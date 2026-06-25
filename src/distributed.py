"""FSDP utilities: data parallelism, optional tensor parallelism, checkpoint I/O."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

_SHARDING = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
}


@dataclass(frozen=True)
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    is_main: bool
    device: torch.device
    tp_size: int
    dp_size: int


_CTX: DistContext | None = None


def dist_context() -> DistContext:
    if _CTX is None:
        raise RuntimeError("Call setup_distributed() before using distributed helpers.")
    return _CTX


def is_distributed() -> bool:
    return _CTX is not None and _CTX.enabled


def setup_distributed(*, tp_size: int = 1, backend: str = "nccl") -> DistContext:
    """Initialize torch.distributed when launched via torchrun. No-op on single process."""
    global _CTX

    if _CTX is not None:
        return _CTX

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _CTX = DistContext(False, 0, 1, 0, True, device, 1, 1)
        return _CTX

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-process training requires CUDA.")

    if world_size % tp_size != 0:
        raise ValueError(f"WORLD_SIZE ({world_size}) must be divisible by FSDP_TP_SIZE ({tp_size}).")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    dp_size = world_size // tp_size
    device = torch.device(f"cuda:{local_rank}")
    _CTX = DistContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=rank == 0,
        device=device,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    logger.info(
        "Distributed init rank=%d world=%d dp=%d tp=%d device=%s",
        rank,
        world_size,
        dp_size,
        tp_size,
        device,
    )
    return _CTX


def teardown_distributed() -> None:
    global _CTX
    if _CTX and _CTX.enabled and dist.is_initialized():
        dist.destroy_process_group()
    _CTX = None


def device_mesh():
    """2D mesh (dp, tp) when tp_size > 1, else 1D (dp,)."""
    from torch.distributed.device_mesh import init_device_mesh

    ctx = dist_context()
    if not ctx.enabled:
        return None
    if ctx.tp_size > 1:
        return init_device_mesh("cuda", (ctx.dp_size, ctx.tp_size), mesh_dim_names=("dp", "tp"))
    return init_device_mesh("cuda", (ctx.world_size,), mesh_dim_names=("dp",))


def _layer_class(model: torch.nn.Module) -> type[torch.nn.Module]:
    explicit = os.getenv("FSDP_TRANSFORMER_LAYER_CLS", "").strip()
    if explicit:
        for module in model.modules():
            if type(module).__name__ == explicit:
                return type(module)
        raise ValueError(f"FSDP_TRANSFORMER_LAYER_CLS={explicit!r} not found in model.")

    for module in model.modules():
        name = type(module).__name__
        if name.endswith("DecoderLayer") or name.endswith("Block"):
            return type(module)
    raise ValueError("Could not auto-detect transformer layer class; set FSDP_TRANSFORMER_LAYER_CLS.")


def _mixed_precision(use_bf16: bool) -> MixedPrecision | None:
    if not use_bf16 or not torch.cuda.is_available():
        return None
    return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)


def apply_tensor_parallel(model: torch.nn.Module) -> torch.nn.Module:
    """Shard attention/MLP linear layers across the tp mesh dimension."""
    ctx = dist_context()
    if not ctx.enabled or ctx.tp_size <= 1:
        return model

    mesh = device_mesh()
    tp_mesh = mesh["tp"]

    try:
        from transformers.integrations.tensor_parallel import shard_model
    except ImportError as exc:
        raise RuntimeError(
            "FSDP_TP_SIZE > 1 requires transformers tensor-parallel integration "
            "(transformers>=4.46, torch>=2.4)."
        ) from exc

    logger.info("Applying tensor parallelism (tp_size=%d)", ctx.tp_size)
    return shard_model(model, tp_mesh, plan="auto")


def wrap_fsdp(
    model: torch.nn.Module,
    *,
    sharding: str = "FULL_SHARD",
    use_bf16: bool = True,
    cpu_offload: bool = False,
    use_orig_params: bool = False,
) -> FSDP:
    """Wrap model with FSDP on the data-parallel mesh dimension."""
    ctx = dist_context()
    if not ctx.enabled:
        return model  # type: ignore[return-value]

    mesh = device_mesh()
    fsdp_mesh = mesh["dp"] if ctx.tp_size > 1 else mesh

    layer_cls = _layer_class(model)
    strategy = _SHARDING.get(sharding.upper(), ShardingStrategy.FULL_SHARD)

    kwargs: dict = {
        "sharding_strategy": strategy,
        "auto_wrap_policy": partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}),
        "mixed_precision": _mixed_precision(use_bf16),
        "device_id": ctx.local_rank,
        "use_orig_params": use_orig_params,
        "sync_module_states": True,
    }
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload

        kwargs["cpu_offload"] = CPUOffload(offload_params=True)
    if fsdp_mesh is not None:
        kwargs["device_mesh"] = fsdp_mesh

    logger.info("Wrapping model with FSDP (%s, layer=%s)", sharding, layer_cls.__name__)
    return FSDP(model, **kwargs)


@contextmanager
def summon_full_params(model: torch.nn.Module, *, writeback: bool = False) -> Iterator[None]:
    """Enable full-parameter forward/generate under FSDP."""
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=writeback):
            yield
    else:
        yield


def fsdp_trainer_config(
    *,
    sharding: str,
    use_bf16: bool,
    use_lora: bool,
    layer_cls_name: str | None = None,
    cpu_offload: bool = False,
) -> tuple[str, dict]:
    """Return (fsdp_flag, fsdp_config) for HuggingFace/TRL TrainingArguments."""
    layer = layer_cls_name or os.getenv("FSDP_TRANSFORMER_LAYER_CLS", "Gemma3DecoderLayer")
    fsdp_flag = "full_shard auto_wrap"
    if cpu_offload:
        fsdp_flag += " offload"

    config = {
        "transformer_layer_cls_to_wrap": [layer],
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "use_orig_params": use_lora,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
        "state_dict_type": "FULL_STATE_DICT",
    }
    if sharding.upper() == "HYBRID_SHARD":
        fsdp_flag = "hybrid_shard auto_wrap"
        if cpu_offload:
            fsdp_flag += " offload"
    elif sharding.upper() == "SHARD_GRAD_OP":
        fsdp_flag = "shard_grad_op auto_wrap"
        if cpu_offload:
            fsdp_flag += " offload"

    if use_bf16:
        config["mixed_precision_policy"] = {
            "param_dtype": "bf16",
            "reduce_dtype": "bf16",
            "buffer_dtype": "bf16",
        }
    return fsdp_flag, config


def distributed_dataloader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    ctx = dist_context()
    sampler = None
    if ctx.enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def save_fsdp_model(model: torch.nn.Module, output_dir: str, tokenizer=None) -> None:
    """Gather full weights on rank 0 and write a standard HF checkpoint."""
    ctx = dist_context()
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
        if ctx.is_main:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
            else:
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        if ctx.enabled:
            dist.barrier()
        return

    if not ctx.is_main and ctx.enabled:
        dist.barrier()
        return

    model.save_pretrained(output_dir, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if ctx.enabled:
        dist.barrier()


def push_to_hub(model: torch.nn.Module, repo_id: str, tokenizer, token: str | None) -> None:
    """All ranks participate in FSDP gather; rank 0 uploads."""
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    ctx = dist_context()
    tmp = tempfile.mkdtemp() if ctx.is_main else "."
    save_fsdp_model(model, tmp, tokenizer if ctx.is_main else None)

    if ctx.is_main:
        HfApi().upload_folder(folder_path=tmp, repo_id=repo_id, token=token)
        shutil.rmtree(tmp)
