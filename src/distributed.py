"""Pure PyTorch distributed training: DDP and FSDP."""

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
from torch.nn.parallel import DistributedDataParallel as DDP
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
    strategy: str  # none | ddp | fsdp


_CTX: DistContext | None = None


def dist_context() -> DistContext:
    if _CTX is None:
        raise RuntimeError("Call setup_distributed() before using distributed helpers.")
    return _CTX


def is_distributed() -> bool:
    return _CTX is not None and _CTX.enabled


def setup_distributed(*, strategy: str = "none", backend: str = "nccl") -> DistContext:
    """Initialize torch.distributed when launched via torchrun."""
    global _CTX

    if _CTX is not None:
        return _CTX

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    strategy = strategy.lower()

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _CTX = DistContext(False, 0, 1, 0, True, device, "none")
        return _CTX

    if strategy not in ("ddp", "fsdp"):
        raise ValueError(f"Multi-GPU training requires DIST_STRATEGY=ddp or fsdp, got {strategy!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-process training requires CUDA.")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    _CTX = DistContext(True, rank, world_size, local_rank, rank == 0, device, strategy)
    logger.info("Distributed init strategy=%s rank=%d world=%d device=%s", strategy, rank, world_size, device)
    return _CTX


def teardown_distributed() -> None:
    global _CTX
    if _CTX and _CTX.enabled and dist.is_initialized():
        dist.destroy_process_group()
    _CTX = None


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


def wrap_ddp(model: torch.nn.Module) -> DDP:
    ctx = dist_context()
    return DDP(
        model,
        device_ids=[ctx.local_rank],
        output_device=ctx.local_rank,
        find_unused_parameters=False,
    )


def wrap_fsdp(
    model: torch.nn.Module,
    *,
    sharding: str = "FULL_SHARD",
    use_bf16: bool = True,
    cpu_offload: bool = False,
    use_orig_params: bool = False,
) -> FSDP:
    ctx = dist_context()
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

    logger.info("Wrapping model with FSDP (%s, layer=%s)", sharding, layer_cls.__name__)
    return FSDP(model, **kwargs)


def wrap_model(
    model: torch.nn.Module,
    *,
    use_bf16: bool = True,
    sharding: str = "FULL_SHARD",
    cpu_offload: bool = False,
    use_orig_params: bool = False,
) -> torch.nn.Module:
    """Wrap model with DDP or FSDP according to the active distributed strategy."""
    ctx = dist_context()
    if not ctx.enabled:
        return model
    if ctx.strategy == "ddp":
        return wrap_ddp(model)
    if ctx.strategy == "fsdp":
        return wrap_fsdp(
            model,
            sharding=sharding,
            use_bf16=use_bf16,
            cpu_offload=cpu_offload,
            use_orig_params=use_orig_params,
        )
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (DDP, FSDP)):
        return model.module
    return model


@contextmanager
def summon_full_params(model: torch.nn.Module, *, writeback: bool = False) -> Iterator[None]:
    """Required for generate() under FSDP."""
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=writeback):
            yield
    else:
        yield


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> None:
    if isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def distributed_dataloader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn=None,
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
        collate_fn=collate_fn,
    )


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def save_model(model: torch.nn.Module, output_dir: str, tokenizer=None) -> None:
    """Save HF checkpoint on rank 0 (handles DDP and FSDP)."""
    ctx = dist_context()
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
        if ctx.is_main:
            base = unwrap_model(model)
            base.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        if ctx.enabled:
            dist.barrier()
        return

    if ctx.enabled and not ctx.is_main:
        dist.barrier()
        return

    unwrap_model(model).save_pretrained(output_dir, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if ctx.enabled:
        dist.barrier()


def push_to_hub(model: torch.nn.Module, repo_id: str, tokenizer, token: str | None) -> None:
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    ctx = dist_context()
    tmp = tempfile.mkdtemp() if ctx.is_main else "."
    save_model(model, tmp, tokenizer if ctx.is_main else None)

    if ctx.is_main:
        HfApi().upload_folder(folder_path=tmp, repo_id=repo_id, token=token)
        shutil.rmtree(tmp)
