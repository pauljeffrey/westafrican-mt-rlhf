import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_africomet():
    from comet import download_model, load_from_checkpoint

    logger.info("Loading AfriCOMET reward model: %s", settings.reward_model_id)
    path = download_model(settings.reward_model_id)
    model = load_from_checkpoint(path)
    model.eval()
    return model


class AfricometReward:
    """Scores (src, model_translation, reference) with AfriCOMET."""

    def __init__(self) -> None:
        self.model = _load_africomet()

    def score_batch(
        self,
        srcs: list[str],
        translations: list[str],
        refs: list[str],
    ) -> list[float]:
        data = [
            {"src": s, "mt": m, "ref": r}
            for s, m, r in zip(srcs, translations, refs, strict=True)
        ]
        output = self.model.predict(
            data,
            batch_size=settings.reward_batch_size,
            gpus=settings.reward_gpus,
        )
        scores = output.scores if hasattr(output, "scores") else output
        return [float(s) for s in scores]

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        if len(scores) <= 1:
            return scores
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = var**0.5
        if std < 1e-8:
            return [0.0 for _ in scores]
        return [(s - mean) / std for s in scores]
