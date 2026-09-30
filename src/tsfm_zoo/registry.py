import importlib
import logging

from .base import TSFM

logger = logging.getLogger("tsfm_zoo")

# family -> module under tsfm_zoo.models; imported lazily so one model's
# dependencies (or env) never break loading another.
MODELS = {
    "chronos": "chronos",
    "timesfm": "timesfm",
    "tirex": "tirex",
    "toto": "toto",
    "flowstate": "flowstate",
    "moment": "moment",
    "timer": "timer",
    "moirai": "moirai",
}


def model_class(family: str) -> type[TSFM]:
    if family not in MODELS:
        raise KeyError(f"unknown model {family!r}; available: {sorted(MODELS)}")
    module = importlib.import_module(f"tsfm_zoo.models.{MODELS[family]}")
    return module.MODEL


def load_tsfm(family: str, variant: str | None = None, *, device, seed: int = 0,
              frozen: bool = True, **load_cfg) -> TSFM:
    model = model_class(family).from_pretrained(variant, seed=seed, device=device, **load_cfg)
    if frozen:
        model.requires_grad_(False)
        model.eval()
        logger.info("%s loaded frozen: requires_grad=False, eval mode", family)
    return model
