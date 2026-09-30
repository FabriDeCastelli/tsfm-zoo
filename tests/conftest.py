import functools

import pytest
import torch

from tsfm_zoo import load_tsfm
from tsfm_zoo.checkpoints import is_cached

# Smallest variant of each distinct architecture, used by the CPU contract tests.
SMALLEST = [
    ("chronos", "chronos-2-small"),
    ("chronos", "bolt-tiny"),
    ("chronos", "t5-tiny"),
    ("timesfm", "2.5"),
    ("timesfm", "3.0"),
    ("tirex", "tirex-1"),
    ("tirex", "tirex-2"),
    ("toto", "2.0-4m"),
    ("flowstate", "r1.1"),
    ("moment", "small"),
    ("timer", "sundial-base-128m"),
    ("timer", "timer-base-84m"),
    ("moirai", "2.0-small"),
    ("moirai", "1.1-small"),
    ("moirai", "moe-small"),
]


def importable(family: str) -> bool:
    try:
        from tsfm_zoo.registry import model_class
        model_class(family)
        return True
    except ImportError:
        return False


@functools.cache
def _load(family: str, variant: str):
    return load_tsfm(family, variant, device="cpu", seed=0)


@pytest.fixture(params=SMALLEST, ids=[v for _, v in SMALLEST])
def model(request):
    family, variant = request.param
    if not importable(family):
        pytest.skip(f"{family} is not installed in this env")
    if not is_cached(family, variant):
        pytest.skip(f"{family}/{variant} not cached; run pytest -m download")
    return _load(family, variant)


def sine_trend(B=2, T=256, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=torch.float32)
    phase = torch.rand(B, 1, C, generator=g) * 6.28
    x = 10 + 0.02 * t[None, :, None] + torch.sin(2 * torch.pi * t[None, :, None] / 24 + phase)
    return x + 0.05 * torch.randn(B, T, C, generator=g)


def inputs_for(model, x) -> dict:
    """What a model needs besides x in the contract tests: hourly data (timestamps where required)."""
    if "timestamps" in model.requires:
        B, T = x.shape[:2]
        return {"timestamps": (1_700_000_000 + 3600 * torch.arange(T)).expand(B, T)}
    return {"freq": "h"} if model.needs_one_of else {}


def native_task(model) -> str:
    return "forecast" if model.tasks["forecast"] == "native" else "reconstruct"


@pytest.fixture
def forecaster(model):
    if model.tasks["forecast"] != "native":
        pytest.skip(f"{model.name} has no zero-shot forecast")
    return model
