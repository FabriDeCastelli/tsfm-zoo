"""GIFT-Eval leaderboard reproduction (GPU). Every config within ±2% relative."""
import pytest
import torch

from tsfm_zoo import gifteval, load_tsfm

from .conftest import importable

# Models that reproduce every config within 2% (all exact except TiRex, within 0.72%).
CASES = [
    ("chronos", "chronos-2"),
    ("timesfm", "3.0"),
    ("timesfm", "2.5"),
    ("tirex", "tirex-1.1-gifteval"),
    ("tirex", "tirex-2-gifteval-zs"),
    ("tirex", "tirex-2-gifteval-pretrain"),
    *[("toto", f"2.0-{size}") for size in ("4m", "22m", "313m", "1B")],
    ("flowstate", "r1.1"),
    ("moirai", "2.0-small"),
]


@pytest.mark.repro
@pytest.mark.parametrize("family,variant", CASES)
@pytest.mark.parametrize("config", gifteval.REPRO_CONFIGS)
def test_leaderboard(family, variant, config):
    if not importable(family):
        pytest.skip(f"{family} not installed in this env")
    if not torch.cuda.is_available():
        pytest.skip("repro runs on GPU")
    import subprocess
    from pathlib import Path
    free = subprocess.run([str(Path.home() / ".claude" / "free-gpu")], capture_output=True, text=True).stdout
    device = f"cuda:{free.strip().split(',')[0]}"
    if device == "cuda:-1":
        pytest.skip("no free GPU")
    settings = gifteval.LEADERBOARD[variant]
    model = load_tsfm(family, variant, device=device, **settings.get("load", {}))
    ours = gifteval.evaluate(model, gifteval.load(config), multivariate=settings["multivariate"], pack=settings["pack"],
                             batch_size=settings.get("batch_size", 64), call=settings.get("call"))
    ref = gifteval.leaderboard(settings["results"], config)
    for metric in ("MASE", "CRPS"):
        assert ours[metric] == pytest.approx(ref[metric], rel=0.02), metric
