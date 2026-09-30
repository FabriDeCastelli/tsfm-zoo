"""GIFT-Eval test windows without gluonts, following gift_eval.data.Dataset.

A config is the leaderboard's name, e.g. "ett1/H/long". Data lives in
$GIFT_EVAL (one HF `load_from_disk` directory per dataset) and is fetched
from the HF dataset Salesforce/GiftEval on first use.
"""
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import base_freq, to_offset

# The fixed repro subset: frequencies 10T to W, short and long horizons, uni- and multivariate.
REPRO_CONFIGS = [
    "m4_hourly/H/short",
    "electricity/H/short",
    "ett1/H/short",
    "ett1/H/long",
    "solar/10T/short",
    "saugeen/W/short",
]

# Leaderboard folder under gift-eval/results and the settings of its replication notebook.
# pack: rows per call when series are forecast jointly (cross-learning), None for independent.
LEADERBOARD = {
    "chronos-2": dict(results="Chronos-2", multivariate=True, pack=100),
    "chronos-2-small": dict(results=None, multivariate=True, pack=100),
    # notebooks/chronos.ipynb: univariate, full history (the pipeline truncates), 20 samples for T5.
    "bolt-small": dict(results="chronos_bolt_small", multivariate=False, pack=None),
    "bolt-base": dict(results="chronos_bolt_base", multivariate=False, pack=None),
    "t5-small": dict(results="Chronos_small", multivariate=False, pack=None),
    "t5-base": dict(results="Chronos_base", multivariate=False, pack=None),
    "t5-large": dict(results="Chronos_large", multivariate=False, pack=None),
    # notebooks/timesfm3.ipynb: multivariate items, 64 per batch.
    "3.0": dict(results="TimesFM-3", multivariate=True, pack=None, batch_size=64),
    # notebooks/timesfm2p5.ipynb: univariate, 1024 per batch (the batch sets the padded context).
    "2.5": dict(results="TimesFM-2.5", multivariate=False, pack=None, batch_size=1024),
    # notebooks/tirex.ipynb: the leakage-controlled 1.1 checkpoint, CUDA kernels, frequency resampling.
    "tirex-1.1-gifteval": dict(results="TiRex", multivariate=False, pack=None, batch_size=512,
                               load=dict(backend="torch", resample_strategy="frequency")),  # notebook: "cuda"
    # notebooks/tirex-2.ipynb: univariate, 512 per batch, the checkpoint's own TTA settings.
    "tirex-2-gifteval-zs": dict(results="TiRex-2-Zeroshot", multivariate=False, pack=None, batch_size=512),
    "tirex-2-gifteval-pretrain": dict(results="TiRex-2-Pretrained", multivariate=False, pack=None, batch_size=512),
    # notebooks/toto_2_0.ipynb: multivariate items, context 4096, 512 per batch.
    **{f"2.0-{size}": dict(results=f"Toto-2.0-{size}", multivariate=True, pack=None, batch_size=512)
       for size in ("4m", "22m", "313m", "1B", "2.5B")},
    # granite-tsfm notebooks/hfdemo/flowstate: univariate, IBM's GIFT wrapper (make_positive on).
    "r1.1": dict(results="Granite-FlowState-r1.1", multivariate=False, pack=None, batch_size=1024),
    # notebooks/sundial.ipynb: univariate, 1024 per batch, 100 samples, bf16 autocast.
    "sundial-base-128m": dict(results="sundial_base_128m", multivariate=False, pack=None, batch_size=1024,
                              load=dict(autocast="bfloat16"), call=dict(num_samples=100)),
    # notebooks/moirai.ipynb: multivariate items, context 4000, patch 32, 100 samples, 512 per batch.
    # The notebook loads moirai-1.0-R-*, the leaderboard's config.json links 1.1-R-*.
    **{f"{v}-{size}": dict(results=f"Moirai_{size}", multivariate=True, pack=None, batch_size=512)
       for v in ("1.0", "1.1") for size in ("small", "base", "large")},
    # notebooks/moirai2.ipynb: univariate, context 4000, 512 per batch.
    "2.0-small": dict(results="Moirai2", multivariate=False, pack=None, batch_size=512),
}

RESULTS_URL = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main/results/{}/all_results.csv"
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
M4_PRED_LENGTH = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TEST_SPLIT, MAX_WINDOW = 0.1, 20
DIR_NAMES = {"saugeen": "saugeenday", "temperature_rain": "temperature_rain_with_missing",
             "kdd_cup_2018": "kdd_cup_2018_with_missing", "car_parts": "car_parts_with_missing"}


def data_root() -> Path:
    return Path(os.environ.get("GIFT_EVAL", Path(os.environ.get("TSFM_ZOO_CACHE", "~/.cache/tsfm_zoo")) / "gift-eval")).expanduser()


def _hf_dir(config: str) -> str:
    name, freq, _ = config.split("/")
    name = DIR_NAMES.get(name, name)
    return name if name.startswith("m4_") else f"{name}/{freq}"


@dataclass
class Window:
    past: list[np.ndarray]    # per series [d, T_i]
    label: list[np.ndarray]   # per series [d, H]


@dataclass
class GiftEvalConfig:
    config: str
    freq: str
    prediction_length: int
    windows: list[Window]
    multivariate: bool
    starts: list[pd.Period]   # per series, the period of its first step


def load(config: str) -> GiftEvalConfig:
    import datasets

    path = data_root() / _hf_dir(config)
    if not path.exists():
        from huggingface_hub import snapshot_download
        snapshot_download("Salesforce/GiftEval", repo_type="dataset", local_dir=data_root(),
                          allow_patterns=[f"{_hf_dir(config)}/*"])
    ds = datasets.load_from_disk(str(path)).with_format("numpy")
    targets = [np.atleast_2d(np.asarray(t, dtype=np.float64)) for t in ds["target"]]
    freq = ds[0]["freq"]
    name, _, term = config.split("/")
    unit = base_freq(freq)
    prediction_length = TERM_MULTIPLIER[term] * (M4_PRED_LENGTH if "m4" in name else PRED_LENGTH)[unit]

    if "m4" in name:
        n_windows = 1
    else:
        min_length = min(t.shape[-1] for t in targets)
        n_windows = min(max(1, math.ceil(TEST_SPLIT * min_length / prediction_length)), MAX_WINDOW)

    windows = []
    for w in range(n_windows):
        past, label = [], []
        for t in targets:
            cut = t.shape[-1] - prediction_length * (n_windows - w)
            past.append(t[:, :cut])
            label.append(t[:, cut:cut + prediction_length])
        windows.append(Window(past, label))
    starts = [pd.Period(pd.Timestamp(np.asarray(t).item()), freq=to_offset(freq)) for t in ds["start"]]
    return GiftEvalConfig(config, freq, prediction_length, windows, multivariate=targets[0].shape[0] > 1,
                          starts=starts)


def leaderboard(results: str, config: str) -> dict[str, float]:
    cache = data_root() / "results" / f"{results}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RESULTS_URL.format(results), cache)
    row = pd.read_csv(cache).set_index("dataset").loc[config]
    return {"MASE": float(row["eval_metrics/MASE[0.5]"]),
            "CRPS": float(row["eval_metrics/mean_weighted_sum_quantile_loss"])}


def _left_pad(series: list[np.ndarray], length: int) -> np.ndarray:
    """[d_i, T_i] each -> [length, sum d_i], NaN-padded on the left, keeping the last `length` steps."""
    cols = []
    for s in series:
        s = s[:, -length:]
        cols.append(np.pad(s, ((0, 0), (length - s.shape[1], 0)), constant_values=np.nan))
    return np.concatenate(cols, axis=0).T


def _timestamps(start: pd.Period, end: int, length: int) -> np.ndarray:
    """Epoch seconds of steps [end - length, end) of a series starting at `start`, left padding included."""
    periods = pd.period_range(start + (end - length), periods=length)
    return periods.to_timestamp().asi8 // 10**9


def _calls(series: list[np.ndarray], pack: int | None, batch_size: int):
    """Group series into model calls: yields (indices, layout) with layout "C" (series side by side
    in one item, forecast jointly) or "B" (one series per batch item, same d)."""
    if pack:
        group, rows = [], 0
        for i, s in enumerate(series):
            group.append(i)
            rows += s.shape[0]
            if rows >= pack:
                yield group, "C"
                group, rows = [], 0
        if group:
            yield group, "C"
    else:
        for start in range(0, len(series), batch_size):
            yield list(range(start, min(start + batch_size, len(series)))), "B"


def evaluate(model, data: GiftEvalConfig, *, multivariate: bool, pack: int | None, batch_size: int = 64,
             quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), call: dict | None = None) -> dict:
    """Forecast every window of `data` and score it like the leaderboard. Windows are
    forecast separately, so joint forecasting never mixes windows of one series."""
    import torch

    from .metrics import gift_eval_metrics

    H = data.prediction_length
    extra = ({"freq": data.freq} if "freq" in model.uses else {}) | (call or {})
    instances = []
    for window in data.windows:
        past, label, starts = window.past, window.label, data.starts
        if data.multivariate and not multivariate:
            starts = [t for s, t in zip(past, starts) for _ in range(s.shape[0])]
            past = [s[i:i + 1] for s in past for i in range(s.shape[0])]
            label = [s[i:i + 1] for s in label for i in range(s.shape[0])]
        for idx, layout in _calls(past, pack, batch_size):
            length = min(max(past[i].shape[1] for i in idx), model.max_context)
            if layout == "C":
                x = torch.tensor(_left_pad([past[i] for i in idx], length), dtype=torch.float32)[None]
            else:
                x = torch.stack([torch.tensor(_left_pad([past[i]], length), dtype=torch.float32) for i in idx])
            if "timestamps" in model.uses:
                # Packed series share one time axis: the first series' (packing is only used without timestamps).
                rows = idx[:1] if layout == "C" else idx
                extra["timestamps"] = torch.from_numpy(
                    np.stack([_timestamps(starts[i], past[i].shape[1], length) for i in rows]))
            with torch.no_grad():
                f = model(x, horizon=H, quantile_levels=list(quantile_levels), **extra)
            med = f.median.cpu().numpy()                                  # [B, H, C]
            qs = None if f.quantiles is None else f.quantiles.cpu().numpy()  # [B, H, C, Q]
            col = 0
            for j, i in enumerate(idx):
                d = past[i].shape[0]
                b, cols = (0, slice(col, col + d)) if layout == "C" else (j, slice(0, d))
                instances.append((past[i], label[i], med[b, :, cols].T,
                                  None if qs is None else qs[b, :, cols].transpose(1, 0, 2)))
                col += d if layout == "C" else 0
    return gift_eval_metrics(instances, quantile_levels, data.freq)
