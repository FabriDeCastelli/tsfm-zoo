"""GIFT-Eval metrics in numpy, following gluonts.ev as called by the leaderboard.

Each instance is one forecast window of one series: `label` [d, H],
`past` [d, T] (the history, for the seasonal error), forecasts [d, H] and
[d, H, Q]. NaN labels are masked, as with `mask_invalid_label=True`.
"""
import warnings

import numpy as np
import pandas as pd

DEFAULT_SEASONALITIES = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4}


def to_offset(freq: str):
    with warnings.catch_warnings():  # GIFT-Eval stores old spellings such as '10T'
        warnings.simplefilter("ignore", FutureWarning)
        return pd.tseries.frequencies.to_offset(freq)


def base_freq(freq: str) -> str:
    """Base unit of a pandas frequency in the old spelling: '10min' -> 'T', 'W-SUN' -> 'W'."""
    name = to_offset(freq).name.split("-")[0]
    if len(name) >= 2 and name.endswith("S") and name not in ("MS", "QS"):
        name = name[:-1]
    return {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S", "us": "U"}.get(name, name)


def seasonality(freq: str) -> int:
    offset = to_offset(freq)
    base = DEFAULT_SEASONALITIES.get(base_freq(freq), 1)
    season, remainder = divmod(base, offset.n)
    return season if not remainder else 1


def seasonal_error(past: np.ndarray, season: int) -> np.ma.MaskedArray:
    """[d, T] -> [d, 1]: mean |y_t - y_{t-m}| over the history."""
    past = np.ma.masked_invalid(past)
    if season > past.shape[-1]:
        season = 1
    return np.abs(past[:, season:] - past[:, :-season]).mean(axis=-1, keepdims=True)


def gift_eval_metrics(instances, quantile_levels, freq: str) -> dict[str, float]:
    """instances: iterable of (past [d, T], label [d, H], median [d, H], quantiles [d, H, Q])."""
    season = seasonality(freq)
    ase, qloss, abs_label = [], [], []
    for past, label, median, quantiles in instances:
        label = np.ma.masked_invalid(label)
        ase.append((np.abs(label - median) / seasonal_error(past, season)).ravel())
        if quantiles is not None:
            q = np.asarray(quantile_levels)
            loss = 2 * np.abs((label[..., None] - quantiles) * ((quantiles >= label[..., None]) - q))
            qloss.append(loss.reshape(-1, len(q)))
            abs_label.append(np.abs(label).ravel())
    out = {"MASE": float(np.ma.concatenate(ase).mean())}
    if qloss:
        total = np.ma.concatenate(qloss).sum(axis=0) / np.ma.concatenate(abs_label).sum()
        out["CRPS"] = float(total.mean())
    return out
