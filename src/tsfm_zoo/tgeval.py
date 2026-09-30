"""Zero-shot forecasting on a tgdata node-forecasting dataset (PeMS, ...), scored like the traffic papers.

Every node is a channel of the model's input. Readings the dataset marks as missing are masked in
the input and left out of the metrics: masked MAE, RMSE and MAPE per horizon step, as in ASTGCN
and STSGCN, on the original scale. A persistence baseline (the last observed value of the window)
runs on the same windows. Needs the `tgdata` extra.
"""
import logging

import torch

from .base import TSFM

logger = logging.getLogger("tsfm_zoo")

REPORTED_STEPS = (3, 6, 12)


class MaskedErrors:
    """Running masked error sums per horizon step, accumulated over batches."""

    def __init__(self, horizon: int):
        self.sums = torch.zeros(4, horizon, dtype=torch.float64)      # |e|, e^2, |e|/|y|, count

    def update(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        """pred, y [B, H, N]; mask True where y was observed."""
        pred, y, mask = pred.double().cpu(), y.double().cpu(), mask.cpu()
        error = (pred - y).abs()
        self.sums += torch.stack([
            (error * mask).sum((0, 2)),
            (error ** 2 * mask).sum((0, 2)),
            (error / torch.where(mask, y.abs(), torch.ones_like(y)) * mask).sum((0, 2)),
            mask.sum((0, 2)).double(),
        ])

    def per_step(self) -> dict[str, list[float]]:
        mae, sq, ape, count = self.sums
        return {"MAE": (mae / count).tolist(), "RMSE": (sq / count).sqrt().tolist(),
                "MAPE": (100 * ape / count).tolist()}


def summarize(per_step: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Values at the reported horizon steps (1-based) and the mean over all steps."""
    return {metric: {**{str(h): values[h - 1] for h in REPORTED_STEPS if h <= len(values)},
                     "avg": sum(values) / len(values)} for metric, values in per_step.items()}


def last_observed(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, T, N] -> [B, N]: the last observed reading of each series, 0 if it has none."""
    steps = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
    last = torch.where(mask, steps, torch.full_like(steps, -1)).max(1).values            # [B, N]
    value = x.gather(1, last.clamp_min(0)[:, None])[:, 0]
    return torch.where(last >= 0, value, torch.zeros_like(value))


def evaluate_forecast(model: TSFM, g, *, window: int, split: str = "test", batch_size: int = 16,
                      data_device="cpu", **inputs) -> dict:
    """Forecast the targets of the dataset's own task on `split`, from `window` steps of context.

    The targets are those of `g.task(split=split)` (for PeMS: 12 steps after every test window of
    12), whatever `window` is, so different contexts are scored on the same targets; a longer
    context only reaches further back. Returns the per-step metrics of `model` and of persistence,
    the horizon and the number of windows. `inputs` are further forecast arguments; `freq` comes
    from the dataset when the model reads it.
    """
    if g.x.shape[-1] != 1:
        raise ValueError(f"{g.name} has {g.x.shape[-1]} channels per node; the evaluation needs one")
    if "freq" in model.uses and "freq" not in inputs:
        inputs["freq"] = g.meta["freq"]
    reference = g.task(split=split, device=data_device)
    horizon, anchors = reference.horizon, reference.anchors
    if anchors.min() < window:
        raise ValueError(f"a context of {window} steps reaches before the start of {g.name} "
                         f"for the first {split} target (step {anchors.min()})")
    task = g.task("node_forecasting", window=window, horizon=horizon, split=None, device=data_device)
    rows = anchors - window                                    # the task holds one window per step, from `window`
    assert (task.anchors[rows] == anchors).all()
    scores, baseline = MaskedErrors(horizon), MaskedErrors(horizon)
    logger.info("%s on %s: %d %s targets of %d steps from %d steps of context, inputs %s",
                model.name, g.name, len(rows), split, horizon, window, inputs)
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = task.__getitems__(rows[i:i + batch_size])
            x, y = batch["x"][..., 0].cpu(), batch["y"][..., 0].cpu()                    # [B, T, N], [B, H, N]
            mask_x, mask_y = batch["mask_x"][..., 0].cpu(), batch["mask_y"][..., 0].cpu()
            extra = {"timestamps": batch["timestamps"][:, :window].cpu()} if (
                "timestamps" in model.uses and "freq" not in inputs) else {}
            prediction = model(x, mask=mask_x, horizon=horizon, **inputs, **extra).median          # [B, H, N]
            scores.update(prediction, y, mask_y)
            baseline.update(last_observed(x, mask_x)[:, None].expand(-1, horizon, -1), y, mask_y)
            if (i // batch_size) % 50 == 0:
                logger.info("batch %d/%d", i // batch_size, len(rows) // batch_size)
    return {"windows": len(rows), "horizon": horizon, "model": scores.per_step(),
            "persistence": baseline.per_step()}
