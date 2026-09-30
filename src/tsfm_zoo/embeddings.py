"""Embeddings of every window of a temporal-graph dataset, cached on disk under a hash of their configuration.

`embed_windows` follows tgdata's "every window of a series, in order" recipe: row k of the saved
array is the embedding of steps k, ..., k + window - 1, so any split of the dataset maps onto it
through `task.window_starts`. Needs the `tgdata` extra.
"""
import hashlib
import json
import logging
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .base import TSFM
from .checkpoints import entry

logger = logging.getLogger("tsfm_zoo")

ARRAY_FILE, CONFIG_FILE = "emb.npy", "config.json"


def config_hash(config: dict) -> str:
    """16 hex characters of the SHA-256 of the canonical JSON of `config`."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def data_fingerprint(x: np.ndarray) -> str:
    """SHA-256 of the readings, so a rebuilt dataset of the same name and shape gets its own cache."""
    digest = hashlib.sha256()
    for start in range(0, len(x), 4096):
        digest.update(np.ascontiguousarray(x[start:start + 4096]).tobytes())
    return digest.hexdigest()[:16]


class WindowEmbeddings:
    """A finished cache directory: `array` is [windows, nodes, D], `config` what determined it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.config = json.loads((self.path / CONFIG_FILE).read_text())["config"]

    @property
    def array(self) -> np.ndarray:
        return np.load(self.path / ARRAY_FILE, mmap_mode="r")

    def rows(self, task) -> np.ndarray:
        """One embedding per sample of `task` (a tgdata node_forecasting task of any split)."""
        if task.g.name != self.config["dataset"]["name"] or task.window != self.config["window"]:
            raise ValueError(f"{self.path.name} holds windows of {self.config['window']} steps of "
                             f"{self.config['dataset']['name']}, not {task.window} of {task.g.name}")
        return self.array[task.window_starts]


def embed_windows(model: TSFM, g, cache_dir, *, window: int, batch_size: int = 64, pool: str | None = None,
                  dtype: str = "float32", normalize: str | None = None, data_device="cpu",
                  **inputs) -> WindowEmbeddings:
    """Embed every window of `g` with `model`, or return the cache of an identical earlier run.

    `g` is a tgdata dataset with one channel per node; its nodes are the model's channels, so a
    cross-channel model sees all nodes of a window together. Each window becomes [nodes, D]:
    the tokens are pooled with `pool` (default: the model's `summary`). Missing readings go to
    the model as a mask. `inputs` are further `encode` arguments (`layer=`, `scale_factor=`, ...);
    `freq` comes from the dataset when the model reads it. Everything that determines the
    result, and nothing else (not the device or the batch size), is hashed into the directory
    name; `config.json` holds it in full.
    """
    if model.spec is None or model.spec["variant"] is None:
        raise ValueError("load the model with load_tsfm and an explicit variant, so the cache key names its checkpoint")
    if g.x.shape[-1] != 1:
        raise ValueError(f"{g.name} has {g.x.shape[-1]} channels per node; embed_windows needs one")
    pool = pool or model.summary
    if "freq" in model.uses and "freq" not in inputs:
        inputs["freq"] = g.meta["freq"]
    spec = model.spec
    config = {
        "model": {**spec, "revision": entry(spec["family"], spec["variant"])["revision"]},
        "dataset": {"name": g.name, "tgdata": _tgdata_version(), "x_shape": list(g.x.shape),
                    "x_sha256": data_fingerprint(g.x)},
        "window": window, "stride": 1, "normalize": normalize, "inputs": inputs, "pool": pool, "dtype": dtype,
    }
    path = Path(cache_dir) / config_hash(config)
    if (path / CONFIG_FILE).exists():
        logger.info("embeddings cached at %s", path)
        return WindowEmbeddings(path)

    task = g.task("node_forecasting", window=window, horizon=0, split=None, normalize=normalize,
                  device=data_device)
    logger.info("embedding %d windows of %s with %s: %s", len(task), g.name, model.name, json.dumps(config))
    partial = path.with_name(f"{path.name}.partial{os.getpid()}")
    partial.mkdir(parents=True)
    out = np.lib.format.open_memmap(partial / ARRAY_FILE, mode="w+", dtype=dtype,
                                    shape=(len(task), g.num_nodes, model.embed_dim))
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(task.loader(batch_size=batch_size)):
            x = batch["x"][..., 0].cpu()                                          # [B, window, N]
            mask = batch["mask_x"][..., 0].cpu() if "mask_x" in batch else None
            extra = {"timestamps": batch["timestamps"].cpu()} if "timestamps" in model.uses and "freq" not in inputs else {}
            z = model.encode(x, mask=mask, **inputs, **extra)                      # [B, N, tokens, D]
            rows = (batch["t"] - task.window).cpu().numpy()
            out[rows] = model.pool(z, pool).float().cpu().numpy().astype(dtype)
            if i % 50 == 0:
                logger.info("window batch %d/%d, %.0fs", i, len(task) // batch_size, time.time() - start)
    out.flush()
    del out
    finished = {"config": config, "hash": path.name, "shape": [len(task), g.num_nodes, model.embed_dim],
                "seconds": round(time.time() - start, 1), "tsfm_zoo": version("tsfm-zoo")}
    (partial / CONFIG_FILE).write_text(json.dumps(finished, indent=1))
    try:
        partial.rename(path)                                                    # a cache is complete or absent
    except OSError:
        shutil.rmtree(partial)                                                  # another run finished first
    return WindowEmbeddings(path)


def _tgdata_version() -> str:
    import tgdata
    return tgdata.__version__
