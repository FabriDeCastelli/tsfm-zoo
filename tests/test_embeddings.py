"""Window-embedding cache, with a stub model and a synthetic tgdata graph."""
import numpy as np
import pytest
import torch

tgdata = pytest.importorskip("tgdata")

from tsfm_zoo.embeddings import config_hash, embed_windows  # noqa: E402

T, N = 50, 6


class StubModel:
    """Embeds a window as [first step, last step, mean] of each node."""
    name, uses, summary, embed_dim = "stub", frozenset({"mask", "freq"}), "last", 3
    spec = {"family": "chronos", "variant": "bolt-tiny", "seed": 0, "load": {}}

    def __init__(self):
        self.calls, self.seen = 0, {}

    def encode(self, x, mask=None, **inputs):
        self.calls += 1
        self.seen = {"mask": mask, **inputs}
        return torch.stack([x[:, 0], x[:, -1], x.mean(1)], -1)[:, :, None]         # [B, N, 1, 3]

    def pool(self, z, how):
        return z[:, :, 0]


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    g = tgdata.TemporalGraph(
        name="toy", domain="synthetic", tasks=["node_forecasting"], time_mode="discrete",
        x=rng.normal(size=(T, N, 1)).astype(np.float32), mask=rng.random((T, N)) > 0.1,
        timestamps=np.arange(T, dtype=np.int64) * 300,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64), edge_weight=np.ones(2, dtype=np.float32),
        splits={"default": tgdata.Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2})},
        meta={"freq": "5min", "default_task": {"name": "node_forecasting", "params": {"window": 4, "horizon": 3}}},
    )
    return g


def test_hash_is_stable_and_sensitive():
    config = {"window": 12, "dataset": {"name": "pems08"}, "pool": "last"}
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))
    assert len(config_hash(config)) == 16
    assert config_hash(config) != config_hash({**config, "window": 13})
    assert config_hash(config) != config_hash({**config, "dataset": {"name": "pems04"}})


def test_rows_are_window_embeddings_and_cache_is_reused(graph, tmp_path):
    model = StubModel()
    cache = embed_windows(model, graph, tmp_path, window=5, batch_size=7)
    assert cache.array.shape == (T - 5 + 1, N, 3)
    x = graph.x[..., 0]
    for k in (0, 17, T - 5):
        np.testing.assert_allclose(cache.array[k], np.stack([x[k], x[k + 4], x[k:k + 5].mean(0)], -1), rtol=1e-6)
    assert model.seen["freq"] == "5min" and model.seen["mask"].shape[1:] == (5, N)

    calls = model.calls
    again = embed_windows(model, graph, tmp_path, window=5, batch_size=3)
    assert again.path == cache.path and model.calls == calls                        # batch size is not part of the key
    assert embed_windows(model, graph, tmp_path, window=6).path != cache.path
    assert not list(tmp_path.glob("*.partial*"))

    for split in ("train", "val", "test"):
        task = graph.task(window=5, horizon=3, split=split, device="cpu")
        np.testing.assert_array_equal(cache.rows(task), cache.array[task.window_starts])
        assert len(cache.rows(task)) == len(task)


def test_rows_reject_a_task_of_another_window(graph, tmp_path):
    cache = embed_windows(StubModel(), graph, tmp_path, window=5)
    with pytest.raises(ValueError, match="5 steps"):
        cache.rows(graph.task(window=4, horizon=3, split="test", device="cpu"))


def test_needs_an_explicit_variant(graph, tmp_path):
    model = StubModel()
    model.spec = {**model.spec, "variant": None}
    with pytest.raises(ValueError, match="explicit variant"):
        embed_windows(model, graph, tmp_path, window=5)


def test_changed_readings_get_their_own_cache(graph, tmp_path):
    first = embed_windows(StubModel(), graph, tmp_path, window=5)
    graph.x[3, 2, 0] += 1.0
    assert embed_windows(StubModel(), graph, tmp_path, window=5).path != first.path
