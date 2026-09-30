"""Masked metrics and the evaluation loop, with a stub forecaster and a synthetic tgdata graph."""
import numpy as np
import pytest
import torch

tgdata = pytest.importorskip("tgdata")

from tsfm_zoo import Forecast  # noqa: E402
from tsfm_zoo.tgeval import MaskedErrors, evaluate_forecast, last_observed, summarize  # noqa: E402


def test_masked_errors_ignore_unobserved_readings():
    errors = MaskedErrors(horizon=1)
    y = torch.tensor([[[10.0, 0.0, 50.0]]])                                          # [B=1, H=1, N=3]
    pred = torch.tensor([[[12.0, 5.0, 40.0]]])
    errors.update(pred, y, mask=torch.tensor([[[True, False, True]]]))
    scores = errors.per_step()
    assert scores["MAE"] == [6.0] and scores["RMSE"] == [pytest.approx((104 / 2) ** 0.5)]
    assert scores["MAPE"] == [pytest.approx(100 * (0.2 + 0.2) / 2)]


def test_summarize_reports_steps_and_mean():
    per_step = {"MAE": [float(h) for h in range(1, 13)]}
    assert summarize(per_step) == {"MAE": {"3": 3.0, "6": 6.0, "12": 12.0, "avg": 6.5}}


def test_last_observed_skips_missing_steps():
    x = torch.tensor([[[1.0, 9.0], [2.0, 9.0], [0.0, 9.0]]])                         # [1, T=3, N=2]
    mask = torch.tensor([[[True, False], [True, False], [False, False]]])
    assert last_observed(x, mask).tolist() == [[2.0, 0.0]]                            # node 1 never observed


class LastPlusOne:
    """Forecasts the last input value plus one, at every step."""
    name, uses = "stub", frozenset({"mask", "freq"})

    def __call__(self, x, mask=None, horizon=None, **inputs):
        assert inputs == {"freq": "5min"} and mask.shape == x.shape
        return Forecast(median=(x[:, -1:] + 1).expand(-1, horizon, -1))


@pytest.fixture
def graph_for_eval():
    rng = np.random.default_rng(0)
    T, N = 60, 4
    return tgdata.TemporalGraph(
        name="toy", domain="synthetic", tasks=["node_forecasting"], time_mode="discrete",
        x=rng.uniform(1, 100, size=(T, N, 1)).astype(np.float32), mask=rng.random((T, N)) > 0.15,
        timestamps=np.arange(T, dtype=np.int64) * 300,
        edge_index=np.array([[0], [1]], dtype=np.int64), edge_weight=np.ones(1, dtype=np.float32),
        splits={"default": tgdata.Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2})},
        meta={"freq": "5min", "default_task": {"name": "node_forecasting", "params": {"window": 4, "horizon": 3}}},
    )


def test_evaluate_forecast_matches_a_numpy_computation(graph_for_eval):
    g, window = graph_for_eval, 6
    x, observed, N = g.x, g.mask, g.num_nodes
    result = evaluate_forecast(LastPlusOne(), g, window=window, batch_size=5)

    reference = g.task(split="test", device="cpu")                        # the default task: window 4, horizon 3
    horizon = reference.horizon
    abs_error = {h: [] for h in range(horizon)}
    persistence_error = {h: [] for h in range(horizon)}
    for t in reference.anchors:
        seen_in = observed[t - window:t]                                            # [window, N]
        steps = np.where(seen_in, np.arange(window)[:, None], -1).max(0)            # last observed step per node
        last = np.where(steps >= 0, x[t - window:t, :, 0][np.maximum(steps, 0), np.arange(N)], 0.0)
        for h in range(horizon):
            y, seen = x[t + h, :, 0], observed[t + h]
            abs_error[h] += list(np.abs(x[t - 1, :, 0] + 1 - y)[seen])
            persistence_error[h] += list(np.abs(last - y)[seen])
    assert result["windows"] == len(reference) and result["horizon"] == horizon == 3
    np.testing.assert_allclose(result["model"]["MAE"], [np.mean(abs_error[h]) for h in range(horizon)], rtol=1e-5)
    np.testing.assert_allclose(result["persistence"]["MAE"],
                               [np.mean(persistence_error[h]) for h in range(horizon)], rtol=1e-5)


def test_targets_do_not_depend_on_the_context(graph_for_eval):
    short = evaluate_forecast(LastPlusOne(), graph_for_eval, window=4)
    long = evaluate_forecast(LastPlusOne(), graph_for_eval, window=9)
    assert short["windows"] == long["windows"] == len(graph_for_eval.task(split="test", device="cpu"))


def test_context_reaching_before_the_series_start_raises(graph_for_eval):
    with pytest.raises(ValueError, match="reaches before"):
        evaluate_forecast(LastPlusOne(), graph_for_eval, window=1000)
