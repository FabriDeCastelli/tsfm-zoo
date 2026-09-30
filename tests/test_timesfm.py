"""TimesFM 3.0 and 2.5 against upstream's own inference paths."""
import math

import numpy as np
import pytest
import torch

from .conftest import importable, sine_trend

pytestmark = pytest.mark.skipif(not importable("timesfm"), reason="timesfm not installed in this env")


def with_gaps():
    x = sine_trend(B=3, T=300, C=2)
    x[0, :60, :] = torch.nan     # leading steps missing in every variate: stripped
    x[1, 100:110, 1] = torch.nan  # interior gap: interpolated
    return x


@pytest.fixture
def tfm3(model):
    if model.name != "timesfm-3.0":
        pytest.skip("TimesFM 3.0 only")
    return model


@pytest.fixture
def tfm25(model):
    if model.name != "timesfm-2.5":
        pytest.skip("TimesFM 2.5 only")
    return model


@pytest.mark.parametrize("horizon,covariates", [(48, False), (200, False), (48, True)])
def test_tfm3_is_upstream_evaluator(tfm3, horizon, covariates):
    from timesfm3.torch.evaluator import TimesFM3Evaluator
    from timesfm3.torch.timesfm3_forecaster import _ModelConfig

    evaluator = TimesFM3Evaluator.__new__(TimesFM3Evaluator)  # reuse the loaded weights
    evaluator.config = _ModelConfig(per_core_batch_size=64, device="cpu")
    evaluator.model, evaluator.device = tfm3.model, torch.device("cpu")

    x = with_gaps()
    B, T, _ = x.shape
    past = torch.randn(B, T, 1) if covariates else None
    fut = torch.randn(B, T + horizon, 2) if covariates else None
    f = tfm3(x, horizon=horizon, past_covariates=past, future_covariates=fut)
    ref = list(evaluator.predict_batch(
        [x[b].T.numpy() for b in range(B)], horizon=horizon,
        past_only_covariates=None if past is None else [past[b].T.numpy() for b in range(B)],
        past_future_covariates=None if fut is None else [fut[b].T.numpy() for b in range(B)]))
    ref = torch.tensor(np.stack([r.quantiles for r in ref])).permute(0, 2, 1, 3)
    torch.testing.assert_close(f.quantiles, ref, rtol=0, atol=0)


def test_tfm3_variate_limit(tfm3):
    with pytest.raises(ValueError, match="channel_groups"):
        tfm3.encode(sine_trend(B=1, T=64, C=33))


def test_tfm3_channel_groups_isolate(tfm3):
    x = sine_trend()
    y = x.clone()
    y[:, :, 2] += torch.randn_like(y[:, :, 2])
    groups = torch.tensor([0, 0, 1])
    a, b = tfm3.encode(x, channel_groups=groups), tfm3.encode(y, channel_groups=groups)
    assert torch.equal(a[:, :2], b[:, :2]) and not torch.equal(a[:, 2], b[:, 2])
    f = tfm3(x, horizon=16, channel_groups=groups)
    assert f.median.shape == (2, 16, 3)


@pytest.mark.parametrize("horizon", [48, 300])  # 300 > 128: exercises the AR decode
def test_tfm25_is_upstream_forecast(tfm25, horizon):
    from timesfm import configs

    x = with_gaps()
    B, _, C = x.shape
    f = tfm25(x, horizon=horizon)
    rows = [r.numpy() for r in x.transpose(1, 2).flatten(0, 1)]
    longest = max(len(r) - int(np.argmax(~np.isnan(r))) for r in rows)
    tfm25.tfm.compile(configs.ForecastConfig(
        max_context=min(math.ceil(longest / 32) * 32, 15360), max_horizon=1024, per_core_batch_size=128,
        **tfm25.FORECAST_FLAGS))
    _, q = tfm25.tfm.forecast(horizon=horizon, inputs=rows)
    ref = torch.tensor(q[..., 1:]).float().view(B, C, horizon, 9).permute(0, 2, 1, 3)
    torch.testing.assert_close(f.quantiles, ref, rtol=0, atol=0)


def test_tfm25_z_independent_of_horizon(tfm25):
    x = with_gaps()
    assert torch.equal(tfm25.encode(x), tfm25.encode(x, horizon=48))
