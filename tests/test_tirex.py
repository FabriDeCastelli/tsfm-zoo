"""TiRex-1 and TiRex-2 against upstream's own forecast API."""
import pytest
import torch

from .conftest import importable, sine_trend

pytestmark = pytest.mark.skipif(not importable("tirex"), reason="tirex not installed in this env")


@pytest.fixture
def tirex1(model):
    if model.name != "tirex-1":
        pytest.skip("TiRex-1 only")
    return model


@pytest.fixture
def tirex2(model):
    if model.name != "tirex-2":
        pytest.skip("TiRex-2 only")
    return model


def with_gaps():
    x = sine_trend(B=2, T=300, C=2)
    x[0, :60, 0] = torch.nan
    x[1, 100:110, 1] = torch.nan
    return x


@pytest.mark.parametrize("horizon", [48, 150])  # 150 > 32: upstream appends NaN patches
def test_tirex1_is_upstream_forecast(tirex1, horizon):
    x = with_gaps()
    B, _, C = x.shape
    f = tirex1(x, horizon=horizon)
    q, _ = tirex1.model.forecast(x.transpose(1, 2).flatten(0, 1), prediction_length=horizon)
    torch.testing.assert_close(f.quantiles, q.view(B, C, horizon, 9).permute(0, 2, 1, 3), rtol=0, atol=0)


def test_tirex1_z_is_upstream_embedding(tirex1):
    x = sine_trend(B=2, T=2048, C=2)
    ref = tirex1.model._embed_context(x.transpose(1, 2).flatten(0, 1), max_context=2048)[:, :, -1]
    assert torch.equal(tirex1.encode(x).flatten(0, 1), ref)


@pytest.mark.parametrize("horizon", [48, 300])
def test_tirex2_is_upstream_forecast(tirex2, horizon):
    import tirex2 as upstream

    x = with_gaps()
    f = tirex2(x, horizon=horizon)
    ref = tirex2.forecaster.forecast([upstream.TimeseriesType(x[b].T, None, None) for b in range(x.shape[0])],
                                     prediction_length=horizon)
    ref = torch.stack(ref) if isinstance(ref, list) else ref                       # [B, C, Q, H]
    torch.testing.assert_close(f.quantiles, ref.permute(0, 3, 1, 2), rtol=0, atol=0)


def test_tirex2_z_independent_of_horizon(tirex2):
    x = with_gaps()
    assert torch.equal(tirex2.encode(x), tirex2.encode(x, horizon=48))


def test_tirex2_horizon_limit(tirex2):
    with pytest.raises(ValueError, match="at most"):
        tirex2(sine_trend(), horizon=tirex2.max_horizon + 1)


def test_tirex2_channel_groups_isolate(tirex2):
    x = sine_trend()
    y = x.clone()
    y[:, :, 2] += torch.randn_like(y[:, :, 2])
    groups = torch.tensor([0, 0, 1])
    a, b = tirex2.encode(x, channel_groups=groups), tirex2.encode(y, channel_groups=groups)
    assert torch.equal(a[:, :2], b[:, :2]) and not torch.equal(a[:, 2], b[:, 2])
