"""Chronos-2 against upstream's own pipeline calls."""
import warnings

import pytest
import torch

from .conftest import SMALLEST, importable, sine_trend

pytestmark = pytest.mark.skipif(not importable("chronos"), reason="chronos not installed in this env")


@pytest.fixture
def chronos2(model):
    if model.name != "chronos-2":
        pytest.skip("Chronos-2 only")
    return model


def test_encode_is_upstream_embed(chronos2):
    x = sine_trend()
    x[0, :40, 0] = torch.nan
    ref, loc_scale = chronos2.pipeline.embed(x.transpose(1, 2))
    z, (loc, scale) = chronos2.encode(x, return_scale=True)
    for b in range(x.shape[0]):
        assert torch.equal(z[b], ref[b])
        assert torch.equal(loc[b], loc_scale[b][0].squeeze(-1))


@pytest.mark.parametrize("horizon", [40, 1100])  # 1100 > 64 patches: exercises the unroll
def test_forecast_is_upstream_predict(chronos2, horizon):
    x = sine_trend()
    B, T, _ = x.shape
    past, fut = torch.randn(B, T, 1), torch.randn(B, T + horizon, 1)
    f = chronos2(x, horizon=horizon, past_covariates=past, future_covariates=fut)
    inputs = [{"target": x[b].T.numpy(),
               "past_covariates": {"p": past[b, :, 0].numpy(), "f": fut[b, :T, 0].numpy()},
               "future_covariates": {"f": fut[b, T:, 0].numpy()}} for b in range(B)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q, _ = chronos2.pipeline.predict_quantiles(inputs, prediction_length=horizon)
    torch.testing.assert_close(f.quantiles, torch.stack(q).permute(0, 2, 1, 3), rtol=0, atol=0)
    assert f.rollout_steps == (1 if horizon > 1024 else 0)


def test_channel_groups_isolate(chronos2):
    x = sine_trend()
    y = x.clone()
    y[:, :, 2] += torch.randn_like(y[:, :, 2])
    groups = torch.tensor([0, 0, 1])
    a, b = chronos2.encode(x, channel_groups=groups), chronos2.encode(y, channel_groups=groups)
    assert torch.equal(a[:, :2], b[:, :2])
    assert not torch.equal(a[:, 2], b[:, 2])


def test_unknown_quantile_level_raises(chronos2):
    with pytest.raises(ValueError, match="interpolate"):
        chronos2(sine_trend(), horizon=16, quantile_levels=[0.15])
    f = chronos2(sine_trend(), horizon=16, quantile_levels=[0.15], interpolate=True)
    assert f.quantiles.shape[-1] == 1


def test_layer_hook(chronos2):
    z = chronos2.encode(sine_trend(), layer=0)
    assert z.shape == chronos2.encode(sine_trend()).shape
    with pytest.raises(ValueError, match="layer"):
        chronos2(sine_trend(), horizon=16, layer=0)


@pytest.fixture
def bolt(model):
    if model.name != "chronos-bolt":
        pytest.skip("Chronos-Bolt only")
    return model


@pytest.fixture
def t5(model):
    if model.name != "chronos-t5":
        pytest.skip("Chronos T5 only")
    return model


def test_bolt_encode_is_upstream_embed(bolt):
    x = sine_trend()
    x[0, :40, 0] = torch.nan
    ref, (loc, scale) = bolt.pipeline.embed(x.transpose(1, 2).flatten(0, 1))
    z, (our_loc, our_scale) = bolt.encode(x, return_scale=True)
    assert torch.equal(z.flatten(0, 1), ref)
    assert torch.equal(our_loc.flatten(), loc)


@pytest.mark.parametrize("horizon", [40, 150])  # 150 > 64: exercises the rollout
def test_bolt_forecast_is_upstream_predict(bolt, horizon):
    x = sine_trend()
    f = bolt(x, horizon=horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = bolt.pipeline.predict(x.transpose(1, 2).flatten(0, 1), prediction_length=horizon)  # [B*C, Q, H]
    B, _, C = x.shape
    torch.testing.assert_close(f.quantiles, ref.view(B, C, 9, horizon).permute(0, 3, 1, 2), rtol=0, atol=0)
    assert f.rollout_steps == (2 if horizon > 128 else 0)


def test_t5_encode_is_upstream_embed(t5):
    x = sine_trend()
    x[0, :40, 0] = torch.nan
    ref, scale = t5.pipeline.embed(x.transpose(1, 2).flatten(0, 1))
    z, (_, our_scale) = t5.encode(x, return_scale=True)
    assert torch.equal(z.flatten(0, 1), ref)
    assert torch.equal(our_scale.flatten(), scale)


def test_t5_forecast_is_upstream_predict(t5):
    """Within the native horizon there is one generate call; seeding the global RNG
    the way the wrapper seeds its own makes upstream draw the same samples."""
    x = sine_trend()
    f = t5(x, horizon=40, num_samples=8)
    torch.manual_seed(t5.seed)
    ref = t5.pipeline.predict(x.transpose(1, 2).flatten(0, 1), prediction_length=40, num_samples=8)
    B, _, C = x.shape
    torch.testing.assert_close(f.samples, ref.view(B, C, 8, 40).permute(0, 2, 3, 1), rtol=0, atol=0)


def test_t5_rollout(t5):
    f = t5(sine_trend(), horizon=100, num_samples=4)
    assert f.samples.shape == (2, 4, 100, 3) and f.rollout_steps == 1
