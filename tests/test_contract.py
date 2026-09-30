"""Contract shared by every wrapper, driven by its class attributes."""
import warnings

import pytest
import torch

from tsfm_zoo import Forecast, MissingInputError, pool

from .conftest import inputs_for, native_task, sine_trend

H = 24


def test_encode_shape(model):
    x = sine_trend()
    z = model.encode(x, **inputs_for(model, x))
    B, T, C = x.shape
    assert z.shape[:2] == (B, C) and z.shape[-1] == model.embed_dim
    assert z.shape[2] == len(model.token_kinds) == len(model.token_span)
    assert model.token_observed.shape == z.shape[:3]
    assert pool(z, model.summary, model.token_kinds, model.token_observed).shape == (B, C, model.embed_dim)


def test_return_scale(model):
    z, (loc, scale) = model.encode(sine_trend(), return_scale=True, **inputs_for(model, sine_trend()))
    assert loc.shape == scale.shape == z.shape[:2]


def test_forecast_shape(forecaster):
    model = forecaster
    x = sine_trend()
    f = model(x, horizon=H, **inputs_for(model, x))
    B, T, C = x.shape
    assert isinstance(f, Forecast)
    assert f.median.shape == (B, H, C)
    if f.quantiles is not None:
        assert f.quantiles.shape == (B, H, C, len(f.quantile_levels))


def test_deterministic(model):
    x = sine_trend()
    torch.manual_seed(1)
    a = model.encode(x, **inputs_for(model, x))
    torch.manual_seed(2)
    b = model.encode(x, **inputs_for(model, x))
    assert torch.equal(a, b)
    task = native_task(model)
    extra = {"horizon": H} if task == "forecast" else {}
    first, second = (model(x, task=task, **extra, **inputs_for(model, x)) for _ in range(2))
    if task == "forecast":
        first, second = first.median, second.median
    assert torch.equal(first, second)


def test_forward_is_head_of_encode(forecaster):
    model = forecaster
    x = sine_trend()
    f = model(x, horizon=H, **inputs_for(model, x))
    z = model.encode(x, horizon=H, **inputs_for(model, x))
    g = model.output_head(z, task="forecast", x=x, horizon=H)
    assert torch.equal(f.median, g.median)


def test_missing_horizon(forecaster):
    model = forecaster
    with pytest.raises(MissingInputError):
        model(sine_trend(), **inputs_for(model, sine_trend()))


def test_missing_required_inputs(model):
    x = sine_trend()
    for name in model.requires:
        others = {k: v for k, v in inputs_for(model, x).items() if k != name}
        with pytest.raises(MissingInputError, match=name):
            model.encode(x, **others)


def test_ignored_input_warns_once(model):
    unused = sorted({"timestamps", "static_covariates", "num_samples"} - model.uses)
    if not unused:
        pytest.skip("model reads every probe input")
    name = unused[0]
    value = {"timestamps": torch.zeros(2, 256, dtype=torch.long), "static_covariates": torch.zeros(2, 1),
             "num_samples": 10}[name]
    model._warned.discard(f"ignored:{name}")
    with pytest.warns(UserWarning, match=name):
        model.encode(sine_trend(), **{name: value}, **inputs_for(model, sine_trend()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.encode(sine_trend(), **{name: value}, **inputs_for(model, sine_trend()))


def test_nan_equals_mask(model):
    x = sine_trend()
    x_nan, mask = x.clone(), torch.ones_like(x, dtype=torch.bool)
    x_nan[0, 100:130, 1] = torch.nan
    mask[0, 100:130, 1] = False
    x_junk = x.clone()
    x_junk[0, 100:130, 1] = 1e6
    assert torch.equal(model.encode(x_nan, **inputs_for(model, x_nan)), model.encode(x_junk, mask=mask, **inputs_for(model, x_nan)))


def test_truncation_warns(model):
    model._warned.discard("truncate")
    x = sine_trend(B=1, T=model.max_context + 40, C=1)
    inputs = inputs_for(model, x)
    with pytest.warns(UserWarning, match="truncated"):
        z = model.encode(x, **inputs)
    if not model.reads_full_history:
        # The truncated input keeps its own timestamps, the last max_context of the full series.
        kept = {k: v[:, -model.max_context:] if k == "timestamps" else v for k, v in inputs.items()}
        assert torch.equal(z, model.encode(x[:, -model.max_context:], **kept))


def test_channel_independence(model):
    x = sine_trend()
    y = x.clone()
    y[:, :, 1] = y[:, :, 1] * 3 + torch.randn_like(y[:, :, 1])
    changed = not torch.allclose(model.encode(x, **inputs_for(model, x))[:, 0], model.encode(y, **inputs_for(model, x))[:, 0])
    assert changed == model.cross_channel


def test_scale_equivariance(forecaster):
    """Forecasts follow an affine map of x; models that only scale (no shift) follow a pure scaling."""
    model = forecaster
    if not model.affine_equivariant:
        pytest.skip(f"{model.name}: sampling is not affine-equivariant under a fixed seed")
    x = sine_trend()
    shift = -7.0 if model.normalization == "affine" else 0.0
    f = model(x, horizon=H, **inputs_for(model, x)).median
    g = model(3.0 * x + shift, horizon=H, **inputs_for(model, x)).median
    torch.testing.assert_close(g, 3.0 * f + shift, rtol=1e-3, atol=1e-2)


def test_sine_smoke(forecaster):
    model = forecaster
    x = sine_trend(T=24 * 20)
    context, target = x[:, :-H], x[:, -H:]
    f = model(context, horizon=H, **inputs_for(model, context))
    mae = (f.median - target).abs().mean()
    persistence = (context[:, -1:] - target).abs().mean()
    assert mae < persistence, (mae, persistence)
