"""FlowState against a direct upstream forward, as IBM's GIFT-Eval wrapper calls it."""
import pytest
import torch

from tsfm_zoo import MissingInputError

from .conftest import importable, sine_trend

pytestmark = pytest.mark.skipif(not importable("flowstate"), reason="granite-tsfm not installed in this env")


@pytest.fixture
def flowstate(model):
    if model.name != "flowstate":
        pytest.skip("FlowState only")
    return model


def test_needs_a_frequency(flowstate):
    with pytest.raises(MissingInputError, match="freq"):
        flowstate.encode(sine_trend())


def test_scale_factor_from_frequency(flowstate):
    assert flowstate._scale_factor("h", None, None) == 1.0
    assert flowstate._scale_factor("10min", None, None) == pytest.approx(1 / 6)
    assert flowstate._scale_factor(None, None, 0.5) == 0.5


def test_is_upstream_forward(flowstate):
    x = sine_trend(B=1, T=300, C=2)
    x[0, :50, 1] = torch.nan  # this series is 250 steps once leading NaNs are dropped
    f = flowstate(x, horizon=40, freq="h")
    for c, start in [(0, 0), (1, 50)]:
        out = flowstate.model(past_values=x[0, start:, c].reshape(-1, 1, 1), scale_factor=1.0, prediction_length=40,
                              batch_first=False)
        ref = out.quantile_outputs.squeeze(-1).transpose(-1, -2)[0].clamp(min=0.0)  # series is positive
        torch.testing.assert_close(f.quantiles[0, :, c], ref, rtol=0, atol=0)


def test_series_with_too_little_data_do_not_crash(flowstate):
    """Upstream fails on fewer than 2 usable steps; the wrapper forecasts those from an all-zero history."""
    nan = float("nan")
    x = torch.linspace(100, 120, 12)[None, :, None].repeat(1, 1, 4)
    x[0, :11, 1] = nan                  # one observed step at the end
    x[0, :, 2] = nan                    # all missing
    x[0, ::4, 3] = nan                  # interior gaps: fine, upstream handles them
    flowstate._warned.discard("short")
    with pytest.warns(UserWarning, match="fewer than 2"):
        f = flowstate(x, horizon=6, freq="5min")
    assert f.median.shape == (1, 6, 4) and torch.isfinite(f.median).all()
    assert (f.median[0, :, 0] > 50).all()                            # a normal series is unaffected
