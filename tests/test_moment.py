"""MOMENT against upstream's own embed and reconstruction."""
import pytest
import torch

from .conftest import importable, sine_trend


@pytest.fixture
def moment(model):
    if model.name != "moment":
        pytest.skip("MOMENT only")
    return model


def test_z_is_upstream_embed(moment):
    x = sine_trend(B=2, T=512, C=2)
    ref = moment.model.embed(x_enc=x.transpose(1, 2).reshape(-1, 1, 512), reduction="none").embeddings
    torch.testing.assert_close(moment.encode(x).flatten(0, 1), ref[:, 0], rtol=0, atol=1e-5)


def test_reconstruct_is_upstream(moment):
    x = sine_trend(B=2, T=512, C=2)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 100:140] = False                            # five patches missing in every channel
    out = moment(x, mask=mask, task="reconstruct")
    rows = x.transpose(1, 2).reshape(-1, 1, 512)
    ref = moment.model.reconstruction(x_enc=rows, input_mask=torch.ones(4, 512),
                                      mask=mask.transpose(1, 2).reshape(-1, 512).long()).reconstruction
    torch.testing.assert_close(out, ref[:, 0].view(2, 2, 512).transpose(1, 2), rtol=0, atol=0)
    assert torch.equal(out, moment.output_head(moment.encode(x, mask=mask), task="reconstruct"))


def test_short_series_are_left_padded(moment):
    out = moment(sine_trend(B=1, T=200, C=3), task="reconstruct")
    assert out.shape == (1, 200, 3)


def test_no_zero_shot_forecast(moment):
    with pytest.raises(ValueError, match="forecast"):
        moment(sine_trend(), horizon=8)
