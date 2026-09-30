"""Toto 2.0 against upstream's GluonTS predictor, as the GIFT-Eval notebook runs it."""
import numpy as np
import pytest
import torch

from .conftest import importable, sine_trend

pytestmark = pytest.mark.skipif(not importable("toto"), reason="toto not installed in this env")


@pytest.fixture
def toto(model):
    if model.name != "toto-2":
        pytest.skip("Toto 2 only")
    return model


@pytest.mark.parametrize("horizon", [48, 200])
def test_toto_is_upstream_predictor(toto, horizon):
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig

    x = sine_trend(B=3, T=400, C=2)
    x[0, :100, :] = torch.nan
    x[1, 200:210, 1] = torch.nan
    B, _, C = x.shape
    f = toto(x, horizon=horizon)
    config = Toto2GluonTSModelConfig(prediction_length=horizon, context_length=4096, target_dim=C)
    predictor = Toto2GluonTSModel(toto.model, config).eval().create_predictor(batch_size=512, device="cpu")
    data = ListDataset([{"start": pd.Period("2020-01-01", "h"), "target": x[b].T.numpy()} for b in range(B)],
                       freq="h", one_dim_target=False)
    ref = np.stack([np.stack([fc.quantile(q) for q in f.quantile_levels.tolist()], -1)
                    for fc in predictor.predict(data)])                      # [B, C, H, Q]
    torch.testing.assert_close(f.quantiles, torch.tensor(ref).permute(0, 2, 1, 3), rtol=0, atol=0)


def test_toto_context_shrinks_with_horizon(toto):
    assert toto.context_steps(32) == 4096 and toto.context_steps(720) == 4096 - 22 * 32


def test_toto_channel_groups_isolate(toto):
    x = sine_trend()
    y = x.clone()
    y[:, :, 2] += torch.randn_like(y[:, :, 2])
    groups = torch.tensor([0, 0, 1])
    a, b = toto.encode(x, channel_groups=groups), toto.encode(y, channel_groups=groups)
    assert torch.equal(a[:, :2], b[:, :2]) and not torch.equal(a[:, 2], b[:, 2])
