# tsfm-zoo

Pretrained time-series foundation models behind one interface. Design: [DESIGN.md](DESIGN.md).

```python
from tsfm_zoo import load_tsfm, pool

model = load_tsfm("chronos", "chronos-2", device="cuda:3")
z = model.encode(x)                              # x [B, T, C] -> z [B, C, N, D]
zp = pool(z, model.summary, model.token_kinds, model.token_observed)  # [B, C, D]
f = model(x, horizon=48)                         # Forecast: median [B, H, C], quantiles [B, H, C, Q]
```

## Environments

| env | models | setup |
|---|---|---|
| `main` | chronos, timesfm, tirex, toto, flowstate, moment | `source .envrc && uv sync --extra main` |
| `legacy` | moirai, timer/sundial | `TSFM_ENV=legacy source .envrc && uv sync --extra legacy` |

`legacy` exists because uni2ts pins gluonts 0.14 / torch 2.4, and
the Timer/Sundial remote code breaks on transformers 5.
`scripts/test_all.sh` runs the CPU tests in every env. `pytest -m repro` runs the GIFT-Eval reproduction on a free GPU.

## Models

Every model takes `x [B, T, C]` (NaN = missing) and `mask`. "Univariate"
models fold C into the batch, so channels never interact. P is the number
of context patches. The native forecast is always upstream's own
inference call; where `z` and the forecast come from one pass, `z` is
captured with a forward hook and the head returns that pass's forecast.

| model (`family:variant`) | extra inputs used | `encode` returns | cross-channel | GIFT-Eval repro (6 configs) | limitations |
|---|---|---|---|---|---|
| Chronos-2 (`chronos:chronos-2`, `chronos-2-small`) | numeric `past_covariates` / `future_covariates`, `channel_groups`, `layer` | `[B, C, P+1+F, 768]` (small: 512): P patches of 16, the REG token, F future-patch tokens. F = 1 without `horizon` (= upstream `embed`), ceil(horizon/16) with it: the encoder is bidirectional, so `z` depends on F | yes, group attention over each item's channels and covariates | exact (Δ 0.00%) | categorical covariates not wrapped; `channel_groups` cannot be combined with covariates; group attention is O(C²) |
| Chronos-Bolt (`chronos:bolt-{tiny,mini,small,base}`) | `layer` | `[B, C, P+1, D]` encoder output, patches of 16 plus REG; D = 256/384/512/768 | no | small, base: exact on 5 configs; ett1/H/long CRPS −6 to −8% | the leaderboard used chronos ≤1.5's rollout past 64 steps (median fed back); we use the current one (quantile paths). The old rollout reproduces the leaderboard to 6 decimals |
| Chronos (T5) (`chronos:t5-{tiny,mini,small,base,large}`) | `layer`, `num_samples` | `[B, C, T'+1, D]` encoder output, one token per step (context ≤ 512) plus EOS; D = 256…1024 | no | small: within 1.4%; base: within seed noise except m4 CRPS; large: m4 CRPS −22%, solar −5%, unexplained | sampled (20 paths in the leaderboard runs), so results move with the seed; slow for long horizons |
| TimesFM 3.0 (`timesfm:3.0`) | `past_covariates`, `future_covariates`, `channel_groups`, `layer` | `[B, C, P, 1280]`, patches of 32, from the same `decode` pass as the forecast | yes, variate attention (at most 32 variates) | exact | non-commercial licence; `symmetric` doubles the cost |
| TimesFM 2.5 (`timesfm:2.5`) | `layer` | `[B, C, P, 1280]` prefill output embeddings, patches of 32 | no | exact | upstream sets its device itself; the wrapper overrides it ([PATCHES.md](PATCHES.md)) |
| TiRex-1 (`tirex:tirex-1`, `tirex-1.1-gifteval`) | `layer` | `[B, C, P, 512]`, patches of 32, context 2048 | no | 1.1-gifteval: within 0.06% | run with `backend="torch"`; the notebook uses the CUDA kernels. With `resample_strategy="frequency"`, `token_span` is None |
| TiRex-2 (`tirex:tirex-2`, `tirex-2-gifteval-{zs,pretrain}`) | `past_covariates`, `future_covariates`, `channel_groups`, `layer` | `[B, C, N, 512]`: context (2048, NaN-padded) plus future patches, independent of the horizon | yes, variate attention | gifteval-zs, gifteval-pretrain: within 0.72% (GPU, CUDA kernels) | the CUDA kernels are JIT-built on first use and need `scripts/setup_cuda.sh` (CUDA 12.8 toolkit); GPU runs in bf16, so GPU and CPU forecasts differ by ~0.1%; trend differencing reads the full history, so truncation changes `z` |
| Toto 2.0 (`toto:2.0-{4m,22m,313m,1B,2.5B}`) | `past_covariates`, `future_covariates`, `channel_groups`, `layer` | `[B, C, P+F, D]`, context patches of 32 plus one token per future patch; D = 256/512/1024/1536 (4m…1B) | yes, variate attention | 4m, 22m, 313m, 1B: exact (2.5B not run) | the context is `4096 − (ceil(H/32)−1)·32`, so `z` depends on the horizon |
| FlowState (`flowstate:r1.1`) | one of `freq`, `timestamps`, `scale_factor` | `[B, C, 1, 512]`: the S5 state after the last context step | no | exact | needs the time scale; context is `4096 / scale_factor` steps; one token, so no per-patch features |
| MOMENT (`moment:{small,base,large}`) | `layer` | `[B, C, 64, D]`, 64 patches of 8 over a fixed 512-step window; D = 512/768/1024 | no | no forecast (head must be trained); ETTh1 zero-shot imputation MSE 0.4024 vs the paper's 0.402 (large) | native task is `reconstruct`; longer inputs are truncated to 512 steps |
| Sundial (`timer:sundial-base-128m`) | `layer`, `num_samples` | `[B, C, P, 768]`, patches of 16 over the last 2880 steps | no | within 0.7%, except m4 CRPS −2.4% (sampled) | runs in `legacy` (transformers 4.40) |
| Timer (`timer:timer-base-84m`) | `layer` | `[B, C, P, 1024]`, patches of 96 over the last 2880 steps | no | no leaderboard entry | point forecast only (no quantiles) |
| Moirai 1.x (`moirai:1.{0,1}-{small,base,large}`) | `num_samples` | `[B, C, P, D]`, context tokens of a separate pass with one masked patch; patch 32, context 4000; D = 384/768/1024 | yes, any-variate attention over the item's channels | 1.1-small: electricity +4 to +6%; 1.1-base: solar and saugeen +5 to +12%. Samples are bit-identical to uni2ts' own GluonTS predictor (uni- and multivariate), so the gap is in the leaderboard run, not the wrapper | sampled (100 paths); patch size fixed at load, where upstream picks it by frequency |
| Moirai-MoE (`moirai:moe-{small,base}`) | `num_samples` | as Moirai 1.x, patch 16 | yes | no leaderboard entry | not affine-equivariant under a fixed seed (rejection sampling in its mixture) |
| Moirai 2.0 (`moirai:2.0-small`) | none | `[B, C, P, 384]`, patches of 16 | no | exact | |

TTM (Tiny Time Mixers) is planned and not implemented yet.

Training corpora (for contamination) are in each class's `training_corpora`.
TiRex, Toto, FlowState, Moirai, TimesFM 3.0 and Sundial train on data that
include GIFT-Eval-Pretrain or LOTSA, which contain PEMS03/04/07/08; TimesFM
2.5's corpus is undisclosed.

## Reproduction

`python examples/compare.py --model chronos:chronos-2 --dataset gifteval:repro --device auto`
runs the fixed subset (m4_hourly/H/short, electricity/H/short, ett1/H/short,
ett1/H/long, solar/10T/short, saugeen/W/short) with each leaderboard
notebook's settings (`tsfm_zoo.gifteval.LEADERBOARD`) and prints MASE/CRPS
next to the model's leaderboard row.

| config | Chronos-2 MASE | ref | CRPS | ref |
|---|---|---|---|---|
| m4_hourly/H/short | 0.7884 | 0.7884 | 0.0232 | 0.0232 |
| electricity/H/short | 0.9779 | 0.9779 | 0.0679 | 0.0679 |
| ett1/H/short | 0.7868 | 0.7868 | 0.1706 | 0.1706 |
| ett1/H/long | 1.3805 | 1.3805 | 0.2752 | 0.2752 |
| solar/10T/short | 0.7784 | 0.7784 | 0.3860 | 0.3860 |
| saugeen/W/short | 1.1876 | 1.1876 | 0.3469 | 0.3469 |

Chronos-2's leaderboard entry forecasts series jointly (cross-learning, 100
series per group), so the reproduction puts them side by side in C.
