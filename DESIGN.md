# tsfm-zoo design

`tsfm_zoo` wraps pretrained Time Series Foundation Models behind one interface.
It is the time-series side of the GFM+TSFM zero-shot project on discrete-time
dynamic graphs; `~/gfm-zoo` is the graph side. The two are separate repos that
share conventions, not code. A spatio-temporal combiner imports both.

Priority: **fidelity to upstream first**. A wrapper that does not reproduce a
published number is not done. `encode` feeds a downstream combiner; `forward`
exists mainly to prove the wrapper is faithful.

`~/TSFM-Graphs/tsfmg` is left untouched. Its adapters are a source of quirk
knowledge, and a later cross-check, not code to migrate.

## Models (in implementation order)

| # | family | variants | default | env |
|---|---|---|---|---|
| 1 | chronos | chronos-2, chronos-2-small, bolt-{tiny,mini,small,base}, t5-{tiny,mini,small,base,large} | chronos-2 | main |
| 2 | timesfm | 3.0, 2.5 | 3.0 | main |
| 3 | tirex | tirex-2, tirex-1 | tirex-2 | main |
| 4 | toto | 2.0-{4m,22m,313m,1B,2.5B} | 2.0-313m | main |
| 5 | ttm | r3, r2 (branch fixed at load) | r3 | main |
| 6 | flowstate | r1.1 | r1.1 | main |
| 7 | timer | sundial-128m, timer-84m | sundial-128m | legacy |
| 8 | moment | large, base, small | large | main |
| 9 | moirai | 2.0-small, moe-{small,base}, 1.1-{small,base,large} | 2.0-small | legacy |

Lag-Llama was implemented and then excluded (2026-09-29, user decision). Its
samples matched upstream's README recipe bit for bit, but it did not reproduce
its leaderboard entry (no replication code published) and forecast worse than
persistence on the contract smoke test. The wrapper and vendored code are
archived in `$TSFM_ZOO_CACHE/excluded/lag-llama.tar.gz`.

Status at v0.1.0: every family above is implemented except **ttm** (row 5), which is
planned; how to validate it against IBM's leaderboard entries is undecided.

Within #1, Chronos-2 is built first and the interface is reviewed with the user
before anything else. Bolt and T5 follow.

Out of scope:
- **Toto 1**: needs its own exact-pinned env; tsfmg already covers it.
- **Timer-XL**: no HF checkpoint (only OpenLTM); timer-84m carries its improvements.
- **Timer-S1**: 8.3B parameters, ≥40GB VRAM.
- **TSPulse, PatchTST-FM**: not requested.

## API

```python
model = load_tsfm(family, variant=None, *, device, seed=0, frozen=True, **load_cfg)

z   = model.encode(x, mask=None, timestamps=None, *, <model inputs>)
out = model.forward(x, mask=None, timestamps=None, *, task="forecast", <model inputs>)
    # == model.output_head(model.encode(...), task=task, <model inputs>)
model.set_head(head)        # nn.Module: (z, **inputs) -> out
z_pooled = pool(z, model.summary)   # explicit helper; encode never pools
```

- `x`: `[B, T, C]` float, raw values.
- `mask`: `[B, T, C]`, 1 = observed. NaN in `x` is treated like `mask = 0`.
- `timestamps`: `[B, T]` int64 epoch **seconds**.

### Keyword inputs (all keyword-only)

| name | shape | meaning |
|---|---|---|
| `freq` | str | pandas frequency; overrides the one inferred from `timestamps` |
| `horizon` | int | required for `task="forecast"`; no default |
| `quantile_levels` | list[float] | default 0.1…0.9 (GIFT-Eval) |
| `interpolate` | bool | allow levels the model does not emit natively |
| `num_samples` | int | for sample-based models |
| `past_covariates` | `[B, T, K]` | known only in the past |
| `future_covariates` | `[B, T+H, K]` | known in past and future |
| `static_covariates` | `[B, S]` | per series |
| `channel_groups` | `[C]` long | opt-in grouping of C for cross-channel models |
| `layer` | int | hidden layer to return, where the model is a plain stack |
| `scale_factor`, `patch_size` | | per-call overrides of values derived from `freq` |

A `Batch` dataclass holds these fields; `model(**batch)` unpacks it.

### Missing and ignored inputs
Same convention as gfm-zoo.
- **Missing required input:** raises `MissingInputError`, naming the argument
  and what it means for this model. No silent fallbacks.
- **Ignored input:** a provided but unused input warns once per instance.

### Class attributes (source of truth for README and contract tests)
- `uses: set[str]`: the inputs the model reads.
- `tasks: dict[str, "native" | "probe" | "unsupported"]`.
- `embed_dim: int`.
- `cross_channel: bool`: whether the backbone mixes information across C.
- `max_context: int`, `patch_size: int | None`.
- `summary: "last" | "mean" | "reg"`: recommended pooling (causal models `last`).
- `token_kinds`: marks special tokens in `z` (REG, EOS, future patch).
- `training_corpora`: for the README's contamination notes (for example,
  GIFT-Eval-Pretrain contains PeMS03/04/07/08).
- `requires` / `needs_one_of`: inputs that must be given (FlowState needs one of
  `freq`, `timestamps`, `scale_factor`).
- Contract-test exemptions, each stating an upstream property:
  - `normalization: "affine" | "scale"`: whether forecasts follow a shift of x
    or only a rescaling (Chronos-T5 is mean-scaled).
  - `affine_equivariant`: False where a fixed seed does not give equivariant
    samples (Moirai-MoE samples its mixture by rejection).
  - `reads_full_history`: True where upstream preprocessing looks past
    `max_context`, so truncating the input changes `z` (TiRex-2's trend check).

## Channels

`C` means the **variates of one entity** by default. To forecast graph nodes
independently, the caller folds nodes into B. To let a cross-channel model
(Chronos-2, TimesFM 3.0, TiRex-2, Toto 2, Moirai) learn relations between
nodes, the caller deliberately puts nodes in C.

- There is no soft limit on C (interface review): Chronos-2 and Toto
  attention is O(C²) and the caller owns the cost. Only a hard upstream limit
  raises (TimesFM 3.0: 32 variates).
- `channel_groups=` is the explicit opt-in to split C into groups that attend
  within themselves. The zoo never groups on its own.
- Univariate models loop over C independently.
- `z` is always per channel, so a learned cross-node mixer downstream works the
  same either way.

## What `encode` returns

`z: [B, C, N, D]`, where N counts tokens.
- Tokens are patches, or timesteps for Chronos-T5.
- D is the native dimension. The zoo never projects `z`.
- The last layer is the default; `layer=` selects another.
- `model.token_span` maps each token to its time indices.

| model | `z` source | token | D | summary |
|---|---|---|---|---|
| Chronos-2 | `pipeline.embed` | patch 16, + REG + 1 future-patch token | 768 / 512 | reg |
| Chronos-Bolt | `pipeline.embed` | patch 16, + REG | 256–768 | reg |
| Chronos-T5 | `pipeline.embed` | per timestep, + EOS | 256–1024 | mean |
| TimesFM 3.0 | aux `transformer_output` | patch 32 | 1280 | last |
| TimesFM 2.5 | `output_embeddings` | patch 32 | 1280 | last |
| TiRex-1 | `_embed_context`, last layer | patch 32 | 512 | last |
| TiRex-2 | pre-head hidden state | patch 32 | 512 | last |
| Toto 2 | pre-`output_head` output | patch 32 | 256–2048 | last |
| TTM | `backbone_hidden_state` | patch (per branch) | per branch | mean |
| FlowState | final SSM state | one token | 512 | last |
| Sundial / Timer | `last_hidden_state` | patch 16 / 96 | 768 / 1024 | last |
| MOMENT | `embed(reduction="none")` | patch 8, fixed 64 tokens | 512–1024 | mean |
| Moirai 1.1 | encoder output before `param_proj`, unflattened | patch 8–128 | 384–1024 | mean |
| Moirai-MoE / 2.0 | decoder output before head | patch 16 | 384 | last |

- **Moirai 1.1** is a masked encoder. `encode` uses one mask patch and
  returns **context tokens only**, so `z` does not depend on the horizon.
  The patch size is fixed at load (32; 16 for MoE), as in the GIFT-Eval
  notebook, rather than upstream's per-frequency `"auto"`; `patch_size=` at
  call time must match it.
- **Same-pass heads.** For TimesFM, TiRex, Toto, FlowState, Moirai 2.0,
  and Sundial/Timer, `z` is captured with a forward hook during
  upstream's own inference call, and the native head returns that call's
  forecast instead of re-deriving it from `z`. `check_head_state` enforces
  that the head receives the `z` of the latest `encode`. The table's `z`
  source column names the hooked module's output.
- **Chronos-2's future-patch tokens** are kept and flagged in `token_kinds`.
  The encoder is bidirectional, so context tokens attend to them and `z`
  depends on their number F: without `horizon`, F = 1 (exactly upstream's
  `embed`); with `horizon`, F = ceil(horizon / 16), which the native head
  reads. They are conditioned on future covariates when those are given.
- **N differs across models.** Only the pooled `[B, C, D]` is comparable.
- **Normalization:** `z` lives in the model's internally normalized space, so it
  carries no level or scale. `encode(..., return_scale=True)` and
  `model.last_scale` expose the per-channel `(loc, scale)`.
- Covariates enter `z` where the model conditions on them (Chronos-2, TimesFM
  3.0, TiRex-2, Moirai). The README states this per model.

## Tasks and outputs

| task | where native | output |
|---|---|---|
| `forecast` | every model except MOMENT (its head needs training) | `Forecast` |
| `reconstruct` | MOMENT | `[B, T, C]`; used for imputation and anomaly scoring |
| `classify`, `regress` | none; probe heads on pooled `z`, README says "probe, not native" | logits / values |

There is no anomaly detection built on forecast residuals. That is a method,
not a model capability.

`Forecast` dataclass:

| field | shape | when |
|---|---|---|
| `median` | `[B, H, C]` | always (point forecast) |
| `quantiles` | `[B, H, C, Q]` | when native, or computed from samples |
| `quantile_levels` | `[Q]` | with `quantiles` |
| `samples` | `[B, S, H, C]` | only when native |
| `dist` | torch distribution | only when native |
| `rollout_steps` | int | autoregressive steps used past the native horizon |

- Point-only models set `quantiles=None`.
- Levels a model does not emit natively raise an error unless `interpolate=True`.
- FlowState's point forecast is the median (`prediction_type="median"`), not
  upstream's default mean.
- Rollout past the native horizon follows upstream's own `predict` logic.

## Model-specific inputs
- **Normalization:** the wrapper feeds raw values and replicates upstream's
  pipeline exactly. Forecasts come back in the original scale.
- **Missing values:** `mask` is converted to the model's convention: NaN, an
  explicit mask, or the upstream-recommended fill plus a warning. Variable-length
  series in one batch are left-padded with `mask = 0`.
- **Context too long:** the wrapper keeps the most recent `max_context` steps and
  warns once. This is upstream behaviour, not a fallback.
- **Context too short or unaligned:** left-padded with `mask = 0`, as upstream does.
- **Frequency:** inferred from `timestamps`. Irregular timestamps with no `freq`
  raise `MissingInputError` for models that need a frequency.
- **Load-time vs. call-time configuration:**
  - What **selects weights** is fixed at load, e.g.
    `load_tsfm("ttm", context_length=512, horizon=96)`. The default is upstream's
    default branch. A call whose horizon exceeds the branch raises.
  - What is **derived from data** is computed per call from `freq` using
    upstream's rule, and can be overridden by a keyword argument. Examples:
    FlowState `scale_factor = 24 / season`. (Moirai 1.1's `patch_size` is
    fixed at load instead; see above.)

## Determinism
`load_tsfm(seed=...)`. Sampling models (Chronos-T5, Moirai 1.x and MoE,
Sundial) run upstream's sampling inside `torch.random.fork_rng` seeded with
the model's seed, so `z` and forecasts are reproducible and the caller's global
RNG state is left untouched. No upstream code is patched for this.

## Modes and devices
- `frozen=True` (default) sets `requires_grad=False`, calls `.eval()` explicitly
  and logs it. `encode` never switches mode. Finetuning is possible (wrappers are
  `nn.Module`s) but untested.
- The device is always explicit; `cuda:0` is never hardcoded.
  - `--device auto` uses `~/.claude/free-gpu` and falls back to CPU.
  - TimesFM 2.5 picks the first GPU itself. The wrapper sets its device to the
    requested one after construction.
  - TiRex-2 accepts only "cuda" (the current device); the wrapper runs it inside
    `torch.cuda.device(i)`. Both are listed in `PATCHES.md`.

## Packaging
- One pyproject and one `uv.lock`, **Python 3.12**. Each env group is an optional
  extra, declared mutually exclusive through `[tool.uv] conflicts`:

| env | models | key pins |
|---|---|---|
| `main` | chronos, timesfm, tirex (1, 2), toto 2, ttm, flowstate, moment | torch 2.11, numpy 2.x, transformers 5.x, gluonts ≥0.16 |
| `legacy` | moirai, timer/sundial | torch 2.4, numpy 1.26, gluonts 0.14.4, transformers 4.40.1 |

  - Each extra syncs into `/raid/f.decastelli/venvs/tsfm-zoo-<env>`.
  - The core package (base class, registry, `Forecast`, metrics, pooling) depends
    only on torch and numpy, so it imports in every env.
- **Setup checks (2026-09-28):** uni2ts installs and imports on 3.12.
  Timer/Sundial remote code fails on transformers 5 (`DynamicCache.seen_tokens`)
  and runs on the card's 4.40.1, so it moved to `legacy`.
- **Cross-env use:** there is no RPC. Frozen `z` is cached to disk (`--save-z`),
  and the combiner reads the files.
- **How upstream code enters:**
  - pinned PyPI releases: chronos-forecasting, timesfm, uni2ts, tirex-ts,
    tirex-2, toto-2, granite-tsfm;
  - **vendored** under `src/tsfm_zoo/_vendor/`, each pinned to a commit with its
    upstream LICENSE: MOMENT 0.1.5 from GitHub main
    (the PyPI 0.1.4 wheel pins transformers 4.33);
  - Timer/Sundial remote code is loaded from HF at a pinned revision;
  - patches are minimal monkeypatches in the wrapper, each listed in `PATCHES.md`.

## Checkpoints and licences
- Downloaded lazily on `load_tsfm` into `$TSFM_ZOO_CACHE` (under `/raid`), with
  `HF_HOME` set there too.
- `checkpoints.toml` lists, per family and variant: HF repo, revision and
  licence. The revision is a commit hash, which is the integrity pin (HF
  content-addresses every file in a commit), so there is no separate sha256.
  TTM branches are resolved to their commit.
- Licence notes are printed once on first load:
  - TimesFM 3.0 weights: non-commercial, non-production.
  - Moirai (all): CC-BY-NC-4.0.
  - TiRex-1: NXAI Community License.
  - All others are Apache-2.0 or MIT.

## Tests
There is no remote CI. `scripts/test_all.sh` runs pytest in each env.

- **Default (CPU):** the smallest variant per family, skipped if not cached;
  nothing large is downloaded implicitly. Shared contract tests driven by the
  class attributes:
  - shapes of `z` and `Forecast`;
  - determinism under a fixed seed;
  - `forward == head(encode)`;
  - `MissingInputError` and the ignored-input warning;
  - NaN equals `mask = 0`;
  - the truncation warning;
  - channel independence: for `cross_channel=False`, perturbing channel 1
    leaves `z[:, 0]` unchanged; for `True`, it changes;
  - scale equivariance: `forecast(a·x + b) ≈ a·forecast(x) + b`;
  - a smoke test on a sine+trend series and on ETTh1.
- `download`: fetches checkpoints.
- `repro`: the GIFT-Eval subset on GPU (see below).
- `dtdg`: encodes PeMS08 nodes; checks shapes and determinism.

## Reproduction and definition of done
- **Forecasters:** a fixed subset of about 6 GIFT-Eval configs spanning
  frequencies and horizons.
  - Candidates: `m4_hourly`, `electricity/H/short`, `ett1/H/{short,long}`,
    `solar/10T/short`, `us_births/D/short`, `saugeenday/W/short`.
  - The final list is chosen from the public per-config leaderboard results.
  - MASE and CRPS must be within **±2% relative** of the model's own leaderboard
    entry. Models not on the leaderboard use their paper or card numbers.
- **MOMENT:** zero-shot imputation MSE on ETTh1 from its paper, within ±2%.
- **Metrics:** our own numpy MASE (seasonal naive, season from frequency) and
  CRPS (weighted quantile loss over 9 levels). They are checked once against
  gluonts on one config.
- **Data:** GIFT-Eval read from HF `Salesforce/GiftEval` with the leaderboard's
  windowing; ETTh1 for smoke tests.

The loop for each model:
1. Implement.
2. Run the contract tests.
3. Run the repro subset **immediately**.
4. If a config is more than 2% off, diagnose (context length, scaling, windowing,
   quantile levels, dtype, missing-value handling), fix, and rerun.
5. **Stop rule:** if a gap survives 3 iterations with the diagnosed causes
   exhausted, write up the evidence and ask the user. Tuning undocumented
   settings until the number matches is fitting, not reproducing.

A model is done when:
1. its contract tests pass on CPU;
2. every repro config is within tolerance;
3. it has a README row (inputs used, what `z` is, limitations, training corpora);
4. its patches are listed in `PATCHES.md`.

## Window embeddings and the tgdata evaluation (v0.2.0)

tgdata (FabriDeCastelli/tgdata, private, pinned tag v0.2.2) is the data and batching base for temporal
graphs, behind the optional `tgdata` extra.

- `embeddings.embed_windows(model, g, cache_dir, window=W)` follows tgdata's "every window of a series"
  recipe (`split=None, horizon=0`) and writes `[windows, nodes, D]` with the tokens pooled (default
  `model.summary`; a dataset with one channel per node). The nodes are the model's channels.
- The cache directory name is the SHA-256 (`hashlib`) of the canonical JSON of the model
  (family, variant, pinned checkpoint revision, seed, load options), the dataset (name, tgdata version,
  shape, fingerprint of the readings), the window, normalisation, pooling, dtype and extra `encode`
  inputs. Not the device or batch size. `config.json` holds it in full; runs are written to a temporary
  directory and renamed, so a cache is complete or absent. `load_tsfm` records `model.spec` for this.
- tgdata puts batches on the CUDA default device unless told otherwise; every call here passes
  `device="cpu"` for the data, and only the model goes on the GPU chosen with `free-gpu`.

## Example
`examples/compare.py --model chronos[:bolt-small] | all --dataset gifteval:<config> | etth1 --device auto [--save-z out.pt]`

- Runs the native head with GIFT-Eval's windowing.
- Prints MASE and CRPS next to the leaderboard reference.
- `all` re-launches itself in each model's env through `uv run` and merges the
  tables. This is the only subprocess in the project, and it stays out of the
  library.

## Repo
- `~/tsfm-zoo` is its own git repo, package `tsfm_zoo`, `src/` layout with
  hatchling.
- `.envrc` sets `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, `TSFM_ZOO_CACHE` and
  `HF_HOME`, all under `/raid/f.decastelli/`.
- No commits until asked.
