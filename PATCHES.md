# Patches to upstream code

Every place where tsfm-zoo changes or works around upstream behaviour. Wrappers
otherwise call upstream's own entry points (encode/decode/predict/generate)
and capture `z` with forward hooks, without modifying the modules.

## Vendored code

Vendored packages live in `src/tsfm_zoo/_vendor/<name>/` with the upstream
LICENSE and an `UPSTREAM` file naming the commit.

- **MOMENT** (moment-timeseries-foundation-model, commit 38f7310, momentfm 0.1.5):
  vendored because the pip package pins old dependencies. Only the imports are
  rewritten to be package-relative; no logic changes.

## Runtime workarounds (no upstream source changed)

- **TimesFM 2.5**: upstream picks the first CUDA device itself when one exists.
  After loading, the wrapper sets `model.device` to the requested device and
  `model.device_count = 1` (`models/timesfm.py`), so it runs on the GPU chosen with `free-gpu`.
- **TiRex-2**: `tirex2.load_model` accepts only `"cuda"`, meaning the current
  device. Loading and inference run inside `torch.cuda.device(i)`
  (`models/tirex.py`).
  - The CUDA path JIT-builds flashrnn kernels through `torch.utils.cpp_extension`.
    That needs `ninja` (in the `main` extra) and an nvcc matching torch's CUDA
    12.8: the system nvcc is 13.x and the pip nvcc wheel has no compiler
    driver. `scripts/setup_cuda.sh` unpacks NVIDIA's CUDA 12.8 redist archives
    into `$CUDA_HOME` (set in `.envrc`, under the cache on /raid).
- **TiRex 1.1**: the GIFT-Eval notebook uses `backend="cuda"` (custom kernels);
  we use `backend="torch"`, which reproduces the leaderboard within 0.06%.
- **Timer / Sundial**: their remote code breaks on transformers 5
  (`DynamicCache.seen_tokens`), so they run in the `legacy` env with
  transformers 4.40.1. The wrapper raises ImportError on transformers >= 5.
