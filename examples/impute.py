"""Zero-shot imputation on ETTh1 with MOMENT's protocol (momentfm InformerDataset + Masking).

Test split = the last 4 months of the first 20 (plus 512 steps of lead-in),
standardized with train statistics; 512-step windows; each channel is
masked separately with random 8-step patches; MSE on the masked steps,
averaged over mask ratios 12.5/25/37.5/50%.

    python examples/impute.py --model moment:large --stride 1 --device auto
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsfm_zoo import load_tsfm
from tsfm_zoo._vendor.moment.utils.masking import Masking

from compare import pick_device

RATIOS = (0.125, 0.25, 0.375, 0.5)


def ett_test_windows(path: Path, stride: int, seq_len: int = 512) -> np.ndarray:
    """[N, 512, 7] windows, as momentfm.data.InformerDataset(task_name="imputation")."""
    df = pd.read_csv(path).drop(columns=["date"]).interpolate(method="cubic")
    n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    mean, std = df.values[:n_train].mean(0), df.values[:n_train].std(0)
    data = (df.values - mean) / std
    test = data[n_train + n_val - seq_len: n_train + n_val + n_test]
    return np.stack([test[s:s + seq_len] for s in range(0, len(test) - seq_len + 1, stride)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moment:large")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data", default=os.path.join(os.environ.get("TSFM_ZOO_CACHE", "."), "ett", "ETTh1.csv"))
    parser.add_argument("--out")
    args = parser.parse_args()

    family, _, variant = args.model.partition(":")
    model = load_tsfm(family, variant or None, device=pick_device(args.device), seed=args.seed)
    windows = torch.tensor(ett_test_windows(Path(args.data), args.stride), dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    results = {}
    for ratio in RATIOS:
        masker = Masking(mask_ratio=ratio)
        sq_err, n = 0.0, 0
        for start in range(0, len(windows), args.batch_size):
            x = windows[start:start + args.batch_size]                          # [B, 512, C]
            B, T, C = x.shape
            # Masking draws from the global RNG; fork it so this script stays seedable.
            with torch.random.fork_rng():
                torch.manual_seed(int(torch.randint(2**31, (1,), generator=generator)))
                keep = masker.generate_mask(x=x.transpose(1, 2).reshape(B * C, 1, T),
                                            input_mask=torch.ones(B * C, T)).view(B, C, T).transpose(1, 2).bool()
            out = model(x, mask=keep, task="reconstruct")
            sq_err += ((out - x)[~keep] ** 2).sum().item()
            n += int((~keep).sum())
        results[ratio] = sq_err / n
        print(f"mask {ratio:.3f}: MSE {results[ratio]:.4f}", flush=True)
    mean = float(np.mean(list(results.values())))
    print(json.dumps({"model": args.model, "stride": args.stride, "seed": args.seed, "per_ratio": results,
                      "MSE": mean, "ref_MSE": 0.402}))
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"model": args.model, "stride": args.stride, "seed": args.seed, "MSE": mean,
                                 "per_ratio": results}) + "\n")


if __name__ == "__main__":
    main()
