"""Zero-shot forecasting on a tgdata traffic dataset: masked MAE / RMSE / MAPE at 15, 30 and 60 minutes.

    python examples/eval_pems08.py                                    # flowstate:r1.1, contexts 12 and 512
    python examples/eval_pems08.py --model chronos:chronos-2 --windows 12 512 --out outputs/eval_pems08.jsonl

Scores the dataset's own test targets (12 steps out) whatever the context, so contexts are
comparable. Prints plain numbers, and appends one JSON line per context length to --out. Needs the `tgdata` extra and TGDATA_ROOT (see .envrc).
"""
import argparse
import json
import logging

import tgdata

from tsfm_zoo import load_tsfm
from tsfm_zoo.device import pick_device
from tsfm_zoo.tgeval import evaluate_forecast, summarize

# Supervised reference from the TIDES paper table (masked MAE averaged over the 12 steps).
TIDES_MAE = {"pems03": 14.55, "pems04": 19.08, "pems07": 19.68, "pems08": 14.09}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flowstate:r1.1", help="family:variant")
    parser.add_argument("--dataset", default="pems08")
    parser.add_argument("--windows", type=int, nargs="+", default=[12, 512], help="context lengths in steps")
    parser.add_argument("--batch-size", type=int, default=16, help="windows per model call (x nodes series)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="append JSON lines of results here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    family, _, variant = args.model.partition(":")
    device = pick_device(args.device)
    model = load_tsfm(family, variant, device=device, seed=args.seed)
    g = tgdata.load(args.dataset)
    header = {"model": args.model, "device": device, "seed": args.seed, "dataset": g.name,
              "tgdata": tgdata.__version__, "x_shape": list(g.x.shape), "freq": g.meta["freq"],
              "training_corpora": model.training_corpora, "tides_mae": TIDES_MAE.get(args.dataset)}
    print(json.dumps(header))

    for window in args.windows:
        result = evaluate_forecast(model, g, window=window, batch_size=args.batch_size)
        record = {**header, "window": window, "horizon": result["horizon"], "windows": result["windows"]}
        for name in ("model", "persistence"):
            record[name] = summarize(result[name])
            for metric, values in record[name].items():
                cells = " ".join(f"{step}={value:.4f}" for step, value in values.items())
                print(f"window={window} {name} {metric}: {cells}", flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
