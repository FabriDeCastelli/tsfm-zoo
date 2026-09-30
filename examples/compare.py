"""Run one TSFM on a GIFT-Eval config and print MASE/CRPS next to the leaderboard.

    python examples/compare.py --model chronos:chronos-2 --dataset gifteval:ett1/H/short --device auto
    python examples/compare.py --model chronos:chronos-2 --dataset gifteval:repro
"""
import argparse
import json
import logging

from tsfm_zoo import gifteval, load_tsfm
from tsfm_zoo.device import pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="family[:variant]")
    parser.add_argument("--dataset", required=True, help="gifteval:<config> or gifteval:repro")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, help="default: the leaderboard notebook's, else 64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="append JSON lines of results here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    family, _, variant = args.model.partition(":")
    device = pick_device(args.device)
    settings = gifteval.LEADERBOARD.get(variant, {})
    model = load_tsfm(family, variant or None, device=device, seed=args.seed, **settings.get("load", {}))
    source, config = args.dataset.split(":", 1)
    if source != "gifteval":
        raise SystemExit(f"unknown dataset source {source!r}")
    configs = gifteval.REPRO_CONFIGS if config == "repro" else [config]
    print(json.dumps({"model": args.model, "device": device, "settings": settings, "seed": args.seed}))

    print(f"{'config':24s} {'MASE':>8s} {'ref':>8s} {'Δ%':>6s} {'CRPS':>8s} {'ref':>8s} {'Δ%':>6s}")
    for cfg in configs:
        data = gifteval.load(cfg)
        ours = gifteval.evaluate(model, data, multivariate=settings.get("multivariate", False),
                                 pack=settings.get("pack"),
                                 batch_size=args.batch_size or settings.get("batch_size", 64),
                                 call=settings.get("call"))
        ref = gifteval.leaderboard(settings["results"], cfg) if settings.get("results") else {}
        cells = []
        for metric in ("MASE", "CRPS"):
            r = ref.get(metric)
            delta = f"{100 * (ours[metric] / r - 1):+6.2f}" if r else "     -"
            cells.append(f"{ours[metric]:8.4f} {r if r else float('nan'):8.4f} {delta}")
        print(f"{cfg:24s} {' '.join(cells)}", flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps({"model": args.model, "config": cfg, **ours,
                                     **{f"ref_{k}": v for k, v in ref.items()}}) + "\n")


if __name__ == "__main__":
    main()
