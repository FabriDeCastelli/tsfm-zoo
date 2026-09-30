"""Embed every window of a tgdata dataset and cache the result under a hash of the configuration.

    python examples/embed_windows.py --model flowstate:r1.1 --dataset pems08 --window 12
    python examples/embed_windows.py --model chronos:chronos-2 --dataset pems08 --window 512 --dtype float16

The cache directory is $TSFM_EMBEDDINGS (see .envrc) unless --cache-dir is given. Running the same
configuration again returns the cached array without touching the model. Prints where the cache
is, and the shape of the embeddings of each split of the dataset's own task.
"""
import argparse
import logging
import os

import tgdata

from tsfm_zoo import load_tsfm
from tsfm_zoo.device import pick_device
from tsfm_zoo.embeddings import embed_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="family:variant")
    parser.add_argument("--dataset", default="pems08")
    parser.add_argument("--window", type=int, default=12, help="input steps per window")
    parser.add_argument("--pool", help="how to pool the tokens: mean, last or reg (default: the model's summary)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--cache-dir", default=os.environ.get("TSFM_EMBEDDINGS"))
    parser.add_argument("--batch-size", type=int, default=64, help="windows per model call")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.cache_dir:
        raise SystemExit("set --cache-dir or TSFM_EMBEDDINGS")

    family, _, variant = args.model.partition(":")
    model = load_tsfm(family, variant, device=pick_device(args.device), seed=args.seed)
    g = tgdata.load(args.dataset)
    cache = embed_windows(model, g, args.cache_dir, window=args.window, batch_size=args.batch_size,
                          pool=args.pool, dtype=args.dtype)
    print(cache.path, cache.array.shape, cache.array.dtype)
    for split in ("train", "val", "test"):
        task = g.task(split=split, device="cpu")                        # the dataset's own task
        if task.window == args.window:
            print(split, cache.rows(task).shape)


if __name__ == "__main__":
    main()
