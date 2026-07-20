#!/usr/bin/env python
"""Evaluate a saved checkpoint on the held-out test identities (leakage-safe).

Examples:
  python evaluate.py --model resnet2d --ckpt artifacts/resnet2d_best.pt
  python evaluate.py --model r2plus1d --ckpt artifacts/r2plus1d_best.pt --ks 1,3,5,10,15 --plot-embed
Prints per-cell leave-one-footwear-out, the competition verification report, and accumulated
rank-1 over a walking pass; saves the per-cell table and (optionally) the embedding plot.
"""
import argparse

import pandas as pd
import torch

from stepup.config import ARTIFACTS, T, dev, seed_everything
from stepup.data import build_datasets
from stepup.eval import (accumulated_identification, cross_footwear_verification,
                         leave_one_footwear_out, open_set_accumulated, plot_embeddings, summarise)
from stepup.models import registry, set_dropout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name (must match the checkpoint)")
    ap.add_argument("--ckpt", default=None, help="path to a checkpoint (default artifacts/{model}_best.pt)")
    ap.add_argument("--ks", default="1,3,5,10", help="accumulation levels for rank-1")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--plot-embed", action="store_true")
    ap.add_argument("--hf-repo", default=None, help="fetch the checkpoint from this HF repo if not local")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    seed_everything()
    import os
    ckpt = args.ckpt or str(ARTIFACTS / f"{args.model}_best.pt")
    if not os.path.exists(ckpt):                 # checkpoint offloaded to HF? fetch it
        if args.hf_repo:
            from stepup.hf import fetch_file
            ckpt = fetch_file(args.hf_repo, f"{args.model}_best.pt", args.hf_token)
            print(f"fetched checkpoint from HF: {ckpt}")
        else:
            raise SystemExit(f"checkpoint not found: {ckpt}\nPass --hf-repo user/name to fetch it from HF.")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    set_dropout(cfg.get("dropout", 0.0))
    data_t = cfg["pack_res"][0] if cfg["pack_res"] else T
    reg = registry(cfg["sample3d"], data_t)
    spec = reg[args.model]
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **spec["kw"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()

    _, ds = build_datasets(cfg)
    target = ds[args.split if args.split in ds else "test"]

    ev = leave_one_footwear_out(net, target)
    print(ev.to_string(index=False))
    s = summarise(ev)
    vr = cross_footwear_verification(net, target)
    print(f"\nsummary  same_rank1 {s.get('same_rank1', float('nan')):.3f}  "
          f"cross_rank1 {s.get('cross_rank1', float('nan')):.3f}  "
          f"cross_eer {s.get('cross_eer', float('nan')):.3f}")
    print(f"verification  EER {vr['eer']:.3f}  BACC {vr['balanced_accuracy']:.3f}  "
          f"F1 {vr['f1']:.3f}  precision {vr['precision']:.3f}  recall {vr['recall']:.3f}  "
          f"FMR {vr['fmr']:.3f}  FNMR {vr['fnmr']:.3f}")
    ks = tuple(int(v) for v in args.ks.split(","))
    acc = accumulated_identification(net, target, ks=ks)
    print("accumulated rank1 (cross-footwear, hard)      " +
          "  ".join(f"{k}-step {v:.3f}" for k, v in acc.items()))
    osa = open_set_accumulated(net, target, ks=ks)
    print("accumulated rank1 (mixed gallery, ref ~0.9)   " +
          "  ".join(f"{k}-step {v:.3f}" for k, v in osa.items()))
    ev.to_parquet(ARTIFACTS / f"eval_{args.model}_{args.split}.parquet", index=False)
    if args.plot_embed:
        p = plot_embeddings(net, target, f"{args.model} {args.split} embeddings",
                            ARTIFACTS / f"embed_{args.model}.png")
        print(f"embedding plot -> {p}")


if __name__ == "__main__":
    main()
