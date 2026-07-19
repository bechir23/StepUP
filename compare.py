#!/usr/bin/env python
"""Comparisons for the thesis.

  # standard vs cross-footwear positive mining on one backbone
  python compare.py --compare mining --model resnet2d --epochs 20

  # leaderboard across every model already trained (reads artifacts/*.parquet)
  python compare.py --compare models
"""
import argparse
import glob

import pandas as pd
import torch

from stepup.args import add_common_args, apply_smoke
from stepup.config import ARTIFACTS, T, build_cfg, seed_everything
from stepup.data import build_datasets
from stepup.engine import train
from stepup.eval import accumulated_identification, leave_one_footwear_out, summarise
from stepup.models import registry, set_dropout


def compare_mining(args):
    cfg = build_cfg(args)
    set_dropout(cfg["dropout"]); seed_everything()
    data_t = cfg["pack_res"][0] if cfg["pack_res"] else T
    reg = registry(cfg["sample3d"], data_t)
    spec = reg[args.model]
    man, ds = build_datasets(cfg)
    P0, K0 = spec["full_pk"]
    P, K = args.P or P0, args.K or K0
    steps = cfg["steps_per_epoch"] or max(1, len(man["train"]) // (P * K))
    rows = []
    for mode in ("standard", "crossfw"):
        net, _, best = train(spec["fn"], man["train"], cfg, tag=f"{args.model}-{mode}",
                             max_epochs=cfg["epochs"], patience=cfg["patience"],
                             steps_per_epoch=steps, P=P, K=K, model_kw=spec["kw"],
                             ds_tr=ds["train"], ds_va=ds["val_mon"], mining=mode)
        s = summarise(leave_one_footwear_out(net, ds["test"]))
        acc = accumulated_identification(net, ds["test"])
        rows.append(dict(mining=mode, val_cross_eer=round(best["val"], 3),
                         cross_rank1=round(s.get("cross_rank1", float("nan")), 3),
                         cross_eer=round(s.get("cross_eer", float("nan")), 3),
                         acc_rank1_5step=round(acc.get(5, float("nan")), 3)))
        del net
    df = pd.DataFrame(rows)
    df.to_parquet(ARTIFACTS / f"mining_compare_{args.model}.parquet", index=False)
    print(df.to_string(index=False))


def compare_models(args):
    rows = []
    for f in sorted(glob.glob(str(ARTIFACTS / "test_*.parquet"))):
        name = f.split("test_")[-1][:-8]
        s = summarise(pd.read_parquet(f))
        row = dict(model=name, cross_rank1=round(s.get("cross_rank1", float("nan")), 3),
                   cross_eer=round(s.get("cross_eer", float("nan")), 3))
        vf = ARTIFACTS / f"verif_{name}.parquet"
        if vf.exists():
            v = pd.read_parquet(vf).iloc[0]
            row.update(eer=round(v["eer"], 3), bacc=round(v["balanced_accuracy"], 3),
                       f1=round(v["f1"], 3))
        af = ARTIFACTS / f"acc_{name}.parquet"
        if af.exists():
            a = pd.read_parquet(af).iloc[0]
            row["acc_r1_5step"] = round(a.get("5", float("nan")), 3)
        rows.append(row)
    if not rows:
        print("no trained models found in artifacts/ (run train.py first)")
        return
    df = pd.DataFrame(rows).sort_values("cross_eer")
    df.to_parquet(ARTIFACTS / "model_leaderboard.parquet", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--compare", required=True, choices=["mining", "models"])
    ap.add_argument("--model", default="resnet2d")
    args = apply_smoke(ap.parse_args())
    (compare_mining if args.compare == "mining" else compare_models)(args)


if __name__ == "__main__":
    main()
