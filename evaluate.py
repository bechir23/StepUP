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

from stepup.config import ARTIFACTS, DEF_KS, T, dev, seed_everything
from stepup.data import build_datasets
from stepup.eval import (accumulated_identification, cross_footwear_verification,
                         leave_one_footwear_out, open_set_accumulated, plot_embeddings, summarise)
from stepup.models import registry, set_adaptive_proj, set_dropout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name (must match the checkpoint)")
    ap.add_argument("--ckpt", default=None, help="path to a checkpoint (default artifacts/{model}_best.pt)")
    ap.add_argument("--ks", default=DEF_KS, help="accumulation levels for rank-1")
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
            # fetch the exact file the user asked for (tagged checkpoints are {model}_{tag}_best.pt,
            # not just {model}_best.pt), falling back to the plain name only when --ckpt was default.
            ckpt = fetch_file(args.hf_repo, os.path.basename(ckpt), args.hf_token)
            print(f"fetched checkpoint from HF: {ckpt}")
        else:
            raise SystemExit(f"checkpoint not found: {ckpt}\nPass --hf-repo user/name to fetch it from HF.")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    set_dropout(cfg.get("dropout", 0.0))
    set_adaptive_proj(cfg.get("adaptive_proj", False))
    data_t = (cfg.get("sample3d") or cfg.get("pack_res") or (T, T, T))[0]   # what the model sees
    if cfg.get("stride_pairs"):
        data_t *= 2                          # a stride = left+right concatenated in time (frames->channels)
    reg = registry(cfg["sample3d"], data_t)
    spec = reg[args.model]
    # Overlay the architecture toggles the model was TRAINED with (hpp/dsu/mixstyle) onto the
    # registry's base kwargs -- hpp changes feat_dim (256->1792), so rebuilding without it fails the
    # strict load. in_frames stays from spec["kw"] (current data). Old ckpts (no saved kw) fall back.
    _saved_kw = ck.get("kw", {}) or {}
    _kw = {**spec["kw"], **{k: _saved_kw[k] for k in ("hpp", "dsu", "mixstyle") if k in _saved_kw}}
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **_kw).to(dev)
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
    print(f"verification  EER {vr['eer']:.3f}  FMR100 {vr['fmr100']:.3f}  AUC {vr['auc']:.3f}  "
          f"BACC {vr['balanced_accuracy']:.3f}  F1 {vr['f1']:.3f}  precision {vr['precision']:.3f}  "
          f"recall {vr['recall']:.3f}  FMR {vr['fmr']:.3f}  FNMR {vr['fnmr']:.3f}")
    # Competition-style metrics at a FIXED decision threshold calibrated on validation (the analog of
    # each team's "submitted threshold"). Away from the equal-error point, FMR, FNMR, ACC and BACC no
    # longer coincide with the EER -- these are the distinct values to tabulate alongside the teams.
    if "val" in ds and args.split == "test":
        import numpy as np
        from sklearn.metrics import roc_curve
        from stepup.eval import cross_footwear_scores
        vs, vl = cross_footwear_scores(net, ds["val"])
        vfpr, vtpr, vthr = roc_curve(np.asarray(vl).astype(int), np.asarray(vs))
        tau_fix = float(vthr[np.nanargmin(np.abs((1 - vtpr) - vfpr))])   # val equal-error threshold
        ts, tl = cross_footwear_scores(net, target)
        tl = np.asarray(tl).astype(int); pred = (np.asarray(ts) >= tau_fix).astype(int)
        tp = int(((pred == 1) & (tl == 1)).sum()); fp = int(((pred == 1) & (tl == 0)).sum())
        tn = int(((pred == 0) & (tl == 0)).sum()); fn = int(((pred == 0) & (tl == 1)).sum())
        fmr_f = fp / max(1, fp + tn); fnmr_f = fn / max(1, fn + tp)
        acc_f = (tp + tn) / max(1, len(tl)); bacc_f = 1 - (fmr_f + fnmr_f) / 2
        print(f"at val-calibrated threshold  ACC {acc_f*100:.2f}  BACC {bacc_f*100:.2f}  "
              f"FNMR {fnmr_f*100:.2f}  FMR {fmr_f*100:.2f}")
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
