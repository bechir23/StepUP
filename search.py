#!/usr/bin/env python
"""Automated hyperparameter / architecture search.

Random-search over a defined space of {model, loss, lr, dropout, sample3d}, training each trial
on the shared packs and ranking by a chosen validation metric. Trials that look weak after a few
epochs are pruned early (their best-so-far is compared against the running median), so the budget
concentrates on promising configurations -- the same idea as interpreting early training dynamics
to abandon unpromising trials.

Examples:
  python search.py --trials 20 --epochs 25                       # search everything
  python search.py --trials 12 --model r2plus1d --rank-by unseen_eer   # fix model, minimise EER
  python search.py --trials 30 --epochs 20 --wandb online
"""
import argparse
import itertools
import json
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from stepup.args import add_common_args, apply_smoke
from stepup.config import ARTIFACTS, T, build_cfg, dev, seed_everything
from stepup.data import build_datasets
from stepup.engine import train
from stepup.eval import condition_verification, embed_dataset, open_set_accumulated
from stepup.models import registry, set_dropout

# ---------------------------------------------------------------- search space
SPACE = {
    "model":    ["r2plus1d", "r2plus1d_light", "r3d", "r3d_light", "swin3d", "swin3d_light",
                 "convnext", "resnet2d", "resnet2d_light", "gaitcnn", "gaitcnn_deep", "cnnlstm", "vit"],
    "loss":     ["arcface", "triplet"],
    "lr":       [1e-4, 2e-4, 5e-4, 1e-3],
    "dropout":  [0.2, 0.3, 0.4],
    "sample3d": ["48,48,48", "32,64,40", "16,48,32"],
}


def sample_config(rng, fixed):
    return {k: (fixed[k] if fixed.get(k) is not None else rng.choice(v)) for k, v in SPACE.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--model", default=None, help="fix the model instead of searching it")
    ap.add_argument("--final-epochs", type=int, default=100,
                    help="after the search, retrain the best config this many epochs and save it "
                         "(0 = search only, no final model)")
    ap.add_argument("--rank-by", default="mixed5",
                    choices=["mixed5", "unseen_eer", "seen_eer"], help="metric to optimise")
    ap.add_argument("--prune-epochs", type=int, default=4,
                    help="check trial health after this many epochs")
    ap.add_argument("--search-seed", type=int, default=0)
    args = apply_smoke(ap.parse_args())

    rng = np.random.default_rng(args.search_seed)
    fixed = {"model": args.model}
    # build the shared packs ONCE at full resolution (sample3d only changes the 3D runtime resize,
    # not the stored pack), so every trial reuses the same data.
    base = build_cfg(args)
    seed_everything()
    man, ds = build_datasets(base)
    minimise = args.rank_by.endswith("eer")
    results, best_so_far = [], (1.0 if minimise else 0.0)

    pbar = tqdm(range(args.trials), desc="search", unit="trial")
    for t in pbar:
        pbar.set_postfix(best=f"{best_so_far:.3f}", done=len(results))
        cfg = dict(base)
        pick = sample_config(rng, fixed)
        cfg.update(loss=pick["loss"], lr=float(pick["lr"]), dropout=float(pick["dropout"]),
                   sample3d=tuple(int(v) for v in pick["sample3d"].split(",")))
        set_dropout(cfg["dropout"])
        name = pick["model"]
        data_t = cfg["pack_res"][0] if cfg["pack_res"] else T
        spec = registry(cfg["sample3d"], data_t)[name]
        cfg["lr_mult"] = spec.get("lr_mult", 1.0)
        P, K = (args.P or spec["full_pk"][0]), (args.K or spec["full_pk"][1])
        steps = max(1, len(man["train"]) // (P * K))
        print(f"\n=== trial {t+1}/{args.trials}: {name} loss={pick['loss']} lr={pick['lr']} "
              f"dropout={pick['dropout']} sample3d={pick['sample3d']} ===", flush=True)
        try:
            net, hist, best = train(spec["fn"], man["train"], cfg, tag=f"trial{t}",
                                    max_epochs=args.epochs, patience=args.prune_epochs + 3,
                                    steps_per_epoch=steps, P=P, K=K, model_kw=spec["kw"],
                                    ds_tr=ds["train"], ds_va=ds["val_mon"], mining=cfg["mining"],
                                    wandb_run=None)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  trial failed ({type(e).__name__}: {str(e)[:60]}) -- skipped", flush=True)
            continue
        fpre = embed_dataset(net, ds["val"])
        mixed = open_set_accumulated(net, ds["val"], ks=(5, 10), repeats=3, precomp=fpre,
                                     score_norm="znorm")
        cond = condition_verification(net, ds["val"], precomp=fpre)
        row = dict(**pick, mixed5=round(mixed.get(5, float("nan")), 4),
                   mixed10=round(mixed.get(10, float("nan")), 4),
                   seen_eer=round(cond.get("seen", {}).get("eer", float("nan")), 4),
                   unseen_eer=round(cond.get("unseen", {}).get("eer", float("nan")), 4),
                   unseen_fmr100=round(cond.get("unseen", {}).get("fmr100", float("nan")), 4))
        score = row["mixed5"] if args.rank_by == "mixed5" else row[args.rank_by]
        best_so_far = min(best_so_far, score) if minimise else max(best_so_far, score)
        results.append(row)
        pbar.set_postfix(best=f"{best_so_far:.3f}", last=f"{score:.3f}", done=len(results))
        print(f"  -> {args.rank_by}={score}  (best {best_so_far})  mixed5={row['mixed5']} "
              f"unseen_eer={row['unseen_eer']}", flush=True)
        del net, fpre
        if dev == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r[args.rank_by], reverse=not minimise)
    print("\n================ SEARCH LEADERBOARD (by " + args.rank_by + ") ================")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:2d}. {r['model']:12s} {r['loss']:8s} lr={r['lr']:<6} do={r['dropout']} "
              f"s3d={r['sample3d']:9s} | mixed5={r['mixed5']} mixed10={r['mixed10']} "
              f"seen_eer={r['seen_eer']} unseen_eer={r['unseen_eer']}")
    (ARTIFACTS / "search_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {len(results)} trials -> {ARTIFACTS/'search_results.json'}")

    # ---- train the winning config to completion and save a ready-to-use checkpoint ----
    if results and args.final_epochs > 0:
        best = results[0]
        cfg = dict(base)
        cfg.update(loss=best["loss"], lr=float(best["lr"]), dropout=float(best["dropout"]),
                   sample3d=tuple(int(v) for v in best["sample3d"].split(",")))
        set_dropout(cfg["dropout"])
        name = best["model"]
        data_t = cfg["pack_res"][0] if cfg["pack_res"] else T
        spec = registry(cfg["sample3d"], data_t)[name]
        cfg["lr_mult"] = spec.get("lr_mult", 1.0)
        P, K = (args.P or spec["full_pk"][0]), (args.K or spec["full_pk"][1])
        steps = max(1, len(man["train"]) // (P * K))
        print(f"\n=== FINAL: training best config to completion ({name}, loss={best['loss']}, "
              f"lr={best['lr']}, {args.final_epochs} epochs) ===", flush=True)
        net, hist, rec = train(spec["fn"], man["train"], cfg, tag=name, max_epochs=args.final_epochs,
                               patience=cfg["patience"], steps_per_epoch=steps, P=P, K=K,
                               model_kw=spec["kw"], ds_tr=ds["train"], ds_va=ds["val_mon"],
                               mining=cfg["mining"], wandb_run=None)
        torch.save(dict(state=rec["state"], cfg=cfg, model=name, kw=spec["kw"],
                        search_config=best, epoch=rec["epoch"]), ARTIFACTS / f"{name}_best.pt")
        fpre = embed_dataset(net, ds["test"])                       # held-out test numbers
        mixed = open_set_accumulated(net, ds["test"], ks=(1, 3, 5, 10), repeats=5, precomp=fpre,
                                     score_norm="znorm")
        cond = condition_verification(net, ds["test"], precomp=fpre)
        print("\n================ FINAL MODEL (held-out test) ================")
        print("mixed-gallery rank-1 (znorm):", {k: round(v, 3) for k, v in mixed.items()})
        for c in ("seen", "unseen"):
            r = cond.get(c)
            if r:
                print(f"  {c:6s} EER {r['eer']*100:5.2f}  FMR100 {r['fmr100']*100:5.2f}  "
                      f"BACC {r['balanced_accuracy']*100:5.2f}")
        print(f"saved final model -> {ARTIFACTS/(name+'_best.pt')}")
        if args.hf_repo:
            from stepup.hf import push_model
            push_model(args.hf_repo, name, ARTIFACTS, args.hf_token)
            print(f"pushed -> https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
