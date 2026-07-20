#!/usr/bin/env python
"""Train one or all StepUP backbones, then evaluate on the held-out test identities.

Examples:
  python train.py --model resnet2d                     # one model, defaults
  python train.py --model all                          # every backbone
  python train.py --model r2plus1d --mining crossfw    # cross-footwear positive mining
  python train.py --model gaitcnn --loss arcface --arc-scale 32
  python train.py --model resnet2d --no-augment --epochs 50 --P 32
  python train.py --model swin3d --plot-embed          # + save the embedding plot
Outputs per model in artifacts/: {name}_best.pt, hist_{name}.parquet, test_{name}.parquet,
verif_{name}.parquet, acc_{name}.parquet, embed_{name}.png (with --plot-embed).
"""
import argparse
import gc

import pandas as pd
import torch

from stepup.args import add_common_args, apply_smoke
from stepup.config import ARTIFACTS, T, build_cfg, seed_everything
from stepup.data import build_datasets
from stepup.engine import train
from stepup.eval import (accumulated_identification, condition_verification,
                         cross_footwear_verification, leave_one_footwear_out,
                         plot_embeddings, plot_history, summarise)
from stepup.models import registry, set_dropout
from stepup.wb import init_run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", default="all", help="a model name or 'all'")
    ap.add_argument("--plot-embed", action="store_true", help="save the test embedding plot")
    args = apply_smoke(ap.parse_args())

    cfg = build_cfg(args)
    set_dropout(cfg["dropout"])
    seed_everything()
    data_t = cfg["pack_res"][0] if cfg["pack_res"] else T
    reg = registry(cfg["sample3d"], data_t)
    names = list(reg) if args.model == "all" else [args.model]
    assert all(n in reg for n in names), f"unknown model; choose from {list(reg)}"

    man, ds = build_datasets(cfg)
    print(f"train {len(man['train']):,} steps / {man['train'].ParticipantID.nunique()} ids  "
          f"| loss={cfg['loss']} mining={cfg['mining']} augment={cfg['augment']} "
          f"pack_res={cfg['pack_res']} sample3d={cfg['sample3d']}")

    for name in names:
        spec = reg[name]
        P0, K0 = spec["full_pk"]
        P, K = args.P or P0, args.K or K0            # --P/--K override the per-model batch
        mkw = dict(spec["kw"])
        if args.mixstyle and name in ("r2plus1d", "r3d"):
            mkw["mixstyle"] = True                    # MixStyle only in the video ResNets
        steps = cfg["steps_per_epoch"] or max(1, len(man["train"]) // (P * K))
        run = init_run(args, cfg, name)
        net, hist, best = train(spec["fn"], man["train"], cfg, tag=name,
                                max_epochs=cfg["epochs"], patience=cfg["patience"],
                                steps_per_epoch=steps, P=P, K=K, model_kw=mkw,
                                ds_tr=ds["train"], ds_va=ds["val_mon"], mining=cfg["mining"],
                                wandb_run=run)
        torch.save(dict(state=best["state"], cfg=cfg, model=name, kw=spec["kw"],
                        val_cross_eer=best["val"], epoch=best["epoch"]),
                   ARTIFACTS / f"{name}_best.pt")
        hist.to_parquet(ARTIFACTS / f"hist_{name}.parquet", index=False)
        plot_history(hist, f"{name} training", ARTIFACTS / f"curves_{name}.png")

        ev = leave_one_footwear_out(net, ds["test"])
        ev.to_parquet(ARTIFACTS / f"test_{name}.parquet", index=False)
        vr = cross_footwear_verification(net, ds["test"])
        pd.DataFrame([vr]).to_parquet(ARTIFACTS / f"verif_{name}.parquet", index=False)
        acc = accumulated_identification(net, ds["test"])
        pd.DataFrame([acc]).to_parquet(ARTIFACTS / f"acc_{name}.parquet", index=False)
        cond = condition_verification(net, ds["test"])                 # competition seen/unseen split
        pd.DataFrame(cond).T.to_parquet(ARTIFACTS / f"cond_{name}.parquet")
        s = summarise(ev)
        print(f"\n{name} TEST  cross rank1 {s.get('cross_rank1', float('nan')):.3f}  "
              f"EER {vr['eer']:.3f}  BACC {vr['balanced_accuracy']:.3f}  F1 {vr['f1']:.3f}  "
              f"recall {vr['recall']:.3f}")
        for c in ("seen", "unseen"):                                    # competition metric set
            r = cond.get(c)
            if r:
                print(f"  {c:6s} EER {r['eer']*100:5.2f}  FMR100 {r['fmr100']*100:5.2f}  "
                      f"ACC {r['accuracy']*100:5.2f}  BACC {r['balanced_accuracy']*100:5.2f}  "
                      f"FNMR {r['fnmr']*100:5.2f}  FMR {r['fmr']*100:5.2f}")
        print("  accumulated rank1  " + "  ".join(f"{k}-step {v:.3f}" for k, v in acc.items()))
        if args.plot_embed:
            p = plot_embeddings(net, ds["test"], f"{name} test embeddings",
                                ARTIFACTS / f"embed_{name}.png")
            print(f"  embedding plot -> {p}")
        if run is not None:
            run.summary["best_cross_eer"] = best["val"]; run.finish()
        if args.hf_repo:                          # push this model's artifacts to HF storage
            from stepup.hf import push_model
            push_model(args.hf_repo, name, ARTIFACTS, args.hf_token)
            print(f"  pushed {name} artifacts -> https://huggingface.co/{args.hf_repo}")
            if args.hf_offload:                   # keep the folder light: models live on HF only
                ckpt = ARTIFACTS / f"{name}_best.pt"
                if ckpt.exists():
                    ckpt.unlink()
                    print(f"  offloaded {ckpt.name} (removed local copy; on HF)")
        del net                                   # free the model + GPU memory before the next
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
