"""Stage 3 --- calibration & competition submission.

Enroll -> cosine -> cohort/z-norm score normalization -> threshold at the EER point, then emit the
competition files scores.txt + threshold.txt and report EER / FMR100 / BACC. No deep network:
the only choices are score-normalization (raw vs z-norm) and the calibrator (identity vs Platt).

  python submit.py --model gaitcnn_snr --hf-repo Bechir23/stepup-footstep --score-norm znorm --k 5
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from stepup.config import ARTIFACTS, DEF_WALK_K, FOOTWEAR, T, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.metrics import enroll_templates, report_from_scores
from stepup.models import registry, set_dropout


def load_backbone(model, ckpt, hf_repo, hf_token, in_frames=0, tag=""):
    fname = f"{model}_{tag}_best.pt" if tag else f"{model}_best.pt"
    if not os.path.exists(ckpt or ""):
        ckpt = str(ARTIFACTS / fname)
    if not os.path.exists(ckpt):
        if not hf_repo:
            raise FileNotFoundError(
                f"backbone checkpoint '{fname}' not found in {ARTIFACTS} and no --hf-repo to fetch it "
                f"from. Train Stage 1 first (saves {fname} after all epochs), or pass --hf-repo.")
        from stepup.hf import fetch_file
        ckpt = fetch_file(hf_repo, fname, hf_token)
        print(f"fetched checkpoint from HF: {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    set_dropout(cfg.get("dropout", 0.0))
    data_t = (cfg.get("sample3d") or cfg.get("pack_res") or (T, T, T))[0]
    if cfg.get("stride_pairs"):
        data_t *= 2                      # stride = left+right concatenated in time (as in train.py)
    if in_frames > 0:
        data_t = in_frames               # manual override if auto mismatches the ckpt
    spec = registry(cfg["sample3d"], data_t)[model]
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **spec["kw"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, cfg


def cross_scores(f, y, fw, score_norm="none"):
    """Pooled cross-footwear (probe shoe != enrol shoe) genuine/impostor scores; optional per-
    template cohort z-norm (standardize each template column by its impostor distribution)."""
    fw = np.asarray(fw)
    scores, labels = [], []
    for enrol in FOOTWEAR:
        g, p = fw == enrol, fw != enrol
        if g.sum() == 0 or p.sum() == 0:
            continue
        templates, ids = enroll_templates(f[g], y[g])
        sim = (F.normalize(f[p]) @ templates.t()).numpy()          # (n_probe, n_templates)
        if score_norm == "znorm":                                   # cohort normalization
            sim = (sim - sim.mean(0, keepdims=True)) / (sim.std(0, keepdims=True) + 1e-6)
        gen = (ids[None, :] == y[p][:, None]).numpy()
        scores.append(sim.ravel()); labels.append(gen.ravel().astype(int))
    return np.concatenate(scores), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr")
    ap.add_argument("--in-frames", type=int, default=0, help="override backbone in_frames if the auto value mismatches the checkpoint")
    ap.add_argument("--tag", default="", help="load {model}_{tag}_best.pt (a specific ablation run)")
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"], help="override the checkpoint's pack device (cuda = fast eval on Colab)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--score-norm", default="znorm", choices=["none", "znorm"])
    ap.add_argument("--k", type=int, default=DEF_WALK_K, help="footsteps per walk (walk mode)")
    ap.add_argument("--walk", action="store_true",
                    help="calibrate at the WALK level (mean-pool k footsteps) so Stage 3 chains from "
                         "Stage 2 -- EER should drop below the single-footstep number")
    ap.add_argument("--agg-ckpt", default="",
                    help="Stage-2 aggregator .pt to pool the walk with (implies --walk); else mean pool")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        rname = f"submit_{args.model}" + (f"_{args.tag}" if args.tag else "")
        wb = _wb.init(project="stepup-footstep", name=rname, mode=args.wandb, config=vars(args))

    net, cfg = load_backbone(args.model, args.ckpt, args.hf_repo, args.hf_token, args.in_frames, args.tag)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device   # override (cuda = fast on Colab, memmap = low RAM)
    _, ds = build_datasets(cfg)
    f, y, fw = embed_dataset(net, ds["test"])
    print(f"embedded test: {len(f)} steps, {len(np.unique(y.numpy()))} ids", flush=True)

    level = "footstep"
    if args.walk or args.agg_ckpt:      # Stage-3-on-Stage-2: calibrate WALK-level scores (chain)
        from aggregate import walk_windows, build_walk_embeds, make_aggregator
        agg = None
        if args.agg_ckpt:
            path = args.agg_ckpt if os.path.exists(args.agg_ckpt) else str(ARTIFACTS / args.agg_ckpt)
            ck = torch.load(path, map_location=dev, weights_only=False)
            agg = make_aggregator(ck.get("aggregator", "pma"), f.shape[1]).to(dev)
            agg.load_state_dict(ck["state"]); agg.eval()
            print(f"loaded Stage-2 aggregator {ck.get('aggregator')} (best epoch {ck.get('best_epoch')})",
                  flush=True)
        win = walk_windows(ds["test"].m.reset_index(drop=True), args.k)
        ew, yw, fww = build_walk_embeds(f, win, agg=agg)     # one embedding per single-shoe walk
        f, y, fw = ew, torch.as_tensor(yw), np.asarray(fww)
        level = f"walk(k={args.k},{'learned' if agg else 'mean'})"
        print(f"walk-level: {len(f)} walks  [{level}]", flush=True)

    # ablation the reference ran: raw vs z-norm
    rows = []
    for norm in ("none", "znorm"):
        s, lab = cross_scores(f, y, fw, score_norm=norm)
        r = report_from_scores(s, lab)
        rows.append({"score_norm": norm, **{k: float(v) for k, v in r.items()}})
        print(f"  score-norm={norm:5s}  EER {r['eer']*100:5.2f}  FMR100 {r['fmr100']*100:5.2f}  "
              f"BACC {r['balanced_accuracy']*100:5.2f}  ACC {r['accuracy']*100:5.2f}", flush=True)
        if wb is not None:
            wb.log({f"eer_{norm}": r["eer"], f"fmr100_{norm}": r["fmr100"],
                    f"bacc_{norm}": r["balanced_accuracy"]})

    # submission files at the chosen normalization, threshold = EER point (min-max -> [0,1])
    s, lab = cross_scores(f, y, fw, score_norm=args.score_norm)
    lo, hi = s.min(), s.max()
    s01 = (s - lo) / (hi - lo + 1e-9)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(lab, s01)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    threshold = float(thr[i])
    # exact competition leaderboard row AT the submitted threshold: EER/FMR100 (optimized) + the
    # operating-point metrics ACC/BACC/FNMR/FMR the organizers score at the team's own threshold.
    gen, imp = s01[lab == 1], s01[lab == 0]
    fnmr = float((gen < threshold).mean()); fmr = float((imp >= threshold).mean())
    acc = float(((gen >= threshold).sum() + (imp < threshold).sum()) / len(s01))
    bacc = 1 - (fmr + fnmr) / 2
    rr = report_from_scores(s, lab)
    print(f"  LEADERBOARD ({args.score_norm}) @ thr {threshold:.3f}:  EER {rr['eer']*100:5.2f}  "
          f"FMR100 {rr['fmr100']*100:5.2f}  ACC {acc*100:5.2f}  BACC {bacc*100:5.2f}  "
          f"FNMR {fnmr*100:5.2f}  FMR {fmr*100:5.2f}", flush=True)
    if wb is not None:
        wb.log({"lb_eer": rr["eer"], "lb_fmr100": rr["fmr100"], "lb_acc": acc,
                "lb_bacc": bacc, "lb_fnmr": fnmr, "lb_fmr": fmr})
    wtag = "" if level == "footstep" else f"_walk{args.k}"   # keep walk-level runs distinct from the
    sfx = (f"_{args.tag}" if args.tag else "") + wtag        # single-footstep competition submission
    scores_f = ARTIFACTS / f"scores_{args.model}{sfx}.txt"
    thr_f = ARTIFACTS / f"threshold_{args.model}{sfx}.txt"
    np.savetxt(scores_f, s01, fmt="%.6f")
    with open(thr_f, "w") as fh:
        fh.write(f"{threshold:.6f}\n")
    # Python-readable leaderboard parquet (raw vs znorm + the operating-point row) for analysis/plots
    import pandas as pd
    rows.append({"score_norm": f"{args.score_norm}@thr", "eer": rr["eer"], "fmr100": rr["fmr100"],
                 "accuracy": acc, "balanced_accuracy": bacc, "fnmr": fnmr, "fmr": fmr,
                 "threshold": threshold})
    lb_f = ARTIFACTS / f"submit_{args.model}{sfx}.parquet"
    pd.DataFrame(rows).to_parquet(lb_f, index=False)
    print(f"wrote {len(s01)} scores -> {scores_f.name}  |  threshold {threshold:.4f} -> {thr_f.name}  "
          f"|  metrics -> {lb_f.name}  (norm={args.score_norm}, k={args.k})  # rename to "
          f"scores.txt/threshold.txt for the competition zip", flush=True)
    if args.hf_repo:                                     # push submission + metrics parquet to HF
        from stepup.hf import push_files
        push_files(args.hf_repo, [scores_f, thr_f, lb_f], args.hf_token)
        print(f"pushed submission + metrics -> https://huggingface.co/{args.hf_repo}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
