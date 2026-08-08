#!/usr/bin/env python
"""Enrollment chapter: how recognition improves as MORE footsteps are enrolled, on the frozen
backbone (no retraining). Compares nearest-prototype (NCM) vs SimpleShot CL2N vs BD-CSPN prototype
rectification (Liu et al., ECCV 2020, prototype_rectification.md), each optionally with AdaBN
adaptation to the probe footwear. Cross-footwear protocol: enroll from ONE shoe, identify in others.

  python enroll.py --model gaitcnn_snr --tag <t> --hf-repo Bechir23/stepup-footstep \
      --pack-device cuda --ks 1,2,3,5,8,10 --z 8

Outputs to <out>/: enroll_results.parquet + 3 plots
  enroll_eer_vs_k.png     EER vs #enrolled footsteps (methods x AdaBN) + 1/sqrt(K) variance trend
  enroll_det.png          DET at k=1/5/10 (BD-CSPN)
  enroll_tsne.png         probe features before/after cross-class rectification
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------- core (unit-testable on arrays) -----------------------------
def preprocess(f, method):
    """CL2N (SimpleShot): center by the global mean then L2-normalize. Else just L2-normalize."""
    if method == "cl2n":
        return F.normalize(f - f.mean(0, keepdim=True), dim=1)
    return F.normalize(f, dim=1)


def bdcspn_rectify(protos, query, z=8, scale=10.0):
    """BD-CSPN: (1) cross-class shift -- align the query mean to the support(prototype) mean;
    (2) intra-class rectification -- pseudo-label queries by nearest prototype and fold the top-Z
    most-confident queries (confidence-weighted) into each prototype. Transductive, inference-only.
    Returns (rectified prototypes, shifted query)."""
    protos = F.normalize(protos, dim=1)
    q = F.normalize(query, dim=1)
    xi = protos.mean(0, keepdim=True) - q.mean(0, keepdim=True)   # cross-class bias
    q = F.normalize(q + xi, dim=1)
    conf = torch.softmax((q @ protos.t()) * scale, dim=1)         # (Q, G) pseudo-label confidence
    new = protos.clone()
    for j in range(len(protos)):
        w = conf[:, j]
        top = torch.topk(w, min(z, len(w))).indices
        if len(top) == 0:
            continue
        ww = w[top].unsqueeze(1)
        num = protos[j:j + 1] + (ww * q[top]).sum(0, keepdim=True)  # support (unit) + weighted queries
        new[j] = (num / (1.0 + ww.sum())).squeeze(0)
    return F.normalize(new, dim=1), q


RANKS = (1, 3, 5)


def _rank_hits(sim, pid, qy, ranks=RANKS):
    """Top-r identification membership: for each probe, is its true identity among the r
    nearest (highest-similarity) prototypes? Same ranking as rank-1, just top-r membership.
    Returns {r: bool array over probes}. r is clipped to the #prototypes available."""
    order = np.argsort(-sim, axis=1)                       # prototypes by descending similarity
    ranked_ids = pid[order]                                # (Q, G) prototype ids in rank order
    hits = {}
    for r in ranks:
        rr = min(r, ranked_ids.shape[1])
        hits[r] = (ranked_ids[:, :rr] == qy[:, None]).any(1)
    return hits


def cross_fw_eval(f, y, fw, k, method="ncm", z=8, rng=None, FOOTWEAR=None):
    """Enroll each identity from ONE footwear (k steps), identify its OTHER-footwear steps.
    Returns pooled genuine/impostor scores + labels (for EER) AND rank-1/3/5 identification
    accuracy (the natural 'did we name the right person' metric -- BD-CSPN plots this)."""
    rng = rng or np.random.default_rng(0)
    fu = preprocess(f, method)
    ids = np.unique(y)
    S, L = [], []
    hit = {r: 0 for r in RANKS}
    total = 0
    for A in FOOTWEAR:
        g = fw == A
        if g.sum() == 0:
            continue
        protos, pid = [], []
        for i in ids:
            idx = np.where(g & (y == i))[0]
            if len(idx) == 0:
                continue
            sel = idx if len(idx) <= k else rng.choice(idx, k, replace=False)
            protos.append(F.normalize(fu[sel].mean(0), dim=0)); pid.append(i)
        if len(protos) < 2:
            continue
        P = torch.stack(protos); pid = np.array(pid)
        pm = fw != A
        qf, qy = fu[pm], y[pm]
        if method == "bdcspn":
            P, qf = bdcspn_rectify(P, qf, z=z)
        sim = (F.normalize(qf) @ F.normalize(P).t()).numpy()
        S.append(sim.ravel()); L.append((pid[None, :] == qy[:, None]).ravel())
        h = _rank_hits(sim, pid, qy)                        # rank-1/3/5 identification
        for r in RANKS:
            hit[r] += int(h[r].sum())
        total += len(qy)
    accs = tuple(hit[r] / max(total, 1) for r in RANKS)     # (rank1, rank3, rank5)
    return np.concatenate(S), np.concatenate(L), accs


def cross_fw_scores(f, y, fw, k, method="ncm", z=8, rng=None, FOOTWEAR=None):
    s, l, _ = cross_fw_eval(f, y, fw, k, method, z, rng, FOOTWEAR)
    return s, l


def metrics_at(f, y, fw, k, method, z, rng, FOOTWEAR):
    from stepup.metrics import report_from_scores
    s, l, accs = cross_fw_eval(f, y, fw, k, method, z, rng, FOOTWEAR)
    r = report_from_scores(s, l)
    return (r["eer"], r["fmr100"], r["fnmr"], *accs)        # (eer, fmr100, fnmr, rank1, rank3, rank5)


def enroll_curve(f, y, fw, ks, method, z, FOOTWEAR, repeats=3):
    """(k, EER, FMR100, FNMR, rank1, rank3, rank5) vs enrolled footsteps, averaged over `repeats` draws."""
    out = []
    for k in ks:
        m = [metrics_at(f, y, fw, k, method, z, np.random.default_rng(r), FOOTWEAR) for r in range(repeats)]
        m = np.array(m)                                     # (repeats, 6) -> mean each column
        out.append((k, *[float(v) for v in m.mean(0)]))     # (k, eer, fmr100, fnmr, rank1, rank3, rank5)
    return out


# ----------------------------- plots -----------------------------
def plot_all(curves, f, y, fw, ks, z, FOOTWEAR, out_dir):
    import pathlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) HEADLINE: rank-1/3/5 identification ACCURACY vs #enrolled footsteps (BD-CSPN's metric).
    # One colour per method; rank-1 solid, rank-3 dashed, rank-5 dotted (p[4]/p[5]/p[6]).
    fig, a = plt.subplots(figsize=(6.4, 4.3))
    styles = [("rank1", 4, "-", "o"), ("rank3", 5, "--", "s"), ("rank5", 6, ":", "^")]
    for ci, (name, pts) in enumerate(curves.items()):
        col = f"C{ci}"
        for lbl, idx, ls, mk in styles:
            a.plot([p[0] for p in pts], [p[idx] * 100 for p in pts], ls=ls, marker=mk, ms=4,
                   color=col, label=f"{name} {lbl}")
    a.set_xlabel("# enrolled footsteps (k)"); a.set_ylabel("rank-1/3/5 identification accuracy (%)")
    a.set_title("Enrollment: more footsteps -> better recognition")
    a.legend(fontsize=6, ncol=len(curves)); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "enroll_acc_vs_k.png", dpi=120); plt.close(fig)

    # 2) EER vs #enrolled footsteps (verification / competition metric) + 1/sqrt(K) variance trend
    fig, a = plt.subplots(figsize=(6, 4.3))
    for name, pts in curves.items():
        a.plot([p[0] for p in pts], [p[1] * 100 for p in pts], marker="o", ms=4, label=name)
    base = curves[list(curves)[0]]
    e1, k1 = base[0][1] * 100, base[0][0]
    a.plot(ks, [e1 * np.sqrt(k1 / k) for k in ks], "k--", lw=1, alpha=.6,
           label="1/√K prototype-variance trend")     # BD-CSPN: variance ~ 1/(K+Z)
    a.set_xlabel("# enrolled footsteps (k)"); a.set_ylabel("cross-fw EER (%)")
    a.set_title("Enrollment: verification EER vs #footsteps")
    a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "enroll_eer_vs_k.png", dpi=120); plt.close(fig)

    # 2) DET at k = 1 / 5 / 10 (BD-CSPN)
    from sklearn.metrics import roc_curve
    fig, a = plt.subplots(figsize=(5.2, 5))
    for k, c in zip([kk for kk in (1, 5, 10) if kk in ks], ["tab:blue", "tab:orange", "tab:green"]):
        s, l = cross_fw_scores(f, y, fw, k, "bdcspn", z, np.random.default_rng(0), FOOTWEAR)
        fpr, tpr, _ = roc_curve(l.astype(int), s); fnr = 1 - tpr
        i = np.nanargmin(np.abs(fnr - fpr))
        a.plot(fpr * 100, fnr * 100, color=c, lw=2, label=f"k={k} (EER {(fpr[i]+fnr[i])/2*100:.1f})")
    a.plot([0, 100], [0, 100], ls=":", c="gray", lw=1)
    a.set_xlim(0, 40); a.set_ylim(0, 40); a.set_xlabel("FMR (%)"); a.set_ylabel("FNMR (%)")
    a.set_title("DET vs #enrolled footsteps (BD-CSPN)"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "enroll_det.png", dpi=120); plt.close(fig)

    # 3) t-SNE of probe features BEFORE vs AFTER cross-class rectification (one enroll shoe)
    try:
        from stepup.eval import project_2d
        A = FOOTWEAR[0]
        fu = preprocess(f, "ncm")
        protos = torch.stack([F.normalize(fu[np.where((fw == A) & (y == i))[0][:5]].mean(0), dim=0)
                              for i in np.unique(y) if ((fw == A) & (y == i)).sum() > 0])
        pm = fw != A
        q0 = F.normalize(fu[pm], dim=1)
        _, q1 = bdcspn_rectify(protos, q0, z=z)
        sub = np.random.default_rng(0).choice(len(q0), min(1500, len(q0)), replace=False)
        yy = y[pm][sub]
        xy0 = project_2d(q0[sub].numpy()); xy1 = project_2d(q1[sub].numpy())
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(xy0[:, 0], xy0[:, 1], c=yy, cmap="tab20", s=7, alpha=.7); ax[0].set_title("probe: before rectification")
        ax[1].scatter(xy1[:, 0], xy1[:, 1], c=yy, cmap="tab20", s=7, alpha=.7); ax[1].set_title("probe: after cross-class shift")
        for aa in ax:
            aa.set_xticks([]); aa.set_yticks([])
        fig.tight_layout(); fig.savefig(out_dir / "enroll_tsne.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"  (t-SNE skipped: {e})", flush=True)
    print(f"  plots -> {out_dir}/enroll_*.png", flush=True)


# ----------------------------- driver -----------------------------
def run(f, y, fw, ks, z, out_dir, FOOTWEAR, adabn_feats=None):
    import pathlib
    import pandas as pd
    curves = {}

    def _show(m, pts):                                # print rank-1/3/5 acc (headline) + EER/FMR100
        print(f"  {m:12s}  " + "  ".join(
            f"k{k}:r1{r1*100:.1f}/r3{r3*100:.1f}/r5{r5*100:.1f}/eer{e*100:.1f}/fmr100{fmr*100:.1f}"
            for k, e, fmr, fnmr, r1, r3, r5 in pts), flush=True)

    for method in ("ncm", "cl2n", "bdcspn"):
        curves[method] = enroll_curve(f, y, fw, ks, method, z, FOOTWEAR); _show(method, curves[method])
    if adabn_feats is not None:                       # BD-CSPN on AdaBN-adapted embeddings
        fa, ya, fwa = adabn_feats
        curves["bdcspn+adabn"] = enroll_curve(fa, ya, fwa, ks, "bdcspn", z, FOOTWEAR)
        _show("bdcspn+adabn", curves["bdcspn+adabn"])
    plot_all(curves, f, y, fw, ks, z, FOOTWEAR, out_dir)
    rows = [dict(method=m, k=k, eer=e, fmr100=fmr, fnmr=fnmr, rank1=r1, rank3=r3, rank5=r5)
            for m, pts in curves.items() for k, e, fmr, fnmr, r1, r3, r5 in pts]
    pd.DataFrame(rows).to_parquet(pathlib.Path(out_dir) / "enroll_results.parquet", index=False)
    return curves


def main():
    from stepup.config import dev, FOOTWEAR
    from stepup.data import build_datasets
    from stepup.eval import embed_dataset
    from finetune import load_backbone_trainable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr"); ap.add_argument("--tag", default="")
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"])
    ap.add_argument("--ks", default="1,2,3,5,8,10", help="enrolled-footstep counts to sweep")
    ap.add_argument("--z", type=int, default=8, help="BD-CSPN: top-Z confident probes per prototype")
    ap.add_argument("--adabn", action="store_true", help="also run BD-CSPN on AdaBN-adapted embeddings")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    ap.add_argument("--wandb-project", default="stepup-footstep")
    args = ap.parse_args()
    ks = [int(v) for v in args.ks.split(",")]

    net, cfg = load_backbone_trainable(args.model, args.hf_repo, args.hf_token, args.tag)
    net = net.to(dev).eval()
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)
    f, y, fw = embed_dataset(net, ds["test"])
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    print(f"enroll {args.model}  test {len(f)} steps / {len(np.unique(y))} ids", flush=True)

    adabn_feats = None
    if args.adabn:
        from quantize import adabn_recalibrate
        from torch.utils.data import DataLoader
        batches = [xb for xb, _, _ in DataLoader(ds["test"], batch_size=64, shuffle=False, num_workers=0)]
        adabn_recalibrate(net, batches[:8], dev)
        fa, ya, fwa = embed_dataset(net, ds["test"])
        adabn_feats = (fa, ya.numpy() if hasattr(ya, "numpy") else np.asarray(ya), fwa)

    curves = run(f, y, fw, ks, args.z, args.out, list(FOOTWEAR), adabn_feats)

    if args.wandb != "disabled":                       # optional: mirror curves + plots to wandb
        import pathlib
        import wandb
        wr = wandb.init(project=args.wandb_project, mode=args.wandb,
                        name=f"enroll_{args.model}" + (f"_{args.tag}" if args.tag else ""))
        tbl = wandb.Table(columns=["method", "k", "eer", "fmr100", "fnmr", "rank1", "rank3", "rank5"],
                          data=[[m, k, e, fmr, fnmr, r1, r3, r5]
                                for m, pts in curves.items() for k, e, fmr, fnmr, r1, r3, r5 in pts])
        wr.log({"enroll/results": tbl})
        for p in sorted(pathlib.Path(args.out).glob("enroll_*.png")):
            wr.log({f"enroll/{p.stem}": wandb.Image(str(p))})
        for m, pts in curves.items():                  # best (largest-k) metrics per method -> summary
            wr.summary[f"{m}_eer_bestk"] = pts[-1][1]
            wr.summary[f"{m}_fmr100_bestk"] = pts[-1][2]
            wr.summary[f"{m}_rank1_bestk"] = pts[-1][4]
            wr.summary[f"{m}_rank3_bestk"] = pts[-1][5]
            wr.summary[f"{m}_rank5_bestk"] = pts[-1][6]
        wr.finish()
        print("  logged curves + plots to wandb", flush=True)


if __name__ == "__main__":
    main()
