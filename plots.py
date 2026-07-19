#!/usr/bin/env python
"""Comparison plots across all trained models, from the artifacts parquets.

  python plots.py                 # writes compare_eer.png, compare_accumulated.png
Reads test_*.parquet / verif_*.parquet / acc_*.parquet in artifacts/ (produced by train.py).
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stepup.config import ARTIFACTS
from stepup.eval import summarise


def main():
    import argparse
    ap = argparse.ArgumentParser(description="comparison plots from artifacts/")
    ap.add_argument("--hf-repo", default=None, help="also push the comparison figures to this HF repo")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    models, eer, r1, acc_curves = [], [], [], {}
    for f in sorted(glob.glob(str(ARTIFACTS / "test_*.parquet"))):
        name = f.split("test_")[-1][:-8]
        s = summarise(pd.read_parquet(f))
        vf = ARTIFACTS / f"verif_{name}.parquet"
        e = pd.read_parquet(vf).iloc[0]["eer"] if vf.exists() else s.get("cross_eer", np.nan)
        models.append(name); eer.append(e); r1.append(s.get("cross_rank1", np.nan))
        af = ARTIFACTS / f"acc_{name}.parquet"
        if af.exists():
            row = pd.read_parquet(af).iloc[0]
            acc_curves[name] = {int(k): float(v) for k, v in row.items()}
    if not models:
        print("no trained models in artifacts/ (run train.py first)")
        return

    order = np.argsort(eer)
    models = [models[i] for i in order]; eer = [eer[i] for i in order]; r1 = [r1[i] for i in order]

    # 1) leaderboard: cross-footwear EER (lower=better) and rank-1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(models, eer, color="#d1495b"); ax[0].set_title("cross-footwear EER (lower=better)")
    ax[0].invert_yaxis()
    ax[1].barh(models, r1, color="#2e86ab"); ax[1].set_title("cross-footwear rank-1 (higher=better)")
    ax[1].invert_yaxis()
    for a in ax:
        a.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "compare_eer.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) accumulated rank-1 as a walking pass grows (per model)
    if acc_curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, curve in acc_curves.items():
            ks = sorted(curve)
            ax.plot(ks, [curve[k] for k in ks], marker="o", ms=4, label=name)
        ax.set_xlabel("footsteps accumulated (k)"); ax.set_ylabel("cross-footwear rank-1")
        ax.set_title("accumulated identification over a walking pass")
        ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(ARTIFACTS / "compare_accumulated.png", dpi=120,
                                        bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {ARTIFACTS/'compare_eer.png'} and {ARTIFACTS/'compare_accumulated.png'}")
    print(pd.DataFrame(dict(model=models, cross_eer=np.round(eer, 3),
                            cross_rank1=np.round(r1, 3))).to_string(index=False))
    if args.hf_repo:
        from stepup.hf import push_files
        push_files(args.hf_repo, [ARTIFACTS / "compare_eer.png",
                                  ARTIFACTS / "compare_accumulated.png",
                                  ARTIFACTS / "model_leaderboard.parquet"], args.hf_token)
        print(f"pushed comparison figures -> https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
