"""Sequential-freeze JOINT fine-tune: backbone + walk aggregator, end-to-end.

Why: every frozen-backbone head we tried (PMA aggregator, disentangling head) equalled or lost to
the raw embedding, because a FROZEN backbone cannot co-adapt to the head. This trains them TOGETHER:
  Phase 1 (freeze):   backbone frozen, train only the aggregator (warm-up).
  Phase 2 (unfreeze):  unfreeze the backbone, fine-tune backbone + aggregator jointly at a LOW lr.
This is the one mechanism that lets the backbone reshape its per-footstep embedding so a set of them
aggregates better -- the "sequential freezing" idea. We report BOTH the walk cross-footwear EER and
the SINGLE-footstep cross-footwear EER (did co-adaptation improve the raw embedding too?).

  python finetune.py --model gaitcnn_snr --tag dsu --pack-device cuda \
      --k 5 --frozen-epochs 8 --joint-epochs 12 --lr-head 1e-3 --lr-backbone 1e-5 --wandb online
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses

from stepup.config import ARTIFACTS, T, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.models import registry, set_adaptive_proj, set_dropout
from aggregate import WalkAggregator, walk_windows, walk_metrics


def load_backbone_trainable(model, hf_repo, hf_token, tag="", in_frames=0, scratch=False):
    fname = f"{model}_{tag}_best.pt" if tag else f"{model}_best.pt"
    ckpt = str(ARTIFACTS / fname)
    if not os.path.exists(ckpt):
        if not hf_repo:
            raise FileNotFoundError(f"{fname} not local and no --hf-repo")
        from stepup.hf import fetch_file
        ckpt = fetch_file(hf_repo, fname, hf_token)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]; set_dropout(cfg.get("dropout", 0.0)); set_adaptive_proj(cfg.get("adaptive_proj", False))
    data_t = (cfg.get("sample3d") or cfg.get("pack_res") or (T, T, T))[0]
    if cfg.get("stride_pairs"):
        data_t *= 2
    if in_frames > 0:
        data_t = in_frames
    spec = registry(cfg["sample3d"], data_t)[model]
    # Rebuild with the architecture toggles the checkpoint was trained with (hpp changes feat_dim),
    # overlaid on the registry base kwargs; old checkpoints without a saved kw fall back cleanly.
    _saved_kw = ck.get("kw", {}) or {}
    _kw = {**spec["kw"], **{k: _saved_kw[k] for k in ("hpp", "dsu", "mixstyle") if k in _saved_kw}}
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **_kw).to(dev)
    if not scratch:
        net.load_state_dict(ck["state"])          # scratch=True -> keep the fresh random init
    return net, cfg


@torch.no_grad()
def eval_all(net, agg, ds, k, ks=(1, 3, 5, 10)):
    """single-footstep (raw backbone) + walk (aggregated) cross-footwear metrics on a split."""
    net.eval(); agg.eval()
    f, y, fw = embed_dataset(net, ds)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y); fw = np.asarray(fw)
    single = walk_metrics(f, y, fw)                          # raw single-footstep embedding
    m = ds.m.reset_index(drop=True)
    win = walk_windows(m, k)
    embs = []
    for _, _, rows in win:
        embs.append(agg(f[rows].unsqueeze(0).to(dev)).squeeze(0).cpu())
    labs = np.array([lab for lab, _, _ in win]); fws = np.array([w for _, w, _ in win])
    walk = walk_metrics(torch.stack(embs), labs, fws)
    return single, walk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr"); ap.add_argument("--tag", default="")
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"])
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--sub-centers", type=int, default=3)
    ap.add_argument("--frozen-epochs", type=int, default=8); ap.add_argument("--joint-epochs", type=int, default=12)
    ap.add_argument("--P", type=int, default=8); ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=1e-3); ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        wb = _wb.init(project="stepup-footstep", name=f"ft_{args.model}" + (f"_{args.tag}" if args.tag else ""),
                      mode=args.wandb, config=vars(args))

    net, cfg = load_backbone_trainable(args.model, args.hf_repo, args.hf_token, args.tag)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)
    D = cfg["embed_dim"]
    agg = WalkAggregator(D).to(dev)
    ids = sorted(np.unique(ds["train"].m.ParticipantID.to_numpy())); id2c = {i: c for c, i in enumerate(ids)}
    arc = losses.SubCenterArcFaceLoss(num_classes=len(ids), embedding_size=D,
                                      sub_centers=args.sub_centers).to(dev)

    # walk row-groups (single-footwear walks) from the train manifest
    win = walk_windows(ds["train"].m.reset_index(drop=True), args.k)
    walks = [(id2c[lab], rows) for lab, _, rows in win if lab in id2c]
    print(f"train walks: {len(walks)} (k={args.k})  ids={len(ids)}", flush=True)

    # baseline BEFORE fine-tuning
    s0, w0 = eval_all(net, agg, ds["test"], args.k)
    print(f"  BEFORE  single EER {s0['eer']:.3f} rank1 {s0['rank1']:.3f} | walk EER {w0['eer']:.3f} "
          f"rank1 {w0['rank1']:.3f}", flush=True)

    rng = np.random.default_rng(0)

    def run_phase(name, epochs, train_backbone, lr):
        for p in net.parameters():
            p.requires_grad_(train_backbone)
        params = list(agg.parameters()) + list(arc.parameters())
        if train_backbone:
            params = params + [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        steps = max(1, len(walks) // (args.P * args.M))
        for ep in range(epochs):
            net.train() if train_backbone else net.eval()
            agg.train(); tot = 0.0
            for _ in range(steps):
                bwalks, by = [], []
                for i in rng.choice(len(ids), min(args.P, len(ids)), replace=False):
                    cand = [w for w in walks if w[0] == i]
                    if not cand:
                        continue
                    for _ in range(args.M):
                        c, rows = cand[rng.integers(len(cand))]
                        bwalks.append(rows); by.append(c)
                if not bwalks:
                    continue
                flat = [r for rows in bwalks for r in rows]                 # all footsteps in the batch
                xs = torch.stack([ds["train"][r][0] for r in flat]).to(dev)  # (B*k, C, T, H, W)
                emb = net(xs).reshape(len(bwalks), args.k, D)               # backbone (grad if unfrozen)
                w = agg(emb)                                                # walk embeddings (B, D)
                loss = arc(w, torch.tensor(by, device=dev))
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            s, wm = eval_all(net, agg, ds["test"], args.k)
            print(f"  [{name}] ep {ep + 1:>2}/{epochs}  loss {tot / steps:6.3f}  "
                  f"single EER {s['eer']:.3f} rank1 {s['rank1']:.3f} | walk EER {wm['eer']:.3f} "
                  f"rank1 {wm['rank1']:.3f}", flush=True)
            if wb is not None:
                wb.log({"phase": name, "single_eer": s["eer"], "single_rank1": s["rank1"],
                        "walk_eer": wm["eer"], "walk_rank1": wm["rank1"]})
        return eval_all(net, agg, ds["test"], args.k)

    print("=== Phase 1: backbone FROZEN, train aggregator ===", flush=True)
    run_phase("frozen", args.frozen_epochs, train_backbone=False, lr=args.lr_head)
    print("=== Phase 2: backbone UNFROZEN, joint fine-tune (low lr) ===", flush=True)
    s1, w1 = run_phase("joint", args.joint_epochs, train_backbone=True, lr=args.lr_backbone)

    print("\n=== RESULT (test cross-footwear) ===", flush=True)
    print(f"  BEFORE   single EER {s0['eer']:.3f} rank1 {s0['rank1']:.3f} | walk EER {w0['eer']:.3f} "
          f"rank1 {w0['rank1']:.3f}", flush=True)
    print(f"  AFTER    single EER {s1['eer']:.3f} rank1 {s1['rank1']:.3f} | walk EER {w1['eer']:.3f} "
          f"rank1 {w1['rank1']:.3f}", flush=True)
    print(f"  --> single EER {(s0['eer'] - s1['eer'])*100:+.2f}pp   walk EER "
          f"{(w0['eer'] - w1['eer'])*100:+.2f}pp  ({'WIN' if s1['eer'] < s0['eer'] - 1e-3 else 'no gain'})",
          flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
