"""Second-stage WALK aggregator on top of a FROZEN footstep-embedding backbone.

Turns the set of per-footstep embeddings from a walking pass into ONE walk embedding via a
Set-Transformer (PMA pooling; Lee et al., ICML 2019), trained with Sub-Center ArcFace (K
sub-centres per identity -> each person's several footwear-conditioned clusters get their own
centre; Deng et al., ECCV 2020). The backbone is FROZEN and the enroll->cosine->threshold
protocol is unchanged -- this only replaces mean-pooling of k steps with a LEARNED pooling.

It reports the cross-footwear (hard) and mixed-gallery accumulated rank-1 k-curves for the
LEARNED aggregator vs the MEAN-pool baseline, so the gain is measured directly.

  python aggregate.py --model gaitcnn_snr --hf-repo Bechir23/stepup-footstep --k 5 --epochs 30
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

from stepup.config import ARTIFACTS, FOOTWEAR, T, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.metrics import enroll_templates
from stepup.models import registry, set_dropout


# ---------------- Set-Transformer (MAB / SAB / PMA) ----------------
class MAB(nn.Module):
    """Multihead attention block: attend q over k, residual + FF (Set Transformer)."""
    def __init__(self, d, h=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, q, k):
        a, _ = self.attn(q, k, k)
        x = self.ln1(q + a)
        return self.ln2(x + self.ff(x))


class SAB(nn.Module):
    """Self-attention block over the set (lets steps talk to each other before pooling)."""
    def __init__(self, d, h=4):
        super().__init__()
        self.mab = MAB(d, h)

    def forward(self, x):
        return self.mab(x, x)


class WalkAggregator(nn.Module):
    """Set-Transformer aggregator: SAB self-attention + PMA pooling (Lee et al., ICML 2019).
    Set of N step embeddings (B,N,D) -> one L2-normalised walk embedding (B,D)."""
    def __init__(self, d=64, h=4, depth=1):
        super().__init__()
        self.enc = nn.Sequential(*[SAB(d, h) for _ in range(depth)])
        self.seed = nn.Parameter(torch.randn(1, 1, d))
        self.pool = MAB(d, h)

    def forward(self, x):                                    # x: (B, N, D)
        x = self.enc(x)
        w = self.pool(self.seed.expand(x.size(0), -1, -1), x).squeeze(1)
        return F.normalize(w, dim=-1)


class NANAggregator(nn.Module):
    """Neural Aggregation Network head (Yang et al., CVPR 2017): TWO cascaded dot-product
    attention blocks (no self-attention). q0 init at zero = average pooling; the second block's
    query is content-adapted q1=tanh(W r0). The reference architecture, so we can ablate it vs PMA."""
    def __init__(self, d=64):
        super().__init__()
        self.q0 = nn.Parameter(torch.zeros(d))               # starts from average pooling
        self.W = nn.Linear(d, d)

    def forward(self, x):                                    # x: (B, N, D)
        a0 = torch.softmax(x @ self.q0, dim=1)               # (B, N)
        r0 = (a0.unsqueeze(-1) * x).sum(1)                   # (B, D)
        q1 = torch.tanh(self.W(r0))                          # content-adapted query
        a1 = torch.softmax((x * q1.unsqueeze(1)).sum(-1), dim=1)
        return F.normalize((a1.unsqueeze(-1) * x).sum(1), dim=-1)


def make_aggregator(kind, d):
    return {"pma": WalkAggregator, "nan": NANAggregator}[kind](d)


# ---------------- backbone + frozen embeddings ----------------
def load_backbone(model, ckpt, hf_repo, hf_token, in_frames=0, tag=""):
    fname = f"{model}_{tag}_best.pt" if tag else f"{model}_best.pt"
    if not os.path.exists(ckpt or ""):
        ckpt = str(ARTIFACTS / fname)
    if not os.path.exists(ckpt):
        from stepup.hf import fetch_file
        ckpt = fetch_file(hf_repo, fname, hf_token)
        print(f"fetched checkpoint from HF: {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    set_dropout(cfg.get("dropout", 0.0))
    data_t = (cfg.get("sample3d") or cfg.get("pack_res") or (T, T, T))[0]
    if cfg.get("stride_pairs"):
        data_t *= 2                      # a stride = left+right concatenated in time (as in train.py)
    if in_frames > 0:                    # manual override if the auto value mismatches the ckpt
        data_t = in_frames
    spec = registry(cfg["sample3d"], data_t)[model]
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **spec["kw"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, cfg


@torch.no_grad()
def frozen(net, ds):
    """Per-footstep embeddings aligned with the manifest rows (needed to group into walks)."""
    f, _, _ = embed_dataset(net, ds)
    return f, ds.m.reset_index(drop=True)


def walk_windows(m, k):
    """List of (label, footwear, row-index-list) for each k-consecutive-step window within one
    (identity, footwear, pass) -- the deployment 'short walk'."""
    pid = m.ParticipantID.to_numpy(); fw = m.Footwear.to_numpy()
    passid = m.PassID.to_numpy(); step = m.FootstepID.to_numpy()
    groups = {}
    for i in range(len(m)):
        groups.setdefault((pid[i], fw[i], passid[i]), []).append(i)
    out = []
    for (p, w, _), rows in groups.items():
        rows = sorted(rows, key=lambda r: step[r])
        for s in range(0, len(rows) - k + 1, k):
            out.append((int(p), w, rows[s:s + k]))
    return out


# ---------------- training the aggregator ----------------
def train_aggregator(agg, f, m, k, ids, id2c, epochs, P, M, lr, wd, sub_centers):
    pools = {i: np.where(m.ParticipantID.to_numpy() == i)[0] for i in ids}
    loss_fn = losses.SubCenterArcFaceLoss(num_classes=len(ids), embedding_size=f.shape[1],
                                          sub_centers=sub_centers).to(dev)
    opt = torch.optim.AdamW(list(agg.parameters()) + list(loss_fn.parameters()), lr=lr,
                            weight_decay=wd)
    rng = np.random.default_rng(0)
    steps = max(1, sum(len(v) for v in pools.values()) // (P * M * k))
    for ep in range(epochs):
        agg.train(); tot = 0.0
        for _ in range(steps):
            bx, by = [], []
            for i in rng.choice(ids, min(P, len(ids)), replace=False):     # P identities
                idx = pools[i]
                for _ in range(M):                                          # M walks each
                    sel = idx[rng.integers(0, len(idx), k)]                 # k random steps (mix shoes)
                    bx.append(f[sel]); by.append(id2c[i])
            xb = torch.stack(bx).to(dev)                                    # (P*M, k, D)
            w = agg(xb)
            loss = loss_fn(w, torch.tensor(by, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  agg ep {ep + 1:>2}/{epochs}  loss {tot / steps:6.3f}", flush=True)
    agg.eval()
    return agg


# ---------------- evaluation: learned vs mean pooling ----------------
@torch.no_grad()
def build_walk_embeds(f, windows, agg=None):
    """One embedding per window: mean-pool (agg=None) or the learned aggregator."""
    embs, labs, fws = [], [], []
    for lab, w, rows in windows:
        steps = f[rows]
        if agg is None:
            e = F.normalize(steps.mean(0), dim=0)
        else:
            e = agg(steps.unsqueeze(0).to(dev)).squeeze(0).cpu()
        embs.append(e); labs.append(lab); fws.append(w)
    return torch.stack(embs), np.array(labs), np.array(fws)


def cross_fw_rank1(embs, labs, fws):
    """Leave-one-footwear-out rank-1 over walk embeddings (probe shoe != enrol shoe)."""
    y = torch.tensor(labs); accs = []
    for enrol in FOOTWEAR:
        g, p = fws == enrol, fws != enrol
        if g.sum() == 0 or p.sum() == 0:
            continue
        templ, ids = enroll_templates(embs[g], y[g])
        pred = ids[(F.normalize(embs[p]) @ templ.t()).argmax(1)]
        accs.append((pred == y[p]).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr")
    ap.add_argument("--in-frames", type=int, default=0, help="override backbone in_frames if the auto value mismatches the checkpoint")
    ap.add_argument("--tag", default="", help="load {model}_{tag}_best.pt (a specific ablation run)")
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"], help="override the checkpoint's pack device (cuda = fast eval on Colab)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--k", type=int, default=5, help="steps per walk the aggregator is trained on")
    ap.add_argument("--ks", default="1,3,5,10", help="k-curve to report")
    ap.add_argument("--sub-centers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=32); ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--aggregator", default="pma", choices=["pma", "nan"],
                    help="pma = Set-Transformer (ours); nan = Neural Aggregation Network head")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        wb = _wb.init(project="stepup-footstep", name=f"agg_{args.aggregator}_K{args.sub_centers}",
                      mode=args.wandb, config=vars(args))

    net, cfg = load_backbone(args.model, args.ckpt, args.hf_repo, args.hf_token, args.in_frames, args.tag)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device   # override (cuda = fast on Colab, memmap = low RAM)
    _, ds = build_datasets(cfg)
    ftr, mtr = frozen(net, ds["train"])
    fte, mte = frozen(net, ds["test"])
    D = ftr.shape[1]
    print(f"backbone loaded: embed_dim={D}  train steps={len(mtr)}  test steps={len(mte)}", flush=True)

    ids = sorted(mtr.ParticipantID.unique()); id2c = {i: c for c, i in enumerate(ids)}
    agg = make_aggregator(args.aggregator, D).to(dev)
    print(f"training aggregator={args.aggregator.upper()} + Sub-Center ArcFace (K={args.sub_centers}) "
          f"on frozen {D}-d embeddings, k={args.k} ...", flush=True)
    train_aggregator(agg, ftr, mtr, args.k, ids, id2c, args.epochs, args.P, args.M,
                     args.lr, args.weight_decay, args.sub_centers)

    print(f"\ncross-footwear (hard) accumulated rank-1  --  MEAN vs LEARNED ({args.aggregator}):", flush=True)
    for k in (int(v) for v in args.ks.split(",")):
        win = walk_windows(mte, k)
        em_mean, lab, fw = build_walk_embeds(fte, win, agg=None)
        em_lrn, _, _ = build_walk_embeds(fte, win, agg=agg)
        r_mean, r_lrn = cross_fw_rank1(em_mean, lab, fw), cross_fw_rank1(em_lrn, lab, fw)
        print(f"  {k:>2}-step   mean {r_mean:.3f}   learned {r_lrn:.3f}   "
              f"delta {r_lrn - r_mean:+.3f}", flush=True)
        if wb is not None:
            wb.log({f"rank1_mean_k{k}": r_mean, f"rank1_learned_k{k}": r_lrn, "k": k})
    ckpt_out = ARTIFACTS / f"agg_{args.model}_{args.aggregator}_K{args.sub_centers}.pt"
    torch.save(agg.state_dict(), ckpt_out)
    print(f"\nsaved aggregator -> {ckpt_out}", flush=True)
    if args.hf_repo:                                     # offload to HF like train.py
        from stepup.hf import push_files
        push_files(args.hf_repo, [ckpt_out], args.hf_token)
        print(f"pushed aggregator -> https://huggingface.co/{args.hf_repo}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
