"""Adversarial-AutoAugment footwear-invariance policy (Zhang et al., ICLR 2020; AutoAugment,
Cubuk CVPR'19), adapted to plantar-pressure.

A learnable augmentation POLICY samples pressure-domain operations (magnitude scale, gamma, blur,
spatial warp, time-frame dropout, region mask) that simulate footwear shift. Each batch is augmented
into M copies; the BACKBONE minimises the identity loss over them (learning invariance) while the
POLICY is updated by REINFORCE to MAXIMISE that loss (inventing the hardest footwear-like shifts).
Unlike the halting-RL (efficiency only), this RL feeds the backbone and targets cross-footwear EER.

We fine-tune from the trained backbone and report cross-footwear EER BEFORE vs AFTER -- shipped only
if it wins. Reference code checked: kakaobrain/fast-autoaugment (MIT), DeepVoltaire/AutoAugment (MIT);
we port the argument, own code.

  python rl_augment.py --model gaitcnn_snr --pack-device cuda --epochs 15 --M 4 \
      --lr-backbone 1e-5 --lr-policy 1e-2 --wandb online
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

from stepup.config import ARTIFACTS, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.rl_viz import RLPolicyLog
from aggregate import walk_metrics
from finetune import load_backbone_trainable


# ---------------- GEOMETRIC augmentation ops (x: (B,1,C,H,W) in [0,1]) ----------------
# The competition WINNERS searched geometric transforms (Rotation, VerticalFlip; NOT HorizontalFlip
# -- that undoes the medial-lateral mirror) with an RL agent. Geometry PRESERVES the pressure
# biometric; intensity/blur ops DESTROY it (the distribution IS the identity). So the policy searches
# rotate / translate / scale / vertical-flip / mild-noise / time-jitter -- cheap and identity-safe.
import math

def _affine(x, mat):                                                                    # mat: (2,3)
    B, _, C, H, W = x.shape
    theta = mat.to(x.device, x.dtype).unsqueeze(0).repeat(B * C, 1, 1)
    grid = F.affine_grid(theta, (B * C, 1, H, W), align_corners=False)
    return F.grid_sample(x.reshape(B * C, 1, H, W), grid, align_corners=False,
                         padding_mode="zeros").reshape(B, 1, C, H, W)

def op_rotate(x, m):
    a = (m - 0.5) * 0.52; c, s = math.cos(a), math.sin(a)                               # +-15 deg
    return _affine(x, torch.tensor([[c, -s, 0.0], [s, c, 0.0]]))
def op_translate(x, m):
    t = (m - 0.5) * 0.3                                                                 # small shift
    return _affine(x, torch.tensor([[1.0, 0, t], [0, 1.0, t]]))
def op_scale(x, m):
    sc = 1 + (m - 0.5) * 0.3                                                            # zoom +-15%
    return _affine(x, torch.tensor([[sc, 0, 0], [0, sc, 0]]))
def op_vflip(x, m):
    return torch.flip(x, dims=[-2]) if m > 0.5 else x                                   # flip foot-length
def op_noise(x, m):
    return (x + torch.randn_like(x) * (0.03 * m)).clamp(0, 1)                           # mild sensor noise
def op_timejitter(x, m):
    sh = int(round((m - 0.5) * 4))                                                      # small cadence shift
    return torch.roll(x, shifts=sh, dims=2) if sh != 0 else x

OPS = [op_rotate, op_translate, op_scale, op_vflip, op_noise, op_timejitter]
OP_NAMES = ["rotate", "translate", "scale", "vflip", "noise", "timejitter"]


class AugPolicy(nn.Module):
    """AutoAugment-style policy: learnable logits over (operation, magnitude-bin). Not input-
    conditioned (dataset-level policy). Sampled M times per batch; updated by REINFORCE."""
    def __init__(self, n_ops=len(OPS), n_mag=10):
        super().__init__()
        self.op_logits = nn.Parameter(torch.zeros(n_ops))
        self.mag_logits = nn.Parameter(torch.zeros(n_ops, n_mag))
        self.n_mag = n_mag

    def sample(self):
        od = torch.distributions.Categorical(logits=self.op_logits)
        op = od.sample()
        md = torch.distributions.Categorical(logits=self.mag_logits[op])
        mag = md.sample()
        logp = od.log_prob(op) + md.log_prob(mag)
        return int(op), (mag.item() + 0.5) / self.n_mag, logp


def apply_op(x, op_idx, mag):
    return OPS[op_idx](x, mag)


@torch.no_grad()
def cross_fw_eer(net, ds):
    net.eval()
    f, y, fw = embed_dataset(net, ds)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return walk_metrics(f, y, np.asarray(fw))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr"); ap.add_argument("--tag", default="")
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"])
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--scratch", action="store_true",
                    help="train the backbone FROM SCRATCH so it co-evolves with the policy (the "
                         "correct Adversarial-AutoAugment setup; fine-tuning a converged net collapses)")
    ap.add_argument("--warmup", type=int, default=6,
                    help="ramp augmentation strength 0->1 over this many epochs so the adversary "
                         "doesn't overwhelm an untrained backbone")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--P", type=int, default=16); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr-backbone", type=float, default=1e-5); ap.add_argument("--lr-policy", type=float, default=1e-2)
    ap.add_argument("--sub-centers", type=int, default=3); ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        wb = _wb.init(project="stepup-footstep", name=f"rlaug_{args.model}" + (f"_{args.tag}" if args.tag else ""),
                      mode=args.wandb, config=vars(args))

    net, cfg = load_backbone_trainable(args.model, args.hf_repo, args.hf_token, args.tag,
                                       scratch=args.scratch)
    print(f"  backbone: {'FROM SCRATCH (co-evolve)' if args.scratch else 'fine-tune from checkpoint'}",
          flush=True)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)
    dstr = ds["train"]
    ids = sorted(np.unique(dstr.m.ParticipantID.to_numpy())); id2c = {i: c for c, i in enumerate(ids)}
    pools = {i: np.where(dstr.m.ParticipantID.to_numpy() == i)[0] for i in ids}
    arc = losses.SubCenterArcFaceLoss(num_classes=len(ids), embedding_size=cfg["embed_dim"],
                                      sub_centers=args.sub_centers).to(dev)
    policy = AugPolicy().to(dev)
    opt_net = torch.optim.AdamW(list(net.parameters()) + list(arc.parameters()),
                                lr=args.lr_backbone, weight_decay=args.weight_decay)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)

    base = cross_fw_eer(net, ds["test"])
    print(f"  BEFORE  test cross-fw EER {base['eer']:.3f} rank1 {base['rank1']:.3f}", flush=True)
    rng = np.random.default_rng(0)

    rl_log = RLPolicyLog(OP_NAMES, eer_label="test cross-fw EER")                        # 6 thesis figures
    sample_x = None                                                                     # real cube for Fig 1
    for ep in range(args.epochs):
        net.train(); tot = 0.0; op_hits = np.zeros(len(OPS))
        strength = min(1.0, (ep + 1) / max(1, args.warmup))                             # ramp adversary in
        for _ in range(args.steps):
            rows = []
            for i in rng.choice(ids, min(args.P, len(ids)), replace=False):
                idx = pools[i]; rows += list(idx[rng.integers(0, len(idx), args.K)] )
            if hasattr(dstr, "gather"):                                                 # cuda pack path
                xb, yb, _ = dstr.gather(rows)                                           # (B,1,2T,H,W), labels
            else:                                                                       # cpu/memmap path
                xb = torch.stack([dstr[r][0] for r in rows]).to(dev)
                yb = torch.tensor([dstr[r][1] for r in rows], device=dev)
            if sample_x is None:
                sample_x = xb[:1].detach().clone()                                      # -> Fig 1 panel
            losses_m, logps = [], []
            for _ in range(args.M):                                                    # M augmented copies
                op, mag, logp = policy.sample(); op_hits[op] += 1
                xa = strength * apply_op(xb, op, mag) + (1 - strength) * xb             # blend: ramp strength
                emb = net(xa)
                emb = emb[0] if isinstance(emb, tuple) else emb
                l = arc(emb, yb)
                losses_m.append(l); logps.append(logp)
            L = torch.stack(losses_m)                                                  # (M,)
            reward = (L.detach() - L.detach().mean())                                   # advantage
            # backbone graph and policy graph are separate -> no retain_graph needed
            opt_net.zero_grad(); L.mean().backward(); opt_net.step()                    # minimise id loss
            pol_loss = -(torch.stack(logps) * reward).mean()                           # REINFORCE: maximise loss
            opt_pol.zero_grad(); pol_loss.backward(); opt_pol.step()
            tot += L.mean().item()
        te = cross_fw_eer(net, ds["test"])
        # ---- canonical RL training scalars (reward, entropy, mean magnitude); full policy state
        #      goes to the six figures at the end, not into per-epoch streams ----
        rl_scalars = rl_log.update(policy, reward=tot / args.steps, eer=te["eer"])
        with torch.no_grad():
            op_p = torch.softmax(policy.op_logits, 0)
        top = OP_NAMES[int(op_p.argmax())]
        print(f"  ep {ep + 1:>2}/{args.epochs}  id {tot / args.steps:6.3f}  "
              f"test EER {te['eer']:.3f} rank1 {te['rank1']:.3f}  | base {base['eer']:.3f}  "
              f"top-op {top}({op_p.max():.2f})  H {rl_scalars['rl/policy_entropy']:.2f}  "
              f"mag {rl_scalars['rl/mean_magnitude']:.2f}", flush=True)
        if wb is not None:
            wb.log({"epoch": ep + 1, "test_eer": te["eer"], "test_rank1": te["rank1"],
                    "base_eer": base["eer"], **rl_scalars})

    fin = cross_fw_eer(net, ds["test"])
    print(f"\n=== RESULT ===\n  BEFORE EER {base['eer']:.3f} rank1 {base['rank1']:.3f}\n"
          f"  AFTER  EER {fin['eer']:.3f} rank1 {fin['rank1']:.3f}\n"
          f"  --> EER {(base['eer'] - fin['eer'])*100:+.2f}pp  "
          f"({'WIN' if fin['eer'] < base['eer'] - 1e-3 else 'no gain'})", flush=True)
    print("  learned op probs:", {OP_NAMES[i]: round(p, 2) for i, p in
                                   enumerate(torch.softmax(policy.op_logits, 0).tolist())}, flush=True)
    print("  learned magnitudes (mean bin/10 per op):",
          {OP_NAMES[i]: round(float(m), 2) for i, m in enumerate(
              (torch.softmax(policy.mag_logits, 1) *
               torch.arange(policy.n_mag, device=policy.mag_logits.device)).sum(1) / policy.n_mag)},
          flush=True)
    # ---- the six interpretability figures (action panel, P(op) traj, entropy, magnitude,
    #      op heatmap, reward-vs-EER) written to artifacts/ and logged to wandb ----
    prefix = f"rlaug_{args.model}" + (f"_{args.tag}" if args.tag else "")
    rl_log.render(policy, ARTIFACTS, prefix=prefix, apply_op=apply_op, sample_x=sample_x, wandb_run=wb)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
