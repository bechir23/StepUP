"""Stage 2 (learnable HEAD) --- footwear-DISENTANGLING head on the frozen backbone embedding.

The dead-end aggregator is replaced by a real learnable head that attacks the actual problem: the
seen/unseen (footwear) gap. On the FROZEN per-footstep embedding e it learns to split identity from
footwear and produce a footwear-invariant identity code z_id used for SINGLE-footstep matching:

    e ─▶ Encoder ─▶ z_id ──▶ ArcFace                       (identity kept)
                 └▶ z_fw ──▶ footwear classifier           (footwear concentrated here)
    GRL(z_id) ─▶ adversarial footwear classifier           (footwear REMOVED from z_id)  [Ganin'16]
    Decoder(z_id,z_fw) ─▶ e^  (reconstruction)             (force a complete, clean split) [VAE]

References: Ganin et al. DANN / gradient-reversal (JMLR 2016) -- the only invariance mechanism that
is a head on a frozen single-sample embedding; disentangled identity/covariate representation
(Wu et al., IEEE 2020); SNR (already in the backbone). We report z_id single-footstep cross-footwear
EER vs the RAW backbone embedding -- the head is only shipped if it BEATS raw.

  python invariance.py --model gaitcnn_snr --pack-device cuda --epochs 40 \
      --lambda-adv 1.0 --lambda-fw 1.0 --lambda-rec 0.1 --wandb online
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

from stepup.config import ARTIFACTS, FOOTWEAR, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.models import registry, set_dropout                       # noqa: F401 (via load_backbone)
from aggregate import load_backbone, frozen, walk_metrics             # reuse frozen-embed + metrics

FW2I = {fw: i for i, fw in enumerate(FOOTWEAR)}


# ---------------- gradient reversal (DANN) ----------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb):
    return _GradReverse.apply(x, lamb)


# ---------------- disentangling head ----------------
class DisentangleHead(nn.Module):
    def __init__(self, d=64, d_id=48, d_fw=16, h=128, n_fw=4, dropout=0.3):
        super().__init__()
        self.d_id, self.d_fw = d_id, d_fw
        self.enc = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(h, d_id + d_fw))
        self.dec = nn.Sequential(nn.Linear(d_id + d_fw, h), nn.ReLU(), nn.Linear(h, d))
        self.fw_head = nn.Sequential(nn.Linear(d_fw, n_fw))           # footwear lives in z_fw
        self.adv_head = nn.Sequential(nn.Linear(d_id, h // 2), nn.ReLU(), nn.Linear(h // 2, n_fw))

    def forward(self, e, lamb=1.0):
        z = self.enc(e)
        z_id, z_fw = z[:, :self.d_id], z[:, self.d_id:]
        recon = self.dec(torch.cat([z_id, z_fw], 1))
        fw_logits = self.fw_head(z_fw)                                # keep footwear in z_fw
        adv_logits = self.adv_head(grad_reverse(z_id, lamb))         # strip footwear from z_id
        return z_id, z_fw, recon, fw_logits, adv_logits

    @torch.no_grad()
    def encode_id(self, e):
        z = self.enc(e)
        return z[:, :self.d_id]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--in-frames", type=int, default=0)
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"])
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--hf-offload", action="store_true")
    ap.add_argument("--d-id", type=int, default=48); ap.add_argument("--d-fw", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--sub-centers", type=int, default=3)
    ap.add_argument("--lambda-adv", type=float, default=1.0, help="GRL footwear-removal weight")
    ap.add_argument("--lambda-fw", type=float, default=1.0, help="footwear-in-z_fw weight")
    ap.add_argument("--lambda-rec", type=float, default=0.1, help="reconstruction weight")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        rname = f"inv_{args.model}" + (f"_{args.tag}" if args.tag else "")
        wb = _wb.init(project="stepup-footstep", name=rname, mode=args.wandb, config=vars(args))

    net, cfg = load_backbone(args.model, args.ckpt, args.hf_repo, args.hf_token, args.in_frames, args.tag)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)

    def emb(split):
        f, y, fw = embed_dataset(net, ds[split])
        return f, y.numpy() if hasattr(y, "numpy") else np.asarray(y), np.asarray(fw)
    ftr, ytr, fwtr = emb("train")
    fva, yva, fwva = emb("val")
    fte, yte, fwte = emb("test")
    D = ftr.shape[1]
    ids = sorted(np.unique(ytr)); id2c = {i: c for c, i in enumerate(ids)}
    print(f"backbone embed_dim={D}  train={len(ftr)} val={len(fva)} test={len(fte)}  ids={len(ids)}",
          flush=True)

    # RAW backbone single-footstep cross-footwear baseline (the bar the head must beat)
    base_val = walk_metrics(fva, yva, fwva)
    base_te = walk_metrics(fte, yte, fwte)
    print(f"  RAW backbone  val: rank1 {base_val['rank1']:.3f} EER {base_val['eer']:.3f}  |  "
          f"test: rank1 {base_te['rank1']:.3f} EER {base_te['eer']:.3f}", flush=True)

    head = DisentangleHead(d=D, d_id=args.d_id, d_fw=args.d_fw,
                           n_fw=len(FOOTWEAR), dropout=cfg.get("dropout", 0.3)).to(dev)
    arc = losses.SubCenterArcFaceLoss(num_classes=len(ids), embedding_size=args.d_id,
                                      sub_centers=args.sub_centers).to(dev)
    opt = torch.optim.AdamW(list(head.parameters()) + list(arc.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    Etr = ftr.to(dev)
    ytr_c = torch.tensor([id2c[i] for i in ytr], device=dev)
    fwtr_i = torch.tensor([FW2I[w] for w in fwtr], device=dev)
    n = len(Etr); rng = np.random.default_rng(0)
    import copy
    best = dict(val=-1.0, state=copy.deepcopy(head.state_dict()), epoch=0, m=base_val)
    bad = 0
    for ep in range(args.epochs):
        head.train()
        lamb = args.lambda_adv * (2.0 / (1.0 + np.exp(-5.0 * ep / args.epochs)) - 1.0)  # DANN ramp
        idx = rng.permutation(n); tot = np.zeros(4)
        for s in range(0, n, args.batch):
            b = idx[s:s + args.batch]
            e = Etr[b]; yb = ytr_c[b]; wb_ = fwtr_i[b]
            z_id, z_fw, recon, fw_logits, adv_logits = head(e, lamb)
            l_id = arc(z_id, yb)
            l_fw = F.cross_entropy(fw_logits, wb_)
            l_adv = F.cross_entropy(adv_logits, wb_)
            l_rec = F.mse_loss(recon, e)
            loss = l_id + args.lambda_fw * l_fw + l_adv + args.lambda_rec * l_rec
            opt.zero_grad(); loss.backward(); opt.step()
            tot += [l_id.item(), l_fw.item(), l_adv.item(), l_rec.item()]
        nb = max(1, n // args.batch); tot /= nb

        head.eval()
        with torch.no_grad():
            zva = head.encode_id(fva.to(dev)).cpu()
        vm = walk_metrics(zva, yva, fwva)
        fit = vm["rank1"]; is_best = fit > best["val"] + 1e-4
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  inv ep {ep + 1:>2}/{args.epochs}  id {tot[0]:5.2f} fw {tot[1]:.2f} "
                  f"adv {tot[2]:.2f} rec {tot[3]:.3f}  lamb {lamb:.2f}  "
                  f"val z_id rank1 {vm['rank1']:.3f} EER {vm['eer']:.3f}  "
                  f"| raw {base_val['rank1']:.3f}/{base_val['eer']:.3f}"
                  f"{'  *' if is_best else ''}", flush=True)
        if wb is not None:
            wb.log({"epoch": ep + 1, "id": tot[0], "fw": tot[1], "adv": tot[2], "rec": tot[3],
                    "val_rank1": vm["rank1"], "val_eer": vm["eer"], "raw_rank1": base_val["rank1"],
                    "raw_eer": base_val["eer"], "lamb": lamb})
        if is_best:
            best.update(val=fit, state=copy.deepcopy(head.state_dict()), epoch=ep + 1, m=vm)
            bad = 0
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"  early stop @ ep {ep + 1} (best val rank1 {best['val']:.3f} @ ep {best['epoch']})",
                      flush=True)
                break

    head.load_state_dict(best["state"]); head.eval()
    with torch.no_grad():
        zte = head.encode_id(fte.to(dev)).cpu()
    tm = walk_metrics(zte, yte, fwte)
    print(f"\n=== TEST single-footstep cross-footwear  (best epoch {best['epoch']}) ===", flush=True)
    print(f"  RAW backbone   rank1 {base_te['rank1']:.3f}  EER {base_te['eer']:.3f}  "
          f"FMR100 {base_te['fmr100']:.3f}  auc {base_te['auc']:.3f}", flush=True)
    print(f"  z_id (head)    rank1 {tm['rank1']:.3f}  EER {tm['eer']:.3f}  "
          f"FMR100 {tm['fmr100']:.3f}  auc {tm['auc']:.3f}", flush=True)
    d_eer = base_te["eer"] - tm["eer"]
    print(f"  --> EER change {d_eer*100:+.2f} pp  ({'HEAD WINS' if d_eer > 0 else 'raw wins / tie'})",
          flush=True)
    if wb is not None:
        wb.log({"test_raw_eer": base_te["eer"], "test_head_eer": tm["eer"],
                "test_raw_rank1": base_te["rank1"], "test_head_rank1": tm["rank1"]})
        wb.finish()
    otag = f"_{args.tag}" if args.tag else ""
    out = ARTIFACTS / f"inv_{args.model}{otag}.pt"
    import pandas as pd
    torch.save(dict(state=best["state"], d_id=args.d_id, d_fw=args.d_fw, embed_dim=D,
                    model=args.model, tag=args.tag, best_epoch=best["epoch"],
                    test_raw=base_te, test_head=tm), out)
    pd.DataFrame([{"which": "raw", **base_te}, {"which": "head", **tm}]).to_parquet(
        ARTIFACTS / f"inv_{args.model}{otag}.parquet", index=False)
    print(f"\nsaved head -> {out.name}   metrics -> inv_{args.model}{otag}.parquet", flush=True)
    if args.hf_repo:
        from stepup.hf import push_files
        push_files(args.hf_repo, [out, ARTIFACTS / f"inv_{args.model}{otag}.parquet"], args.hf_token)
        if args.hf_offload:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
