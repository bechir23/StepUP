"""Triplet mining and the training criterion.

Criterion = BNNeck bag-of-tricks: an ID loss on the post-BN feature (label-smoothed cross
entropy by default, starts ~5; SubCenter-ArcFace s=32 optional) plus a triplet loss on the
pre-BN feature (the competition winner's loss). Mining is 'standard' batch-hard or 'crossfw'
cross-footwear positive mining (the footwear-invariance contribution).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

from .arcface import ArcFace


def _stack(a, p, n, device):
    if not a:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty, empty
    return (torch.tensor(a, device=device), torch.stack(p).to(device), torch.stack(n).to(device))


def batchhard_triplets(emb, labels, fw=None):
    """Hardest positive = any same-identity step; hardest negative = nearest other identity."""
    d = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos, neg = same & ~eye, ~same
    a, p, n = [], [], []
    for i in range(len(labels)):
        if pos[i].any() and neg[i].any():
            a.append(i)
            p.append(torch.where(pos[i], d[i], d.new_tensor(-1.0)).argmax())
            n.append(torch.where(neg[i], d[i], d.new_tensor(float("inf"))).argmin())
    return _stack(a, p, n, emb.device)


def crossfootwear_triplets(emb, labels, fw):
    """Hardest positive = same identity, DIFFERENT shoe; negative prefers the anchor's shoe."""
    d = torch.cdist(emb, emb)
    same_id = labels[:, None] == labels[None, :]
    same_fw = fw[:, None] == fw[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos = same_id & ~same_fw & ~eye
    neg_same_fw = (~same_id) & same_fw
    neg_any = ~same_id
    a, p, n = [], [], []
    for i in range(len(labels)):
        if not pos[i].any():
            continue
        neg_mask = neg_same_fw[i] if neg_same_fw[i].any() else neg_any[i]
        if not neg_mask.any():
            continue
        a.append(i)
        p.append(torch.where(pos[i], d[i], d.new_tensor(-1.0)).argmax())
        n.append(torch.where(neg_mask, d[i], d.new_tensor(float("inf"))).argmin())
    return _stack(a, p, n, emb.device)


MINERS = {"standard": batchhard_triplets, "crossfw": crossfootwear_triplets}


class Criterion(nn.Module):
    """ID loss (CE or ArcFace) on f_i + triplet on f_t. Its parameters join the optimizer."""

    def __init__(self, cfg, n_ids, embed_dim):
        super().__init__()
        self.kind = cfg.get("loss", "arcface")
        if self.kind == "arcface":
            # single-center ArcFace on the L2-normalized embedding, no triplet -- the reference
            # recipe, verified to beat SubCenter+triplet on clean identities. Margin ramps 0->m.
            self.arc = ArcFace(embed_dim, n_ids, s=cfg.get("arc_scale", 16.0), m=0.5)
            self._m_target = 0.5
            self.arc.set_margin(0.0)
            self.ce = nn.CrossEntropyLoss()
        elif self.kind == "triplet":
            # pure online-mined triplet loss on the pre-BN feature -- the loss the StepUP
            # competition WINNER used (R(2+1)D + triplet + aug). Mining is handled in the engine.
            self.triplet = losses.TripletMarginLoss(margin=cfg.get("triplet_margin", 0.3))
        elif self.kind == "supcon":
            # supervised contrastive (Khosla et al. 2020), the CodaBench StepUP baseline objective:
            # pulls same-identity embeddings together, pushes others apart, directly on the unit
            # sphere. No classifier head, no margin -- an alternative to ArcFace to compare against.
            self.supcon = losses.SupConLoss(temperature=cfg.get("supcon_temp", 0.1))
        else:
            # BNNeck "bag of tricks": label-smoothed CE ID head + triplet (kept for the ce ablation)
            self.id_head = nn.Linear(embed_dim, n_ids)
            self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.triplet = losses.TripletMarginLoss(margin=0.3)

    def set_margin_frac(self, frac):
        """ArcFace only: ramp the angular margin 0 -> target (called each epoch during warmup)."""
        if self.kind == "arcface":
            self.arc.set_margin(self._m_target * max(0.0, min(1.0, frac)))

    def forward(self, f_t, f_i, yb, mined):
        if self.kind == "arcface":
            l_id = self.ce(self.arc(F.normalize(f_i), yb), yb)   # ArcFace on L2-normed embedding
            return l_id, l_id.item(), 0.0                         # no triplet (matches reference)
        if self.kind == "triplet":
            l = self.triplet(f_t, yb, mined)                      # pure triplet (competition winner)
            return l, 0.0, l.item()
        if self.kind == "supcon":
            l = self.supcon(F.normalize(f_i), yb)                 # supervised contrastive
            return l, l.item(), 0.0
        l_id = self.ce(self.id_head(f_i), yb)
        l_tri = self.triplet(f_t, yb, mined)
        return l_id + l_tri, l_id.item(), l_tri.item()
