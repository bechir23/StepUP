"""Triplet mining and the training criterion.

Criterion = BNNeck bag-of-tricks: an ID loss on the post-BN feature (label-smoothed cross
entropy by default, starts ~5; SubCenter-ArcFace s=32 optional) plus a triplet loss on the
pre-BN feature (the competition winner's loss). Mining is 'standard' batch-hard or 'crossfw'
cross-footwear positive mining (the footwear-invariance contribution).
"""
import torch
import torch.nn as nn
from pytorch_metric_learning import losses


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
        self.kind = cfg.get("loss", "ce")
        if self.kind == "arcface":
            self.arc = losses.SubCenterArcFaceLoss(num_classes=n_ids, embedding_size=embed_dim,
                                                   sub_centers=3, scale=cfg.get("arc_scale", 32))
        else:
            self.id_head = nn.Linear(embed_dim, n_ids)
            self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.triplet = losses.TripletMarginLoss(margin=0.3)

    def forward(self, f_t, f_i, yb, mined):
        l_id = self.arc(f_i, yb) if self.kind == "arcface" else self.ce(self.id_head(f_i), yb)
        l_tri = self.triplet(f_t, yb, mined)
        return l_id + l_tri, float(l_id), float(l_tri)
