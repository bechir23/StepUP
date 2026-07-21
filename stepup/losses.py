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
from .config import FOOTWEAR


class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015): identity forward, negated-and-scaled
    gradient backward. Placed before a footwear discriminator, it turns 'classify the shoe' into
    'make the shoe unclassifiable from the embedding' for the backbone -- domain-adversarial
    footwear invariance, applied directly on the eval feature f_i (unlike the triplet on f_t)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


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


def crossfw_center_align(f_i, labels, fw):
    """Cross-footwear center alignment on the EVAL feature f_i: for each identity present with >=2
    shoes in the batch, pull its per-shoe centroids toward the identity's global centroid. Unlike a
    triplet (which only needs the hardest pair to satisfy a margin) or the adversarial head (whose
    discriminator never engaged), this directly and unconditionally compresses a person's
    different-shoe embeddings together in the space the metric is measured in -- the most direct
    footwear-invariance objective. Needs --mining crossfw so each id spans shoes in the batch."""
    f = F.normalize(f_i)
    total, n = f.new_tensor(0.0), 0
    for uid in labels.unique():
        m = labels == uid
        shoes = fw[m].unique()
        if len(shoes) < 2:
            continue
        cents = torch.stack([f[m & (fw == s)].mean(0) for s in shoes])   # per-shoe centroids
        gc = cents.mean(0, keepdim=True)                                 # identity centroid
        total = total + ((cents - gc) ** 2).sum(1).mean()               # spread across shoes
        n += 1
    return total / max(n, 1)


class Criterion(nn.Module):
    """ID loss (CE or ArcFace) on f_i + triplet on f_t. Its parameters join the optimizer."""

    def __init__(self, cfg, n_ids, embed_dim):
        super().__init__()
        self.kind = cfg.get("loss", "arcface")
        if self.kind in ("arcface", "arcfw", "arcadv", "arccal"):
            # single-center ArcFace on the L2-normalized embedding. 'arcface' = identity only (the
            # clean-identity recipe). 'arcfw' additionally applies a triplet on f_t over the MINED
            # pairs -- with --mining crossfw those pairs are same-identity/different-shoe, so this
            # term is a footwear-invariance objective (pull a person's steps together ACROSS shoes).
            # 'arcadv' instead adds a gradient-reversed footwear discriminator on f_i, which removes
            # footwear information from the eval embedding directly. ArcFace alone never sees
            # footwear, leaving cross-footwear (unseen) EER far above same-shoe (seen) EER.
            self.arc = ArcFace(embed_dim, n_ids, s=cfg.get("arc_scale", 16.0), m=0.5)
            self._m_target = 0.5
            self.arc.set_margin(0.0)
            # Label smoothing gives the ID loss a FINITE optimum. Plain softmax/ArcFace keeps
            # pushing logits apart forever, so once the training identities are perfectly
            # separated it goes on widening margins on already-solved data -- large gradients,
            # zero transfer, and validation drifts down. Smoothing stops the push once reached,
            # which is what removes the peak-then-decay shape.
            self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smooth", 0.0))
            if self.kind == "arcfw":
                self.triplet = losses.TripletMarginLoss(margin=cfg.get("triplet_margin", 0.3))
                self.fw_w = cfg.get("fw_triplet_weight", 1.0)
            if self.kind == "arcadv":
                nfw = len(FOOTWEAR)
                self.fw_disc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                             nn.Linear(embed_dim, nfw))
                self.fw_ce = nn.CrossEntropyLoss()
                self.adv_w = cfg.get("adv_weight", 1.0)          # gradient-reversal strength lambda
            if self.kind == "arccal":
                self.cal_w = cfg.get("cal_weight", 0.5)          # cross-footwear center-align weight
        elif self.kind == "triplet":
            # pure online-mined triplet loss on the pre-BN feature -- the loss the StepUP
            # competition WINNER used (R(2+1)D + triplet + aug). Mining is handled in the engine.
            self.triplet = losses.TripletMarginLoss(margin=cfg.get("triplet_margin", 0.3))
        elif self.kind == "circle":
            # CircleLoss (Sun et al. CVPR 2020) with the organisers' baseline settings
            # (m=0.25, gamma=256): re-weights each similarity by how far it is from its optimum,
            # which handles the wide within-identity spread that different footwear creates
            # better than a single fixed angular margin.
            self.circle = losses.CircleLoss(m=cfg.get("circle_m", 0.25),
                                            gamma=cfg.get("circle_gamma", 256))
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
        if self.kind in ("arcface", "arcfw", "arcadv", "arccal"):
            self.arc.set_margin(self._m_target * max(0.0, min(1.0, frac)))

    def forward(self, f_t, f_i, yb, mined, fwb=None):
        if self.kind == "arcface":
            l_id = self.ce(self.arc(F.normalize(f_i), yb), yb)   # ArcFace on L2-normed embedding
            return l_id, l_id.item(), 0.0                         # no triplet (matches reference)
        if self.kind == "arcfw":
            l_id = self.ce(self.arc(F.normalize(f_i), yb), yb)   # identity (angular margin)
            l_tri = self.triplet(f_t, yb, mined)                 # cross-footwear pull (invariance)
            return l_id + self.fw_w * l_tri, l_id.item(), float(l_tri.item())
        if self.kind == "arcadv":
            l_id = self.ce(self.arc(F.normalize(f_i), yb), yb)   # identity (angular margin)
            logits_fw = self.fw_disc(grad_reverse(F.normalize(f_i), self.adv_w))
            l_adv = self.fw_ce(logits_fw, fwb.long())            # discriminator learns; backbone unlearns
            return l_id + l_adv, l_id.item(), float(l_adv.item())
        if self.kind == "arccal":
            l_id = self.ce(self.arc(F.normalize(f_i), yb), yb)   # identity (angular margin)
            l_cal = crossfw_center_align(f_i, yb, fwb)           # compress per-id cross-shoe spread
            return l_id + self.cal_w * l_cal, l_id.item(), float(l_cal.item())
        if self.kind == "triplet":
            l = self.triplet(f_t, yb, mined)                      # pure triplet (competition winner)
            return l, 0.0, l.item()
        if self.kind == "circle":
            l = self.circle(F.normalize(f_i), yb)                 # CircleLoss (baseline recipe)
            return l, l.item(), 0.0
        if self.kind == "supcon":
            l = self.supcon(F.normalize(f_i), yb)                 # supervised contrastive
            return l, l.item(), 0.0
        l_id = self.ce(self.id_head(f_i), yb)
        l_tri = self.triplet(f_t, yb, mined)
        return l_id + l_tri, l_id.item(), l_tri.item()
