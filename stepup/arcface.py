"""Single-center ArcFace head (additive angular margin), the standard open-set metric-learning
loss (Deng et al., "ArcFace", CVPR 2019). Used only at train time on top of the L2-normalized
embedding, then discarded -- the deployed model is just the embedder.

Why this and not SubCenter+triplet: on a small set of clean identities the sub-center variant
(3 centers/identity) fragments each identity and an added triplet term fights the angular
objective. A controlled head-to-head on the same 15 identities / same full-res data showed this
plain single-center head reaching mixed-gallery rank-1 ~0.69 vs ~0.55 for SubCenter+triplet, and
it starts at the healthy CE floor ln(n_ids) (~2.7 for 15) rather than an inflated ~6. Matches the
reference StepUP implementation (s=16, m=0.5 rad, angular-margin warm-up 0->m over a few epochs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, embed_dim, num_classes, s=16.0, m=0.5, easy_margin=False):
        super().__init__()
        self.s, self.easy_margin = s, easy_margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)
        self.set_margin(m)

    def set_margin(self, m):
        """Set the angular margin (radians). Called during warm-up to ramp 0 -> target."""
        self.m = m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)              # cos(pi - m)
        self.mm = math.sin(math.pi - m) * m          # fallback for the far side

    def forward(self, embeddings, labels):
        # embeddings are L2-normalized by the caller; normalize the class weights too.
        w = F.normalize(self.weight, dim=1)
        cosine = (embeddings @ w.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * self.cos_m - sine * self.sin_m               # cos(theta + m)
        phi = torch.where(cosine > 0, phi, cosine) if self.easy_margin \
            else torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=w.size(0)).to(cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s   # scaled logits -> feed to CE
