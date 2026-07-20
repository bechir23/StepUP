"""Backbones (all small-input-corrected), the BNNeck embedder, MixStyle, and the registry.

Correction rule for the 75x40 pressure map (docs/BACKBONE_CATALOG.md): never a 7x7-s2 +
maxpool stem; use a 3x3 s1 stem and last_stride=1 (OpenGait/GaitBase; ReID bag of tricks).
Transformers are re-patched to tile the small grid. Every model wraps a shared BNNeck head.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torchvision.models import resnet18

from .config import T, H, W

_DROPOUT = 0.0


def set_dropout(p):
    global _DROPOUT
    _DROPOUT = p


class MixStyle(nn.Module):
    """Mix instance-level feature statistics across a batch (Zhou et al. 2021), 5D-adapted."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        dims = [2, 3, 4] if x.dim() == 5 else [2, 3]
        mu = x.mean(dim=dims, keepdim=True)
        sig = (x.var(dim=dims, keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        xn = (x - mu) / sig
        lmda = self.beta.sample((x.size(0), *([1] * (x.dim() - 1)))).to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        return xn * (sig * lmda + sig[perm] * (1 - lmda)) + (mu * lmda + mu[perm] * (1 - lmda))


class Embedder(nn.Module):
    """Backbone (-> feat_dim) + BNNeck head: f_t (pre-BN, triplet), f_i (post-BN, ID/cosine)."""

    def __init__(self, backbone, feat_dim, embed_dim=128, n_classes=None):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(_DROPOUT)
        self.embed = nn.Linear(feat_dim, embed_dim)
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, n_classes, bias=False) if n_classes else None

    def forward(self, x):
        h = self.drop(self.backbone(x))
        f_t = self.embed(h)
        f_i = self.bnneck(f_t)
        logits = self.classifier(f_i) if self.classifier is not None else None
        return f_t, f_i, logits


class TimeAsChannels(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


class Resize3D(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode="trilinear", align_corners=False)


def _wrap3d(net, resize, embed_dim, n_classes, feat_dim=512):
    backbone = nn.Sequential(Resize3D(resize), net) if resize else net
    return Embedder(backbone, feat_dim=feat_dim, embed_dim=embed_dim, n_classes=n_classes)


# --------------------------------------------------------------------- 2D
def make_resnet2d(embed_dim=128, n_classes=None, in_frames=T):
    """ResNet-18 with a gait-style small-input stem: 3x3 s1, no maxpool, last_stride=1
    (OpenGait/GaitBase; ReID bag of tricks). Keeps 75x40 -> ~19x10 instead of 3x2."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(in_frames, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.layer4[0].conv1.stride = (1, 1)
    net.layer4[0].downsample[0].stride = (1, 1)
    net.fc = nn.Identity()
    return Embedder(nn.Sequential(TimeAsChannels(), net), feat_dim=512,
                    embed_dim=embed_dim, n_classes=n_classes)


class GaitCNN(nn.Module):
    """Compact 2D-CNN for the 75x40 map (reference plantar-pressure shape): 4 gentle blocks."""

    def __init__(self, in_frames=T, widths=(64, 128, 256, 256)):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        c0, c1, c2, c3 = widths
        self.features = nn.Sequential(
            blk(in_frames, c0), nn.MaxPool2d(2), blk(c0, c1), nn.MaxPool2d(2),
            blk(c1, c2), nn.MaxPool2d(2), blk(c2, c3), nn.AdaptiveAvgPool2d(1))
        self.out_dim = c3

    def forward(self, x):
        return self.features(x.squeeze(1)).flatten(1)


def make_gaitcnn(embed_dim=128, n_classes=None, in_frames=T):
    net = GaitCNN(in_frames=in_frames)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


class ResBlock2d(nn.Module):
    """Standard 3x3 residual block; the skip keeps gradients healthy as depth grows."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(cout)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(out)) + self.short(x), inplace=True)


class GaitCNNDeep(nn.Module):
    """Deeper residual 2D-CNN over the 75x40 map (8 res-blocks, widths to 512): more capacity
    than the 4-block gaitcnn, with residual skips. Only helps if regularized (dropout + aug) and
    trained at a low LR -- otherwise the extra capacity just memorizes faster."""

    def __init__(self, in_frames=T, widths=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = widths
        self.stem = nn.Sequential(nn.Conv2d(in_frames, c0, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(c0), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock2d(c0, c0), ResBlock2d(c0, c1, 2),      # 75x40 -> 38x20
            ResBlock2d(c1, c1), ResBlock2d(c1, c2, 2),      # -> 19x10
            ResBlock2d(c2, c2), ResBlock2d(c2, c3, 2),      # -> 10x5
            ResBlock2d(c3, c3), nn.AdaptiveAvgPool2d(1))
        self.out_dim = c3

    def forward(self, x):
        return self.layers(self.stem(x.squeeze(1))).flatten(1)


def make_gaitcnn_deep(embed_dim=128, n_classes=None, in_frames=T):
    net = GaitCNNDeep(in_frames=in_frames)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


class CNNLSTM(nn.Module):
    """LRCN (Donahue et al. 2015): ResNet-18 frame encoder + LSTM over subsampled frames."""

    def __init__(self, n_frames=32, lstm_dim=256):
        super().__init__()
        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        enc.fc = nn.Identity()
        self.enc = enc
        self.lstm = nn.LSTM(512, lstm_dim, batch_first=True)
        self.n_frames, self.out_dim = n_frames, lstm_dim

    def forward(self, x):
        b, _, t, h, w = x.shape
        idx = torch.linspace(0, t - 1, self.n_frames, device=x.device).long()
        frames = x[:, :, idx].transpose(1, 2).reshape(b * self.n_frames, 1, h, w)
        seq = self.enc(frames).reshape(b, self.n_frames, 512)
        _, (hn, _) = self.lstm(seq)
        return hn[-1]


def make_cnnlstm(embed_dim=128, n_classes=None, n_frames=32, lstm_dim=256):
    net = CNNLSTM(n_frames=n_frames, lstm_dim=lstm_dim)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


# --------------------------------------------------------------------- 3D (torchvision, corrected)
def _to_single_channel(net):
    """Adapt a torchvision video ResNet to 1-channel input: swap the stem's first conv to one
    input channel and keep the network's standard spatiotemporal downsampling. The 3D nets are
    fed a resized ~16-frame Kinetics-style clip (see registry), so the original strides are
    correct and no stride surgery is needed -- the stride-1 surgery on a 101-frame full cube is
    exactly what OOMs a 40GB GPU (a single conv3d activation reaches ~18GB)."""
    s0 = net.stem[0]
    net.stem[0] = nn.Conv3d(1, s0.out_channels, s0.kernel_size, s0.stride, s0.padding, bias=False)
    net.fc = nn.Identity()
    return net


class MixStyleVideo(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.mix = MixStyle()

    def forward(self, x):
        n = self.net
        x = n.stem(x)
        x = self.mix(n.layer1(x))
        x = self.mix(n.layer2(x))
        x = n.layer4(n.layer3(x))
        return torch.flatten(n.avgpool(x), 1)


def make_r2plus1d(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    from torchvision.models.video import r2plus1d_18
    net = _to_single_channel(r2plus1d_18(weights=None))
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 512)


def make_r3d(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    from torchvision.models.video import r3d_18
    net = _to_single_channel(r3d_18(weights=None))
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 512)


def make_swin3d(embed_dim=128, n_classes=None, resize=None):
    """Video Swin re-parameterised for the small map: patch (2,2,2), window (8,4,4)."""
    from torchvision.models.video.swin_transformer import SwinTransformer3d
    net = SwinTransformer3d(patch_size=[2, 2, 2], embed_dim=96, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=[8, 4, 4], num_classes=1)
    pe = net.patch_embed.proj
    net.patch_embed.proj = nn.Conv3d(1, pe.out_channels, pe.kernel_size, pe.stride, pe.padding)
    net.head = nn.Identity()
    return _wrap3d(net, resize, embed_dim, n_classes, feat_dim=768)


class VideoViT(nn.Module):
    """Tubelet ViViT/VideoMAE encoder from scratch; patch sized to tile the small grid."""

    def __init__(self, in_ch=1, dim=512, depth=8, heads=8, patch=(4, 8, 8), input_size=(T, H, W)):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_ch, dim, patch, patch)
        with torch.no_grad():
            n = self.patch_embed(torch.zeros(1, in_ch, *input_size)).flatten(2).shape[-1]
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.out_dim, self.n_tokens = dim, n

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.enc(x + self.pos[:, :x.size(1)])
        return self.norm(x).mean(1)


def make_vit(embed_dim=128, n_classes=None, resize=None):
    net = VideoViT(in_ch=1, input_size=resize or (T, H, W))
    return _wrap3d(net, resize, embed_dim, n_classes, feat_dim=net.out_dim)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def registry(sample3d, data_t=T):
    """Build the model registry for a given 3D input size and 2D frame count.
    full_pk is the per-model (P, K): the light 2D/recurrent nets take a big batch P=128 (512),
    the 3D/transformer nets a smaller P=8 (32). Override per run with --P/--K.

    3D video nets cannot consume the full 101-frame cube -- R(2+1)D/R3D were designed for short
    ~16-frame clips (Tran et al. 2018, "A Closer Look at Spatiotemporal Convolutions"), and 3D
    convs over 101x75x40 OOM even a 40GB A100. They are therefore fed a resized Kinetics-style
    clip (<=16 frames, <=64 spatial); the 2D nets still see the full-resolution cube."""
    pk2d, pk3d = (128, 4), (8, 4)
    clip3d = (min(16, sample3d[0]), min(64, sample3d[1]), min(64, sample3d[2]))
    return {
        "gaitcnn":  dict(fn=make_gaitcnn, kw=dict(in_frames=data_t), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_deep": dict(fn=make_gaitcnn_deep, kw=dict(in_frames=data_t), full_pk=pk2d,
                             smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "resnet2d": dict(fn=make_resnet2d, kw=dict(in_frames=data_t), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "cnnlstm":  dict(fn=make_cnnlstm, kw=dict(n_frames=min(32, data_t)), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(n_frames=min(16, data_t))),
        "r2plus1d": dict(fn=make_r2plus1d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "r3d":      dict(fn=make_r3d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "swin3d":   dict(fn=make_swin3d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "vit":      dict(fn=make_vit, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
    }
