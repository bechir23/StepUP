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


class DSU(nn.Module):
    """Domain Shift with Uncertainty (Li et al., ICLR 2022). Model each channel's instance
    mean/std as a Gaussian whose *scope* is the batch std of those statistics, resample new
    stats via the reparameterization trick, and re-apply AdaIN-style. Footwear shifts these
    per-channel statistics like 'style', so perturbing them during training makes the embedding
    footwear-robust WITHOUT deleting identity -- and it is trained INTO the backbone (a frozen-
    embedding head cannot replicate it). Parameter-free; off at eval, so it adds no state_dict keys."""

    def __init__(self, p=0.5, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        dims = [2, 3, 4] if x.dim() == 5 else [2, 3]
        mu = x.mean(dim=dims, keepdim=True)                     # (B,C,1,1) per-sample channel stats
        sig = (x.var(dim=dims, keepdim=True) + self.eps).sqrt()
        mu_std = mu.std(dim=0, keepdim=True) + self.eps         # (1,C,1,1) uncertainty scope
        sig_std = sig.std(dim=0, keepdim=True) + self.eps       # = batch std of the statistics
        beta = mu + torch.randn_like(mu) * mu_std               # resampled mean   (reparam)
        gamma = sig + torch.randn_like(sig) * sig_std           # resampled std
        xn = (x - mu.detach()) / sig.detach()
        return gamma * xn + beta


class HPP(nn.Module):
    """Horizontal Pyramid Pooling (Fu et al. 2018; GaitSet, GaitPart, OpenGait -- and Salehi's
    FootPart for THIS dataset). Split the feature map into horizontal strips (foot heel->arch->toe
    regions) at several scales and pool each strip separately (avg+max), instead of one global
    pool. Footwear changes some regions (arch/heel loading under a stiff sole) far more than others
    (the toe/forefoot pressure signature, which is identity-specific and footwear-stable); a global
    pool blends them, part-based pooling keeps the invariant regions able to carry identity. This is
    the standard covariate-invariance lever in gait recognition ('clothing-invariant gait via
    part-based features'). Output dim = C * sum(scales)."""

    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, x):                                  # (B, C, H, W)
        outs = []
        for s in self.scales:
            a = F.adaptive_avg_pool2d(x, (s, 1))           # split the foot-length axis into s strips
            m = F.adaptive_max_pool2d(x, (s, 1))
            outs.append((a + m).flatten(1))                # (B, C*s)   avg+max, GaitSet-style
        return torch.cat(outs, 1)                          # (B, C*sum(scales))


class SNR(nn.Module):
    """Style Normalisation and Restitution (Jin et al., CVPR 2020).

    Plain InstanceNorm removes footwear style but takes identity signal with it -- measured
    here: IN lifted cross-footwear EER 30.7 -> 29.2 while mixed-gallery rank-1 fell
    0.644 -> 0.536. SNR fixes exactly that: normalise the style away, then look at what was
    removed (R = x - IN(x)) and add back only the identity-relevant channels, selected by a
    small squeeze-excite gate. Style is discarded, discriminative content is restituted."""

    def __init__(self, c, r=8):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c, affine=True)
        hidden = max(4, c // r)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, hidden, 1),
                                  nn.ReLU(inplace=True), nn.Conv2d(hidden, c, 1), nn.Sigmoid())

    def forward(self, x):
        xn = self.norm(x)              # style-normalised
        r = x - xn                     # what IN threw away (style + some identity)
        return xn + self.gate(r) * r   # restitute the identity-relevant part only


class SNR3d(nn.Module):
    """SNR for 5D (video) features -- same idea as SNR, with InstanceNorm3d and a 3D gate, so the
    video backbones can carry the same footwear-style removal as the 2D ones."""

    def __init__(self, c, r=8):
        super().__init__()
        self.norm = nn.InstanceNorm3d(c, affine=True)
        hidden = max(4, c // r)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(c, hidden, 1),
                                  nn.ReLU(inplace=True), nn.Conv3d(hidden, c, 1), nn.Sigmoid())

    def forward(self, x):
        xn = self.norm(x)
        r = x - xn
        return xn + self.gate(r) * r


def _add_snr(net, chans=(64, 128), three_d=False):
    """Insert SNR after the first two residual stages of a torchvision-style backbone.

    Early stages only: those carry the per-sample statistics that encode footwear, while the last
    stage carries identity and is left untouched (normalising there costs discrimination)."""
    mod = SNR3d if three_d else SNR
    net.layer1 = nn.Sequential(net.layer1, mod(chans[0]))
    net.layer2 = nn.Sequential(net.layer2, mod(chans[1]))
    return net


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


def make_resnet2d_light(embed_dim=128, n_classes=None, in_frames=T):
    """Lighter ResNet-2D: ResNet-18 with the last (512-ch) stage dropped -> ~2.8M params, feat 256."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(in_frames, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.layer4 = nn.Identity()                       # drop the deepest stage; layer3 out = 256
    net.fc = nn.Identity()
    return Embedder(nn.Sequential(TimeAsChannels(), net), feat_dim=256,
                    embed_dim=embed_dim, n_classes=n_classes)


class GaitCNN(nn.Module):
    """Compact 2D-CNN for the 75x40 map (reference plantar-pressure shape): 4 gentle blocks.
    With mixstyle=True, MixStyle is inserted after the first two blocks (early stages only -- never
    the last, which carries identity): footwear changes the per-channel feature STATISTICS ('style')
    of the pressure map more than its structure, so mixing instance stats across the batch during
    training makes the embedding footwear-invariant. Off at eval. (Zhou et al. 2021.)"""

    def __init__(self, in_frames=T, widths=(64, 128, 256, 256), mixstyle=False, dsu=False,
                 hpp=False, norm="bn"):
        super().__init__()
        # norm="in" swaps BatchNorm for InstanceNorm. BN normalises with batch statistics and so
        # PRESERVES each sample's own style; IN normalises per-sample per-channel and therefore
        # REMOVES it. Footwear is carried largely in those per-sample statistics (a stiff sole
        # spreads load, barefoot concentrates it), so IN is a footwear-invariance mechanism built
        # into the architecture -- this is what the organisers' own baseline uses (InstanceNorm3d).
        def nrm(c):
            return nn.InstanceNorm2d(c, affine=True) if norm == "in" else nn.BatchNorm2d(c)

        def blk(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nrm(cout), nn.ReLU(inplace=True))
        c0, c1, c2, c3 = widths
        ms1 = MixStyle() if mixstyle else nn.Identity()      # after stage 1
        ms2 = MixStyle() if mixstyle else nn.Identity()      # after stage 2
        # norm="snr": keep BatchNorm inside the blocks (so identity discrimination is intact)
        # and insert SNR after the early/mid stages, where style lives.
        s1 = SNR(c0) if norm == "snr" else nn.Identity()
        s2 = SNR(c1) if norm == "snr" else nn.Identity()
        s3 = SNR(c2) if norm == "snr" else nn.Identity()
        hpp_scales = (1, 2, 4)
        pool = HPP(hpp_scales) if hpp else nn.AdaptiveAvgPool2d(1)   # part-based vs global pooling
        # Build the stack as a list; DSU is only INSERTED when --dsu (so with dsu=False the module
        # indices stay identical to the original -> pre-existing checkpoints still load).
        feat = [blk(in_frames, c0), s1, ms1]
        if dsu:
            feat.append(DSU())
        feat += [nn.MaxPool2d(2), blk(c0, c1), s2, ms2]
        if dsu:
            feat.append(DSU())
        feat += [nn.MaxPool2d(2), blk(c1, c2), s3, nn.MaxPool2d(2), blk(c2, c3), pool]
        self.features = nn.Sequential(*feat)
        self.out_dim = c3 * sum(hpp_scales) if hpp else c3

    def forward(self, x):
        return self.features(x.squeeze(1)).flatten(1)


def make_gaitcnn(embed_dim=128, n_classes=None, in_frames=T, mixstyle=False, dsu=False, hpp=False):
    net = GaitCNN(in_frames=in_frames, mixstyle=mixstyle, dsu=dsu, hpp=hpp)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


def make_gaitcnn_in(embed_dim=128, n_classes=None, in_frames=T, mixstyle=False):
    """GaitCNN with InstanceNorm instead of BatchNorm -- architecture-level footwear-style removal,
    matching the organisers' baseline (which uses InstanceNorm3d in its SpatioTemporalConv)."""
    net = GaitCNN(in_frames=in_frames, mixstyle=mixstyle, norm="in")
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


def make_gaitcnn_snr(embed_dim=128, n_classes=None, in_frames=T, mixstyle=False, dsu=False, hpp=False):
    """GaitCNN + SNR modules: InstanceNorm-based style removal with identity restitution.
    Targets the measured IN trade-off (better cross-footwear EER, worse rank-1). --dsu adds
    Domain-Shift-with-Uncertainty statistic perturbation on top of SNR (both attack footwear style).
    --hpp swaps global pooling for Horizontal Pyramid (part-based) pooling of the foot regions."""
    net = GaitCNN(in_frames=in_frames, mixstyle=mixstyle, dsu=dsu, hpp=hpp, norm="snr")
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


def make_gaitcnn_tiny(embed_dim=128, n_classes=None, in_frames=T, mixstyle=False):
    """Low-capacity GaitCNN (~16x fewer conv params). The full model memorises the training
    identities within 2-3 epochs (train rank-1 -> 1.00, train cross-footwear EER -> ~1%), after
    which the loss is satisfied, gradients vanish and validation stops moving. Shrinking capacity
    keeps the training task non-trivial for longer, forcing features that transfer to new people
    instead of an identity lookup table."""
    net = GaitCNN(in_frames=in_frames, widths=(16, 32, 64, 64), mixstyle=mixstyle)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


class R21DBlock(nn.Module):
    """Factorised (2+1)D convolution (Tran et al., CVPR 2018, 'A Closer Look at Spatiotemporal
    Convolutions'): a 2D spatial conv (1,3,3) over the pressure map, then a 1D temporal conv (3,1,1)
    over the gait cycle, each BN+ReLU. Learns the spatial and temporal filters SEPARATELY -- models
    footstep dynamics (which the time-as-channels GaitCNN discards) at ~1/3 the FLOPs of a full 3D
    (3,3,3) conv, and the extra ReLU between them adds nonlinearity a single 3D conv lacks."""

    def __init__(self, cin, cout, mid=None):
        super().__init__()
        mid = mid or cout
        self.spatial = nn.Conv3d(cin, mid, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        self.temporal = nn.Conv3d(mid, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.spatial(x)))
        return self.act(self.bn2(self.temporal(x)))


class GaitR21D(nn.Module):
    """Lightweight (2+1)D backbone for the pressure cube (B,1,T,H,W): 4 factorised spatiotemporal
    blocks that keep the gait DYNAMICS the time-as-channels GaitCNN throws away, far cheaper than a
    full-3D net (which needs a downsized clip and OOMs on the full cube). Same footwear-invariance
    hooks as GaitCNN: norm='snr' inserts SNR3d after the early stages; mixstyle/dsu perturb per-sample
    feature statistics (5D-aware); hpp does part-based heel->toe pooling after a temporal collapse."""

    def __init__(self, widths=(32, 64, 128, 128), mixstyle=False, dsu=False, hpp=False, norm="snr"):
        super().__init__()
        c0, c1, c2, c3 = widths
        ms1 = MixStyle() if mixstyle else nn.Identity()
        ms2 = MixStyle() if mixstyle else nn.Identity()
        s1 = SNR3d(c0) if norm == "snr" else nn.Identity()   # style removal after early stages only
        s2 = SNR3d(c1) if norm == "snr" else nn.Identity()
        s3 = SNR3d(c2) if norm == "snr" else nn.Identity()
        sp = lambda: nn.MaxPool3d((1, 2, 2), ceil_mode=True)   # spatial-only downsample
        spt = lambda: nn.MaxPool3d((2, 2, 2), ceil_mode=True)  # spatial + temporal downsample
        feat = [R21DBlock(1, c0), s1, ms1]                     # input channel = 1 (single pressure map)
        if dsu:
            feat.append(DSU())
        feat += [sp(), R21DBlock(c0, c1), s2, ms2]
        if dsu:
            feat.append(DSU())
        feat += [spt(), R21DBlock(c1, c2), s3, spt(), R21DBlock(c2, c3)]
        self.features = nn.Sequential(*feat)
        self.hpp, self.scales = hpp, (1, 2, 4)
        self.pool = HPP(self.scales) if hpp else None
        self.out_dim = c3 * sum(self.scales) if hpp else c3

    def forward(self, x):                                      # (B,1,T,H,W)
        h = self.features(x)                                   # (B,c3,T',H',W')
        if self.hpp:
            return self.pool(h.mean(2))                        # temporal-pool -> part-based spatial pool
        return F.adaptive_avg_pool3d(h, 1).flatten(1)          # global (T,H,W) pool -> (B,c3)


def make_gaitcnn_r21d(embed_dim=128, n_classes=None, in_frames=T, mixstyle=False, dsu=False, hpp=False):
    """Factorised (2+1)D GaitCNN + SNR: spatial-2D-then-temporal-1D convs model the footstep dynamics
    the time-as-channels GaitCNN drops, at ~1/3 the cost of full 3D. --mixstyle/--dsu/--hpp behave as
    in gaitcnn_snr. Consumes the same (B,1,T,H,W) pressure pack as the 2D nets (no clip resize)."""
    net = GaitR21D(mixstyle=mixstyle, dsu=dsu, hpp=hpp, norm="snr")   # in_frames unused (T is dynamic)
    return Embedder(net, feat_dim=net.out_dim, embed_dim=embed_dim, n_classes=n_classes)


def make_resnet2d_snr(embed_dim=128, n_classes=None, in_frames=T):
    """ResNet-2D with SNR after stages 1-2 -- the same footwear-style removal as gaitcnn_snr, so
    the backbones can be compared with the invariance mechanism held constant."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(in_frames, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.layer4[0].conv1.stride = (1, 1)
    net.layer4[0].downsample[0].stride = (1, 1)
    net.fc = nn.Identity()
    net = _add_snr(net, chans=(64, 128))
    return Embedder(nn.Sequential(TimeAsChannels(), net), feat_dim=512,
                    embed_dim=embed_dim, n_classes=n_classes)


def make_convnext(embed_dim=128, n_classes=None, in_frames=T):
    """ConvNeXt-Tiny (2D, frames-as-channels) with a small-input stem (stride-2 patchify instead
    of 4) so the 75x40 map isn't over-downsampled. One of the architectures the top teams explored;
    ~28M params, modern conv design that often generalizes better than ResNet."""
    from torchvision.models import convnext_tiny
    net = convnext_tiny(weights=None)
    old = net.features[0][0]                         # Conv2d(3, 96, k=4, s=4) patchify stem
    net.features[0][0] = nn.Conv2d(in_frames, old.out_channels, kernel_size=4, stride=2, padding=1)
    net.classifier[2] = nn.Identity()               # drop final Linear -> keep LayerNorm+Flatten=768
    backbone = nn.Sequential(TimeAsChannels(), net.features, net.avgpool, net.classifier)
    return Embedder(backbone, feat_dim=768, embed_dim=embed_dim, n_classes=n_classes)


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


def make_r2plus1d_snr(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    """R(2+1)D with SNR3d after stages 1-2. R(2+1)D is the backbone the strongest published
    StepUP systems converged on, so this is the like-for-like comparison against gaitcnn_snr."""
    from torchvision.models.video import r2plus1d_18
    net = _to_single_channel(r2plus1d_18(weights=None))
    net = _add_snr(net, chans=(64, 128), three_d=True)
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 512)


def make_r2plus1d_light(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    """Lighter R(2+1)D: drop the last (heaviest, 512-channel) residual stage -> ~14M params vs
    31M, feature dim 256. Fewer blocks + fewer params memorize the train identities more slowly,
    which is the direction that helps open-set transfer to unseen footwear (not more depth)."""
    from torchvision.models.video import r2plus1d_18
    net = _to_single_channel(r2plus1d_18(weights=None))
    net.layer4 = nn.Identity()                       # remove the deepest stage; layer3 out = 256
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 256)


def make_r3d(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    from torchvision.models.video import r3d_18
    net = _to_single_channel(r3d_18(weights=None))
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 512)


def make_r3d_light(embed_dim=128, n_classes=None, resize=None, mixstyle=False):
    """Lighter R3D: drop the last 512-channel stage -> ~14M params, feature dim 256."""
    from torchvision.models.video import r3d_18
    net = _to_single_channel(r3d_18(weights=None))
    net.layer4 = nn.Identity()
    return _wrap3d(MixStyleVideo(net) if mixstyle else net, resize, embed_dim, n_classes, 256)


def make_swin3d(embed_dim=128, n_classes=None, resize=None, depths=(2, 2, 6, 2), feat_dim=768):
    """Video Swin re-parameterised for the small map: patch (2,2,2), window (8,4,4)."""
    from torchvision.models.video.swin_transformer import SwinTransformer3d
    heads = [3, 6, 12, 24][:len(depths)]
    net = SwinTransformer3d(patch_size=[2, 2, 2], embed_dim=96, depths=list(depths),
                            num_heads=heads, window_size=[8, 4, 4], num_classes=1)
    pe = net.patch_embed.proj
    net.patch_embed.proj = nn.Conv3d(1, pe.out_channels, pe.kernel_size, pe.stride, pe.padding)
    net.head = nn.Identity()
    return _wrap3d(net, resize, embed_dim, n_classes, feat_dim=feat_dim)


def make_swin3d_light(embed_dim=128, n_classes=None, resize=None):
    """Lighter video Swin: 3 stages with a shallow [2,2,2] depth instead of [2,2,6,2]."""
    return make_swin3d(embed_dim, n_classes, resize, depths=(2, 2, 2), feat_dim=384)


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
    # 3D nets resize to this clip. Cap at 48 frames / 64 spatial so the competition winner's exact
    # (48,48,48) input (--sample3d 48,48,48) passes through unchanged; the full 101-frame cube is
    # still downsampled (it OOMs 3D convs otherwise).
    clip3d = (min(48, sample3d[0]), min(64, sample3d[1]), min(64, sample3d[2]))
    reg = {
        "gaitcnn":  dict(fn=make_gaitcnn, kw=dict(in_frames=data_t), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_deep": dict(fn=make_gaitcnn_deep, kw=dict(in_frames=data_t), full_pk=pk2d,
                             smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "resnet2d_snr": dict(fn=make_resnet2d_snr, kw=dict(in_frames=data_t), full_pk=pk2d,
                             smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_snr": dict(fn=make_gaitcnn_snr, kw=dict(in_frames=data_t), full_pk=pk2d,
                            smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_r21d": dict(fn=make_gaitcnn_r21d, kw=dict(in_frames=data_t), full_pk=pk2d,
                             smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_in": dict(fn=make_gaitcnn_in, kw=dict(in_frames=data_t), full_pk=pk2d,
                           smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "gaitcnn_tiny": dict(fn=make_gaitcnn_tiny, kw=dict(in_frames=data_t), full_pk=pk2d,
                             smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "convnext": dict(fn=make_convnext, kw=dict(in_frames=data_t), full_pk=(64, 4),
                         smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "resnet2d": dict(fn=make_resnet2d, kw=dict(in_frames=data_t), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "resnet2d_light": dict(fn=make_resnet2d_light, kw=dict(in_frames=data_t), full_pk=pk2d,
                              smoke_pk=(2, 4), smoke_kw=dict(in_frames=data_t)),
        "cnnlstm":  dict(fn=make_cnnlstm, kw=dict(n_frames=min(32, data_t)), full_pk=pk2d,
                         smoke_pk=(2, 4), smoke_kw=dict(n_frames=min(16, data_t))),
        "r2plus1d": dict(fn=make_r2plus1d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "r2plus1d_snr": dict(fn=make_r2plus1d_snr, kw=dict(resize=clip3d), full_pk=pk3d,
                             smoke_pk=(2, 4), smoke_kw=dict(resize=(8, 16, 12))),
        "r2plus1d_light": dict(fn=make_r2plus1d_light, kw=dict(resize=clip3d), full_pk=pk3d,
                              smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "r3d":      dict(fn=make_r3d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "r3d_light": dict(fn=make_r3d_light, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "swin3d":   dict(fn=make_swin3d, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "swin3d_light": dict(fn=make_swin3d_light, kw=dict(resize=clip3d), full_pk=pk3d,
                            smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
        "vit":      dict(fn=make_vit, kw=dict(resize=clip3d), full_pk=pk3d,
                         smoke_pk=(2, 4), smoke_kw=dict(resize=(16, 32, 24))),
    }
    # per-model LR factor: heavy nets overfit fast and need a much lower LR than the light 2D CNNs.
    # Calibrated from the r2plus1d dynamics: val kept climbing while the (warming-up) LR was in the
    # 3-8e-4 band and only cratered once it crossed ~1.1e-3, so we place the *peak* at ~5e-4 for the
    # video ResNets (batch 256 -> 1e-3*sqrt(2)*0.35 = 5e-4) -- safely inside the climbing band, well
    # below the crash. Transformers are even more LR-brittle, so they stay lower.
    # The sqrt-batch rule alone lands the 2D nets at 2e-3 (batch 512) -- too hot for AdamW here, so
    # they peaked early and memorized exactly like the 3D nets did: the "every model early-stops"
    # symptom was one LR bug, not many. Factors below place every model's PEAK LR in the sane
    # 1.5e-4..8e-4 band regardless of its batch.
    for n, spec in reg.items():
        spec["lr_mult"] = 1.0
    for n in ("gaitcnn", "gaitcnn_in", "gaitcnn_snr", "resnet2d_snr", "gaitcnn_deep", "gaitcnn_tiny", "resnet2d", "resnet2d_light", "cnnlstm"):
        reg[n]["lr_mult"] = 0.4                           # 2D CNNs: 2e-3 -> ~8e-4 at batch 512
    for n in ("r2plus1d", "r2plus1d_snr", "r2plus1d_light", "r3d", "r3d_light", "gaitcnn_r21d"):
        reg[n]["lr_mult"] = 0.35                          # video ResNets + (2+1)D: peak ~5e-4 at batch 256
    for n in ("swin3d", "swin3d_light", "vit"):
        reg[n]["lr_mult"] = 0.2                           # transformers: gentler still
    reg["convnext"]["lr_mult"] = 0.5                       # ConvNeXt (28M) also wants a gentler LR
    return reg
