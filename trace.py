#!/usr/bin/env python
"""Print each backbone's feature-map shape after every stage on the real (101,75,40)
footstep, so you can confirm the small-input stems keep a strong map instead of shrinking it
aggressively (docs/BACKBONE_CATALOG.md). A large final map / many tokens = rich features.

  python trace.py                      # all models, full resolution
  python trace.py --model resnet2d     # one model
  python trace.py --sample3d 48,48,24  # trace the 3D models at a smaller input
"""
import argparse

import torch

from stepup.config import H, T, W, dev
from stepup.models import registry, set_dropout


def trace(name, spec, x, mixstyle=False):
    kw = dict(spec["kw"])
    if mixstyle and name in ("r2plus1d", "r3d"):
        kw["mixstyle"] = True
    net = spec["fn"](embed_dim=128, n_classes=None, **kw)
    net = net.to(dev).train() if (mixstyle and name in ("r2plus1d", "r3d")) else net.to(dev).eval()
    stage_names = {"conv1", "layer1", "layer2", "layer3", "layer4", "stem"}
    stage_cls = ("MaxPool2d", "PatchMerging", "SwinTransformerBlock", "LSTM",
                 "TransformerEncoder", "PatchEmbed3d", "MixStyle")
    logs, handles = [], []

    def hook(tag):
        def fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if hasattr(t, "shape"):
                logs.append((tag, tuple(t.shape[1:])))
        return fn

    for mn, m in net.named_modules():
        last = mn.split(".")[-1]
        cls = type(m).__name__
        if last in stage_names or cls in stage_cls:
            handles.append(m.register_forward_hook(hook(last if last in stage_names else cls)))
    with torch.no_grad():
        net(x.to(dev))
        for h in handles:
            h.remove()
        key = [logs[0]] + [logs[i] for i in range(1, len(logs)) if logs[i][0] != logs[i - 1][0]]
        prog = " -> ".join(f"{c}{s}" for c, s in key[:8])
        fi = net(x.to(dev))[1]
    n_params = sum(p.numel() for p in net.parameters())
    print(f"{name:9s} in {tuple(x.shape[1:])}  |  {prog}  |  embed {tuple(fi.shape[1:])}  "
          f"| {n_params/1e6:.1f}M")
    del net


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all")
    ap.add_argument("--sample3d", default="full", help="3D input size: 'full' or 't,h,w'")
    ap.add_argument("--mixstyle", action="store_true",
                    help="trace r2plus1d/r3d with MixStyle inserted (shows the shape after it)")
    args = ap.parse_args()
    set_dropout(0.0)
    s3d = (T, H, W) if args.sample3d == "full" else tuple(int(v) for v in args.sample3d.split(","))
    reg = registry(s3d, T)
    names = list(reg) if args.model == "all" else [args.model]
    x = torch.randn(2, 1, T, H, W)
    note = " (+MixStyle on r2plus1d/r3d)" if args.mixstyle else ""
    print(f"feature-map progression on the real (1,101,75,40) footstep{note} "
          "(large final map / many tokens = strong features):\n")
    for n in names:
        try:
            trace(n, reg[n], x, mixstyle=args.mixstyle)
        except RuntimeError as e:
            print(f"{n:9s} skipped ({str(e)[:60]})")


if __name__ == "__main__":
    main()
