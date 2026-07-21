#!/usr/bin/env python
"""Build the StepUP packs ONCE, at full resolution, into artifacts/.

Storage resolution is fixed when the pack is written, so this builds at the native
(101, 75, 40) -- the superset. Every later run picks the resolution the model actually sees
with --sample3d, which is applied at load time, so nothing needs rebuilding to change it.

Packs are written UNMIRRORED (mirror_right=False): a right footstep keeps its true footprint
instead of being flipped onto the left. That is what --stride-pairs needs, and left/right gait
asymmetry measured as real signal (cross-footwear EER -2.5pp, same-footwear -3.4pp).

  python build_packs.py                 # full res (101,75,40), unmirrored  [default]
  python build_packs.py --res 48,48,32  # smaller pack if disk is tight
  python build_packs.py --mirrored      # also build the mirrored pack (for --no-stride-pairs)
  python build_packs.py --check         # report what exists, build nothing
"""
import argparse

from stepup.config import ARTIFACTS, T, H, W
from stepup.data import build_pack, existing_pack_path, get_manifest, get_split, stride_pairs


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default="full", help="'full' (101,75,40) or 't,h,w'")
    ap.add_argument("--mirrored", action="store_true",
                    help="also build the mirrored pack (only needed for --no-stride-pairs runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="report status only, build nothing")
    args = ap.parse_args()

    res = None if args.res == "full" else tuple(int(v) for v in args.res.split(","))
    shape = tuple(res or (T, H, W))
    per_step = shape[0] * shape[1] * shape[2]          # uint8, 1 channel

    split = get_split()
    sizes = {k: len(v) for k, v in split.items()}
    print(f"split: {sizes}   (identity-disjoint)")
    assert sizes["train"] + sizes["val"] + sizes["test"] == 150, "unexpected split total"
    print(f"pack resolution: {shape}  ({human(per_step)} per footstep)\n")

    modes = [False] + ([True] if args.mirrored else [])
    total = 0
    for mirror in modes:
        tag = "mirrored" if mirror else "UNMIRRORED (for --stride-pairs)"
        print(f"=== {tag} ===")
        for s in ("train", "val", "test"):
            man = get_manifest(s, split=split)
            n = len(man)
            est = n * per_step
            total += est
            path = existing_pack_path(s, man, res, mirror=mirror)
            done = path.exists()
            note = "exists" if done else "to build"
            print(f"  {s:5s} {n:>8,} footsteps  ~{human(est):>9}  [{note}] {path.name}")
            if not mirror:
                print(f"        -> {len(stride_pairs(man)):>8,} left/right strides")
            if not args.check:
                p = build_pack(s, man, res=res, mirror_right=mirror, workers=args.workers)
                print(f"        built -> {p}")
    print(f"\nestimated total on disk: {human(total)}")
    print(f"artifacts dir: {ARTIFACTS}")
    if args.check:
        print("\n(--check: nothing was built)")


if __name__ == "__main__":
    main()
