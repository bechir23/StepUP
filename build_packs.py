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
import json

import stepup.data as _sd
from stepup.config import ARTIFACTS, T, H, W
from stepup.data import (build_pack, build_split, existing_pack_path, get_manifest, get_split,
                         stride_pairs)


def resolved_split():
    """Identity lists per split, repaired here if the installed stepup.data hands back anything
    else. An older data.py could return a cached identity_split.json verbatim -- including a
    legacy file that stores counts instead of id lists -- so validate the shape at the point of
    use rather than trusting it."""
    split = get_split()
    ok = isinstance(split, dict) and all(isinstance(v, list) for v in split.values())
    if not ok:
        print(f"identity_split.json is not a roster of ids (got "
              f"{ {k: type(v).__name__ for k, v in (split or {}).items()} }); rebuilding")
        split = build_split()
        (ARTIFACTS / "identity_split.json").write_text(json.dumps(split, indent=1))
    return split


def verify_pack(path, n_probe=400, seed=0):
    """Check a pack actually holds footstep data, not just the right shape.

    build_pack preallocates the whole file before writing anything, so a build that was
    interrupted leaves a correctly-shaped file full of zeros -- and the reuse check only compares
    row count and resolution, so it would be accepted silently. Sample random rows and report how
    many are entirely zero."""
    import numpy as np
    if not path.exists():
        return None
    a = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a), size=min(n_probe, len(a)), replace=False)
    empty = 0
    mx = 0.0
    for i in sorted(idx.tolist()):
        row = np.asarray(a[i])
        m = int(row.max())
        mx = max(mx, m)
        if m == 0:
            empty += 1
    del a
    return dict(probed=len(idx), empty=empty, max=mx)


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
    ap.add_argument("--verify", action="store_true",
                    help="probe existing packs for real content (catches an interrupted build "
                         "that left a correctly-shaped file full of zeros)")
    args = ap.parse_args()

    res = None if args.res == "full" else tuple(int(v) for v in args.res.split(","))
    shape = tuple(res or (T, H, W))
    per_step = shape[0] * shape[1] * shape[2]          # uint8, 1 channel

    print(f"using stepup.data from: {_sd.__file__}")
    split = resolved_split()
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
            if args.verify and done:
                v = verify_pack(path)
                if v:
                    state = "OK" if v["empty"] == 0 else f"*** {v['empty']}/{v['probed']} EMPTY ***"
                    print(f"        content: probed {v['probed']} rows, max={v['max']}  {state}")
            if not args.check:
                before = path.exists()
                p = build_pack(s, man, res=res, mirror_right=mirror, workers=args.workers)
                print(f"        {'reused' if before else 'built'} -> {p}")
    print(f"\nestimated total on disk: {human(total)}")
    print(f"artifacts dir: {ARTIFACTS}")
    if args.check:
        print("\n(--check: nothing was built)")


if __name__ == "__main__":
    main()
