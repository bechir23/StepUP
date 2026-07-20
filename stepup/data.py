"""Data layer: metadata, identity split, preprocessing, datasets, PK samplers, and packs.

A footstep = one cube slice (101,75,40). Metadata is built once and cached; the split is the
frozen identity-disjoint 100/25/25. Training reads from a per-split uint8 pack (built once,
resolution-tagged, read from the artifacts folder) or streams the original npz trials.
"""
import itertools
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import affine as tv_affine

from .config import (ARTIFACTS, COLAB, DATA, FW2CODE, FOOTWEAR, ROOT, SEED, SPEEDS,
                     T, H, W, seed_everything)

TRIAL_CACHE = 8


# --------------------------------------------------------------------- metadata + split
def get_metadata():
    """One row per footstep over every trial, cached to parquet. Returns (steps, clean)."""
    path = ARTIFACTS / "footstep_metadata.parquet"
    if path.exists():
        steps = pd.read_parquet(path)
    else:
        trials = list(itertools.product(range(1, 151), FOOTWEAR, SPEEDS))

        def read(t):
            pid, fw, sp = t
            return pd.read_csv(DATA / f"{pid:03d}" / fw / sp / "metadata.csv")

        with ThreadPoolExecutor(max_workers=16 if COLAB else 4) as ex:
            steps = pd.concat(list(ex.map(read, trials)), ignore_index=True)
        steps.to_parquet(path, index=False)
    steps["row"] = steps.groupby(["ParticipantID", "Footwear", "Speed"]).cumcount()
    return steps, steps.query("Exclude == False")


def build_split(seed=1337):
    """Identity-disjoint 100/25/25, stratified by sex x foot-length tertile x age tertile."""
    people = pd.read_csv(ROOT / "participant_metadata.csv")
    rng = np.random.default_rng(seed)
    df = people.copy()
    df["foot_bin"] = pd.qcut(df["RightFootLength (cm)"], 3, labels=False, duplicates="drop")
    df["age_bin"] = pd.qcut(df.Age, 3, labels=False, duplicates="drop")
    pattern = ["train"] * 4 + ["val", "test"]
    out = {"train": [], "val": [], "test": []}
    cursor = 0
    for _, grp in df.groupby(["Sex", "foot_bin", "age_bin"], sort=True):
        ids = grp.ParticipantID.to_numpy().copy()
        rng.shuffle(ids)
        for pid in ids:
            out[pattern[cursor % len(pattern)]].append(int(pid))
            cursor += 1
    return {k: sorted(v) for k, v in out.items()}


def get_split():
    path = ARTIFACTS / "identity_split.json"
    if path.exists():
        return json.loads(path.read_text())
    split = build_split()
    path.write_text(json.dumps(split, indent=1))
    return split


def get_manifest(split_name, clean=None, split=None):
    """One row per clean footstep for a split: identity, footwear, pass, side, cube row.
    Reuses the cached manifest_{split}.parquet if present (this is the exact order the packs
    were built from, so the pack rows stay aligned) and only rebuilds it otherwise."""
    cached = ARTIFACTS / f"manifest_{split_name}.parquet"
    if cached.exists():
        return pd.read_parquet(cached).reset_index(drop=True)
    clean = clean if clean is not None else get_metadata()[1]
    split = split if split is not None else get_split()
    pids = set(split[split_name])
    sub = clean[clean.ParticipantID.isin(pids) & clean.Speed.isin(SPEEDS)]
    man = sub[["ParticipantID", "Footwear", "Speed", "PassID",
               "FootstepID", "Side", "row"]].reset_index(drop=True)
    man.to_parquet(cached, index=False)
    return man


# --------------------------------------------------------------------- preprocessing
@lru_cache(maxsize=TRIAL_CACHE)
def _trial_cube(pid, fw, sp):
    with np.load(DATA / f"{pid:03d}" / fw / sp / "pipeline_1.npz") as z:
        return z["arr_0"].astype(np.float32)


def preprocess_cube(cube, side, mirror_right=True):
    """Peak-normalise to 0..1 and mirror right footsteps onto the left (removes side)."""
    x = cube
    if mirror_right and side == "Right":
        x = x[..., ::-1]
    peak = x.max()
    if peak > 0:
        x = x / peak
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------- datasets
class _AugMixin:
    def _aug(self, x):
        """Winner recipe plus regularizers (gamma, random erasing) to fight overfitting.
        No horizontal flip (that undoes the medial-lateral mirror)."""
        x = x + 0.02 * torch.randn_like(x)
        if torch.rand(1).item() < 0.3:
            x = x * (torch.rand(x.shape[-2:], device=x.device) > 0.05).float()
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        if torch.rand(1).item() < 0.5:              # rotation + footprint scale + small shift
            angle = float(torch.empty(1).uniform_(-15, 15))
            scale = float(torch.empty(1).uniform_(0.8, 1.25))   # contact-area robustness (ref)
            tx = int(torch.randint(-3, 4, (1,)))
            ty = int(torch.randint(-3, 4, (1,)))
            x = tv_affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0])
        if torch.rand(1).item() < 0.5:
            peak = x.max()
            if peak > 0:
                g = float(torch.empty(1).uniform_(0.7, 1.5))
                x = (x / peak).clamp_(min=0).pow(g) * peak
        if torch.rand(1).item() < 0.3:
            h, w = x.shape[-2], x.shape[-1]
            dh = int(torch.randint(1, max(2, h // 3), (1,)))
            dw = int(torch.randint(1, max(2, w // 3), (1,)))
            h0 = int(torch.randint(0, max(1, h - dh), (1,)))
            w0 = int(torch.randint(0, max(1, w - dw), (1,)))
            x[..., h0:h0 + dh, w0:w0 + dw] = 0
        return x.clamp_(min=0)


class FootstepData(Dataset, _AugMixin):
    """Streams one preprocessed cube (1,101,75,40) and its identity label from the npz trials."""

    def __init__(self, manifest, in_memory=False, mirror_right=True, augment=False):
        self.m = manifest.reset_index(drop=True)
        self.mirror_right, self.augment, self.packed = mirror_right, augment, None
        self.pid2label = {p: i for i, p in enumerate(sorted(self.m.ParticipantID.unique()))}
        self.label = self.m.ParticipantID.map(self.pid2label).to_numpy()
        self.fw = self.m.Footwear.map(FW2CODE).to_numpy()
        if in_memory:
            buf = np.empty((len(self.m), 1, T, H, W), np.float32)
            for i, r in enumerate(self.m.itertuples()):
                raw = _trial_cube(int(r.ParticipantID), r.Footwear, r.Speed)[int(r.row)]
                buf[i, 0] = preprocess_cube(raw, r.Side, mirror_right)
            self.packed = torch.from_numpy(buf)
            _trial_cube.cache_clear()

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        if self.packed is not None:
            x = self.packed[i]
        else:
            r = self.m.iloc[i]
            raw = _trial_cube(int(r.ParticipantID), r.Footwear, r.Speed)[int(r.row)]
            x = torch.from_numpy(preprocess_cube(raw, r.Side, self.mirror_right)).unsqueeze(0)
        if self.augment:
            x = self._aug(x.clone())
        return x, int(self.label[i]), int(self.fw[i])


class PKSampler(Sampler):
    """Each batch is P identities x K steps. Yields index lists of length P*K."""

    def __init__(self, manifest, P=8, K=4, batches=None, seed=SEED):
        self.m = manifest.reset_index(drop=True)
        self.P, self.K = P, K
        self.by_pid = {p: g.index.to_numpy() for p, g in self.m.groupby("ParticipantID")}
        self.pids = list(self.by_pid)
        self.batches = batches or (len(self.m) // (P * K))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for p in self.rng.choice(self.pids, self.P, replace=False):
                pool = self.by_pid[p]
                batch += list(self.rng.choice(pool, self.K, replace=len(pool) < self.K))
            yield batch


class FootwearSpanningSampler(Sampler):
    """PK batches whose K steps per identity span footwear, so cross-shoe positives exist."""

    def __init__(self, manifest, P=8, K=4, batches=None, seed=SEED):
        self.m = manifest.reset_index(drop=True)
        self.P, self.K = P, K
        self.idx_by = {}
        for pid, g in self.m.groupby("ParticipantID"):
            self.idx_by[pid] = {fw: sub.index.to_numpy()
                                for fw, sub in g.groupby(g.Footwear.map(FW2CODE))}
        self.pids = list(self.idx_by)
        self.batches = batches or (len(self.m) // (P * K))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for p in self.rng.choice(self.pids, self.P, replace=False):
                fws = list(self.idx_by[p])
                chosen = self.rng.choice(fws, self.K, replace=len(fws) < self.K)
                for fw in chosen:
                    batch.append(int(self.rng.choice(self.idx_by[p][fw])))
            yield batch


# --------------------------------------------------------------------- packs
def _res_tag(res):
    r = res or (T, H, W)
    return f"{r[0]}x{r[1]}x{r[2]}"


def pack_path(split_name, res):
    return ARTIFACTS / f"pack_{split_name}_{_res_tag(res)}_u8.npy"


def _matches(path, n, want):
    """True if an existing pack file has the right length and resolution."""
    if not path.exists():
        return False
    ex = np.load(path, mmap_mode="r")
    ok = len(ex) == n and tuple(ex.shape[2:]) == tuple(want)
    del ex
    return ok


def existing_pack_path(split_name, manifest, res):
    """The pack file to read: the resolution-tagged name, or the legacy untagged
    `pack_{split}_u8.npy` (from earlier runs) if it exists and matches the resolution."""
    want = tuple(res or (T, H, W))
    n = len(manifest)
    tagged = pack_path(split_name, res)
    if _matches(tagged, n, want):
        return tagged
    legacy = ARTIFACTS / f"pack_{split_name}_u8.npy"
    if _matches(legacy, n, want):
        return legacy
    return tagged                                     # neither present -> build here


def build_pack(split_name, manifest, res=None, mirror_right=True, overwrite=False, workers=4):
    """Write every preprocessed footstep of a split into one uint8 array, once. Reuses an
    existing pack (tagged or legacy-untagged) that matches, so packs built by the notebook are
    not rebuilt."""
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    want = tuple(res or (T, H, W))
    if not overwrite:
        found = existing_pack_path(split_name, manifest, res)
        if found.exists():
            return found
    path = pack_path(split_name, res)
    m = manifest.reset_index(drop=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(m), 1, *want))
    groups = list(m.groupby(["ParticipantID", "Footwear", "Speed"]))

    def fill(item):
        (pid, fw, sp), grp = item
        with np.load(DATA / f"{int(pid):03d}" / fw / sp / "pipeline_1.npz") as z:
            cube = z["arr_0"]
        for i, r in zip(grp.index.to_numpy(), grp.itertuples(index=False)):
            x = preprocess_cube(cube[int(r.row)], r.Side, mirror_right)
            if res is not None:
                x = F.interpolate(torch.from_numpy(x)[None, None], size=res,
                                  mode="trilinear", align_corners=False)[0, 0].numpy()
            mm[i, 0] = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(tqdm(ex.map(fill, groups), total=len(groups), desc=f"pack {split_name}"))
    mm.flush()
    return path


class PackedData(Dataset, _AugMixin):
    """Footsteps from a prebuilt pack read from the artifacts folder. device: cpu (RAM array),
    memmap (paged), or cuda (VRAM-resident, gathered on the GPU). rows selects a subset."""

    def __init__(self, split_name, manifest, res=None, augment=False, device="cpu", rows=None):
        from .config import dev
        path = existing_pack_path(split_name, manifest, res)   # tagged or legacy untagged pack
        self.device = device
        full = manifest.reset_index(drop=True)
        self.rows = np.arange(len(full)) if rows is None else np.asarray(rows, dtype=np.int64)
        self.m = full.iloc[self.rows].reset_index(drop=True)
        self.pid2label = {p: i for i, p in enumerate(sorted(self.m.ParticipantID.unique()))}
        self.label = self.m.ParticipantID.map(self.pid2label).to_numpy()
        self.fw = self.m.Footwear.map(FW2CODE).to_numpy()
        self.augment = augment
        if device == "cuda":
            self.pack = torch.from_numpy(np.load(path)).to(dev)
            self.label_t = torch.as_tensor(self.label, device=dev)
            self.fw_t = torch.as_tensor(self.fw, device=dev)
            n = len(self.pack)
        else:
            self.mm = np.load(path, mmap_mode="r" if device == "memmap" else None)
            n = len(self.mm)
        assert n == len(full), "pack and manifest out of sync; rebuild the pack"

    def __len__(self):
        return len(self.rows)

    def gather(self, idx):
        from .config import dev
        idx = np.asarray(idx)
        ii = torch.as_tensor(self.rows[idx], device=dev)
        xb = self.pack[ii].to(torch.float32).div_(255.0)
        if self.augment:
            xb = torch.stack([self._aug(x) for x in xb])
        jj = torch.as_tensor(idx, device=dev)
        return xb, self.label_t[jj], self.fw_t[jj]

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.mm[self.rows[i]], dtype=np.float32) / 255.0)
        if self.augment:
            x = self._aug(x)
        return x, int(self.label[i]), int(self.fw[i])


def build_datasets(cfg):
    """Build packs (once) and return (manifests, datasets) per cfg. datasets has keys
    train/val/test/val_mon; val_mon is a small stratified subset for per-epoch early stopping."""
    _, clean = get_metadata()
    split = get_split()
    man = {s: get_manifest(s, clean, split) for s in ("train", "val", "test")}
    lim = cfg.get("limit_ids", 0)
    if lim:                                                    # subset for smoke/local runs
        for s in man:
            pids = sorted(man[s].ParticipantID.unique())[:lim]
            man[s] = man[s][man[s].ParticipantID.isin(pids)].reset_index(drop=True)
    res = cfg["pack_res"]
    if cfg["use_pack"]:
        pw = 1 if lim else 4             # single worker keeps peak RAM to one float64 cube
        for s in ("train", "val", "test"):
            build_pack(s, man[s], res=res, workers=pw)
        ds_train = PackedData("train", man["train"], res=res, augment=cfg["augment"],
                              device=cfg["pack_device"])
        ds_val = PackedData("val", man["val"], res=res, device="memmap")
        ds_test = PackedData("test", man["test"], res=res, device="memmap")
        rng = np.random.default_rng(SEED)
        mv = man["val"].reset_index(drop=True)
        keep = []
        for _fw, g in mv.groupby("Footwear"):
            k = max(1, round(cfg["val_monitor"] * len(g) / len(mv)))
            keep += list(rng.choice(g.index.to_numpy(), min(k, len(g)), replace=False))
        ds_val_mon = PackedData("val", man["val"], res=res, device="memmap", rows=np.sort(keep))
    else:
        ds_train = FootstepData(man["train"], augment=cfg["augment"])
        ds_val = ds_test = ds_val_mon = FootstepData(man["val"])
    return man, dict(train=ds_train, val=ds_val, test=ds_test, val_mon=ds_val_mon)
