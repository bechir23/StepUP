"""Paths, constants, device, seeding, and the run-config builder.

ROOT is located from STEPUP_ROOT, then Colab Drive, then by searching upward from this file
for the folder that holds Data/ and participant_metadata.csv (so the repo can live inside the
existing Footsteps folder). Data/ holds the pipeline_1 footstep trials; artifacts/ is written.
"""
import os
import pathlib
import random
import sys

import numpy as np
import torch

SEED = 42
COLAB = "google.colab" in sys.modules
T, H, W = 101, 75, 40
FOOTWEAR = ["BF", "ST", "P1", "P2"]
FW2CODE = {fw: i for i, fw in enumerate(FOOTWEAR)}
FOOTWEAR_LABEL = {"BF": "barefoot/sock", "ST": "standard shoe",
                  "P1": "personal shoe 1", "P2": "personal shoe 2"}
SPEEDS = ["W1", "W2", "W3", "W4"]


def find_root():
    if os.environ.get("STEPUP_ROOT"):
        return pathlib.Path(os.environ["STEPUP_ROOT"])
    def _has_data(p):
        p = pathlib.Path(p)
        return (p / "Data").exists() or (p / "participant_metadata.csv").exists()

    # 1) explicit override
    if os.environ.get("STEPUP_ROOT"):
        return pathlib.Path(os.environ["STEPUP_ROOT"])
    # 2) Google Drive, detected by the FILESYSTEM mount (/content/drive), not the google.colab
    #    import -- a `!python` subprocess does not import google.colab, so COLAB is False there,
    #    but the Drive mount is still visible. Mount it if we are the notebook kernel and it is
    #    not there yet.
    if COLAB and not pathlib.Path("/content/drive").exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            pass
    if pathlib.Path("/content/drive/MyDrive").exists():
        for p in ("/content/drive/MyDrive", "/content/drive/MyDrive/Footsteps",
                  "/content/drive/MyDrive/StepUP", "/content/drive/MyDrive/stepup"):
            if _has_data(p):
                return pathlib.Path(p)
    # 3) search upward from this file (local checkouts that sit inside the data folder)
    here = pathlib.Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _has_data(parent):
            return parent
    # 4) last resort
    return pathlib.Path("/content/drive/MyDrive") if pathlib.Path("/content/drive").exists() \
        else pathlib.Path.cwd()


ROOT = find_root()
DATA = ROOT / "Data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
dev = os.environ.get("STEPUP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def seed_everything(seed=SEED, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def build_cfg(args):
    """Assemble the run config dict from parsed CLI args (train/eval share this)."""
    return dict(
        P=args.P, K=args.K, epochs=args.epochs, workers=args.workers,
        amp=args.amp, embed_dim=args.embed_dim, lr=args.lr,
        sample3d=(T, H, W) if args.sample3d == "full" else tuple(int(v) for v in args.sample3d.split(",")),
        augment=args.augment, pack_res=None if args.pack_res == "full"
        else tuple(int(v) for v in args.pack_res.split(",")),
        use_pack=args.use_pack, pack_device=args.pack_device,
        stride_pairs=getattr(args, 'stride_pairs', False),
        val_monitor=args.val_monitor, warmup_frac=args.warmup_frac,
        dropout=args.dropout, weight_decay=args.weight_decay,
        loss=args.loss, arc_scale=args.arc_scale, mining=args.mining,
        fw_triplet_weight=getattr(args, "fw_triplet_weight", 1.0),
        adv_weight=getattr(args, "adv_weight", 1.0),
        cal_weight=getattr(args, "cal_weight", 0.5),
        label_smooth=getattr(args, "label_smooth", 0.0),
        patience=args.patience, steps_per_epoch=args.steps_per_epoch,
        limit_ids=getattr(args, "limit_ids", 0), log_every=getattr(args, "log_every", 100),
        margin_warmup_frac=getattr(args, "margin_warmup_frac", 0.1),
    )
