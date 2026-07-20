"""Shared argparse options for the train / eval / compare CLIs."""


def add_common_args(ap):
    g = ap.add_argument_group("run")
    g.add_argument("--P", type=int, default=0,
                   help="identities per batch (batch = P*K); 0 = the model's own full_pk")
    g.add_argument("--K", type=int, default=0, help="steps per identity; 0 = the model's full_pk")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--patience", type=int, default=10, help="early-stop patience (val cross EER)")
    g.add_argument("--steps-per-epoch", type=int, default=0,
                   help="0 = a full pass over the training footsteps at the model's batch")
    g.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW LR (reference-tuned 1e-3; raise for very large batches)")
    g.add_argument("--warmup-frac", type=float, default=0.05,
                   help="linear LR warmup fraction of epochs (short; reference ~3 epochs)")
    g.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (strong reg)")
    g.add_argument("--margin-warmup-frac", type=float, default=0.1,
                   help="ArcFace only: ramp the angular margin 0->target over this fraction of "
                        "epochs (prevents early embedding collapse)")
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--embed-dim", type=int, default=128)
    g.add_argument("--workers", type=int, default=8, help="DataLoader workers (Colab has cores)")
    g.add_argument("--log-every", type=int, default=0,
                   help="log train loss every N steps; 0 = auto (~20 logs/epoch, so the curve has "
                        "the SAME density for any batch size). val is logged per epoch")
    g.add_argument("--amp", action="store_true", help="mixed precision (default off = FP32)")

    d = ap.add_argument_group("data")
    d.add_argument("--pack-res", default="full", help="'full' (101,75,40) or 't,h,w' to downsample")
    d.add_argument("--sample3d", default="full", help="3D-model input size: 'full' or 't,h,w'")
    d.add_argument("--pack-device", default="cpu", choices=["cpu", "memmap", "cuda"])
    d.add_argument("--no-pack", dest="use_pack", action="store_false",
                   help="stream original npz instead of building a pack")
    d.add_argument("--val-monitor", type=int, default=3000, help="val steps embedded per epoch")
    d.add_argument("--augment", action="store_true", help="training augmentation on")
    d.add_argument("--no-augment", dest="augment", action="store_false")
    d.set_defaults(augment=True, use_pack=True)

    l = ap.add_argument_group("loss / mining")
    l.add_argument("--loss", default="arcface", choices=["arcface", "ce"],
                   help="ID loss. Default SubCenter-ArcFace s=16 (the reference recipe: angular "
                        "margin -> tight open-set clusters that transfer to unseen ids). CE is "
                        "kept for ablation only -- it caps open-set rank-1 ~0.5 (verified), so it "
                        "is NOT the default.")
    l.add_argument("--arc-scale", type=float, default=16.0,
                   help="ArcFace scale s (reference uses 16; lower = gentler)")
    l.add_argument("--mining", default="standard", choices=["standard", "crossfw"],
                   help="triplet mining: batch-hard or cross-footwear positive")
    l.add_argument("--mixstyle", action="store_true",
                   help="insert MixStyle in the video ResNets (r2plus1d/r3d) for domain generalization")

    w = ap.add_argument_group("logging")
    w.add_argument("--wandb", default="disabled", choices=["online", "offline", "disabled"])
    w.add_argument("--wandb-project", default="stepup-footstep")
    w.add_argument("--wandb-entity", default=None)
    w.add_argument("--hf-repo", default=None,
                   help="push artifacts to this HF model repo (e.g. Bechir23/stepup-footstep) "
                        "so heavy checkpoints live in HF storage, not on the disk")
    w.add_argument("--hf-token", default=None, help="HF token (else HF_TOKEN env / cached login)")
    w.add_argument("--hf-offload", action="store_true",
                   help="after pushing to HF, delete the local .pt checkpoint (keep metrics/plots) "
                        "so the artifacts folder stays light; evaluate.py refetches it from HF")

    s = ap.add_argument_group("smoke / subset")
    s.add_argument("--limit-ids", type=int, default=0,
                   help="keep only the first N identities per split (0 = all)")
    s.add_argument("--smoke", action="store_true",
                   help="tiny local run: few ids, stream (no pack), few epochs, small input")
    return ap


def apply_smoke(args):
    """Override to a fast CPU/4GB smoke run when --smoke is set."""
    if not args.smoke:
        return args
    args.limit_ids = args.limit_ids or 3
    args.use_pack = True                 # tiny pack (built one cube at a time) beats streaming
    args.pack_res = "24,32,24"           # small footstep -> small pack, low RAM
    args.pack_device = "cpu"
    args.epochs = min(args.epochs, 3)
    args.patience = 3
    args.steps_per_epoch = args.steps_per_epoch or 6
    args.workers = 0
    args.val_monitor = 200
    args.log_every = 2                               # tiny epochs -> log every 2 steps
    args.P, args.K = args.P or 2, args.K or 4        # keep user --P/--K if given
    if args.sample3d == "full":
        args.sample3d = "24,32,24"
    return args
