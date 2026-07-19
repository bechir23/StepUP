# StepUP — Footwear-Invariant Footstep Biometrics (repo)

Modular port of `stepup_150.ipynb`. Open-set metric learning on UNB StepUP-P150
plantar-pressure footsteps: learn an embedding that matches the same person across
*different footwear*; test identities are disjoint from train. See `../docs/WORKFLOW.md`
for the full, cited methodology.

## Layout
```
stepup/            package
  config.py        paths, device, seed, cfg builder
  data.py          metadata, split, datasets, PK samplers, packs, build_datasets()
  models.py        7 small-input-corrected backbones + BNNeck + registry(full_pk)
  losses.py        mining (standard/crossfw) + Criterion (CE / ArcFace + triplet)
  metrics.py       identification, verification, competition report
  eval.py          leave-one-footwear-out, accumulated rank-1, embedding plot
  engine.py        training loop (warmup+cosine, early stop, per-epoch log)
  args.py          shared CLI options
train.py           train + test a model (or all)
evaluate.py        score a saved checkpoint
compare.py         mining comparison / model leaderboard
```

## Setup
```bash
pip install -r requirements.txt
export STEPUP_ROOT=/path/to/Footsteps    # folder with Data/ and participant_metadata.csv
                                         # (auto-detected if the repo sits inside it)
```

## Models
`gaitcnn` (compact pressure CNN), `resnet2d` (gait-style stem), `cnnlstm` (LRCN),
`r2plus1d`, `r3d` (torchvision Kinetics, small-input stems), `swin3d` (re-patched Video
Swin), `vit` (ViViT/VideoMAE encoder). All keep the 75×40 map instead of crushing it.

## Inspect model shapes first (verify no aggressive shrinking)
```bash
python trace.py                       # every backbone's feature-map shape per stage, full res
python trace.py --model resnet2d      # e.g. resnet2d keeps 75x40 -> 19x10 (not 3x2)
```

## Train
```bash
python train.py --model resnet2d                      # one model, defaults (CE+triplet, augment)
python train.py --model all                           # every backbone
python train.py --model r2plus1d --mining crossfw     # cross-footwear positive mining
python train.py --model r2plus1d --mixstyle           # + MixStyle domain generalization
python train.py --model gaitcnn  --loss arcface --arc-scale 32
python train.py --model resnet2d --no-augment --P 32 --K 4 --epochs 50
python train.py --model swin3d   --plot-embed         # + save embedding plot
python train.py --model all      --wandb online       # log every epoch to wandb
```
**Batch size**: each model has a default in the registry `full_pk` (2D=64, 3D/transformer=32);
override with `--P N --K M` (batch = P*K). One epoch = a full pass unless `--steps-per-epoch`.
`--mixstyle` applies to the video ResNets (r2plus1d/r3d).

### What `--model all` does (and what it does NOT)
`--model all` trains the **7 backbones once, each with the defaults**: `loss=ce`,
`mining=standard`, `augment=on`, `mixstyle=off`, each model's own batch (`full_pk`: 2D=64,
3D/transformer=32), `lr=1e-4`, `dropout=0.2`, `weight_decay=5e-4`, warmup+cosine, 100 epochs,
early stop, FP32. It writes per-model artifacts and prints test + accumulated rank-1.

It **does NOT sweep** augment/mining/mixstyle/loss — each of those is a **separate command**
(only the named flag changes from the defaults). See `COMMANDS.md` (quick) and
`COMMANDS_EXPANDED.md` (every default + how to correct a run). MixStyle is applied only to the
models that support it (r2plus1d/r3d); the big 3D/transformer nets use a smaller batch than the
light 2D nets. To run **every model at the same batch 64**, add `--P 16 --K 4`.

Logging is **step-based** (`--log-every 0` = auto ~20 points/epoch, same density at any batch),
so even a short run gives a dense training curve; val is logged per epoch.

## Evaluate a checkpoint
```bash
python evaluate.py --model resnet2d --ckpt artifacts/resnet2d_best.pt --ks 1,3,5,10 --plot-embed
```
Prints per-cell rank-1/EER, the competition verification report, and **accumulated rank-1**
over a walking pass (k=1,3,5,10 — the deployment metric that reaches high values at k=5–10).

## Compare
```bash
python compare.py --compare mining --model resnet2d --epochs 20   # standard vs crossfw
python compare.py --compare models                                # leaderboard of trained models
```

## Key knobs (all `--flags`)
`--model --P --K --epochs --lr --loss {ce,arcface} --arc-scale --mining {standard,crossfw}`
`--augment/--no-augment --dropout --weight-decay --warmup-frac --pack-res --sample3d`
`--pack-device {cpu,memmap,cuda} --val-monitor --ks --wandb {online,offline,disabled}`

## Outputs (in `artifacts/`)
`{model}_best.pt`, `hist_{model}.parquet` (per-epoch curves), `test_{model}.parquet`
(per-cell), `verif_{model}.parquet` (competition report), `acc_{model}.parquet` (accumulated
rank-1), `embed_{model}.png`, `model_leaderboard.parquet`.
