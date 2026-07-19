# Expanded commands — exactly what each run does, and how to change it

Every flag has a default. A command only changes the flags you pass; everything else stays at
the default below. Use this to know what config a run used and how to correct it.

## Defaults (the Colab-production config)
| flag | default | meaning |
|---|---|---|
| `--model` | `all` | which backbone(s); `all` = the 7 below, one after another |
| `--P` / `--K` | `0` / `0` | batch = P*K; `0` = use the model's own `full_pk` (table below) |
| `--epochs` | `100` | max epochs |
| `--patience` | `10` | early-stop patience on val cross-EER |
| `--steps-per-epoch` | `0` | `0` = a full pass, `len(train)//(P*K)` steps |
| `--lr` | `1e-4` | AdamW learning rate |
| `--warmup-frac` | `0.1` | linear warmup fraction, then cosine decay |
| `--weight-decay` | `5e-4` | AdamW weight decay |
| `--dropout` | `0.2` | dropout in the BNNeck head |
| `--workers` | `8` | DataLoader workers |
| `--log-every` | `100` | log train loss every N steps (curve is **step-based**) |
| `--amp` | off | precision; default **FP32** |
| `--pack-res` | `full` | stored footstep resolution = `(101,75,40)` |
| `--sample3d` | `full` | 3D-model input size = `(101,75,40)` |
| `--pack-device` | `cpu` | pack in RAM (`memmap`/`cuda` also available) |
| `--val-monitor` | `3000` | val steps embedded per epoch for early stopping |
| `--augment` | **on** | flip+rotation+noise+dropout+gamma+random-erase |
| `--loss` | `ce` | label-smoothed cross-entropy ID loss (+ triplet) |
| `--arc-scale` | `32` | ArcFace scale (only if `--loss arcface`) |
| `--mining` | `standard` | batch-hard triplet mining |
| `--mixstyle` | off | MixStyle DG (only r2plus1d/r3d) |
| `--wandb` | `disabled` | pass `online` to log to your dashboard |
| `--ks` | `1,3,5,10` | accumulation levels for rank-1 (evaluate.py) |

## Per-model batch (`full_pk`, used when `--P/--K` are 0)
| model | P | K | batch |
|---|---|---|---|
| gaitcnn, resnet2d, cnnlstm | 16 | 4 | **64** |
| r2plus1d, r3d, swin3d, vit | 8 | 4 | **32** |
Change per run with `--P N --K M` (overrides the table).

## What `python train.py --model all` does
It **trains each of the 7 backbones once, with the defaults**: `loss=ce`, `mining=standard`,
`augment=on`, `mixstyle=off`, each model's own `full_pk` batch, 100 epochs, early stop. It
writes per-model `{m}_best.pt / hist / curves / test / verif / acc` and prints the test report
+ accumulated rank-1. **It does NOT sweep** augment on/off, mining, mixstyle, or ce-vs-arcface
— those are **separate commands** (below). So `--model all` = the baseline recipe on all models;
the comparisons are extra runs you launch explicitly.

---

## Each experiment = one explicit command

**Baseline, all models** (ce, standard, augment on, per-model batch):
```bash
python train.py --model all
```

**One model, custom batch:**
```bash
python train.py --model resnet2d --P 32 --K 4        # batch 128 (override full_pk)
python train.py --model r2plus1d --P 8  --K 4        # batch 32
```

**Augmentation on vs off** (only `--augment` differs):
```bash
python train.py --model resnet2d                     # augment ON  (default)
python train.py --model resnet2d --no-augment        # augment OFF
```

**Mining standard vs cross-footwear** (only `--mining` differs):
```bash
python train.py --model r2plus1d                     # mining standard (default)
python train.py --model r2plus1d --mining crossfw    # cross-footwear positive mining
python compare.py --compare mining --model r2plus1d  # trains BOTH, prints the diff table
```

**MixStyle on vs off** (only `--mixstyle` differs; video ResNets only):
```bash
python train.py --model r2plus1d                     # no MixStyle (default)
python train.py --model r2plus1d --mixstyle          # + MixStyle
```

**Loss ce vs arcface** (only `--loss`/`--arc-scale` differ):
```bash
python train.py --model gaitcnn                                  # ce (default)
python train.py --model gaitcnn --loss arcface --arc-scale 32    # SubCenter-ArcFace
```

**Evaluate a checkpoint / accumulated rank over a pass:**
```bash
python evaluate.py --model r2plus1d --ks 1,3,5,10,15 --plot-embed
```

**Leaderboard + plots (after training):**
```bash
python compare.py --compare models        # table from the saved test/verif/acc parquets
python plots.py                            # compare_eer.png + compare_accumulated.png
```

**Log to wandb** (append to any train/compare command):
```bash
python train.py --model all --wandb online
```

---

## If a run misbehaves — how to correct the config
- **Out of memory (3D/transformer)**: lower the batch, e.g. `--P 4 --K 4` (batch 16); or
  `--sample3d 64,64,32` to shrink only the 3D input.
- **Overfitting** (val worse after a few epochs): keep `--augment`, raise `--dropout 0.3`,
  `--weight-decay 1e-3`, or lower `--lr 5e-5`.
- **Curve too sparse**: lower `--log-every` (e.g. 50). Too noisy: raise it (e.g. 200).
- **Epoch too long / short**: set `--steps-per-epoch N` (0 = a full data pass).
- **RAM tight**: `--pack-device memmap` (pages from disk instead of holding the pack in RAM).
- **Reproduce one comparison arm**: copy its exact command above; only the named flag changed
  from the defaults table.
