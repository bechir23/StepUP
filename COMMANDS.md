# Run sheet — every experiment as a command

On Colab, first mount Drive and install the one missing package. `Data/` and `artifacts/`
are auto-detected under `/content/drive/MyDrive` (or set `STEPUP_ROOT`).

```bash
from google.colab import drive; drive.mount('/content/drive')     # in a notebook cell, OR:
# %cd /content/drive/MyDrive/stepup_repo   (wherever you put the repo)
pip install pytorch-metric-learning
# optional: pip install umap-learn wandb
export STEPUP_ROOT=/content/drive/MyDrive        # only if Data/ isn't auto-found
```

Defaults are the **Colab production** config (full `(101,75,40)` resolution, augment on,
100 epochs, per-model batch, FP32). Add `--smoke` for a tiny local sanity run.

---

## 0. Inspect model shapes (no aggressive shrink)
```bash
python trace.py                          # every backbone, per-stage shape on (101,75,40)
python trace.py --mixstyle               # r2plus1d/r3d with MixStyle in the path
python trace.py --model swin3d           # one model
```

## 1. Train the models (the rank / metrics per model)
```bash
python train.py --model all              # all 7 backbones, default recipe
python train.py --model gaitcnn          # single model
python train.py --model r2plus1d --plot-embed
```
Each writes to `artifacts/`: `{m}_best.pt`, `hist_{m}.parquet`, `curves_{m}.png` (loss/val
curves), `test_{m}.parquet`, `verif_{m}.parquet`, `acc_{m}.parquet` (accumulated rank-1),
and `embed_{m}.png` with `--plot-embed`. Console prints per-epoch rows + the test report +
**accumulated rank-1 (1/3/5/10-step)**.

## 2. Batch size (P, K) — tune per run
```bash
python train.py --model resnet2d --P 32 --K 4      # batch 128
python train.py --model r2plus1d --P 8  --K 4      # batch 32
python train.py --model gaitcnn  --P 64 --K 4      # batch 256
```

## 3. Augmentation on vs off (the DG difference)
```bash
python train.py --model resnet2d                   # augment ON (default)
python train.py --model resnet2d --no-augment      # augment OFF
```

## 4. Mining: standard vs cross-footwear (the invariance contribution)
```bash
python train.py --model r2plus1d --mining standard
python train.py --model r2plus1d --mining crossfw
python compare.py --compare mining --model r2plus1d   # trains both, prints the diff
```

## 5. MixStyle domain generalization (video ResNets)
```bash
python train.py --model r2plus1d                   # no MixStyle
python train.py --model r2plus1d --mixstyle        # + MixStyle
```

## 6. Loss: CE+triplet vs ArcFace+triplet
```bash
python train.py --model gaitcnn --loss ce                       # label-smoothed CE (default)
python train.py --model gaitcnn --loss arcface --arc-scale 32   # SubCenter-ArcFace
```

## 7. Evaluate a checkpoint (rank over a walking pass)
```bash
python evaluate.py --model r2plus1d                             # uses artifacts/r2plus1d_best.pt
python evaluate.py --model r2plus1d --ks 1,3,5,10,15 --plot-embed
```

## 8. Compare / leaderboard / plots
```bash
python compare.py --compare models          # table: cross EER/rank-1/F1/BACC/accumulated
python plots.py                             # compare_eer.png + compare_accumulated.png
```

## 9. wandb logging (per-epoch to your dashboard)
```bash
wandb login                                 # once
python train.py --model all --wandb online
```

---

## One-shot: everything
```bash
python trace.py                                            # verify shapes
python train.py --model all                                # train all 7 (rank + metrics + curves)
python train.py --model r2plus1d --mining crossfw          # + invariance run
python train.py --model r2plus1d --mixstyle                # + DG run
python train.py --model gaitcnn  --loss arcface --arc-scale 32
python train.py --model resnet2d --no-augment              # augment ablation
python compare.py --compare models                         # leaderboard
python plots.py                                            # comparison figures
```

## Full flag list
`--model --P --K --epochs --patience --steps-per-epoch --lr --warmup-frac --weight-decay`
`--dropout --embed-dim --workers --amp --pack-res --sample3d --pack-device {cpu,memmap,cuda}`
`--no-pack --val-monitor --augment/--no-augment --loss {ce,arcface} --arc-scale`
`--mining {standard,crossfw} --mixstyle --wandb {online,offline,disabled} --ks --plot-embed --smoke`
