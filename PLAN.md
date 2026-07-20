# Experiment plan — keeping comparisons clean after the changes

## What changed (and why it matters for comparisons)
1. **LR scaled to batch** — `eff_lr = base_lr * sqrt(P*K/128)`, `base_lr=1e-3`. So every model
   gets an LR matched to its batch (2D 512 → ~2e-3, 3D 256 → ~1.4e-3). Fixes the "val plateaus
   right after warmup" seen with the old flat 1e-4.
2. **Stronger reg + short warmup** — `weight_decay=1e-2`, `warmup_frac=0.05`.
3. **Scale augmentation** — affine rotation + footprint scale (0.8–1.25) + small shift.
4. **ArcFace path fixed** — `s=16` + angular-margin warmup (0→target), the reference recipe.
5. **Two val metrics logged every epoch**:
   - `val_r1(cross)` / `cross_eer` — strict leave-one-footwear-out (our thesis metric; early-stop
     is on `cross_eer`).
   - `val_r1(mixed5)` — reference open-set protocol (mixed gallery, 5-step) — the fair
     comparison to the reference's ~0.9.
6. **No repack / no re-split** — these are optimizer/loss/aug/eval changes. Reuse the same Drive
   `Data/`, `split.json`, manifests, and packs.

## The one baseline to fix first
Get the 2D baseline strong before trusting anything else. Run and read the curves:
```bash
python train.py --model gaitcnn   --wandb online --hf-repo Bechir23/stepup-footstep --hf-offload
python train.py --model resnet2d  --wandb online --hf-repo Bechir23/stepup-footstep --hf-offload
```
Judge on: does `cross_eer` keep dropping past epoch ~10 now (LR fix), and does `val_r1(mixed5)`
climb toward the reference? If the baseline is healthy, the rest will be.

## Controlled comparison matrix (change ONE thing at a time)
Hold everything else at the defaults; only the named flag differs, so numbers stay comparable.

| axis | command (only this flag changes) |
|---|---|
| **models** | `--model all` (baseline recipe on all 7) |
| **loss** | `--loss ce` (default) vs `--loss arcface` |
| **mining** | `--mining standard` vs `--mining crossfw` |
| **DG** | (none) vs `--mixstyle` (r2plus1d / r3d only) |
| **augment** | (on, default) vs `--no-augment` |
| **batch** | default `full_pk` vs `--P … --K …` |

Rule: **never vary two axes in one run** — otherwise you can't attribute the change.

## Suggested order (cheap → expensive, each builds on the last)
1. **Baseline all models** → `python train.py --model all` → `compare.py --compare models` +
   `plots.py`. Establishes the leaderboard.
2. **Loss**: `gaitcnn --loss arcface` vs the baseline gaitcnn. Pick the better ID loss.
3. **Mining**: `compare.py --compare mining --model r2plus1d` (standard vs crossfw) — the
   footwear-invariance contribution; watch `cross_rank1`.
4. **DG**: `r2plus1d --mixstyle` vs baseline r2plus1d.
5. **Augment ablation**: `resnet2d --no-augment` vs baseline.
6. Re-run `compare.py --compare models` + `plots.py` for the final table + figures.

## Which number decides a winner
- **Thesis metric**: `cross_eer` / cross-footwear accumulated rank-1 (`acc_*.parquet`) — footwear
  invariance is the point.
- **Reference comparison**: `val_r1(mixed5)` and `evaluate.py … ` mixed-gallery accumulated —
  to show we reach the ~0.9 the reference reports under *their* protocol.
- Always read both; report both in the thesis (single-step cross, and accumulated per k).

## Evaluate / read results
```bash
python evaluate.py --model r2plus1d --hf-repo Bechir23/stepup-footstep --ks 1,3,5,10,15
# prints cross-footwear accumulated AND mixed-gallery accumulated side by side
```

## If a heavy 3D model OOMs at batch 256
Lower only its batch: `--model r2plus1d --P 16 --K 4` (or `--sample3d 64,64,32`). Note it in the
write-up (that model used a smaller batch) so the comparison stays honest.
