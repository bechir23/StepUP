"""Training loop: BNNeck ID+triplet loss, PK sampling, warmup+cosine, early stop on cross EER."""
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import SEED, dev, seed_everything
from .data import FootstepData, FootwearSpanningSampler, PKSampler
from .eval import (condition_verification, embed_dataset, leave_one_footwear_out,
                   open_set_accumulated, summarise)
from .losses import MINERS, Criterion


class ModelEMA:
    """Exponential moving average of the model weights (ultralytics/YOLO). A shadow copy is nudged
    toward the live weights every optimizer step; evaluating on it is smoother and generalizes
    better -- it damps the late-training memorization that makes the raw model's val decline. This
    is the main training-loop difference vs a plain loop and usually the single biggest val gain."""

    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))   # ramps 0 -> decay early on

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def train(model_fn, man_tr, cfg, tag, max_epochs=40, patience=8, steps_per_epoch=40,
          monitor="cross_eer", P=None, K=None, model_kw=None, ds_va=None, ds_tr=None,
          mining="standard", wandb_run=None):
    """Train one backbone; validate leave-one-footwear-out; early-stop on cross EER.
    Returns (best-checkpoint net, per-epoch history df, best record)."""
    seed_everything(SEED)
    P, K = P or cfg["P"], K or cfg["K"]
    n_ids = man_tr.ParticipantID.nunique()
    P = min(P, n_ids)                                         # can't sample more ids than exist
    net = model_fn(embed_dim=cfg["embed_dim"], n_classes=None, **(model_kw or {})).to(dev)
    ema = ModelEMA(net)                                  # YOLO-style weight EMA; val runs on this
    crit = Criterion(cfg, n_ids, cfg["embed_dim"]).to(dev)
    mine = MINERS[mining]
    make_sampler = FootwearSpanningSampler if mining == "crossfw" else PKSampler
    # scale the LR to our batch (sqrt rule, suited to AdamW): cfg["lr"] is tuned for batch 128,
    # our 2D batch is 512 and 3D 256, so a bigger batch gets a proportionally bigger LR.
    # LR scaled to batch (sqrt rule), then times a per-model factor: the heavy 3D nets (r2plus1d/
    # r3d/swin3d/vit) overfit fast and want a much lower LR than the 2D nets -- without this the
    # default lands at ~1.4e-3 for r2plus1d, which memorizes in ~10 epochs and craters val.
    eff_lr = cfg["lr"] * (P * K / 128) ** 0.5 * cfg.get("lr_mult", 1.0)
    clip_params = list(net.parameters()) + list(crit.parameters())   # for gradient clipping
    opt = torch.optim.AdamW(clip_params, lr=eff_lr, weight_decay=cfg.get("weight_decay", 5e-4))
    # YOLO-style per-iteration schedule (ultralytics): warm up over max(frac*total, 100) iters
    # (a floor of 100 so short runs still warm up in STEPS, not a wasted epoch), LR ramping 0->target
    # and AdamW beta1 ramping warmup_mom->0.9; then cosine-decay to lr*lrf (NOT 0 -- a nonzero floor
    # keeps the tail stable instead of freezing the model). Set per batch in the loop below.
    total_steps = max(1, max_epochs * steps_per_epoch)
    nw = max(round(cfg.get("warmup_frac", 0.03) * total_steps), 100)
    lrf = cfg.get("lrf", 0.05)                            # final LR = eff_lr * lrf
    warmup_mom = 0.85
    beta2 = opt.param_groups[0]["betas"][1]

    def lr_cosine(it):                                   # eff_lr -> eff_lr*lrf over all steps
        p = min(1.0, it / total_steps)
        return eff_lr * (((1 + math.cos(math.pi * p)) / 2) * (1 - lrf) + lrf)

    def set_schedule(it):                                # YOLO warmup + cosine, per iteration
        target = lr_cosine(it)
        if it < nw:
            lr_now = float(np.interp(it, [0, nw], [0.0, target]))
            mom = float(np.interp(it, [0, nw], [warmup_mom, 0.9]))
        else:
            lr_now, mom = target, 0.9
        for g in opt.param_groups:
            g["lr"] = lr_now
            g["betas"] = (mom, beta2)
        return lr_now
    use_amp = cfg["amp"] and dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if ds_tr is None:
        ds_tr = FootstepData(man_tr, in_memory=True, augment=cfg.get("augment", False))
    if ds_va is None:
        ds_va = ds_tr

    sampler = make_sampler(ds_tr.m, P=P, K=K, batches=steps_per_epoch, seed=SEED)
    on_gpu = getattr(ds_tr, "device", "cpu") == "cuda"
    if on_gpu:
        loader = None
        def epoch_batches():
            for idx in sampler:
                yield ds_tr.gather(idx)
    else:
        dl_kw = dict(batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=(dev == "cuda"))
        if cfg["workers"] > 0:
            dl_kw.update(persistent_workers=True, prefetch_factor=4)
        loader = DataLoader(ds_tr, **dl_kw)
        def epoch_batches():
            for xb, yb, fwb in loader:
                yield xb.to(dev, non_blocking=True), yb.to(dev), fwb.to(dev)

    # 0 = auto: ~20 logs per epoch, so the step-based curve has the same density at any batch
    log_every = cfg.get("log_every", 0) or max(1, steps_per_epoch // 20)
    if wandb_run is not None:                                # step-based x-axis on wandb
        wandb_run.define_metric("step")
        wandb_run.define_metric("*", step_metric="step")

    # fixed ~5-epoch angular-margin ramp (the reference schedule): starting the margin near 0
    # keeps epoch-1 loss at the healthy CE floor ~ln(n_ids) and lets the embedding spread before
    # the hard margin bites. A fraction-of-epochs ramp is too fast on short runs (margin already
    # ~half-target at epoch 1), which inflates the early loss and hurts convergence.
    margin_warmup = min(max_epochs, 5)
    patience = patience if patience and patience > 0 else float("inf")   # YOLO: 0 -> train all epochs
    # Early-stop robustness (small 15-id val gallery -> noisy per-epoch EER). Two guards so a run
    # is stopped only when it has *genuinely* stopped improving, not on a single unlucky epoch:
    #  (1) a min-epoch floor: never stop before the margin has finished ramping and the schedule
    #      has come off warmup -- stopping in that volatile window was the "it always early-stops"
    #      symptom; (2) the stop/best decision is on a 3-epoch trailing mean of the fitness, so one
    #      noisy dip doesn't reset the model or one noisy spike doesn't lock in a fluke best.
    min_stop_epoch = max(margin_warmup + 5, max_epochs // 4)
    fit_window = []
    best = dict(val=-float("inf"), state=None, epoch=-1)   # maximise the (smoothed) fitness
    hist, bad, gstep = [], 0, 0
    win = dict(loss=[], id=[], tri=[])                       # window since the last step-log
    for epoch in range(max_epochs):
        crit.set_margin_frac((epoch + 1) / margin_warmup)   # ArcFace margin ramp (no-op for CE)
        net.train(); crit.train()
        ep_loss, ep_id, ep_tri, ep_gn = [], [], [], []
        steps = tqdm(epoch_batches(), total=steps_per_epoch, leave=False,
                     desc=f"{tag} ep {epoch + 1}/{max_epochs}")
        for xb, yb, fwb in steps:
            set_schedule(gstep)                              # YOLO-style per-iteration LR + momentum
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=use_amp):
                f_t, f_i, _ = net(xb)
                loss, l_id, l_tri = crit(f_t, f_i, yb, mine(f_t, yb, fwb), fwb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                         # unscale, then clip gradients (YOLO: 10.0)
            # capture the pre-clip total gradient norm: this is the direct read on whether the
            # model is still learning. If it collapses toward 0 while val is flat, the objective
            # has saturated (the model has memorised the training identities) and no amount of
            # further epochs will help -- that is the "climbs then blocks" signature.
            gn = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=10.0)
            ep_gn.append(float(gn))
            scaler.step(opt); scaler.update()
            ema.update(net)                              # nudge the EMA weights toward the live net
            gstep += 1
            lv = loss.item()
            ep_loss.append(lv); ep_id.append(l_id); ep_tri.append(l_tri)
            win["loss"].append(lv); win["id"].append(l_id); win["tri"].append(l_tri)
            steps.set_postfix(loss=f"{lv:.2f}", id=f"{l_id:.2f}", tri=f"{l_tri:.2f}")
            if gstep % log_every == 0:                       # dense step-based train curve
                r = dict(step=gstep, epoch=epoch, train_loss=float(np.mean(win["loss"])),
                         id_loss=float(np.mean(win["id"])), triplet_loss=float(np.mean(win["tri"])))
                hist.append(r)
                if wandb_run is not None:
                    wandb_run.log(r)
                win = dict(loss=[], id=[], tri=[])

        lr = opt.param_groups[0]["lr"]                        # LR now stepped per batch, not here
        eval_net = ema.ema                                   # validate on the smoother EMA weights
        fyf = embed_dataset(eval_net, ds_va)                 # embed val once, reuse for both
        s = summarise(leave_one_footwear_out(eval_net, ds_va, precomp=fyf))   # cross-footwear (hard)
        # repeats=5 (was 2): mixed5 averages over random enrollment draws, so few draws make the
        # per-epoch number bounce +-0.03-0.04 purely from sampling. More draws = a materially
        # smoother curve at negligible cost (the embeddings are already computed), which stops
        # noise from being mistaken for the model oscillating.
        mixed5 = open_set_accumulated(eval_net, ds_va, ks=(5,), repeats=5, precomp=fyf,
                                      score_norm="znorm").get(5, float("nan"))   # cohort-normalized
        s["mixed_r5"] = mixed5                               # reference-protocol 5-step rank-1
        cond = condition_verification(eval_net, ds_va, precomp=fyf)   # competition seen/unseen EER
        for c in ("seen", "unseen"):
            if cond.get(c):
                s[f"{c}_eer"] = cond[c]["eer"]; s[f"{c}_fmr100"] = cond[c]["fmr100"]
        del fyf                                              # free the per-epoch val embeddings
        import gc; gc.collect()                              # reclaim RAM between epochs (low-RAM cards)
        if dev == "cuda":
            torch.cuda.empty_cache()
        # Composite validation FITNESS (higher = better), the YOLO idea: don't early-stop on one
        # noisy metric (cross_eer plateaus early while the model still improves elsewhere). Weight
        # the generalization signals that matter -- rank-1, unseen-footwear EER, cross-footwear EER.
        fitness = (0.40 * mixed5
                   + 0.35 * (1 - s.get("unseen_eer", 0.5))   # unseen footwear -- the hard target
                   + 0.25 * (1 - s.get("cross_eer", 0.5)))   # cross-footwear invariance
        s["fitness"] = fitness
        grad_norm = float(np.mean(ep_gn)) if ep_gn else float("nan")
        er = dict(step=gstep, epoch=epoch, lr=lr, train_loss=float(np.mean(ep_loss)),
                  id_loss=float(np.mean(ep_id)), triplet_loss=float(np.mean(ep_tri)),
                  grad_norm=grad_norm, **s)
        hist.append(er)
        if wandb_run is not None:
            wandb_run.log({"step": gstep, "epoch": epoch, "lr": lr, **s})
        print(f"{tag} ep {epoch + 1:>3}/{max_epochs} step {gstep}  loss {er['train_loss']:6.3f}  "
              f"id {er['id_loss']:6.3f}  tri {er['triplet_loss']:5.3f}  lr {lr:.2e}  "
              f"val_eer {s.get('cross_eer', float('nan')):.3f}  "
              f"val_r1(cross) {s.get('cross_rank1', float('nan')):.3f}  "
              f"val_r1(mixed5) {mixed5:.3f}  "
              f"EER(seen/unseen) {s.get('seen_eer', float('nan'))*100:.1f}/"
              f"{s.get('unseen_eer', float('nan'))*100:.1f}  gn {grad_norm:.2f}  "
              f"fit {fitness:.3f}", flush=True)

        fit_window.append(fitness)                           # smoothed fitness = mean of last 3
        smooth = float(np.mean(fit_window[-3:]))
        if smooth > best["val"] + 1e-4:                      # maximise the smoothed fitness
            best.update(val=smooth, epoch=epoch,             # save the EMA weights (what we eval)
                        state={k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience and epoch + 1 >= min_stop_epoch:   # floor: give every run a fair shot
                print(f"  early stop: no improvement in {bad} epochs (best fit {best['val']:.3f} "
                      f"@ ep {best['epoch']+1})", flush=True)
                break

    del loader
    if best["state"] is not None:
        ema.ema.load_state_dict(best["state"])
    if dev == "cuda":
        torch.cuda.empty_cache()
    return ema.ema, pd.DataFrame(hist), best             # return the EMA model as the trained net
