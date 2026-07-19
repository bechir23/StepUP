"""Training loop: BNNeck ID+triplet loss, PK sampling, warmup+cosine, early stop on cross EER."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import SEED, dev, seed_everything
from .data import FootstepData, FootwearSpanningSampler, PKSampler
from .eval import leave_one_footwear_out, summarise
from .losses import MINERS, Criterion


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
    crit = Criterion(cfg, n_ids, cfg["embed_dim"]).to(dev)
    mine = MINERS[mining]
    make_sampler = FootwearSpanningSampler if mining == "crossfw" else PKSampler
    opt = torch.optim.AdamW(list(net.parameters()) + list(crit.parameters()),
                            lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 5e-4))
    warmup = max(1, round(cfg.get("warmup_frac", 0.1) * max_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, milestones=[warmup],
        schedulers=[torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs - warmup))])
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

    best = dict(val=float("inf"), state=None, epoch=-1)
    hist, bad = [], 0
    for epoch in range(max_epochs):
        net.train(); crit.train()
        ep_loss, ep_id, ep_tri = [], [], []
        steps = tqdm(epoch_batches(), total=steps_per_epoch, leave=False,
                     desc=f"{tag} ep {epoch + 1}/{max_epochs}")
        for xb, yb, fwb in steps:
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=use_amp):
                f_t, f_i, _ = net(xb)
                loss, l_id, l_tri = crit(f_t, f_i, yb, mine(f_t, yb, fwb))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss.append(loss.item()); ep_id.append(l_id); ep_tri.append(l_tri)
            steps.set_postfix(loss=f"{loss.item():.2f}")

        lr = opt.param_groups[0]["lr"]; sched.step()
        s = summarise(leave_one_footwear_out(net, ds_va))
        val = s.get(monitor, float("inf"))
        row = dict(epoch=epoch, train_loss=float(np.mean(ep_loss)), id_loss=float(np.mean(ep_id)),
                   triplet_loss=float(np.mean(ep_tri)), lr=lr, **s)
        hist.append(row)
        if wandb_run is not None:
            wandb_run.log(row)
        print(f"{tag} ep {epoch + 1:>3}/{max_epochs}  loss {row['train_loss']:6.3f}  "
              f"id {row['id_loss']:6.3f}  tri {row['triplet_loss']:5.3f}  lr {lr:.2e}  "
              f"val_eer {s.get('cross_eer', float('nan')):.3f}  "
              f"val_r1 {s.get('cross_rank1', float('nan')):.3f}", flush=True)

        if val < best["val"] - 1e-4:
            best.update(val=val, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    del loader
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    if dev == "cuda":
        torch.cuda.empty_cache()
    return net, pd.DataFrame(hist), best
