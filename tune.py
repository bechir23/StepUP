#!/usr/bin/env python
"""Optuna hyperparameter search for gaitcnn_snr, pruned by the per-epoch validation fitness.

Wraps the REAL train() (so every flag means exactly what it means in train.py) and MAXIMISES
best["val"] -- the composite validation fitness (0.90*cross_mAP + 0.07*val_auc + 0.03*(1-unseen_eer)).
A Hyperband/ASHA pruner reports the per-epoch fitness through the engine's epoch_cb and kills weak
trials early (the same "predict from early dynamics, then stop" idea as the competition organizers'
GRM search, without a custom regressor). Optuna's study plots (optimization history, hyperparameter
importance, slice, parallel-coordinate, intermediate-values) are written to artifacts/ and, if wandb
is on, logged to one run.

Search space = weight_decay, arc_scale, dropout, label_smooth, K in 4..8, and the mixstyle/dsu/hpp
architecture toggles. FIXED (not searched): lr=1e-4, embed_dim=256, P=256 (the engine clamps P to the
number of training identities).
(margin_warmup_frac and channel widths are NOT searched yet -- the engine hardcodes the margin ramp
and the gaitcnn_snr factory hardcodes widths; both need a 1-line change before they're searchable.)

  python tune.py --trials 40 --search-epochs 60 --pack-device cuda --wandb online
"""
import argparse
import copy
import gc
import os

import optuna
import torch

from stepup.args import add_common_args, apply_smoke
from stepup.config import ARTIFACTS, H, T, W, build_cfg, seed_everything
from stepup.data import build_datasets
from stepup.engine import train
from stepup.models import registry, set_dropout

MODEL = "gaitcnn_snr"


def build_objective(ds, man, args):
    def objective(trial):
        a = copy.deepcopy(args)                              # inherit the REAL run args (data cfg etc.)
        a.epochs = args.search_epochs                        # so the model matches the built dataset
        # ---- searched hyperparameters (all already consumed by build_cfg / the model) ----
        a.lr = 1e-4                                          # FIXED (not searched)
        a.weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)   # HIGH (reg)
        a.arc_scale = trial.suggest_categorical("arc_scale", [8, 16, 24, 32, 48, 64])  # HIGH
        a.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        a.embed_dim = 256                                    # FIXED (not searched)
        a.label_smooth = trial.suggest_float("label_smooth", 0.0, 0.25)
        P = 256                                              # FIXED (engine clamps to #train IDs)
        K = trial.suggest_int("K", 4, 8)                     # steps per identity, 4..8
        toggles = dict(mixstyle=trial.suggest_categorical("mixstyle", [True, False]),
                       dsu=trial.suggest_categorical("dsu", [True, False]),
                       hpp=trial.suggest_categorical("hpp", [True, False]))   # the discrete arch search
        cfg = build_cfg(a)
        set_dropout(cfg["dropout"])
        seed_everything()
        data_t = (cfg["sample3d"] or cfg["pack_res"] or (T, H, W))[0]
        if cfg.get("stride_pairs"):
            data_t *= 2
        spec = registry(cfg["sample3d"], data_t)[MODEL]
        cfg["lr_mult"] = spec.get("lr_mult", 1.0)
        mkw = dict(spec["kw"]); mkw.update(toggles)          # gaitcnn_snr honours mixstyle/dsu/hpp
        steps = cfg["steps_per_epoch"] or max(1, len(man["train"]) // (P * K))

        def cb(ep, fit):
            trial.report(fit, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            _, _, best = train(spec["fn"], man["train"], cfg, tag=f"tune_t{trial.number}",
                               max_epochs=cfg["epochs"], patience=cfg["patience"],
                               steps_per_epoch=steps, P=P, K=K, model_kw=mkw,
                               ds_tr=ds["train"], ds_va=ds["val_mon"],
                               ds_tr_mon=ds.get("train_mon"), mining=cfg["mining"], epoch_cb=cb)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return best["val"]
    return objective


def save_plots(study, run=None):
    """Write the Optuna study plots to artifacts/ and (optionally) log them to the wandb run."""
    try:
        import optuna.visualization.matplotlib as vis
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (optuna plots skipped: {e})", flush=True)
        return
    plots = {"opt_history": vis.plot_optimization_history,
             "param_importance": vis.plot_param_importances,
             "parallel_coordinate": vis.plot_parallel_coordinate,
             "slice": vis.plot_slice,
             "intermediate_values": vis.plot_intermediate_values}
    for key, fn in plots.items():
        try:
            res = fn(study)
            ax0 = res.ravel()[0] if hasattr(res, "ravel") else res     # plot_slice returns an array
            fig = ax0.figure
            p = ARTIFACTS / f"tune_{key}.png"
            fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
            if run is not None:
                import wandb
                run.log({f"tune/{key}": wandb.Image(str(p))})
            print(f"  plot -> {p.name}", flush=True)
        except Exception as e:
            print(f"  ({key} skipped: {e})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--search-epochs", type=int, default=60,
                    help="epochs per trial during search (reduced); retrain the winner at full epochs")
    ap.add_argument("--study-name", default="gaitcnn_snr")
    args = apply_smoke(ap.parse_args())
    optuna.logging.set_verbosity(optuna.logging.WARNING)     # silence per-trial INFO (log_trial covers it)

    cfg0 = build_cfg(args)
    man, ds = build_datasets(cfg0)                           # build the pack ONCE; reuse every trial
    print(f"search: {args.trials} trials x <= {args.search_epochs} ep  "
          f"| train {len(man['train']):,} steps / {man['train'].ParticipantID.nunique()} ids", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min(8, max(1, args.search_epochs // 2)),   # stays < max_resource on tiny runs
            max_resource=args.search_epochs, reduction_factor=3),
        study_name=args.study_name,
        storage=f"sqlite:///{ARTIFACTS / (args.study_name + '.db')}", load_if_exists=True)

    run = None
    if args.wandb != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=f"tune_{args.study_name}", mode=args.wandb, config=vars(args))

    def log_trial(study, trial):                             # per-trial summary: state + combination
        v = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        best = f"{study.best_value:.4f}" if study.best_trial is not None else "n/a"
        combo = "  ".join(f"{k}={round(x, 5) if isinstance(x, float) else x}"
                          for k, x in trial.params.items())         # the sampled hyperparameters
        print(f"[trial {trial.number:>3}/{args.trials}]  {trial.state.name:8s}  fit={v}  best={best}\n"
              f"        {combo}", flush=True)                        # PRUNED trials also show their combo
        if run is not None:
            run.log({"trial": trial.number,
                     "trial_value": (trial.value if trial.value is not None else float("nan")),
                     "best_value": (study.best_value if study.best_trial is not None else float("nan"))})

    objective = build_objective(ds, man, args)
    study.optimize(objective, n_trials=args.trials, callbacks=[log_trial],   # log_trial = clean per-trial
                   show_progress_bar=os.environ.get("STEPUP_PROGRESS") == "1")   # bar off by default

    print("\n=== BEST ===")
    print(f"  value : {study.best_value:.4f}")
    print(f"  params: {study.best_params}")
    flags = []                                               # reproduce the winner FAITHFULLY, incl. toggles
    for k, v in study.best_params.items():
        if isinstance(v, bool):
            if k == "mixstyle":
                flags.append("" if v else "--no-mixstyle")   # mixstyle DEFAULTS True -> must disable explicitly
            else:
                flags.append(f"--{k}" if v else "")          # dsu / hpp default False -> only add when True
        else:
            flags.append(f"--{k.replace('_', '-')} {v}")
    cmd = f"python train.py --model {MODEL} " + " ".join(f for f in flags if f)
    print(f"  retrain the winner at full epochs:\n    {cmd}")
    save_plots(study, run)
    if run is not None:
        run.summary["best_value"] = study.best_value
        for k, v in study.best_params.items():
            run.summary[f"best_{k}"] = v
        run.finish()


if __name__ == "__main__":
    main()
