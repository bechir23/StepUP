"""Weights & Biases logging. init_run returns a run (or None if --wandb disabled), and the
engine logs each epoch's row to it. Standard, no conditional wrapper at the call site."""


def init_run(args, cfg, name):
    if args.wandb == "disabled":
        return None
    import wandb
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name,
                      mode=args.wandb, reinit=True, config={**cfg, "model": name})
