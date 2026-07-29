"""Interpretability figures for the RL-learned augmentation policy (Adversarial-AutoAugment,
Zhang ICLR'20; AutoAugment, Cubuk CVPR'19).

Six thesis figures, produced DURING training and logged to Weights & Biases, so the policy is
never a black box -- you can see what it searched and whether it paid off:

  1 aug_panel    the 6 geometric ops applied to a REAL footstep      -> the action space
  2 policy_traj  P(op) per epoch (softmax of op logits)              -> is the policy learning?
  3 entropy      policy entropy per epoch, vs ln(n_ops) ceiling      -> explore -> exploit
  4 mag_heatmap  learned P(magnitude | op)                            -> how hard each op is applied
  5 op_heatmap   op x epoch selection probability (compact traj.)     -> which ops persist
  6 reward_eer   adversary reward (backbone loss) vs downstream EER   -> did the min-max pay off?

Both the standalone rl_augment.py and the integrated engine.py --rl-augment path drive the SAME
RLPolicyLog, so the two produce an identical figure set. Nothing here is illustrative: every curve
is the real per-epoch policy state of the run that emitted it.
"""
import math

import numpy as np


class RLPolicyLog:
    """Accumulates the per-epoch policy state and renders the six figures at the end of a run.

    Call `.update(policy, reward, eer)` once per epoch (it also returns the scalars to log to
    wandb that epoch), then `.render(...)` once after training to write the PNGs + parquet and
    push the images to the wandb run.
    """

    def __init__(self, op_names, eer_label="cross-fw EER"):
        self.op_names = list(op_names)
        self.n_ops = len(op_names)
        self.eer_label = eer_label
        self.op_prob_hist = []      # (epochs, n_ops)  P(op)               -> figures 2 & 5
        self.mag_hist = []          # (epochs, n_ops)  magnitude per op     -> figure 4 / parquet
        self.ent_hist = []          # (epochs,)        policy entropy       -> figure 3
        self.mag_mean_hist = []     # (epochs,)        expected magnitude   -> canonical scalar
        self.reward_hist = []       # (epochs,)        adversary reward     -> figure 6
        self.eer_hist = []          # (epochs,)        downstream EER       -> figure 6

    def update(self, policy, reward, eer):
        """Read the live policy; store P(op)/magnitude/entropy/reward/EER for the end-of-run figures.
        Returns ONLY the three canonical per-epoch RL training scalars for wandb -- the quantities the
        augmentation-policy papers actually track: the adversary REWARD (the augmented-batch loss it
        maximises, Adversarial-AutoAugment), the policy ENTROPY (explore->exploit, REINFORCE), and the
        mean augmentation MAGNITUDE (= difficulty; Adversarial-AutoAugment Fig 2 shows it rising with
        training). The full per-op breakdown is in the figures + parquet, not noisy per-epoch streams."""
        import torch
        with torch.no_grad():
            op_p = torch.softmax(policy.op_logits, 0)
            ent = float(-(op_p * op_p.clamp_min(1e-9).log()).sum())
            mag_p = torch.softmax(policy.mag_logits, 1)                       # P(magnitude | op)
            mean_bin = (mag_p * torch.arange(policy.n_mag, device=mag_p.device)).sum(1)
            op_mag = ((mean_bin + 0.5) / policy.n_mag)                        # magnitude per op in [0,1]
            mean_mag = float((op_p * op_mag).sum())                          # expected applied magnitude
        self.op_prob_hist.append([float(v) for v in op_p.tolist()])
        self.mag_hist.append([float(v) for v in op_mag.tolist()])
        self.ent_hist.append(ent)
        self.mag_mean_hist.append(mean_mag)
        self.reward_hist.append(float(reward))
        self.eer_hist.append(float(eer))
        return {"rl/reward": float(reward), "rl/policy_entropy": ent, "rl/mean_magnitude": mean_mag}

    def render(self, policy, out_dir, prefix, apply_op=None, sample_x=None, wandb_run=None):
        """Write the six figures + a parquet of the series to `out_dir` with `{prefix}_*` names,
        and log every image to `wandb_run`. `apply_op(x, op_idx, mag)` + `sample_x` (one real
        footstep cube) are needed for figure 1; if either is missing figure 1 is skipped."""
        import pathlib
        out_dir = pathlib.Path(out_dir)
        paths = {}
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  (RL figures skipped: matplotlib unavailable: {e})", flush=True)
            return paths
        import torch

        H = np.asarray(self.op_prob_hist)                          # (epochs, n_ops)
        epochs = np.arange(1, len(H) + 1)
        with torch.no_grad():
            mag_p = torch.softmax(policy.mag_logits, 1).cpu().numpy()   # (n_ops, n_mag)
        n_mag = mag_p.shape[1]
        mag_mean = (mag_p * np.arange(n_mag)).sum(1) / n_mag       # learned mean magnitude per op
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_ops))

        def _save(fig, key):
            p = out_dir / f"{prefix}_{key}.png"
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths[key] = p
            if wandb_run is not None:
                try:
                    import wandb
                    wandb_run.log({f"rl/{key}": wandb.Image(str(p))})
                except Exception as e:
                    print(f"  (wandb image {key} skipped: {e})", flush=True)

        # -- Fig 1: the action space -- each op on a REAL footstep, at its LEARNED magnitude ------
        if apply_op is not None and sample_x is not None:
            try:
                x = sample_x.detach().float()
                if x.dim() == 4:
                    x = x.unsqueeze(0)                             # -> (1,1,C,H,W)

                def peak(t):                                       # peak-pressure image (H,W)
                    a = t[0, 0].detach().cpu().numpy()
                    a = a.max(0) if a.ndim == 3 else a
                    lo, hi = float(a.min()), float(a.max())
                    return (a - lo) / (hi - lo + 1e-9)
                panels = [("original", peak(x))]
                for i, n in enumerate(self.op_names):
                    xa = apply_op(x, i, float(mag_mean[i]))
                    panels.append((f"{n}\n(mag {mag_mean[i]:.2f})", peak(xa)))
                cols = len(panels)
                fig, ax = plt.subplots(1, cols, figsize=(2.05 * cols, 2.5))
                for a, (ttl, img) in zip(np.atleast_1d(ax), panels):
                    a.imshow(img, cmap="magma", aspect="auto")
                    a.set_title(ttl, fontsize=8); a.axis("off")
                fig.suptitle("Fig 1  augmentation ops on a real footstep (at learned magnitude)",
                             fontsize=10)
                _save(fig, "1_aug_panel")
            except Exception as e:
                print(f"  (Fig 1 aug_panel skipped: {e})", flush=True)

        # -- Fig 2: policy P(op) trajectory ------------------------------------------------------
        fig, a = plt.subplots(figsize=(6.2, 4))
        for i, n in enumerate(self.op_names):
            a.plot(epochs, H[:, i], label=n, color=colors[i], lw=2)
        a.axhline(1.0 / self.n_ops, ls="--", c="gray", lw=1, label="uniform (no learning)")
        a.set_title("Fig 2  policy P(op) over training"); a.set_xlabel("epoch")
        a.set_ylabel("P(op)"); a.set_ylim(0, 1); a.legend(fontsize=8, ncol=2); a.grid(alpha=.3)
        _save(fig, "2_policy_trajectory")

        # -- Fig 3: policy entropy (explore -> exploit) ------------------------------------------
        fig, a = plt.subplots(figsize=(6.2, 4))
        a.plot(epochs, self.ent_hist, "k-", lw=2)
        a.axhline(math.log(self.n_ops), ls="--", c="tab:green", lw=1,
                  label=f"max = ln({self.n_ops}) = {math.log(self.n_ops):.2f} (uniform)")
        a.set_title("Fig 3  policy entropy  (explore -> exploit)"); a.set_xlabel("epoch")
        a.set_ylabel("H(op)  [nats]"); a.set_ylim(0, math.log(self.n_ops) * 1.08)
        a.legend(fontsize=8); a.grid(alpha=.3)
        _save(fig, "3_entropy")

        # -- Fig 4: magnitude distribution per op ------------------------------------------------
        fig, a = plt.subplots(figsize=(6.2, 4))
        im = a.imshow(mag_p, aspect="auto", cmap="viridis", origin="lower",
                      extent=[0.5, n_mag + 0.5, -0.5, self.n_ops - 0.5])
        a.set_yticks(range(self.n_ops)); a.set_yticklabels(self.op_names)
        a.set_xlabel("magnitude bin (1 = gentle .. %d = strong)" % n_mag)
        a.set_title("Fig 4  P(magnitude | op)")
        fig.colorbar(im, ax=a, fraction=.046, pad=.04, label="probability")
        _save(fig, "4_magnitude_heatmap")

        # -- Fig 5: op x epoch selection heatmap -------------------------------------------------
        fig, a = plt.subplots(figsize=(6.2, 4))
        im = a.imshow(H.T, aspect="auto", cmap="magma", origin="lower",
                      extent=[0.5, len(epochs) + 0.5, -0.5, self.n_ops - 0.5], vmin=0, vmax=1)
        a.set_yticks(range(self.n_ops)); a.set_yticklabels(self.op_names)
        a.set_xlabel("epoch"); a.set_title("Fig 5  op selection P(op) across training")
        fig.colorbar(im, ax=a, fraction=.046, pad=.04, label="P(op)")
        _save(fig, "5_op_heatmap")

        # -- Fig 6: adversary reward vs downstream EER -------------------------------------------
        fig, a1 = plt.subplots(figsize=(6.2, 4))
        a1.plot(epochs, self.reward_hist, color="tab:red", lw=2, label="adversary reward (loss)")
        a1.set_xlabel("epoch"); a1.set_ylabel("backbone loss (adversary maximises)", color="tab:red")
        a1.tick_params(axis="y", labelcolor="tab:red"); a1.grid(alpha=.3)
        a2 = a1.twinx()
        a2.plot(epochs, np.asarray(self.eer_hist) * 100, color="tab:blue", lw=2,
                label=self.eer_label)
        a2.set_ylabel(f"{self.eer_label} (%)", color="tab:blue")
        a2.tick_params(axis="y", labelcolor="tab:blue")
        a1.set_title("Fig 6  adversary reward vs downstream EER")
        _save(fig, "6_reward_eer")

        # -- series parquet (so the figures can be re-plotted / analysed later) -------------------
        try:
            import pandas as pd
            df = pd.DataFrame(H, columns=[f"prob_{n}" for n in self.op_names])
            for i, n in enumerate(self.op_names):
                df[f"mag_{n}"] = np.asarray(self.mag_hist)[:, i]
            df["epoch"] = epochs
            df["entropy"] = self.ent_hist
            df["mean_magnitude"] = self.mag_mean_hist       # Adversarial-AA difficulty-over-training
            df["reward"] = self.reward_hist
            df["eer"] = self.eer_hist
            pq = out_dir / f"{prefix}_policy.parquet"
            df.to_parquet(pq, index=False)
            paths["parquet"] = pq
        except Exception as e:
            print(f"  (RL parquet skipped: {e})", flush=True)

        names = ", ".join(sorted(k for k in paths if k != "parquet"))
        print(f"  RL interpretability figures -> {prefix}_*  [{names}]", flush=True)
        return paths
