"""Stage 4 --- RL active-sensing: an early-classification HALTING POLICY over footsteps.

Given a probe walk (consecutive footsteps of one identity in one shoe), decide AT EACH footstep
whether to HALT and identify or WAIT for another step, to reach the same accuracy in the FEWEST
footsteps. The policy is trained by REINFORCE (EARLIEST, Hartvigsen et al. KDD'19) on top of the
FROZEN Stage-1 embedding (optionally pooled by the FROZEN Stage-2 aggregator); nothing in the
backbone is retrained. This is a *runtime* halting policy -- distinct from the competition winner's
Q-learning, which searched hyperparameters/architectures offline.

Baselines on the SAME accuracy-vs-#footsteps axes (the references' comparison):
  * fixed-k         -- halt at a fixed k (this is Stage 2's k-curve, the "fixed-halting" baseline)
  * confidence/SPRT -- halt when top-1 similarity crosses a threshold (Wald's SPRT idea, a
                       threshold on the sequential evidence), swept over thresholds
Metrics (SPN, Zhang'21; EARLIEST): accuracy, earliness = avg #footsteps / max, and their
Harmonic Mean HM = 2(1-earliness)*acc / ((1-earliness)+acc). The trade-off is swept by lambda.

  python rl_agent.py --model gaitcnn_snr --hf-repo Bechir23/stepup-footstep --max-steps 10 \
      --episodes 40 --lambda 0.02 --pack-device cuda --wandb online
  # add --agg-ckpt agg_gaitcnn_snr_pma_K3.pt to pool with the Stage-2 aggregator instead of mean
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stepup.config import ARTIFACTS, DEF_KS, DEF_WALK_K, FOOTWEAR, T, dev
from stepup.data import build_datasets
from stepup.eval import embed_dataset
from stepup.metrics import enroll_templates
from stepup.models import registry, set_dropout


# ---------------- frozen Stage-1 backbone (same loader as aggregate.py / submit.py) --------------
def load_backbone(model, ckpt, hf_repo, hf_token, in_frames=0, tag=""):
    fname = f"{model}_{tag}_best.pt" if tag else f"{model}_best.pt"
    if not os.path.exists(ckpt or ""):
        ckpt = str(ARTIFACTS / fname)
    if not os.path.exists(ckpt):
        if not hf_repo:
            raise FileNotFoundError(
                f"backbone checkpoint '{fname}' not found in {ARTIFACTS} and no --hf-repo to fetch it "
                f"from. Train Stage 1 first (saves {fname} after all epochs), or pass --hf-repo.")
        from stepup.hf import fetch_file
        ckpt = fetch_file(hf_repo, fname, hf_token)
        print(f"fetched checkpoint from HF: {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    set_dropout(cfg.get("dropout", 0.0))
    data_t = (cfg.get("sample3d") or cfg.get("pack_res") or (T, T, T))[0]
    if cfg.get("stride_pairs"):
        data_t *= 2
    if in_frames > 0:
        data_t = in_frames
    spec = registry(cfg["sample3d"], data_t)[model]
    net = spec["fn"](embed_dim=cfg["embed_dim"], n_classes=None, **spec["kw"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, cfg


def load_aggregator(agg_ckpt, hf_repo, hf_token, d):
    """Optional frozen Stage-2 pooling head; returns None -> caller uses running mean."""
    if not agg_ckpt:
        return None
    from aggregate import make_aggregator                      # lazy: pulls pytorch_metric_learning
    path = agg_ckpt if os.path.exists(agg_ckpt) else str(ARTIFACTS / agg_ckpt)
    if not os.path.exists(path):
        from stepup.hf import fetch_file
        path = fetch_file(hf_repo, os.path.basename(agg_ckpt), hf_token)
    ck = torch.load(path, map_location=dev, weights_only=False)
    agg = make_aggregator(ck.get("aggregator", "pma"), d).to(dev)
    agg.load_state_dict(ck["state"]); agg.eval()
    print(f"loaded Stage-2 aggregator: {ck.get('aggregator')} K{ck.get('sub_centers')} "
          f"(best epoch {ck.get('best_epoch')})", flush=True)
    return agg


# ---------------- build the running-evidence tensors (precompute so rollout is cheap) ------------
@torch.no_grad()
def running_states(f, m, max_k, agg=None):
    """For every cross-footwear probe walk, precompute per-step (k=1..max_k) the identification
    STATE the policy sees and whether the running top-1 prediction is CORRECT. Enrollment templates
    come from each held-out footwear in turn (leave-one-footwear-out, the hard cross-shoe setting).

    Returns dict of tensors, all shape [N_walks, max_k, *]:
      state  [N, max_k, 5] = (top1_sim, margin=top1-top2, mean_sim, std_sim, k/max)
      top1   [N, max_k]     running top-1 similarity (for the SPRT/confidence baseline)
      correct[N, max_k]     running top-1 prediction is the true id (0/1)
      valid  [N, max_k]     this step exists in the walk (walks shorter than max_k are masked)
    """
    f = F.normalize(f)
    m = m.reset_index(drop=True)
    fw = m.Footwear.to_numpy(); pid = m.ParticipantID.to_numpy()
    passid = m.PassID.to_numpy(); stepid = m.FootstepID.to_numpy()
    S, TOP1, COR, VAL = [], [], [], []
    for enrol in FOOTWEAR:
        g = fw == enrol
        if g.sum() == 0 or (fw != enrol).sum() == 0:
            continue
        templ, ids = enroll_templates(f[g], torch.as_tensor(pid[g]))
        passes = {}
        for i in np.where(fw != enrol)[0]:
            passes.setdefault((pid[i], fw[i], passid[i]), []).append(int(i))
        for (p, _, _), rows in passes.items():
            rows = sorted(rows, key=lambda i: stepid[i])
            st = torch.zeros(max_k, 5); t1 = torch.zeros(max_k)
            cor = torch.zeros(max_k); val = torch.zeros(max_k)
            for k in range(min(len(rows), max_k)):
                win = rows[:k + 1]
                if agg is None:
                    emb = F.normalize(f[win].mean(0), dim=0)
                else:
                    emb = agg(f[win].unsqueeze(0).to(dev)).squeeze(0).cpu()
                sims = (emb @ templ.t())
                top = torch.topk(sims, min(2, sims.numel())).values
                top1 = top[0].item(); top2 = top[1].item() if top.numel() > 1 else 0.0
                st[k] = torch.tensor([top1, top1 - top2, sims.mean().item(),
                                      sims.std().item(), (k + 1) / max_k])
                t1[k] = top1; val[k] = 1.0
                cor[k] = float(ids[sims.argmax()].item() == p)
            S.append(st); TOP1.append(t1); COR.append(cor); VAL.append(val)
    return dict(state=torch.stack(S), top1=torch.stack(TOP1),
                correct=torch.stack(COR), valid=torch.stack(VAL))


# ---------------- the halting policy (EARLIEST controller) ----------------
class HaltPolicy(nn.Module):
    def __init__(self, d_state=5, hidden=32, halt_bias=-1.0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_state, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.halt = nn.Linear(hidden, 1)        # P(halt now)
        self.value = nn.Linear(hidden, 1)       # baseline for REINFORCE variance reduction
        nn.init.constant_(self.halt.bias, halt_bias)  # <0 starts by WAITING (explores full walks so
        #   REINFORCE doesn't collapse to halt-at-1); but too negative + a long warmup makes it
        #   OVER-wait. -1.0 (p_halt~0.27) balances both; tune with --halt-bias.

    def forward(self, s):
        h = self.body(s)
        return torch.sigmoid(self.halt(h)).squeeze(-1), self.value(h).squeeze(-1)


def rollout(policy, batch, lam, sample=True):
    """One REINFORCE rollout over a batch of walks. Returns (loss, accuracy, avg_steps, mean_reward).
    Steps left-to-right; each still-running walk halts when it samples 'halt' (greedy: p>0.5) or
    hits its last valid step. Reward = correct(+1/0) - lambda * steps_used."""
    state, correct, valid = batch["state"].to(dev), batch["correct"].to(dev), batch["valid"].to(dev)
    B, Lmax, _ = state.shape
    p_halt, value = policy(state.reshape(B * Lmax, -1))
    p_halt = p_halt.reshape(B, Lmax).clamp(1e-6, 1 - 1e-6)
    value = value.reshape(B, Lmax)
    last = valid.long().sum(1) - 1                                   # index of each walk's last step
    active = torch.ones(B, dtype=torch.bool, device=dev)
    logp = torch.zeros(B, device=dev); K = last.clone()
    for k in range(Lmax):
        on = active & (valid[:, k] > 0)
        if not on.any():
            break
        ph = p_halt[:, k]
        if sample:
            halt_now = (torch.bernoulli(ph) > 0.5)
        else:
            halt_now = (ph > 0.5)
        force = (k == last)                                          # must halt at the last step
        do_halt = on & (halt_now | force)
        logp = logp + on.float() * torch.where(do_halt, torch.log(ph), torch.log(1 - ph))
        K = torch.where(do_halt & active, torch.full_like(K, k), K)
        active = active & ~do_halt
    idx = torch.arange(B, device=dev)
    steps = (K + 1).float()
    hit = correct[idx, K]
    reward = hit - lam * steps
    baseline = value[idx, K]
    adv = (reward - baseline).detach()
    pg = -(logp * adv).mean()
    vloss = F.smooth_l1_loss(baseline, reward.detach())
    ent = -(p_halt * torch.log(p_halt) + (1 - p_halt) * torch.log(1 - p_halt))
    ent = (ent * valid).sum() / valid.sum()
    loss = pg + 0.5 * vloss - 0.01 * ent
    return loss, hit.mean().item(), steps.mean().item(), reward.mean().item()


# ---------------- baselines ----------------
def fixed_k_curve(batch, ks):
    correct, valid = batch["correct"], batch["valid"]
    last = valid.long().sum(1) - 1
    out = {}
    for k in ks:
        kk = torch.clamp(torch.full_like(last, k - 1), max=last)    # clip to each walk's length
        acc = correct[torch.arange(len(last)), kk].mean().item()
        out[k] = (acc, float(min(k, (last + 1).float().mean().item())))
    return out


def sprt_curve(batch, thresholds):
    """SPRT/confidence baseline: halt at the first step whose running top-1 sim exceeds tau."""
    top1, correct, valid = batch["top1"], batch["correct"], batch["valid"]
    last = valid.long().sum(1) - 1
    B, Lmax = top1.shape
    out = {}
    for tau in thresholds:
        cross = (top1 >= tau) & (valid > 0)
        first = torch.where(cross.any(1), cross.float().argmax(1), last)
        first = torch.minimum(first, last)
        idx = torch.arange(B)
        out[round(float(tau), 3)] = (correct[idx, first].mean().item(), (first + 1).float().mean().item())
    return out


def harmonic_mean(acc, avg_steps, max_steps):
    earliness = avg_steps / max_steps
    a, e = acc, 1 - earliness
    return 2 * a * e / (a + e + 1e-9)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr")
    ap.add_argument("--tag", default="", help="load {model}_{tag}_best.pt (a specific ablation run)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--agg-ckpt", default="", help="optional Stage-2 aggregator .pt (else running mean)")
    ap.add_argument("--in-frames", type=int, default=0)
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"],
                    help="override the checkpoint's pack device (cuda = fast eval on Colab)")
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--hf-offload", action="store_true",
                    help="after pushing to HF, delete the local .pt (keep parquet); use on Colab to "
                         "keep Drive light -- locally omit it so the policy stays, as before")
    ap.add_argument("--max-steps", type=int, default=10, help="longest walk the policy may observe")
    ap.add_argument("--episodes", type=int, default=40, help="training epochs over the walk set")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=0.05,
                    help="earliness penalty (per footstep); higher = halt sooner")
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                    help="fraction of episodes to ramp lambda 0->target (short: just enough to learn "
                         "accuracy before penalizing steps; too long -> over-waits)")
    ap.add_argument("--halt-bias", type=float, default=-1.0,
                    help="init halt-head bias; <0 = start waiting (explore). -1.0 ~ p_halt 0.27")
    ap.add_argument("--ks", default=DEF_KS, help="fixed-k baselines to report")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    args = ap.parse_args()
    wb = None
    if args.wandb != "disabled":
        import wandb as _wb
        rname = f"rl_{args.model}_lam{args.lam}" + (f"_{args.tag}" if args.tag else "")
        wb = _wb.init(project="stepup-footstep", name=rname, mode=args.wandb, config=vars(args))

    net, cfg = load_backbone(args.model, args.ckpt, args.hf_repo, args.hf_token, args.in_frames, args.tag)
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)
    f, y, fw = embed_dataset(net, ds["test"])
    D = f.shape[1]
    agg = load_aggregator(args.agg_ckpt, args.hf_repo, args.hf_token, D)
    print(f"backbone embed_dim={D}  test steps={len(f)}  building running-evidence up to "
          f"k={args.max_steps} ...", flush=True)
    data = running_states(f, ds["test"].m, args.max_steps, agg=agg)
    N = data["state"].shape[0]
    print(f"probe walks: {N}  (cross-footwear, leave-one-footwear-out)", flush=True)

    # split walks 70/30 into policy-train / policy-eval (the backbone never sees these labels)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    ntr = int(0.7 * N); tr, te = perm[:ntr], perm[ntr:]
    sub = lambda keys, ix: {k: data[k][ix] for k in keys}
    train = sub(data.keys(), tr); test = sub(data.keys(), te)

    policy = HaltPolicy(d_state=5, halt_bias=args.halt_bias).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"training halting policy (REINFORCE, lambda={args.lam}, warmup={args.warmup_frac}, "
          f"halt_bias={args.halt_bias}) ...", flush=True)
    warmup = max(1, int(args.episodes * args.warmup_frac))
    for ep in range(args.episodes):
        lam_ep = args.lam * min(1.0, ep / warmup)      # warmup: learn accuracy first, then penalize
        policy.train()
        idx = torch.randperm(ntr)
        tot = 0.0; nb = 0
        for s in range(0, ntr, args.batch):
            b = {k: train[k][idx[s:s + args.batch]] for k in train}
            loss, _, _, _ = rollout(policy, b, lam_ep, sample=True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        policy.eval()
        with torch.no_grad():
            _, acc, steps, rew = rollout(policy, test, args.lam, sample=False)
        hm = harmonic_mean(acc, steps, args.max_steps)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep + 1:>3}/{args.episodes}  loss {tot / nb:6.3f}  "
                  f"acc {acc:.3f}  avg_steps {steps:.2f}  HM {hm:.3f}", flush=True)
        if wb is not None:
            wb.log({"epoch": ep + 1, "loss": tot / nb, "rl_acc": acc, "rl_avg_steps": steps,
                    "rl_hm": hm, "reward": rew})

    # ---- final comparison on the SAME axes: RL vs fixed-k vs SPRT/confidence ----
    policy.eval()
    with torch.no_grad():
        _, rl_acc, rl_steps, _ = rollout(policy, test, args.lam, sample=False)
    rl_hm = harmonic_mean(rl_acc, rl_steps, args.max_steps)
    ks = [int(v) for v in args.ks.split(",")]
    fk = fixed_k_curve(test, ks)
    sp = sprt_curve(test, np.linspace(0.2, 0.8, 7))
    print("\n=== Stage 4: accuracy vs avg #footsteps (lower steps + higher acc = better) ===", flush=True)
    print(f"  RL policy (lambda={args.lam})   acc {rl_acc:.3f}  avg_steps {rl_steps:.2f}  HM {rl_hm:.3f}",
          flush=True)
    print("  fixed-k baseline (= Stage 2 k-curve):", flush=True)
    for k, (a, st) in fk.items():
        print(f"    k={k:<2d}  acc {a:.3f}  steps {st:.2f}  HM {harmonic_mean(a, st, args.max_steps):.3f}",
              flush=True)
    print("  SPRT/confidence-threshold baseline:", flush=True)
    for tau, (a, st) in sp.items():
        print(f"    tau={tau:<4}  acc {a:.3f}  avg_steps {st:.2f}  "
              f"HM {harmonic_mean(a, st, args.max_steps):.3f}", flush=True)
    if wb is not None:
        wb.log({"final_rl_acc": rl_acc, "final_rl_steps": rl_steps, "final_rl_hm": rl_hm,
                **{f"fixedk_acc_k{k}": a for k, (a, _) in fk.items()}})
        wb.finish()
    otag = f"_{args.tag}" if args.tag else ""            # include tag so 12/25-ID runs don't collide
    stem = f"rl_{args.model}{otag}_lam{args.lam}"
    out = ARTIFACTS / f"{stem}.pt"
    torch.save(dict(state=policy.state_dict(), lam=args.lam, max_steps=args.max_steps,
                    rl=(rl_acc, rl_steps, rl_hm), fixed_k=fk, sprt=sp, model=args.model, tag=args.tag),
               out)
    # Python-readable frontier parquet (RL + every fixed-k + every SPRT tau) for analysis/plots
    import pandas as pd
    rows = [{"method": "RL", "param": args.lam, "acc": rl_acc, "avg_steps": rl_steps, "hm": rl_hm}]
    rows += [{"method": "fixed_k", "param": k, "acc": a, "avg_steps": st,
              "hm": harmonic_mean(a, st, args.max_steps)} for k, (a, st) in fk.items()]
    rows += [{"method": "sprt", "param": tau, "acc": a, "avg_steps": st,
              "hm": harmonic_mean(a, st, args.max_steps)} for tau, (a, st) in sp.items()]
    fr_f = ARTIFACTS / f"{stem}.parquet"
    pd.DataFrame(rows).to_parquet(fr_f, index=False)
    print(f"\nsaved policy -> {out.name}   frontier -> {fr_f.name}", flush=True)
    if args.hf_repo:
        from stepup.hf import push_files
        push_files(args.hf_repo, [out, fr_f], args.hf_token)
        print(f"pushed policy + frontier -> https://huggingface.co/{args.hf_repo}", flush=True)
        if args.hf_offload:                              # Colab: drop the heavy .pt locally, keep parquet
            out.unlink(missing_ok=True)
            print(f"  offloaded {out.name} (removed local copy; on HF)", flush=True)


if __name__ == "__main__":
    main()
