"""Leakage-safe, deployment-realistic evaluation."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import FOOTWEAR, FOOTWEAR_LABEL, SEED, dev
from .metrics import enroll_templates, identification, report_from_scores, verification


@torch.no_grad()
def embed_dataset(net, ds, batch=256):
    """L2-normalised f_i for every footstep, plus its footwear. GPU-resident packs slice on GPU."""
    net = net.to(dev).eval()
    feats, labs = [], []
    if getattr(ds, "device", "cpu") == "cuda":
        for i in range(0, len(ds), batch):
            xb, yb, _ = ds.gather(range(i, min(i + batch, len(ds))))
            _, f_i, _ = net(xb)
            feats.append(F.normalize(f_i).cpu()); labs.append(yb.cpu())
    else:
        for xb, yb, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
            _, f_i, _ = net(xb.to(dev))
            feats.append(F.normalize(f_i).cpu()); labs.append(yb)
    return torch.cat(feats), torch.cat(labs), ds.m.Footwear.to_numpy()


def leave_one_footwear_out(net, ds, precomp=None):
    """Per enrol->probe shoe pair: rank-1/5, mAP, EER, TAR@1%; same vs cross footwear."""
    f, y, fw = precomp if precomp is not None else embed_dataset(net, ds)
    fw = np.asarray(fw)
    m = ds.m.reset_index(drop=True)
    first_pass = m.groupby("ParticipantID").PassID.transform("min").to_numpy()
    passid = m.PassID.to_numpy()
    rows = []
    for enrol in FOOTWEAR:
        g = fw == enrol
        if g.sum() == 0:
            continue
        for probe in FOOTWEAR:
            p = fw == probe
            if p.sum() == 0:
                continue
            if enrol == probe:
                g_sel = g & (passid == first_pass)
                p_sel = p & (passid != first_pass)
            else:
                g_sel, p_sel = g, p
            if g_sel.sum() == 0 or p_sel.sum() == 0:
                continue
            cmc, mp = identification(f[p_sel], y[p_sel], f[g_sel], y[g_sel])
            eer, tar = verification(f[p_sel], y[p_sel], f[g_sel], y[g_sel])
            rows.append(dict(enrol=enrol, probe=probe, kind="same" if enrol == probe else "cross",
                             rank1=cmc[1], rank5=cmc[5], mAP=mp, eer=eer, tar1=tar,
                             n_probe=int(p_sel.sum())))
    import pandas as pd
    return pd.DataFrame(rows)


def cross_footwear_verification(net, ds):
    """Pooled cross-footwear genuine/impostor scores -> full competition report."""
    f, y, fw = embed_dataset(net, ds)
    fw = np.asarray(fw)
    scores, labels = [], []
    for enrol in FOOTWEAR:
        g = fw == enrol
        p = fw != enrol
        if g.sum() == 0 or p.sum() == 0:
            continue
        templates, ids = enroll_templates(f[g], y[g])
        sim = (F.normalize(f[p]) @ templates.t()).numpy()
        genuine = (ids[None, :] == y[p][:, None]).numpy()
        scores.append(sim.ravel()); labels.append(genuine.ravel())
    return report_from_scores(np.concatenate(scores), np.concatenate(labels))


def accumulated_identification(net, ds, ks=(1, 3, 5, 10)):
    """Cross-footwear rank-1 as a walking pass accumulates: probe = running mean of k
    consecutive footsteps of one (identity, shoe, pass). Rank-1 climbs with k (deployment)."""
    f, y, fw = embed_dataset(net, ds)
    f = F.normalize(f)
    fw = np.asarray(fw)
    m = ds.m.reset_index(drop=True)
    pid = m.ParticipantID.to_numpy(); passid = m.PassID.to_numpy(); stepid = m.FootstepID.to_numpy()
    per_k = {k: [] for k in ks}
    for enrol in FOOTWEAR:
        g = fw == enrol
        if g.sum() == 0:
            continue
        templates, ids = enroll_templates(f[g], y[g])
        probe_rows = np.where(fw != enrol)[0]
        passes = {}
        for i in probe_rows:
            passes.setdefault((pid[i], fw[i], passid[i]), []).append(int(i))
        for k in ks:
            hit = tot = 0
            for _, rows in passes.items():
                rows = sorted(rows, key=lambda i: stepid[i])
                for s in range(0, len(rows) - k + 1, k):
                    win = rows[s:s + k]
                    probe = F.normalize(f[win].mean(0), dim=0)
                    pred = ids[(probe @ templates.t()).argmax()]
                    hit += int(pred.item() == int(y[win[0]])); tot += 1
            per_k[k].append(hit / tot if tot else float("nan"))
    return {k: float(np.nanmean(v)) for k, v in per_k.items()}


def open_set_accumulated(net, ds, n_enroll=5, ks=(1, 3, 5, 10), repeats=3, precomp=None):
    """Reference open-set protocol (mixed-footwear gallery, NOT cross-footwear): enroll
    n_enroll random steps per identity (any footwear) into one template, then probe = running
    mean of k consecutive steps within one (identity, footwear, speed) trial; rank-1 per k,
    averaged over random enroll draws. This mirrors the reference eval that reaches ~0.9 at
    k=5-10, so it is the fair comparison to that number -- contrast with
    accumulated_identification, which is strict leave-one-footwear-out (much harder)."""
    f, y, fw = precomp if precomp is not None else embed_dataset(net, ds)
    f = F.normalize(f)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    m = ds.m.reset_index(drop=True)
    speed = m.Speed.to_numpy() if "Speed" in m else np.zeros(len(m), int)
    passid = m.PassID.to_numpy(); stepid = m.FootstepID.to_numpy(); fwc = np.asarray(fw)
    ids = np.unique(y)
    per_k = {k: [] for k in ks}
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        templates, tids, enrolled = [], [], np.zeros(len(y), bool)
        for i in ids:
            idx = np.where(y == i)[0]; rng.shuffle(idx)
            enrolled[idx[:n_enroll]] = True                    # mixed-footwear enroll steps
            templates.append(F.normalize(f[idx[:n_enroll]].mean(0), dim=0)); tids.append(i)
        templ = torch.stack(templates); tids = np.array(tids)
        passes = {}
        for j in range(len(y)):
            if not enrolled[j]:
                passes.setdefault((y[j], fwc[j], speed[j]), []).append(j)
        for k in ks:
            hit = tot = 0
            for _, rows in passes.items():
                rows = sorted(rows, key=lambda r: (passid[r], stepid[r]))
                for s in range(0, len(rows) - k + 1, k):
                    win = rows[s:s + k]
                    probe = F.normalize(f[win].mean(0), dim=0)
                    pred = tids[(probe @ templ.t()).argmax()]
                    hit += int(pred == y[win[0]]); tot += 1
            per_k[k].append(hit / tot if tot else float("nan"))
    return {k: float(np.nanmean(v)) for k, v in per_k.items()}


def summarise(df):
    out = {}
    for kind in ("same", "cross"):
        sub = df[df.kind == kind]
        if len(sub):
            out[f"{kind}_rank1"] = float(np.nanmean(sub.rank1))
            out[f"{kind}_eer"] = float(np.nanmean(sub.eer))
    return out


def plot_history(hist_df, title, out_path):
    """Step-based curves: train losses (total/id/triplet) logged every N steps -- a dense
    curve even with few epochs -- and val cross-EER / rank-1 at each epoch's step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = hist_df.copy()
    xcol = "step" if "step" in h else "epoch"
    if "cross_eer" in h:
        tr = h[h["cross_eer"].isna()]                 # dense step-granular train rows
        ep = h[h["cross_eer"].notna()]                # per-epoch val rows
        if len(tr) == 0:                              # e.g. epochs shorter than log_every
            tr = h
    else:
        tr, ep = h, h
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(tr[xcol], tr.train_loss, label="total")
    ax[0].plot(tr[xcol], tr.id_loss, label="id")
    ax[0].plot(tr[xcol], tr.triplet_loss, label="triplet")
    ax[0].set_title("train loss"); ax[0].legend(fontsize=7)
    if "cross_eer" in ep:
        ax[1].plot(ep[xcol], ep.cross_eer, marker="o", ms=3); ax[1].set_ylim(0, 0.6)
    ax[1].set_title("val cross EER (lower=better)")
    if "cross_rank1" in ep:
        ax[2].plot(ep[xcol], ep.cross_rank1, marker="o", ms=3); ax[2].set_ylim(0, 1)
    ax[2].set_title("val cross rank-1")
    for a in ax:
        a.set_xlabel(xcol); a.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def project_2d(feats):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                         random_state=SEED).fit_transform(feats)
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine", init="pca",
                    perplexity=30, random_state=SEED).fit_transform(feats)


def plot_embeddings(net, ds, title, out_path, max_pts=2500):
    """2D projection coloured by identity and by footwear; saved to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, y, fw = embed_dataset(net, ds)
    if len(f) > max_pts:
        idx = np.random.default_rng(SEED).choice(len(f), max_pts, replace=False)
        f, y, fw = f[idx], y[idx], fw[idx]
    xy = project_2d(f.numpy())
    fw = np.asarray(fw)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ax[0].scatter(xy[:, 0], xy[:, 1], c=y.numpy(), cmap="tab20", s=7, alpha=0.7)
    ax[0].set_title("coloured by identity\n(tight separated clusters = good)")
    for name in FOOTWEAR:
        mask = fw == name
        ax[1].scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=0.7, label=FOOTWEAR_LABEL[name])
    ax[1].legend(markerscale=2, fontsize=8)
    ax[1].set_title("coloured by footwear\n(mixed within clusters = footwear-invariant)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
