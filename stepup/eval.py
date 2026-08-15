"""Leakage-safe, deployment-realistic evaluation."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_score

from .config import FOOTWEAR, FOOTWEAR_LABEL, SEED, dev
from .metrics import enroll_templates, identification, report_from_scores, verification


def _pooled_cross_scores(f, y, fw):
    """Pool cross-footwear genuine/impostor cosine scores: for each enrol shoe, template each id
    from that shoe, probe with every OTHER shoe (the hard, generalization-relevant pairs)."""
    fw = np.asarray(fw)
    y = y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))
    scores, labels = [], []
    for enrol in FOOTWEAR:
        g, p = fw == enrol, fw != enrol
        if g.sum() == 0 or p.sum() == 0:
            continue
        templates, ids = enroll_templates(f[g], y[g])
        sim = (F.normalize(f[p]) @ templates.t()).numpy()
        gen = (ids[None, :] == y[p][:, None]).numpy()
        scores.append(sim.ravel()); labels.append(gen.ravel())
    return np.concatenate(scores), np.concatenate(labels).astype(int)


def separability_panel(f, y, fw):
    """NEW smooth generalization metrics computed from the FULL embedding set (no sampled 15-id
    gallery), so they are far less per-epoch-noisy than rank-1 / EER:
      auc      -- cross-footwear verification ROC-AUC (whole curve, not one threshold like EER)
      dprime   -- (mu_genuine - mu_impostor) / pooled-std: raw separability of the score dists
      emb_margin -- mean(intra-id cosine to own centroid) - mean(inter-id centroid cosine)
      silhouette -- cosine silhouette of the identity clustering
    All are 'higher = better' and read the representation directly, which is what actually
    over-fits (train->1.0, val->collapse). Cheap: reuses the already-computed embeddings."""
    y = y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))
    s, lab = _pooled_cross_scores(f, y, fw)
    gen, imp = s[lab == 1], s[lab == 0]
    dprime = float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))
    auc = float(roc_auc_score(lab, s)) if lab.min() != lab.max() else float("nan")
    fn = F.normalize(f).numpy(); yy = y.numpy()
    ids = np.unique(yy)
    cents, intra = [], []
    for i in ids:
        v = fn[yy == i]; c = v.mean(0); c = c / (np.linalg.norm(c) + 1e-12)
        cents.append(c); intra.append(float((v @ c).mean()))
    cents = np.stack(cents)
    inter = (cents @ cents.T)[np.triu_indices(len(ids), 1)] if len(ids) > 1 else np.array([0.0])
    margin = float(np.mean(intra) - float(inter.mean()))
    try:
        sil = float(silhouette_score(fn, yy, metric="cosine")) if len(ids) > 1 else float("nan")
    except Exception:
        sil = float("nan")
    return dict(auc=auc, dprime=dprime, emb_margin=margin, silhouette=sil)


def representation_metrics(f, y, fw, seed=0):
    """Research-backed embedding-QUALITY metrics that diagnose memorization vs generalization,
    computed on the VAL (unseen-id) embeddings each epoch (cheap):
      alignment  -- Wang&Isola 2020: E||x_i - x_j||^2 over same-id CROSS-FOOTWEAR positive pairs
                    (lower=better; positives should be close *despite the shoe*). Finite optimum.
      uniformity -- Wang&Isola 2020: log E exp(-2||x_i - x_j||^2) over random pairs (lower=more
                    spread on the hypersphere). Finite optimum. Over-separation lowers this while
                    HURTING alignment -- the trade-off is the memorization signal.
      fisher     -- trace(between-class scatter)/trace(within-class scatter); separability that
                    should transfer if it is genuine identity structure, not memorized.
      erank      -- RankMe (Garrido 2023): effective rank = exp(entropy of normalized singular
                    values) of the embedding matrix; representation richness, NO labels. Collapse
                    (memorizing a few directions) drives it down.
      davies_bouldin -- mean cluster compactness/separation ratio (lower=better)."""
    y = y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))
    fn = F.normalize(f).numpy(); yy = y.numpy(); fw = np.asarray(fw)
    ids = np.unique(yy)
    # alignment over same-id cross-footwear positive pairs
    al = []
    for i in ids:
        idx = np.where(yy == i)[0]
        if len(idx) < 2:
            continue
        V = fn[idx]; sim = V @ V.T
        diff_fw = fw[idx][:, None] != fw[idx][None, :]
        if diff_fw.any():
            al.append(float((2 - 2 * sim[diff_fw]).mean()))     # ||a-b||^2 = 2-2cos on the sphere
    alignment = float(np.mean(al)) if al else float("nan")
    # uniformity over a random pair sample
    rng = np.random.default_rng(seed)
    n = len(fn); m = min(4000, n * (n - 1) // 2)
    a = rng.integers(0, n, m); b = rng.integers(0, n, m); ok = a != b
    d2 = 2 - 2 * (fn[a[ok]] * fn[b[ok]]).sum(1)
    uniformity = float(np.log(np.exp(-2 * d2).mean() + 1e-12))
    # Fisher discriminant ratio (trace form)
    mu = fn.mean(0); sw = sb = 0.0
    for i in ids:
        v = fn[yy == i]; mc = v.mean(0)
        sw += ((v - mc) ** 2).sum(); sb += len(v) * ((mc - mu) ** 2).sum()
    fisher = float(sb / (sw + 1e-12))
    # RankMe effective rank of the embedding matrix
    s = np.linalg.svd(fn - fn.mean(0), compute_uv=False)
    p = s / (s.sum() + 1e-12); erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    try:
        db = float(davies_bouldin_score(fn, yy)) if len(ids) > 1 else float("nan")
    except Exception:
        db = float("nan")
    return dict(alignment=alignment, uniformity=uniformity, fisher=fisher,
                erank=erank, davies_bouldin=db)


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
            cmc, mp = identification(f[p_sel], y[p_sel], f[g_sel], y[g_sel], topk=(1, 3, 5))
            eer, tar = verification(f[p_sel], y[p_sel], f[g_sel], y[g_sel])
            rows.append(dict(enrol=enrol, probe=probe, kind="same" if enrol == probe else "cross",
                             rank1=cmc[1], rank3=cmc[3], rank5=cmc[5], mAP=mp, eer=eer, tar1=tar,
                             n_probe=int(p_sel.sum())))
    import pandas as pd
    return pd.DataFrame(rows)


def cross_footwear_scores(net, ds, precomp=None):
    """Pooled cross-footwear genuine/impostor similarity scores + binary genuine labels -- the raw
    material shared by the verification report and the DET curve (so both use identical scores)."""
    f, y, fw = precomp if precomp is not None else embed_dataset(net, ds)
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
    return np.concatenate(scores), np.concatenate(labels)


def cross_footwear_verification(net, ds):
    """Pooled cross-footwear genuine/impostor scores -> full competition report."""
    scores, labels = cross_footwear_scores(net, ds)
    return report_from_scores(scores, labels)


def condition_verification(net, ds, precomp=None):
    """Competition-style verification, split by SEEN vs UNSEEN footwear condition, reporting the
    full metric set the StepUP competition uses (EER, FMR100, ACC, BACC, FNMR, FMR). 'seen' =
    probe footwear matches an enrolled shoe (enrol/probe split by pass to avoid the same sample);
    'unseen' = cross-footwear (probe shoe never enrolled) -- the hard, generalization case."""
    f, y, fw = precomp if precomp is not None else embed_dataset(net, ds)
    fw = np.asarray(fw)
    m = ds.m.reset_index(drop=True)
    first_pass = m.groupby("ParticipantID").PassID.transform("min").to_numpy()
    passid = m.PassID.to_numpy()
    out = {}
    for cond in ("seen", "unseen"):
        scores, labels = [], []
        for enrol in FOOTWEAR:
            g = fw == enrol
            if g.sum() == 0:
                continue
            if cond == "seen":
                g_sel = g & (passid == first_pass); p_sel = g & (passid != first_pass)
            else:
                g_sel = g; p_sel = fw != enrol
            if g_sel.sum() == 0 or p_sel.sum() == 0:
                continue
            templates, ids = enroll_templates(f[g_sel], y[g_sel])
            sim = (F.normalize(f[p_sel]) @ templates.t()).numpy()
            genuine = (ids[None, :] == y[p_sel][:, None]).numpy()
            scores.append(sim.ravel()); labels.append(genuine.ravel())
        if scores:
            out[cond] = report_from_scores(np.concatenate(scores), np.concatenate(labels))
    return out


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


def accumulated_identification_bdcspn(net, ds, ks=(1, 3, 5, 10), z=8, scale=10.0):
    """Live-monitoring variant of accumulated_identification: before the accumulated match,
    the gallery prototypes are rectified with BD-CSPN using the probe batch (transductive),
    then each pass window is matched against the rectified prototypes. Combines the two
    deployment mechanisms -- probe-side accumulation and prototype-side BD-CSPN -- for the
    continuous-walk setting. Strict leave-one-footwear-out (cross-footwear).

    The gallery here is the FULL reference-footwear prototype -- enroll_templates() averages
    ALL steps of the enrol footwear per identity -- NOT the k-limited gallery of enroll.py's
    cross_fw_eval (which builds the prototype from only k enrolled steps). So this measures
    BD-CSPN on top of the full gallery, while tab:enroll measures it as the gallery grows with k."""
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
        protos, ids = enroll_templates(f[g], y[g])
        probe_rows = np.where(fw != enrol)[0]
        if len(probe_rows) == 0:
            continue
        # BD-CSPN: shift probes to the prototype centre, then fold confident probes in.
        q = f[probe_rows]
        xi = protos.mean(0, keepdim=True) - q.mean(0, keepdim=True)      # cross-class shift
        q = F.normalize(q + xi, dim=1)
        conf = torch.softmax((q @ protos.t()) * scale, dim=1)            # pseudo-labels
        rect = protos.clone()
        for j in range(len(protos)):
            w = conf[:, j]
            top = torch.topk(w, min(z, len(w))).indices
            ww = w[top].unsqueeze(1)
            rect[j] = ((protos[j:j + 1] + (ww * q[top]).sum(0, keepdim=True)) / (1.0 + ww.sum())).squeeze(0)
        rect = F.normalize(rect, dim=1)
        qrow = {int(r): q[i] for i, r in enumerate(probe_rows)}          # shifted probe by row
        passes = {}
        for i in probe_rows:
            passes.setdefault((pid[i], fw[i], passid[i]), []).append(int(i))
        for k in ks:
            hit = tot = 0
            for _, rows in passes.items():
                rows = sorted(rows, key=lambda i: stepid[i])
                for s in range(0, len(rows) - k + 1, k):
                    win = rows[s:s + k]
                    probe = F.normalize(torch.stack([qrow[r] for r in win]).mean(0), dim=0)
                    pred = ids[(probe @ rect.t()).argmax()]
                    hit += int(pred.item() == int(y[win[0]])); tot += 1
            per_k[k].append(hit / tot if tot else float("nan"))
    return {k: float(np.nanmean(v)) for k, v in per_k.items()}


def open_set_accumulated(net, ds, n_enroll=5, ks=(1, 3, 5, 10), repeats=3, precomp=None,
                         score_norm="none"):
    """Reference open-set protocol (mixed-footwear gallery, NOT cross-footwear): enroll
    n_enroll random steps per identity (any footwear) into one template, then probe = running
    mean of k consecutive steps within one (identity, footwear, speed) trial; rank-1 per k,
    averaged over random enroll draws. This mirrors the reference eval that reaches ~0.9 at
    k=5-10, so it is the fair comparison to that number -- contrast with
    accumulated_identification, which is strict leave-one-footwear-out (much harder).

    score_norm='znorm' applies cohort (Z-norm) score normalization: each template's similarity
    column is standardized by its own impostor distribution over all probes, which de-biases
    'hub' templates that are spuriously close to everything and typically lifts rank-1 a few
    points (the cohort-normalization the competition baseline used)."""
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
            probes, trues = [], []
            for _, rows in passes.items():
                rows = sorted(rows, key=lambda r: (passid[r], stepid[r]))
                for s in range(0, len(rows) - k + 1, k):
                    win = rows[s:s + k]
                    probes.append(F.normalize(f[win].mean(0), dim=0)); trues.append(int(y[win[0]]))
            if not probes:
                per_k[k].append(float("nan")); continue
            scores = (torch.stack(probes) @ templ.t()).numpy()          # (W, n_templates)
            if score_norm == "znorm":                                    # cohort normalization
                scores = (scores - scores.mean(0, keepdims=True)) / (scores.std(0, keepdims=True) + 1e-6)
            preds = tids[scores.argmax(1)]
            per_k[k].append(float((preds == np.array(trues)).mean()))
    return {k: float(np.nanmean(v)) for k, v in per_k.items()}


def summarise(df):
    out = {}
    for kind in ("same", "cross"):
        sub = df[df.kind == kind]
        if len(sub):
            out[f"{kind}_rank1"] = float(np.nanmean(sub.rank1))
            out[f"{kind}_eer"] = float(np.nanmean(sub.eer))
            # impactful identification metrics already computed per shoe-pair but previously
            # discarded: mAP (full-ranking quality) and TAR@1%FAR (deployment gate accept rate).
            out[f"{kind}_map"] = float(np.nanmean(sub.mAP))
            out[f"{kind}_rank3"] = float(np.nanmean(sub.rank3))
            out[f"{kind}_rank5"] = float(np.nanmean(sub.rank5))
            out[f"{kind}_tar1"] = float(np.nanmean(sub.tar1))
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


def embedding_panel(f, y, fw, title, max_pts=2500):
    """2D UMAP/t-SNE projection as a 2-panel figure: coloured by identity, and by footwear. Takes
    ALREADY-EMBEDDED (f, y, fw) so it can be reused for a periodic per-epoch snapshot (on the val
    embeddings computed anyway) and for the final test plot. Returns the matplotlib figure; the
    caller saves it and/or logs it to wandb, then closes it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(f) > max_pts:
        idx = np.random.default_rng(SEED).choice(len(f), max_pts, replace=False)
        f, y, fw = f[idx], y[idx], np.asarray(fw)[idx]
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
    return fig


def plot_embeddings(net, ds, title, out_path, max_pts=2500):
    """Embed a dataset and save the 2-panel identity/footwear projection to out_path."""
    import matplotlib.pyplot as plt
    f, y, fw = embed_dataset(net, ds)
    fig = embedding_panel(f, y, fw, title, max_pts)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_det(scores, labels, title, out_path=None):
    """Detection Error Tradeoff (StepUP competition Fig 2 style): FNMR vs FMR in %, with the EER
    point on the FNMR=FMR diagonal and the FMR=1% operating line whose FNMR is the reported FMR100.
    wandb cannot draw this itself. Returns the figure (out_path=None) or saves + returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    scores = np.asarray(scores); labels = np.asarray(labels).astype(int)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); eer = float((fpr[i] + fnr[i]) / 2)
    fmr100 = float(np.interp(0.01, fpr, fnr))                    # FNMR at FMR=1% (competition metric)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(fpr * 100, fnr * 100, lw=2, color="tab:blue")
    ax.plot([0, 100], [0, 100], ls=":", c="gray", lw=1)         # FNMR = FMR diagonal
    ax.scatter([eer * 100], [eer * 100], color="tab:red", zorder=5, label=f"EER = {eer * 100:.2f}%")
    ax.axvline(1.0, ls="--", c="tab:green", lw=1)               # FMR = 1% operating point
    ax.scatter([1.0], [fmr100 * 100], color="tab:green", zorder=5,
               label=f"FMR100 = {fmr100 * 100:.2f}%")
    hi = max(5.0, min(100.0, eer * 100 * 3))                    # zoom to the informative region
    ax.set_xlim(0, hi); ax.set_ylim(0, max(hi, fmr100 * 100 * 1.1))
    ax.set_xlabel("FMR (%)"); ax.set_ylabel("FNMR (%)")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120); plt.close(fig); return out_path
    return fig
