"""Identification and verification metrics (leakage-safe, competition-style)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def enroll_templates(feats, labels):
    """One L2-normalised template per identity, the mean of its gallery embeddings."""
    ids = torch.unique(labels)
    templates = torch.stack([F.normalize(feats[labels == i].mean(0), dim=0) for i in ids])
    return templates, ids


def identification(probe_f, probe_y, gallery_f, gallery_y, topk=(1, 5)):
    """Rank-k and mAP for probes scored against per-identity templates."""
    templates, ids = enroll_templates(gallery_f, gallery_y)
    sim = F.normalize(probe_f) @ templates.t()
    order = sim.argsort(dim=1, descending=True)
    hit = ids[order] == probe_y[:, None]
    cmc = {k: hit[:, :k].any(1).float().mean().item() for k in topk}
    aps = []
    for row in hit:
        rel = row.float()
        if rel.sum() == 0:
            continue
        prec = rel.cumsum(0) / torch.arange(1, len(rel) + 1)
        aps.append(((prec * rel).sum() / rel.sum()).item())
    return cmc, float(np.mean(aps)) if aps else 0.0


def verification(probe_f, probe_y, gallery_f, gallery_y, far_target=0.01):
    """EER and TAR@FAR from genuine vs impostor probe-template scores."""
    templates, ids = enroll_templates(gallery_f, gallery_y)
    sim = (F.normalize(probe_f) @ templates.t()).numpy()
    genuine = (ids[None, :] == probe_y[:, None]).numpy()
    scores, labels = sim.ravel(), genuine.ravel().astype(int)
    if labels.min() == labels.max():
        return float("nan"), float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)
    tar = float(tpr[np.searchsorted(fpr, far_target, side="left").clip(0, len(tpr) - 1)])
    return eer, tar


def report_from_scores(scores, labels):
    """Full verification report at the equal-error threshold (EER/BACC/F1/precision/recall/…)."""
    scores, labels = np.asarray(scores), np.asarray(labels).astype(int)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    tau = thr[i]
    pred = (scores >= tau).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    fmr = fp / max(1, fp + tn)
    fnmr = fn / max(1, fn + tp)
    # FMR100 (competition security metric): false-match rate at the threshold where FNMR <= 1%.
    fmr100 = float(fpr[np.argmax(fnr <= 0.01)]) if (fnr <= 0.01).any() else 1.0
    return dict(eer=float((fpr[i] + fnr[i]) / 2), fmr100=fmr100,
                accuracy=(tp + tn) / max(1, len(labels)),
                balanced_accuracy=1 - (fmr + fnmr) / 2, precision=prec, recall=rec,
                f1=2 * prec * rec / max(1e-9, prec + rec), fmr=fmr, fnmr=fnmr)
