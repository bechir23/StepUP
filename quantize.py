#!/usr/bin/env python
"""Quantization comparison for the frozen footstep backbone, via ONNX Runtime, evaluated on the
SAME cross-footwear EER/FMR100 harness as training -- so FP32 vs INT8 is apples-to-apples.

Rungs compared:  FP32 | FP16 | dynamic-INT8 | static-INT8(PTQ) | static-INT8 + AdaBN
  (static-INT8 mirrors ultralytics: calibrate on ~N real val footsteps, QDQ graph, per-channel
   weights, then re-evaluate on the identical metric. AdaBN = re-estimate BatchNorm running stats
   on the deployment footsteps before quantizing, so the model normalizes to the real distribution.)

Loads the winning checkpoint from Hugging Face by args (same loader as finetune.py):
  python quantize.py --model gaitcnn_snr --tag <t> --hf-repo Bechir23/stepup-footstep \
      --pack-device cuda --calib 300 --out artifacts

Outputs to <out>/: quant_results.parquet + 4 plots (EER-latency Pareto, EER-size, DET FP32-vs-INT8,
BN-ablation bar), and the .onnx artifacts.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- model plumbing -----------------------------
class EmbWrap(nn.Module):
    """Wrap the Embedder so ONNX exports exactly the matching feature: L2-normalized f_i."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        f_i = out[1] if isinstance(out, (tuple, list)) else out
        return F.normalize(f_i, dim=1)


def adabn_recalibrate(net, batches, dev):
    """AdaBN (Li et al. 2016): reset BatchNorm running stats and re-accumulate them on target
    footsteps (no labels, no backprop) so the model normalizes to the deployment distribution.
    InstanceNorm (SNR) is untouched -- it has no running stats."""
    bns = [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                       # cumulative moving average = true mean over the pass
        m.train()
    with torch.no_grad():
        for xb in batches:
            net(xb.to(dev))
    net.eval()
    return len(bns)


# ----------------------------- ONNX export + quantize -----------------------------
def export_fp32(wrap, dummy, path):
    wrap.eval()
    # legacy TorchScript exporter: embeds weights in ONE .onnx file and writes clean
    # dynamic-batch shapes. (The dynamo exporter stored weights externally -> a stub graph
    # with inconsistent shape annotations that ONNX strict shape-inference rejects at quant time.)
    torch.onnx.export(
        wrap,
        dummy,
        path,
        input_names=["x"],
        output_names=["emb"],
        dynamic_axes={"x": {0: "B"}, "emb": {0: "B"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,          # force the legacy TorchScript exporter (single-file weights)
    )
    return path

def to_fp16(fp32_path, out_path):
    from onnxconverter_common import float16
    import onnx
    m = onnx.load(fp32_path)
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), out_path)
    return out_path


def dyn_int8(fp32_path, out_path):
    # dynamic quant emits ConvInteger for Conv, which the ORT CPU EP does NOT implement -> restrict
    # to MatMul/Gemm (Linear). This is also the honest behaviour: dynamic INT8 helps Linear-heavy
    # nets, barely a conv net -- a real data point for the comparison. Static-INT8 (QDQ/QLinearConv)
    # is the one that quantizes the convs.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp32_path, out_path, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    return out_path


class ArrCalib:
    """CalibrationDataReader over a list of numpy input batches (static-INT8 calibration)."""

    def __init__(self, arrays, input_name="x"):
        self._it = iter([{input_name: a.astype(np.float32)} for a in arrays])

    def get_next(self):
        return next(self._it, None)


def static_int8(fp32_path, out_path, reader):
    from onnxruntime.quantization import (quantize_static, QuantFormat, QuantType,
                                          CalibrationMethod)
    src = fp32_path
    try:                                            # shape-inference + graph opt (folds BN into Conv)
        from onnxruntime.quantization.shape_inference import quant_pre_process
        src = fp32_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(fp32_path, src)
    except Exception as e:
        print(f"  (quant pre-process skipped: {e})", flush=True); src = fp32_path
    quantize_static(src, out_path, calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                    calibrate_method=CalibrationMethod.MinMax)
    return out_path


# ----------------------------- ONNX inference / metrics / latency -----------------------------
def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                 # fixed threads -> fair FP32-vs-INT8 latency
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def ort_embed(path, ds, batch=256):
    """Embed a dataset through an ONNX model (mirrors stepup.eval.embed_dataset, via ORT)."""
    sess = _session(path)
    name = sess.get_inputs()[0].name
    feats, labs = [], []
    on_gpu = getattr(ds, "device", "cpu") == "cuda"
    if on_gpu:
        for i in range(0, len(ds), batch):
            xb, yb, _ = ds.gather(range(i, min(i + batch, len(ds))))
            f = sess.run(None, {name: xb.cpu().numpy().astype(np.float32)})[0]
            feats.append(torch.from_numpy(f)); labs.append(yb.cpu())
    else:
        from torch.utils.data import DataLoader
        for xb, yb, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
            f = sess.run(None, {name: xb.numpy().astype(np.float32)})[0]
            feats.append(torch.from_numpy(f)); labs.append(yb)
    return torch.cat(feats), torch.cat(labs), ds.m.Footwear.to_numpy()


def latency_ms(path, dummy_np, n=200, warmup=20):
    sess = _session(path)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: dummy_np})
    ts = []
    for _ in range(n):
        t = time.perf_counter(); sess.run(None, {name: dummy_np}); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def eval_scores(f, y, fw, ds=None):
    """Overall cross-footwear EER/FMR100 (+ seen/unseen if the manifest is available)."""
    from stepup.eval import cross_footwear_scores
    from stepup.metrics import report_from_scores
    sc, lb = cross_footwear_scores(None, ds, precomp=(f, y, fw))
    rep = report_from_scores(sc, lb)
    out = dict(eer=rep["eer"], fmr100=rep["fmr100"], scores=sc, labels=lb)
    if ds is not None and hasattr(ds, "m"):
        try:
            from stepup.eval import condition_verification
            cond = condition_verification(None, ds, precomp=(f, y, fw))
            out["seen_eer"] = cond.get("seen", {}).get("eer", float("nan"))
            out["unseen_eer"] = cond.get("unseen", {}).get("eer", float("nan"))
        except Exception as e:
            print(f"  (seen/unseen skipped: {e})", flush=True)
    return out


# ----------------------------- plots -----------------------------
def _det(ax, scores, labels, label, color):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.asarray(labels).astype(int), np.asarray(scores))
    fnr = 1 - tpr
    ax.plot(fpr * 100, fnr * 100, lw=2, label=label, color=color)


def make_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pathlib
    out_dir = pathlib.Path(out_dir)
    ok = [r for r in rows if r.get("eer") is not None]

    # 1) EER vs latency Pareto
    fig, a = plt.subplots(figsize=(5.6, 4.2))
    for r in ok:
        a.scatter(r["latency_ms"], r["eer"] * 100, s=60)
        a.annotate(r["name"], (r["latency_ms"], r["eer"] * 100), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("CPU latency / footstep (ms, batch=1)"); a.set_ylabel("cross-fw EER (%)")
    a.set_title("Quantization: EER vs latency"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "quant_eer_latency.png", dpi=120); plt.close(fig)

    # 2) EER vs model size
    fig, a = plt.subplots(figsize=(5.6, 4.2))
    for r in ok:
        a.scatter(r["size_mb"], r["eer"] * 100, s=60)
        a.annotate(r["name"], (r["size_mb"], r["eer"] * 100), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("model size (MB)"); a.set_ylabel("cross-fw EER (%)")
    a.set_title("Quantization: EER vs size"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "quant_eer_size.png", dpi=120); plt.close(fig)

    # 3) DET overlay: FP32 vs best INT8 (static)
    fp32 = next((r for r in ok if r["name"] == "fp32"), None)
    sint = next((r for r in ok if r["name"] == "static_int8"), None)
    if fp32 is not None and sint is not None:
        fig, a = plt.subplots(figsize=(5.2, 5))
        _det(a, fp32["scores"], fp32["labels"], f"FP32 (EER {fp32['eer']*100:.2f})", "tab:blue")
        _det(a, sint["scores"], sint["labels"], f"static-INT8 (EER {sint['eer']*100:.2f})", "tab:red")
        a.plot([0, 100], [0, 100], ls=":", c="gray", lw=1)
        hi = max(5, min(60, fp32["eer"] * 100 * 3))
        a.set_xlim(0, hi); a.set_ylim(0, hi)
        a.set_xlabel("FMR (%)"); a.set_ylabel("FNMR (%)"); a.set_title("DET: FP32 vs static-INT8")
        a.legend(fontsize=8); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / "quant_det_fp32_vs_int8.png", dpi=120); plt.close(fig)

    # 4) BN-ablation bar: FP32 / INT8 frozen-BN / INT8 AdaBN
    order = ["fp32", "static_int8", "static_int8_adabn"]
    bars = [(r["name"], r["eer"] * 100) for n in order for r in ok if r["name"] == n]
    if bars:
        fig, a = plt.subplots(figsize=(5.2, 4))
        a.bar([b[0] for b in bars], [b[1] for b in bars],
              color=["tab:gray", "tab:orange", "tab:green"][:len(bars)])
        for i, b in enumerate(bars):
            a.text(i, b[1], f"{b[1]:.2f}", ha="center", va="bottom", fontsize=9)
        a.set_ylabel("cross-fw EER (%)"); a.set_title("BN ablation: frozen vs AdaBN under INT8")
        a.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / "quant_bn_ablation.png", dpi=120); plt.close(fig)
    print(f"  plots -> {out_dir}/quant_*.png", flush=True)


# ----------------------------- driver -----------------------------
def run_all(net, ds_test, ds_calib, dummy, out_dir, dev, calib_n=300, do_adabn=True):
    """Build every rung, evaluate + time it, and emit the plots. Returns the results rows."""
    import pathlib
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dummy_np = dummy.cpu().numpy().astype(np.float32)
    size_mb = lambda p: os.path.getsize(p) / 1e6

    # calibration batches (numpy) from the calib dataset
    calib_arrs, seen = [], 0
    from torch.utils.data import DataLoader
    for xb, _, _ in DataLoader(ds_calib, batch_size=32, shuffle=False, num_workers=0):
        calib_arrs.append(xb.numpy().astype(np.float32)); seen += len(xb)
        if seen >= calib_n:
            break

    wrap = EmbWrap(net).to(dev).eval()
    fp32 = export_fp32(wrap, dummy, str(out_dir / "emb_fp32.onnx"))
    rows = []

    def add(name, path):
        f, y, fw = ort_embed(path, ds_test)
        m = eval_scores(f, y, fw, ds_test)
        lat, p95 = latency_ms(path, dummy_np)
        rows.append(dict(name=name, size_mb=size_mb(path), latency_ms=lat, latency_p95=p95, **m))
        print(f"  {name:20s}  EER {m['eer']*100:5.2f}  size {size_mb(path):5.2f}MB  "
              f"lat {lat:6.2f}ms", flush=True)

    add("fp32", fp32)
    try:
        add("fp16", to_fp16(fp32, str(out_dir / "emb_fp16.onnx")))
    except Exception as e:
        print(f"  (fp16 skipped: {e})", flush=True)
    add("dynamic_int8", dyn_int8(fp32, str(out_dir / "emb_dyn_int8.onnx")))
    add("static_int8", static_int8(fp32, str(out_dir / "emb_static_int8.onnx"), ArrCalib(calib_arrs)))

    if do_adabn:
        n_bn = adabn_recalibrate(net, [torch.from_numpy(a) for a in calib_arrs], dev)
        print(f"  AdaBN: recalibrated {n_bn} BatchNorm layers on {seen} footsteps", flush=True)
        wrap2 = EmbWrap(net).to(dev).eval()
        fp32a = export_fp32(wrap2, dummy, str(out_dir / "emb_fp32_adabn.onnx"))
        add("static_int8_adabn", static_int8(fp32a, str(out_dir / "emb_static_int8_adabn.onnx"),
                                             ArrCalib(calib_arrs)))

    make_plots(rows, out_dir)
    import pandas as pd
    tbl = pd.DataFrame([{k: v for k, v in r.items() if k not in ("scores", "labels")} for r in rows])
    tbl.to_parquet(out_dir / "quant_results.parquet", index=False)
    print("\n" + tbl.to_string(index=False), flush=True)
    return rows


def main():
    from stepup.config import dev
    from stepup.data import build_datasets
    from finetune import load_backbone_trainable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaitcnn_snr"); ap.add_argument("--tag", default="")
    ap.add_argument("--hf-repo", default=None); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--pack-device", default="", choices=["", "cuda", "memmap", "cpu"])
    ap.add_argument("--calib", type=int, default=300, help="calibration footsteps for static INT8")
    ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    ap.add_argument("--wandb-project", default="stepup-footstep")
    args = ap.parse_args()

    net, cfg = load_backbone_trainable(args.model, args.hf_repo, args.hf_token, args.tag)
    net = net.to(dev).eval()
    if args.pack_device:
        cfg["pack_device"] = args.pack_device
    _, ds = build_datasets(cfg)
    ds_test = ds["test"]; ds_calib = ds.get("val") or ds.get("val_mon") or ds["train"]
    x0 = ds_test[0][0]                                      # one real footstep -> exact input shape
    dummy = x0.unsqueeze(0).to(dev)
    # 2D backbones consume the cube as frames-as-channels (squeeze the singleton dim); show that
    # effective 2D shape (B, frames, H, W) rather than the 5D storage wrapper. 3D nets keep the 5D.
    _3d = ("r2plus1d", "r3d", "swin3d", "vit")
    _shape = tuple(dummy.shape) if args.model.startswith(_3d) else tuple(dummy.squeeze(1).shape)
    print(f"quantize {args.model}  input {_shape}  test {len(ds_test)} steps", flush=True)
    rows = run_all(net, ds_test, ds_calib, dummy, args.out, dev, calib_n=args.calib, do_adabn=args.adabn)

    if args.wandb != "disabled":                       # optional: mirror the table + plots to wandb
        import pathlib
        import wandb
        run = wandb.init(project=args.wandb_project, mode=args.wandb,
                         name=f"quant_{args.model}" + (f"_{args.tag}" if args.tag else ""))
        cols = ["method", "eer", "fmr100", "size_mb", "latency_ms", "latency_p95"]
        tbl = wandb.Table(columns=cols, data=[[r["name"], r.get("eer"), r.get("fmr100"),
                          r["size_mb"], r["latency_ms"], r["latency_p95"]] for r in rows])
        run.log({"quant/results": tbl})
        for p in sorted(pathlib.Path(args.out).glob("quant_*.png")):
            run.log({f"quant/{p.stem}": wandb.Image(str(p))})
        int8 = [r["eer"] for r in rows if "int8" in r["name"] and r.get("eer") is not None]
        fp32 = next((r["eer"] for r in rows if r["name"] == "fp32"), None)
        if int8 and fp32 is not None:
            run.summary["fp32_eer"] = fp32; run.summary["best_int8_eer"] = min(int8)
            run.summary["int8_eer_drop"] = min(int8) - fp32     # <- "did EER drop too much?" in one number
        run.finish()
        print("  logged results + plots to wandb", flush=True)


if __name__ == "__main__":
    main()
