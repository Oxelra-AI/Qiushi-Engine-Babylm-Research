#!/usr/bin/env python
"""BOS-only AoA: extract P(word | <bos>) across checkpoints (single-BOS, fixed path)
and plot learning-curve small-multiples by category (averaging + k*ln(V) ceiling +
robust fit — the final scorer mechanism).

Run from babylm-eval-hidden/strict, or set BABYLM_STRICT to that path.
Point HF_HOME at scratch if disk is tight; PURGE_PER_CKPT keeps only one revision at a time.
"""
import os
import sys
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------- config ----------------
MODEL = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
BACKEND = "causal"           # causal / mlm / mntp / enc_dec_mask / enc_dec_prefix
TRACK = "non-strict-small"   # "strict-small" (19 ckpts) or "non-strict-small" (28)
MIN_CONTEXT = 20             # word set: same filter as the context run
PURGE_PER_CKPT = True        # delete each HF revision after use (low disk)
EVAL_DATASET = "BabyLM-community/BabyLM-2026-Strict-Evals"  # source of the AoA/CDI data
REPO_STRICT = Path(os.environ.get("BABYLM_STRICT", ".")).resolve()
OUT_DIR = REPO_STRICT / "results" / (Path(MODEL).stem + "_BOS")
# ----------------------------------------

sys.path.insert(0, str(REPO_STRICT))
from evaluation_pipeline.AoA_word.eval_util import StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

WORD_PATH = REPO_STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json"
CDI = REPO_STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv"
SURPRISAL = OUT_DIR / "surprisal.json"
PNG = OUT_DIR / "aoa_bos_curves.png"


def ensure_data():
    # Fetch the AoA/CDI eval files from HF if they aren't present locally.
    if WORD_PATH.exists() and CDI.exists():
        return
    from huggingface_hub import snapshot_download
    print(f"[data] downloading AoA eval data from {EVAL_DATASET}", flush=True)
    snapshot_download(
        repo_id=EVAL_DATASET,
        repo_type="dataset",
        local_dir=str(REPO_STRICT),
        allow_patterns=["evaluation_data/full_eval/aoa/*"],
    )


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d


def extract():
    import torch
    hub = Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None

    def purge():
        if PURGE_PER_CKPT and hub and hub.exists():
            shutil.rmtree(hub, ignore_errors=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    words, _ = load_eval(WORD_PATH, MIN_CONTEXT, debug=False)
    cfg = StepConfig(resume=False, track=TRACK, file_path=None, debug=False)
    ex = StepSurprisalExtractor(config=cfg, model_name=MODEL, backend=BACKEND, device=dev)
    print(f"[extract] {MODEL}  words={len(words)}  steps={len(cfg.steps)}  device={dev}", flush=True)
    rows = []
    for step, wc in zip(cfg.steps, cfg.word_counts):
        purge()
        print(f"[extract] {step}", flush=True)
        model = ex.load_model_for_step(step)
        proc, tok = ex.load_tokenizer_for_step(step)
        for w in words:
            s = ex.compute_surprisal(model, proc, tok, "", w, use_bos_only=True)
            rows.append({"step": step, "word_count": wc, "target_word": w,
                         "surprisal": s, "use_bos_only": True})
    purge()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"results": rows}, SURPRISAL.open("w"))
    print(f"[extract] DONE rows={len(rows)} -> {SURPRISAL}", flush=True)


def robust_fit(ls, neg):
    from scipy.optimize import curve_fit
    rng = neg.max() - neg.min()
    p0 = [rng, 1.0, ls.mean(), neg.min()]
    lb = [0.0, 0.0, ls.min() - 1, neg.min() - 2 * rng - 1]
    ub = [10 * rng + 1, 100.0, ls.max() + 1, neg.max() + 1]
    return curve_fit(sigmoid, ls, neg, p0=p0, bounds=(lb, ub), maxfev=20000)


def classify(us, means, k, V):
    chance = k * np.log(V)
    thr = chance - 0.5 * (chance - means.min())
    neg = -means
    ls = np.log10(us + 1)
    try:
        popt, _ = robust_fit(ls, neg)
    except Exception:
        return None, "fit-fail", None, thr
    a, b, c, d = popt
    nt = -thr
    if b <= 1e-6 or a <= 1e-6:
        return None, "bad-shape", popt, thr
    if nt <= d:
        return None, "already-acq", popt, thr
    if nt >= a + d:
        return None, "never-reached", popt, thr
    la = c - np.log((a / (nt - d)) - 1) / b
    if 10 ** la - 1 < us[0] or 10 ** la - 1 > us[-1]:
        return None, "outside-range", popt, thr
    return la, "ok", popt, thr


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = tok.vocab_size
    pref = len(tok("The", add_special_tokens=False)["input_ids"])

    def klen(w):
        return max(1, len(tok("The " + w, add_special_tokens=False)["input_ids"]) - pref)

    ev = AoAEvaluator(CDI)
    wd = {}
    for r in json.load(SURPRISAL.open())["results"]:
        st = ev.extract_step_number(r["step"])
        wd.setdefault(r["target_word"], {"steps": [], "s": []})
        wd[r["target_word"]]["steps"].append(st)
        wd[r["target_word"]]["s"].append(float(r["surprisal"]))

    info = {}
    for w, d in wd.items():
        us = np.array(sorted(set(d["steps"])), float)
        means = np.array([np.mean([x for s2, x in zip(d["steps"], d["s"]) if s2 == u]) for u in us])
        aoa, cat, popt, thr = classify(us, means, klen(w), V)
        info[w] = dict(us=us, means=means, cat=cat, popt=popt, aoa=aoa, thr=thr,
                       chance=klen(w) * np.log(V), minm=float(means.min()), k=klen(w))
    print("category counts:", dict(Counter(i["cat"] for i in info.values()).most_common()))

    C_MEAN, C_RAW, C_FIT, C_THR, C_ANCH, C_CROSS = "#111", "#CCC", "#0072B2", "#009E73", "#999", "#D55E00"
    counts = Counter(i["cat"] for i in info.values())
    cats = ["ok"] + [c for c, _ in counts.most_common() if c != "ok"][:4]
    ncol = 3
    fig, axes = plt.subplots(len(cats), ncol, figsize=(11, 2.6 * len(cats)),
                             constrained_layout=True, squeeze=False)
    for ri, cat in enumerate(cats):
        words = sorted([w for w, i in info.items() if i["cat"] == cat])[:ncol]
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci >= len(words):
                ax.axis("off")
                continue
            w = words[ci]; I = info[w]; d = wd[w]
            st = np.array(d["steps"], float); s = np.array(d["s"], float)
            ax.scatter(np.log10(st), s, s=5, c=C_RAW, alpha=0.35, zorder=1, linewidths=0)
            ax.plot(np.log10(I["us"]), I["means"], "-o", color=C_MEAN, ms=3, lw=1.5, zorder=4)
            if I["popt"] is not None:
                xs = np.linspace(np.log10(I["us"].min() + 1), np.log10(I["us"].max() + 1), 200)
                ax.plot(xs, -sigmoid(xs, *I["popt"]), color=C_FIT, lw=1.6, zorder=3)
            ax.axhline(I["chance"], color=C_ANCH, ls=":", lw=1.0)
            ax.axhline(I["minm"], color=C_ANCH, ls=":", lw=1.0)
            ax.axhline(I["thr"], color=C_THR, lw=1.6)
            if I["aoa"] is not None:
                ax.axvline(I["aoa"], color=C_CROSS, ls="--", lw=1.2)
            ax.set_title(f"{w}  (k={I['k']})", fontsize=8.5)
            ax.grid(True, color="#F0F0F0", lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(f"{cat}\nsurprisal (nats)", fontsize=8)
    for ci in range(ncol):
        axes[-1][ci].set_xlabel("log10(training words)", fontsize=8)
    handles = [
        plt.Line2D([], [], color=C_MEAN, marker="o", ms=3, lw=1.5, label="mean surprisal / ckpt"),
        plt.Line2D([], [], color=C_FIT, lw=1.6, label="robust sigmoid fit"),
        plt.Line2D([], [], color=C_THR, lw=1.6, label="threshold = midpoint"),
        plt.Line2D([], [], color=C_ANCH, ls=":", lw=1.0, label="anchors: k·ln(V) & min"),
        plt.Line2D([], [], color=C_CROSS, ls="--", lw=1.2, label="crossing (AoA)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"BOS-only AoA curves  —  {Path(MODEL).stem}", y=1.04, fontsize=11)
    fig.savefig(PNG, dpi=130, bbox_inches="tight")
    print(f"[plot] -> {PNG}")


if __name__ == "__main__":
    ensure_data()
    if not SURPRISAL.exists():
        extract()
    else:
        print(f"[extract] {SURPRISAL} exists, skipping extraction")
    plot()
