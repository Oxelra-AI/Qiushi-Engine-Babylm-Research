#!/usr/bin/env python3
"""research full-surface family-alias control.

Companion to full_surface_expanded_alias_probe.py. It adds the missing
family-stable alias mode on the same expanded full-sentence task, separating
small alias-vocabulary effects from per-item identity-orbit randomization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import identity_orbit_anchor_test as base  # noqa: E402
import full_surface_expanded_alias_probe as fs  # noqa: E402

OUT_DIR = base.WORKSPACE / "data/full_surface_family_alias_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODE = "family_alias"
ARM_K = [("zero", 0), ("true", 8), ("shuffled", 8), ("exposure", 8)]
N_SEEDS = 5


def aliases_family(mode: str, ev: dict, row_key: str):
    if mode == "family_alias":
        return base.family_alias_map(ev["family_id"])
    return fs.aliases(mode, ev, row_key)

# Monkeypatch full-surface rendering to include family_alias.
fs.aliases = aliases_family


def summarize(vals):
    vals = [float(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]} if vals else {"mean": float("nan"), "std": float("nan"), "n": 0}


def main():
    t0 = time.time()
    if base.DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    train_fams, held_fams = base.load_families()
    train_events = base.select_events(fs.extract_events(train_fams), base.TRAIN_EVENT_LIMIT, 257)
    eval_train_events = base.select_events(fs.extract_events(train_fams), base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(fs.extract_events(held_fams), base.EVAL_HELD_EVENT_LIMIT, 2257)

    base_anchor = fs.make_rows(train_events, fs.ANCHOR_TEMPLATES, MODE, "family_full_train_anchor")
    for r in base_anchor:
        r["train_kind"] = "anchor"
    evals = {
        "anchor_heldfam": fs.make_rows(eval_held_events, fs.ANCHOR_TEMPLATES, MODE, "family_full_eval_anchor"),
        "probe_heldfam": fs.make_rows(eval_held_events, fs.PROBE_TEMPLATES, MODE, "family_full_eval_probe"),
        "probe_trainfam": fs.make_rows(eval_train_events, fs.PROBE_TEMPLATES, MODE, "family_full_eval_probe_trainfam"),
        "heldtemplate_heldfam": fs.make_rows(eval_held_events, fs.HELD_TEMPLATES, MODE, "family_full_eval_heldtemplate"),
    }
    all_rows = list(base_anchor)
    for rs in evals.values():
        all_rows.extend(rs)
    for arm, k in ARM_K:
        if k:
            all_rows.extend(fs.make_sparse(train_events, MODE, k, arm, 999))
    vocab = base.build_vocab({MODE: all_rows})[MODE]

    print(json.dumps({"status": "FULL_SURFACE_FAMILY_ALIAS_CONTROL_START", "device": base.DEVICE, "vocab": len(vocab), "arm_k": ARM_K, "n_seeds": N_SEEDS}), flush=True)
    raw = []
    for arm, k in ARM_K:
        for seed_idx in range(N_SEEDS):
            seed = 257650 + seed_idx * 1129 + k * 47 + base.stable_int(MODE + arm) % 809
            sparse = fs.make_sparse(train_events, MODE, k, arm, seed) if k else []
            train_rows = base_anchor + sparse
            res = base.train_eval(train_rows, evals, vocab, seed)
            res.update({"mode": MODE, "arm": arm, "k_per_template": k, "seed_idx": seed_idx, "seed": seed, "n_train_rows": len(train_rows), "n_sparse_rows": len(sparse), "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT)})
            raw.append(res)
            print(json.dumps({"mode": MODE, "arm": arm, "k": k, "seed_idx": seed_idx, "train": round(res["train_acc"], 4), "probeH": round(res["probe_heldfam"], 4), "heldT": round(res["heldtemplate_heldfam"], 4), "fit": res["train_fit"]}), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["arm"], r["k_per_template"])].append(r)
    metrics = ["train_acc", "anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"]
    summary = {}
    for (arm, k), items in grouped.items():
        key = f"{MODE}|{arm}|k{k}"
        summary[key] = {"mode": MODE, "arm": arm, "k_per_template": k, "n_runs": len(items), "fit_count": sum(x["train_fit"] for x in items)}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])

    out = {"status": "FULL_SURFACE_FAMILY_ALIAS_CONTROL", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": round(time.time() - t0, 2), "config": {"mode": MODE, "arm_k": ARM_K, "n_seeds": N_SEEDS, "device": base.DEVICE, "vocab_size": len(vocab)}, "summary": summary, "raw": raw}
    out_json = OUT_DIR / "full_surface_family_alias_control_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    lines = ["# research full-surface family-alias control\n\n", f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s. Same expanded full-sentence task, family-stable aliases only.\n\n", "| mode | arm | k/template | fit | train | probe-held | probe-trainfam | held-template |\n", "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for arm, k in ARM_K:
        s = summary[f"{MODE}|{arm}|k{k}"]
        lines.append(f"| {MODE} | {arm} | {k} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldfam']['mean']:.3f} | {s['probe_trainfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "full_surface_family_alias_control_summary.md").write_text("".join(lines))
    print("\nFULL-SURFACE FAMILY-ALIAS CONTROL SUMMARY")
    for arm, k in ARM_K:
        s = summary[f"{MODE}|{arm}|k{k}"]
        print(f"{MODE:<15s} {arm:<9s} k={k:<2d} fit={s['fit_count']}/{s['n_runs']} train={s['train_acc']['mean']:.3f} probeH={s['probe_heldfam']['mean']:.3f} probeTrain={s['probe_trainfam']['mean']:.3f} heldT={s['heldtemplate_heldfam']['mean']:.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
