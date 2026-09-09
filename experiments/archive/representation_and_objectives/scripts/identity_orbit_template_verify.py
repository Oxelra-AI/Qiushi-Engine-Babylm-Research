#!/usr/bin/env python3
"""research focused verification of the per-item alias/anchor result.

Imports the research identity-orbit experiment and reruns the load-bearing arms
with more seeds and per-template evaluations. This is not a new research route;
it verifies whether the strongest result is stable across random seeds and not
driven by a single probe template.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import identity_orbit_anchor_test as base  # noqa: E402

OUT_DIR = base.WORKSPACE / "data/identity_orbit_template_verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["fixed_names", "per_item_alias"]
ARM_K = [("zero", 0), ("true", 2), ("shuffled", 2), ("exposure", 2), ("true", 8), ("shuffled", 8), ("exposure", 8)]
N_SEEDS = 5


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]}


def make_eval_sets(mode: str, eval_train_events: List[dict], eval_held_events: List[dict]) -> Dict[str, List[dict]]:
    evals: Dict[str, List[dict]] = {
        "anchor_heldfam": base.make_rows(eval_held_events, base.ANCHOR_TEMPLATES, mode, "verify_held_anchor"),
        "probe_heldfam": base.make_rows(eval_held_events, base.PROBE_TEMPLATES, mode, "verify_held_probe"),
        "probe_trainfam": base.make_rows(eval_train_events, base.PROBE_TEMPLATES, mode, "verify_train_probe"),
        "heldtemplate_heldfam": base.make_rows(eval_held_events, base.HELD_TEMPLATES, mode, "verify_heldtemplate"),
    }
    for tid in base.PROBE_TEMPLATES:
        evals[f"probe_T{tid:02d}_heldfam"] = base.make_rows(eval_held_events, [tid], mode, f"verify_probeT{tid}")
    for tid in base.HELD_TEMPLATES:
        evals[f"held_T{tid:02d}_heldfam"] = base.make_rows(eval_held_events, [tid], mode, f"verify_heldT{tid}")
    return evals


def main():
    t0 = time.time()
    if base.DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))

    train_fams, held_fams = base.load_families()
    train_events_all = base.extract_events(train_fams)
    held_events_all = base.extract_events(held_fams)
    train_events = base.select_events(train_events_all, base.TRAIN_EVENT_LIMIT, 257)
    eval_train_events = base.select_events(train_events_all, base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(held_events_all, base.EVAL_HELD_EVENT_LIMIT, 2257)

    # Build vocabularies over all rows used in this verification per mode.
    rows_by_mode = defaultdict(list)
    datasets = {}
    for mode in MODES:
        base_anchor = base.make_rows(train_events, base.ANCHOR_TEMPLATES, mode, "verify_train_anchor")
        for r in base_anchor:
            r["train_kind"] = "anchor"
        evals = make_eval_sets(mode, eval_train_events, eval_held_events)
        datasets[mode] = {"base_anchor": base_anchor, "evals": evals}
        rows_by_mode[mode].extend(base_anchor)
        for rows in evals.values():
            rows_by_mode[mode].extend(rows)
        for arm, k in ARM_K:
            if k == 0:
                continue
            rows_by_mode[mode].extend(base.make_sparse_probe_rows(train_events, mode, k, arm, 999))
    vocabs = base.build_vocab(rows_by_mode)

    raw = []
    print(json.dumps({
        "status": "TEMPLATE_VERIFY_START",
        "device": base.DEVICE,
        "modes": MODES,
        "arm_k": ARM_K,
        "n_seeds": N_SEEDS,
        "train_events": len(train_events),
        "eval_held_events": len(eval_held_events),
    }), flush=True)

    for mode in MODES:
        base_anchor = datasets[mode]["base_anchor"]
        evals = datasets[mode]["evals"]
        vocab = vocabs[mode]
        print(f"\n=== VERIFY MODE {mode} vocab={len(vocab)} ===", flush=True)
        for arm, k in ARM_K:
            for seed_idx in range(N_SEEDS):
                if k == 0:
                    sparse = []
                    arm_name = "zero"
                else:
                    arm_name = arm
                    seed_sparse = 257700 + seed_idx * 101 + k * 31 + base.stable_int(mode + arm) % 503
                    sparse = base.make_sparse_probe_rows(train_events, mode, k, arm, seed_sparse)
                seed = 257900 + seed_idx * 1301 + k * 41 + base.stable_int(mode + arm_name) % 997
                train_rows = base_anchor + sparse
                res = base.train_eval(train_rows, evals, vocab, seed)
                res.update({
                    "mode": mode,
                    "arm": arm_name,
                    "k_per_probe_template": k,
                    "seed_idx": seed_idx,
                    "seed": seed,
                    "n_train_rows": len(train_rows),
                    "n_sparse_rows": len(sparse),
                    "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT),
                })
                raw.append(res)
                print(json.dumps({
                    "mode": mode, "arm": arm_name, "k": k, "seed_idx": seed_idx,
                    "train": round(res["train_acc"], 4),
                    "probeH": round(res["probe_heldfam"], 4),
                    "heldT": round(res["heldtemplate_heldfam"], 4),
                    "fit": res["train_fit"],
                }), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"], r["k_per_probe_template"])].append(r)
    metrics = ["train_acc", "anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"]
    metrics += [f"probe_T{tid:02d}_heldfam" for tid in base.PROBE_TEMPLATES]
    metrics += [f"held_T{tid:02d}_heldfam" for tid in base.HELD_TEMPLATES]
    summary = {}
    for (mode, arm, k), items in grouped.items():
        key = f"{mode}|{arm}|k{k}"
        summary[key] = {"mode": mode, "arm": arm, "k_per_probe_template": k, "n_runs": len(items), "fit_count": sum(x["train_fit"] for x in items)}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])
        fit_items = [x for x in items if x["train_fit"]]
        if fit_items:
            for m in metrics:
                summary[key][m + "__fit_only"] = summarize([x[m] for x in fit_items if m in x])

    out = {
        "status": "IDENTITY_ORBIT_TEMPLATE_VERIFY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "config": {
            "modes": MODES,
            "arm_k": ARM_K,
            "n_seeds": N_SEEDS,
            "train_event_limit": base.TRAIN_EVENT_LIMIT,
            "eval_held_event_limit": base.EVAL_HELD_EVENT_LIMIT,
            "device": base.DEVICE,
            "model": {"emb_dim": base.EMB_DIM, "hidden_dim": base.HIDDEN_DIM, "epochs": base.EPOCHS, "batch_size": base.BATCH_SIZE, "lr": base.LR},
        },
        "summary": summary,
        "raw": raw,
    }
    out_json = OUT_DIR / "identity_orbit_template_verify_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    lines = []
    lines.append("# research focused template verification\n\n")
    lines.append(f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s. Same train/eval event selection as the main research identity-orbit run; 5 seeds for fixed-names and per-item aliases.\n\n")
    lines.append("## Aggregate held-family transfer\n\n")
    lines.append("| mode | arm | k/template | fit | train | probe-held | held-template |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for mode in MODES:
        for arm, k in ARM_K:
            key = f"{mode}|{arm}|k{k}"
            s = summary[key]
            lines.append(f"| {mode} | {arm} | {k} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append("\n## Per-item alias per-template held-family accuracy\n\n")
    lines.append("| arm | k/template | " + " | ".join([f"T{tid:02d}" for tid in base.PROBE_TEMPLATES + base.HELD_TEMPLATES]) + " |\n")
    lines.append("|---|---:|" + "---:|" * (len(base.PROBE_TEMPLATES) + len(base.HELD_TEMPLATES)) + "\n")
    for arm, k in ARM_K:
        key = f"per_item_alias|{arm}|k{k}"
        s = summary[key]
        vals = []
        for tid in base.PROBE_TEMPLATES:
            vals.append(s[f"probe_T{tid:02d}_heldfam"]["mean"])
        for tid in base.HELD_TEMPLATES:
            vals.append(s[f"held_T{tid:02d}_heldfam"]["mean"])
        lines.append(f"| {arm} | {k} | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "identity_orbit_template_verify_summary.md").write_text("".join(lines))

    print("\n" + "=" * 88)
    print("FOCUSED TEMPLATE VERIFY SUMMARY")
    print("=" * 88)
    for mode in MODES:
        for arm, k in ARM_K:
            key = f"{mode}|{arm}|k{k}"
            s = summary[key]
            print(f"{mode:<15s} {arm:<9s} k={k:<2d} fit={s['fit_count']}/{s['n_runs']} train={s['train_acc']['mean']:.3f} probeH={s['probe_heldfam']['mean']:.3f} heldT={s['heldtemplate_heldfam']['mean']:.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
