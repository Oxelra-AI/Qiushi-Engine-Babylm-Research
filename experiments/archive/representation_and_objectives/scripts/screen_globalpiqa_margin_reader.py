#!/usr/bin/env python3
"""research: GlobalPIQA all-option readout for disposable packet-screen checkpoints.

This measures whether the non-submission disposable role-switch learning screen
moves the original official-compatible GlobalPIQA hard surfaces. It performs no
training and must not be interpreted as a legal endpoint score because the
screen checkpoints add packet exposure after an already-complete 100M run.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import globalpiqa_margin_reader as gp  # noqa: E402

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
DEFAULT_OUT_ROOT = A01_WS / "data" / "screen_globalpiqa_margin_reader"

TARGETS: dict[str, dict[str, Any]] = {
    "base_reheat": {
        "label": "Base reheat 100M before disposable packet screen",
        "model_root": A01_WS / "training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M",
        "revision": None,
    },
    "screen_treatment": {
        "label": "Disposable role-switch treatment screen final (not submission-legal)",
        "model_root": A01_WS / "training/runs/disposable_role_switch_learning_screen/treatment/hf_model/screen_final",
        "revision": None,
    },
    "screen_role_fixed": {
        "label": "Disposable role-fixed control screen final (not submission-legal)",
        "model_root": A01_WS / "training/runs/disposable_role_switch_learning_screen/role_fixed/hf_model/screen_final",
        "revision": None,
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ("scores", "completions", "metadata")}
        rr.update({f"score_{i}": s for i, s in enumerate(r.get("scores", []))})
        rr.update({f"completion_{i}": c for i, c in enumerate(r.get("completions", []))})
        flat.append(rr)
    keys = sorted({k for rr in flat for k in rr.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def delta_summary(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in sorted(set(base.get("modes", {})) & set(target.get("modes", {}))):
        b = base["modes"][mode]
        t = target["modes"][mode]
        d = {}
        for key in ["accuracy", "chance_adjusted_accuracy"]:
            if key in b and key in t:
                d[key] = t[key] - b[key]
        if "always_wrong_subset" in b and "always_wrong_subset" in t:
            d["always_wrong_subset_accuracy"] = t["always_wrong_subset"].get("accuracy", 0) - b["always_wrong_subset"].get("accuracy", 0)
            d["always_wrong_subset_mean_top_minus_correct"] = t["always_wrong_subset"].get("mean_top_minus_correct", 0) - b["always_wrong_subset"].get("mean_top_minus_correct", 0)
        if "all_rows_margin_summary" in b and "all_rows_margin_summary" in t:
            d["all_rows_mean_top_minus_correct"] = t["all_rows_margin_summary"].get("mean_top_minus_correct", 0) - b["all_rows_margin_summary"].get("mean_top_minus_correct", 0)
        out[mode] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--modes", nargs="+", default=["parallel", "nonparallel"], choices=sorted(gp.TASKS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=32)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    gp.OUT_ROOT = out_root
    gp.TARGETS = {k: TARGETS[k] for k in args.targets}

    combined: dict[str, Any] = {
        "status": "SCREEN_GLOBALPIQA_MARGIN_READER_DONE",
        "created_utc": now_utc(),
        "legal_status": "not_submission_facing; measures disposable screen checkpoints only",
        "targets": {},
        "deltas_vs_base_reheat": {},
    }
    full_summaries: dict[str, Any] = {}
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "modes": args.modes}), flush=True)
        res = gp.run_target(target, args.modes, args.batch_size, args.non_causal_batch_size, args.max_items, args.threads)
        target_json = out_root / f"{target}_margins.json"
        target_json.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_only = {"label": res.get("label"), "model_root": res.get("model_root"), "revision": res.get("revision"), "modes": {}}
        for mode, md in res.get("modes", {}).items():
            rows = md.get("rows", [])
            write_rows_csv(out_root / f"{target}_{mode}_rows.csv", rows)
            summary_only["modes"][mode] = md.get("summary", {})
        combined["targets"][target] = summary_only
        full_summaries[target] = summary_only
        print(json.dumps({"event": "target_done", "target": target, "summaries": summary_only["modes"]}, indent=2), flush=True)

    if "base_reheat" in full_summaries:
        for target in full_summaries:
            if target != "base_reheat":
                combined["deltas_vs_base_reheat"][target] = delta_summary(full_summaries["base_reheat"], full_summaries[target])
    if "screen_treatment" in full_summaries and "screen_role_fixed" in full_summaries:
        combined["treatment_minus_role_fixed"] = delta_summary(full_summaries["screen_role_fixed"], full_summaries["screen_treatment"])

    combined_json = out_root / "screen_globalpiqa_margin_summary.json"
    combined_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": combined["status"], "combined_json": str(combined_json)}), flush=True)


if __name__ == "__main__":
    main()
