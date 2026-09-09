#!/usr/bin/env python3
"""research: evaluate Muon 20M screens against research legal 20M baseline.

Uses the research evaluator with --arm reinvest as template, overriding
--run-dir, --target, --endpoint, and --columns for each Muon arm.
Then compares against the fixed research 20M baseline.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')

EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
BASELINE_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json')

# Arms to evaluate
ARMS = {
    "muon_lr008": {
        "label": "Muon LR=0.008",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_seed43022_20M')),
        "target": "muon_lr008_20M",
    },
    "muon_lr012": {
        "label": "Muon LR=0.012",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr012_seed43022_20M')),
        "target": "muon_lr012_20M",
    },
}

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_20M_collate')


def extract_scores(payload: dict) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp_vals = []
    for c in GP_COLS:
        rec = tasks.get(c, {})
        if "score" in rec and rec["score"] is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def run_eval(arm_key: str, arm: dict, gpu: int) -> Path | None:
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model" / "chck_20M"
    if not (model_path / "model.safetensors").exists():
        print(f"SKIP {arm_key}: no checkpoint at {model_path}", flush=True)
        return None

    target = arm["target"]
    per_target = _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target') / f"{target}.json"
    if per_target.exists():
        print(f"EXISTS {arm_key}: {per_target}", flush=True)
        return per_target

    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", "chck_20M",
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    print(f"RUN {arm_key}: {target} gpu={gpu}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        err = (proc.stderr or "")[-3000:]
        out = (proc.stdout or "")[-2000:]
        print(f"FAIL {arm_key}:\nSTDOUT: {out}\nSTDERR: {err}", flush=True)
        return None

    if per_target.exists():
        print(f"OK {arm_key}: {per_target}", flush=True)
        return per_target
    else:
        print(f"WARN {arm_key}: evaluator succeeded but no per-target JSON at {per_target}", flush=True)
        return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="", help="Evaluate only this arm key")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8")) if BASELINE_JSON.exists() else None
    baseline_scores = extract_scores(baseline) if baseline else None
    baseline_c7 = cheap7(baseline_scores) if baseline_scores else None

    results = {}
    arms_to_run = {args.arm: ARMS[args.arm]} if args.arm and args.arm in ARMS else ARMS
    for arm_key, arm in arms_to_run.items():
        rp = run_eval(arm_key, arm, args.gpu)
        if rp and rp.exists():
            payload = json.loads(rp.read_text(encoding="utf-8"))
            scores = extract_scores(payload)
            c7 = cheap7(scores)
            deltas = {}
            for c in CHEAP_COLUMNS:
                if scores.get(c) is not None and baseline_scores and baseline_scores.get(c) is not None:
                    deltas[c] = scores[c] - baseline_scores[c]
                else:
                    deltas[c] = None
            deltas["cheap7"] = (c7 - baseline_c7) if c7 is not None and baseline_c7 is not None else None
            results[arm_key] = {"label": arm["label"], "scores": scores, "cheap7": c7, "deltas": deltas}
        else:
            results[arm_key] = {"label": arm["label"], "scores": None, "cheap7": None, "deltas": None}

    # Decision
    for arm_key, r in results.items():
        d = r.get("deltas")
        if d and d.get("cheap7") is not None:
            dc7 = float(d["cheap7"])
            damaged = any(d.get(c) is not None and d[c] <= -1.5 for c in ["EWoK", "GlobalPIQA", "Reading"])
            if dc7 >= 0.35 and not damaged:
                r["decision"] = "strong_broad_continue"
            elif dc7 >= 0.15 and not damaged:
                r["decision"] = "moderate_extend_with_review"
            elif dc7 <= -0.15 or damaged:
                r["decision"] = "close_or_rebuild"
            else:
                r["decision"] = "mixed_needs_review"
        else:
            r["decision"] = "incomplete"

    comparison = {"status": "MUON_20M_EVAL", "baseline_cheap7": baseline_c7, "baseline_scores": baseline_scores, "arms": results}
    (_public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/muon_20M_comparison.json')).write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    # Markdown summary
    lines = ["# research Muon-hidden 20M screen", ""]
    lines.append(f"Baseline (research legal 20M) cheap7: **{baseline_c7:.4f}**" if baseline_c7 else "Baseline: N/A")
    lines.append("")
    arms_v = list(results.values())
    hdr = "| column | baseline |" + "|".join(f" {v['label']} " for v in arms_v) + "|" + "|".join(f" Δ{v['label'][:6]} " for v in arms_v) + "|"
    sep = "|---|---:|" + "---:|" * len(arms_v) + "---:|" * len(arms_v)
    lines.append(hdr)
    lines.append(sep)
    for c in CHEAP_COLUMNS:
        b = f"{baseline_scores[c]:.2f}" if baseline_scores and baseline_scores.get(c) is not None else ""
        cells = [b]
        for v in arms_v:
            s = (v.get("scores") or {}).get(c)
            cells.append(f"{s:.2f}" if s is not None else "")
        for v in arms_v:
            dd = (v.get("deltas") or {}).get(c)
            cells.append(f"{dd:+.2f}" if dd is not None else "")
        lines.append("| " + c + " | " + " | ".join(cells) + " |")
    cells = [f"{baseline_c7:.4f}" if baseline_c7 else ""]
    for v in arms_v:
        cells.append(f"{v['cheap7']:.4f}" if v.get("cheap7") else "")
    for v in arms_v:
        dd = (v.get("deltas") or {}).get("cheap7")
        cells.append(f"{dd:+.4f}" if dd is not None else "")
    lines.append("| **cheap7** | " + " | ".join(cells) + " |")
    lines.append("")
    for arm_key, v in results.items():
        lines.append(f"- **{v['label']}**: `{v.get('decision','incomplete')}`")
    (_public_path('research/documents/frontier_consolidation/data/muon_20M_eval/muon_20M_comparison.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "MUON_20M_EVAL", "baseline_cheap7": baseline_c7,
                       "arms": {k: {"cheap7": v.get("cheap7"), "delta": (v.get("deltas") or {}).get("cheap7"), "decision": v.get("decision")} for k, v in results.items()},
                       "out": str(OUT_ROOT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
