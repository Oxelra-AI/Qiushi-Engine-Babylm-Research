#!/usr/bin/env python3
"""research/244: selected cheap-column panel for DeBERTa positional-package interaction.

The panel is for already-trained checkpoints. It scores or reuses four cells:

  full_compact, full_repeat, nodis_compact, nodis_repeat

at requested mature checkpoints (default 80M and 100M), then computes:

  full_delta  = full_compact - full_repeat
  nodis_delta = nodis_compact - nodis_repeat
  interaction = nodis_delta - full_delta

on cheap7 and stable family summaries. research hardened this script: reused and
new per-target payloads must have the expected run/checkpoint provenance,
completed selected tasks, finite scores, and existing prediction files before a
row can enter an interaction.

No SuperGLUE, AoA, upload, or leaderboard submission is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
WRAPPER = WS / "scripts/eval_custom_checkpoint.py"
DEFAULT_OUT = WS / "data/architecture_interaction_selected_panel"
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_REQUIRED_TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
STABLE_KEYS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum", "Supplement", "Entity", "COMPS"]
DEFAULT_CHECKPOINTS = ["chck_80M", "chck_100M"]

ARMS: dict[str, dict[str, Any]] = {
    "full_compact": {
        "run_dir": WS / "training/runs/complianttok_reinvest_seed43022_r2",
        "variant": "full_p2c_c2p_abs",
        "data_arm": "compact",
        "role": "existing legal full-DeBERTa compact cell",
    },
    "full_repeat": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
        "variant": "full_p2c_c2p_abs",
        "data_arm": "repeat",
        "role": "new legal full-DeBERTa repeat cell",
    },
    "nodis_compact": {
        "run_dir": WS / "training/runs/no_disentangle_abs_compact_deberta100M_seed43022",
        "variant": "no_disentangle_abs",
        "data_arm": "compact",
        "role": "compact cell with c2p/p2c score terms removed",
    },
    "nodis_repeat": {
        "run_dir": WS / "training/runs/no_disentangle_abs_repeat_deberta100M_seed43022",
        "variant": "no_disentangle_abs",
        "data_arm": "repeat",
        "role": "repeat cell with c2p/p2c score terms removed",
    },
}

KNOWN_EXISTING_PER_TARGETS: dict[tuple[str, str], Path] = {
    ("full_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    ("full_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def endpoint_words(ck: str) -> int:
    if not ck.startswith("chck_") or not ck.endswith("M"):
        raise ValueError(f"endpoint must look like chck_80M: {ck}")
    return int(ck[len("chck_"):-1]) * 1_000_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(x: str | Path | None) -> Path | None:
    """Resolve paths recorded in JSONs against the user root, not caller cwd."""
    if x is None or str(x) == "":
        return None
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def summary_path(out_dir: Path, arm: str, ck: str) -> Path:
    target = f"arch_{arm}_{ck}"
    return out_dir / arm / ck / f"{target}_summary.json"


def existing_per_target_path(arm_name: str, ck: str) -> Path | None:
    p = KNOWN_EXISTING_PER_TARGETS.get((arm_name, ck))
    return p if p is not None and p.exists() else None


def endpoint_exists(arm_name: str, ck: str) -> bool:
    rd = Path(ARMS[arm_name]["run_dir"])
    return (rd / "hf_model" / ck / "model.safetensors").exists() or (rd / "hf_model" / ck / "pytorch_model.bin").exists()


def read_metrics_brief(arm_name: str) -> dict[str, Any]:
    p = Path(ARMS[arm_name]["run_dir"]) / "scientific_metrics.json"
    if not p.exists():
        return {"exists": False, "path": str(p)}
    m = read_json(p)
    return {
        "exists": True,
        "path": str(p),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "saved_checkpoint_count": len(m.get("saved_checkpoints", [])),
    }


def extract_scores_from_per_target(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp: list[float] = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp.append(float(rec["score"]))
    out["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7_from_scores(scores: dict[str, Any]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return float(mean([float(v) for v in vals]))


def derived(scores: dict[str, Any]) -> dict[str, float | None]:
    b, s, e, en, c, g, r = [scores.get(k) for k in CHEAP_COLS]

    def ok(vals: list[Any]) -> bool:
        return all(v is not None and math.isfinite(float(v)) for v in vals)

    return {
        "cheap6_no_GlobalPIQA": float(mean([float(x) for x in [b, s, e, en, c, r]])) if ok([b, s, e, en, c, r]) else None,
        "cheap5_no_GlobalPIQA_Reading": float(mean([float(x) for x in [b, s, e, en, c]])) if ok([b, s, e, en, c]) else None,
        "EWoK_plus_Entity_sum": float(e) + float(en) if e is not None and en is not None else None,
    }


def validate_selected_per_target(payload: dict[str, Any], arm_name: str, ck: str, path: Path) -> dict[str, Any]:
    problems: list[str] = []
    tasks = payload.get("tasks") or {}
    if payload.get("endpoint") != ck:
        problems.append(f"endpoint mismatch: {payload.get('endpoint')} != {ck}")
    expected_run_path = resolve_recorded_path(ARMS[arm_name]["run_dir"])
    got_run_raw = payload.get("run_dir", "")
    got_run_path = resolve_recorded_path(got_run_raw)
    if got_run_raw and (got_run_path is None or expected_run_path is None or got_run_path.resolve() != expected_run_path.resolve()):
        problems.append(f"run_dir mismatch: {got_run_raw} != {expected_run_path}")
    expected_model_path = expected_run_path / "hf_model" / ck if expected_run_path is not None else None
    got_model_raw = payload.get("model_path", "")
    got_model_path = resolve_recorded_path(got_model_raw)
    if got_model_raw and (got_model_path is None or expected_model_path is None or got_model_path.resolve() != expected_model_path.resolve()):
        problems.append(f"model_path mismatch: {got_model_raw} != {expected_model_path}")
    missing_tasks = [t for t in RAW_REQUIRED_TASKS if t not in tasks]
    if missing_tasks:
        problems.append(f"missing selected tasks: {missing_tasks}")
    for t in RAW_REQUIRED_TASKS:
        rec = tasks.get(t)
        if not isinstance(rec, dict):
            continue
        if rec.get("returncode") != 0:
            problems.append(f"{t} returncode {rec.get('returncode')}")
        if t == "Reading":
            if not finite((rec.get("scores") or {}).get("Reading", rec.get("score"))):
                problems.append("Reading score missing or non-finite")
        elif not finite(rec.get("score")):
            problems.append(f"{t} score missing or non-finite")
        pred = resolve_recorded_path(rec.get("predictions"))
        if t != "Reading" and not (pred and pred.exists()):
            problems.append(f"{t} predictions missing: {rec.get('predictions')}")
    scores = extract_scores_from_per_target(payload)
    missing_scores = [c for c in CHEAP_COLS if not finite(scores.get(c))]
    if missing_scores:
        problems.append(f"cheap score fields missing/non-finite: {missing_scores}")
    return {"path": str(path), "ok": len(problems) == 0, "problems": problems, "scores": scores}


def score_record_complete(sc: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if sc.get("returncode") != 0:
        problems.append(f"returncode {sc.get('returncode')}")
    pt = resolve_recorded_path(sc.get("per_target"))
    if not (pt and pt.exists()):
        problems.append(f"per_target missing: {sc.get('per_target')}")
    scores = sc.get("scores") or {}
    for c in CHEAP_COLS:
        if not finite(scores.get(c)):
            problems.append(f"score {c} missing/non-finite: {scores.get(c)}")
    if not finite(sc.get("cheap7")):
        problems.append(f"cheap7 missing/non-finite: {sc.get('cheap7')}")
    validation = sc.get("reuse_or_payload_validation")
    if isinstance(validation, dict) and not validation.get("ok", False):
        problems.extend([f"validation: {p}" for p in validation.get("problems", [])])
    return len(problems) == 0, problems


def summary_from_existing_per_target(path: Path, arm_name: str, ck: str) -> dict[str, Any]:
    payload = read_json(path)
    validation = validate_selected_per_target(payload, arm_name, ck, path)
    scores = validation["scores"]
    return {
        "returncode": 0 if validation["ok"] else 2,
        "scores": scores,
        "cheap7": cheap7_from_scores(scores),
        "per_target": str(path),
        "summary_path": str(path),
        "loaded_from_existing_per_target": True,
        "reuse_or_payload_validation": validation,
    }


def load_summary(path: Path, arm_name: str, ck: str) -> dict[str, Any]:
    payload = read_json(path)
    rec = payload.get("record", {})
    per_target = resolve_recorded_path(rec.get("per_target"))
    validation = None
    if per_target and per_target.exists():
        try:
            validation = validate_selected_per_target(read_json(per_target), arm_name, ck, per_target)
        except Exception as exc:
            validation = {"ok": False, "problems": [f"per-target validation exception: {exc!r}"], "path": str(per_target)}
    return {
        "returncode": rec.get("returncode"),
        "scores": rec.get("scores", {}),
        "cheap7": rec.get("cheap7"),
        "per_target": rec.get("per_target"),
        "summary_path": str(path),
        "loaded_from_existing_per_target": bool(rec.get("loaded_from_existing_per_target", False)),
        "reuse_or_payload_validation": validation,
    }


def make_plan(out_dir: Path, arms: list[str], checkpoints: list[str]) -> dict[str, Any]:
    plan = {
        "status": "ARCHITECTURE_INTERACTION_SELECTED_PANEL_PLAN",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "wrapper": str(WRAPPER),
        "arms": {
            a: {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in ARMS[a].items()},
                "metrics": read_metrics_brief(a),
                "requested_endpoints": {ck: endpoint_exists(a, ck) for ck in checkpoints},
                "existing_per_target_reuse": {ck: str(existing_per_target_path(a, ck)) if existing_per_target_path(a, ck) else None for ck in checkpoints},
            }
            for a in arms
        },
        "checkpoints": checkpoints,
        "primary_interaction": "(compact - repeat)_no_disentangle_abs minus (compact - repeat)_full_p2c_c2p_abs",
        "readout_columns": ["cheap7", *STABLE_KEYS, *CHEAP_COLS],
        "interpretation_boundary": "Survival of full-like compact-minus-repeat under no_disentangle_abs shows c2p/p2c score terms are not necessary. Collapse initially implicates the whole removed score-term package plus parameter/normalization changes, not c2p versus p2c separately. EWoK_plus_Entity_sum is a score-column sum, not the item-weighted EWoK_plus_Entity pool used in bootstrap files.",
        "no_training_superglue_aoa_upload_or_leaderboard": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_panel_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def launch_one(out_dir: Path, arm_name: str, ck: str, gpu: int, force: bool) -> tuple[subprocess.Popen, Any]:
    target = f"arch_{arm_name}_{ck}"
    out_base = out_dir / arm_name / ck
    out_base.mkdir(parents=True, exist_ok=True)
    sp = summary_path(out_dir, arm_name, ck)
    if sp.exists() and not force:
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"]), None
    existing_pt = existing_per_target_path(arm_name, ck)
    if existing_pt is not None and not force:
        sc = summary_from_existing_per_target(existing_pt, arm_name, ck)
        payload = {
            "status": "REUSED_EXISTING_PER_TARGET",
            "record": {
                "target": target,
                "run_dir": str(ARMS[arm_name]["run_dir"]),
                "endpoint": ck,
                "returncode": sc["returncode"],
                "scores": sc["scores"],
                "cheap7": sc["cheap7"],
                "per_target": str(existing_pt),
                "loaded_from_existing_per_target": True,
                "reuse_or_payload_validation": sc.get("reuse_or_payload_validation"),
            },
        }
        sp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"]), None
    if not endpoint_exists(arm_name, ck):
        raise FileNotFoundError(f"Missing checkpoint for {arm_name} {ck}: {Path(ARMS[arm_name]['run_dir']) / 'hf_model' / ck}")
    cmd = [
        sys.executable, "-B", str(WRAPPER),
        "--run-dir", str(ARMS[arm_name]["run_dir"]),
        "--endpoint", ck,
        "--target", target,
        "--out-base", str(out_base),
        "--gpu", str(gpu),
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = (out_base / "panel_driver_stdout.log").open("w", encoding="utf-8")
    log.write(json.dumps({"event": "launch", "utc": now(), "arm": arm_name, "checkpoint": ck, "gpu": gpu, "cmd": cmd}) + "\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True), log


def run_panel(out_dir: Path, arms: list[str], checkpoints: list[str], force: bool) -> None:
    for ck in checkpoints:
        pending = list(arms)
        while pending:
            batch = pending[:2]
            pending = pending[2:]
            procs: list[tuple[str, subprocess.Popen, Any]] = []
            for gpu, arm_name in enumerate(batch):
                print(json.dumps({"event": "start_eval", "utc": now(), "arm": arm_name, "checkpoint": ck, "gpu": gpu}), flush=True)
                p, log = launch_one(out_dir, arm_name, ck, gpu, force)
                procs.append((arm_name, p, log))
            bad = 0
            for arm_name, p, log in procs:
                rc = p.wait()
                if log is not None:
                    log.close()
                print(json.dumps({"event": "finish_eval", "utc": now(), "arm": arm_name, "checkpoint": ck, "returncode": rc}), flush=True)
                if rc != 0:
                    bad = max(bad, rc)
            if bad:
                summarize(out_dir, arms, checkpoints)
                raise SystemExit(bad)


def delta(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        try:
            out[k] = float(av) - float(bv)
        except Exception:
            out[k] = None
    return out


def subtract_delta(x: dict[str, float | None], y: dict[str, float | None], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        xv, yv = x.get(k), y.get(k)
        out[k] = float(xv) - float(yv) if xv is not None and yv is not None else None
    return out


def summarize(out_dir: Path, arms: list[str], checkpoints: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by: dict[tuple[str, str], dict[str, Any]] = {}
    problems: list[str] = []
    for ck in checkpoints:
        for arm_name in arms:
            sp = summary_path(out_dir, arm_name, ck)
            if sp.exists():
                sc = load_summary(sp, arm_name, ck)
            else:
                existing_pt = existing_per_target_path(arm_name, ck)
                if existing_pt is not None:
                    sc = summary_from_existing_per_target(existing_pt, arm_name, ck)
                else:
                    problems.append(f"missing summary {arm_name} {ck}: {sp}")
                    continue
            complete, completeness_problems = score_record_complete(sc)
            if not complete:
                problems.append(f"incomplete/invalid summary {arm_name} {ck}: {completeness_problems}")
                continue
            scores = dict(sc.get("scores", {}))
            row = {
                "arm": arm_name,
                "variant": ARMS[arm_name]["variant"],
                "data_arm": ARMS[arm_name]["data_arm"],
                "checkpoint": ck,
                "words": endpoint_words(ck),
                "cheap7": float(sc.get("cheap7")),
                **scores,
                **derived(scores),
                "summary_path": sc.get("summary_path", str(sp)),
                "per_target": sc.get("per_target"),
                "loaded_from_existing_per_target": sc.get("loaded_from_existing_per_target", False),
                "payload_validation_ok": True,
            }
            rows.append(row)
            by[(arm_name, ck)] = row
    keys = ["cheap7", *STABLE_KEYS, *CHEAP_COLS]
    interactions: list[dict[str, Any]] = []
    for ck in checkpoints:
        needed = [("full_compact", ck), ("full_repeat", ck), ("nodis_compact", ck), ("nodis_repeat", ck)]
        if not all(k in by for k in needed):
            problems.append(f"interaction missing at {ck}")
            continue
        full_delta = delta(by[("full_compact", ck)], by[("full_repeat", ck)], keys)
        nodis_delta = delta(by[("nodis_compact", ck)], by[("nodis_repeat", ck)], keys)
        interaction = subtract_delta(nodis_delta, full_delta, keys)
        interactions.append({
            "checkpoint": ck,
            "words": endpoint_words(ck),
            "full_compact_minus_repeat": full_delta,
            "nodis_compact_minus_repeat": nodis_delta,
            "interaction_nodis_minus_full": interaction,
        })
    common_cks = [r["checkpoint"] for r in interactions]
    mean_interaction: dict[str, Any] = {}
    if common_cks:
        for label, field in [("full_compact_minus_repeat", "full_compact_minus_repeat"), ("nodis_compact_minus_repeat", "nodis_compact_minus_repeat"), ("interaction_nodis_minus_full", "interaction_nodis_minus_full")]:
            mean_interaction[label] = {}
            for k in keys:
                vals = [rec[field].get(k) for rec in interactions if rec[field].get(k) is not None]
                mean_interaction[label][k] = float(mean(vals)) if vals else None
    reading = None
    if mean_interaction:
        reading = {
            "common_checkpoints": common_cks,
            "full_cheap6_no_GlobalPIQA_delta": mean_interaction["full_compact_minus_repeat"].get("cheap6_no_GlobalPIQA"),
            "nodis_cheap6_no_GlobalPIQA_delta": mean_interaction["nodis_compact_minus_repeat"].get("cheap6_no_GlobalPIQA"),
            "interaction_cheap6_no_GlobalPIQA": mean_interaction["interaction_nodis_minus_full"].get("cheap6_no_GlobalPIQA"),
            "EWoK_plus_Entity_is_column_sum_not_mean": True,
            "interpretation": "Use stable-family interaction with paired item intervals before drawing mechanism conclusions. Survival means no_disentangle compact-minus-repeat remains comparable to full on stable families. Collapse means the whole removed score-term package/parameterization is implicated but does not identify c2p vs p2c. EWoK_plus_Entity_sum is a score-column sum; do not compare it numerically with item-weighted EWoK_plus_Entity_item_pool intervals.",
        }
    payload = {
        "status": "ARCHITECTURE_INTERACTION_SELECTED_PANEL_SUMMARY",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "arms": arms,
        "checkpoints": checkpoints,
        "row_count": len(rows),
        "problems": problems,
        "rows": rows,
        "interactions": interactions,
        "mean_over_common_checkpoints": mean_interaction,
        "reading": reading,
        "boundary": "Selected cheap-column panel only; no SuperGLUE, AoA, upload, or leaderboard submission.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "architecture_interaction_selected_panel_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cols = ["arm", "variant", "data_arm", "checkpoint", "words", *CHEAP_COLS, "cheap7", *STABLE_KEYS, "summary_path", "per_target"]
    with (out_dir / "architecture_interaction_selected_panel_rows.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in cols})
    md = ["# research/244 architecture-interaction selected panel", "", f"Created UTC: `{payload['created_utc']}`", "", "## Interaction", ""]
    for rec in interactions:
        md.append(f"### {rec['checkpoint']}")
        md.append(f"- full compact-minus-repeat cheap6_no_GlobalPIQA: `{rec['full_compact_minus_repeat'].get('cheap6_no_GlobalPIQA')}`")
        md.append(f"- no_disentangle compact-minus-repeat cheap6_no_GlobalPIQA: `{rec['nodis_compact_minus_repeat'].get('cheap6_no_GlobalPIQA')}`")
        md.append(f"- interaction cheap6_no_GlobalPIQA: `{rec['interaction_nodis_minus_full'].get('cheap6_no_GlobalPIQA')}`")
        md.append(f"- interaction EWoK_plus_Entity_sum: `{rec['interaction_nodis_minus_full'].get('EWoK_plus_Entity_sum')}`")
        md.append("")
    if problems:
        md.append("## Problems")
        for p in problems:
            md.append(f"- {p}")
        md.append("")
    md.append("## Boundary")
    md.append(payload["boundary"])
    md.append("Collapse initially implicates the removed positional score package plus parameterization, not c2p versus p2c separately.")
    md.append("`EWoK_plus_Entity_sum` is a score-column sum; item bootstrap files use an item-weighted pool with a different scale/weighting.")
    (out_dir / "architecture_interaction_selected_panel_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_dir / "architecture_interaction_selected_panel_summary.json"), "row_count": len(rows), "problem_count": len(problems)}, indent=2), flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; allowed {sorted(ARMS)}")
    make_plan(args.out_dir, args.arms, args.checkpoints)
    if args.plan_only:
        print(json.dumps({"status": "ARCHITECTURE_INTERACTION_SELECTED_PANEL_PLAN_WRITTEN", "out": str(args.out_dir / "selected_panel_plan.json"), "arms": args.arms, "checkpoints": args.checkpoints}, indent=2), flush=True)
        return
    if not args.summarize_only:
        run_panel(args.out_dir, args.arms, args.checkpoints, force=args.force)
    summarize(args.out_dir, args.arms, args.checkpoints)


if __name__ == "__main__":
    main()
