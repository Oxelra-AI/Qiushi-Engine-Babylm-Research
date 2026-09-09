#!/usr/bin/env python3
"""research: selected official-compatible readout panel for extractive-view DeBERTa arms.

This script is prepared while research source-only trainings are running.  It does
not train, upload, run SuperGLUE, run AoA, or submit to the leaderboard.  Once
the balanced and wide extractive runs are complete, it can score the late-band
the checkpoints on the cheap official-compatible columns and summarize the source-only
contrast against the legal compact reference. The clean-Qwen 80M control is optional context only and is not part of the default primary panel.

Scientific role: determine whether source-only selection at either side of the
research density/coverage tradeoff can reproduce the stable selected competence of
natural generated compact views under the research stock DeBERTa recipe.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY
WRAPPER = WS / "scripts/eval_custom_checkpoint.py"
DEFAULT_OUT = WS / "data/extractive_selected_eval_panel"
MANIFEST_100M = WS / "data/extractive_view_100m_streams/extractive_100m_streams_manifest.json"
PREFLIGHT_224 = WS / "data/extractive_view_pool_preflight/extractive_view_pool_preflight.json"
A01_REPEAT_POSITION = ROOT / "experiments/archive/representation_and_objectives/data/repeat_position_coverage_audit/repeat_position_coverage_audit.json"
A01_TRIANGLE = ROOT / "experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/triangle_noaoa_summary.json"

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE_KEYS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "Supplement", "Entity", "COMPS"]
DEFAULT_CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]


@dataclass(frozen=True)
class Arm:
    name: str
    run_dir: Path
    role: str
    treatment: str
    default_include: bool = True
    matched_tokenizer_coordinate: bool = True


ARMS: dict[str, Arm] = {
    "extractive_balanced": Arm(
        name="extractive_balanced",
        run_dir=WS / "training/runs/extractive_balanced_deberta100M_seed43022",
        role="density/token-mass-nearest source-only view",
        treatment="source words in source order, compact-like density, higher source coverage, no source-absent content",
    ),
    "extractive_wide": Arm(
        name="extractive_wide",
        run_dir=WS / "training/runs/extractive_wide_deberta100M_seed43022",
        role="coverage/content-maximized source-only view",
        treatment="source words in source order, high density and near-complete source coverage, no source-absent content",
    ),
    "legal_compact": Arm(
        name="legal_compact",
        run_dir=WS / "training/runs/complianttok_reinvest_seed43022_r2",
        role="natural generated compact-view legal reference under research stock DeBERTa recipe",
        treatment="source plus generated compact views with saved words reinvested into additional legal rows",
    ),
    "bridge_compactgap": Arm(
        name="bridge_compactgap",
        run_dir=WS / "training/runs/bridge_compactgap_deberta100M_seed43022",
        role="compact-gap-matched adjacency source-only bridge (gap1=0.777)",
        treatment="source words in source order optimized for adjacency, zero novel content, compact-like gap1 and span",
    ),
    "bridge_mid": Arm(
        name="bridge_mid",
        run_dir=WS / "training/runs/bridge_mid_deberta100M_seed43022",
        role="intermediate adjacency source-only bridge (gap1=0.671)",
        treatment="source words in source order with intermediate adjacency, zero novel content",
    ),
    "clean_qwen_control": Arm(
        name="clean_qwen_control",
        run_dir=WS / "training/runs/complianttok_cleanqwen_seed43022_80M",
        role="fixed-tokenizer clean-Qwen 80M contextual control; optional effect-size context only, not a default late-band primary comparator",
        treatment="no FineWeb generated compact changed block; tokenizer fitted on reinvest pool; available only through 80M",
        default_include=False,
    ),
    "a01_hash_repeat_context": Arm(
        name="a01_hash_repeat_context",
        run_dir=ROOT / "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2",
        role="historical hash-rotated repeat context from A01; use as contextual mechanism reference, not matched-tokenizer primary comparator",
        treatment="hash-rotated cyclic source repetition, lower content fraction than compact and different tokenizer label",
        default_include=False,
        matched_tokenizer_coordinate=False,
    ),
}

KNOWN_EXISTING_PER_TARGETS: dict[tuple[str, str], Path] = {
    ("legal_compact", "chck_70M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    ("legal_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    ("legal_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def endpoint_words(endpoint: str) -> int:
    if not endpoint.startswith("chck_") or not endpoint.endswith("M"):
        raise ValueError(f"endpoint must look like chck_80M: {endpoint}")
    return int(endpoint[len("chck_"):-1]) * 1_000_000



def summary_path(out_dir: Path, arm: str, ck: str) -> Path:
    target = f"step226_{arm}_{ck}"
    return out_dir / arm / ck / f"{target}_summary.json"

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
    return float(mean(vals))


def load_per_target_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_from_per_target(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    scores = extract_scores_from_per_target(payload)
    return {
        "returncode": 0,
        "cheap7": cheap7_from_scores(scores),
        "scores": scores,
        "per_target": str(path),
        "summary_path": str(path),
        "loaded_from_existing_per_target": True,
    }


def existing_per_target_path(arm_name: str, ck: str) -> Path | None:
    p = KNOWN_EXISTING_PER_TARGETS.get((arm_name, ck))
    return p if p is not None and p.exists() else None


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rec = payload.get("record", {})
    return {
        "returncode": rec.get("returncode"),
        "cheap7": rec.get("cheap7"),
        "scores": rec.get("scores", {}),
        "per_target": rec.get("per_target"),
        "summary_path": str(path),
        "loaded_from_existing_per_target": bool(rec.get("loaded_from_existing_per_target", False)),
    }


def derived(scores: dict[str, Any]) -> dict[str, float | None]:
    b, s, e, en, c, g, r = [scores.get(k) for k in CHEAP_COLS]
    out: dict[str, float | None] = {}
    out["cheap6_no_GlobalPIQA"] = float(mean([b, s, e, en, c, r])) if all(v is not None for v in [b, s, e, en, c, r]) else None
    out["cheap5_no_GlobalPIQA_Reading"] = float(mean([b, s, e, en, c])) if all(v is not None for v in [b, s, e, en, c]) else None
    out["EWoK_plus_Entity"] = float(e + en) if e is not None and en is not None else None
    return out


def endpoint_exists(arm: Arm, ck: str) -> bool:
    return (arm.run_dir / "hf_model" / ck / "model.safetensors").exists() or (arm.run_dir / "hf_model" / ck / "pytorch_model.bin").exists()


def read_metrics_brief(arm: Arm) -> dict[str, Any]:
    p = arm.run_dir / "scientific_metrics.json"
    if not p.exists():
        return {"exists": False, "path": str(p)}
    m = json.loads(p.read_text(encoding="utf-8"))
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


def make_plan(out_dir: Path, arms: list[str], checkpoints: list[str]) -> dict[str, Any]:
    plan_arms: dict[str, Any] = {}
    for arm_name in arms:
        arm = ARMS[arm_name]
        plan_arms[arm_name] = {
            "run_dir": str(arm.run_dir),
            "role": arm.role,
            "treatment": arm.treatment,
            "matched_tokenizer_coordinate": arm.matched_tokenizer_coordinate,
            "metrics": read_metrics_brief(arm),
              "requested_endpoints": {ck: endpoint_exists(arm, ck) for ck in checkpoints},
              "existing_per_target_reuse": {ck: str(existing_per_target_path(arm_name, ck)) if existing_per_target_path(arm_name, ck) else None for ck in checkpoints},
        }
    plan = {
        "status": "EXTRACTIVE_SELECTED_PANEL_PLAN",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "wrapper": str(WRAPPER),
        "manifest_100M": str(MANIFEST_100M),
        "preflight": str(PREFLIGHT_224),
        "a01_repeat_position_context": str(A01_REPEAT_POSITION),
        "a01_triangle_context": str(A01_TRIANGLE),
        "arms": plan_arms,
        "checkpoints": checkpoints,
        "cheap_columns": CHEAP_COLS,
        "primary_readout": {
            "stable_columns": STABLE_KEYS,
            "primary_reference": "legal_compact",
            "secondary_control": "clean_qwen_control_optional_80M_only",
            "interpretation": "source-only success means an extractive arm approaches legal_compact on stable late selected families; the existing clean-Qwen 80M arm is optional effect-size context, while compact superiority over both source-only arms remains a bundled natural-compact difference rather than an isolated single-factor proof",
        },
        "staged_execution": {
            "stage1_checkpoints": ["chck_80M", "chck_100M"],
            "why_stage1_is_lowest_cost_reliable_first_readout": "legal_compact per-target payloads already exist for both 80M and 100M, so only four extractive selected evaluations are needed to detect a clear source-only failure or near-compact signal before spending on the full 60/70/90M late band and new legal_compact 60/90 references",
            "continue_to_full_late_band_if": "stage1 is ambiguous, a source-only arm is close to legal_compact on stable families, or the two checkpoints disagree enough that temporal redistribution could change the scientific interpretation",
        },
        "no_training_upload_superglue_aoa_or_leaderboard": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_panel_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def launch_one(out_dir: Path, arm_name: str, ck: str, gpu: int, force: bool) -> tuple[subprocess.Popen, Any]:
    arm = ARMS[arm_name]
    target = f"step226_{arm_name}_{ck}"
    out_base = out_dir / arm_name / ck
    out_base.mkdir(parents=True, exist_ok=True)
    sp = summary_path(out_dir, arm_name, ck)
    if sp.exists() and not force:
        p = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
        return p, None
    existing_pt = existing_per_target_path(arm_name, ck)
    if existing_pt is not None and not force:
        scores = summary_from_per_target(load_per_target_payload(existing_pt), existing_pt)
        payload = {
            "status": "REUSED_EXISTING_PER_TARGET",
            "record": {
                "target": target,
                "run_dir": str(arm.run_dir),
                "endpoint": ck,
                "returncode": 0,
                "scores": scores["scores"],
                "cheap7": scores["cheap7"],
                "per_target": str(existing_pt),
                "loaded_from_existing_per_target": True,
            },
        }
        sp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_base / f"{target}_summary.md").write_text(f"# {target}\n\nReused existing per-target payload: `{existing_pt}`\n\ncheap7: {scores['cheap7']:.4f}\n", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
        return p, None
    if not endpoint_exists(arm, ck):
        raise FileNotFoundError(f"Missing checkpoint for {arm_name} {ck}: {arm.run_dir / 'hf_model' / ck}")
    cmd = [
        sys.executable, "-B", str(WRAPPER),
        "--run-dir", str(arm.run_dir),
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
    log_path = out_base / "panel_driver_stdout.log"
    log = log_path.open("w", encoding="utf-8")
    log.write(json.dumps({"event": "launch", "utc": now(), "arm": arm_name, "checkpoint": ck, "gpu": gpu, "cmd": cmd}) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p, log


def run_panel(out_dir: Path, arms: list[str], checkpoints: list[str], force: bool) -> None:
    # For each checkpoint, keep at most two concurrent GPU jobs.  The default arm
    # ordering pairs balanced/wide, then compact/clean, which uses both H100s.
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


def delta_dict(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        out[k] = float(av - bv) if av is not None and bv is not None else None
    return out


def summarize(out_dir: Path, arms: list[str], checkpoints: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    by_arm_ck: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
    for ck in checkpoints:
        for arm_name in arms:
            sp = summary_path(out_dir, arm_name, ck)
            if sp.exists():
                sc = load_summary(sp)
            else:
                existing_pt = existing_per_target_path(arm_name, ck)
                if existing_pt is not None:
                    sc = summary_from_per_target(load_per_target_payload(existing_pt), existing_pt)
                else:
                    problems.append(f"missing summary {arm_name} {ck}: {sp}")
                    continue
            if sc.get("returncode") not in (0, None):
                problems.append(f"nonzero returncode {arm_name} {ck}: {sc.get('returncode')}")
            scores = dict(sc.get("scores", {}))
            der = derived(scores)
            row = {
                "arm": arm_name,
                "checkpoint": ck,
                "words": endpoint_words(ck),
                "cheap7": sc.get("cheap7"),
                **scores,
                **der,
                "summary_path": sc.get("summary_path", str(sp)),
                "per_target": sc.get("per_target"),
                "loaded_from_existing_per_target": sc.get("loaded_from_existing_per_target", False),
            }
            by_arm_ck[arm_name][ck] = row
            rows.append(row)

    contrasts: list[dict[str, Any]] = []
    for ck in checkpoints:
        compact = by_arm_ck.get("legal_compact", {}).get(ck)
        clean = by_arm_ck.get("clean_qwen_control", {}).get(ck)
        for arm_name in [a for a in arms if a.startswith("extractive_")]:
            row = by_arm_ck.get(arm_name, {}).get(ck)
            if row is None:
                continue
            rec: dict[str, Any] = {"arm": arm_name, "checkpoint": ck, "words": endpoint_words(ck)}
            if compact is not None:
                rec["minus_legal_compact"] = delta_dict(row, compact, ["cheap7", *STABLE_KEYS, *CHEAP_COLS])
            if clean is not None:
                rec["minus_clean_qwen_control"] = delta_dict(row, clean, ["cheap7", *STABLE_KEYS, *CHEAP_COLS])
            contrasts.append(rec)

    arm_summary: dict[str, Any] = {}
    for arm_name in arms:
        valid = [by_arm_ck[arm_name][ck] for ck in checkpoints if ck in by_arm_ck.get(arm_name, {}) and by_arm_ck[arm_name][ck].get("cheap7") is not None]
        if not valid:
            arm_summary[arm_name] = {"n_valid": 0}
            continue
        best = max(valid, key=lambda r: r["cheap7"])
        arm_summary[arm_name] = {
            "n_valid": len(valid),
            "best_checkpoint_by_cheap7": best["checkpoint"],
            "best_cheap7": best["cheap7"],
            "best_scores": {k: best.get(k) for k in [*CHEAP_COLS, *STABLE_KEYS]},
            "late_mean": {k: float(mean([r[k] for r in valid if r.get(k) is not None])) if any(r.get(k) is not None for r in valid) else None for k in ["cheap7", *STABLE_KEYS, *CHEAP_COLS]},
        }

    def mean_over_common(arm_name: str, cks: list[str]) -> dict[str, float | None]:
        keys = ["cheap7", *STABLE_KEYS, *CHEAP_COLS]
        out: dict[str, float | None] = {}
        for k in keys:
            vals = [by_arm_ck[arm_name][ck].get(k) for ck in cks if by_arm_ck.get(arm_name, {}).get(ck, {}).get(k) is not None]
            out[k] = float(mean(vals)) if vals else None
        return out

    source_only_reading: dict[str, Any] = {}
    for arm_name in [a for a in arms if a.startswith("extractive_")]:
        common_cks = [
            ck for ck in checkpoints
            if ck in by_arm_ck.get(arm_name, {})
            and ck in by_arm_ck.get("legal_compact", {})
            and by_arm_ck[arm_name][ck].get("cheap7") is not None
            and by_arm_ck["legal_compact"][ck].get("cheap7") is not None
        ]
        if not common_cks:
            source_only_reading[arm_name] = {"ready": False, "reason": "missing common extractive/legal_compact selected rows"}
            continue
        a_late = mean_over_common(arm_name, common_cks)
        c_late = mean_over_common("legal_compact", common_cks)
        vs_compact = delta_dict(a_late, c_late, ["cheap7", *STABLE_KEYS])

        clean_cks = [ck for ck in common_cks if ck in by_arm_ck.get("clean_qwen_control", {}) and by_arm_ck["clean_qwen_control"][ck].get("cheap7") is not None]
        if clean_cks:
            a_clean_window = mean_over_common(arm_name, clean_cks)
            q_late = mean_over_common("clean_qwen_control", clean_cks)
            vs_clean = delta_dict(a_clean_window, q_late, ["cheap7", *STABLE_KEYS])
        else:
            vs_clean = {}
        near_compact_stable = bool(
            vs_compact.get("cheap6_no_GlobalPIQA") is not None and vs_compact["cheap6_no_GlobalPIQA"] >= -0.25
            and vs_compact.get("cheap5_no_GlobalPIQA_Reading") is not None and vs_compact["cheap5_no_GlobalPIQA_Reading"] >= -0.25
            and vs_compact.get("EWoK_plus_Entity") is not None and vs_compact["EWoK_plus_Entity"] >= -0.5
        )
        above_clean_stable = bool(
            vs_clean
            and vs_clean.get("cheap6_no_GlobalPIQA") is not None and vs_clean["cheap6_no_GlobalPIQA"] > 0
            and vs_clean.get("cheap5_no_GlobalPIQA_Reading") is not None and vs_clean["cheap5_no_GlobalPIQA_Reading"] > 0
        )
        source_only_reading[arm_name] = {
            "ready": True,
            "common_checkpoints_vs_legal_compact": common_cks,
            "n_common_vs_legal_compact": len(common_cks),
            "extractive_common_late_mean": a_late,
            "legal_compact_common_late_mean": c_late,
            "late_mean_minus_legal_compact": vs_compact,
            "clean_qwen_common_checkpoints": clean_cks,
            "late_mean_minus_clean_qwen_control": vs_clean or None,
            "near_compact_on_stable_common_late_mean": near_compact_stable,
            "above_clean_on_stable_common_late_mean": above_clean_stable,
            "reading": "source-only selection is competitive if it is near legal_compact on stable means over the same checkpoints; clean_qwen_control, when explicitly included, is only contextual effect-size evidence. Otherwise compact retains a bundled natural-data advantage",
        }
    payload = {
        "status": "EXTRACTIVE_SELECTED_PANEL_SUMMARY",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "arms": arms,
        "checkpoints": checkpoints,
        "row_count": len(rows),
        "problems": problems,
        "per_arm_summary": arm_summary,
        "contrasts": contrasts,
        "source_only_reading": source_only_reading,
        "primary_readout_note": "Use stable late selected families, not GlobalPIQA-only or Reading-only movement, to interpret the extractive contrast.",
        "interpretation_boundary": "If both source-only arms lag legal_compact, the result is bundled evidence about natural generated compact data and does not isolate fluency, source-absent content, coverage, or token mass alone.",
        "no_training_upload_superglue_aoa_or_leaderboard": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extractive_selected_panel_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = out_dir / "extractive_selected_panel_rows.csv"
    cols = ["arm", "checkpoint", "words", *CHEAP_COLS, "cheap7", *STABLE_KEYS, "summary_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in cols})
    md = [
        "# research extractive selected readout panel",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "This file summarizes selected cheap-column scores only. It does not include SuperGLUE, AoA, upload, or leaderboard submission.",
        "",
        "## Per-arm late summary",
        "",
    ]
    for arm_name, rec in arm_summary.items():
        md.append(f"- `{arm_name}`: n_valid={rec.get('n_valid')}, best={rec.get('best_checkpoint_by_cheap7')} cheap7={rec.get('best_cheap7')}")
    md += ["", "## Source-only reading", ""]
    for arm_name, rec in source_only_reading.items():
        md.append(f"- `{arm_name}`: {json.dumps(rec, ensure_ascii=False, sort_keys=True)}")
    (out_dir / "extractive_selected_panel_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_dir / "extractive_selected_panel_summary.json"), "row_count": len(rows), "problem_count": len(problems)}, indent=2), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=[name for name, arm in ARMS.items() if arm.default_include])
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; allowed {sorted(ARMS)}")
    make_plan(args.out_dir, args.arms, args.checkpoints)
    if args.plan_only:
        print(json.dumps({"status": "EXTRACTIVE_SELECTED_PANEL_PLAN_WRITTEN", "out": str(args.out_dir / "selected_panel_plan.json"), "arms": args.arms, "checkpoints": args.checkpoints}, indent=2), flush=True)
        return
    if not args.summarize_only:
        run_panel(args.out_dir, args.arms, args.checkpoints, force=args.force)
    summarize(args.out_dir, args.arms, args.checkpoints)


if __name__ == "__main__":
    main()
