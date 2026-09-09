#!/usr/bin/env python3
"""research selected evaluation and interaction reader for HS/LS/HD/LD RoBERTa factorial.

This driver evaluates already trained RoBERTa checkpoints through the existing
official-compatible cheap-column wrapper, then integrates the factorial
contrasts:

  same-anchor compact effect   = HS - LS
  deranged-anchor compact      = HD - LD
  contextual-anchor interaction = (HS - LS) - (HD - LD)

The intended scientific readout is downstream movement on stable selected
families (Supplement, EWoK, Entity, COMPS, BLiMP/cheap5/cheap6), not local NLL
or GlobalPIQA/Reading-only changes.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
DERIVED_KEYS = ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity"]
PRIMARY_KEYS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "Supplement", "EWoK", "Entity", "COMPS", "EWoK_plus_Entity"]
DEFAULT_CHECKPOINTS = ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]
DEFAULT_ARMS = ["hs", "ls", "hd", "ld"]


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
EVAL_WRAPPER = ROOT / "experiments/archive/frontier_consolidation/scripts/eval_custom_checkpoint.py"
DEFAULT_OUT = ROOT / "experiments/archive/representation_and_objectives/data/factorial_selected_eval"
RUN_DIRS = {
    "hs": ROOT / "experiments/archive/representation_and_objectives/training/runs/factorial_hs_roberta_100M_accum_mb16",
    "ls": ROOT / "experiments/archive/representation_and_objectives/training/runs/factorial_ls_roberta_100M_accum_mb16",
    "hd": ROOT / "experiments/archive/representation_and_objectives/training/runs/factorial_hd_roberta_100M_accum_mb16",
    "ld": ROOT / "experiments/archive/representation_and_objectives/training/runs/factorial_ld_roberta_100M_accum_mb16",
}
ARM_LABELS = {
    "hs": "factorial_hs_compact_same_anchors",
    "ls": "factorial_ls_repeat_same_anchors",
    "hd": "factorial_hd_compact_deranged_anchors",
    "ld": "factorial_ld_repeat_deranged_anchors",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_path(out_dir: Path, arm: str, ck: str) -> Path:
    return out_dir / arm / ck / f"factorial_{arm}_{ck}_summary.json"


def verify_training(arms: list[str], checkpoints: list[str]) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {}
    problems: list[str] = []
    for arm in arms:
        run = RUN_DIRS[arm]
        metrics_path = run / "scientific_metrics.json"
        arm_rep: dict[str, Any] = {"run_dir": str(run), "metrics_exists": metrics_path.exists(), "checkpoints": {}}
        if not metrics_path.exists():
            problems.append(f"{arm}: missing scientific_metrics.json at {metrics_path}")
        else:
            m = read_json(metrics_path)
            arm_rep.update({
                "status": m.get("status"),
                "word_exposure": m.get("word_exposure"),
                "target_word_exposure": m.get("target_word_exposure"),
                "actual_training_steps": m.get("actual_training_steps"),
                "optimizer_steps": m.get("optimizer_steps"),
                "planned_training_steps": m.get("planned_training_steps"),
                "batch_size": m.get("batch_size"),
                "effective_optimizer_batch_size": m.get("effective_optimizer_batch_size"),
                "micro_batch_size": m.get("micro_batch_size"),
                "parameter_count": m.get("parameter_count"),
                "seed": m.get("seed"),
                "extra_init_seed": m.get("extra_init_seed"),
                "train_rng_seed": m.get("train_rng_seed"),
                "loss_first": m.get("loss_first"),
                "loss_last": m.get("loss_last"),
            })
            expected = {
                "word_exposure": 99999910,
                "actual_training_steps": 2529,
                "optimizer_steps": 2529,
                "planned_training_steps": 2529,
                "batch_size": 256,
                "effective_optimizer_batch_size": 256,
                "micro_batch_size": 16,
                "parameter_count": 30528064,
                "seed": 43,
                "extra_init_seed": 43022,
                "train_rng_seed": 43023,
            }
            for k, v in expected.items():
                if arm_rep.get(k) != v:
                    problems.append(f"{arm}: {k}={arm_rep.get(k)} expected {v}")
        for ck in checkpoints:
            model_file = run / "hf_model" / ck / "model.safetensors"
            arm_rep["checkpoints"][ck] = {"model_safetensors": str(model_file), "exists": model_file.exists()}
            if not model_file.exists():
                problems.append(f"{arm}: missing {ck} model at {model_file}")
        report[arm] = arm_rep
    return report, problems


def run_one_eval(out_dir: Path, arm: str, ck: str, gpu: int, force: bool) -> subprocess.Popen:
    run = RUN_DIRS[arm]
    target = f"factorial_{arm}_{ck}"
    out_base = out_dir / arm / ck
    out_base.mkdir(parents=True, exist_ok=True)
    if summary_path(out_dir, arm, ck).exists() and not force:
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    cmd = [
        sys.executable, "-B", str(EVAL_WRAPPER),
        "--run-dir", str(run),
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
    logf = (out_base / "factorial_eval_driver_stdout.log").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)


def read_score(path: Path) -> dict[str, Any]:
    d = read_json(path)
    rec = d.get("record", {})
    scores = rec.get("scores", {})
    out = {"returncode": rec.get("returncode"), "cheap7": rec.get("cheap7"), "scores": scores, "per_target": rec.get("per_target"), "summary_path": str(path)}
    out.update(derived(scores, rec.get("cheap7")))
    return out


def derived(scores: dict[str, Any], cheap7_value: float | None = None) -> dict[str, float | None]:
    b = scores.get("BLiMP"); s = scores.get("Supplement"); e = scores.get("EWoK")
    en = scores.get("Entity"); c = scores.get("COMPS"); g = scores.get("GlobalPIQA"); r = scores.get("Reading")
    out: dict[str, float | None] = {"cheap7": cheap7_value if cheap7_value is not None else None}
    if out["cheap7"] is None and all(v is not None for v in [b, s, e, en, c, g, r]):
        out["cheap7"] = float(mean([b, s, e, en, c, g, r]))
    out["cheap6_no_GlobalPIQA"] = float(mean([b, s, e, en, c, r])) if all(v is not None for v in [b, s, e, en, c, r]) else None
    out["cheap5_no_GlobalPIQA_Reading"] = float(mean([b, s, e, en, c])) if all(v is not None for v in [b, s, e, en, c]) else None
    out["EWoK_plus_Entity"] = float(e + en) if e is not None and en is not None else None
    return out


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    # Return a-b for scores and derived values.
    out: dict[str, float | None] = {}
    for k in CHEAP_COLS:
        av = a.get("scores", {}).get(k)
        bv = b.get("scores", {}).get(k)
        out[k] = float(av - bv) if av is not None and bv is not None else None
    for k in DERIVED_KEYS:
        av = a.get(k)
        bv = b.get(k)
        out[k] = float(av - bv) if av is not None and bv is not None else None
    return out


def sub_delta(x: dict[str, float | None], y: dict[str, float | None]) -> dict[str, float | None]:
    keys = sorted(set(x) | set(y))
    return {k: (float(x[k] - y[k]) if x.get(k) is not None and y.get(k) is not None else None) for k in keys}


def integrate(out_dir: Path, arms: list[str], checkpoints: list[str], training_report: dict[str, Any] | None = None, training_problems: list[str] | None = None) -> dict[str, Any]:
    per_ck: list[dict[str, Any]] = []
    problems = list(training_problems or [])
    for ck in checkpoints:
        rec: dict[str, Any] = {"ck": ck, "arms": {}}
        missing = False
        for arm in arms:
            sp = summary_path(out_dir, arm, ck)
            if not sp.exists():
                rec["arms"][arm] = {"missing": True, "summary_path": str(sp)}
                problems.append(f"{ck}/{arm}: missing eval summary {sp}")
                missing = True
            else:
                rec["arms"][arm] = read_score(sp)
        if not missing and all(a in rec["arms"] for a in DEFAULT_ARMS):
            hs_ls = delta(rec["arms"]["hs"], rec["arms"]["ls"])
            hd_ld = delta(rec["arms"]["hd"], rec["arms"]["ld"])
            interaction = sub_delta(hs_ls, hd_ld)
            same_minus_deranged = {"HS_minus_LS": hs_ls, "HD_minus_LD": hd_ld, "interaction": interaction}
            rec.update(same_minus_deranged)
        per_ck.append(rec)

    ready = [r for r in per_ck if "interaction" in r]
    mean_all: dict[str, dict[str, float | None]] = {}
    for name in ["HS_minus_LS", "HD_minus_LD", "interaction"]:
        mean_all[name] = {}
        for k in DERIVED_KEYS + CHEAP_COLS:
            vals = [r[name].get(k) for r in ready if r[name].get(k) is not None]
            mean_all[name][k] = float(mean(vals)) if vals else None

    late = [r for r in ready if r["ck"] in ["chck_60M", "chck_80M", "chck_100M"]]
    late_mean: dict[str, dict[str, float | None]] = {}
    for name in ["HS_minus_LS", "HD_minus_LD", "interaction"]:
        late_mean[name] = {}
        for k in DERIVED_KEYS + CHEAP_COLS:
            vals = [r[name].get(k) for r in late if r[name].get(k) is not None]
            late_mean[name][k] = float(mean(vals)) if vals else None

    primary_inter = late_mean["interaction"] if late else mean_all["interaction"]
    primary_vals = {k: primary_inter.get(k) for k in PRIMARY_KEYS}
    nonnull_primary = [v for v in primary_vals.values() if v is not None]
    positive_primary = [k for k, v in primary_vals.items() if v is not None and v > 0]
    nonpositive_primary = [k for k, v in primary_vals.items() if v is not None and v <= 0]
    stable_contextual_signature = bool(
        primary_vals.get("cheap6_no_GlobalPIQA") is not None and primary_vals["cheap6_no_GlobalPIQA"] > 0 and
        primary_vals.get("cheap5_no_GlobalPIQA_Reading") is not None and primary_vals["cheap5_no_GlobalPIQA_Reading"] > 0 and
        len(positive_primary) >= 5
    )
    if stable_contextual_signature:
        reading = "positive contextual-anchor interaction on the selected stable downstream surface; inspect item/subtask movements before authorizing any extension or seed replication"
    elif nonnull_primary and (primary_vals.get("cheap6_no_GlobalPIQA") is not None and primary_vals["cheap6_no_GlobalPIQA"] <= 0) and len(nonpositive_primary) >= 4:
        reading = "no coherent positive contextual-anchor downstream interaction on the selected stable surface; this weakens the current mechanism and should redirect mechanism work rather than extending same-premise arms"
    else:
        reading = "mixed or incomplete factorial interaction; inspect per-checkpoint and per-column movement, then decide the smallest additional readout needed"

    payload = {
        "status": "FACTORIAL_SELECTED_INTERACTION",
        "created_utc": now(),
        "meaning": "Official-compatible selected cheap-column readout for the HS/LS/HD/LD RoBERTa contextual-anchor factorial. Main scientific object is downstream interaction (HS-LS)-(HD-LD), not local NLL.",
        "checkpoints": checkpoints,
        "arms": arms,
        "run_dirs": {a: str(RUN_DIRS[a]) for a in arms},
        "training_report": training_report,
        "per_checkpoint": per_ck,
        "mean_all_ready_checkpoints": mean_all,
        "late_mean_chck60_80_100_if_available": late_mean,
        "primary_interaction_values": primary_vals,
        "primary_positive_keys": positive_primary,
        "primary_nonpositive_keys": nonpositive_primary,
        "stable_contextual_signature": stable_contextual_signature,
        "scientific_reading": reading,
        "problems": problems,
        "no_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out_dir / "factorial_selected_interaction_summary.json", payload)
    md_lines = [
        "# research factorial selected interaction",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "Main contrast: `(HS-LS)-(HD-LD)` on selected downstream families.",
        "",
        f"Ready checkpoints integrated: `{[r['ck'] for r in ready]}`",
        f"Stable contextual signature: `{stable_contextual_signature}`",
        f"Primary interaction values: `{json.dumps(primary_vals, sort_keys=True)}`",
        "",
        reading,
        "",
        f"Problems: `{json.dumps(problems[:20], ensure_ascii=False)}`",
    ]
    (out_dir / "factorial_selected_interaction_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=DEFAULT_ARMS)
    ap.add_argument("--verify-training-only", action="store_true")
    ap.add_argument("--integrate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    training_report, training_problems = verify_training(args.arms, args.checkpoints)
    write_json(out_dir / "training_integrity_report.json", {"created_utc": now(), "report": training_report, "problems": training_problems})
    print(json.dumps({"event": "training_integrity", "problems": len(training_problems), "out": str(out_dir / 'training_integrity_report.json')}), flush=True)
    if args.verify_training_only:
        if training_problems:
            print(json.dumps({"status": "TRAINING_NOT_READY", "problems": training_problems[:20]}, indent=2), flush=True)
            raise SystemExit(2)
        print(json.dumps({"status": "TRAINING_READY", "arms": args.arms, "checkpoints": args.checkpoints}, indent=2), flush=True)
        return

    if not args.integrate_only:
        if training_problems:
            print(json.dumps({"status": "TRAINING_NOT_READY_FOR_EVAL", "problems": training_problems[:20]}, indent=2), flush=True)
            raise SystemExit(2)
        # For each checkpoint, evaluate HS/LS in parallel then HD/LD in parallel.
        # This keeps GPU use high while avoiding four simultaneous model evaluations.
        pairs = [("hs", "ls"), ("hd", "ld")]
        for ck in args.checkpoints:
            for left, right in pairs:
                if left not in args.arms or right not in args.arms:
                    continue
                print(json.dumps({"event": "eval_pair_start", "utc": now(), "ck": ck, "arms": [left, right]}), flush=True)
                p0 = run_one_eval(out_dir, left, ck, 0, args.force)
                p1 = run_one_eval(out_dir, right, ck, 1, args.force)
                rc0 = p0.wait(); rc1 = p1.wait()
                print(json.dumps({"event": "eval_pair_done", "utc": now(), "ck": ck, left: rc0, right: rc1}), flush=True)
                if rc0 != 0 or rc1 != 0:
                    integrate(out_dir, args.arms, args.checkpoints, training_report, training_problems)
                    raise SystemExit(max(rc0, rc1))

    payload = integrate(out_dir, args.arms, args.checkpoints, training_report, training_problems)
    print(json.dumps({"status": payload["status"], "stable_contextual_signature": payload["stable_contextual_signature"], "primary_interaction_values": payload["primary_interaction_values"], "out": str(out_dir / 'factorial_selected_interaction_summary.json')}, indent=2), flush=True)


if __name__ == "__main__":
    main()
