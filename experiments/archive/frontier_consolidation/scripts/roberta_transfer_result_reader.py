#!/usr/bin/env python3
"""research: mechanical result reader for the pending RoBERTa compact-vs-repeat transfer experiment.

This file is intentionally read-only with respect to the active training/evaluation outputs unless
called later in normal read mode.  The --selftest mode used in research reads only completed scaffold
and smoke artifacts plus synthetic score payloads, without reading unfinished training outputs.

Scientific purpose: when the 100M RoBERTa compact/repeat arms and their selected official-compatible
scores arrive, read the result in the intended coordinate: exact legal exposure and checkpoints first,
then compact-minus-repeat across the late band on stable BabyLM readouts rather than a single volatile
column.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


CHECKPOINTS = [
    "chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M",
    "chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M",
]
LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
STABLE_KEYS = [
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "EWoK_plus_Entity",
    "Supplement",
    "Entity",
    "COMPS",
]
ALL_DELTA_KEYS = [
    "cheap7",
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "EWoK_plus_Entity",
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA",
    "Reading",
]

EXPECTED = {
    "word_exposure": 100_000_000,
    "actual_training_steps": 2529,
    "parameter_count": 30_528_064,
    "vocab_size": 16_384,
    "tokenizer_json_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "tokenizer_label": "compliant16k_reinvest10M",
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "lr_total_steps": 2529,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
}


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_COMPACT_RUN = WS / "training/runs/roberta_compact_reinvest_100M_seed43022"
DEFAULT_REPEAT_RUN = WS / "training/runs/roberta_repeat_compact_reinvest_100M_seed43022"
DEFAULT_INTEGRATED = WS / "data/roberta_full100m_integrated/roberta_transfer_integrated.json"
DEFAULT_SELECTED_ROOT = WS / "data/roberta_full100m_selected_eval"
DEFAULT_SCAFFOLD = WS / "data/roberta_full100m_transfer_scaffold/roberta_full100m_transfer_scaffold.json"
DEFAULT_TOKENIZER_JSON = WS / "data/compliant_tokenizer/tokenizer.json"
DEFAULT_SMOKE_METRICS = WS / "training/runs/roberta_smoke_train_2row/scientific_metrics.json"
DEFAULT_OUT_DIR = WS / "data/roberta_transfer_result_reader"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def close_float(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if not finite_number(a) or not finite_number(b):
        return False
    return abs(float(a) - float(b)) <= tol


def compare_expected(metrics: dict[str, Any]) -> dict[str, Any]:
    matches: dict[str, Any] = {}
    for k, expected in EXPECTED.items():
        observed = metrics.get(k)
        if isinstance(expected, float):
            ok = close_float(observed, expected, tol=1e-12)
        else:
            ok = observed == expected
        matches[k] = {"observed": observed, "expected": expected, "ok": ok}
    return matches


def read_training_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False}
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    rows = 0
    nonfinite_loss_rows = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows += 1
            if first is None:
                first = rec
            last = rec
            if not finite_number(rec.get("loss")):
                nonfinite_loss_rows += 1
    return {
        "exists": True,
        "rows": rows,
        "first": first,
        "last": last,
        "nonfinite_loss_rows": nonfinite_loss_rows,
        "loss_first": first.get("loss") if first else None,
        "loss_last": last.get("loss") if last else None,
        "last_cumulative_word_exposure": last.get("cumulative_word_exposure") if last else None,
    }


def read_run(run_dir: Path, tokenizer_json: Path) -> dict[str, Any]:
    metrics_path = run_dir / "scientific_metrics.json"
    hf_dir = run_dir / "hf_model"
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }
    if not metrics_path.exists():
        out["ready"] = False
        out["missing_reason"] = "scientific_metrics.json absent"
        return out

    metrics = read_json(metrics_path)
    checkpoint_dirs = {ck: (hf_dir / ck).exists() for ck in CHECKPOINTS}
    saved_names = [x.get("name") for x in metrics.get("saved_checkpoints", []) if isinstance(x, dict)]
    expected_matches = compare_expected(metrics)
    tokenizer_sha = sha256_file(tokenizer_json) if tokenizer_json.exists() else None
    log_summary = read_training_log(run_dir / "training_log.jsonl")

    problems: list[str] = []
    for k, rec in expected_matches.items():
        if not rec["ok"]:
            problems.append(f"metrics[{k}]={rec['observed']!r} expected {rec['expected']!r}")
    if tokenizer_sha != EXPECTED["tokenizer_json_sha256"]:
        problems.append(f"tokenizer sha {tokenizer_sha!r} expected {EXPECTED['tokenizer_json_sha256']!r}")
    missing_ck = [ck for ck, exists in checkpoint_dirs.items() if not exists]
    if missing_ck:
        problems.append(f"missing checkpoint dirs: {missing_ck}")
    if log_summary.get("exists"):
        if log_summary.get("rows") != EXPECTED["actual_training_steps"]:
            problems.append(f"training_log rows {log_summary.get('rows')} expected {EXPECTED['actual_training_steps']}")
        if log_summary.get("nonfinite_loss_rows"):
            problems.append(f"nonfinite loss rows {log_summary.get('nonfinite_loss_rows')}")
        if log_summary.get("last_cumulative_word_exposure") != EXPECTED["word_exposure"]:
            problems.append(f"log last cumulative words {log_summary.get('last_cumulative_word_exposure')} expected {EXPECTED['word_exposure']}")
    else:
        problems.append("training_log.jsonl absent")

    out.update({
        "ready": True,
        "status": metrics.get("status"),
        "metrics_selected_fields": {k: metrics.get(k) for k in sorted(EXPECTED)},
        "expected_matches": expected_matches,
        "tokenizer_json_sha256": tokenizer_sha,
        "saved_checkpoint_names": saved_names,
        "checkpoint_dirs": checkpoint_dirs,
        "training_log": log_summary,
        "problems": problems,
        "mechanically_sound": len(problems) == 0,
    })
    return out


def mean(vals: list[float]) -> float | None:
    vals2 = [float(v) for v in vals if finite_number(v)]
    return sum(vals2) / len(vals2) if vals2 else None


def summarize_integrated_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("per_checkpoint", [])
    by_ck = {r.get("ck"): r for r in rows if isinstance(r, dict)}
    missing_ck = [ck for ck in CHECKPOINTS if ck not in by_ck or by_ck[ck].get("missing")]
    late_rows = [by_ck[ck] for ck in LATE if ck in by_ck and not by_ck[ck].get("missing")]

    late_means: dict[str, float | None] = {}
    late_positive_counts: dict[str, int] = {}
    per_late: list[dict[str, Any]] = []
    for key in ALL_DELTA_KEYS:
        vals = [r.get("compact_minus_repeat", {}).get(key) for r in late_rows]
        late_means[key] = mean([v for v in vals if finite_number(v)])
        late_positive_counts[key] = sum(1 for v in vals if finite_number(v) and float(v) > 0)
    for r in late_rows:
        d = r.get("compact_minus_repeat", {})
        per_late.append({"ck": r.get("ck"), **{k: d.get(k) for k in ALL_DELTA_KEYS if k in d}})

    stored = payload.get("late_band_mean_compact_minus_repeat", {})
    stored_diffs: dict[str, Any] = {}
    for key, val in stored.items():
        recomputed = late_means.get(key)
        stored_diffs[key] = {
            "stored": val,
            "recomputed": recomputed,
            "abs_diff": abs(float(val) - float(recomputed)) if finite_number(val) and finite_number(recomputed) else None,
        }

    stable_vals = {k: late_means.get(k) for k in STABLE_KEYS}
    stable_positive = [k for k, v in stable_vals.items() if finite_number(v) and float(v) > 0]
    stable_negative_or_zero = [k for k, v in stable_vals.items() if finite_number(v) and float(v) <= 0]
    cheap7 = late_means.get("cheap7")
    cheap6 = late_means.get("cheap6_no_GlobalPIQA")
    cheap5 = late_means.get("cheap5_no_GlobalPIQA_Reading")
    ewok_entity = late_means.get("EWoK_plus_Entity")
    globalpiqa = late_means.get("GlobalPIQA")
    reading = late_means.get("Reading")

    stable_late_positive = (
        finite_number(cheap6) and float(cheap6) > 0 and
        finite_number(cheap5) and float(cheap5) > 0 and
        finite_number(ewok_entity) and float(ewok_entity) >= 0 and
        len(stable_positive) >= 4
    )
    volatile_carried = (
        finite_number(cheap7) and float(cheap7) > 0 and
        (not finite_number(cheap6) or float(cheap6) <= 0 or not finite_number(cheap5) or float(cheap5) <= 0)
    )
    if stable_late_positive:
        reading_text = "compact is late-positive on stable readouts in this RoBERTa coordinate; the next scientific test should be an independent-seed paired replication before treating this as a robust learning principle"
    elif volatile_carried:
        reading_text = "compact has a positive cheap7 mean without a stable-family mean; treat it as volatile-column movement rather than transfer of the compact data marginal"
    else:
        reading_text = "compact is neutral or negative on stable late readouts in this tested RoBERTa coordinate; this bounds the effect here and redirects mechanism work to content density, source-wide coverage, lexical recurrence, and diversity reinvestment without naming a single architectural cause"

    return {
        "status": payload.get("status"),
        "missing_checkpoints": missing_ck,
        "late_rows_present": [r.get("ck") for r in late_rows],
        "late_means_recomputed": late_means,
        "late_positive_counts_out_of_5": late_positive_counts,
        "stored_late_mean_diffs": stored_diffs,
        "stable_late_positive": stable_late_positive,
        "volatile_carried_readout": volatile_carried,
        "stable_positive_keys": stable_positive,
        "stable_negative_or_zero_keys": stable_negative_or_zero,
        "per_late_checkpoint_deltas": per_late,
        "scientific_reading": reading_text,
        "primary_readout_reminder": "Use late-band cheap6_no_GlobalPIQA, cheap5_no_GlobalPIQA_Reading, EWoK_plus_Entity, Supplement, Entity, and COMPS; do not promote a GlobalPIQA/Reading-only movement.",
    }


def read_integrated(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out.update({
            "ready": False,
            "scientific_reading": "selected official-compatible RoBERTa readout has not arrived at this path",
        })
        return out
    payload = read_json(path)
    out.update({"ready": True, "summary": summarize_integrated_payload(payload)})
    return out


def make_synthetic_payload(kind: str) -> dict[str, Any]:
    rows = []
    for ck in CHECKPOINTS:
        is_late = ck in LATE
        if kind == "stable_positive" and is_late:
            d = {
                "cheap7": 0.25,
                "cheap6_no_GlobalPIQA": 0.20,
                "cheap5_no_GlobalPIQA_Reading": 0.18,
                "EWoK_plus_Entity": 0.35,
                "BLiMP": 0.02,
                "Supplement": 0.25,
                "EWoK": 0.15,
                "Entity": 0.20,
                "COMPS": 0.12,
                "GlobalPIQA": -0.10,
                "Reading": 0.01,
            }
        elif kind == "volatile_carried" and is_late:
            d = {
                "cheap7": 0.15,
                "cheap6_no_GlobalPIQA": -0.05,
                "cheap5_no_GlobalPIQA_Reading": -0.08,
                "EWoK_plus_Entity": -0.20,
                "BLiMP": -0.05,
                "Supplement": -0.10,
                "EWoK": -0.10,
                "Entity": -0.10,
                "COMPS": -0.05,
                "GlobalPIQA": 1.50,
                "Reading": 0.04,
            }
        else:
            d = {k: 0.0 for k in ALL_DELTA_KEYS}
        rows.append({"ck": ck, "compact_minus_repeat": d})
    late_summary = {k: mean([r["compact_minus_repeat"].get(k) for r in rows if r["ck"] in LATE]) for k in ["cheap7", *STABLE_KEYS]}
    return {
        "status": f"SYNTHETIC_{kind}",
        "checkpoints": CHECKPOINTS,
        "late_band": LATE,
        "per_checkpoint": rows,
        "late_band_mean_compact_minus_repeat": late_summary,
    }


def selftest(scaffold_path: Path, smoke_metrics_path: Path, out_dir: Path) -> dict[str, Any]:
    scaffold = read_json(scaffold_path) if scaffold_path.exists() else None
    smoke = read_json(smoke_metrics_path) if smoke_metrics_path.exists() else None
    synthetic_stable = summarize_integrated_payload(make_synthetic_payload("stable_positive"))
    synthetic_volatile = summarize_integrated_payload(make_synthetic_payload("volatile_carried"))

    out = {
        "status": "ROBERTA_TRANSFER_RESULT_READER_SELFTEST",
        "created_utc": now_utc(),
        "running_tasks_touched": False,
        "scaffold_path": str(scaffold_path),
        "scaffold_exists": scaffold is not None,
        "scaffold_ok": scaffold.get("ok") if isinstance(scaffold, dict) else None,
        "pair100_text_different_rows": scaffold.get("pair100", {}).get("text_different_rows") if isinstance(scaffold, dict) else None,
        "pair100_word_mismatches": scaffold.get("pair100", {}).get("word_mismatches") if isinstance(scaffold, dict) else None,
        "tokenization_note": scaffold.get("tokenization_note_from_step210", {}).get("estimated_100m_total_diff_by_10x_one_pass") if isinstance(scaffold, dict) else None,
        "smoke_metrics_path": str(smoke_metrics_path),
        "smoke_exists": smoke is not None,
        "smoke_selected_fields": {k: smoke.get(k) for k in ["parameter_count", "vocab_size", "tokenizer_label", "seed", "extra_init_seed", "train_rng_seed", "lr_total_steps", "masking_curriculum", "loss_last"]} if isinstance(smoke, dict) else None,
        "synthetic_stable_positive_reader": {
            "stable_late_positive": synthetic_stable["stable_late_positive"],
            "volatile_carried_readout": synthetic_stable["volatile_carried_readout"],
            "scientific_reading": synthetic_stable["scientific_reading"],
        },
        "synthetic_volatile_reader": {
            "stable_late_positive": synthetic_volatile["stable_late_positive"],
            "volatile_carried_readout": synthetic_volatile["volatile_carried_readout"],
            "scientific_reading": synthetic_volatile["scientific_reading"],
        },
        "reader_outputs_when_real_results_arrive": [
            "mechanical summaries for compact and repeat run dirs",
            "selected-score integrated late-band recomputation",
            "stable-family versus volatile-column interpretation text",
        ],
    }
    write_json(out_dir / "reader_selftest.json", out)
    md = [
        "# research RoBERTa transfer result reader self-test",
        "",
        f"Created UTC: {out['created_utc']}",
        "",
        "This self-test did not read the active RoBERTa training directories.",
        "",
        f"Scaffold exists: {out['scaffold_exists']} / scaffold OK: {out['scaffold_ok']}",
        f"100M pair text-different rows: {out['pair100_text_different_rows']}; word mismatches: {out['pair100_word_mismatches']}",
        f"Estimated compact-minus-repeat 100M tokenization difference: {out['tokenization_note']}",
        "",
        "Synthetic stable-positive reader: " + out["synthetic_stable_positive_reader"]["scientific_reading"],
        "",
        "Synthetic volatile-carried reader: " + out["synthetic_volatile_reader"]["scientific_reading"],
        "",
        "When real results arrive, run this script in normal read mode and inspect `reader_report.json` before forming the next research action.",
    ]
    (out_dir / "reader_selftest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def read_mode(args: argparse.Namespace) -> dict[str, Any]:
    compact = read_run(args.compact_run, args.tokenizer_json)
    repeat = read_run(args.repeat_run, args.tokenizer_json)
    integrated = read_integrated(args.integrated)
    problems: list[str] = []
    for arm_name, arm in [("compact", compact), ("repeat", repeat)]:
        if not arm.get("ready"):
            problems.append(f"{arm_name}: {arm.get('missing_reason')}")
        elif not arm.get("mechanically_sound"):
            problems.append(f"{arm_name}: {arm.get('problems')}")
    if not integrated.get("ready"):
        problems.append("integrated selected readout absent")
    else:
        missing = integrated.get("summary", {}).get("missing_checkpoints", [])
        if missing:
            problems.append(f"integrated selected readout missing checkpoints: {missing}")
    payload = {
        "status": "ROBERTA_TRANSFER_RESULT_READER_REPORT",
        "created_utc": now_utc(),
        "compact_run": compact,
        "repeat_run": repeat,
        "integrated_selected_readout": integrated,
        "problems": problems,
        "ready_for_scientific_interpretation": len(problems) == 0,
        "human_submission_boundary": "No leaderboard submission is made or authorized by this reader.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "reader_report.json", payload)
    md = [
        "# research RoBERTa transfer result reader report",
        "",
        f"Created UTC: {payload['created_utc']}",
        f"Ready for interpretation: {payload['ready_for_scientific_interpretation']}",
        "",
        "## Mechanical reading",
        f"Compact ready: {compact.get('ready')} sound: {compact.get('mechanically_sound')}",
        f"Repeat ready: {repeat.get('ready')} sound: {repeat.get('mechanically_sound')}",
        f"Integrated selected readout ready: {integrated.get('ready')}",
        "",
    ]
    if problems:
        md += ["## Problems", *[f"- {p}" for p in problems], ""]
    if integrated.get("ready"):
        s = integrated["summary"]
        md += [
            "## Late-band compact-minus-repeat reading",
            f"Late means: `{json.dumps(s.get('late_means_recomputed'), sort_keys=True)}`",
            f"Stable positive keys: {s.get('stable_positive_keys')}",
            f"Stable nonpositive keys: {s.get('stable_negative_or_zero_keys')}",
            s.get("scientific_reading", ""),
        ]
    (args.out_dir / "reader_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-run", type=Path, default=DEFAULT_COMPACT_RUN)
    ap.add_argument("--repeat-run", type=Path, default=DEFAULT_REPEAT_RUN)
    ap.add_argument("--integrated", type=Path, default=DEFAULT_INTEGRATED)
    ap.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    ap.add_argument("--scaffold", type=Path, default=DEFAULT_SCAFFOLD)
    ap.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER_JSON)
    ap.add_argument("--smoke-metrics", type=Path, default=DEFAULT_SMOKE_METRICS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        out = selftest(args.scaffold, args.smoke_metrics, args.out_dir)
    else:
        out = read_mode(args)
    print(json.dumps({
        "status": out.get("status"),
        "out_dir": str(args.out_dir),
        "ready_for_scientific_interpretation": out.get("ready_for_scientific_interpretation"),
        "running_tasks_touched": out.get("running_tasks_touched"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
