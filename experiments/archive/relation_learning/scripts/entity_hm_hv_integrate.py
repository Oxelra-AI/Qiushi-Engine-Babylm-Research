#!/usr/bin/env python3
"""research: finish/parse HM-HV Entity evaluation with metadata-keyed relevant-update groups.

This repairs the research parser problem: official BabyLM Entity predictions are keyed by
UID blocks such as regular_0_ops and ambiref_3_ops.  The correct stratification is by
(uid, item_index) joined to research entity_item_metadata.csv, following research.

If --run-missing-hv100 is passed, the script runs only the missing HV chck_100M official
Entity evaluation, preserving the research output directory.  It then parses all available
HM/HV prediction files and compares them with the existing seed43022 C/R/V Entity rows.
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
import pathlib
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/entity_hm_hv_integrate.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration')
research = _public_path('experiments/archive/relation_learning/data/entity_hm_hv')
META_ROWS = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_item_metadata.csv')
BASE_LATE = _public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_accuracy_by_group.csv')
STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
ENTITY_DATA = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
NLP_DATA_ROOT = _public_path('experiments/archive/initial_model_studies/data/nltk_data')
RUN_DIR = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022')
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = [
        "seed", "arm", "checkpoint", "group", "contrast", "n", "n_checkpoints",
        "accuracy_pct", "late_mean_accuracy_pct", "delta_accuracy_pct",
        "late_mean_delta_accuracy_pct", "acc_a", "acc_b", "late_mean_acc_a", "late_mean_acc_b",
        "relevant_updates", "mean_relevant_updates", "mean_total_ops", "mean_prefix_words",
        "stale_pick_pct_among_wrong_stale_available_not_gold", "source_family", "prediction_path",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def load_meta() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with META_ROWS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["uid"], int(r["item_index"]))] = r
    return out


def parse_score(text: str) -> float | None:
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5 <= val <= 105 else None
    return None


def eval_env(arm: str, ck: str, gpu: int) -> dict[str, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "NLTK_DATA": str(NLP_DATA_ROOT)}
    cache = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/cache') / f"{arm}_{ck}"
    tmp = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/tmp') / f"{arm}_{ck}"
    for key, p in {
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": tmp,
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    return env


def pred_path_for(arm: str, ck: str) -> pathlib.Path | None:
    out_dir = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/eval_output') / f"{arm}_{ck}"
    files = sorted(out_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def report_path_for(arm: str, ck: str) -> pathlib.Path | None:
    out_dir = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/eval_output') / f"{arm}_{ck}"
    files = sorted(out_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_hv100_if_missing(gpu: int, timeout: int = 900) -> dict[str, Any]:
    arm, ck = "HV", "chck_100M"
    if pred_path_for(arm, ck) is not None:
        return {"arm": arm, "checkpoint": ck, "status": "already_present", "predictions": rel(pred_path_for(arm, ck))}
    model_path = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model') / ck
    if not model_path.exists():
        return {"arm": arm, "checkpoint": ck, "status": "model_not_found", "model_path": rel(model_path)}
    out_dir = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/eval_output') / f"{arm}_{ck}"
    log_path = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/logs') / f"entity_{arm}_{ck}_step036.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    revision = f"step036_{arm}_{ck}_Entity"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(out_dir.resolve()),
    ]
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "ck": ck}) + "\n")
        try:
            proc = subprocess.run(argv, cwd=str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')), env=eval_env(arm, ck, gpu),
                                  stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            return {"arm": arm, "checkpoint": ck, "status": "timeout", "elapsed": round(time.time()-t0, 1), "log": rel(log_path)}
    report = report_path_for(arm, ck)
    pred = pred_path_for(arm, ck)
    score = parse_score(report.read_text(encoding="utf-8", errors="replace")) if report else None
    return {"arm": arm, "checkpoint": ck, "status": "ok" if rc == 0 and pred and score is not None else f"exit_{rc}",
            "elapsed": round(time.time()-t0, 1), "score": score, "predictions": rel(pred) if pred else None,
            "report": rel(report) if report else None, "log": rel(log_path), "rc": rc}


def available_prediction_records() -> list[dict[str, Any]]:
    rows = []
    for arm in ["HM", "HV"]:
        for ck in CKS:
            pred = pred_path_for(arm, ck)
            report = report_path_for(arm, ck)
            score = parse_score(report.read_text(encoding="utf-8", errors="replace")) if report else None
            rows.append({
                "arm": arm,
                "checkpoint": ck,
                "available": pred is not None,
                "predictions": rel(pred) if pred else "",
                "report": rel(report) if report else "",
                "score": score if score is not None else "",
            })
    return rows


def load_prediction_rows(meta: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_meta: list[tuple[str, int]] = []
    for rec in available_prediction_records():
        if not rec["available"]:
            continue
        arm = rec["arm"]
        ck = rec["checkpoint"]
        pred_path = ROOT / rec["predictions"]
        obj = read_json(pred_path)
        for uid, block in obj.items():
            preds = block.get("predictions", []) if isinstance(block, dict) else block
            for i, pr in enumerate(preds):
                key = (uid, i)
                if key not in meta:
                    missing_meta.append(key)
                    continue
                m = meta[key]
                pred = str(pr.get("pred", "") if isinstance(pr, dict) else pr)
                gold = str(m.get("gold", ""))
                stale = str(m.get("stale_initial", ""))
                rows.append({
                    "seed": 43022,
                    "arm": arm,
                    "checkpoint": ck,
                    "uid": uid,
                    "item_index": i,
                    "correct": int(norm(pred) == norm(gold)),
                    "reported_numops": int(m["reported_numops"]),
                    "relevant_updates": int(m["relevant_updates"]),
                    "total_ops": int(m["total_ops"]),
                    "prefix_words": int(m["prefix_words"]),
                    "stale_available": int(m.get("stale_available", 0)),
                    "stale_is_gold": int(m.get("stale_is_gold", 0)),
                    "pred_is_stale_initial": int(norm(pred) == norm(stale) and stale != ""),
                    "source_family": "hm_hv_partial",
                    "prediction_path": rel(pred_path),
                })
    if missing_meta:
        raise RuntimeError({"missing_metadata_examples": missing_meta[:10], "count": len(missing_meta)})
    return rows


def groups_for(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    total = int(r["total_ops"])
    groups = ["ALL", f"rel_updates_{relu}", f"total_ops_{total}"]
    if relu == 0:
        groups.append("rel_eq0")
        groups.append(f"rel0_total_ops_{total}")
    if relu >= 1:
        groups.append("rel_ge1")
    if relu >= 2:
        groups.append("rel_ge2")
    if relu >= 3:
        groups.append("rel_ge3")
    if relu >= 4:
        groups.append("rel_ge4")
    return groups


def summarize_by_checkpoint(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in groups_for(r):
            bins[(r["arm"], r["checkpoint"], g)].append(r)
    out = []
    for (arm, ck, group), vals in sorted(bins.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_wrong = [v for v in wrong if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        out.append({
            "seed": 43022,
            "arm": arm,
            "checkpoint": ck,
            "group": group,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_prefix_words": mean([v["prefix_words"] for v in vals]),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in stale_wrong) / len(stale_wrong) if stale_wrong else float("nan"),
        })
    return out


def late_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary:
        bins[(r["arm"], r["group"])].append(r)
    out = []
    for (arm, group), vals in sorted(bins.items()):
        vals = sorted(vals, key=lambda r: r["checkpoint"])
        out.append({
            "seed": 43022,
            "arm": arm,
            "group": group,
            "n_checkpoints": len(vals),
            "checkpoints": ";".join(v["checkpoint"] for v in vals),
            "n": int(vals[0]["n"]),
            "late_mean_accuracy_pct": mean([v["accuracy_pct"] for v in vals]),
            "mean_relevant_updates": mean([v["mean_relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["mean_total_ops"] for v in vals]),
            "mean_prefix_words": mean([v["mean_prefix_words"] for v in vals]),
            "late_mean_stale_pick_pct_among_wrong_stale_available_not_gold": mean([v["stale_pick_pct_among_wrong_stale_available_not_gold"] for v in vals]),
        })
    return out


def shared_hm_hv_contrasts(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Direct same-checkpoint HM-HV contrasts; robust to HV chck_100M absence.
    idx = {(r["arm"], r["checkpoint"], r["group"]): r for r in summary}
    groups = sorted({r["group"] for r in summary})
    rows = []
    for g in groups:
        vals = []
        for ck in CKS:
            a = idx.get(("HM", ck, g))
            b = idx.get(("HV", ck, g))
            if a and b:
                vals.append((ck, float(a["accuracy_pct"]), float(b["accuracy_pct"]), int(a["n"])))
        if vals:
            rows.append({
                "seed": 43022,
                "contrast": "HMminusHV",
                "group": g,
                "n": vals[0][3],
                "n_checkpoints": len(vals),
                "checkpoints": ";".join(v[0] for v in vals),
                "late_mean_delta_accuracy_pct": mean([v[1]-v[2] for v in vals]),
                "late_mean_acc_a": mean([v[1] for v in vals]),
                "late_mean_acc_b": mean([v[2] for v in vals]),
            })
    return rows


def read_baseline_late() -> list[dict[str, Any]]:
    rows = []
    with BASE_LATE.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] in {"C", "R", "V"}:
                rows.append(r)
    return rows


def combined_contrasts(hm_late: list[dict[str, Any]], baseline_late: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = []
    for r in baseline_late:
        combined.append({**r, "source_family": "baseline_step019"})
    for r in hm_late:
        combined.append({**r, "source_family": "hm_hv_step036"})
    idx = {(r["arm"], r["group"]): r for r in combined}
    groups = sorted({r["group"] for r in combined})
    pairs = [
        ("HMminusHV", "HM", "HV"), ("HMminusC", "HM", "C"), ("HVminusC", "HV", "C"),
        ("HMminusR", "HM", "R"), ("HVminusR", "HV", "R"), ("HMminusV", "HM", "V"),
        ("HVminusV", "HV", "V"),
    ]
    out = []
    for g in groups:
        for cname, a, b in pairs:
            if (a, g) not in idx or (b, g) not in idx:
                continue
            ra, rb = idx[(a, g)], idx[(b, g)]
            out.append({
                "seed": 43022,
                "group": g,
                "contrast": cname,
                "n": int(ra["n"]),
                "n_checkpoints_a": int(ra["n_checkpoints"]),
                "n_checkpoints_b": int(rb["n_checkpoints"]),
                "checkpoints_a": ra.get("checkpoints", "80M;90M;100M"),
                "checkpoints_b": rb.get("checkpoints", "80M;90M;100M"),
                "late_mean_delta_accuracy_pct": float(ra["late_mean_accuracy_pct"]) - float(rb["late_mean_accuracy_pct"]),
                "late_mean_acc_a": float(ra["late_mean_accuracy_pct"]),
                "late_mean_acc_b": float(rb["late_mean_accuracy_pct"]),
                "mean_relevant_updates": float(ra["mean_relevant_updates"]),
                "mean_total_ops": float(ra["mean_total_ops"]),
                "note": "HM/HV use available checkpoints; use shared_hm_hv_contrasts.csv for exact HM-HV shared-checkpoint comparison.",
            })
    return out


def make_note(pred_recs: list[dict[str, Any]], late: list[dict[str, Any]], shared: list[dict[str, Any]], comb: list[dict[str, Any]], run_result: dict[str, Any] | None) -> None:
    def get_late(arm: str, group: str) -> dict[str, Any] | None:
        for r in late:
            if r["arm"] == arm and r["group"] == group:
                return r
        return None
    def get_shared(group: str) -> dict[str, Any] | None:
        for r in shared:
            if r["group"] == group:
                return r
        return None
    def get_comb(c: str, group: str) -> dict[str, Any] | None:
        for r in comb:
            if r["contrast"] == c and r["group"] == group:
                return r
        return None
    lines = []
    lines.append("# research HM/HV Entity relevant-update integration")
    lines.append("")
    lines.append("This note repairs the research Entity parser by joining official prediction blocks keyed by UID to `entity_item_metadata.csv` with `(uid,item_index)`, following the research parser. It treats the HM/HV result as a test of identity-practice increments in Entity behavior, not as evidence for a single trust component.")
    lines.append("")
    if run_result:
        lines.append("## Missing checkpoint run")
        lines.append("")
        lines.append(f"HV chck_100M run status: `{run_result.get('status')}`, score: `{run_result.get('score')}`. Log/report/predictions are recorded in the JSON result.")
        lines.append("")
    lines.append("## Available official predictions")
    lines.append("")
    lines.append("| arm | checkpoint | available | score |")
    lines.append("|---|---|---:|---:|")
    for r in pred_recs:
        lines.append(f"| {r['arm']} | {r['checkpoint']} | {r['available']} | {r['score']} |")
    lines.append("")
    lines.append("## Late accuracy by relevant-update group")
    lines.append("")
    lines.append("| group | HM acc | HV acc | HM−HV shared ckpts | C acc | R acc | V acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    # baseline rows are embedded only in combined contrasts; read directly for table values
    base = { (r["arm"], r["group"]): r for r in read_baseline_late() }
    for group in ["ALL", "rel_eq0", "rel_ge1", "rel_ge2", "rel_ge3", "rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5"]:
        hm = get_late("HM", group); hv = get_late("HV", group); sh = get_shared(group)
        c = base.get(("C", group)); rr = base.get(("R", group)); v = base.get(("V", group))
        fmt = lambda x: f"{float(x):.2f}" if x not in [None, ""] else ""
        lines.append("| " + " | ".join([
            group,
            fmt(hm.get("late_mean_accuracy_pct") if hm else None),
            fmt(hv.get("late_mean_accuracy_pct") if hv else None),
            fmt(sh.get("late_mean_delta_accuracy_pct") if sh else None),
            fmt(c.get("late_mean_accuracy_pct") if c else None),
            fmt(rr.get("late_mean_accuracy_pct") if rr else None),
            fmt(v.get("late_mean_accuracy_pct") if v else None),
        ]) + " |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    sh0 = get_shared("rel_eq0")
    shge3 = get_shared("rel_ge3")
    hm0 = get_late("HM", "rel_eq0"); hv0 = get_late("HV", "rel_eq0")
    hm_all = get_late("HM", "ALL"); hv_all = get_late("HV", "ALL")
    if sh0 and shge3:
        lines.append(f"Across checkpoints shared by HM and HV, HM is above HV at zero relevant updates by {float(sh0['late_mean_delta_accuracy_pct']):+.2f} points and at rel≥3 by {float(shge3['late_mean_delta_accuracy_pct']):+.2f} points. The zero-update difference is large enough to support an identity-practice increment in Entity behavior, but it does not unify the compact readout and unrelated-source effects into one latent component.")
    if hm0 and hv0:
        lines.append(f"Late rel_eq0 means are HM {float(hm0['late_mean_accuracy_pct']):.2f}% over {hm0['n_checkpoints']} checkpoint(s) and HV {float(hv0['late_mean_accuracy_pct']):.2f}% over {hv0['n_checkpoints']} checkpoint(s). Overall Entity means are HM {float(hm_all['late_mean_accuracy_pct']):.2f}% and HV {float(hv_all['late_mean_accuracy_pct']):.2f}%.")
    lines.append("The result should be described as partial if HV chck_100M is missing; if the missing run completed, the table above records three-checkpoint HV means. In either case, this readout is a behavioral companion to the relation-typed readout evidence rather than a headline mechanism.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for p in [
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/prediction_inventory.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_prediction_rows.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_accuracy_by_checkpoint_group.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_late_accuracy_by_group.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_hm_hv_shared_checkpoint_contrasts_by_group.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_with_baseline_partial_contrasts_by_group.csv'),
        _public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/result.json'),
    ]:
        lines.append(f"- `{rel(p)}`")
    (_public_path('research/notes/relation_learning/entity_hm_hv_partial_integration.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-missing-hv100", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_result = None
    if args.run_missing_hv100:
        run_result = run_hv100_if_missing(args.gpu, args.timeout)
        write_json(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/hv100_run_result.json'), run_result)
    pred_recs = available_prediction_records()
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/prediction_inventory.csv'), pred_recs)
    meta = load_meta()
    pred_rows = load_prediction_rows(meta)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_prediction_rows.csv'), pred_rows)
    summary = summarize_by_checkpoint(pred_rows)
    late = late_summary(summary)
    shared = shared_hm_hv_contrasts(summary)
    base = read_baseline_late()
    comb = combined_contrasts(late, base)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_accuracy_by_checkpoint_group.csv'), summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_late_accuracy_by_group.csv'), late)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_hm_hv_shared_checkpoint_contrasts_by_group.csv'), shared)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_with_baseline_partial_contrasts_by_group.csv'), comb)
    result = {
        "status": "ENTITY_HM_HV_INTEGRATION_DONE",
        "created_utc": now(),
        "run_missing_hv100": run_result,
        "prediction_inventory": rel(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/prediction_inventory.csv')),
        "n_prediction_rows": len(pred_rows),
        "n_checkpoint_group_rows": len(summary),
        "n_late_rows": len(late),
        "note": rel(_public_path('research/notes/relation_learning/entity_hm_hv_partial_integration.md')),
        "key_groups": {
            "shared_HMminusHV_rel_eq0": next((r for r in shared if r["group"] == "rel_eq0"), None),
            "shared_HMminusHV_rel_ge3": next((r for r in shared if r["group"] == "rel_ge3"), None),
            "HM_rel_eq0": next((r for r in late if r["arm"] == "HM" and r["group"] == "rel_eq0"), None),
            "HV_rel_eq0": next((r for r in late if r["arm"] == "HV" and r["group"] == "rel_eq0"), None),
        },
        "files": {
            "checkpoint_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_accuracy_by_checkpoint_group.csv')),
            "late_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_late_accuracy_by_group.csv')),
            "shared_contrasts": rel(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_hm_hv_shared_checkpoint_contrasts_by_group.csv')),
            "baseline_contrasts": rel(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_with_baseline_partial_contrasts_by_group.csv')),
        },
    }
    write_json(_public_path('experiments/archive/relation_learning/data/entity_hm_hv_integration/result.json'), result)
    make_note(pred_recs, late, shared, comb, run_result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
