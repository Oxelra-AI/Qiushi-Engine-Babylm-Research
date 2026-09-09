#!/usr/bin/env python3
"""research: explicit reference-arm decomposition for the fixed-budget broad leg.

This file-only readout is a successor to research's allocation merger.  It keeps
three references separate:

* R (exact repeat): useful for measuring duplicate-recurrence cost, but it fuses
  source relatedness with exact duplication and therefore is not the clean broad
  mechanism reference.
* B (same-population independent sentence breadth): the mechanism reference for
  compact source-conditioned companions.  V-B asks whether the companion must be
  about its own source rather than simply spending the same words/rows/passes on
  unrelated text from the same population.
* C (clean filler): the absolute fixed-budget placement.  V-C says whether the
  full compact-packet stream beats matched clean experience; B-C says whether
  the breadth stream is itself different from clean text.

Identities checked whenever scores are present:

    V-R = (V-B) + (B-R)
    V-C = (V-B) + (B-C)
    B-C = (B-R) + (R-C)

The script loads existing/future official-compatible stable-family score JSONs
and the older 1x-geometry clean rows from the research harvest.  It performs no
model loading, no training, no evaluation, no GPU use, no GlobalPIQA, no
SuperGLUE, no AoA, no upload, and no leaderboard action.
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
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'

FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
NO_READING = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
EX_ENTITY_NO_READING = ["BLiMP", "Supplement", "EWoK", "COMPS"]
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
AGGREGATES = {
    "cheap6_no_GlobalPIQA": FAMILIES,
    "cheap5_no_GlobalPIQA_Reading": NO_READING,
    "exEntity5": EX_ENTITY,
    "exEntity4_noReading": EX_ENTITY_NO_READING,
    "EWoK_plus_Entity_sum": ["EWoK", "Entity"],
}
WINDOWS = {
    "common10_80": (10_000_000, 80_000_000),
    "late80_100": (80_000_000, 100_000_000),
    "endpoint80": (80_000_000, 80_000_000),
    "endpoint100": (100_000_000, 100_000_000),
}

JSON_ARMS: dict[str, dict[str, Any]] = {
    "D1_V": {
        "role": "V", "reference": "deberta_basin1", "seed": 43022,
        "root": WS / "data" / "dose_ladder_stable_eval" / "eval" / "per_target",
        "prefix": "dose_max_view",
        "description": "DeBERTa basin1 MAX compact source-conditioned view",
    },
    "D1_R": {
        "role": "R", "reference": "deberta_basin1", "seed": 43022,
        "root": WS / "data" / "dose_ladder_stable_eval" / "eval" / "per_target",
        "prefix": "dose_max_repeat",
        "description": "DeBERTa basin1 MAX exact repeat",
    },
    "D1_B": {
        "role": "B", "reference": "deberta_basin1", "seed": 43022,
        "root": WS / "data" / "breadth_entity_ewok_eval" / "eval" / "per_target",
        "prefix": "max_breadth_seed43022",
        "description": "DeBERTa basin1 MAX same-population unrelated breadth",
    },
    "D1_Cmax": {
        "role": "C", "reference": "deberta_basin1_maxgeom", "seed": 43022,
        "root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target",
        "prefix": "deberta_maxgeom_clean_seed43022",
        "description": "DeBERTa basin1 MAX-geometry clean control",
    },
    "D2_V": {
        "role": "V", "reference": "deberta_basin2", "seed": 43122,
        "root": WS / "data" / "second_basin_entity_ewok_eval" / "view_eval" / "per_target",
        "prefix": "second_basin_max_view_seed43122",
        "description": "DeBERTa basin2 MAX compact source-conditioned view",
    },
    "D2_R": {
        "role": "R", "reference": "deberta_basin2", "seed": 43122,
        "root": WS / "data" / "second_basin_entity_ewok_eval" / "repeat_eval" / "per_target",
        "prefix": "second_basin_max_repeat_seed43122",
        "description": "DeBERTa basin2 MAX exact repeat",
    },
    "D2_Cmax": {
        "role": "C", "reference": "deberta_basin2_maxgeom", "seed": 43122,
        "root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target",
        "prefix": "deberta_maxgeom_clean_seed43122",
        "description": "DeBERTa basin2 MAX-geometry clean control",
    },
    "Rbt_V": {
        "role": "V", "reference": "roberta", "seed": 43022,
        "root": WS / "data" / "roberta_viewclean_total_stable_eval" / "eval" / "per_target",
        "prefix": "roberta_viewclean_total_view",
        "description": "RoBERTa MAX compact source-conditioned view",
    },
    "Rbt_R": {
        "role": "R", "reference": "roberta", "seed": 43022,
        "root": WS / "data" / "roberta_maxdose_repeat_eval" / "eval" / "per_target",
        "prefix": "roberta_max_repeat_seed43022",
        "description": "RoBERTa MAX exact repeat",
    },
    "Rbt_C": {
        "role": "C", "reference": "roberta_maxgeom", "seed": 43022,
        "root": WS / "data" / "roberta_viewclean_total_stable_eval" / "eval" / "per_target",
        "prefix": "roberta_viewclean_total_clean",
        "description": "RoBERTa MAX-geometry clean control",
    },
}
OLD_CLEAN_CSV = WS / "data" / "score_harvest_readout" / "harvested_deberta_stable_rows.csv"
OLD_CLEAN_ARM = "D1_Cold"

PAIR_CONTRASTS = [
    ("D1_VminusR", "D1_V", "D1_R"),
    ("D1_VminusB", "D1_V", "D1_B"),
    ("D1_BminusR", "D1_B", "D1_R"),
    ("D1_VminusCold", "D1_V", OLD_CLEAN_ARM),
    ("D1_RminusCold", "D1_R", OLD_CLEAN_ARM),
    ("D1_BminusCold", "D1_B", OLD_CLEAN_ARM),
    ("D1_VminusCmax", "D1_V", "D1_Cmax"),
    ("D1_RminusCmax", "D1_R", "D1_Cmax"),
    ("D1_BminusCmax", "D1_B", "D1_Cmax"),
    ("D2_VminusR", "D2_V", "D2_R"),
    ("D2_VminusCmax", "D2_V", "D2_Cmax"),
    ("D2_RminusCmax", "D2_R", "D2_Cmax"),
    ("Rbt_VminusC", "Rbt_V", "Rbt_C"),
    ("Rbt_VminusR", "Rbt_V", "Rbt_R"),
    ("Rbt_RminusC", "Rbt_R", "Rbt_C"),
]
IDENTITIES = [
    ("D1_VR_equals_VB_plus_BR", "D1_VminusR", "D1_VminusB", "D1_BminusR", +1.0),
    ("D1_VCold_equals_VB_plus_BCold", "D1_VminusCold", "D1_VminusB", "D1_BminusCold", +1.0),
    ("D1_BCold_equals_BR_plus_RCold", "D1_BminusCold", "D1_BminusR", "D1_RminusCold", +1.0),
    ("D1_VCmax_equals_VB_plus_BCmax", "D1_VminusCmax", "D1_VminusB", "D1_BminusCmax", +1.0),
    ("D1_BCmax_equals_BR_plus_RCmax", "D1_BminusCmax", "D1_BminusR", "D1_RminusCmax", +1.0),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def ck_words(ck: str) -> int:
    return int(str(ck).split("_")[1].rstrip("M")) * 1_000_000


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_path(arm: str, ck: str) -> pathlib.Path:
    cfg = JSON_ARMS[arm]
    return pathlib.Path(cfg["root"]) / f"{cfg['prefix']}_{ck}.json"


def task_score(payload: dict[str, Any], family: str) -> float | None:
    task = (payload.get("tasks") or {}).get(family)
    if not isinstance(task, dict) or task.get("returncode") != 0:
        return None
    if family == "Reading":
        scores = task.get("scores") if isinstance(task.get("scores"), dict) else {}
        val = scores.get("Reading") if scores else task.get("score")
    else:
        val = task.get("score")
    return float(val) if finite(val) else None


def collect_json_scores() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    miss: list[dict[str, Any]] = []
    for arm, cfg in JSON_ARMS.items():
        for ck in CHECKPOINTS:
            p = payload_path(arm, ck)
            if not p.exists():
                for fam in FAMILIES:
                    miss.append({"arm": arm, "checkpoint": ck, "family": fam, "reason": "payload_missing", "path": rel(p)})
                continue
            try:
                payload = read_json(p)
            except Exception as exc:
                for fam in FAMILIES:
                    miss.append({"arm": arm, "checkpoint": ck, "family": fam, "reason": f"payload_unreadable:{exc!r}", "path": rel(p)})
                continue
            for fam in FAMILIES:
                val = task_score(payload, fam)
                base = {
                    "arm": arm,
                    "role": cfg["role"],
                    "reference": cfg["reference"],
                    "seed": cfg["seed"],
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "family": fam,
                    "payload": rel(p),
                    "description": cfg["description"],
                }
                if val is None:
                    miss.append({**base, "reason": "task_missing_or_no_score"})
                else:
                    rows.append({**base, "score": val})
    return rows, miss


def collect_old_clean() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    miss: list[dict[str, Any]] = []
    if not OLD_CLEAN_CSV.exists():
        for ck in CHECKPOINTS:
            for fam in FAMILIES:
                miss.append({"arm": OLD_CLEAN_ARM, "checkpoint": ck, "family": fam, "reason": "old_clean_csv_missing", "path": rel(OLD_CLEAN_CSV)})
        return rows, miss
    with OLD_CLEAN_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("arm")) != "clean0":
                continue
            ck = str(r.get("checkpoint"))
            for fam in FAMILIES:
                val = fnum(r.get(fam))
                base = {
                    "arm": OLD_CLEAN_ARM,
                    "role": "C",
                    "reference": "deberta_old_1x_geometry_clean",
                    "seed": 43022,
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "family": fam,
                    "payload": r.get("source_path") or rel(OLD_CLEAN_CSV),
                    "description": "Older 1x-geometry clean control from research harvest; useful historical absolute reference but geometry-confounded for MAX.",
                }
                if val is None:
                    miss.append({**base, "reason": "old_clean_score_missing"})
                else:
                    rows.append({**base, "score": val})
    return rows, miss


def lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    return {(r["arm"], r["checkpoint"], r["family"]): float(r["score"]) for r in rows if finite(r.get("score"))}


def agg(vals: list[float | None]) -> float | None:
    if not vals or any(v is None or not finite(v) for v in vals):
        return None
    return statistics.mean([float(v) for v in vals])


def make_score_rows_with_aggs(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(score_rows)
    lut = lookup(score_rows)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for r in score_rows:
        meta.setdefault((r["arm"], r["checkpoint"]), r)
    for (arm, ck), m in sorted(meta.items()):
        for q, fams in AGGREGATES.items():
            val = agg([lut.get((arm, ck, fam)) for fam in fams])
            if val is not None:
                out.append({
                    "arm": arm,
                    "role": m["role"],
                    "reference": m["reference"],
                    "seed": m["seed"],
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "family": q,
                    "score": val,
                    "payload": m.get("payload"),
                    "description": m.get("description"),
                    "is_aggregate": True,
                })
    return out


def make_contrasts(score_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lut = {(r["arm"], r["checkpoint"], r["family"]): float(r["score"]) for r in score_rows if finite(r.get("score"))}
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    quantities = FAMILIES + list(AGGREGATES)
    for cname, a, b in PAIR_CONTRASTS:
        for ck in CHECKPOINTS:
            for q in quantities:
                ka = (a, ck, q)
                kb = (b, ck, q)
                if ka in lut and kb in lut:
                    rows.append({
                        "contrast": cname,
                        "arm_a": a,
                        "arm_b": b,
                        "checkpoint": ck,
                        "words": ck_words(ck),
                        "quantity": q,
                        "delta": lut[ka] - lut[kb],
                        "score_a": lut[ka],
                        "score_b": lut[kb],
                    })
                else:
                    missing.append({"contrast": cname, "arm_a": a, "arm_b": b, "checkpoint": ck, "words": ck_words(ck), "quantity": q, "missing_a": ka not in lut, "missing_b": kb not in lut})
    return rows, missing


def make_identity_rows(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lut = {(r["contrast"], r["checkpoint"], r["quantity"]): float(r["delta"]) for r in contrast_rows if finite(r.get("delta"))}
    rows: list[dict[str, Any]] = []
    for iname, lhs, rhs1, rhs2, sign2 in IDENTITIES:
        for ck in CHECKPOINTS:
            for q in FAMILIES + list(AGGREGATES):
                a = lut.get((lhs, ck, q))
                b = lut.get((rhs1, ck, q))
                c = lut.get((rhs2, ck, q))
                if all(finite(x) for x in [a, b, c]):
                    rows.append({
                        "identity": iname,
                        "checkpoint": ck,
                        "words": ck_words(ck),
                        "quantity": q,
                        "lhs_contrast": lhs,
                        "lhs": float(a),
                        "rhs1_contrast": rhs1,
                        "rhs1": float(b),
                        "rhs2_contrast": rhs2,
                        "rhs2": float(c),
                        "rhs_sum": float(b) + sign2 * float(c),
                        "residual": (float(b) + sign2 * float(c)) - float(a),
                    })
    return rows


def summarize_windows(rows: list[dict[str, Any]], value_key: str, id_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        for wname, (lo, hi) in WINDOWS.items():
            if lo <= int(r["words"]) <= hi:
                key = tuple([r.get(k) for k in id_keys] + [wname])
                groups.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = [float(r[value_key]) for r in rs if finite(r.get(value_key))]
        if not vals:
            continue
        rec = {id_keys[i]: key[i] for i in range(len(id_keys))}
        rec["window"] = key[-1]
        rec.update({
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "checkpoints": ";".join(r["checkpoint"] for r in sorted(rs, key=lambda x: int(x["words"]))),
        })
        out.append(rec)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt(x: Any) -> str:
    return "NA" if not finite(x) else f"{float(x):+.4f}"


def get_summary(summary_rows: list[dict[str, Any]], contrast: str, q: str, window: str) -> dict[str, Any] | None:
    return next((r for r in summary_rows if r.get("contrast") == contrast and r.get("quantity") == q and r.get("window") == window), None)


def build_md(payload: dict[str, Any]) -> str:
    sr = payload["contrast_window_summaries"]
    ir = payload["identity_window_summaries"]
    lines: list[str] = []
    lines.append("# research reference-arm decomposition readout\n\n")
    lines.append("This file-only readout fixes the reference logic before broad breadth and clean scores are interpreted. R measures duplicate recurrence; B is the mechanism reference for whether compact companions must be source-related; C places the full stream against clean finite experience.\n\n")
    lines.append(f"Score rows present including aggregates: {payload['score_row_count']}; missing raw score rows: {payload['missing_score_row_count']}. Contrast rows: {payload['contrast_row_count']}; missing contrast rows: {payload['missing_contrast_row_count']}. Identity rows: {payload['identity_row_count']}.\n\n")
    lines.append("## Selected broad quantities\n\n")
    selected = [
        ("D1_VminusB", "exEntity5", "common10_80"),
        ("D1_BminusCold", "exEntity5", "common10_80"),
        ("D1_VminusCold", "exEntity5", "common10_80"),
        ("D1_RminusCold", "exEntity5", "common10_80"),
        ("D1_VminusR", "exEntity5", "common10_80"),
        ("D1_VminusB", "cheap6_no_GlobalPIQA", "common10_80"),
        ("D1_BminusCold", "cheap6_no_GlobalPIQA", "common10_80"),
        ("D1_VminusCold", "cheap6_no_GlobalPIQA", "common10_80"),
        ("D1_VminusB", "exEntity5", "late80_100"),
        ("D1_BminusCmax", "exEntity5", "late80_100"),
        ("D1_VminusCmax", "exEntity5", "late80_100"),
        ("Rbt_VminusC", "exEntity5", "late80_100"),
    ]
    for contrast, q, w in selected:
        r = get_summary(sr, contrast, q, w)
        if r:
            lines.append(f"- {contrast} {q} {w}: n={r['n']} mean={fmt(r['mean'])} range=[{fmt(r['min'])},{fmt(r['max'])}] checkpoints={r['checkpoints']}\n")
        else:
            lines.append(f"- {contrast} {q} {w}: not yet complete\n")
    lines.append("\n## Completed identities of interest\n\n")
    for r in ir:
        if r.get("quantity") in {"Entity", "exEntity5", "cheap6_no_GlobalPIQA"} and r.get("window") in {"common10_80", "late80_100", "endpoint80"}:
            lines.append(f"- {r['identity']} {r['quantity']} {r['window']}: n={r['n']} residual mean={fmt(r['mean'])} range=[{fmt(r['min'])},{fmt(r['max'])}] checkpoints={r['checkpoints']}\n")
    lines.append("\n## Reference interpretation\n\n")
    lines.append("- If broad ex-Entity `D1_VminusB` is near zero when the breadth ladder lands, then the old positive ex-Entity `V-C` is not evidence for a companion mechanism; it mainly sits in `B-C`/stream composition or geometry.\n")
    lines.append("- If broad ex-Entity `D1_VminusB` is substantially positive across families/checkpoints, then within-packet source-related re-expression becomes the mechanism-bearing quantity. The matched clean controls then locate that mechanism relative to clean experience, but do not replace it.\n")
    lines.append("- `D1_VminusCmax` and `D2_VminusCmax` are absolute placements under matched row/update geometry. They can confirm a useful total stream effect, but a positive total with `V-B≈0` would not support source-conditioned companions.\n")
    lines.append("- The permuted-companion arm becomes high-value only if broad `V-B` survives, because it equalizes compact-rewrite fertility and multiset properties that B cannot match while breaking own-source correspondence.\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(WS / "data" / "reference_decomposition_readout"))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_scores, json_missing = collect_json_scores()
    old_scores, old_missing = collect_old_clean()
    score_rows_raw = json_scores + old_scores
    missing_raw = json_missing + old_missing
    score_rows = make_score_rows_with_aggs(score_rows_raw)
    contrast_rows, missing_contrasts = make_contrasts(score_rows)
    identity_rows = make_identity_rows(contrast_rows)
    contrast_summary = summarize_windows(contrast_rows, "delta", ["contrast", "quantity"])
    identity_summary = summarize_windows(identity_rows, "residual", ["identity", "quantity"])

    files = {
        "score_rows_csv": rel(out_dir / "score_rows_with_aggregates.csv"),
        "missing_score_rows_csv": rel(out_dir / "missing_score_rows.csv"),
        "contrast_rows_csv": rel(out_dir / "contrast_rows.csv"),
        "missing_contrast_rows_csv": rel(out_dir / "missing_contrast_rows.csv"),
        "contrast_window_summaries_csv": rel(out_dir / "contrast_window_summaries.csv"),
        "identity_rows_csv": rel(out_dir / "identity_rows.csv"),
        "identity_window_summaries_csv": rel(out_dir / "identity_window_summaries.csv"),
        "summary_json": rel(out_dir / "reference_decomposition_summary.json"),
        "summary_md": rel(out_dir / "reference_decomposition_summary.md"),
    }
    payload = {
        "status": "REFERENCE_DECOMPOSITION_READOUT_INCOMPLETE" if missing_raw or missing_contrasts else "REFERENCE_DECOMPOSITION_READOUT_COMPLETE",
        "created_utc": now(),
        "score_row_count": len(score_rows),
        "missing_score_row_count": len(missing_raw),
        "contrast_row_count": len(contrast_rows),
        "missing_contrast_row_count": len(missing_contrasts),
        "identity_row_count": len(identity_rows),
        "scientific_reading": "Use B, not R, as the broad mechanism reference: V-B isolates whether compact companions outperform same-population unrelated breadth. Use C as absolute placement and R as duplicate-recurrence cost. Do not infer a companion mechanism from V-C unless V-B is positive.",
        "contrast_window_summaries": contrast_summary,
        "identity_window_summaries": identity_summary,
        "files": files,
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_csv(out_dir / "score_rows_with_aggregates.csv", score_rows)
    write_csv(out_dir / "missing_score_rows.csv", missing_raw)
    write_csv(out_dir / "contrast_rows.csv", contrast_rows)
    write_csv(out_dir / "missing_contrast_rows.csv", missing_contrasts)
    write_csv(out_dir / "contrast_window_summaries.csv", contrast_summary)
    write_csv(out_dir / "identity_rows.csv", identity_rows)
    write_csv(out_dir / "identity_window_summaries.csv", identity_summary)
    write_json(out_dir / "reference_decomposition_summary.json", payload)
    (out_dir / "reference_decomposition_summary.md").write_text(build_md(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "score_row_count": len(score_rows),
        "contrast_row_count": len(contrast_rows),
        "identity_row_count": len(identity_rows),
        "summary_md": files["summary_md"],
        "scientific_reading": payload["scientific_reading"],
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
