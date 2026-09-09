#!/usr/bin/env python3
"""research: combined readout for split decisive GPU scoring roots.

Reads only the corrected research split scorer outputs:
  - register_decisive_gpu_eval
  - incorpus_decisive_gpu_eval
and joins them to the research matched clean reference, research register
prediction, and research in-corpus rate commitment. It intentionally excludes the
failed research/research partial scorer roots.

File-only: no model loading, no official evaluation, no training, no upload.
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


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
DEFAULT_ROOTS = [
    WS / "data" / "register_decisive_gpu_eval",
    WS / "data" / "incorpus_decisive_gpu_eval",
]
CLEAN = WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target"
PRED_REGISTER = WS / "data" / "distribution_proximity_prediction" / "prediction_commitment.json"
PRED_REGISTER_RATE = WS / "data" / "register_removal_rate_and_reference_spread" / "prediction_commitment.json"
PRED_RATE = WS / "data" / "incorpus_rate_prediction" / "prediction_commitment.json"
PRED_STATIC_INCORPUS = WS / "data" / "incorpus_static_profile_prediction" / "prediction_commitment.json"
PRED_INCORPUS_SPREAD = WS / "data" / "incorpus_rate_spread_interpretation" / "incorpus_rate_spread_interpretation.json"
OUT = WS / "data" / "combined_decisive_readout"
FAMS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
EXE4 = ["BLiMP", "Supplement", "EWoK", "COMPS"]
EXE5_WITH_READING = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    scores: dict[str, float | None] = {}
    for fam in FAMS + ["Reading"]:
        rec = tasks.get(fam) if isinstance(tasks, dict) else None
        s = rec.get("score") if isinstance(rec, dict) else None
        scores[fam] = float(s) if finite(s) else None
    return scores


def avg(scores: dict[str, Any], fams: list[str]) -> float | None:
    vals = [scores.get(f) for f in fams]
    return statistics.mean(float(v) for v in vals) if all(finite(v) for v in vals) else None


def diff(a: Any, b: Any) -> float | None:
    return round(float(a) - float(b), 6) if finite(a) and finite(b) else None


def load_payloads(roots: list[pathlib.Path]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    files: list[dict[str, Any]] = []
    for root in roots:
        per = root / "per_target"
        if not per.exists():
            files.append({"root": rel(root), "per_target_exists": False, "n_files": 0})
            continue
        paths = sorted(per.glob("*.json"))
        files.append({"root": rel(root), "per_target_exists": True, "n_files": len(paths)})
        for p in paths:
            try:
                payload = read_json(p)
            except Exception as exc:
                files.append({"path": rel(p), "error": repr(exc)})
                continue
            arm = str(payload.get("arm") or "")
            ck = str(payload.get("checkpoint") or "")
            if not arm or not ck:
                continue
            key = (arm, ck, rel(p) or "")
            if key in seen:
                continue
            seen.add(key)
            rec = {
                "arm": arm,
                "checkpoint": ck,
                "source_root": rel(root),
                "source_payload": rel(p),
                "role": payload.get("role"),
                "scores": extract_scores(payload),
                "aggregates_from_payload": payload.get("aggregates"),
            }
            arms.setdefault(arm, {})[ck] = rec
    return arms, files


def load_clean() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not CLEAN.exists():
        return out
    for p in sorted(CLEAN.glob("*.json")):
        d = read_json(p)
        ck = str(d.get("endpoint") or d.get("checkpoint") or p.stem.split("_")[-1])
        out[ck] = {"arm": "clean_maxgeom", "checkpoint": ck, "source_payload": rel(p), "scores": extract_scores(d)}
    return out


def arm_rows(arms: dict[str, dict[str, dict[str, Any]]], clean: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, ckmap in sorted(arms.items()):
        for ck, rec in sorted(ckmap.items()):
            s = rec["scores"]
            cs = (clean.get(ck) or {}).get("scores", {})
            row: dict[str, Any] = {"arm": arm, "checkpoint": ck, "source_root": rec.get("source_root"), "source_payload": rec.get("source_payload")}
            for fam in FAMS:
                row[fam] = s.get(fam)
                row[f"delta_{fam}_vs_clean"] = diff(s.get(fam), cs.get(fam))
            row["exEntity4"] = avg(s, EXE4)
            row["cheap5_noReading"] = avg(s, FAMS)
            row["delta_exEntity4_vs_clean"] = diff(row["exEntity4"], avg(cs, EXE4))
            row["delta_cheap5_vs_clean"] = diff(row["cheap5_noReading"], avg(cs, FAMS))
            row["complete_families"] = sum(1 for f in FAMS if finite(s.get(f)))
            rows.append(row)
    return rows


def add_contrast(out: list[dict[str, Any]], arms: dict[str, dict[str, dict[str, Any]]], clean: dict[str, dict[str, Any]], name: str, a: str, b: str, ck: str) -> None:
    ra = arms.get(a, {}).get(ck)
    rb = clean.get(ck) if b == "clean_maxgeom" else arms.get(b, {}).get(ck)
    if not ra or not rb:
        return
    sa, sb = ra["scores"], rb["scores"]
    row: dict[str, Any] = {"contrast": name, "checkpoint": ck, "arm_A": a, "arm_B": b}
    for fam in FAMS:
        row[fam] = diff(sa.get(fam), sb.get(fam))
    row["exEntity4"] = diff(avg(sa, EXE4), avg(sb, EXE4))
    row["cheap5_noReading"] = diff(avg(sa, FAMS), avg(sb, FAMS))
    row["complete_families"] = sum(1 for f in FAMS if finite(sa.get(f)) and finite(sb.get(f)))
    out.append(row)


def contrast_rows(arms: dict[str, dict[str, dict[str, Any]]], clean: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ck in ["chck_80M", "chck_100M"]:
        add_contrast(out, arms, clean, "childspeech_minus_adultprose", "regmax_childspeech", "regmax_adultprose", ck)
        add_contrast(out, arms, clean, "regmax_childspeech_minus_clean", "regmax_childspeech", "clean_maxgeom", ck)
        add_contrast(out, arms, clean, "regmax_adultprose_minus_clean", "regmax_adultprose", "clean_maxgeom", ck)
    for ck in ["chck_70M", "chck_80M", "chck_100M"]:
        add_contrast(out, arms, clean, "incorpus_minus_clean", "incorpus_adultprose", "clean_maxgeom", ck)
    add_contrast(out, arms, clean, "incorpus_minus_subdose_full", "incorpus_adultprose", "subdose_full", "chck_70M")
    add_contrast(out, arms, clean, "subdose_full_minus_clean", "subdose_full", "clean_maxgeom", "chck_70M")
    return out


def read_optional_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        return {"read_error": repr(exc), "path": rel(path)}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):+.4f}"


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research combined decisive readout\n\n")
    lines.append("This file-only readout uses only the corrected research split scorer roots and excludes failed partial scorer outputs.\n\n")
    lines.append("## Roots\n\n")
    for r in payload["input_roots"]:
        lines.append(f"- {r}\n")
    lines.append("\n## Arms seen\n\n")
    lines.append(json.dumps(payload["arms_seen"], indent=2, ensure_ascii=False))
    lines.append("\n\n## Contrast table\n\n")
    lines.append("| contrast | checkpoint | BLiMP | Supplement | EWoK | COMPS | Entity | exEntity4 | cheap5 | complete families |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["contrast_rows"]:
        lines.append(f"| {r['contrast']} | {r['checkpoint']} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('Entity'))} | {fmt(r.get('exEntity4'))} | {fmt(r.get('cheap5_noReading'))} | {r.get('complete_families')} |\n")
    pred = payload.get("register_prediction_excerpt", {})
    rate = payload.get("incorpus_rate_commitment_excerpt", {})
    static = payload.get("incorpus_static_profile_commitment_excerpt", {})
    lines.append("\n## Frozen prediction anchors\n\n")
    lines.append(f"- Register childspeech-minus-adultprose profile exEntity5 prediction: {pred.get('profile_exEntity5')}; word-control exEntity5: {pred.get('word_control_exEntity5')}; strict task-text profile exEntity5: {pred.get('strict_profile_exEntity5')}.\n")
    reg_rate = payload.get("register_rate_commitment_excerpt", {})
    lines.append(f"- Register removal-rate commitment: {reg_rate.get('rate_based_sign_commitment')}\n")
    lines.append(f"- Register removed-block late rates: child/speech {reg_rate.get('childspeech_removed_block_rate_60_to_100')}; adult prose {reg_rate.get('adultprose_removed_block_rate_60_to_100')}; adult-minus-child {reg_rate.get('adult_minus_child_removed_block_rate')}\n")
    lines.append(f"- Register interpretation consequence: {reg_rate.get('interpretation')}\n")
    lines.append(f"- In-corpus rate commitment: {rate.get('primary_commitment')}\n")
    lines.append(f"- In-corpus clean-prior position: {rate.get('incorpus_clean_prior_position')}\n")
    lines.append(f"- In-corpus static profile commitment: {static.get('primary_profile_predictions_mass_scaled')}\n")
    lines.append(f"- In-corpus word-JS control commitment: {static.get('word_js_control_predictions_mass_scaled')}\n")
    inc_spread = payload.get("incorpus_rate_spread_interpretation_excerpt", {})
    lines.append(f"- In-corpus measured-band interpretation: status={inc_spread.get('future_rate_available')}, classification={inc_spread.get('classification')}, rate_60_to_100={inc_spread.get('rate_60_to_100')}.\n")
    lines.append(f"- In-corpus measured-band note: {inc_spread.get('interpretation')}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=[str(p) for p in DEFAULT_ROOTS])
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    roots = [pathlib.Path(x) if pathlib.Path(x).is_absolute() else ROOT / x for x in args.roots]
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arms, root_status = load_payloads(roots)
    clean = load_clean()
    rows = arm_rows(arms, clean)
    contrasts = contrast_rows(arms, clean)
    reg_pred = read_optional_json(PRED_REGISTER)
    reg_rate_pred = read_optional_json(PRED_REGISTER_RATE)
    rate_pred = read_optional_json(PRED_RATE)
    static_pred = read_optional_json(PRED_STATIC_INCORPUS)
    inc_spread_pred = read_optional_json(PRED_INCORPUS_SPREAD)

    files = {
        "summary_json": rel(out_dir / "combined_decisive_readout.json"),
        "summary_md": rel(out_dir / "combined_decisive_readout.md"),
        "arm_csv": rel(out_dir / "arm_scores_vs_clean.csv"),
        "contrast_csv": rel(out_dir / "decisive_contrasts.csv"),
    }
    payload = {
        "status": "COMBINED_DECISIVE_READOUT",
        "created_utc": now(),
        "input_roots": [rel(r) for r in roots],
        "root_status": root_status,
        "clean_score_dir": rel(CLEAN),
        "arms_seen": {a: sorted(ckmap) for a, ckmap in arms.items()},
        "n_arm_rows": len(rows),
        "n_contrast_rows": len(contrasts),
        "arm_rows": rows,
        "contrast_rows": contrasts,
        "register_prediction_excerpt": {
            "contrast_definition": reg_pred.get("contrast_definition"),
            "profile_exEntity5": (((reg_pred.get("committed_profile_js_model") or {}).get("predictions") or {}).get("exEntity5")),
            "strict_profile_exEntity5": (((reg_pred.get("strict_eval_text_extraction_robustness") or {}).get("aggregate_predictions") or {}).get("profile_js", {}) or {}).get("exEntity5"),
            "word_control_exEntity5": (((reg_pred.get("lexical_word_js_control") or {}).get("predictions") or {}).get("exEntity5")),
        },
        "register_rate_commitment_excerpt": {
            "rate_based_sign_commitment": reg_rate_pred.get("rate_based_sign_commitment"),
            "childspeech_removed_block_rate_60_to_100": reg_rate_pred.get("childspeech_removed_block_rate_60_to_100"),
            "adultprose_removed_block_rate_60_to_100": reg_rate_pred.get("adultprose_removed_block_rate_60_to_100"),
            "adult_minus_child_removed_block_rate": reg_rate_pred.get("adult_minus_child_removed_block_rate"),
            "interpretation": reg_rate_pred.get("interpretation"),
        },
        "incorpus_rate_commitment_excerpt": {
            "primary_commitment": rate_pred.get("primary_commitment"),
            "incorpus_clean_prior_position": rate_pred.get("incorpus_clean_prior_position"),
            "rate_classification_thresholds": rate_pred.get("rate_classification_thresholds"),
        },
        "incorpus_static_profile_commitment_excerpt": {
            "primary_profile_predictions_mass_scaled": static_pred.get("primary_profile_predictions_mass_scaled"),
            "word_js_control_predictions_mass_scaled": static_pred.get("word_js_control_predictions_mass_scaled"),
            "interpretation_rule": static_pred.get("interpretation_rule"),
        },
        "incorpus_rate_spread_interpretation_excerpt": {
            "future_rate_available": inc_spread_pred.get("future_rate_available"),
            "classification": ((inc_spread_pred.get("rate_classification") or {}).get("classification")),
            "interpretation": ((inc_spread_pred.get("rate_classification") or {}).get("interpretation")),
            "rate_60_to_100": ((inc_spread_pred.get("future_rate") or {}).get("rate_60_to_100")),
            "transition_bands": ((inc_spread_pred.get("rate_classification") or {}).get("transition_bands")),
            "rate_fit_spread": ((inc_spread_pred.get("rate_classification") or {}).get("rate_to_late_exEntity5_fit_spread")),
        },
        "files": files,
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_csv(out_dir / "arm_scores_vs_clean.csv", rows)
    write_csv(out_dir / "decisive_contrasts.csv", contrasts)
    write_json(out_dir / "combined_decisive_readout.json", payload)
    write_md(payload, out_dir / "combined_decisive_readout.md")
    print(json.dumps({
        "status": payload["status"],
        "arms_seen": payload["arms_seen"],
        "n_contrast_rows": len(contrasts),
        "summary_md": files["summary_md"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
