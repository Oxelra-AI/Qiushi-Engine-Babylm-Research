#!/usr/bin/env python3
"""research relation_learning: read inherited frontier_consolidation evidence cells.

This script is file-only. It reads already-produced frontier_consolidation scoring and rate
artifacts, computes register seed-replication and in-corpus contrasts, and
writes relation_learning research-facing data/notes. It does not load models, train,
score official tasks, use GPU, upload, or touch leaderboards.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


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
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive" / 'frontier_consolidation'
FUNCTIONAL_RELATION_STUDIES = ROOT / "experiments/archive" / 'relation_learning'
OUT = FUNCTIONAL_RELATION_STUDIES / "data" / "frontier_consolidation_readout"
NOTE = FUNCTIONAL_RELATION_STUDIES / "notes" / "002_frontier_consolidation_seed43122_incorpus_readout.md"

FAMS4 = ["BLiMP", "Supplement", "EWoK", "COMPS"]
FAMS5 = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
CHECKPOINTS = ["chck_70M", "chck_80M", "chck_100M"]
REG_CKS = ["chck_80M", "chck_100M"]

REG_43022 = REPRESENTATION_FRONTIER_STUDIES / "data" / "register_readout_only" / "decisive_contrasts.csv"
REG_43122 = REPRESENTATION_FRONTIER_STUDIES / "data" / "register_seed43122_decisive_eval"
INC_ROOT = REPRESENTATION_FRONTIER_STUDIES / "data" / "incorpus_decisive_gpu_eval" / "per_target"
CLEAN_ROOT = REPRESENTATION_FRONTIER_STUDIES / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target"
INC_RATE = REPRESENTATION_FRONTIER_STUDIES / "data" / "incorpus_rate_ladder_after_training" / "prediction_commitment.json"
INC_STATIC = REPRESENTATION_FRONTIER_STUDIES / "data" / "incorpus_static_profile_prediction" / "prediction_commitment.json"
REG_STATIC_NOTE = REPRESENTATION_FRONTIER_STUDIES / "notes" / "distribution_proximity_prediction.md"
REG_RATE = REPRESENTATION_FRONTIER_STUDIES / "data" / "register_removal_rate_and_reference_spread" / "prediction_commitment.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return x is not None and x != "" and math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any) -> float | None:
    return float(x) if finite(x) else None


def round6(x: Any) -> float | None:
    return round(float(x), 6) if finite(x) else None


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def avg(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if finite(v)]
    return statistics.mean(xs) if len(xs) == len(vals) and xs else None


def scores_from_payload(path: pathlib.Path) -> dict[str, float | None]:
    d = read_json(path)
    tasks = d.get("tasks") or {}
    out: dict[str, float | None] = {}
    for fam in FAMS5 + ["Reading"]:
        rec = tasks.get(fam) if isinstance(tasks, dict) else None
        if isinstance(rec, dict):
            if fam == "Reading" and isinstance(rec.get("scores"), dict):
                out[fam] = fnum(rec.get("scores", {}).get("Reading", rec.get("score")))
            else:
                out[fam] = fnum(rec.get("score"))
        else:
            out[fam] = None
    return out


def add_aggregates(scores: dict[str, Any]) -> dict[str, Any]:
    s = dict(scores)
    s["exEntity4"] = avg([s.get(f) for f in FAMS4])
    s["cheap5_noReading"] = avg([s.get(f) for f in FAMS5])
    return s


def subtract(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    keys = FAMS5 + ["exEntity4", "cheap5_noReading", "Reading"]
    return {k: round6(float(a[k]) - float(b[k])) if finite(a.get(k)) and finite(b.get(k)) else None for k in keys}


def load_seed43022_register() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrast_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for r in read_csv(REG_43022):
        row = {"seed": "43022", "source": rel(REG_43022)}
        for k, v in r.items():
            row[k] = fnum(v) if k in FAMS5 + ["exEntity4", "cheap5_noReading"] else v
        if row.get("contrast") == "childspeech_minus_adultprose":
            contrast_rows.append(row)
        else:
            arm_rows.append(row)
    return arm_rows, contrast_rows


def load_seed43122_register() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    arm_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for arm_short, arm_dir, arm_name in [
        ("childspeech", "childspeech", "regmax_childspeech_seed43122"),
        ("adultprose", "adultprose", "regmax_adultprose_seed43122"),
    ]:
        for ck in REG_CKS:
            path = REG_43122 / arm_dir / "per_target" / f"{arm_name}_{ck}.json"
            if not path.exists():
                missing.append(rel(path) or str(path))
                continue
            scores = add_aggregates(scores_from_payload(path))
            row: dict[str, Any] = {
                "seed": "43122",
                "arm_short": arm_short,
                "arm": arm_name,
                "checkpoint": ck,
                "source_payload": rel(path),
                "complete_broad4": all(finite(scores.get(f)) for f in FAMS4),
                "complete_entity": finite(scores.get("Entity")),
            }
            for k in FAMS5 + ["exEntity4", "cheap5_noReading", "Reading"]:
                row[k] = scores.get(k)
            arm_rows.append(row)
            by[(arm_short, ck)] = scores
    contrast_rows: list[dict[str, Any]] = []
    for ck in REG_CKS:
        child = by.get(("childspeech", ck))
        adult = by.get(("adultprose", ck))
        row: dict[str, Any] = {
            "seed": "43122",
            "checkpoint": ck,
            "contrast": "childspeech_minus_adultprose",
            "source": rel(REG_43122),
            "complete_broad4": bool(child and adult and all(finite(child.get(f)) and finite(adult.get(f)) for f in FAMS4)),
            "complete_entity": bool(child and adult and finite(child.get("Entity")) and finite(adult.get("Entity"))),
        }
        diffs = subtract(child or {}, adult or {})
        for k, v in diffs.items():
            row[k] = v
        contrast_rows.append(row)
    return arm_rows, contrast_rows, missing


def clean_payload_path(ck: str) -> pathlib.Path:
    return CLEAN_ROOT / f"deberta_maxgeom_clean_seed43022_{ck}.json"


def inc_payload_path(ck: str) -> pathlib.Path:
    return INC_ROOT / f"incorpus_adultprose_{ck}.json"


def subdose_payload_path(ck: str = "chck_70M") -> pathlib.Path:
    return INC_ROOT / f"subdose_full_{ck}.json"


def load_incorpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    arm_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    scoremaps: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, label, cks, path_fun in [
        ("clean_maxgeom", "clean", CHECKPOINTS, clean_payload_path),
        ("incorpus_adultprose", "incorpus", CHECKPOINTS, inc_payload_path),
        ("subdose_full", "subdose_full", ["chck_70M"], subdose_payload_path),
    ]:
        for ck in cks:
            path = path_fun(ck)
            if not path.exists():
                missing.append(rel(path) or str(path))
                continue
            scores = add_aggregates(scores_from_payload(path))
            row: dict[str, Any] = {
                "arm": arm,
                "label": label,
                "checkpoint": ck,
                "source_payload": rel(path),
                "complete_families5": all(finite(scores.get(f)) for f in FAMS5),
            }
            for k in FAMS5 + ["exEntity4", "cheap5_noReading", "Reading"]:
                row[k] = scores.get(k)
            arm_rows.append(row)
            scoremaps[(arm, ck)] = scores
    contrasts: list[dict[str, Any]] = []
    pairs = []
    for ck in CHECKPOINTS:
        pairs.append(("incorpus_minus_clean", "incorpus_adultprose", "clean_maxgeom", ck))
    pairs += [
        ("incorpus_minus_subdose_full", "incorpus_adultprose", "subdose_full", "chck_70M"),
        ("subdose_full_minus_clean", "subdose_full", "clean_maxgeom", "chck_70M"),
    ]
    for name, a, b, ck in pairs:
        sa = scoremaps.get((a, ck))
        sb = scoremaps.get((b, ck))
        row: dict[str, Any] = {"contrast": name, "checkpoint": ck, "arm_A": a, "arm_B": b, "complete_families5": False}
        if sa and sb:
            row["complete_families5"] = all(finite(sa.get(f)) and finite(sb.get(f)) for f in FAMS5)
            for k, v in subtract(sa, sb).items():
                row[k] = v
        contrasts.append(row)
    return arm_rows, contrasts, missing


def mean_rows(rows: list[dict[str, Any]], where: dict[str, str], quantity: str) -> float | None:
    vals = []
    for r in rows:
        if all(str(r.get(k)) == v for k, v in where.items()) and finite(r.get(quantity)):
            vals.append(float(r[quantity]))
    return statistics.mean(vals) if vals else None


def sign(x: Any) -> str:
    if not finite(x):
        return "missing"
    x = float(x)
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def distance(a: Any, b: Any) -> float | None:
    return abs(float(a) - float(b)) if finite(a) and finite(b) else None


def build_predictor_rows(register_contrasts: list[dict[str, Any]], inc_contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inc_rate = read_json(INC_RATE) if INC_RATE.exists() else {}
    inc_static = read_json(INC_STATIC) if INC_STATIC.exists() else {}
    reg_rate = read_json(REG_RATE) if REG_RATE.exists() else {}

    reg_mean_43022 = mean_rows(register_contrasts, {"seed": "43022", "contrast": "childspeech_minus_adultprose"}, "exEntity4")
    reg_mean_43122 = mean_rows(register_contrasts, {"seed": "43122", "contrast": "childspeech_minus_adultprose"}, "exEntity4")
    inc_endpoint = mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_100M"}, "exEntity4")
    inc_late = statistics.mean([x for x in [
        mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_80M"}, "exEntity4"),
        mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_100M"}, "exEntity4"),
    ] if x is not None])
    inc_full_70 = mean_rows(inc_contrasts, {"contrast": "incorpus_minus_subdose_full", "checkpoint": "chck_70M"}, "exEntity4")
    static_preds = inc_static.get("primary_profile_predictions_mass_scaled") or {}
    word_preds = inc_static.get("word_js_control_predictions_mass_scaled") or {}

    rows = [
        {
            "test_lens": "register_static_profile_or_removed_block_late_rate",
            "quantity": "seed43022_exEntity4_mean_chck80_100_child_minus_adult",
            "prediction_before_scores": "positive around +0.75 to +0.77 from profile; positive from removed-block late-rate difference",
            "observed": round6(reg_mean_43022),
            "observed_sign": sign(reg_mean_43022),
            "reading": "opposite sign, so seed43022 rejects the simple positive register predictor",
            "source": rel(REG_43022),
        },
        {
            "test_lens": "register_static_profile_or_removed_block_late_rate",
            "quantity": "seed43122_exEntity4_mean_chck80_100_child_minus_adult",
            "prediction_before_scores": "same positive sign if the predictor is seed-stable",
            "observed": round6(reg_mean_43122),
            "observed_sign": sign(reg_mean_43122),
            "reading": "positive broad4 sign, but the sign reverses relative to seed43022 and BLiMP remains negative; register opportunity value is basin-conditional, not a stable scalar",
            "source": rel(REG_43122),
        },
        {
            "test_lens": "in_corpus_static_profile",
            "quantity": "incorpus_minus_clean_exEntity4_chck100",
            "prediction_before_scores": static_preds.get("incorpus_minus_clean_exEntity4_noReading"),
            "observed": round6(inc_endpoint),
            "abs_error": round6(distance(inc_endpoint, static_preds.get("incorpus_minus_clean_exEntity4_noReading"))),
            "observed_sign": sign(inc_endpoint),
            "reading": "positive endpoint movement is compatible with a profile/proximity term",
            "source": rel(INC_STATIC),
        },
        {
            "test_lens": "in_corpus_late_residual_rate",
            "quantity": "incorpus_minus_clean_exEntity4_mean_chck80_100",
            "prediction_before_scores": inc_rate.get("predicted_late_exEntity5_from_rate_fit_if_available"),
            "observed": round6(inc_late),
            "abs_error": round6(distance(inc_late, inc_rate.get("predicted_late_exEntity5_from_rate_fit_if_available"))),
            "observed_sign": sign(inc_late),
            "reading": "adult-prose rows remained late-active and the official broad4 movement is positive; this supports residual work as a real measured term for DeBERTa, though the comparison lacks Reading and uses a three-arm rate fit",
            "source": rel(INC_RATE),
        },
        {
            "test_lens": "in_corpus_vs_out_of_corpus_full1x",
            "quantity": "incorpus_minus_subdose_full_exEntity4_chck70",
            "prediction_before_scores": static_preds.get("incorpus_minus_full1x_exEntity4_noReading"),
            "observed": round6(inc_full_70),
            "abs_error": round6(distance(inc_full_70, static_preds.get("incorpus_minus_full1x_exEntity4_noReading"))),
            "observed_sign": sign(inc_full_70),
            "reading": "in-corpus is positive relative to clean but below full1x on broad4 at 70M; official adult prose is not dead experience, yet FineWeb/source-view experience may still carry additional structure",
            "source": rel(INC_ROOT),
        },
        {
            "test_lens": "word_unigram_control",
            "quantity": "incorpus_minus_clean_exEntity4_chck100",
            "prediction_before_scores": word_preds.get("incorpus_minus_clean_exEntity4_noReading"),
            "observed": round6(inc_endpoint),
            "observed_sign": sign(inc_endpoint),
            "reading": "word-unigram proximity predicts the wrong sign for the in-corpus cell and cannot explain the positive DeBERTa conversion",
            "source": rel(INC_STATIC),
        },
    ]
    # Add explicit register rate numbers so downstream readers do not have to reopen the old JSON.
    for r in rows[:2]:
        r["register_child_removed_late_rate_mean"] = (reg_rate.get("childspeech_removed_block_rate_60_to_100") or {}).get("mean")
        r["register_adult_removed_late_rate_mean"] = (reg_rate.get("adultprose_removed_block_rate_60_to_100") or {}).get("mean")
    return rows


def fmt(x: Any, plus: bool = True) -> str:
    if not finite(x):
        return "NA"
    return ("{:+.3f}" if plus else "{:.3f}").format(float(x))


def write_note(payload: dict[str, Any]) -> None:
    reg = payload["register_summary"]
    inc = payload["incorpus_summary"]
    files = payload["files"]
    lines: list[str] = []
    lines.append("# research readout of inherited frontier_consolidation register and in-corpus cells\n\n")
    lines.append("This note is research-facing memory for relation_learning. It reads already-produced frontier_consolidation artifacts and does not run model inference, training, official scoring, uploads, or leaderboard actions.\n\n")
    lines.append("## Predictors tied to these cells before any new training\n\n")
    lines.append("The old frontier_consolidation files already contain the numerical predictors used here: research profile-register prediction, research removed-block late-rate prediction, research in-corpus static profile prediction, and research post-training in-corpus late-rate readout. relation_learning uses those numbers as comparators and asks whether they supply a computable conversion term rather than simply describe outcomes.\n\n")
    lines.append("- Register childspeech-minus-adultprose: profile distance and removed-block late rate both predict a positive broad contrast if adult-prose removal is costlier; word-unigram predicts a small negative contrast. A seed-stable principle would preserve the sign across seed43022 and seed43122.\n")
    lines.append("- In-corpus adult prose: static profile predicts a small positive broad4/cheap5 movement; the measured in-corpus 60M→100M admitted-block loss drop predicts a positive late broad gain close to the view/breadth regime; the word-unigram control predicts a small negative movement; a pure out-of-corpus-novelty reading expects the in-corpus cell to stay near clean while FineWeb full1x remains higher.\n")
    lines.append("- The open term in this study is conversion: whether residual work is connected to reusable learner coordinates rather than privately fitted. The present file-only readout cannot measure that term; it identifies where a checkpoint-computable shared/private fitting index must enter next.\n\n")

    lines.append("## Register seed replication\n\n")
    lines.append("Positive means the arm that removes child/spoken material scores above the arm that removes adult prose, with identical admitted MAX FineWeb text.\n\n")
    lines.append("| seed | checkpoint | BLiMP | Supplement | EWoK | COMPS | exEntity4 | Entity/cheap if present |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["register_contrast_rows"]:
        if r.get("contrast") != "childspeech_minus_adultprose":
            continue
        lines.append(f"| {r.get('seed')} | {r.get('checkpoint')} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('exEntity4'))} | {fmt(r.get('cheap5_noReading'))} |\n")
    lines.append("\n")
    lines.append(f"Seed43022 mean exEntity4 over 80/100M is {fmt(reg['seed43022_exEntity4_mean'])}; seed43122 mean exEntity4 is {fmt(reg['seed43122_exEntity4_mean'])}. The broad sign therefore flips between basins. Seed43122 gives positive broad4 mostly through EWoK/COMPS/Supplement while BLiMP remains negative; seed43022 gave a negative broad4 contrast. This makes the sacrificed-register sign real but not stable enough to be the general principle.\n\n")

    lines.append("## In-corpus adult-prose cell\n\n")
    lines.append("| contrast | checkpoint | BLiMP | Supplement | EWoK | COMPS | Entity | exEntity4 | cheap5 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["incorpus_contrast_rows"]:
        lines.append(f"| {r.get('contrast')} | {r.get('checkpoint')} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('Entity'))} | {fmt(r.get('exEntity4'))} | {fmt(r.get('cheap5_noReading'))} |\n")
    lines.append("\n")
    lines.append(f"In-corpus minus clean is positive on broad4 at 70M/80M/100M ({fmt(inc['incorpus_minus_clean_exEntity4_70M'])}, {fmt(inc['incorpus_minus_clean_exEntity4_80M'])}, {fmt(inc['incorpus_minus_clean_exEntity4_100M'])}), with late 80/100M mean {fmt(inc['incorpus_minus_clean_exEntity4_late_mean'])}. The post-training admitted-block loss drop from 60M to 100M is {fmt(inc['incorpus_owner_loss_reduction_60_to_100'], plus=False)}, near the view/breadth active range rather than the repeat-like low range. At 70M, in-corpus is below full1x on broad4 ({fmt(inc['incorpus_minus_subdose_full_exEntity4_70M'])}) but above it on Entity ({fmt(inc['incorpus_minus_subdose_full_Entity_70M'])}), so Entity should remain separate from the broad learning principle.\n\n")

    lines.append("## What this changes for the principle\n\n")
    lines.append("The in-corpus result gives residual work numerical content in DeBERTa: official adult-prose rows still carried late reducible loss and their training moved broad evaluation scores. The register replicate prevents turning removal-side register opportunity into a seed-invariant scalar: the same sign does not hold across seed43022 and seed43122. The stronger relation_learning proposition should therefore be coordinate conversion, not register mixture by itself: finite experience helps when late residual work is aligned with a learner coordinate that carries improvement beyond the trained rows. The next execution must measure that alignment directly as shared versus private fitting on existing checkpoints, especially DeBERTa versus RoBERTa on identical MAX admitted view text.\n\n")

    lines.append("## Files\n\n")
    for k, v in files.items():
        lines.append(f"- {k}: `{v}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg43022_arm, reg43022_contrasts = load_seed43022_register()
    reg43122_arm, reg43122_contrasts, reg_missing = load_seed43122_register()
    inc_arm_rows, inc_contrasts, inc_missing = load_incorpus()
    register_contrasts = reg43022_contrasts + reg43122_contrasts
    register_arm_rows = reg43022_arm + reg43122_arm
    predictor_rows = build_predictor_rows(register_contrasts, inc_contrasts)

    inc_rate = read_json(INC_RATE) if INC_RATE.exists() else {}
    rate_metrics = inc_rate.get("incorpus_owner_rate_metrics") or {}

    register_summary = {
        "seed43022_exEntity4_mean": mean_rows(register_contrasts, {"seed": "43022", "contrast": "childspeech_minus_adultprose"}, "exEntity4"),
        "seed43122_exEntity4_mean": mean_rows(register_contrasts, {"seed": "43122", "contrast": "childspeech_minus_adultprose"}, "exEntity4"),
        "seed43022_cheap5_mean": mean_rows(register_contrasts, {"seed": "43022", "contrast": "childspeech_minus_adultprose"}, "cheap5_noReading"),
        "seed43122_cheap5_mean": mean_rows(register_contrasts, {"seed": "43122", "contrast": "childspeech_minus_adultprose"}, "cheap5_noReading"),
        "seed43122_entity_available": any(finite(r.get("Entity")) for r in reg43122_contrasts),
        "scientific_reading": "Seed43122 reverses broad4 sign relative to seed43022; register opportunity cost is basin-conditional and cannot by itself be the general data-efficient learning principle.",
    }
    inc_summary = {
        "incorpus_minus_clean_exEntity4_70M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_70M"}, "exEntity4"),
        "incorpus_minus_clean_exEntity4_80M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_80M"}, "exEntity4"),
        "incorpus_minus_clean_exEntity4_100M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_100M"}, "exEntity4"),
        "incorpus_minus_clean_cheap5_100M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_100M"}, "cheap5_noReading"),
        "incorpus_minus_clean_exEntity4_late_mean": statistics.mean([x for x in [
            mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_80M"}, "exEntity4"),
            mean_rows(inc_contrasts, {"contrast": "incorpus_minus_clean", "checkpoint": "chck_100M"}, "exEntity4"),
        ] if x is not None]),
        "incorpus_minus_subdose_full_exEntity4_70M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_subdose_full", "checkpoint": "chck_70M"}, "exEntity4"),
        "incorpus_minus_subdose_full_Entity_70M": mean_rows(inc_contrasts, {"contrast": "incorpus_minus_subdose_full", "checkpoint": "chck_70M"}, "Entity"),
        "incorpus_owner_loss_reduction_60_to_100": rate_metrics.get("loss_reduction_60_to_100"),
        "predicted_late_exEntity5_from_rate_fit": inc_rate.get("predicted_late_exEntity5_from_rate_fit_if_available"),
        "scientific_reading": "In-corpus adult prose remains late-active and produces positive DeBERTa broad4 movement, but does not remove the need for a learner-coordinate conversion term.",
    }
    files = {
        "summary_json": rel(OUT / "frontier_consolidation_readout_summary.json"),
        "register_arm_csv": rel(OUT / "register_arm_scores.csv"),
        "register_contrast_csv": rel(OUT / "register_seed_contrasts.csv"),
        "incorpus_arm_csv": rel(OUT / "incorpus_arm_scores.csv"),
        "incorpus_contrast_csv": rel(OUT / "incorpus_contrasts.csv"),
        "predictor_csv": rel(OUT / "predictor_outcomes.csv"),
        "note_md": rel(NOTE),
    }
    payload = {
        "status": "relation_learning_STEP002_READOUT_DONE",
        "created_utc": now(),
        "source_session": "frontier_consolidation",
        "register_sources": {
            "seed43022_contrasts": rel(REG_43022),
            "seed43122_root": rel(REG_43122),
            "static_profile_note": rel(REG_STATIC_NOTE),
            "register_rate_json": rel(REG_RATE),
        },
        "incorpus_sources": {
            "per_target_root": rel(INC_ROOT),
            "clean_root": rel(CLEAN_ROOT),
            "post_training_rate_json": rel(INC_RATE),
            "static_profile_json": rel(INC_STATIC),
        },
        "missing_files": sorted(set(reg_missing + inc_missing)),
        "register_summary": register_summary,
        "incorpus_summary": inc_summary,
        "register_arm_rows": register_arm_rows,
        "register_contrast_rows": register_contrasts,
        "incorpus_arm_rows": inc_arm_rows,
        "incorpus_contrast_rows": inc_contrasts,
        "predictor_rows": predictor_rows,
        "principle_update": {
            "kept": "Late residual work has measurable content in the in-corpus DeBERTa cell, and static word-unigram proximity is not sufficient.",
            "changed": "Register child-versus-adult removal sign is not stable across the two available seeds, so the principle should not be a single register-mixture or removed-register value scalar.",
            "next_measured_term": "Compute shared-versus-private fitting on existing checkpoints: identical admitted view text, DeBERTa converting arm versus RoBERTa non-converting arm, with clean-prior and cross-architecture alignment measured before using benchmark scores.",
        },
        "files": files,
        "no_model_loading_training_official_scoring_gpu_upload_or_leaderboard": True,
    }
    write_csv(OUT / "register_arm_scores.csv", register_arm_rows)
    write_csv(OUT / "register_seed_contrasts.csv", register_contrasts)
    write_csv(OUT / "incorpus_arm_scores.csv", inc_arm_rows)
    write_csv(OUT / "incorpus_contrasts.csv", inc_contrasts)
    write_csv(OUT / "predictor_outcomes.csv", predictor_rows)
    write_json(OUT / "frontier_consolidation_readout_summary.json", payload)
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "register_seed43022_exEntity4_mean": round6(register_summary["seed43022_exEntity4_mean"]),
        "register_seed43122_exEntity4_mean": round6(register_summary["seed43122_exEntity4_mean"]),
        "incorpus_late_exEntity4_mean": round6(inc_summary["incorpus_minus_clean_exEntity4_late_mean"]),
        "incorpus_owner_loss_reduction_60_to_100": inc_summary["incorpus_owner_loss_reduction_60_to_100"],
        "note": files["note_md"],
        "summary": files["summary_json"],
        "missing_count": len(payload["missing_files"]),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
