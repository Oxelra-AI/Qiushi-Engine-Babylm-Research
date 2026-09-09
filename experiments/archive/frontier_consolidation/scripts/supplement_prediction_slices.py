#!/usr/bin/env python3
"""research: join official Supplement predictions to tokenizer-affected row slices.

CPU-only.  This script is for post-evaluation interpretation of compliant
endpoints.  It maps a Supplement predictions.json file produced by the official
sentence_zero_shot runner back to official Supplement rows and to the research
newline/<unk> structure.  It does not run model inference and does not change
scores; it only makes row subsets inspectable after official predictions exist.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
STRICT_REPO = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
SUPP_DIR = STRICT_REPO / "evaluation_data/full_eval/supplement_filtered"
READ_FILES_PATH = STRICT_REPO / "evaluation_pipeline/sentence_zero_shot/read_files.py"
DEFAULT_SYMMETRY = STUDY / "data/supplement_newline_symmetry/supplement_newline_symmetry.json"
DEFAULT_OUT_DIR = STUDY / "data/supplement_prediction_slices"

spec = importlib.util.spec_from_file_location("strict_sentence_read_files_step043_preds", READ_FILES_PATH)
read_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(read_mod)  # type: ignore[arg-type]


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if raw:
                yield line_no, raw, json.loads(raw)


def build_manifest(symmetry_json: pathlib.Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sym = json.loads(symmetry_json.read_text(encoding="utf-8"))
    by_task_line = {(r["task"], int(r["line"])): r for r in sym.get("rows", [])}
    manifest: dict[str, dict[str, Any]] = {}
    task_counts = Counter()
    for path in sorted(SUPP_DIR.glob("*.jsonl")):
        task = path.stem
        for line_no, raw_line, raw in read_jsonl(path):
            idx = task_counts[task]
            task_counts[task] += 1
            pred_id = f"{task}_{idx}"
            decoded_rows = read_mod.decode(raw_line, path, "blimp", False, None)
            if len(decoded_rows) != 1:
                raise RuntimeError(f"unexpected decoded row count {len(decoded_rows)} for {path}:{line_no}")
            dec = decoded_rows[0]
            sym_rec = by_task_line.get((task, line_no), {})
            good = dec["sentences"][0]
            bad = dec["sentences"][1]
            manifest[pred_id] = {
                "id": pred_id,
                "task": task,
                "line": line_no,
                "path": str(path),
                "sentence_good": good,
                "sentence_bad": bad,
                "raw_row": raw,
                "affected": bool(sym_rec.get("affected")),
                "both_candidates_affected": bool(sym_rec.get("both_candidates_affected")),
                "same_unk_count_good_bad": bool(sym_rec.get("same_unk_count_good_bad")),
                "same_unk_span_multiset": bool(sym_rec.get("same_unk_span_multiset")),
                "same_unk_span_and_offset": bool(sym_rec.get("same_unk_span_and_offset")),
                "local8_same_at_unk": bool(sym_rec.get("local8_same_at_unk")),
                "local24_same_at_unk": bool(sym_rec.get("local24_same_at_unk")),
                "target_unk_tokens_pair": int(sym_rec.get("target_unk_tokens_pair", 0) or 0),
                "target_unk_fraction_step35_pair": float(sym_rec.get("target_unk_fraction_step35_pair", 0.0) or 0.0),
            }
    meta = {
        "supplement_dir": str(SUPP_DIR),
        "symmetry_json": str(symmetry_json),
        "rows": len(manifest),
        "task_counts": dict(task_counts),
    }
    return manifest, meta


def load_predictions(path: pathlib.Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preds: dict[str, str] = {}
    for uid, rec in payload.items():
        entries = rec.get("predictions") if isinstance(rec, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError(f"predictions for {uid} not a list in {path}")
        for ent in entries:
            if not isinstance(ent, dict) or "id" not in ent or "pred" not in ent:
                raise RuntimeError(f"malformed prediction entry under {uid}: {ent}")
            pid = str(ent["id"])
            if pid in preds:
                raise RuntimeError(f"duplicate prediction id {pid} in {path}")
            preds[pid] = str(ent["pred"])
    return preds


def bucket_key(row: dict[str, Any]) -> str:
    if not row["affected"]:
        return "unaffected"
    if row["same_unk_span_and_offset"]:
        return "affected_same_span_offset"
    if row["local8_same_at_unk"]:
        return "affected_offsetdiff_local8same"
    return "affected_offsetdiff_local8diff"


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def score_predictions(manifest: dict[str, dict[str, Any]], preds: dict[str, str]) -> dict[str, Any]:
    missing = sorted(set(manifest) - set(preds))
    extra = sorted(set(preds) - set(manifest))
    rows = []
    counters = {
        "overall": Counter(),
        "by_task": defaultdict(Counter),
        "by_affected": defaultdict(Counter),
        "by_structural_slice": defaultdict(Counter),
        "by_task_and_affected": defaultdict(Counter),
    }
    examples_wrong_affected = []
    examples_right_affected = []
    for pid, row in manifest.items():
        pred = preds.get(pid)
        if pred is None:
            continue
        correct = pred == row["sentence_good"]
        wrong_chose_bad = pred == row["sentence_bad"]
        other_prediction = (not correct) and (not wrong_chose_bad)
        affected_key = "affected" if row["affected"] else "unaffected"
        structural = bucket_key(row)
        task_aff_key = f"{row['task']}::{affected_key}"
        joined = {
            **{k: row[k] for k in [
                "id", "task", "line", "affected", "same_unk_span_and_offset",
                "local8_same_at_unk", "local24_same_at_unk", "target_unk_tokens_pair",
                "target_unk_fraction_step35_pair",
            ]},
            "pred": pred,
            "correct": correct,
            "wrong_chose_bad": wrong_chose_bad,
            "other_prediction": other_prediction,
            "structural_slice": structural,
        }
        rows.append(joined)
        for c in [
            counters["overall"],
            counters["by_task"][row["task"]],
            counters["by_affected"][affected_key],
            counters["by_structural_slice"][structural],
            counters["by_task_and_affected"][task_aff_key],
        ]:
            c["n"] += 1
            c["correct"] += int(correct)
            c["wrong_chose_bad"] += int(wrong_chose_bad)
            c["other_prediction"] += int(other_prediction)
        if row["affected"] and not correct and len(examples_wrong_affected) < 24:
            examples_wrong_affected.append({
                "id": pid,
                "task": row["task"],
                "line": row["line"],
                "pred": pred,
                "sentence_good": row["sentence_good"],
                "sentence_bad": row["sentence_bad"],
                "structural_slice": structural,
            })
        if row["affected"] and correct and len(examples_right_affected) < 12:
            examples_right_affected.append({
                "id": pid,
                "task": row["task"],
                "line": row["line"],
                "pred": pred,
                "structural_slice": structural,
            })

    def enrich_counter(c: Counter) -> dict[str, Any]:
        d = dict(c)
        d["accuracy"] = safe_div(c.get("correct", 0), c.get("n", 0))
        d["error_rate"] = safe_div(c.get("n", 0) - c.get("correct", 0), c.get("n", 0))
        return d

    by_task = {k: enrich_counter(v) for k, v in sorted(counters["by_task"].items())}
    official_macro = sum((v["accuracy"] or 0.0) for v in by_task.values()) / len(by_task) if by_task else None
    result = {
        "prediction_rows_joined": len(rows),
        "prediction_rows_expected": len(manifest),
        "missing_prediction_ids": missing[:100],
        "missing_prediction_count": len(missing),
        "extra_prediction_ids": extra[:100],
        "extra_prediction_count": len(extra),
        "overall_micro": enrich_counter(counters["overall"]),
        "official_supplement_subtask_macro_accuracy_0to1": official_macro,
        "official_supplement_subtask_macro_accuracy_0to100": official_macro * 100.0 if official_macro is not None else None,
        "by_task": by_task,
        "by_affected": {k: enrich_counter(v) for k, v in sorted(counters["by_affected"].items())},
        "by_structural_slice": {k: enrich_counter(v) for k, v in sorted(counters["by_structural_slice"].items())},
        "by_task_and_affected": {k: enrich_counter(v) for k, v in sorted(counters["by_task_and_affected"].items())},
        "examples_wrong_affected": examples_wrong_affected,
        "examples_right_affected": examples_right_affected,
        "rows": rows,
    }
    return result


def write_md(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    s = payload["score"]
    lines = []
    lines.append(f"# research Supplement prediction slices — {payload['tag']}")
    lines.append("")
    lines.append("This CPU-only join maps official Supplement predictions to the research-tokenizer newline-affected row structure. It is an interpretation layer; official column scores still come from the standard evaluator and pristine collation.")
    lines.append("")
    lines.append(f"Predictions: `{payload['predictions_path']}`")
    lines.append(f"Rows joined: {s['prediction_rows_joined']} / {s['prediction_rows_expected']} (missing {s['missing_prediction_count']}, extra {s['extra_prediction_count']}).")
    lines.append(f"Subtask-macro Supplement accuracy: {s['official_supplement_subtask_macro_accuracy_0to100']:.6f}")
    lines.append("")
    lines.append("## By subtask")
    lines.append("")
    lines.append("| subtask | n | correct | accuracy | affected n | affected accuracy | unaffected accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for task, d in s["by_task"].items():
        aff = s["by_task_and_affected"].get(f"{task}::affected", {})
        una = s["by_task_and_affected"].get(f"{task}::unaffected", {})
        aff_acc = aff.get("accuracy")
        una_acc = una.get("accuracy")
        aff_acc_s = f"{100.0 * aff_acc:.3f}" if aff_acc is not None else ""
        una_acc_s = f"{100.0 * una_acc:.3f}" if una_acc is not None else ""
        lines.append(f"| {task} | {d.get('n', 0)} | {d.get('correct', 0)} | {100.0 * (d.get('accuracy') or 0.0):.3f} | {aff.get('n', 0)} | {aff_acc_s} | {una_acc_s} |")
    lines.append("")
    lines.append("## Affected versus unaffected")
    lines.append("")
    lines.append("| slice | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for k, d in s["by_affected"].items():
        lines.append(f"| {k} | {d.get('n', 0)} | {d.get('correct', 0)} | {100.0 * (d.get('accuracy') or 0.0):.3f} |")
    lines.append("")
    lines.append("## Structural slices among research-affected rows")
    lines.append("")
    lines.append("| structural slice | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for k, d in s["by_structural_slice"].items():
        lines.append(f"| {k} | {d.get('n', 0)} | {d.get('correct', 0)} | {100.0 * (d.get('accuracy') or 0.0):.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use this after each compliant endpoint finishes official Supplement scoring. If research and byte-alphabet endpoints differ mostly on `affected` rows, especially QA or turn-taking, the difference is consistent with the tokenizer-surface mechanism quantified in research/43. If they differ mainly on unaffected rows or other columns, the movement reflects broader training/tokenization interactions rather than newline `<unk>` exposure alone.")
    lines.append("")
    lines.append(f"Full JSON: `{payload['out_json']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--symmetry-json", default=str(DEFAULT_SYMMETRY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    pred_path = pathlib.Path(args.predictions)
    symmetry_json = pathlib.Path(args.symmetry_json)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_meta = build_manifest(symmetry_json)
    preds = load_predictions(pred_path)
    score = score_predictions(manifest, preds)
    out_json = out_dir / f"{args.tag}_supplement_prediction_slices.json"
    out_md = out_dir / f"{args.tag}_supplement_prediction_slices.md"
    payload = {
        "status": "SUPPLEMENT_PREDICTION_SLICES",
        "tag": args.tag,
        "predictions_path": str(pred_path),
        "symmetry_json": str(symmetry_json),
        "manifest_meta": manifest_meta,
        "score": score,
        "elapsed_sec": round(time.time() - t0, 3),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "tag": args.tag,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "joined": score["prediction_rows_joined"],
        "expected": score["prediction_rows_expected"],
        "supplement_macro": score["official_supplement_subtask_macro_accuracy_0to100"],
        "affected": score["by_affected"].get("affected"),
        "unaffected": score["by_affected"].get("unaffected"),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
