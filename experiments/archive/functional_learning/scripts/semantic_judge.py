#!/usr/bin/env python3
"""research: scalable semantic judge calibration for compact rewrites.

This does not admit production compacts by itself. It prepares and parses prompts for
an independent LLM semantic judge, calibrated against the independently reviewed pilot sample.
The scientific purpose is to determine whether a scalable reviewer can distinguish
faithful shortening from information loss and altered meaning before any learner
training uses the full generated candidate pool.

Generation is external. The prepare command writes prompt and manifest JSONL;
parse accepts existing response JSONL via --generation-output. Response rows use
index to join the manifest and output (or text/completion) for the response text.
Original generation defaults: llama3.1-8b-instruct, batch size 4,
max_new_tokens 180, temperature 0.0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

STUDY = Path("experiments/archive/functional_learning")
DATA = STUDY / "data/semantic_judge"
REVIEW_DATA = STUDY / "data/compact_semantic_review"
DEFAULT_SAMPLE = REVIEW_DATA / "semantic_review_sample.jsonl"
DEFAULT_LABELS = REVIEW_DATA / "independent_review_semantic_labels_pilot_sample.jsonl"
DEFAULT_FULL_JOINED = STUDY / "data/compact_pilot/full_generation_joined_26567.jsonl"

LABELS = ["faithful_shortening", "supported_summary_with_lost_detail", "altered_meaning", "unclear"]

SYSTEM_INSTRUCTION = """You are checking candidate compact rewrites for a BabyLM pretraining experiment. Compare the compact candidate primarily to the original source text. The current inherited rewrite is only a reference and may itself be imperfect.

Choose exactly one label:
- faithful_shortening: compact is shorter and preserves the source meaning needed for training: who did what, to whom, under what polarity, time/order, condition, modality, comparison, quotation/speech act, and numeric/entity attachment.
- supported_summary_with_lost_detail: compact says only source-supported things but deletes substantive source information, event structure, conditions, roles, order, or discourse acts. This is not an equivalent replacement.
- altered_meaning: compact introduces an unsupported claim, swaps roles/attachments, changes order, turns possibility/intention/condition into fact, changes polarity, or otherwise changes the source claim.
- unclear: source is too ambiguous, malformed, or context-dependent to judge safely.

Do not reward word savings, low lexical overlap, or number/entity string preservation by themselves. Complete numeral recall can still have wrong attachment; lower Jaccard can be omitted content. Output exactly one compact JSON object and no other text:
{"label":"...","confidence":0.0-1.0,"brief_reason":"...","critical_errors":["..."]}
"""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_prompt(rec: Dict[str, Any]) -> str:
    return (
        SYSTEM_INSTRUCTION
        + "\nPAIR_ID: " + str(rec.get("pair_id", ""))
        + "\nSOURCE: " + str(rec.get("source", ""))
        + "\n\nORIGINAL SOURCE TEXT:\n" + str(rec.get("original", "")).strip()
        + "\n\nCURRENT INHERITED REWRITE:\n" + str(rec.get("current_rewrite", "")).strip()
        + "\n\nCOMPACT CANDIDATE:\n" + str(rec.get("compact_rewrite", "")).strip()
        + "\n\nReturn JSON only."
    )


def prepare(args: argparse.Namespace) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    # If labels are provided, restrict/order to labeled pair IDs.
    label_map = {}
    if args.labels and Path(args.labels).exists():
        for r in load_jsonl(Path(args.labels)):
            label_map[r["pair_id"]] = r
    if label_map:
        records_by_id = {r["pair_id"]: r for r in records}
        ordered = []
        missing = []
        for pid in label_map:
            if pid in records_by_id:
                ordered.append(records_by_id[pid])
            else:
                missing.append(pid)
        records = ordered
    else:
        missing = []

    if args.max_records and args.max_records > 0:
        records = records[: int(args.max_records)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            prompt = make_prompt(r)
            f.write(json.dumps({"prompt": prompt, "pair_id": r.get("pair_id"), "index": i}, ensure_ascii=False) + "\n")
            m = {"index": i, "pair_id": r.get("pair_id"), "source": r.get("source", "")}
            if r.get("pair_id") in label_map:
                m["independent_review_label"] = label_map[r["pair_id"]].get("label")
                m["independent_review_note"] = label_map[r["pair_id"]].get("note", "")
            manifest.append(m)
    (out.parent / (out.stem + "_manifest.jsonl")).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
    summary = {"status": "SEMANTIC_JUDGE_PROMPTS_PREPARED", "n_prompts": len(records), "prompt_path": str(out), "manifest": str(out.parent / (out.stem + "_manifest.jsonl")), "missing_labeled": missing}
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Remove code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def normalize_label(label: str) -> str:
    lab = (label or "").strip().lower()
    lab = lab.replace(" ", "_").replace("-", "_")
    aliases = {
        "faithful": "faithful_shortening",
        "faithful_compression": "faithful_shortening",
        "supported_summary": "supported_summary_with_lost_detail",
        "information_losing_summary": "supported_summary_with_lost_detail",
        "information_losing": "supported_summary_with_lost_detail",
        "altered": "altered_meaning",
        "changed_meaning": "altered_meaning",
        "unsafe": "unclear",
        "unclear_or_unsafe_source": "unclear",
        "unclear_or_information_losing": "supported_summary_with_lost_detail",
    }
    return aliases.get(lab, lab if lab in LABELS else "parse_failed")


def parse_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = load_jsonl(Path(args.generation_output))
    manifest = load_jsonl(Path(args.manifest))
    man_by_idx = {int(m.get("index", i)): m for i, m in enumerate(manifest)}
    rows = []
    for j, r in enumerate(gen):
        idx = int(r.get("index", j))
        meta = man_by_idx.get(idx, {})
        raw = r.get("output", r.get("text", r.get("completion", "")))
        obj = extract_json(str(raw))
        label = normalize_label(str(obj.get("label", ""))) if obj else "parse_failed"
        rows.append({
            "index": idx,
            "pair_id": meta.get("pair_id", r.get("pair_id")),
            "source": meta.get("source", ""),
            "judge_label": label,
            "judge_confidence": obj.get("confidence") if obj else None,
            "judge_reason": obj.get("brief_reason") if obj else None,
            "judge_errors": obj.get("critical_errors") if obj else None,
            "raw_output": raw,
            "independent_review_label": meta.get("independent_review_label"),
            "independent_review_note": meta.get("independent_review_note"),
        })

    # Save parsed rows.
    parsed_path = out_dir / "semantic_judge_parsed.jsonl"
    with parsed_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labeled = [r for r in rows if r.get("independent_review_label")]
    conf = Counter()
    for r in labeled:
        true = normalize_label(str(r["independent_review_label"]))
        # independent review distinguishes unclear_or_unsafe_source and unclear_or_information_losing; for admission both are non-faithful.
        if true in {"unclear_or_unsafe_source", "unclear_or_information_losing"}:
            true = "unclear"
        pred = r["judge_label"]
        conf[(true, pred)] += 1
    faithful_true = [r for r in labeled if normalize_label(str(r["independent_review_label"])) == "faithful_shortening"]
    faithful_pred = [r for r in labeled if r["judge_label"] == "faithful_shortening"]
    faithful_good = [r for r in faithful_pred if normalize_label(str(r["independent_review_label"])) == "faithful_shortening"]
    false_admit = [r for r in faithful_pred if normalize_label(str(r["independent_review_label"])) != "faithful_shortening"]
    missed_faithful = [r for r in faithful_true if r["judge_label"] != "faithful_shortening"]

    summary = {
        "status": "SEMANTIC_JUDGE_PARSED",
        "n_outputs": len(rows),
        "n_parse_failed": sum(1 for r in rows if r["judge_label"] == "parse_failed"),
        "label_counts": dict(Counter(r["judge_label"] for r in rows)),
        "has_independent_review_labels": bool(labeled),
        "n_labeled": len(labeled),
        "confusion_true_pred": {f"{k[0]}->{k[1]}": v for k, v in conf.items()},
        "faithful_precision_on_labeled": (len(faithful_good) / len(faithful_pred)) if faithful_pred else None,
        "faithful_recall_on_labeled": (len(faithful_good) / len(faithful_true)) if faithful_true else None,
        "false_admit_ids": [{"pair_id": r["pair_id"], "independent_review_label": r["independent_review_label"], "judge_reason": r.get("judge_reason"), "independent_review_note": r.get("independent_review_note")} for r in false_admit],
        "missed_faithful_ids": [{"pair_id": r["pair_id"], "judge_label": r["judge_label"], "judge_reason": r.get("judge_reason"), "independent_review_note": r.get("independent_review_note")} for r in missed_faithful],
        "parsed_path": str(parsed_path),
    }
    (out_dir / "semantic_judge_calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("prepare")
    p.add_argument("--input", default=str(DEFAULT_SAMPLE))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--output", default=str(DATA / "calibration_prompts.jsonl"))
    p.add_argument("--max-records", type=int, default=0)

    q = sub.add_parser("parse")
    q.add_argument("--generation-output", default=str(DATA / "calibration_outputs.jsonl"))
    q.add_argument("--manifest", default=str(DATA / "calibration_prompts_manifest.jsonl"))
    q.add_argument("--output-dir", default=str(DATA))

    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "parse":
        parse_outputs(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
