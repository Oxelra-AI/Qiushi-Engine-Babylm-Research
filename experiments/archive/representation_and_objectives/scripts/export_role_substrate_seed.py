#!/usr/bin/env python3
"""research: export stable role-substrate seeds and repair queue.

Consumes research automatic labels and research review, with tolerant label parsing.
Outputs compact JSONL files for the next construction/execution step:
- strict_retained_cases.jsonl: all four facts, all contexts, both teachers stable
- full_context_stable_facts.jsonl: individual facts stable across source/bridge/natural
- source_bridge_stable_facts.jsonl: individual facts stable on source and source-attested bridge
- repair_queue.jsonl: nonstable facts with interpretable reasons
"""
from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
AUTO = ROOT / "data/auto_role_fact_substrate"
OUT = ROOT / "data/role_substrate_review"
TEACHERS = ["qwen3.5-9b", "llama3.1-8b-instruct"]
CONTEXTS = ["source", "source_attested_bridge", "natural_compact_reference"]
SOURCE_BRIDGE = ["source", "source_attested_bridge"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tolerant_parse(raw: str) -> str | None:
    s = str(raw or "").strip().upper()
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\bNOT[_ ]?ENTAIL(?:ED|LED)?\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAIL(?:ED|LED)?\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def label_tuple(by_key: dict[tuple[str, str, str, str], dict[str, Any]], cid: str, fid: str, context: str, teacher: str) -> str | None:
    r = by_key.get((cid, fid, context, teacher))
    return r.get("tol_label") if r else None


def all_expected(by_key: dict[tuple[str, str, str, str], dict[str, Any]], cid: str, fid: str, contexts: list[str], gold: str) -> bool:
    return all(label_tuple(by_key, cid, fid, ctx, t) == gold for ctx in contexts for t in TEACHERS)


def labels_by_context(by_key: dict[tuple[str, str, str, str], dict[str, Any]], cid: str, fid: str) -> dict[str, dict[str, str | None]]:
    return {ctx: {t: label_tuple(by_key, cid, fid, ctx, t) for t in TEACHERS} for ctx in CONTEXTS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(AUTO / "generated_fact_cases.jsonl")
    rows = read_jsonl(AUTO / "all_teacher_label_rows.jsonl")
    for r in rows:
        r["tol_label"] = tolerant_parse(r.get("output_raw", ""))
        r["tol_expected"] = (r["tol_label"] == r["gold_label"])
    by_key = {(r["case_id"], r["fact_id"], r["context_type"], r["teacher"]): r for r in rows}

    strict_cases = []
    full_stable_facts = []
    source_bridge_facts = []
    repair_queue = []
    reason_counts = collections.Counter()

    for c in cases:
        if not c.get("ok_case"):
            repair_queue.append({
                "case_id": c.get("case_id"),
                "level": "case",
                "reason": "fact_generation_parse_or_count_failure",
                "parse_error": c.get("parse_error"),
                "fact_generation_raw_head": str(c.get("fact_generation_raw", ""))[:1200],
                "source_text": c.get("source_text"),
                "source_attested_bridge": c.get("source_attested_bridge"),
                "natural_compact_reference": c.get("natural_compact_reference"),
            })
            reason_counts["fact_generation_parse_or_count_failure"] += 1
            continue
        case_full_ok = True
        case_facts = []
        for f in c["facts"]:
            cid, fid, gold = c["case_id"], f["fact_id"], f["gold_label"]
            full_ok = all_expected(by_key, cid, fid, CONTEXTS, gold)
            sb_ok = all_expected(by_key, cid, fid, SOURCE_BRIDGE, gold)
            fact_record = {
                "case_id": cid,
                "pair_id": c.get("pair_id"),
                "tier": c.get("tier"),
                "edit_class": c.get("edit_class"),
                "bucket": c.get("bucket"),
                "fact_id": fid,
                "gold_label": gold,
                "role_type": f.get("role_type"),
                "hypothesis": f.get("hypothesis"),
                "source_text": c.get("source_text"),
                "source_attested_bridge": c.get("source_attested_bridge"),
                "natural_compact_reference": c.get("natural_compact_reference"),
                "labels_by_context": labels_by_context(by_key, cid, fid),
            }
            if full_ok:
                full_stable_facts.append(dict(fact_record, stable_scope="source_bridge_natural"))
            if sb_ok:
                source_bridge_facts.append(dict(fact_record, stable_scope="source_bridge"))
            if not full_ok:
                case_full_ok = False
                reasons = []
                # Source overreach or bad negative: both teachers contradict gold on source.
                source_labels = [label_tuple(by_key, cid, fid, "source", t) for t in TEACHERS]
                bridge_labels = [label_tuple(by_key, cid, fid, "source_attested_bridge", t) for t in TEACHERS]
                natural_labels = [label_tuple(by_key, cid, fid, "natural_compact_reference", t) for t in TEACHERS]
                if all(x and x != gold for x in source_labels):
                    if gold == "ENTAILED":
                        reasons.append("generated_positive_unsupported_by_source")
                    else:
                        reasons.append("generated_negative_accepted_by_source")
                if all(x and x != gold for x in bridge_labels):
                    if gold == "ENTAILED":
                        reasons.append("source_attested_bridge_lost_positive_fact")
                    else:
                        reasons.append("generated_negative_accepted_by_bridge")
                if all(x and x != gold for x in natural_labels):
                    if gold == "ENTAILED":
                        reasons.append("natural_reference_lost_positive_fact")
                    else:
                        reasons.append("generated_negative_accepted_by_natural")
                if any(label_tuple(by_key, cid, fid, ctx, "qwen3.5-9b") != gold and label_tuple(by_key, cid, fid, ctx, "llama3.1-8b-instruct") == gold for ctx in CONTEXTS):
                    reasons.append("qwen_stricter_or_qwen_miss")
                if any(label_tuple(by_key, cid, fid, ctx, "llama3.1-8b-instruct") != gold and label_tuple(by_key, cid, fid, ctx, "qwen3.5-9b") == gold for ctx in CONTEXTS):
                    reasons.append("llama_overaccept_or_llama_miss")
                if not reasons:
                    reasons.append("mixed_context_teacher_disagreement")
                for rr in reasons:
                    reason_counts[rr] += 1
                repair_queue.append(dict(fact_record, level="fact", reasons=reasons))
            case_facts.append(fact_record)
        if case_full_ok:
            strict_cases.append({
                "case_id": c.get("case_id"),
                "pair_id": c.get("pair_id"),
                "tier": c.get("tier"),
                "edit_class": c.get("edit_class"),
                "bucket": c.get("bucket"),
                "source_text": c.get("source_text"),
                "source_attested_bridge": c.get("source_attested_bridge"),
                "natural_compact_reference": c.get("natural_compact_reference"),
                "facts": case_facts,
                "status": "all_4_facts_x_3_contexts_x_2_teachers_expected_after_parser_repair",
            })

    write_jsonl(OUT / "strict_retained_cases.jsonl", strict_cases)
    write_jsonl(OUT / "full_context_stable_facts.jsonl", full_stable_facts)
    write_jsonl(OUT / "source_bridge_stable_facts.jsonl", source_bridge_facts)
    write_jsonl(OUT / "repair_queue.jsonl", repair_queue)
    summary = {
        "status": "ROLE_SUBSTRATE_SEED_EXPORT",
        "created_utc": now(),
        "strict_retained_cases": len(strict_cases),
        "full_context_stable_facts": len(full_stable_facts),
        "source_bridge_stable_facts": len(source_bridge_facts),
        "repair_queue_rows": len(repair_queue),
        "reason_counts": dict(reason_counts),
        "strict_case_ids": [c["case_id"] for c in strict_cases],
        "files": {
            "strict_retained_cases": str(OUT / "strict_retained_cases.jsonl"),
            "full_context_stable_facts": str(OUT / "full_context_stable_facts.jsonl"),
            "source_bridge_stable_facts": str(OUT / "source_bridge_stable_facts.jsonl"),
            "repair_queue": str(OUT / "repair_queue.jsonl"),
        },
    }
    (OUT / "seed_export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
