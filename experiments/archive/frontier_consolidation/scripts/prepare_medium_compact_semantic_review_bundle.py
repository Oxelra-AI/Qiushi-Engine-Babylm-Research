#!/usr/bin/env python3
"""Prepare a bounded semantic-review bundle for medium FineWeb compact views.

The automatic filter is lexical. This bundle exposes accepted compact source/rewrite
pairs that stress relation preservation, negation/modality, roles/coreference, and
entity/number retention before any BabyLM training is launched.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
from typing import Any, Callable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_ROWS = ROOT / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data/medium_compact_analysis/medium_compact_ws_summary.json"
DEFAULT_OUT_JSON = ROOT / "data/medium_semantic_review/medium_compact_semantic_review_bundle.json"
DEFAULT_OUT_MD = (ROOT.parents[2] / 'research/notes/frontier_consolidation/medium_compact_semantic_review_bundle.md')

RELATION_RE = re.compile(r"\b(because|caused?|cause[sd]? by|due to|led to|result(?:ed|s)? in|so that|therefore|thus|help(?:s|ed)?|allow(?:s|ed)?|prevent(?:s|ed)?|reduce(?:s|d)?|increase(?:s|d)?|risk|effect|impact|in order to)\b", re.I)
NEG_MODAL_RE = re.compile(r"\b(no|not|never|without|unless|except|avoid|prevent|fail(?:s|ed)?|cannot|can't|won't|must|should|could|would|may|might|likely|unlikely|probably|possibly|suspect(?:s|ed)?|suggest(?:s|ed)?|suppose|hypothesis|if|although|despite|whereas|rather than)\b", re.I)
COREF_RE = re.compile(r"\b(who|whom|whose|which|that|this|these|those|it|its|they|their|them|he|his|she|her)\b|,\s*(?:who|which|a|an|the)\b|\([^)]{8,}\)", re.I)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def slim(r: dict[str, Any], group: str) -> dict[str, Any]:
    return {
        "group": group,
        "index": r.get("index"),
        "prompt_id": r.get("prompt_id"),
        "sentence_id": r.get("sentence_id"),
        "doc_id": r.get("doc_id"),
        "domain_hits": r.get("domain_hits") or [],
        "source_words": r.get("source_words"),
        "rewrite_words": r.get("rewrite_words"),
        "pair_words": r.get("pair_words"),
        "length_ratio": r.get("length_ratio"),
        "content_recall": r.get("content_recall"),
        "content_overlap": r.get("content_overlap"),
        "entity_recall": r.get("entity_recall"),
        "number_recall": r.get("number_recall"),
        "source_numbers": r.get("source_numbers") or [],
        "output_numbers": r.get("output_numbers") or [],
        "source_entities": r.get("source_entities") or [],
        "missing_entities": r.get("missing_entities") or [],
        "new_entity_like": r.get("new_entity_like") or [],
        "source_risks": r.get("source_risks") or [],
        "hard_reasons": r.get("hard_reasons") or [],
        "soft_flags": r.get("soft_flags") or [],
        "source_text": r.get("source_text"),
        "rewrite_text": r.get("rewrite_text"),
    }


def row_key(r: dict[str, Any]) -> str:
    return str(r.get("prompt_id") or f"{r.get('sentence_id')}|{r.get('doc_id')}|{r.get('index')}")


def take_ranked(seen: set[str], rows: list[dict[str, Any]], n: int, group: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(slim(r, group))
        if len(out) >= n:
            break
    return out


def contains(rx: re.Pattern[str], r: dict[str, Any]) -> bool:
    return bool(rx.search(str(r.get("source_text") or "")) or rx.search(str(r.get("rewrite_text") or "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--seed", type=int, default=829150)
    ap.add_argument("--per-group", type=int, default=16)
    args = ap.parse_args()

    rows_path = pathlib.Path(args.rows)
    summary_path = pathlib.Path(args.summary)
    all_rows = read_jsonl(rows_path)
    accepted = [r for r in all_rows if r.get("accepted_for_next_construction")]
    rng = random.Random(args.seed)

    def shuffled(xs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ys = list(xs)
        rng.shuffle(ys)
        return ys

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    groups.append((
        "compact_low_content_train_rows",
        sorted(accepted, key=lambda r: (r.get("content_recall") or 0.0, r.get("content_overlap") or 0.0, r.get("length_ratio") or 0.0)),
    ))
    groups.append((
        "compact_entity_or_number_sensitive",
        sorted(
            [r for r in accepted if (r.get("source_numbers") or []) or (r.get("source_entities") or []) or (r.get("entity_recall") or 1.0) < 1.0 or (r.get("number_recall") or 1.0) < 1.0],
            key=lambda r: ((r.get("entity_recall") or 1.0), (r.get("number_recall") or 1.0), -(len(r.get("source_entities") or []) + len(r.get("source_numbers") or []))),
        ),
    ))
    groups.append((
        "compact_relation_causal_roles",
        shuffled([r for r in accepted if "causal_relational" in (r.get("domain_hits") or []) or contains(RELATION_RE, r)]),
    ))
    groups.append((
        "compact_negation_modality_conditionals",
        shuffled([r for r in accepted if contains(NEG_MODAL_RE, r)]),
    ))
    groups.append((
        "compact_coreference_apposition_roles",
        shuffled([r for r in accepted if contains(COREF_RE, r) or any(x in (r.get("source_risks") or []) for x in ["long_parenthetical", "apostle_or_title_apposition"])]),
    ))
    groups.append(("compact_random_train_rows", shuffled(accepted)))

    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    for name, candidates in groups:
        examples += take_ranked(seen, candidates, args.per_group, name)

    eligible_counts = {
        "accepted": len(accepted),
        "low_content_candidates": len(accepted),
        "entity_or_number_sensitive_candidates": len(groups[1][1]),
        "relation_causal_candidates": len(groups[2][1]),
        "negation_modality_candidates": len(groups[3][1]),
        "coreference_apposition_candidates": len(groups[4][1]),
        "random_pool": len(accepted),
    }
    group_counts = dict(collections.Counter(e["group"] for e in examples))
    overall_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    payload = {
        "status": "MEDIUM_COMPACT_SEMANTIC_REVIEW_BUNDLE_PREPARED",
        "purpose": "Bounded independent semantic assessment of accepted medium compact views before training: relation, negation/modality, roles/coreference, entity and number fidelity.",
        "inputs": {"rows": str(rows_path), "summary": str(summary_path)},
        "counts": {
            "all_rows": len(all_rows),
            "accepted_rows": len(accepted),
            "examples": len(examples),
            "groups": group_counts,
            "eligible": eligible_counts,
        },
        "automatic_summary_overall": overall_summary.get("summary", {}).get("overall", {}),
        "automatic_top_hard_reasons": overall_summary.get("summary", {}).get("top_hard_reason_prefixes", []),
        "automatic_top_soft_flags": overall_summary.get("summary", {}).get("top_soft_flags", []),
        "examples": examples,
    }

    out_json = pathlib.Path(args.out_json)
    out_md = pathlib.Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research medium compact semantic review bundle",
        "",
        payload["purpose"],
        "",
        f"Rows: `{rows_path}`",
        f"Summary: `{summary_path}`",
        f"Bundle JSON: `{out_json}`",
        "",
        "## Automatic summary",
        "",
        json.dumps(payload["automatic_summary_overall"], indent=2, ensure_ascii=False),
        "",
        "## Examples",
        "",
    ]
    for i, e in enumerate(examples, 1):
        lines.append(f"### {i}. {e['group']} | {e['prompt_id']}")
        lines.append(
            "Metrics: "
            f"source_words={e['source_words']} rewrite_words={e['rewrite_words']} ratio={e['length_ratio']} "
            f"recall={e['content_recall']} overlap={e['content_overlap']} entity={e['entity_recall']} number={e['number_recall']} "
            f"domains={e['domain_hits']} risks={e['source_risks']} flags={e['soft_flags']} hard={e['hard_reasons']}"
        )
        if e["source_entities"] or e["missing_entities"] or e["source_numbers"]:
            lines.append(f"Entities: source={e['source_entities']} missing={e['missing_entities']} new={e['new_entity_like']} | numbers source={e['source_numbers']} output={e['output_numbers']}")
        lines.append(f"SOURCE: {e['source_text']}")
        lines.append(f"REWRITE: {e['rewrite_text']}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "json": str(out_json),
        "md": str(out_md),
        "counts": payload["counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
