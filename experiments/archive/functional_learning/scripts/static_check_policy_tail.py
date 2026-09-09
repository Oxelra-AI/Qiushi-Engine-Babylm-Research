#!/usr/bin/env python3
"""Static checks for research structural policy tail and segment metadata."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, hashlib, re, sys
from collections import Counter
from typing import Any, Dict

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DECISIONS = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/structural_nonexact_decisions.jsonl')
TAIL = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest/compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl')
OUT = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/static_tail_check.json')
DEMONSTRATED = {"rw2s0_030370", "rw2s0_029078", "rw_035551"}


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def wc(t): return len((t or "").strip().split())

def norm(t): return re.sub(r"\s+", " ", (t or "").strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'"))

def iter_jsonl(path):
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)


def main():
    dec: Dict[str, Dict[str, Any]] = {r["pair_id"]: r for r in iter_jsonl(DECISIONS)}
    admitted = [r for r in dec.values() if r.get("admit_compact")]
    exact_admitted = [r["pair_id"] for r in admitted if r.get("compact_equals_original") or norm(r.get("compact_rewrite")) == norm(r.get("original"))]
    demonstrated_admitted = [r["pair_id"] for r in admitted if r["pair_id"] in DEMONSTRATED]
    bad_label_admitted = [r["pair_id"] for r in admitted if r.get("semantic_label") == "faithful_shortening"]
    kind_counts = Counter(r.get("candidate_kind") for r in dec.values())
    admitted_kind_counts = Counter(r.get("candidate_kind") for r in admitted)

    qwen_rows = matched_rows = modified_rows = topup_rows = 0
    row_words_sum = 0
    compact_occ = 0
    bad_segments = []
    bad_modified = []
    modified_ids = Counter()
    segment_kind_counts = Counter()
    for i, row in enumerate(iter_jsonl(TAIL)):
        row_words_sum += int(row.get("words", wc(row.get("text", ""))))
        text = row.get("text", "")
        if row.get("source") == "qwen_pair_packed":
            qwen_rows += 1
            segs = row.get("qwen_pair_segments") or []
            if segs:
                matched_rows += 1
            for seg in segs:
                segment_kind_counts[seg.get("candidate_kind")] += 1
                vs, ve = int(seg.get("view_start", -1)), int(seg.get("view_end", -1))
                ss, se = int(seg.get("source_start", -1)), int(seg.get("source_end", -1))
                if not (0 <= ss <= se <= len(text)) or not (0 <= vs <= ve <= len(text)):
                    bad_segments.append({"row": i, "pair_id": seg.get("pair_id"), "reason": "span_out_of_range"})
                    continue
                if text[ss:se] != seg.get("source_text", "") or text[vs:ve] != seg.get("view_text", ""):
                    bad_segments.append({"row": i, "pair_id": seg.get("pair_id"), "reason": "span_text_mismatch", "source_ok": text[ss:se] == seg.get("source_text", ""), "view_ok": text[vs:ve] == seg.get("view_text", "")})
                    if len(bad_segments) > 20: break
        if row.get("compact_modified_pair_ids"):
            modified_rows += 1
            ids = row.get("compact_modified_pair_ids") or []
            compact_occ += len(ids)
            for pid in ids:
                modified_ids[pid] += 1
                d = dec.get(pid)
                if not d or not d.get("admit_compact") or d.get("candidate_kind") != "unverified_structural_shortening_candidate":
                    bad_modified.append({"row": i, "pair_id": pid, "decision": None if d is None else {"admit": d.get("admit_compact"), "kind": d.get("candidate_kind")}})
        if str(row.get("source", "")).startswith("structural_compact_"):
            topup_rows += 1
    failures = {
        "exact_source_return_admitted": exact_admitted[:20],
        "demonstrated_admitted": demonstrated_admitted,
        "faithful_label_admitted": bad_label_admitted[:20],
        "bad_segments_first20": bad_segments[:20],
        "bad_modified_first20": bad_modified[:20],
    }
    passed = not any(failures.values())
    out = {
        "status": "STATIC_TAIL_CHECK_PASS" if passed else "STATIC_TAIL_CHECK_FAIL",
        "decisions": rel(DECISIONS),
        "tail": rel(TAIL),
        "n_decisions": len(dec),
        "n_admitted": len(admitted),
        "decision_kind_counts": dict(kind_counts),
        "admitted_kind_counts": dict(admitted_kind_counts),
        "qwen_rows": qwen_rows,
        "qwen_rows_with_segments": matched_rows,
        "modified_qwen_rows": modified_rows,
        "compact_modified_pair_occurrences": compact_occ,
        "modified_unique_pairs": len(modified_ids),
        "modified_occurrence_minmax": [min(modified_ids.values()) if modified_ids else 0, max(modified_ids.values()) if modified_ids else 0],
        "topup_rows": topup_rows,
        "tail_total_words": row_words_sum,
        "legal_total_words_if_from_coherent86": 86005295 + row_words_sum,
        "segment_kind_counts": dict(segment_kind_counts),
        "failures": failures,
        "interpretation": "Pass means the built tail preserves provenance: no faithful_shortening labels, no exact-source returns admitted, no demonstrated failures admitted, and qwen segment char spans align with row text.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    sys.exit(0 if passed else 1)

if __name__ == "__main__": main()
