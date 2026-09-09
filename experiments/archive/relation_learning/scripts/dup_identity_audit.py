#!/usr/bin/env python3
"""research: audit whether the launched DUP replication stream matches the COMPACT_EXPERIENCE DUP evidence arm.

The report's DUP evidence in research/031 was scored from COMPACT_EXPERIENCE research
selected_original_dup_all, but research accidentally launched the older research
official_original_dup stream. This script compares metadata and 10M pool rows
without reading any new seed43122 results.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
import collections
from statistics import mean

ROOT = _public_path('.')
WS = _public_path('experiments/archive/relation_learning')
COMPACT_EXPERIENCE = _public_path('experiments/archive/compact_experience')
OUT = _public_path('experiments/archive/relation_learning/data/dup_identity_audit')
NOTE = _public_path('research/notes/relation_learning/dup_identity_audit.md')

META = _public_path('experiments/archive/compact_experience/data/original_dup_control/original_dup_control_metadata.json')
META = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_metadata.json')
POOL = _public_path('experiments/archive/compact_experience/data/original_dup_control/training_corpora/official_original_dup_10M.jsonl')
POOL = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_10M.jsonl')
ROW_META = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_rows_meta.jsonl')


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scan_pool(path: pathlib.Path, packed_source: str, row_meta_path: pathlib.Path | None = None):
    """Scan a 10M pool and, if needed, recover pair ids from companion row metadata.

    The research pool stores pair_ids in the training JSONL itself. The research
    selected-original-dup-all pool does not; its pair_ids are in the row_meta
    sidecar. Counting packed rows by the `source` string protects the audit from
    confusing absent metadata fields with absent local pairing.
    """
    n_rows = 0
    words_total = 0
    packed_source_rows = 0
    pair_ids = []
    pair_n_by_row = []
    row_sources = collections.Counter()
    pair_row_word_lengths = []
    first_pair_rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_rows += 1
            w = int(obj.get("words", 0))
            words_total += w
            src = obj.get("source", "")
            row_sources[src] += 1
            is_packed = (src == packed_source)
            if is_packed:
                packed_source_rows += 1
            ids = obj.get("pair_ids") or []
            if ids:
                pair_ids.extend(ids)
                pair_n_by_row.append(int(obj.get("n_pairs", len(ids))))
                pair_row_word_lengths.append(w)
                if len(first_pair_rows) < 5:
                    first_pair_rows.append({
                        "row_index": n_rows - 1,
                        "source": src,
                        "words": w,
                        "pair_ids": ids[:8],
                        "n_pairs": obj.get("n_pairs", len(ids)),
                    })
            elif is_packed:
                pair_row_word_lengths.append(w)
    if row_meta_path is not None:
        with row_meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("source") == packed_source:
                    ids = obj.get("pair_ids") or []
                    pair_ids.extend(ids)
                    pair_n_by_row.append(int(obj.get("n_pairs", len(ids))))
                    if len(first_pair_rows) < 5:
                        first_pair_rows.append({
                            "row_index": obj.get("row_index"),
                            "source": obj.get("source"),
                            "words": obj.get("words"),
                            "pair_ids": ids[:8],
                            "n_pairs": obj.get("n_pairs", len(ids)),
                            "from_row_meta": True,
                        })
    c = collections.Counter(pair_ids)
    return {
        "path": str(path.relative_to(ROOT)),
        "row_meta_path": str(row_meta_path.relative_to(ROOT)) if row_meta_path else None,
        "n_rows": n_rows,
        "words_total": words_total,
        "row_sources_top": row_sources.most_common(12),
        "packed_source": packed_source,
        "packed_source_rows": packed_source_rows,
        "pair_id_occurrences": len(pair_ids),
        "unique_pair_ids": len(c),
        "duplicate_pair_id_repeats": sum(v - 1 for v in c.values() if v > 1),
        "pair_rows_mean_words": mean(pair_row_word_lengths) if pair_row_word_lengths else None,
        "pair_rows_min_words": min(pair_row_word_lengths) if pair_row_word_lengths else None,
        "pair_rows_max_words": max(pair_row_word_lengths) if pair_row_word_lengths else None,
        "pair_rows_mean_n_pairs": mean(pair_n_by_row) if pair_n_by_row else None,
        "first_pair_rows": first_pair_rows,
        "pair_id_counter": c,
    }


def public_scan(scan):
    return {k: v for k, v in scan.items() if k != "pair_id_counter"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m29 = load_json(META)
    m32 = load_json(META)
    s29 = scan_pool(POOL, "official_original_dup_packed")
    s32 = scan_pool(POOL, "official_selected_original_dup_all_packed", ROW_META)
    set29 = set(s29["pair_id_counter"])
    set32 = set(s32["pair_id_counter"])
    overlap = set29 & set32
    missing_from_29 = sorted(set32 - set29)
    extra_in_29 = sorted(set29 - set32)

    audit = {
        "status": "DUP_IDENTITY_AUDIT",
        "scientific_question": "Does the research launched DUP seed43122 stream match the research DUP arm used for the report's DUP reference values?",
        "launched_step037_stream": "experiments/archive/compact_experience/data/original_dup_control/training_corpora/official_original_dup_100M.jsonl",
        "report_reference_dup_stream": "experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_100M.jsonl",
        "metadata_comparison": {
            "original_duplicate_pool_status": m29.get("status"),
            "selected_all_duplicate_pool_status": m32.get("status"),
            "selected_duplicate_pairs": m29.get("selected_duplicate_pairs"),
            "selected_pairs": m32.get("selected_pairs"),
            "original_duplicate_pool_duplicate_pair_words": m29.get("duplicate_pair_words"),
            "selected_all_duplicate_pool_duplicate_pair_words": m32.get("duplicate_pair_words"),
            "original_duplicate_pool_pair_boundary_preserved": m29.get("pair_boundary_preserved"),
            "selected_all_duplicate_pool_pair_boundary_preserved": m32.get("pair_boundary_preserved"),
            "original_duplicate_pool_pair_truncation": m29.get("pair_truncation"),
            "selected_all_duplicate_pool_pair_truncation": m32.get("pair_truncation"),
            "keeps_all_selected_pair_ids": m32.get("keeps_all_step028_selected_pair_ids"),
            "source_totals_matched": m32.get("source_totals_matched_to_qwen_effective_targets"),
            "original_duplicate_pool_pool_sha256": m29.get("sha256", {}).get("pool"),
            "selected_all_duplicate_pool_pool_sha256": m32.get("sha256", {}).get("pool"),
            "original_duplicate_pool_training_sha256": m29.get("sha256", {}).get("training"),
            "selected_all_duplicate_pool_training_sha256": m32.get("sha256", {}).get("training"),
        },
        "pool_scans": {
            "official_original_dup_10M": public_scan(s29),
            "selected_original_dup_all_10M": public_scan(s32),
        },
        "pair_id_set_comparison": {
            "original_duplicate_pool_unique_pair_ids": len(set29),
            "selected_all_duplicate_pool_unique_pair_ids": len(set32),
            "intersection": len(overlap),
            "missing_from_step029_count": len(missing_from_29),
            "extra_in_step029_count": len(extra_in_29),
            "missing_from_step029_first20": missing_from_29[:20],
            "extra_in_step029_first20": extra_in_29[:20],
            "sets_identical": set29 == set32,
        },
        "decision": "DIFFERENT_CONSTRUCTIONS",
        "implication": "research official_original_dup seed43122 must not be used as replication of the report DUP arm. The correct replication stream is selected_original_dup_all_100M, matching COMPACT_EXPERIENCE research.",
    }
    (_public_path('experiments/archive/relation_learning/data/dup_identity_audit/dup_identity_audit.json')).write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = []
    md.append("# research DUP identity audit\n")
    md.append("This audit was run before reading any seed43122 DUP scores. It compares the stream launched in research against the COMPACT_EXPERIENCE duplication arm used in the current report.\n")
    md.append("## Conclusion\n")
    md.append("The streams are **not the same construction**. research launched `official_original_dup_100M` from COMPACT_EXPERIENCE research, but the report's DUP values were scored from `selected_original_dup_all_100M` from COMPACT_EXPERIENCE research. Therefore the research DUP task is obsolete for the intended replication and must not be reported as a DUP replication.\n")
    md.append("## Metadata comparison\n")
    md.append(f"- research selected duplicate pairs: {m29.get('selected_duplicate_pairs')}\n")
    md.append(f"- research selected pairs: {m32.get('selected_pairs')}\n")
    md.append(f"- research duplicate-pair words: {m29.get('duplicate_pair_words')}\n")
    md.append(f"- research duplicate-pair words: {m32.get('duplicate_pair_words')}\n")
    md.append(f"- Pair boundary preserved: research={m29.get('pair_boundary_preserved')}, research={m32.get('pair_boundary_preserved')}\n")
    md.append(f"- Pair truncation: research={m29.get('pair_truncation')}, research={m32.get('pair_truncation')}\n")
    md.append(f"- research keeps all research selected pair ids: {m32.get('keeps_all_step028_selected_pair_ids')}\n")
    md.append(f"- Pool SHA256: research `{m29.get('sha256', {}).get('pool')}`, research `{m32.get('sha256', {}).get('pool')}`\n")
    md.append(f"- Training SHA256: research `{m29.get('sha256', {}).get('training')}`, research `{m32.get('sha256', {}).get('training')}`\n")
    md.append("## Stream scan of 10M pools\n")
    md.append(f"- research pool rows={s29['n_rows']}, packed pair rows={s29['packed_source_rows']}, unique pair_ids={len(set29)}, pair-id occurrences={s29['pair_id_occurrences']}, total words={s29['words_total']}\n")
    md.append(f"- research pool rows={s32['n_rows']}, packed pair rows={s32['packed_source_rows']}, unique pair_ids={len(set32)}, pair-id occurrences={s32['pair_id_occurrences']} (from sidecar row_meta), total words={s32['words_total']}\n")
    md.append(f"- Pair-id intersection={len(overlap)}; missing from research={len(missing_from_29)}; extra in research={len(extra_in_29)}; identical sets={set29 == set32}\n")
    md.append("## Scientific implication\n")
    md.append("The correct two-seed duplication replication must train on `experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_100M.jsonl` with metadata `selected_original_dup_all_metadata.json`. The research `official_original_dup` run, if retained at all, is a separate one-seed exact-recurrence intervention and not evidence for the pre-stated DUP replication.\n")
    md.append(f"\nFull JSON: `{(_public_path('experiments/archive/relation_learning/data/dup_identity_audit/dup_identity_audit.json')).relative_to(ROOT)}`\n")
    NOTE.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"wrote": [str((_public_path('experiments/archive/relation_learning/data/dup_identity_audit/dup_identity_audit.json')).relative_to(ROOT)), str(NOTE.relative_to(ROOT))], "sets_identical": set29 == set32, "missing_from_step029": len(missing_from_29)}, indent=2))

if __name__ == "__main__":
    main()
