#!/usr/bin/env python3
"""Audit row-structure quality of cached INITIAL_MODEL_STUDIES FineWeb-Edu random-quality corpus."""
from __future__ import annotations

import json
import pathlib
import collections
from typing import Any

INP = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
OUT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/fineweb_row_structure_audit.json")
NOTE = pathlib.Path("research/notes/representation_and_objectives/cached_fineweb_row_structure_audit.md")

BAD_PREFIXES = (
    'including ', 'of ', 'and ', 'or ', 'to ', 'in ', 'for ', 'with ', 'from ', 'regarded ',
    'soldiers ', 'theme ', 'blog post ', 'systematic ',
)


def startish(text: str) -> bool:
    t = str(text or '').lstrip()
    if not t:
        return False
    low = t.lower()
    if low.startswith(BAD_PREFIXES):
        return False
    if t.startswith(('|', '--', ',', ';', ':')):
        return False
    if not (t[0].isupper() or t[0].isdigit() or t[0] in '“"\'('):
        return False
    return True


def main() -> None:
    doc_count_dist: collections.Counter[str] = collections.Counter()
    total_rows = 0
    total_words = 0
    single_rows = 0
    single_words = 0
    startish_rows = 0
    single_startish_rows = 0
    single_startish_words = 0
    by_single_doc: collections.Counter[str] = collections.Counter()
    by_single_startish_doc: collections.Counter[str] = collections.Counter()
    examples_multi_or_frag: list[dict[str, Any]] = []
    examples_single_startish: list[dict[str, Any]] = []
    with INP.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            ids = o.get('doc_ids') or []
            words = int(o.get('words', len(str(o.get('text','')).split())))
            txt = str(o.get('text',''))
            s_ok = startish(txt)
            doc_count_dist[str(len(ids))] += 1
            total_rows += 1
            total_words += words
            if s_ok:
                startish_rows += 1
            if len(ids) == 1:
                single_rows += 1
                single_words += words
                by_single_doc[str(ids[0])] += 1
                if s_ok:
                    single_startish_rows += 1
                    single_startish_words += words
                    by_single_startish_doc[str(ids[0])] += 1
                    if len(examples_single_startish) < 8:
                        examples_single_startish.append({'row': i, 'doc_ids': ids, 'words': words, 'start': txt[:400]})
            if len(examples_multi_or_frag) < 12 and (len(ids) != 1 or not s_ok):
                examples_multi_or_frag.append({'row': i, 'doc_ids': ids, 'words': words, 'startish': s_ok, 'start': txt[:400]})
    multi_rows = sum(v for k, v in doc_count_dist.items() if int(k) > 1)
    payload = {
        'status': 'CACHED_FINEWEB_ROW_STRUCTURE_AUDIT',
        'input': str(INP),
        'rows': total_rows,
        'words': total_words,
        'doc_id_count_distribution': dict(doc_count_dist),
        'multi_doc_rows': multi_rows,
        'multi_doc_fraction': multi_rows / total_rows if total_rows else None,
        'single_doc_rows': single_rows,
        'single_doc_words': single_words,
        'single_doc_fraction_words': single_words / total_words if total_words else None,
        'startish_rows': startish_rows,
        'startish_fraction': startish_rows / total_rows if total_rows else None,
        'single_doc_startish_rows': single_startish_rows,
        'single_doc_startish_words': single_startish_words,
        'single_doc_startish_fraction_words': single_startish_words / total_words if total_words else None,
        'unique_single_docs': len(by_single_doc),
        'unique_single_startish_docs': len(by_single_startish_doc),
        'single_rows_per_doc_top10': by_single_doc.most_common(10),
        'single_startish_rows_per_doc_top10': by_single_startish_doc.most_common(10),
        'examples_multi_or_fragmentish': examples_multi_or_frag,
        'examples_single_doc_startish': examples_single_startish,
        'interpretation': 'The cached INITIAL_MODEL_STUDIES 3M FineWeb-Edu file is public and broad but row-packed: many rows cross document IDs and many chunks start mid-fragment. A cleaner future arm should prefer at least the single-doc subset unless the scientific question intentionally tests noisy web packing.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research cached FineWeb row-structure audit\n\n',
        f"Input: `{INP}`\n\n",
        f"Rows/words: {total_rows:,} / {total_words:,}.\n\n",
        f"Document-ID count distribution: `{dict(doc_count_dist)}`. Multi-doc rows: {multi_rows:,} ({multi_rows/total_rows:.2%}).\n\n",
        f"Single-document rows: {single_rows:,} ({single_words:,} words, {single_words/total_words:.2%} of cached words).\n\n",
        f"Startish rows by heuristic: {startish_rows:,} ({startish_rows/total_rows:.2%}); single-doc startish words: {single_startish_words:,}.\n\n",
        'Interpretation: use the full 3M cached arm only as a broad but noisy fallback. A cleaner candidate should use the single-document subset (~1.77M words) plus matched official filler, because same-source semantic-view work already showed that row-boundary/coherence confounds can matter.\n\n',
        f"JSON: `{OUT}`\n",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'json': str(OUT), 'note': str(NOTE), 'multi_doc_fraction': payload['multi_doc_fraction'], 'single_doc_words': single_words, 'single_doc_startish_words': single_startish_words}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
