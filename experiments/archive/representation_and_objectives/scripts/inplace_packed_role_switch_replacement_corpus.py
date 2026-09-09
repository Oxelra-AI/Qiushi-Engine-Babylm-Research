#!/usr/bin/env python3
"""research: in-place geometry-preserving role-switch replacement corpora.

This is the strictest candidate data object for the minimal from-scratch screen.
It removes exactly 138 intact 160-word OpenSubtitles rows from the original 10M
compact-view pool and replaces those exact row positions with 138 packed packet
rows of 160 words. Treatment and role-fixed control have matched row positions,
word positions, row counts, source-order geometry, and total <=10M word budget.

No model training is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path('.')
SCRIPT_DIR = ROOT / 'experiments/archive/representation_and_objectives/scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import packed_role_switch_replacement_corpus as prev  # noqa: E402

OUT = ROOT / 'experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus'


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_inplace_arm(base_rows: list[dict[str, Any]], removed_idx: set[int], packed_rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    packed_i = 0
    for i, r in enumerate(base_rows):
        if i in removed_idx:
            rr = dict(packed_rows[packed_i])
            rr['replaces_base_row_index'] = i
            rr['replaces_base_example_id'] = base_rows[i].get('example_id')
            packed_i += 1
            rows.append(rr)
        else:
            rows.append(dict(r))
    if packed_i != len(packed_rows):
        raise RuntimeError(f'{arm}: used {packed_i} packed rows but have {len(packed_rows)}')
    if len(rows) != len(base_rows):
        raise RuntimeError(f'{arm}: row count {len(rows)} != source {len(base_rows)}')
    words = prev.count_words(rows)
    if words != 10_000_000:
        raise RuntimeError(f'{arm}: words {words} != 10M')
    return rows


def repeat_rows(rows: list[dict[str, Any]], passes: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in range(passes):
        for r in rows:
            rr = dict(r)
            rr['pass_id'] = p
            out.append(rr)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src_sha = prev.sha256_file(prev.SRC_10M)
    if src_sha != prev.EXPECTED_SRC_SHA:
        raise RuntimeError(f'source sha mismatch {src_sha} != {prev.EXPECTED_SRC_SHA}')
    base_rows = prev.read_jsonl(prev.SRC_10M)
    if prev.count_words(base_rows) != 10_000_000:
        raise RuntimeError('source corpus is not 10M words')
    removed_idx, removed_rows = prev.select_removed_rows(base_rows, prev.REMOVE_WORDS)
    if len(removed_rows) != prev.PACKED_ROWS or any(int(r['words']) != prev.WORDS_PER_PACKED_ROW for r in removed_rows):
        raise RuntimeError('removed rows are not the expected 138 intact 160-word rows')

    treat = prev.packet_word_rows(prev.PKT_DIR / 'train_treatment.jsonl')
    fixed = prev.packet_word_rows(prev.PKT_DIR / 'train_role_fixed_control.jsonl')
    alignment = prev.compare_packet_alignment(treat, fixed)
    if not alignment['all_row_word_counts_equal']:
        raise RuntimeError('packet treatment/control are not position-word-count matched')

    packed = {
        'role_switch': prev.pack_packet_stream(treat, 'role_switch'),
        'role_fixed': prev.pack_packet_stream(fixed, 'role_fixed'),
    }
    arms = {arm: build_inplace_arm(base_rows, removed_idx, rows, arm) for arm, rows in packed.items()}

    files = {'removed_rows': str(OUT / 'removed_open_subtitles_rows.jsonl')}
    write_jsonl(OUT / 'removed_open_subtitles_rows.jsonl', removed_rows)
    for arm, rows in arms.items():
        p10 = OUT / f'{arm}_inplace_packed_replacement_10M.jsonl'
        p100 = OUT / f'{arm}_inplace_packed_replacement_100M.jsonl'
        write_jsonl(p10, rows)
        write_jsonl(p100, repeat_rows(rows, prev.PASSES))
        files[f'{arm}_10M'] = str(p10)
        files[f'{arm}_100M'] = str(p100)

    sorted_removed = sorted(removed_idx)
    manifest: dict[str, Any] = {
        'status': 'INPLACE_PACKED_ROLE_SWITCH_REPLACEMENT_CORPUS',
        'legal_status': 'candidate <=10M from-beginning replacement corpora; tokenizer/model must be trained from each revised corpus for submission-facing evidence',
        'scientific_reason': 'minimum from-scratch screen for natural transfer: preserve original row count and replacement positions while comparing role-switch against structurally matched role-fixed text',
        'source_10M': str(prev.SRC_10M),
        'source_10M_sha256': src_sha,
        'source_10M_words': 10_000_000,
        'source_10M_rows': len(base_rows),
        'removal_policy': {
            'source': prev.REMOVE_SOURCE,
            'words': prev.REMOVE_WORDS,
            'rows': len(removed_rows),
            'words_per_row': prev.WORDS_PER_PACKED_ROW,
            'base_row_index_first_last': [sorted_removed[0], sorted_removed[-1]],
            'removed_example_id_first_last': [removed_rows[0].get('example_id'), removed_rows[-1].get('example_id')],
            'in_place_replacement': True,
        },
        'packet_inputs': {
            'manifest': str(prev.PKT_DIR / 'manifest.json'),
            'role_switch_words': prev.count_words(treat),
            'role_fixed_words': prev.count_words(fixed),
            'original_packet_rows_per_arm': len(treat),
            'packed_rows_per_arm': prev.PACKED_ROWS,
            'packed_words_per_row': prev.WORDS_PER_PACKED_ROW,
            'position_alignment': alignment,
        },
        'arms': {},
        'files': files,
        'code_hash': prev.sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/inplace_packed_role_switch_replacement_corpus.py')),
    }
    for arm, rows in arms.items():
        p10 = pathlib.Path(files[f'{arm}_10M'])
        p100 = pathlib.Path(files[f'{arm}_100M'])
        manifest['arms'][arm] = {
            'rows_10M': len(rows),
            'words_10M': prev.count_words(rows),
            'rows_100M': len(rows) * prev.PASSES,
            'words_100M': prev.count_words(rows) * prev.PASSES,
            'source_counts_10M': prev.source_counts(rows),
            'sha256_10M': prev.sha256_file(p10),
            'sha256_100M': prev.sha256_file(p100),
        }
    man = OUT / 'manifest.json'
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'INPLACE_PACKED_REPLACEMENT_CORPUS_DONE',
        'source_rows': len(base_rows),
        'removed_rows': len(removed_rows),
        'inserted_rows': prev.PACKED_ROWS,
        'row_count_preserved': all(v['rows_10M'] == len(base_rows) for v in manifest['arms'].values()),
        'replacement_base_row_index_first_last': manifest['removal_policy']['base_row_index_first_last'],
        'role_switch_10M': files['role_switch_10M'],
        'role_fixed_10M': files['role_fixed_10M'],
        'manifest': str(man),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
