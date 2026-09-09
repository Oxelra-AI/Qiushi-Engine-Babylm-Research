#!/usr/bin/env python3
"""research: geometry-preserving packed role-switch replacement corpora.

research made legal <=10M replacement corpora, but inserted 1,560 short packet
rows after removing 138 ordinary 160-word OpenSubtitles rows. That is legal for
corpus accounting and treatment/control share the same row count, but it changes
the optimizer-batch geometry relative to the compact-view anchor. This script
builds the stricter version for the minimal from-scratch screen: remove exactly
the same 138 intact OpenSubtitles rows (22,080 words) and insert the same packet
word stream packed back into exactly 138 160-word rows. The role-switch and
role-fixed arms therefore have the same total words and row count as the source
10M pool, with matched packet/control word positions.

No model training is performed here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

ROOT = pathlib.Path('.')
SRC_10M = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
PKT_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/role_switch_expanded_packets'
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/packed_role_switch_replacement_corpus'
EXPECTED_SRC_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
REMOVE_SOURCE = 'open_subtitles'
REMOVE_WORDS = 22_080
WORDS_PER_PACKED_ROW = 160
PACKED_ROWS = REMOVE_WORDS // WORDS_PER_PACKED_ROW
PASSES = 10


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def count_words(rows: list[dict[str, Any]]) -> int:
    return sum(int(r.get('words', r.get('word_count', len(str(r['text']).split())))) for r in rows)


def select_removed_rows(base_rows: list[dict[str, Any]], target_words: int):
    remove_idx: list[int] = []
    total = 0
    for i, r in enumerate(base_rows):
        words = int(r['words'])
        if r.get('source') != REMOVE_SOURCE:
            continue
        if total + words > target_words:
            continue
        remove_idx.append(i)
        total += words
        if total == target_words:
            break
    if total != target_words:
        raise RuntimeError(f'could not select exact {target_words} words from {REMOVE_SOURCE}; got {total}')
    return set(remove_idx), [base_rows[i] for i in remove_idx]


def packet_word_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        text = str(r['text'])
        words = text.split()
        wc = int(r.get('word_count', len(words)))
        if wc != len(words):
            raise RuntimeError(f'packet word mismatch {path.name} row {i}: field={wc} actual={len(words)}')
        out.append({**r, 'words': wc, '_tokens': words})
    if count_words(out) != REMOVE_WORDS:
        raise RuntimeError(f'{path.name} words {count_words(out)} != {REMOVE_WORDS}')
    return out


def compare_packet_alignment(treat: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    if len(treat) != len(fixed):
        raise RuntimeError(f'packet row count mismatch {len(treat)} vs {len(fixed)}')
    mismatched_lengths = []
    mismatched_keys = []
    for i, (a, b) in enumerate(zip(treat, fixed)):
        if a['words'] != b['words']:
            mismatched_lengths.append({'i': i, 'treatment_words': a['words'], 'role_fixed_words': b['words']})
        key_a = (a.get('family'), a.get('template_id'), a.get('pair_id'), a.get('direction'))
        key_b = (b.get('family'), b.get('template_id'), b.get('pair_id'), b.get('direction'))
        if key_a != key_b:
            mismatched_keys.append({'i': i, 'treatment_key': key_a, 'role_fixed_key': key_b})
    return {
        'rows': len(treat),
        'all_row_word_counts_equal': not mismatched_lengths,
        'mismatched_length_count': len(mismatched_lengths),
        'mismatched_length_examples': mismatched_lengths[:10],
        'all_metadata_keys_equal': not mismatched_keys,
        'mismatched_key_count': len(mismatched_keys),
        'mismatched_key_examples': mismatched_keys[:10],
    }


def pack_packet_stream(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    stream: list[str] = []
    packet_meta: list[dict[str, Any]] = []
    cursor = 0
    for i, r in enumerate(rows):
        toks = list(r['_tokens'])
        start = cursor
        stream.extend(toks)
        cursor += len(toks)
        packet_meta.append({
            'row_index': i,
            'start_word_offset': start,
            'end_word_offset_exclusive': cursor,
            'family': r.get('family'),
            'template_id': r.get('template_id'),
            'pair_id': r.get('pair_id'),
            'direction': r.get('direction'),
            'word_count': len(toks),
        })
    if len(stream) != REMOVE_WORDS:
        raise RuntimeError(f'{arm} packet stream words {len(stream)} != {REMOVE_WORDS}')
    packed: list[dict[str, Any]] = []
    for j in range(PACKED_ROWS):
        start = j * WORDS_PER_PACKED_ROW
        end = start + WORDS_PER_PACKED_ROW
        words = stream[start:end]
        if len(words) != WORDS_PER_PACKED_ROW:
            raise RuntimeError(f'{arm} packed row {j} has {len(words)} words')
        # Include compact family counts for audit without storing huge per-word maps.
        fam = Counter()
        for m in packet_meta:
            if m['end_word_offset_exclusive'] <= start:
                continue
            if m['start_word_offset'] >= end:
                break
            fam[str(m.get('family'))] += 1
        packed.append({
            'text': ' '.join(words),
            'words': WORDS_PER_PACKED_ROW,
            'example_id': 9_146_000_000 + (0 if arm == 'role_switch' else 1_000_000) + j,
            'source': f'step146_{arm}_packet_packed160',
            'packed_row_index': j,
            'packet_word_start': start,
            'packet_word_end_exclusive': end,
            'packet_family_touch_counts': dict(sorted(fam.items())),
        })
    return packed


def build_arm(base_rows: list[dict[str, Any]], removed_idx: set[int], packed_rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    rows = [dict(r) for i, r in enumerate(base_rows) if i not in removed_idx]
    rows.extend(packed_rows)
    if count_words(rows) != 10_000_000:
        raise RuntimeError(f'{arm} word total {count_words(rows)} != 10M')
    if len(rows) != len(base_rows):
        raise RuntimeError(f'{arm} row count {len(rows)} != source row count {len(base_rows)}')
    return rows


def repeat_rows(rows: list[dict[str, Any]], passes: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in range(passes):
        for r in rows:
            rr = dict(r)
            rr['pass_id'] = p
            out.append(rr)
    return out


def source_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(); wc = Counter()
    for r in rows:
        src = str(r.get('source', ''))
        c[src] += 1
        wc[src] += int(r['words'])
    return {'rows': dict(sorted(c.items())), 'words': dict(sorted(wc.items()))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_file(SRC_10M)
    if src_sha != EXPECTED_SRC_SHA:
        raise RuntimeError(f'source 10M sha mismatch {src_sha} != {EXPECTED_SRC_SHA}')
    base_rows = read_jsonl(SRC_10M)
    if count_words(base_rows) != 10_000_000:
        raise RuntimeError('source corpus is not exactly 10M words')
    removed_idx, removed_rows = select_removed_rows(base_rows, REMOVE_WORDS)
    if len(removed_rows) != PACKED_ROWS or any(int(r['words']) != WORDS_PER_PACKED_ROW for r in removed_rows):
        raise RuntimeError('removed rows are not exactly 138 intact 160-word rows')

    treat = packet_word_rows(PKT_DIR / 'train_treatment.jsonl')
    fixed = packet_word_rows(PKT_DIR / 'train_role_fixed_control.jsonl')
    alignment = compare_packet_alignment(treat, fixed)
    if not alignment['all_row_word_counts_equal']:
        raise RuntimeError('treatment/control packet row word counts are not position matched')

    packed = {
        'role_switch': pack_packet_stream(treat, 'role_switch'),
        'role_fixed': pack_packet_stream(fixed, 'role_fixed'),
    }
    arms = {arm: build_arm(base_rows, removed_idx, rows, arm) for arm, rows in packed.items()}

    files = {'removed_rows': str(OUT / 'removed_open_subtitles_rows.jsonl')}
    write_jsonl(OUT / 'removed_open_subtitles_rows.jsonl', removed_rows)
    for arm, rows in arms.items():
        p10 = OUT / f'{arm}_packed_replacement_10M.jsonl'
        p100 = OUT / f'{arm}_packed_replacement_100M.jsonl'
        write_jsonl(p10, rows)
        write_jsonl(p100, repeat_rows(rows, PASSES))
        files[f'{arm}_10M'] = str(p10)
        files[f'{arm}_100M'] = str(p100)

    manifest: dict[str, Any] = {
        'status': 'PACKED_ROLE_SWITCH_REPLACEMENT_CORPUS',
        'legal_status': 'candidate <=10M from-beginning replacement corpora; train tokenizer/model from each revised corpus for submission-facing evidence',
        'scientific_reason': 'preserve source 10M row count and optimizer-batch geometry while testing role-switch vs role-fixed exchange-specific natural transfer',
        'source_10M': str(SRC_10M),
        'source_10M_sha256': src_sha,
        'source_10M_words': 10_000_000,
        'source_10M_rows': len(base_rows),
        'removal_policy': {
            'source': REMOVE_SOURCE,
            'words': REMOVE_WORDS,
            'rows': len(removed_rows),
            'words_per_row': WORDS_PER_PACKED_ROW,
            'removed_example_id_first_last': [removed_rows[0].get('example_id'), removed_rows[-1].get('example_id')],
        },
        'packet_inputs': {
            'manifest': str(PKT_DIR / 'manifest.json'),
            'role_switch_words': count_words(treat),
            'role_fixed_words': count_words(fixed),
            'original_packet_rows_per_arm': len(treat),
            'packed_rows_per_arm': PACKED_ROWS,
            'packed_words_per_row': WORDS_PER_PACKED_ROW,
            'position_alignment': alignment,
        },
        'arms': {},
        'files': files,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/packed_role_switch_replacement_corpus.py')),
    }
    for arm, rows in arms.items():
        p10 = pathlib.Path(files[f'{arm}_10M'])
        p100 = pathlib.Path(files[f'{arm}_100M'])
        manifest['arms'][arm] = {
            'rows_10M': len(rows),
            'words_10M': count_words(rows),
            'rows_100M': len(rows) * PASSES,
            'words_100M': count_words(rows) * PASSES,
            'source_counts_10M': source_counts(rows),
            'sha256_10M': sha256_file(p10),
            'sha256_100M': sha256_file(p100),
        }
    man = OUT / 'manifest.json'
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'PACKED_REPLACEMENT_CORPUS_DONE',
        'source_rows': len(base_rows),
        'removed_rows': len(removed_rows),
        'inserted_packed_rows': PACKED_ROWS,
        'row_count_preserved': all(v['rows_10M'] == len(base_rows) for v in manifest['arms'].values()),
        'role_switch_10M': files['role_switch_10M'],
        'role_fixed_10M': files['role_fixed_10M'],
        'manifest': str(man),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
