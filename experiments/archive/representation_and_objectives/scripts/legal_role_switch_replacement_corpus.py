#!/usr/bin/env python3
"""research: legal from-beginning role-switch replacement corpus constructor.

Builds revised <=10M BabyLM Strict-Small corpora by replacing equal word mass
from the original compact-view reinvest 10M pool with either role-switch packet
text or a structurally matched role-fixed packet control. It does not train a
model. The resulting corpora are legal candidates only if their tokenizers and
models are trained from these revised 10M pools from the beginning.

Removal policy in this first constructor: remove 138 intact 160-word open_subtitles
rows (22,080 words) from the common filler part of the compact-view pool. This is
chosen because research packet train words are exactly 22,080 and open_subtitles is
a large, noisy filler source where previous replacement removed ~97,760 words in
the compact-view construction without destroying the compact mechanism. The policy
is auditable and can be revised before H100 training.
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
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/legal_role_switch_replacement_corpus'
EXPECTED_SRC_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
REMOVE_SOURCE = 'open_subtitles'
REMOVE_WORDS = 22080
PASSES = 10


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def count_words(rows):
    return sum(int(r.get('words', r.get('word_count', len(str(r['text']).split())))) for r in rows)


def normalize_packet_rows(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        text = r['text']
        wc = int(r.get('word_count', len(text.split())))
        if wc != len(text.split()):
            raise RuntimeError(f'packet word mismatch {arm} row {i}: {wc} vs {len(text.split())}')
        out.append({
            'text': text,
            'words': wc,
            'example_id': 9_145_000_000 + (0 if arm == 'role_switch' else 1_000_000) + i,
            'source': f'step145_{arm}_packet',
            'packet_family': r.get('family'),
            'packet_template_id': r.get('template_id'),
            'packet_direction': r.get('direction'),
            'packet_pair_id': r.get('pair_id'),
        })
    return out


def select_removed_rows(base_rows: list[dict[str, Any]], target_words: int):
    # Deterministic: remove the earliest intact rows from REMOVE_SOURCE whose
    # cumulative words exactly hits target. In this corpus, rows are usually 160
    # words; target is chosen as 138*160.
    remove_idx = []
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
        raise RuntimeError(f'could not select exact {target_words} words from source={REMOVE_SOURCE}; got {total}')
    return set(remove_idx), [base_rows[i] for i in remove_idx]


def build_arm(base_rows, removed_idx, packet_rows, arm):
    kept = [r for i, r in enumerate(base_rows) if i not in removed_idx]
    rows = kept + packet_rows
    # Preserve original row order for all kept text and put the replacement block
    # at the end; future runs can test placement separately only after this basic
    # route is validated. The union corpus is still exactly 10M words.
    for new_id, r in enumerate(rows):
        r.setdefault('example_id', new_id)
    words = count_words(rows)
    if words != 10_000_000:
        raise RuntimeError(f'{arm} word total {words} != 10M')
    return rows


def repeat_rows(rows, passes):
    out = []
    for p in range(passes):
        for r in rows:
            rr = dict(r)
            rr['pass_id'] = p
            out.append(rr)
    return out


def source_counts(rows):
    c = Counter()
    wc = Counter()
    for r in rows:
        src = r.get('source','')
        c[src] += 1
        wc[src] += int(r['words'])
    return {'rows': dict(sorted(c.items())), 'words': dict(sorted(wc.items()))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_file(SRC_10M)
    if src_sha != EXPECTED_SRC_SHA:
        raise RuntimeError(f'source 10M sha mismatch {src_sha} != {EXPECTED_SRC_SHA}')
    base_rows = read_jsonl(SRC_10M)
    base_words = count_words(base_rows)
    if base_words != 10_000_000:
        raise RuntimeError(f'base words {base_words} != 10M')

    treat_packets = normalize_packet_rows(read_jsonl(PKT_DIR / 'train_treatment.jsonl'), 'role_switch')
    fixed_packets = normalize_packet_rows(read_jsonl(PKT_DIR / 'train_role_fixed_control.jsonl'), 'role_fixed')
    if count_words(treat_packets) != REMOVE_WORDS or count_words(fixed_packets) != REMOVE_WORDS:
        raise RuntimeError('packet arm word counts do not match removal target')

    removed_idx, removed_rows = select_removed_rows(base_rows, REMOVE_WORDS)
    arms = {
        'role_switch': build_arm(base_rows, removed_idx, treat_packets, 'role_switch'),
        'role_fixed': build_arm(base_rows, removed_idx, fixed_packets, 'role_fixed'),
    }

    files = {'removed_rows': str(OUT / 'removed_open_subtitles_rows.jsonl')}
    write_jsonl(OUT / 'removed_open_subtitles_rows.jsonl', removed_rows)
    for arm, rows in arms.items():
        p10 = OUT / f'{arm}_replacement_10M.jsonl'
        p100 = OUT / f'{arm}_replacement_100M.jsonl'
        write_jsonl(p10, rows)
        write_jsonl(p100, repeat_rows(rows, PASSES))
        files[f'{arm}_10M'] = str(p10)
        files[f'{arm}_100M'] = str(p100)

    manifest = {
        'status': 'LEGAL_ROLE_SWITCH_REPLACEMENT_CORPUS',
        'legal_status': 'candidate corpora are <=10M-word from-beginning replacements; tokenizer and model must be trained on each revised corpus from scratch for legal submission-facing evidence',
        'source_10M': str(SRC_10M),
        'source_10M_sha256': src_sha,
        'source_10M_words': base_words,
        'removal_policy': {
            'source': REMOVE_SOURCE,
            'words': REMOVE_WORDS,
            'rows': len(removed_rows),
            'exact_whole_rows': True,
            'removed_example_id_first_last': [removed_rows[0].get('example_id'), removed_rows[-1].get('example_id')],
        },
        'packet_inputs': {
            'manifest': str(PKT_DIR / 'manifest.json'),
            'role_switch_words': count_words(treat_packets),
            'role_fixed_words': count_words(fixed_packets),
            'role_switch_rows': len(treat_packets),
            'role_fixed_rows': len(fixed_packets),
        },
        'arms': {},
        'files': files,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/legal_role_switch_replacement_corpus.py')),
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
    man_path = OUT / 'manifest.json'
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'LEGAL_REPLACEMENT_CORPUS_DONE',
        'removed_rows': len(removed_rows),
        'removed_words': count_words(removed_rows),
        'role_switch_10M': files['role_switch_10M'],
        'role_fixed_10M': files['role_fixed_10M'],
        'manifest': str(man_path),
    }, indent=2), flush=True)

if __name__ == '__main__':
    main()
