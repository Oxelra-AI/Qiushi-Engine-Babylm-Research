#!/usr/bin/env python3
"""research: shared legal tokenizer on the common intersection of role-switch/fixed corpora.

Purpose:
  Remove the research representation confound. The role-switch and role-fixed
  in-place packed corpora differ only in 138 packet rows / 22,080 words. This
  script extracts the exact row-aligned common intersection (9,977,920 words),
  trains one legal 40k byte-BPE tokenizer on that intersection, and audits both
  final corpora under the single tokenizer.

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
import time
from collections import Counter
from typing import Any, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path('.')
CORPUS_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus'
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/shared_intersection_tokenizer'
TEMPLATE_TOK_JSON = ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json'
ORIGINAL_LEGAL40K = ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
VOCAB_SIZE = 40000
SEQ = 256
ARMS = {
    'role_switch': CORPUS_DIR / 'role_switch_inplace_packed_replacement_10M.jsonl',
    'role_fixed': CORPUS_DIR / 'role_fixed_inplace_packed_replacement_10M.jsonl',
}
EXPECTED_COMMON_WORDS = 9_977_920
EXPECTED_PACKET_WORDS = 22_080
EXPECTED_ROWS = 64_740
EXPECTED_PACKET_ROWS = 138


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def row_fingerprint(obj: dict[str, Any]) -> str:
    # Common identity must use the actual language string plus the accounting source/words.
    payload = json.dumps({
        'text': obj.get('text'),
        'words': int(obj.get('words', len(str(obj.get('text', '')).split()))),
        'source': obj.get('source'),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def iter_rows(path: pathlib.Path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def count_words(rows: Iterable[dict[str, Any]]) -> int:
    return sum(int(r.get('words', len(str(r.get('text', '')).split()))) for r in rows)


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_common_intersection() -> dict[str, Any]:
    rs = read_rows(ARMS['role_switch'])
    rf = read_rows(ARMS['role_fixed'])
    if len(rs) != EXPECTED_ROWS or len(rf) != EXPECTED_ROWS:
        raise RuntimeError(f'row count mismatch {len(rs)} {len(rf)}')
    common: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    common_hash = hashlib.sha256()
    for i, (a, b) in enumerate(zip(rs, rf)):
        fa = row_fingerprint(a); fb = row_fingerprint(b)
        if fa == fb:
            common.append(dict(a))
            common_hash.update((fa + '\n').encode('ascii'))
        else:
            diff_rows.append({
                'row_index': i,
                'role_switch_source': a.get('source'),
                'role_fixed_source': b.get('source'),
                'role_switch_words': int(a.get('words', len(str(a.get('text','')).split()))),
                'role_fixed_words': int(b.get('words', len(str(b.get('text','')).split()))),
                'role_switch_text': a.get('text'),
                'role_fixed_text': b.get('text'),
            })
    common_words = count_words(common)
    diff_words_rs = sum(d['role_switch_words'] for d in diff_rows)
    diff_words_rf = sum(d['role_fixed_words'] for d in diff_rows)
    if len(diff_rows) != EXPECTED_PACKET_ROWS:
        raise RuntimeError(f'expected {EXPECTED_PACKET_ROWS} diff rows, found {len(diff_rows)}')
    if common_words != EXPECTED_COMMON_WORDS:
        raise RuntimeError(f'common words {common_words} != {EXPECTED_COMMON_WORDS}')
    if diff_words_rs != EXPECTED_PACKET_WORDS or diff_words_rf != EXPECTED_PACKET_WORDS:
        raise RuntimeError(f'diff words {diff_words_rs}/{diff_words_rf} != {EXPECTED_PACKET_WORDS}')
    common_path = OUT / 'common_intersection_9977920w.jsonl'
    diff_path = OUT / 'differing_packet_rows_22080w.jsonl'
    write_jsonl(common_path, common)
    write_jsonl(diff_path, diff_rows)
    return {
        'common_path': str(common_path),
        'diff_path': str(diff_path),
        'common_rows': len(common),
        'common_words': common_words,
        'diff_rows': len(diff_rows),
        'diff_words_role_switch': diff_words_rs,
        'diff_words_role_fixed': diff_words_rf,
        'diff_row_index_first_last': [diff_rows[0]['row_index'], diff_rows[-1]['row_index']],
        'common_row_fingerprint_sha256': common_hash.hexdigest(),
        'common_file_sha256': sha256_file(common_path),
        'diff_file_sha256': sha256_file(diff_path),
    }


def text_iterator(path: pathlib.Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield str(json.loads(line)['text'])


def train_shared_tokenizer(common_path: pathlib.Path) -> pathlib.Path:
    out_dir = OUT / 'tokenizers' / 'shared_intersection_legal_byte_bpe_40k'
    tok_file = out_dir / 'tokenizer.json'
    if tok_file.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    template = Tokenizer.from_file(str(TEMPLATE_TOK_JSON))
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.normalizer = template.normalizer
    tok.pre_tokenizer = template.pre_tokenizer
    tok.post_processor = template.post_processor
    tok.decoder = template.decoder
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(text_iterator(common_path), trainer=trainer)
    tok.save(str(tok_file))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', mask_token='<mask>',
        model_max_length=1024,
    )
    hf_tok.save_pretrained(str(out_dir))
    meta = {
        'status': 'trained_shared_intersection_tokenizer',
        'vocab_size_requested': VOCAB_SIZE,
        'vocab_size_actual': tok.get_vocab_size(),
        'training_pool': str(common_path),
        'training_pool_words': EXPECTED_COMMON_WORDS,
        'training_pool_sha256': sha256_file(common_path),
        'tokenizer_json_sha256': sha256_file(tok_file),
        'special_token_ids': {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        'missing_bytelevel_alphabet': sorted(set(ByteLevel.alphabet()) - set(tok.get_vocab().keys())),
        'elapsed_sec': round(time.time() - t0, 3),
    }
    (out_dir / 'tokenizer_route_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return out_dir


def token_surface(tok, corpus: pathlib.Path) -> dict[str, Any]:
    total_words = total_tokens = rows = over_seq = 0
    packet_words = packet_tokens = packet_rows = 0
    src_words = Counter(); src_tokens = Counter(); src_rows = Counter()
    packet_lengths: list[int] = []
    packet_positions: list[int] = []
    row_token_hash = hashlib.sha256()
    common_row_token_hash = hashlib.sha256()
    packet_row_token_counts: dict[int, int] = {}
    packet_row_id_hashes: dict[int, str] = {}
    for row_i, obj in enumerate(iter_rows(corpus)):
        text = str(obj['text'])
        words = int(obj.get('words', len(text.split())))
        ids = tok.encode(text, add_special_tokens=False)
        nt = len(ids)
        rows += 1; total_words += words; total_tokens += nt; over_seq += int(nt > SEQ)
        row_digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
        row_token_hash.update(f'{row_i}:{row_digest}\n'.encode('ascii'))
        src = str(obj.get('source', ''))
        src_rows[src] += 1; src_words[src] += words; src_tokens[src] += nt
        if src.startswith('step146_'):
            packet_rows += 1; packet_words += words; packet_tokens += nt
            packet_lengths.append(nt); packet_positions.append(row_i)
            packet_row_token_counts[row_i] = nt
            packet_row_id_hashes[row_i] = row_digest
        else:
            common_row_token_hash.update(f'{row_i}:{row_digest}\n'.encode('ascii'))
    packet_stats = None
    if packet_lengths:
        s = sorted(packet_lengths)
        packet_stats = {
            'n': len(s), 'min': min(s), 'mean': sum(s)/len(s), 'max': max(s),
            'p50': s[len(s)//2], 'p90': s[int(0.9*(len(s)-1))], 'over_seq256': sum(x > SEQ for x in s),
        }
    return {
        'rows': rows,
        'words': total_words,
        'tokens': total_tokens,
        'tokens_per_word': total_tokens / total_words,
        'rows_over_seq256': over_seq,
        'packet_rows': packet_rows,
        'packet_words': packet_words,
        'packet_tokens': packet_tokens,
        'packet_tokens_per_word': packet_tokens / packet_words if packet_words else None,
        'packet_row_token_stats': packet_stats,
        'packet_row_index_first_last': [packet_positions[0], packet_positions[-1]] if packet_positions else None,
        'source_rows': dict(sorted(src_rows.items())),
        'source_tokens_per_word': {s: src_tokens[s] / src_words[s] for s in sorted(src_words)},
        'row_token_hash': row_token_hash.hexdigest(),
        'common_row_token_hash': common_row_token_hash.hexdigest(),
        'packet_row_token_counts': packet_row_token_counts,
        'packet_row_id_hashes': packet_row_id_hashes,
    }


def tokenizer_json_delta(path_a: pathlib.Path, path_b: pathlib.Path) -> dict[str, Any]:
    a = json.loads((path_a / 'tokenizer.json').read_text(encoding='utf-8'))
    b = json.loads((path_b / 'tokenizer.json').read_text(encoding='utf-8'))
    va = a['model']['vocab']; vb = b['model']['vocab']
    mismatches = [(t, va[t], vb[t]) for t in va if t in vb and va[t] != vb[t]]
    return {
        'normalizer_equal': a.get('normalizer') == b.get('normalizer'),
        'pre_tokenizer_equal': a.get('pre_tokenizer') == b.get('pre_tokenizer'),
        'post_processor_equal': a.get('post_processor') == b.get('post_processor'),
        'decoder_equal': a.get('decoder') == b.get('decoder'),
        'vocab_dict_equal': va == vb,
        'vocab_set_equal': set(va) == set(vb),
        'merges_equal': a['model'].get('merges', []) == b['model'].get('merges', []),
        'id_mismatch_count_common_tokens': len(mismatches),
        'id_mismatch_examples': mismatches[:30],
    }


def vocab_overlap(tok_a, tok_b) -> dict[str, Any]:
    va = set(tok_a.get_vocab().keys()); vb = set(tok_b.get_vocab().keys())
    return {
        'a_size': len(va), 'b_size': len(vb), 'shared': len(va & vb),
        'a_only': len(va - vb), 'b_only': len(vb - va),
        'shared_frac_a': len(va & vb)/len(va), 'shared_frac_b': len(va & vb)/len(vb),
        'sample_a_only': sorted(va - vb)[:30], 'sample_b_only': sorted(vb - va)[:30],
    }


def first_n_rows_token_hash(tok, corpus: pathlib.Path, n: int = 256) -> dict[str, Any]:
    h = hashlib.sha256(); words = tokens = 0; rows = 0
    packet_seen = False
    for row_i, obj in enumerate(iter_rows(corpus)):
        if row_i >= n:
            break
        ids = tok.encode(str(obj['text']), add_special_tokens=False)
        rows += 1; words += int(obj.get('words', len(str(obj['text']).split()))); tokens += len(ids)
        if str(obj.get('source', '')).startswith('step146_'):
            packet_seen = True
        h.update(hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).digest())
    return {'rows': rows, 'words': words, 'tokens': tokens, 'contains_packet_rows': packet_seen, 'token_hash': h.hexdigest()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intersection = build_common_intersection()
    common_path = pathlib.Path(intersection['common_path'])
    print(json.dumps({'event': 'intersection_done', **intersection}), flush=True)
    tok_path = train_shared_tokenizer(common_path)
    print(json.dumps({'event': 'shared_tokenizer_done', 'path': str(tok_path)}), flush=True)

    shared_tok = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    orig_tok = AutoTokenizer.from_pretrained(str(ORIGINAL_LEGAL40K), use_fast=True)
    surfaces = {
        'shared_intersection_legal40k': {arm: token_surface(shared_tok, path) for arm, path in ARMS.items()},
        'original_legal40k': {arm: token_surface(orig_tok, path) for arm, path in ARMS.items()},
    }
    packet_counts_a = surfaces['shared_intersection_legal40k']['role_switch']['packet_row_token_counts']
    packet_counts_b = surfaces['shared_intersection_legal40k']['role_fixed']['packet_row_token_counts']
    packet_count_diffs = []
    for k in sorted(set(packet_counts_a) | set(packet_counts_b)):
        if packet_counts_a.get(k) != packet_counts_b.get(k):
            packet_count_diffs.append([k, packet_counts_a.get(k), packet_counts_b.get(k)])
    packet_id_hash_diffs = []
    packet_hash_a = surfaces['shared_intersection_legal40k']['role_switch']['packet_row_id_hashes']
    packet_hash_b = surfaces['shared_intersection_legal40k']['role_fixed']['packet_row_id_hashes']
    for k in sorted(set(packet_hash_a) | set(packet_hash_b)):
        if packet_hash_a.get(k) != packet_hash_b.get(k):
            packet_id_hash_diffs.append(k)
    first256 = {arm: first_n_rows_token_hash(shared_tok, path, 256) for arm, path in ARMS.items()}

    # Remove verbose per-packet hashes from the public summary while preserving counts and top-level hashes.
    compact_surfaces = json.loads(json.dumps(surfaces))
    for tok_label in compact_surfaces:
        for arm in compact_surfaces[tok_label]:
            compact_surfaces[tok_label][arm].pop('packet_row_id_hashes', None)
            # Keep only a short sample of packet counts; exact full counts are reconstructible by rerun.
            counts = compact_surfaces[tok_label][arm].get('packet_row_token_counts', {})
            compact_surfaces[tok_label][arm]['packet_row_token_count_sample'] = list(counts.items())[:12]
            compact_surfaces[tok_label][arm].pop('packet_row_token_counts', None)

    summary = {
        'status': 'SHARED_INTERSECTION_TOKENIZER',
        'scientific_reason': 'remove treatment/control representation confound before any role-switch from-scratch screen',
        'legal_status': 'shared tokenizer trained on the 9,977,920-word row-aligned intersection present in both <=10M corpora; no model training in this script',
        'intersection': intersection,
        'shared_tokenizer_path': str(tok_path),
        'shared_tokenizer_metadata': json.loads((tok_path / 'tokenizer_route_metadata.json').read_text(encoding='utf-8')),
        'corpora': {arm: {'path': str(path), 'sha256': sha256_file(path)} for arm, path in ARMS.items()},
        'surfaces': compact_surfaces,
        'shared_tokenizer_internal_self_delta': tokenizer_json_delta(tok_path, tok_path),
        'shared_vs_original_vocab_overlap': vocab_overlap(shared_tok, orig_tok),
        'first256_shared_tokenization': first256,
        'first256_role_switch_vs_role_fixed_identical': first256['role_switch'] == first256['role_fixed'],
        'common_row_token_hash_identical_under_shared_tokenizer': surfaces['shared_intersection_legal40k']['role_switch']['common_row_token_hash'] == surfaces['shared_intersection_legal40k']['role_fixed']['common_row_token_hash'],
        'packet_row_token_count_diff_count_under_shared_tokenizer': len(packet_count_diffs),
        'packet_row_token_count_diff_examples_under_shared_tokenizer': packet_count_diffs[:30],
        'packet_row_id_hash_diff_count_under_shared_tokenizer': len(packet_id_hash_diffs),
        'packet_row_id_hash_diff_examples_under_shared_tokenizer': packet_id_hash_diffs[:30],
        'special_ids': {k: getattr(shared_tok, k + '_token_id') for k in ['unk','bos','eos','pad','mask']},
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/shared_intersection_tokenizer.py')),
    }
    out_path = OUT / 'shared_intersection_tokenizer_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'SHARED_INTERSECTION_TOKENIZER_DONE',
        'summary': str(out_path),
        'shared_tokenizer_path': str(tok_path),
        'first256_identical': summary['first256_role_switch_vs_role_fixed_identical'],
        'common_token_hash_identical': summary['common_row_token_hash_identical_under_shared_tokenizer'],
        'packet_count_diff_count': len(packet_count_diffs),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
