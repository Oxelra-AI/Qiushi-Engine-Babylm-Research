#!/usr/bin/env python3
"""research: train a clean-Qwen-pool tokenizer for diagnostic comparison only.

This does not launch a model retrain and does not change the submission-relevant
reinvest endpoint.  It quantifies how much the running clean-Qwen fixed-tokenizer
control differs from a hypothetical end-to-end clean-Qwen system whose tokenizer
would be fit on its own 10M pool.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import time
from typing import Any, Dict, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

USER_ROOT = pathlib.Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
CLEAN_POOL = USER_ROOT / 'experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl'
REINVEST_POOL = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
REINVEST_TOKENIZER = WORKSPACE / 'data/compliant_tokenizer/tokenizer.json'
TEMPLATE_TOKENIZER = USER_ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json'
OUT = WORKSPACE / 'data/clean_qwen_own_tokenizer_diagnostic'
TOK_DIR = OUT / 'clean_qwen_own_10M_tokenizer'
VOCAB_SIZE = 16384
SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
SEQ_LEN = 256


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def iter_text(path: pathlib.Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get('text', '')
            if text:
                yield str(text)


def count_words(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: dict[str, int] = {}
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            w = int(obj.get('words', len(str(obj.get('text', '')).split())))
            rows += 1
            words += w
            src = str(obj.get('source', ''))
            source_words[src] = source_words.get(src, 0) + w
    return {'rows': rows, 'words': words, 'source_word_counts': source_words, 'sha256': sha256_file(path)}


def train_clean_tokenizer() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    template = Tokenizer.from_file(str(TEMPLATE_TOKENIZER))
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.normalizer = template.normalizer
    tok.pre_tokenizer = template.pre_tokenizer
    tok.post_processor = template.post_processor
    tok.decoder = template.decoder
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, min_frequency=2, show_progress=True)
    t0 = time.time()
    tok.train_from_iterator(iter_text(CLEAN_POOL), trainer=trainer)
    elapsed = time.time() - t0
    tok_json = TOK_DIR / 'tokenizer.json'
    tok.save(str(tok_json))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', mask_token='<mask>'
    )
    hf_tok.save_pretrained(str(TOK_DIR))
    special_ids = {st: tok.token_to_id(st) for st in SPECIAL_TOKENS}
    return {
        'path': str(TOK_DIR),
        'tokenizer_json': str(tok_json),
        'tokenizer_json_sha256': sha256_file(tok_json),
        'vocab_size': tok.get_vocab_size(),
        'special_ids': special_ids,
        'elapsed_sec': round(elapsed, 2),
        'source_pool': str(CLEAN_POOL),
    }


def fast_tok(path: pathlib.Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', mask_token='<mask>'
    )


def summarize(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {'n': 0, 'mean': None, 'std': None, 'p50': None, 'p95': None, 'max': None}
    ys = sorted(xs)
    return {
        'n': len(xs),
        'mean': round(sum(xs) / len(xs), 6),
        'std': round(statistics.pstdev(xs), 6) if len(xs) > 1 else 0.0,
        'p50': round(ys[len(ys)//2], 6),
        'p95': round(ys[min(len(ys)-1, int(0.95 * (len(ys)-1)))], 6),
        'max': round(ys[-1], 6),
    }


def compare_vocab(tok_a: pathlib.Path, tok_b: pathlib.Path) -> dict[str, Any]:
    a = json.loads(tok_a.read_text(encoding='utf-8'))['model']['vocab']
    b = json.loads(tok_b.read_text(encoding='utf-8'))['model']['vocab']
    sa, sb = set(a), set(b)
    return {
        'a_path': str(tok_a),
        'b_path': str(tok_b),
        'a_vocab': len(sa),
        'b_vocab': len(sb),
        'shared_tokens': len(sa & sb),
        'shared_fraction_of_a': len(sa & sb) / len(sa),
        'shared_fraction_of_b': len(sa & sb) / len(sb),
        'only_a_examples': sorted(list(sa - sb))[:50],
        'only_b_examples': sorted(list(sb - sa))[:50],
    }


def compare_pool(pool: pathlib.Path, tok_a: PreTrainedTokenizerFast, tok_b: PreTrainedTokenizerFast, label_a: str, label_b: str) -> dict[str, Any]:
    ratios_a = []
    ratios_b = []
    token_deltas = []
    rows = 0
    words_total = 0
    tokens_a_total = 0
    tokens_b_total = 0
    trunc_a = 0
    trunc_b = 0
    larger_b = 0
    examples: list[dict[str, Any]] = []
    with pool.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get('text', ''))
            words = int(obj.get('words', len(text.split())))
            ids_a = tok_a(text, add_special_tokens=True, truncation=False)['input_ids']
            ids_b = tok_b(text, add_special_tokens=True, truncation=False)['input_ids']
            la, lb = len(ids_a), len(ids_b)
            rows += 1
            words_total += words
            tokens_a_total += la
            tokens_b_total += lb
            ratios_a.append(la / max(1, words))
            ratios_b.append(lb / max(1, words))
            delta = lb - la
            token_deltas.append(delta)
            if la > SEQ_LEN:
                trunc_a += 1
            if lb > SEQ_LEN:
                trunc_b += 1
            if delta > 0:
                larger_b += 1
            if len(examples) < 30 or abs(delta) > min(abs(x.get('delta', 0)) for x in examples):
                rec = {'example_id': obj.get('example_id'), 'source': obj.get('source'), 'words': words, 'len_' + label_a: la, 'len_' + label_b: lb, 'delta': delta, 'text_preview': text[:240]}
                examples.append(rec)
                examples = sorted(examples, key=lambda x: abs(int(x['delta'])), reverse=True)[:30]
    return {
        'pool': str(pool),
        'rows': rows,
        'words': words_total,
        f'total_tokens_{label_a}': tokens_a_total,
        f'total_tokens_{label_b}': tokens_b_total,
        f'tokens_per_word_{label_a}': summarize(ratios_a),
        f'tokens_per_word_{label_b}': summarize(ratios_b),
        f'{label_b}_minus_{label_a}_token_delta': summarize([float(x) for x in token_deltas]),
        f'{label_b}_over_{label_a}_token_ratio_total': tokens_b_total / tokens_a_total if tokens_a_total else None,
        f'seq{SEQ_LEN}_truncated_rows_{label_a}': trunc_a,
        f'seq{SEQ_LEN}_truncated_rows_{label_b}': trunc_b,
        f'rows_where_{label_b}_longer': larger_b,
        f'rows_where_{label_b}_shorter': sum(1 for d in token_deltas if d < 0),
        f'rows_equal_length': sum(1 for d in token_deltas if d == 0),
        'largest_abs_delta_examples': examples,
    }


def main() -> None:
    started = time.time()
    clean_tok_meta = train_clean_tokenizer()
    assert clean_tok_meta['vocab_size'] == VOCAB_SIZE, clean_tok_meta
    assert clean_tok_meta['special_ids'] == {st: i for i, st in enumerate(SPECIAL_TOKENS)}, clean_tok_meta['special_ids']
    clean_tok_json = pathlib.Path(clean_tok_meta['tokenizer_json'])
    reinvest_tok = fast_tok(REINVEST_TOKENIZER)
    clean_tok = fast_tok(clean_tok_json)
    result = {
        'status': 'CLEAN_QWEN_OWN_TOKENIZER_DIAGNOSTIC',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'purpose': 'Diagnostic only: quantify tokenizer-corpus mismatch for the fixed-reinvest-tokenizer clean-Qwen control; no model retrain is launched.',
        'clean_tokenizer': clean_tok_meta,
        'input_pools': {
            'clean_qwen_10M': count_words(CLEAN_POOL),
            'reinvest_10M': count_words(REINVEST_POOL),
        },
        'vocab_overlap_reinvest_tokenizer_vs_clean_own_tokenizer': compare_vocab(REINVEST_TOKENIZER, clean_tok_json),
        'pool_tokenization': {
            'clean_qwen_pool_reinvesttok_vs_cleanowntok': compare_pool(CLEAN_POOL, reinvest_tok, clean_tok, 'reinvesttok', 'cleanowntok'),
            'reinvest_pool_reinvesttok_vs_cleanowntok': compare_pool(REINVEST_POOL, reinvest_tok, clean_tok, 'reinvesttok', 'cleanowntok'),
        },
        'interpretation': 'If the fixed-tokenizer clean-Qwen control moves relative to prior clean-Qwen, this diagnostic helps separate pretraining-corpus effects from tokenizer-corpus mismatch. It does not make the running clean-Qwen control end-to-end compliant because that run still uses the reinvest-trained tokenizer.',
        'elapsed_sec': round(time.time() - started, 2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / 'clean_qwen_own_tokenizer_diagnostic.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/clean_qwen_own_tokenizer_diagnostic/clean_qwen_own_tokenizer_diagnostic.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    vo = result['vocab_overlap_reinvest_tokenizer_vs_clean_own_tokenizer']
    c = result['pool_tokenization']['clean_qwen_pool_reinvesttok_vs_cleanowntok']
    r = result['pool_tokenization']['reinvest_pool_reinvesttok_vs_cleanowntok']
    md = []
    md.append('# research — clean-Qwen own-tokenizer diagnostic\n')
    md.append('Diagnostic tokenizer trained on the clean-Qwen 10M pool to quantify the fixed-tokenizer control interpretation. No model retraining is launched.\n\n')
    md.append(f"- Clean-own tokenizer SHA256: `{clean_tok_meta['tokenizer_json_sha256']}`; vocab {clean_tok_meta['vocab_size']}; specials {clean_tok_meta['special_ids']}.\n")
    md.append(f"- Vocab overlap with reinvest-trained compliant tokenizer: {vo['shared_tokens']:,}/16,384 = {vo['shared_fraction_of_a']:.4f} of reinvest vocab and {vo['shared_fraction_of_b']:.4f} of clean-own vocab.\n")
    md.append(f"- Clean pool total token ratio clean-own/reinvest-tokenizer: {c['cleanowntok_over_reinvesttok_token_ratio_total']:.6f}; seq256 truncation {c['seq256_truncated_rows_reinvesttok']} -> {c['seq256_truncated_rows_cleanowntok']}.\n")
    md.append(f"- Reinvest pool total token ratio clean-own/reinvest-tokenizer: {r['cleanowntok_over_reinvesttok_token_ratio_total']:.6f}; seq256 truncation {r['seq256_truncated_rows_reinvesttok']} -> {r['seq256_truncated_rows_cleanowntok']}.\n")
    md.append('\nInterpretation: the running clean-Qwen model is a fixed-reinvest-tokenizer scientific control only. This diagnostic says how large the tokenizer-corpus mismatch is in token geometry; it cannot replace a model result.\n')
    out_md.write_text(''.join(md), encoding='utf-8')
    print(json.dumps({
        'status': result['status'],
        'out_json': str(out_json),
        'out_md': str(out_md),
        'clean_tokenizer_sha256': clean_tok_meta['tokenizer_json_sha256'],
        'shared_vocab_fraction': vo['shared_fraction_of_a'],
        'clean_pool_token_ratio_cleanown_over_reinvesttok': c['cleanowntok_over_reinvesttok_token_ratio_total'],
        'reinvest_pool_token_ratio_cleanown_over_reinvesttok': r['cleanowntok_over_reinvesttok_token_ratio_total'],
        'elapsed_sec': result['elapsed_sec'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
