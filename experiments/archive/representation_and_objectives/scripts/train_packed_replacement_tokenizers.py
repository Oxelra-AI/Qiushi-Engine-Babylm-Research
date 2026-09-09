#!/usr/bin/env python3
"""research: train/audit legal 40k byte-BPE tokenizers for packed replacement corpora.

This mirrors research tokenizer training but points at the geometry-preserving
packed corpora from research. It performs no model training.
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
CORPUS_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/packed_role_switch_replacement_corpus'
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/packed_replacement_tokenizers'
TEMPLATE_TOK_JSON = ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json'
ORIGINAL_LEGAL40K = ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
VOCAB_SIZE = 40000
SEQ = 256
ARMS = {
    'role_switch': CORPUS_DIR / 'role_switch_packed_replacement_10M.jsonl',
    'role_fixed': CORPUS_DIR / 'role_fixed_packed_replacement_10M.jsonl',
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_iterator(path: pathlib.Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield str(json.loads(line)['text'])


def iter_rows(path: pathlib.Path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def train_tokenizer(label: str, corpus: pathlib.Path) -> pathlib.Path:
    out_dir = OUT / 'tokenizers' / f'{label}_packed_legal_byte_bpe_40k'
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
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, min_frequency=2, initial_alphabet=ByteLevel.alphabet(), show_progress=True)
    t0 = time.time()
    tok.train_from_iterator(text_iterator(corpus), trainer=trainer)
    tok.save(str(tok_file))
    hf_tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file), unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', mask_token='<mask>', model_max_length=1024)
    hf_tok.save_pretrained(str(out_dir))
    meta = {
        'status': 'trained',
        'label': label,
        'vocab_size_requested': VOCAB_SIZE,
        'vocab_size_actual': tok.get_vocab_size(),
        'training_pool': str(corpus),
        'training_pool_sha256': sha256_file(corpus),
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
    source_words = Counter(); source_tokens = Counter(); source_rows = Counter()
    packed_row_token_lengths: list[int] = []
    for obj in iter_rows(corpus):
        text = str(obj['text'])
        words = int(obj.get('words', len(text.split())))
        ids = tok.encode(text, add_special_tokens=False)
        nt = len(ids)
        rows += 1; total_words += words; total_tokens += nt
        over_seq += int(nt > SEQ)
        src = str(obj.get('source', ''))
        source_rows[src] += 1; source_words[src] += words; source_tokens[src] += nt
        if src.startswith('step146_'):
            packet_rows += 1; packet_words += words; packet_tokens += nt; packed_row_token_lengths.append(nt)
    packed_stats = None
    if packed_row_token_lengths:
        s = sorted(packed_row_token_lengths)
        packed_stats = {
            'n': len(s), 'min': min(s), 'mean': sum(s) / len(s), 'max': max(s),
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
        'packet_tokens_per_word': (packet_tokens / packet_words) if packet_words else None,
        'packed_packet_row_token_stats': packed_stats,
        'source_rows': dict(sorted(source_rows.items())),
        'source_tokens_per_word': {s: source_tokens[s] / source_words[s] for s in sorted(source_words)},
    }


def vocab_overlap(tok_a, tok_b) -> dict[str, Any]:
    va = set(tok_a.get_vocab().keys()); vb = set(tok_b.get_vocab().keys())
    return {
        'a_size': len(va), 'b_size': len(vb), 'shared': len(va & vb),
        'a_only': len(va - vb), 'b_only': len(vb - va),
        'shared_frac_a': len(va & vb) / len(va), 'shared_frac_b': len(va & vb) / len(vb),
        'sample_a_only': sorted(va - vb)[:30], 'sample_b_only': sorted(vb - va)[:30],
    }


def tokenizer_json_delta(path_a: pathlib.Path, path_b: pathlib.Path) -> dict[str, Any]:
    a = json.loads((path_a / 'tokenizer.json').read_text(encoding='utf-8'))
    b = json.loads((path_b / 'tokenizer.json').read_text(encoding='utf-8'))
    va = a['model']['vocab']; vb = b['model']['vocab']
    id_mismatches = [(t, va[t], vb[t]) for t in va.keys() if t in vb and va[t] != vb[t]]
    return {
        'normalizer_equal': a.get('normalizer') == b.get('normalizer'),
        'pre_tokenizer_equal': a.get('pre_tokenizer') == b.get('pre_tokenizer'),
        'post_processor_equal': a.get('post_processor') == b.get('post_processor'),
        'decoder_equal': a.get('decoder') == b.get('decoder'),
        'vocab_dict_equal': va == vb,
        'vocab_set_equal': set(va) == set(vb),
        'merges_equal': a['model'].get('merges', []) == b['model'].get('merges', []),
        'id_mismatch_count_common_tokens': len(id_mismatches),
        'id_mismatch_examples': id_mismatches[:30],
    }


def target_tokenization(tok) -> dict[str, Any]:
    words = ['ball','cat','book','cup','apple','key','Tom','Sam','Ben','Max','open','closed','on','off','full','empty','locked','unlocked','inside','outside','higher','lower']
    return {w: {'plain': tok.encode(w, add_special_tokens=False), 'leading_space': tok.encode(' ' + w, add_special_tokens=False)} for w in words}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok_paths = {}
    for arm, corpus in ARMS.items():
        print(json.dumps({'event': 'train_tokenizer_start', 'arm': arm, 'corpus': str(corpus)}), flush=True)
        tok_paths[arm] = train_tokenizer(arm, corpus)
        print(json.dumps({'event': 'train_tokenizer_done', 'arm': arm, 'path': str(tok_paths[arm])}), flush=True)

    tokenizers = {'original_legal40k': AutoTokenizer.from_pretrained(str(ORIGINAL_LEGAL40K), use_fast=True)}
    tokenizers.update({arm: AutoTokenizer.from_pretrained(str(path), use_fast=True) for arm, path in tok_paths.items()})

    surfaces: dict[str, Any] = {}
    for tlabel, tok in tokenizers.items():
        surfaces[tlabel] = {}
        for arm, corpus in ARMS.items():
            print(json.dumps({'event': 'surface_start', 'tokenizer': tlabel, 'corpus': arm}), flush=True)
            surfaces[tlabel][arm] = token_surface(tok, corpus)
            print(json.dumps({'event': 'surface_done', 'tokenizer': tlabel, 'corpus': arm, 'tpw': surfaces[tlabel][arm]['tokens_per_word'], 'over256': surfaces[tlabel][arm]['rows_over_seq256'], 'packet_tpw': surfaces[tlabel][arm]['packet_tokens_per_word']}), flush=True)

    overlaps = {
        'role_switch_vs_original': vocab_overlap(tokenizers['role_switch'], tokenizers['original_legal40k']),
        'role_fixed_vs_original': vocab_overlap(tokenizers['role_fixed'], tokenizers['original_legal40k']),
        'role_switch_vs_role_fixed': vocab_overlap(tokenizers['role_switch'], tokenizers['role_fixed']),
    }
    tok_delta = tokenizer_json_delta(tok_paths['role_switch'], tok_paths['role_fixed'])
    tokenization = {label: target_tokenization(tok) for label, tok in tokenizers.items()}
    special_ids = {label: {k: getattr(tok, k + '_token_id') for k in ['unk','bos','eos','pad','mask']} for label, tok in tokenizers.items()}

    summary = {
        'status': 'PACKED_REPLACEMENT_TOKENIZERS',
        'legal_status': 'each tokenizer trained only on its own revised packed 10M corpus; no model training performed',
        'vocab_size': VOCAB_SIZE,
        'corpora': {arm: {'path': str(path), 'sha256': sha256_file(path)} for arm, path in ARMS.items()},
        'tokenizer_paths': {arm: str(path) for arm, path in tok_paths.items()},
        'surfaces': surfaces,
        'vocab_overlaps': overlaps,
        'role_switch_vs_role_fixed_tokenizer_json_delta': tok_delta,
        'special_ids': special_ids,
        'target_tokenization': tokenization,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/train_packed_replacement_tokenizers.py')),
    }
    out_path = OUT / 'packed_replacement_tokenizer_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'PACKED_TOKENIZER_DONE', 'summary': str(out_path), 'paths': {arm: str(path) for arm, path in tok_paths.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
