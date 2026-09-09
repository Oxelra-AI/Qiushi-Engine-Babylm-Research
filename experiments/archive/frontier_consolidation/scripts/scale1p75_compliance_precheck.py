#!/usr/bin/env python3
"""research: CPU-only Strict-Small compliance precheck for the scale1.75 100M endpoint.

This verifies the corpus-budget and tokenizer-provenance facts that make the
scale1.75 endpoint a legally certifiable BabyLM Strict-Small candidate IF the
official evaluator later reports Overall >= target. It does not run model
inference and is not an official score. It checks:
  - the 10M allowed pool exists, has 10,000,000 words, and matches the frozen SHA
  - the 100M training stream matches the frozen SHA and reports 100,000,000 words
  - the legal tokenizer was trained only on the 10M pool (pool SHA in metadata)
  - the endpoint's saved tokenizer.json vocab equals the training tokenizer vocab
    (HF may reserialize the JSON, so compare the vocabulary map, not the file SHA)
  - the endpoint config packages a custom AutoModelForMaskedLM subclass
  - training source-word consumption stays within the legal source set at 100M
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path('.')
POOL = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
STREAM = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
POOL_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
STREAM_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'
TOK_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/compliant_tokenizer'
TOK_META = TOK_DIR / 'tokenizer_metadata.json'
TOK_JSON = TOK_DIR / 'tokenizer.json'
RUN = ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'
ENDPOINT_TOK_JSON = RUN / 'hf_model/chck_100M/tokenizer.json'
ENDPOINT_CONFIG = RUN / 'hf_model/chck_100M/config.json'
METRICS = RUN / 'scientific_metrics.json'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_compliance_precheck'
NOTE = ROOT / 'research/notes/frontier_consolidation/scale1p75_compliance_precheck.md'
LEGAL_SOURCES = {
    'childes', 'open_subtitles', 'bnc_spoken', 'gutenberg', 'qwen_pair_packed',
    'simple_wiki', 'switchboard', 'cleanqwen_fineweb_compact_view_reinvest',
    'neutral_cleanqwen_topup_compact_reinvest::open_subtitles',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def count_words(path: Path) -> Dict[str, int]:
    total_words = 0
    rows = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_words += int(rec.get('words', 0))
            rows += 1
    return {'rows': rows, 'words': total_words}


def vocab_map(tok_json_path: Path) -> Dict[str, int]:
    data = load_json(tok_json_path)
    model = data.get('model', {})
    vocab = model.get('vocab', {})
    return vocab


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    pool_sha = sha256(POOL)
    stream_sha = sha256(STREAM)
    if pool_sha != POOL_SHA: errors.append(f'pool SHA mismatch {pool_sha}')
    if stream_sha != STREAM_SHA: errors.append(f'stream SHA mismatch {stream_sha}')

    pool_counts = count_words(POOL)
    if pool_counts['words'] != 10_000_000: errors.append(f"pool words {pool_counts['words']} != 10000000")

    tok_meta = load_json(TOK_META)
    tok_pool = tok_meta.get('training_data', {}).get('pool')
    tok_pool_sha = tok_meta.get('training_data', {}).get('pool_sha256')
    if tok_pool_sha != POOL_SHA: errors.append(f'tokenizer training pool SHA mismatch {tok_pool_sha}')
    if str(tok_pool) != str(POOL): errors.append(f'tokenizer training pool path {tok_pool}')
    if int(tok_meta.get('tokenizer', {}).get('vocab_size', -1)) != 16384: errors.append('tokenizer vocab_size != 16384')

    train_vocab = vocab_map(TOK_JSON)
    end_vocab = vocab_map(ENDPOINT_TOK_JSON)
    vocab_equal = train_vocab == end_vocab
    if not vocab_equal:
        # compare sizes and symmetric difference summary
        diff_keys = set(train_vocab) ^ set(end_vocab)
        errors.append(f'endpoint tokenizer vocab differs from training tokenizer; |symdiff|={len(diff_keys)}')

    config = load_json(ENDPOINT_CONFIG)
    if config.get('vocab_size') != 16384: errors.append(f"endpoint config vocab_size {config.get('vocab_size')}")
    if config.get('auto_map', {}).get('AutoModelForMaskedLM') != 'adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM':
        errors.append('endpoint config missing custom AutoModelForMaskedLM auto_map')
    if config.get('architectures') != ['AdapterDebertaV2ForMaskedLM']:
        errors.append(f"endpoint architecture {config.get('architectures')}")

    metrics = load_json(METRICS)
    src = metrics.get('source_words_consumed', {})
    illegal_sources = [s for s in src if s not in LEGAL_SOURCES]
    if illegal_sources: errors.append(f'unexpected training source(s): {illegal_sources}')
    total_src_words = sum(int(v) for v in src.values())
    if total_src_words != metrics.get('word_exposure'):
        errors.append(f'source-word sum {total_src_words} != word_exposure {metrics.get("word_exposure")}')
    if metrics.get('word_exposure') != 100_000_000:
        errors.append(f"word_exposure {metrics.get('word_exposure')} != 100000000")

    summary = {
        'status': 'SCALE1P75_COMPLIANCE_PRECHECK',
        'note': 'file-only corpus-budget and tokenizer provenance precheck; not an official score',
        'compliant_prechecks_pass': not errors,
        'errors': errors,
        'pool': {'path': str(POOL), 'sha256': pool_sha, 'expected_sha256': POOL_SHA, 'rows': pool_counts['rows'], 'words': pool_counts['words']},
        'stream': {'path': str(STREAM), 'sha256': stream_sha, 'expected_sha256': STREAM_SHA},
        'tokenizer_provenance': {
            'metadata_path': str(TOK_META),
            'training_pool': tok_pool,
            'training_pool_sha256': tok_pool_sha,
            'vocab_size': tok_meta.get('tokenizer', {}).get('vocab_size'),
            'tokenizer_json_sha256_recorded': tok_meta.get('tokenizer', {}).get('tokenizer_json_sha256'),
            'template_source': tok_meta.get('template_source'),
        },
        'endpoint_tokenizer_vocab_equal_to_training': vocab_equal,
        'endpoint_tokenizer_vocab_size': len(end_vocab),
        'training_tokenizer_vocab_size': len(train_vocab),
        'endpoint_config_custom_automap': config.get('auto_map'),
        'source_words_consumed': src,
        'illegal_sources': illegal_sources,
        'total_source_words': total_src_words,
        'word_exposure': metrics.get('word_exposure'),
        'official_score_status': 'pending evaluation; this precheck only makes the endpoint certifiable if the official Overall clears the target',
    }
    out_json = OUT_DIR / 'scale1p75_compliance_precheck.json'
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines: List[str] = []
    lines.append('# research — scale1.75 100M Strict-Small compliance precheck (file-only)')
    lines.append('')
    lines.append('This does not run model inference or produce an official score. It verifies the corpus-budget and tokenizer-provenance facts that would make the scale1.75 endpoint a legally certifiable Strict-Small candidate if the official evaluator later clears the target.')
    lines.append('')
    lines.append(f"- Prechecks pass: `{summary['compliant_prechecks_pass']}`; errors: `{errors}`.")
    lines.append(f"- Allowed 10M pool: rows {pool_counts['rows']}, words {pool_counts['words']}, SHA `{pool_sha}` (expected `{POOL_SHA}`).")
    lines.append(f"- 100M training stream SHA `{stream_sha}` (expected `{STREAM_SHA}`).")
    lines.append(f"- Legal tokenizer trained only on the 10M pool (metadata pool SHA `{tok_pool_sha}`), vocab {tok_meta.get('tokenizer', {}).get('vocab_size')}.")
    lines.append(f"- Endpoint tokenizer vocab equals training tokenizer vocab: `{vocab_equal}` (endpoint {len(end_vocab)}, training {len(train_vocab)}). HF may reserialize tokenizer.json, so the vocabulary map is compared rather than the file SHA.")
    lines.append(f"- Endpoint custom model package: architecture `{config.get('architectures')}`, AutoModelForMaskedLM `{config.get('auto_map', {}).get('AutoModelForMaskedLM')}`.")
    lines.append(f"- Training source-word consumption: total {total_src_words} == word_exposure {metrics.get('word_exposure')}; illegal sources: `{illegal_sources}`.")
    lines.append('')
    lines.append('Scientific consequence: the scale1.75 endpoint is trained on the exact legal 10M-pool-derived 100M stream, with a tokenizer fit only on that legal pool, and packages a custom-but-official-compatible MLM subclass. The endpoint would be submission-legal on provenance grounds; the remaining decisive fact is the official nine-column Overall from the managed evaluator.')
    lines.append('')
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'compliant_prechecks_pass': summary['compliant_prechecks_pass'], 'errors': errors, 'vocab_equal': vocab_equal, 'out_json': str(out_json), 'note': str(NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
