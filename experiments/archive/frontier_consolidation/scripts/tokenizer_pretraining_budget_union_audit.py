#!/usr/bin/env python3
"""research: audit tokenizer-training/pretraining corpus-budget union semantics.

The compliant tokenizer was fit on the compact_view_reinvest 10M pool.  That is
end-to-end compatible for the reinvest endpoint because the tokenizer fitting
text and the first-pass pretraining pool are identical.  A matched clean-Qwen
run using this fixed tokenizer is scientifically useful, but its tokenizer text
is not its own pretraining pool.  This script quantifies the row/text overlap and
word union so later score interpretation separates submission-relevant endpoint
status from fixed-tokenizer causal control status.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any, Dict

USER_ROOT = pathlib.Path('.').resolve()
OUT_DIR = USER_ROOT / 'experiments/archive/frontier_consolidation/data/tokenizer_pretraining_budget_union_audit'
OUT_JSON = OUT_DIR / 'tokenizer_pretraining_budget_union_audit.json'
OUT_MD = (USER_ROOT / 'research/documents/frontier_consolidation/data/tokenizer_pretraining_budget_union_audit/tokenizer_pretraining_budget_union_audit.md')
TOKENIZER_POOL = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
REINVEST_POOL = TOKENIZER_POOL
CLEAN_POOL = USER_ROOT / 'experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl'
META = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json'
COMPLIANT_TOKENIZER = USER_ROOT / 'experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json'
OLD_TOKENIZER = USER_ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M/tokenizer.json'


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_jsonl_stats(path: pathlib.Path) -> Dict[str, Any]:
    rows = 0
    words = 0
    sources: dict[str, int] = {}
    example_ids: set[str] = set()
    text_hashes: dict[str, int] = {}
    row_hashes: dict[str, int] = {}
    first_hashes: list[str] = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            words += int(obj.get('words', len(str(obj.get('text', '')).split())))
            src = str(obj.get('source', ''))
            sources[src] = sources.get(src, 0) + int(obj.get('words', len(str(obj.get('text', '')).split())))
            if 'example_id' in obj:
                example_ids.add(str(obj['example_id']))
            text = str(obj.get('text', ''))
            th = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
            rh = hashlib.sha256(line.rstrip('\n').encode('utf-8', errors='replace')).hexdigest()
            text_hashes[th] = text_hashes.get(th, 0) + 1
            row_hashes[rh] = row_hashes.get(rh, 0) + 1
            if len(first_hashes) < 5:
                first_hashes.append(th)
    return {
        'path': str(path),
        'file_sha256': sha256_file(path),
        'rows': rows,
        'words': words,
        'source_word_counts': sources,
        'unique_example_ids': len(example_ids),
        'unique_text_hashes': len(text_hashes),
        'unique_row_hashes': len(row_hashes),
        'first_text_hashes': first_hashes,
        '_text_hashes': text_hashes,
        '_row_hashes': row_hashes,
    }


def multiset_intersection_count(a: dict[str, int], b: dict[str, int]) -> int:
    return sum(min(v, b.get(k, 0)) for k, v in a.items())


def pair_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    text_common = multiset_intersection_count(a['_text_hashes'], b['_text_hashes'])
    row_common = multiset_intersection_count(a['_row_hashes'], b['_row_hashes'])
    return {
        'text_multiset_common_rows': text_common,
        'row_multiset_common_rows': row_common,
        'text_common_fraction_of_a': text_common / a['rows'] if a['rows'] else None,
        'text_common_fraction_of_b': text_common / b['rows'] if b['rows'] else None,
        'row_common_fraction_of_a': row_common / a['rows'] if a['rows'] else None,
        'row_common_fraction_of_b': row_common / b['rows'] if b['rows'] else None,
    }


def strip_private(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in stats.items() if not k.startswith('_')}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = load_jsonl_stats(TOKENIZER_POOL)
    reinvest = load_jsonl_stats(REINVEST_POOL)
    clean = load_jsonl_stats(CLEAN_POOL)
    tok_reinvest_overlap = pair_overlap(tok, reinvest)
    tok_clean_overlap = pair_overlap(tok, clean)
    reinvest_clean_overlap = pair_overlap(reinvest, clean)

    # Because rows are repacked differently, exact-text overlap underestimates
    # the shared common filler.  Use construction metadata to record the exact
    # designed shared-filler/overlay decomposition as the scientific budget fact.
    metadata = json.loads(META.read_text(encoding='utf-8'))
    designed = {
        'common_filler_words_shared_between_clean_qwen_and_reinvest': metadata['base_split']['common_filler_words'],
        'common_filler_rows': metadata['base_split']['common_filler_rows'],
        'heldout_clean_qwen_words_replaced_by_reinvest_changed_block': metadata['base_split']['heldout_words'],
        'changed_block_budget_words': metadata['changed_block_budget_words'],
        'reinvest_changed_block_words': metadata['families']['compact_reinvest']['pair_words'] + metadata['families']['compact_reinvest']['neutral_cleanqwen_topup_words_inside_changed_block'],
        'reinvest_pair_words': metadata['families']['compact_reinvest']['pair_words'],
        'reinvest_unique_docs_in_changed_block': metadata['pair_summaries']['compact_reinvest']['unique_docs'],
    }
    designed['designed_tokenizer_plus_clean_pretraining_union_words'] = designed['common_filler_words_shared_between_clean_qwen_and_reinvest'] + designed['heldout_clean_qwen_words_replaced_by_reinvest_changed_block'] + designed['changed_block_budget_words']
    designed['union_excess_over_10M_words_for_clean_control'] = designed['designed_tokenizer_plus_clean_pretraining_union_words'] - 10_000_000
    designed['tokenizer_plus_reinvest_pretraining_union_words'] = 10_000_000

    tokenizer_meta = {
        'compliant_tokenizer_json': str(COMPLIANT_TOKENIZER),
        'compliant_tokenizer_sha256': sha256_file(COMPLIANT_TOKENIZER),
        'old_reference_tokenizer_json': str(OLD_TOKENIZER),
        'old_reference_tokenizer_sha256': sha256_file(OLD_TOKENIZER),
    }

    result = {
        'status': 'TOKENIZER_PRETRAINING_BUDGET_UNION_AUDIT',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tokenizer_fitting_pool': strip_private(tok),
        'reinvest_pretraining_pool_10M': strip_private(reinvest),
        'clean_qwen_pretraining_pool_10M': strip_private(clean),
        'overlap_exact_text_rows': {
            'tokenizer_vs_reinvest': tok_reinvest_overlap,
            'tokenizer_vs_clean_qwen': tok_clean_overlap,
            'reinvest_vs_clean_qwen': reinvest_clean_overlap,
        },
        'designed_budget_decomposition': designed,
        'tokenizer_metadata': tokenizer_meta,
        'interpretation': {
            'reinvest': 'submission-relevant end-to-end corpus-budget-valid endpoint if retrain/evaluation complete: tokenizer was fit on the same 10M pool used for pretraining, then repeated for <=10 epochs.',
            'clean_qwen': 'conditional scientific control only: it isolates the effect of the pretraining corpus under the fixed reinvest-trained tokenizer, but tokenizer fitting text plus clean-Qwen pretraining text has a designed union of 10,423,520 words, exceeding the Strict-Small 10M union.',
            'treatment_effect_language': 'Any reinvest-minus-clean result from these two runs is a fixed-tokenizer causal contrast, not an end-to-end compliant comparison between two independently legal systems.'
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = []
    md.append('# research — tokenizer/pretraining corpus-budget union audit\n')
    md.append('The compliant tokenizer was fit on `cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`.  This makes the reinvest retrain the submission-relevant endpoint, because tokenizer fitting text and pretraining text are the same 10M pool.  The clean-Qwen retrain remains useful as a fixed-tokenizer scientific control, but not as an independently valid Strict-Small system.\n')
    md.append('## Quantitative status\n')
    md.append(f"- Tokenizer pool words: {tok['words']:,}; rows: {tok['rows']:,}; SHA256 `{tok['file_sha256']}`.\n")
    md.append(f"- Reinvest pretraining 10M words: {reinvest['words']:,}; exact text overlap with tokenizer pool: {tok_reinvest_overlap['text_multiset_common_rows']:,}/{reinvest['rows']:,} rows.\n")
    md.append(f"- Clean-Qwen pretraining 10M words: {clean['words']:,}; exact text-row overlap with tokenizer pool: {tok_clean_overlap['text_multiset_common_rows']:,}/{clean['rows']:,} rows.\n")
    md.append(f"- Designed shared common filler: {designed['common_filler_words_shared_between_clean_qwen_and_reinvest']:,} words.\n")
    md.append(f"- Reinvest changed block: {designed['changed_block_budget_words']:,} words; clean held-out official rows replaced by that block: {designed['heldout_clean_qwen_words_replaced_by_reinvest_changed_block']:,} words.\n")
    md.append(f"- Tokenizer+reinvest pretraining union: {designed['tokenizer_plus_reinvest_pretraining_union_words']:,} words.\n")
    md.append(f"- Tokenizer+clean-Qwen pretraining designed union: {designed['designed_tokenizer_plus_clean_pretraining_union_words']:,} words, excess {designed['union_excess_over_10M_words_for_clean_control']:,} words over the 10M budget.\n")
    md.append('\n## Interpretation\n')
    md.append('- The reinvest run is the only current end-to-end submission-relevant compliant-tokenizer endpoint.\n')
    md.append('- The clean-Qwen run should be kept and evaluated because it isolates the pretraining-corpus effect under a fixed tokenizer, but its score must not be described as a valid Strict-Small submission artifact.\n')
    md.append('- If reinvest-minus-clean becomes scientifically central, state explicitly that it is fixed-tokenizer causality, separated from end-to-end rule-valid comparison.\n')
    OUT_MD.write_text(''.join(md), encoding='utf-8')
    print(json.dumps({
        'status': result['status'],
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'reinvest_union_words': designed['tokenizer_plus_reinvest_pretraining_union_words'],
        'clean_control_union_words': designed['designed_tokenizer_plus_clean_pretraining_union_words'],
        'clean_control_excess_words': designed['union_excess_over_10M_words_for_clean_control'],
        'clean_exact_text_overlap_rows': tok_clean_overlap['text_multiset_common_rows'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
