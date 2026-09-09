# seqsafe96 interpretive foundation trainer-exact tokenizer/masking exposure audit

This supersedes the earlier quick exposure note because the trainer uses `add_special_tokens=False`, `max_length=256`, and padding to 256.

Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`; vocab=16384; seq_len=256; fixed WWM mask probability=0.15.

## Aggregate exposure per 10M-word epoch

| arm | rows | words | candidate tokens seen | candidate tok/word | WWM groups | tok/group | token trunc frac | padding tokens/row |
|-----|------|-------|-----------------------|--------------------|------------|-----------|------------------|--------------------|
| treatment FineWeb | 75339 | 10000000 | 14401632 | 1.4402 | 9823206 | 1.4661 | 0.02244 | 64.84 |
| control official | 75339 | 10000000 | 14473097 | 1.4473 | 9823026 | 1.4734 | 0.02240 | 63.89 |

## Deltas (treatment - control)

- candidate tokens seen: -71465 (-0.494% relative)
- candidate tok/word: -0.0071
- token truncation fraction: 0.00004
- WWM groups seen: 180 (0.002% relative)
- expected masked tokens/epoch at p=0.15: -10719.8
- expected masked WWM groups/epoch at p=0.15: 27.0

## Source-block exposure

### Treatment

- `official_identical_tail_after_seqsafe_fineweb_block`: words=6589920, candidate_tok/word=1.4533, WWM_groups/word=0.9733, trunc_frac=0.03312
- `fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies`: words=1753280, candidate_tok/word=1.4627, WWM_groups/word=0.9998, trunc_frac=0.00038
- `qwen_pair_packed`: words=1656800, candidate_tok/word=1.3640, WWM_groups/word=0.9995, trunc_frac=0.00070

### Control

- `official_identical_tail_after_seqsafe_fineweb_block`: words=6589920, candidate_tok/word=1.4533, WWM_groups/word=0.9733, trunc_frac=0.03312
- `official_lengthmatched_to_seqsafe_fineweb`: words=1753280, candidate_tok/word=1.5035, WWM_groups/word=0.9997, trunc_frac=0.00076
- `qwen_pair_packed`: words=1656800, candidate_tok/word=1.3640, WWM_groups/word=0.9995, trunc_frac=0.00070

## Interpretation

Under the exact trainer tokenization, the two seqsafe96 arms are materially token-matched if candidate_tokens_seen_rel_pct is near zero. A positive downstream FineWeb effect cannot be explained by larger treatment token exposure if this delta is near-zero or negative. WWM expected masked-token budget is proportional to candidate tokens, while expected masked-group budget follows WWM group counts and reflects lexical segmentation differences.

Full JSON: `experiments/archive/representation_and_objectives/data/tokenizer_exposure_audit/tokenizer_exposure_audit_trainer_exact.json`
