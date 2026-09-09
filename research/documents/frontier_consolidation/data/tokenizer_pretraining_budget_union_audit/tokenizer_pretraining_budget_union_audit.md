# compliance control interpretation and eval harness — tokenizer/pretraining corpus-budget union audit
The compliant tokenizer was fit on `cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`.  This makes the reinvest retrain the submission-relevant endpoint, because tokenizer fitting text and pretraining text are the same 10M pool.  The clean-Qwen retrain remains useful as a fixed-tokenizer scientific control, but not as an independently valid Strict-Small system.
## Quantitative status
- Tokenizer pool words: 10,000,000; rows: 64,740; SHA256 `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`.
- Reinvest pretraining 10M words: 10,000,000; exact text overlap with tokenizer pool: 64,740/64,740 rows.
- Clean-Qwen pretraining 10M words: 10,000,000; exact text-row overlap with tokenizer pool: 61,734/64,381 rows.
- Designed shared common filler: 9,576,480 words.
- Reinvest changed block: 423,520 words; clean held-out official rows replaced by that block: 423,520 words.
- Tokenizer+reinvest pretraining union: 10,000,000 words.
- Tokenizer+clean-Qwen pretraining designed union: 10,423,520 words, excess 423,520 words over the 10M budget.

## Interpretation
- The reinvest run is the only current end-to-end submission-relevant compliant-tokenizer endpoint.
- The clean-Qwen run should be kept and evaluated because it isolates the pretraining-corpus effect under a fixed tokenizer, but its score must not be described as a valid Strict-Small submission artifact.
- If reinvest-minus-clean becomes scientifically central, state explicitly that it is fixed-tokenizer causality, separated from end-to-end rule-valid comparison.
