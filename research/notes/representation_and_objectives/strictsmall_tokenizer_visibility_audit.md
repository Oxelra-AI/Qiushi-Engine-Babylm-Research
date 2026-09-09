# strictsmall tokenizer visibility audit Strict-Small tokenizer visibility audit

JSON: `experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_visibility_audit/strictsmall_tokenizer_visibility_audit.json`

## Why this audit matters

The corrected earlier analysis retrains use a 16k tokenizer trained only on the exact allowed 10M compact-view pool. This audit measures the representation-level change before downstream scores arrive.

## Main comparison

- Full 10M pool mean tokens/word: inherited 1.475552 → strict-small 1.463215 (delta -0.012337).
- Full 10M rows over 256 tokens (no special tokens, matching trainer): inherited 15883 → strict-small 15143 (delta -740).
- Changed block mean tokens/word: inherited 1.462849 → strict-small 1.436011 (delta -0.026839).
- Changed block rows over 256 tokens: inherited 78 → strict-small 54 (delta -24).
- Full source+rewrite visible pairs under actual seq256: inherited 12074/12155 (0.993336) → strict-small 12098/12155 (0.995311); delta 24 pairs.
- Source-visible pairs: inherited 12142 → strict-small 12147 (delta 5).

## Reading

The representation repair changes token density and therefore the exact visible evidence and WWM group structure. It may help or hurt the endpoint independently of compliance. The ongoing H100 retrains are still necessary because this audit is only input geometry, not competence evidence.
