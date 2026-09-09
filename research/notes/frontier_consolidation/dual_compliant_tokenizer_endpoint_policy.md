# dual compliant tokenizer endpoint policy — dual compliant tokenizer endpoint policy after row-level `<unk>` exposure

## Research position

The bytealphabet tokenizer repair and retrain priority byte-alphabet repair remains the right resource priority because it is the standard byte-level BPE construction and removes an avoidable official-input coverage flaw without changing the corpus, architecture, seeds, objective, optimizer, sequence length, batch size, or word exposure.

However, the spatial repair route status tokenizer endpoint is not rule-invalid. It was fitted only on the allowed 10M `compact_view_reinvest` pool. Missing byte-level alphabet entries are an empirical tokenizer-construction weakness, not a compliance failure. If the original same-pool reinvest training completes, its official result should be preserved and evaluated in a separate target coordinate. Pristine official results, not the preference for the byte-alphabet repair, should decide between the two legal endpoints.

Tokenizer design is now frozen. The sole structural repair is the standard `ByteLevel.alphabet()` initial alphabet. No vocabulary changes may be derived from benchmark strings or benchmark tokenization analysis.

## Evidence produced in dual compliant tokenizer endpoint policy

### Corrected official-decode row exposure

Script:
- `experiments/archive/frontier_consolidation/scripts/unk_scored_row_exposure.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/unk_scored_row_exposure/unk_scored_row_exposure.json`
- `research/documents/frontier_consolidation/data/unk_scored_row_exposure/unk_scored_row_exposure.md`

The analysis uses the actual strict `sentence_zero_shot/read_files.py` decode functions and the MLM offset-span scoring logic. It repaired an earlier rough scan that did not decode every task family.

Key results under the official zero-shot decode path:

| family | decoded rows | spatial repair route status rows with `<unk>` | spatial repair route status rows with target-span `<unk>` | spatial repair route status `<unk>` tokens | ByteAlpha `<unk>` tokens | token ratio B/A |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 59,875 | 0 | 0 | 0 | 0 | 1.000089 |
| Supplement | 5,218 | 473 | 473 | 946 | 0 | 1.000868 |
| EWoK | 7,618 | 0 | 0 | 0 | 0 | 1.017346 |
| Entity | 6,780 | 0 | 0 | 0 | 0 | 1.000000 |
| COMPS | 91,028 | 0 | 0 | 0 | 0 | 1.000628 |
| GlobalPIQA parallel | 103 | 0 | 0 | 0 | 0 | 1.000880 |
| GlobalPIQA nonparallel | 100 | 0 | 0 | 0 | 0 | 1.001022 |

Supplement localization:
- `qa_congruence_easy`: 64/64 rows have target-span spatial repair route status `<unk>` from the newline; 128 `<unk>` tokens across good/bad candidates.
- `qa_congruence_tricky`: 165/165 rows affected; 330 `<unk>` tokens.
- `turn_taking`: 244/280 rows affected; 488 `<unk>` tokens.
- `hypernym` and `subject_aux_inversion`: 0 affected rows.

For MLM zero-shot, tokens whose offsets overlap the completion span are individually masked and scored. In the Supplement files, the completion equals the full string, so these newline `<unk>` tokens are target-span exposure rather than only context exposure.

SuperGLUE finetuning input exposure is sparse:
- 92,959 train/valid text rows scanned.
- 129 rows with spatial repair route status `<unk>` (0.001388 row fraction).
- 523 spatial repair route status `<unk>` tokens, mostly BoolQ train/valid and QQP valid.
- Byte-alphabet tokenizer has 0 `<unk>` on these strings.
- token ratio B/A is 1.000472.

Reading/AoA input-surface scan:
- 16,430 strings scanned.
- 44 spatial repair route status `<unk>` tokens.
- 0 byte-alphabet `<unk>` tokens.
- token ratio B/A is 1.000431.

### Repair-token training frequency

Script:
- `experiments/archive/frontier_consolidation/scripts/bytealpha_repair_token_frequency.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/bytealpha_repair_token_frequency/bytealpha_repair_token_frequency.json`
- `research/documents/frontier_consolidation/data/bytealpha_repair_token_frequency/bytealpha_repair_token_frequency.md`

This asked whether the byte-alphabet tokens replacing spatial repair route status `<unk>` spans are actually seen in the allowed 10M pretraining pool.

Allowed-pool byte-alphabet tokenization:
- 14,798,756 tokens over 64,740 rows.
- 16,268 unique tokens seen out of 16,384 vocabulary entries.

Zero-shot spatial repair route status-`<unk>` repair tokens:
- 1,892 repair-token occurrences across official zero-shot strings.
- 37 unique repair token strings.
- 946 occurrences have training frequency 0, all the newline byte token `Ċ`.
- 962 occurrences have training frequency <100.
- Top repair tokens: `Ċ` 946 occurrences / 0 training frequency, `ĠSarah` 327 / 2030, `ĠB` 290 / 8508, `ĠNo` 90 / 8981, `ĠYes` 86 / 4402.

SuperGLUE spatial repair route status-`<unk>` repair tokens:
- 1,148 repair-token occurrences.
- 81 unique repair token strings.
- 523 occurrences have training frequency 0.
- 595 occurrences have training frequency <10.
- 1,069 occurrences have training frequency <100.

Interpretation: the byte-alphabet tokenizer removes `<unk>`, but the main zero-shot repair token `Ċ` is not present as a pretraining token in the allowed 10M pool because the pool has no literal newlines inside `text` fields. The spatial repair route status `<unk>` token also has no positive MLM target exposure in the same pool. Therefore the byte-alphabet repair is principled and standard, but not guaranteed to improve Supplement: it changes the representation of unseen newline from special `<unk>` to a normal byte-level token with no direct target history. Official model results must decide.

## Endpoint policy

### Byte-Alphabet Compliant Reinvest Endpoint

Scientific role: resource-priority submission-relevant endpoint under the standard byte-level BPE construction.

Expected run directory:
- `experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022`

Expected tokenizer:
- `experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet`
- SHA256 `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`

Use target name:
- `bytealphatok_reinvest_seed43022`

Dry-run post-delivery driver already passed in bytealphabet tokenizer repair and retrain priority. When delivered, inspect for complete 100M exposure, `hf_model/chck_100M`, full checkpoint ladder, vocab 16,384, and expected tokenizer SHA, then evaluate under the compliant endpoint evaluation policy and driver hard upper-bound continuation policy.

### Same-Pool Compliant Reinvest Endpoint

Scientific role: legal compliant endpoint with known official-input `<unk>` exposure; not invalidated in advance. It can still surpass the byte-alphabet endpoint or live leader and should be evaluated if it completes without taking resources from the byte-alphabet priority run.

Expected run directory:
- `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`

Expected tokenizer:
- `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- SHA256 `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

Use target name:
- `complianttok_reinvest_seed43022`

dual compliant tokenizer endpoint policy dry-run post-delivery command passed and points to isolated output:
- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json`
- collate summary path `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json`

Do not call this endpoint unusable before official evaluation. Its likely vulnerable columns are Supplement, sparse SuperGLUE input rows, and small Reading/AoA surface effects; BLiMP/EWoK/Entity/COMPS/GlobalPIQA have no spatial repair route status `<unk>` under the official zero-shot decode scan.

## Practical next action

After completed training results become available:

1. Evaluate the completed byte-alphabet reinvest endpoint in its isolated target coordinate.
2. Preserve the original same-pool reinvest arm as a legal endpoint and evaluate it separately under `complianttok_reinvest_seed43022`.
3. Compare only pristine official nine-column collations. Row-exposure and token-frequency analyses explain score movement but cannot replace official scoring.
4. The original clean-Qwen control is not selected for relaunch. If a fixed-tokenizer clean control becomes scientifically valuable after the endpoint results, use the byte-alphabet tokenizer and justify the comparison from those outcomes.
