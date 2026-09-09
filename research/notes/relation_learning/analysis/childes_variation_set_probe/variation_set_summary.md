# Natural CHILDES variation-set probe

Built **732** surface-held-out adult-utterance pairs from **346** transcripts, yielding **4,222** masked-token records. The available pair counts are low=498, partial=217, high=17.

## What `held out` means

Neither cleaned utterance in any retained pair occurs as an exact contiguous tokenizer-ID sequence in any actual CLEAN, VIEW, or REPEAT 10M training-pool row. This direct screen is necessary because the file called `heldout_cleanqwen_rows.jsonl` is repacked into the CLEAN control and is therefore a row-holdout for arm construction, not an evaluation holdout. Exact surface absence does not prove semantic independence or absence from tokenizer fitting.

## Construction

Transcript boundaries are the `= = = childes/...cha = = =` markers. A candidate joins consecutive non-`CHI` speaker lines within three physical lines; both sides have at least three content-word occurrences, utterance 2 has at least two unique novel content words, at least one unique word overlaps, and the intact sequence is at most 256 model tokens. Content words and Jaccard bins follow the proposal. The literal non-`CHI` rule can include sibling codes such as BRO/SIS; use the stored speaker fields for a canonical-caregiver sensitivity analysis.

The tokenizer has no named CLS/SEP IDs, so BOS `<s>` is used as conceptual CLS and EOS `</s>` as conceptual SEP. Only whole content-word occurrences represented by one tokenizer token are targets. Each retained target has paired intact/replaced records. Controls are selected from another retained pair and always from a different transcript, within 30% of original context token length, preferring zero lexical overlap with the target utterance.

## Counts

| bin | pairs | overlap records/condition | non-overlap records/condition |
|---|---:|---:|---:|
| low | 498 | 441 | 1002 |
| partial | 217 | 227 | 390 |
| high | 17 | 21 | 30 |

## Files

- `variation_set_pairs.jsonl`: pair text, transcript/speaker provenance, overlap metrics, control assignment, and token lengths.
- `probe_records.jsonl`: ready-to-score masked inputs and target metadata.
- `probe_stats.json`: full construction, exposure, balance, target, and control audit.
- `validation_results.json`: independent invariant checks produced by the validator.
- `manual_sample_audit.md`: qualitative review and interpretation cautions from a 30-pair sample.
- `exposure_confounded_*`: the largest ≥100/bin sensitivity instrument possible under the structural rules; it is explicitly not held out and must not support the primary causal claim.

## Feasibility result

The requested ≥100 pairs per bin is **not feasible** under direct surface-held-out screening. The high-overlap bin has only 17 retained pairs. The exposure-confounded sensitivity set has low=500, partial=500, high=444; use it only to measure how strongly training exposure changes the readout.

Score `target_token_id` at `mask_position`, then aggregate paired `intact - replaced` log-probability by model arm, overlap class, and overlap bin. Use pair-clustered or transcript-clustered uncertainty; do not treat the two conditions as independent observations.
