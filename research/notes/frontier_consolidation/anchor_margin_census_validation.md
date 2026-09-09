# anchor margin census validation anchor-margin census: validation and column-distribution finding

Status: scorer validated; full anchor-margin census launched, not yet complete.

## What was repaired and validated

The margin census scorer `scripts/anchor_margin_alpha_census.py`
had three real defects that would have manufactured wrong margins before any
inference:

1. GlobalPIQA was parsed as a 2-candidate task with a fabricated prefix built from
   arbitrary keys (`goal`, `context`, ...). The official task is 4-way parallel and
   2-way nonparallel, with `sentences = prompt + solution_i`,
   `completions = " " + solution_i`, and `label` indexing the gold solution. This
   caused an `IndexError` (labels 2/3 out of range for a 2-element list).
2. Entity candidates were built from a non-existent `story`/`context`/`input` key
   with a fallback concatenation. The official reader uses `input_prefix + option`
   as the candidate sentence and the raw `option` string as the completion.
3. The dynamic-module cache (`HF_MODULES_CACHE`) was set after the first
   `transformers` import, so loading the custom-code model tried to write into the
   shared read-only cache. Cache setup now happens before the `transformers`
   import, and model/tokenizer load lines (accidentally dropped in an earlier edit)
   were restored.

All three were caught and repaired before any inference.

## Scorer validation (CPU, no GPU contention)

Two CPU smokes were run:

- `data/anchor_margin_alpha_census_cpu_smoke` (12 items across
  pattern classes): rescored anchor decisions agree with saved payload decisions
  12/12.
- `data/anchor_margin_targeted_smoke` (13 items covering Entity
  5-way, GlobalPIQA 2/4-way, EWoK 2-way, Supplement 2-way): agreement 13/13, and
  margin signs are correct (anchor-correct items have positive gold-vs-best-other
  margins; anchor-wrong items negative).

The scorer reproduces official-style candidate scoring (sum of completion-token
log-probs; length-normalized for GlobalPIQA). Total across both smokes: 25/25
agreement.

## Column distribution of alpha-sensitive items (build-only manifest)

`data/anchor_margin_alpha_census/anchor_margin_item_manifest.jsonl`
has exactly 6,363 alpha-sensitive items (matching the coherent86 multiarm mechanism reading changed-item count).

Column counts:
- COMPS: 4,336 (68.1%)
- BLiMP: 1,497 (23.5%)
- EWoK: 297 (4.7%)
- Entity: 140 (2.2%)
- Supplement: 88 (1.4%)
- GlobalPIQA: 5 (0.08%)

Pattern counts (activation monotonic 0001/0011/0111 vs damage monotonic
1110/1100/1000 vs nonmonotonic):
- 0001: 792, 0011: 771, 0111: 1553 (activation, 3,116)
- 1110: 819, 1100: 837, 1000: 1575 (damage, 3,231)
- nonmonotonic: 0010: 2, 0100: 1, 0110: 6, 1001: 5, 1011: 1, 1101: 1 (16)

Scientific significance: the alpha-sensitive churn is dominated by COMPS and BLiMP
(both grammaticality/acceptability families), not the declining relation/state
families (EWoK/Entity). The anchor-margin census will show whether monotonic
damage concentrates at low anchor margins (supporting an anchor-confidence gate)
or spreads across the margin distribution (closing the separability premise). The
column skew means COMPS-dominated margin aggregates must be read per-column, not
pooled.

## Running census

Full census over all 6,363 items was in progress, writing to
`data/anchor_margin_alpha_census_full/`. It scores only the
anchor model (chck_82M), 6,363 items, ~few minutes on GPU. The output JSON records
per-group margin summaries (mean/median/quantiles, fraction below thresholds) and
activation-vs-damage separability AUC by signed and absolute margin, plus a
per-column breakdown.

## Retry note (path fix)

The initial census attempt failed before producing a result. A retry was initiated; the failed attempt does not establish model behavior.
