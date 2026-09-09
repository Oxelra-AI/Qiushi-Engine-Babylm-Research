# roberta stratified bridge and early local response: RoBERTa compact-vs-repeat stratified bridge and early local response

## Why this work was done

The compact order result and roberta transfer decision RoBERTa compact-vs-repeat 100M arms and selected evaluation are still in progress. No new training, selected scoring, or SuperGLUE/AoA evaluation was started. The analysis built a bridge between local and downstream evidence: when aggregate RoBERTa trajectories arrive, relate them to the already frozen roberta reader and compact marginal atlas pair strata (tail/source-wide coverage, compact content density, source-absent compact fraction) before deciding any next expensive data arm.

This bridge is interpretive evidence, not a causal intervention. It becomes scientifically important only when read with the selected official-compatible stable-family deltas.

## Frozen event set

Script:

- `scripts/roberta_pair_stratified_response_probe.py`

Event artifacts:

- `data/roberta_pair_stratified_response_probe/frozen_events.jsonl`
- `data/roberta_pair_stratified_response_probe/stratum_manifest.json`
- `data/roberta_pair_stratified_response_probe/stratum_manifest.md`

Initial roberta stratified bridge and early local response event-set invariants below were superseded by the earlier analysis undefined-tail repair recorded later in this note. Use the earlier analysis values (5,470 events / 4,338 pairs / SHA `c172b378873553f209a6bfd9bf53a63e0bde251c1f77690f8abea0ef2d18709a`) for current work.

- Superseded selected events: 5,224
- Superseded unique selected pairs: 4,221
- Superseded frozen events SHA256: `8b32af5730fc0ecb230dec4f04d0706bb1a9225bc756cfcb844e7aa9eccb0d2d`
- Pair file SHA256: `6d0ac85dec1718e5f5663d09123e2f62a23f7cceb0b491b83a34f7bfca59d32d`
- Atlas CSV SHA256: `7fb667e17d5198a2b44d1ea5381bd0fea7fc32161e81103f9ca8be6c3b8f223f`
- Legal tokenizer SHA256: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- Selected by view/category:
  - `compact|function_other`: 1,076
  - `compact|retained_content`: 1,071
  - `compact|source_absent_content`: 934
  - `repeat|function_other`: 1,069
  - `repeat|retained_content`: 1,074

Important repair: the first event-set build used a slightly different content-word heuristic from the roberta reader and compact marginal atlas atlas, causing auxiliary/inflectional words such as `became`, `uses`, `using` to appear as `compact|source_absent_content` even in pairs whose atlas source-absent fraction bin was `zero`. The script now matches roberta reader and compact marginal atlas's `STOPWORDS + len>=4/digit` content definition. The rebuilt manifest reports:

- `compact_source_absent_events_by_source_absent_pair_bin`: `{high_positive: 532, low_positive: 402}`
- `compact_source_absent_events_in_zero_source_absent_pairs`: `0`
- `ok_no_source_absent_events_in_zero_pairs`: `true`

This repair matters because the whole bridge would otherwise misread the source-absent stratum.

## Evaluation-mode mechanical smoke

A tiny CPU smoke used the completed earlier analysis two-row RoBERTa smoke checkpoint as both compact and repeat arms, checkpoint `chck_1M`, on 10 frozen events:

- Output: `data/roberta_pair_stratified_response_probe_eval_smoke/stratified_response.json`
- Expected identity invariant held: all compact-minus-repeat NLL deltas were exactly `0.0`.

This establishes that the local-response evaluator sign and checkpoint-loading path are mechanically sane in a non-active completed checkpoint.

## Early local RoBERTa response on completed checkpoints only

Using only already written `chck_10M`/`chck_20M` directories from the running compact order result and roberta transfer decision arms, the CPU local-response bridge produced:

- Output: `data/roberta_pair_stratified_response_probe_early_cpu/stratified_response.{json,md}`
- Sign: positive `repeat_minus_compact_advantage` means the compact-trained RoBERTa predicts the same masked event better than the repeat-trained RoBERTa.

### chck_10M

- Overall local advantage: `+0.018918` nats over 5,224 events / 7,499 pieces.
- `compact|source_absent_content`: `+0.079156`
- `compact|retained_content`: `+0.058274`
- `compact|function_other`: `-0.061760`
- `repeat|retained_content`: `+0.061794`
- `repeat|function_other`: `-0.115326`
- Compact-content-density contrast on compact source-absent events: `+0.036467` (high-minus-low).
- Source-absent-fraction contrast on compact source-absent events: `+0.001012` (high-positive-minus-low-positive).
- Tail-coverage contrast on compact source-absent events: `-0.012585` (high-minus-low).

### chck_20M

- Overall local advantage: `+0.043986` nats over 5,224 events / 7,499 pieces.
- `compact|source_absent_content`: `+0.127381`
- `compact|retained_content`: `+0.049170`
- `compact|function_other`: `+0.019272`
- `repeat|retained_content`: `+0.010129`
- `repeat|function_other`: `+0.003070`
- Compact-content-density contrast on compact source-absent events: `+0.073293` (high-minus-low).
- Source-absent-fraction contrast on compact source-absent events: `+0.006420` (high-positive-minus-low-positive).
- Tail-coverage contrast on compact source-absent events: `-0.031709` (high-minus-low).

Early read: by 10M/20M, compact-trained RoBERTa already has a local advantage on compact-side source-absent content events. The advantage is more connected to compact content density than to tail-coverage or source-absent-fraction bins at this early stage. However, this is still local denoising evidence, not downstream BabyLM evidence.

## Bridge integration

Script:

- `scripts/roberta_stratified_bridge_integrator.py`

Current output with only early local response and no official selected trajectory:

- `data/roberta_stratified_bridge_integrator/stratified_bridge_integration.{json,md}`
- Reading: only local stratified response is available; do not launch a new expensive arm from local response alone.

The integrator is intended to be rerun after the selected official-compatible trajectory arrives, preferably after `scripts/roberta_transfer_result_reader.py` has produced a ready `reader_report.json`.

## How to read the eventual RoBERTa result

Use the official-compatible selected trajectory first:

- late-band cheap6 without GlobalPIQA
- late-band cheap5 without GlobalPIQA/Reading
- EWoK+Entity
- Supplement, Entity, COMPS

Then use the stratified local response to ask where the compact-trained response is concentrated:

- source-absent compact-content local advantage versus retained/function/repeat controls
- high-minus-low compact content density response
- high-minus-low tail/source-wide coverage response
- high-positive versus low/zero source-absent-fraction response

Scientific readings:

1. Stable downstream gains + source-absent-selective, feature-concentrated local advantage: supports a transferable compact-marginal mechanism; next expensive work should be a paired independent-seed RoBERTa replication, not a factor sweep.
2. Stable downstream gains without matching local feature structure: compact marginal may transfer, but the mechanism is not the measured source-absent/tail/density channel; inspect item/family motion before one clean intervention.
3. Local source-absent advantage without stable downstream gains: repeats the ordered/scrambled local-vs-selected dissociation; do not spend on replication solely from the local channel.
4. Neutral/negative downstream and no matching local structure: bound this RoBERTa coordinate and return to compact-marginal decomposition with one measured ingredient changed at a time.

## Relation to prior evidence

The interpretation is refined without changing the protected endpoint:

- The DeBERTa legal compact-reinvestment data effect remains the strongest corpus evidence.
- Ordered/scrambled DeBERTa showed a local source-absent channel by 40M while selected stable families worsened; local source-absent learning alone is not sufficient.
- Decoder-only GPT2 did not show broad transfer of the same compact marginal.
- The pending RoBERTa pair tests a stock absolute-position bidirectional MLM coordinate; a positive or negative result should not be over-read as a single architectural cause.

No leaderboard submission is permitted or performed.


## earlier analysis update: feature-mismatch repair + repeat arm complete

### Repaired frozen event set (fully audited)

The roberta stratified bridge and early local response audit's 16 `feature_mismatch` cases were `compact_tail_content_coverage` null-in-atlas
vs 0.0-in-event. Root cause: event `features` stored `fnum(atlas)` which coerces atlas `null`
(undefined tail-content region: 53/12,155 pairs) to 0.0, and `assign_bins` silently placed them
in the "low" tail bin.

Repair (faithful, not cosmetic):
- Event `features[feature]` now preserves `None` when the atlas value is undefined.
- `assign_bins` maps a non-finite (null) feature to an explicit `"undefined"` bin instead of "low".

Rebuilt frozen event set:
- events 5,470 / pairs 4,338; SHA256 `c172b378873553f209a6bfd9bf53a63e0bde251c1f77690f8abea0ef2d18709a`;
  manifest `frozen_events_sha256` updated to the same value.
- tail bins: low 1838, mid 1772, high 1615, **undefined 245** (feature value `None`).
- source_absent bins: zero 1249, low_positive 2184, high_positive 2037; **0 compact source_absent_content events in the zero bin** (earlier repair preserved).
- `audit_stratified_events.py` now reports `mechanically_sound: true`, all problem counts 0,
  event SHA matches manifest.

Downstream consistency:
- `after_roberta_bridge_wait.py` references the frozen events by path (no hardcoded SHA), so it
  uses the repaired set automatically.
- `roberta_stratified_bridge_integrator.py` reads the manifest SHA dynamically and compares to
  the response `events_sha256`; both come from the regenerated file, so they match.

### Repaired early local signal is robust (chck_10M/chck_20M)

Rerun on the repaired 5,470-event set (`data/roberta_pair_stratified_response_probe_early_cpu_repaired/`):
- Overall compact-minus-repeat local advantage: 10M +0.01560 (boot p05/p50/p95 +0.00917/+0.01542/+0.02171),
  20M +0.03254 (+0.02432/+0.03260/+0.04062), p>0 = 1.0.
- `compact|source_absent_content`: 10M +0.08037 (+0.06402/+0.08036/+0.09523), 20M +0.10810 (+0.08899/+0.10785/+0.12768), p>0 = 1.0.
- Concentrating feature is content density: `compact_content_fraction` high_minus_low for compact source_absent
  = +0.02605 (10M), +0.06975 (20M). Source-absent-fraction and tail-coverage strata do NOT concentrate the advantage.
- Same qualitative pattern as pre-repair; the undefined-tail fix does not change the finding. This is local
  pseudolikelihood only, not downstream evidence.

### Repeat 100M arm COMPLETE (matched control)

The repeat-control training completed. `training/runs/roberta_repeat_compact_reinvest_100M_seed43022/scientific_metrics.json`:
- RoBERTa 8x480, 30,528,064 params, vocab 16,384, legal spatial repair route status tokenizer `compliant16k_reinvest10M`.
- Exact 100,000,000-word exposure, 2,529 steps, seeds 43/43022/43023, AdamW betas (0.9,0.98) lr 0.001 wd 0.01 warmup 0.06,
  WWM p=0.15, batch 256, seq 256, lr_total_steps 2529.
- loss_first 9.804081 -> loss_last 4.214375. All 10 checkpoints chck_10M..chck_100M valid.
- example_jsonl `cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl` (matched first-N repeat control).

The compact 100M arm was at chck_70M at file-check; selected evaluation and the CPU bridge were still running. When both training arms + selected eval deliver, read the official
late-band stable-family deltas (cheap6_no_GlobalPIQA, cheap5, EWoK+Entity, Supplement, Entity, COMPS) and combine
with the repaired late-band stratified response. Concordance (stable downstream gains + feature-structured local
response) points to independent-seed replication; local-only or volatile-column-carried gains do not.
