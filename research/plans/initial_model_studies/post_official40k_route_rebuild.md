# earlier analysis — route rebuild after official40k reversal

## Current scientific state

The protected strong route is **baseline16k DeBERTa-v2 8×480 WWM**, not official40k:

- baseline16k DeBERTa-v2 b256 near-complete coordinate: BLiMP 66.76, Supplement 59.88, Entity 22.62, COMPS 52.19, SuperGLUE 68.02, GlobalPIQA 35.635, Reading 7.62, AoA 0.0 degenerate, EWoK full missing/fast-interim 46.18.
- matched BERT b256/b512 controls show DeBERTa gains are architecture/package, not update geometry.
- official40k reduced full-cycle truncation and improved GlobalPIQA mean by +0.97, but Reading is a direct higher-is-better leaderboard column and official40k collapsed Reading 7.62 → 0.89 while also slightly hurting BLiMP, Supplement, and Entity.
- official40k vs baseline16k known six-column delta sum is -6.81. It would require an unsupported ~+6.81 SuperGLUE gain to break even if EWoK stays equal.

Therefore official40k is a useful diagnostic, not the current best Overall candidate. Do not launch expensive official40k follow-up merely from the GlobalPIQA gain. A cheap Reading implementation audit remains useful, because the 40k Reading collapse could reflect tokenizer/word-surprisal alignment rather than pure capability, but current official-score evidence lowers 40k priority.

## Rule grounding

From `notes/official_rules_and_landscape.md` and official Knowledge sources:

- Strict-Small data budget is ≤10M whitespace words; exposure cap is ≤100M words / ≤10 epochs.
- Custom or swapped data are allowed with a datasheet, still within ≤10M words.
- Tokenizer or learned-on-language tools count toward budget.
- Synthetic data is allowed only as a closed system: augmenter's training data counts; external-model distillation revealing tokenizer/weights/hidden states/output distribution is not allowed unless counted.

The first next mechanism should avoid rule ambiguity by using **only the official corpus** and changing the masking distribution. Synthetic relation-state data should be second-stage only, with explicit word-budget replacement and no external model unless rule accounting is solved.

## Central hypothesis after diagnostics

The remaining baseline16k DeBERTa gaps are not primarily token length or scorer artifacts:

- Entity gap is broad across regular/ambiref/move_contents. Actual Entity option strings have identical baseline16k and official40k token lengths, so answer-option compression is not the mechanism.
- GlobalPIQA official length-normalized MLM scoring is not the source of the gap; raw-sum scoring is worse for parallel. The model fails minimal relation, object-property, affordance, spatial, counting, and temporal contrasts.
- official40k shows lower truncation can improve GlobalPIQA parallel by about two items, but it damages Reading and does not fix Entity.

Working hypothesis: **entity identity, relation binding, and state/property update are under-supervised by ordinary WWM.** The desired improvement must add recoverable relation/state pressure while preserving the natural-text surprisal structure that gives baseline16k strong Reading.

## Ranked next mechanisms

### 1. Official-corpus entity–relation-biased WWM (highest priority)

Keep the protected backbone, tokenizer, data, order, word exposure, optimizer schedule, and normal WWM objective. Change only the mask selection distribution on a controlled fraction of examples/batches so that selected whole-word groups are enriched for recoverable entity, relation, and state/property words.

A simple version can use deterministic rules over tokenized official examples, without external parsers:

- entity/referent groups: repeated content nouns in the same 160-word chunk, pronouns, common entity/object nouns, possible proper-name-like words when casing is recoverable;
- spatial/relation words: in, on, under, over, above, below, behind, beside, inside, outside, left, right, near, far, before, after;
- transfer/state/event words: put, move, take, give, get, bring, leave, enter, exit, open, close, contain, hold, break, fall, spill, fill, empty;
- property/material/affordance words: hard, soft, heavy, light, sharp, dull, wet, dry, glass, metal, paper, wood, rubber, cloth, water, fire, heat, cold, transparent, bend, cut, float, sink.

Important design: preserve expected mask count and WWM span structure. Prefer selecting exactly `round(mask_prob * num_valid_groups)` groups per example via weighted sampling without replacement, rather than independent Bernoulli that changes supervision density. Record telemetry: number of relation/entity candidate groups, number selected, selected-token count, and relation-token selected fraction.

Controls:

- baseline WWM: existing baseline16k DeBERTa full run checkpoints can supply the baseline at matched exposure.
- relation-biased WWM: weighted group sampling with low-to-moderate bias.
- matched shuffled/random control: same exact number of groups per example and same weight distribution, but weights randomly permuted among groups or sampled uniformly with the same K. This controls for mask density and exact-k sampling.

First falsification experiment:

- Train from scratch to 20M exposure, not 100M, using same DeBERTa-v2 8×480, baseline16k, official corpus, seed/init/order, batch/update geometry, and `lr_total_steps=2442` to match the first 20M segment of the full baseline trajectory.
- Arms: relation-biased WWM and matched shuffled/random WWM. Existing baseline16k DeBERTa `chck_20M` is the ordinary WWM reference; if implementation changes make this questionable, also train a 20M exact-k WWM control.
- Evaluate at `chck_10M` and `chck_20M` on targeted official-compatible columns: Entity, GlobalPIQA parallel/nonparallel, Reading, BLiMP, Supplement, COMPS. Save per-item predictions for paired wrong→correct analysis.

Signal that justifies a 100M run:

- Entity official macro improves over baseline and matched random, not only one split.
- GlobalPIQA parallel target categories show wrong→correct movement, not only one/two-item noise.
- Reading does not drop materially; baseline16k Reading is an advantage and must be protected.
- BLiMP/Supplement do not lose the near-leader profile.
- Telemetry confirms the intervention actually selected relation/entity groups more often than controls.

Abandon or revise if gains do not beat matched random, if Reading drops sharply, if Entity changes are only pooled/position artifacts, or if BLiMP/Supplement degrade.

### 2. Low-dose relation-state synthetic/data replacement, probe-gated

Use only after route 1 is implemented or if route 1 fails to create relation pressure. Generate 2% and 5% replacement data within the 10M-word corpus, using rule-based templates with explicit state algebra and no external model. Keep coherent and corrupted/control arms with identical word multiset/length where possible.

Do not repeat the previous procedural-state failure blindly. That failure did not teach the target skill under low target-token budget. Any revived synthetic route must first show learned state representations on held-out templates/probes and then transfer to Entity/GlobalPIQA without Reading damage.

### 3. Baseline16k longer sequence only after a window audit

official40k reduced full-cycle truncation but hurt Overall. It does not prove longer context is beneficial. Before training length 512, do a window audit under actual packing: repeated noun/pronoun co-window, relation verb + participant coverage, state/event terms crossing the 256 boundary, and same-document continuation fraction. If 512 adds real relation coverage, run a controlled baseline16k length-512 experiment with accumulation and Reading checks. If not, deprioritize.

### 4. 16k relation-aware ordering/curriculum

Lower priority. Earlier pure ordering gave tradeoffs. Revisit only if relation-biased masking shows useful target categories and we need a low-risk way to present relation-rich examples without changing tokenizer or synthetic data.

## Cheap diagnostics before or alongside construction

1. **official40k Reading audit**: within the official algorithm, compare baseline16k vs official40k Reading word-surprisal extraction: number of scored words, subword count per word, surprisal distribution, correlation between model surprisals, and raw baseline vs baseline+surprisal R². This diagnoses whether the 40k Reading collapse is tokenizer alignment or real surprisal calibration. It does not change current official-score judgment.

2. **Entity scoring interpretation cleanup**: official Entity score is macro over operation-count subsets. Pooled 21.95 and official 22.62 should not be mixed. Since options are scored independently as strings, option order is not part of model input; randomizing options is mostly a scorer sanity check, not a likely route-changing mechanism.

3. **Window audit**: needed only before length-512 training.

## Implementation plan

Build a trainer variant from the trusted baseline16k DeBERTa full-cycle trainer, not from the official40k accumulation fork unless memory requires accumulation. Suggested path:

`training/scripts/babylm_masked_train_relation_wwm.py`

Minimal additions:

- add `mask_mode` choices: `wwm`, `wwm_exact`, `relation_wwm`, `relation_wwm_shuffled`;
- keep `wwm` unchanged for compatibility;
- implement group-level word reconstruction from `input_ids` and `word_group` using tokenizer token strings;
- define relation/entity lexicons in the script and a small stopword list;
- for `wwm_exact`, choose exactly K groups uniformly per example;
- for `relation_wwm`, choose exactly K groups weighted by `1 + alpha * entity + beta * relation + gamma * state_property`;
- for `relation_wwm_shuffled`, compute the same per-group weights but randomly permute weights among valid groups before sampling, preserving weight distribution without tying it to the relation group;
- log per-step telemetry: selected groups/tokens, candidate relation/entity groups, selected relation/entity groups, and effective K;
- write a `masking_telemetry_summary.json` at end.

First smoke:

- DeBERTa-v2 8×480 baseline16k, 80k words, batch 16 or 32, `relation_wwm`, check no NaN, save/load, telemetry nonzero.

First scientific training:

- 20M exposure relation_wwm and relation_wwm_shuffled, same seeds and schedule as baseline16k DeBERTa full run up to 20M.
- Evaluate targeted available official columns at `chck_10M` and `chck_20M`.

This is the next step that best serves the Overall SOTA goal after official40k's corrected mixed/negative result.
