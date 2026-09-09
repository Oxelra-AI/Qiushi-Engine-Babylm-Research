# Core-competence route reassessment

## Judgment

The earlier analysis reset is scientifically warranted. The 4M paired BSM trace gives strong negative evidence against continuing the current synthetic-binding family as the main route:

- continuous switch margins stayed essentially zero for coherent, swapped, and token-matched official reference;
- pair-level both-correct stayed 0.000;
- targeted losses fell, so the rows were learnable, but the intended cross-row entity→value consistency was not learned;
- coherent−swapped at 4M moved EWoK by +3.09 but Entity by −0.04 and GlobalPIQA mean by −1.975;
- coherent−token-matched-reference moved EWoK by +3.55 but Entity by −0.15 and GlobalPIQA mean by −4.93.

This is not a promising SOTA path. It creates an EWoK-like displacement without the intended binding mechanism and with a GlobalPIQA tradeoff.

The reset is also supported by the leaderboard structure. Strict-Small rank 2 reaches Overall 41.53 with Entity only 16.59, below the internal best, by high BLiMP/Supplement/COMPS. Rank 3 has Entity near ours and wins mainly through BLiMP/Supplement. Thus high Entity is not necessary for a 41+ Overall score. Entity remains important, but it should no longer dominate the research line by default.

The corrected scoring formula matters:

\[
Overall=\frac{3}{28}\sum_{7\,NLP\ columns}+\frac18(Reading+AoA).
\]

A broad +10 points across NLP columns is worth about +1.07 Overall. This makes multi-column improvement in BLiMP, Supplement, COMPS, EWoK, GlobalPIQA, and SuperGLUE a more realistic route than another isolated Entity mechanism, especially since the protected model already has Supplement and Reading advantages that must be preserved.

## Boundary on 40k tokenization

The reset must not be misunderstood as “try 40k again.” wwm official40k 1m profile already tested official-corpus 40k ByteLevel-BPE as a fixed WWM replacement at 1M. Mean official40k−16k deltas were:

- BLiMP −0.47;
- Supplement −3.20;
- EWoK +1.77;
- Entity −0.06;
- COMPS −0.41;
- Reading eye −8.31;
- Reading self-paced −3.56.

So standalone official40k is a poor next move. If tokenizer returns later, it must be as an interaction with data/curriculum and capacity accounting, not as a direct replacement.

## Weakness in the current reset note

The phrase “core-competence corpus/curriculum” can become too vague. To be useful, the next experiment must produce a controlled scientific answer: does a legal, model-free high-clarity official-text selection improve weighted multi-column ability relative to a source/length-matched official control?

This is different from ordinary source-ratio tuning. The contrast should match source and length bins and vary row quality within those bins, so the result speaks to experience quality rather than simply “more SimpleWiki” or “less dialogue.”

## Recommended next experiment

### Mechanism question

Does training on clearer, more complete, higher-signal official text rows improve grammar/semantic competence columns under the protected DeBERTa-v2 WWM recipe, while preserving GlobalPIQA and Reading?

This is an official-only route, so it avoids custom-data licensing ambiguity and does not depend on external simplification models.

### Arms for a first 4M screen

Use protected DeBERTa-v2 8×480, baseline16k tokenizer, WWM, same optimizer, same word budget, same seed, exact checkpoint names.

1. `official_flat_reference`
   - Ordinary official corpus baseline using the existing protected recipe.

2. `source_length_matched_random`
   - Official rows sampled to match the selected arm by source and word-length bin, but random within each bin.
   - This separates high-clarity selection from source mix and length distribution.

3. `clarity_selected_random_order`
   - Official rows selected within each source/length bin by a transparent model-free clarity score, then shuffled.
   - This tests row-quality selection without order effects.

A fourth arm can be added only after the above three are materialized cleanly:

4. `clarity_selected_easy_to_hard`
   - Same selected rows as arm 3, ordered by clarity/length from simpler to harder using a no-shuffle trainer path.
   - This isolates ordering from selection. Do not mix it into the first contrast unless the trainer can preserve order exactly.

### Suggested model-free clarity score

Compute row features from official text only:

- word count and tokenizer kept-token fraction;
- alphabetic-character fraction;
- punctuation/digit/noise fraction;
- sentence-final punctuation;
- proportion of very short fragments;
- type/token ratio within a safe range;
- dialogue-marker density;
- truncation risk under baseline16k;
- source label and length bin.

The score should prefer complete, grammatical, moderately short, low-noise rows, not merely SimpleWiki rows. Every selected row should be traceable to its source file and row index.

### Required evidence from the screen

Evaluate all 4M endpoints on:

- BLiMP fast;
- Supplement fast;
- EWoK fast;
- Entity fast;
- COMPS;
- GlobalPIQA parallel and nonparallel;
- Reading.

Compute the observed weighted proxy:

\[
\Delta Proxy=\frac{3}{28}\Delta(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA)+\frac18\Delta Reading.
\]

SuperGLUE and AoA are not in the first fast screen, so this is not final Overall. It is a route measurement for whether the selection mechanism is worth scaling.

### What would make the result useful

The screen should change the route only if `clarity_selected_random_order` beats `source_length_matched_random` on the weighted proxy through multiple NLP columns, not only EWoK, and without a substantial GlobalPIQA or Reading loss. A good early pattern would be BLiMP/Supplement/COMPS/EWoK moving together with GlobalPIQA stable. If the selected arm only increases EWoK and damages GlobalPIQA, it repeats the BSM failure in a different form.

## Proposed construction

1. Patch the checkpoint naming bug before new developmental runs: checkpoint directory names must encode exact exposure, e.g. `chck_250k`, `chck_500k`, `chck_1000k`, `chck_1250k`, rather than overwriting `chck_1M`.

2. Build `scripts/materialize_core_competence_screen.py`:
   - load the official Strict-Small corpus revision;
   - compute row features and clarity score;
   - produce the three official-only JSONL arms above with exact word counts, source/length-bin accounting, row hashes, and feature summaries;
   - save `data/core_competence_screen/metadata.json`.

3. Build or adapt a trainer path that can train the three JSONL arms under the protected DeBERTa-v2 WWM recipe with exact word exposure and non-overwriting checkpoints.

4. Build `scripts/eval_weighted_proxy.py` to run the fast columns and compute the weighted proxy with per-column deltas.

The next step should construct and smoke these artifacts. It should not start a large run until the materialized arms are inspected for source/length matching and exact word accounting.
