# earlier analysis — Route comparison and first matched-control screen for BabyLM Strict-Small

## Research target now

The live BabyLM 2026 Strict-Small target is the public leaderboard leader `wwm_curriculum_simplification_40k` at Overall 41.8. The INITIAL_MODEL_STUDIES internal coordinate near 40.7028 is below the live target and is not a new mechanism. The next research must create a reusable data-efficient learning principle, not only recover a strong DeBERTa-v2 WWM baseline.

Official implications already refreshed in COMPACT_EXPERIENCE:

- Strict-Small budget: at most 10M training words and at most 100M word exposures under the current challenge surface \cite{choshen2026babylm}.
- Displayed Overall is the mean of nine columns: BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS, (Super)GLUE, GlobalPIQA, Reading, AoA.
- Missing or old-format outputs can be zeroed by current leaderboard processing; Entity and AoA need current accepted output keys.
- New runs must publish the full checkpoint schedule: 1M through 10M in 1M increments, then 20M through 100M in 10M increments.

Evidence carriers:

- `research/notes/compact_experience/babylm2026_surface_refresh.md`
- `research/notes/compact_experience/hf_top_repo_surface.md`
- `experiments/archive/initial_model_studies/data/current_best_internal_coordinate.json`

## What INITIAL_MODEL_STUDIES actually closed, and what remains open

INITIAL_MODEL_STUDIES earlier analysis closed a narrow same-content file-order question. Three 4M files had exactly the same text multiset; only order changed. The ordered arm had small gains on BLiMP, EWoK, Entity, and Reading, but lost Supplement and GlobalPIQA strongly, giving partial-task proxy -0.357 against the mean of two random orders.

This does not answer the content-selection question. The files named `composition_selected_*` and `official_flat_*` in the earlier analysis folder are only 99,840 words / 195 rows and have different source composition. They cannot serve as 4M source-and-length-matched content-selection arms. Therefore the INITIAL_MODEL_STUDIES route left one scientifically meaningful gap:

> In a fixed BabyLM word budget, tokenizer, model, optimizer, and exposure schedule, does raising the density of recoverable compositional and relational prediction constraints per word produce broader reusable language competence than source-and-length-matched text with lower such density?

Evidence carriers:

- `research/notes/compact_experience/initial_model_core_screen_inspection.md`
- `research/notes/initial_model_studies/curriculum_eval.md`
- `research/plans/initial_model_studies/mechanism_comparison_before_construction.md`

## Frontier signals from 2025–2026 Strict-Small

The strongest public 2026 surface gives three useful signals, none of which is yet a causal explanation:

1. **Aligned alternative expressions help enough to reach 41.8.** The leader trains a 34.7M DeBERTa-v2-style MLM on original FineWeb sentences paired with simplified rewrites, 9,999,969 words, 40k SentencePiece BPE, LAMB, 10 epochs, length schedule, and WWM followed by token-level masking.
2. **Simple data ordering is not enough.** Several curriculum-style or story/textbook/synthetic attempts in visible model cards do not surpass a careful GPT-BERT/Muon or DeBERTa-style run. The BabyLM 2025 findings also emphasize objective/architecture choices over plain curriculum \cite{charpentier2025findingsa}.
3. **Objective-level masking matters.** `Mask and You Shall Receive` reports DeBERTa-v2 Strict-Small gains from hard adaptive masking and a decaying mask schedule, while morphology n-hot embeddings have mixed effects and can damage broad columns \cite{edman2025mask}.

The most valuable hypothesis that connects these signals is not “curriculum order,” but:

> Small-data pretraining improves when each word exposure supplies prediction constraints that are (i) recoverable from surrounding context, (ii) structurally compositional, and (iii) stable across different surface expressions of the same proposition.

This principle can be decomposed and tested without mixing all interventions at once.

## Route comparison

### Route A — Same-content order

Status: weak as a main route.

- Already tested at 4M under DeBERTa-v2 WWM in INITIAL_MODEL_STUDIES earlier analysis.
- Negative partial proxy and strong GlobalPIQA/Supplement harm.
- The leader name contains `curriculum`, but its card also changes data source, rewrite pairing, tokenizer, and masking; the name does not isolate file order.

Use only as a control variable: data-loader order must be fixed and recorded so order noise does not masquerade as a content effect.

### Route B — Source-and-length-matched compositional content selection

Status: first direct COMPACT_EXPERIENCE screen.

Mechanism: complete predication, argument structure, clauses, modifiers, relations, and entity chains turn WWM into prediction problems that require combining structure rather than fitting local fragments. If the effect is real, gains should appear most clearly across BLiMP, Supplement, COMPS, and some EWoK; an isolated GlobalPIQA jump is not enough.

Why this route is worth the first screen:

- It is the major INITIAL_MODEL_STUDIES gap that remains untested at the correct 4M scale.
- It uses official in-budget text, so legality and data accounting are straightforward.
- It directly asks whether unit-word experience structure changes reusable competence.
- It can be tested before building a large paired-rewrite pipeline.

### Route C — Recoverable signal-density selection

Status: pair with Route B, but keep scores separate.

Mechanism: the best text is not necessarily syntactically complex; it may contain many learnable masked-token opportunities where context narrows the target without collapsing into duplicated templates. Diversity alone is not sufficient; extreme lexical novelty can starve repetition. The score should balance recoverability, coverage, moderate repetition, and noise suppression.

Why keep it separate from Route B:

- High compositional structure and high recoverable signal may overlap, but they are not the same object.
- S-high improving EWoK/COMPS/GlobalPIQA while C-high improves BLiMP/Supplement would reveal different mechanisms.
- If both score functions select the same rows, the next step must redesign them before interpreting a positive result.

### Route D — Aligned simplification/paraphrase multi-view training

Status: second COMPACT_EXPERIENCE screen, but start data reconnaissance in parallel.

Mechanism: original and simplified/paraphrased sentences provide two expressions of the same latent proposition. The cleanest internal contrast is not paired text versus natural text; it is the same original-rewrite multiset globally shuffled versus locally paired in the same update window, then with relation-aware hard masking or a small representation-consistency loss.

Why not make this the very first training screen:

- The exact leader data construction is not yet reproduced or quality-filtered locally.
- Pair quality, meaning preservation, source distribution, and simplification artifacts must be inspected before training.
- A paired route can easily become a source/style/entropy confound unless its controls are stronger than a simple content-selection screen.

This route is likely essential for surpassing the live leader later, but the first H100 runs should close the INITIAL_MODEL_STUDIES unresolved content-selection mechanism while a paired-rewrite dataset is inspected and filtered.

### Route E — Objective, tokenizer, and optimizer consolidation

Status: attach after a content or paired-data signal appears, not as a mixed first test.

- Hard adaptive masking is well motivated by \cite{edman2025mask}; a structure-weighted variant should emphasize finite predicates, arguments, relations, negation, quantity, comparison, and entity-chain positions while suppressing random names, digits, and subword fragments.
- Tokenizer effects are likely strong because WWM depends on word/subword boundaries. But leaderboard surfaces use 16k, 32k, 40k, and 75k vocabularies; no single vocabulary is established as dominant across data types.
- Muon/AdaMuon and tail averaging are strong consolidation candidates after a data mechanism is found, especially for (Super)GLUE, but they should not obscure the first data comparison.

## Chosen first experiment family: C/S matched content selection

The first COMPACT_EXPERIENCE experiment should materialize a 4M-word matched-content screen on the INITIAL_MODEL_STUDIES DeBERTa-v2 WWM backbone. Training was contingent on execution readiness and had not started in this record.

### Fixed training recipe for the first screen

- Backbone: INITIAL_MODEL_STUDIES protected DeBERTa-v2 WWM recipe, 8 layers × 480 hidden if this is the strongest reproducible INITIAL_MODEL_STUDIES backbone.
- Tokenizer: keep the tokenizer used by the INITIAL_MODEL_STUDIES 40.7028 coordinate for the first screen, so the data mechanism is isolated.
- Objective: standard WWM with the same mask schedule as the INITIAL_MODEL_STUDIES baseline arm; no structural AMLM in the first content-selection screen.
- Optimizer and LR schedule: use the INITIAL_MODEL_STUDIES recipe that produced the current best internal coordinate.
- Budget: 4M unique whitespace-counted words, trained for the same exposure schedule across arms.
- Checkpoints: save 1M, 2M, 3M, and 4M exposure checkpoints for the screen.
- Seeds: at minimum two content/order replicas for the random reference and one initial seed for C-high/S-high; if the first pass shows a potentially important effect, rerun the most informative contrast under a second seed before scaling.

### Data arms

All arms must be built from the same legal candidate pool and match source × length-bin word budgets. They must report whitespace words and tokenizer tokens.

1. **R-a / R-b: stratified random references**
   - Same source-file quotas and length-bin quotas as all treatment arms.
   - Independent draws or independent row orders, used to estimate run and order variation.

2. **C-high: high compositional structure density**
   - Select within each source × length bin for finite predicates, argument cues, subordinate or complement clauses, modifiers, relation words, complete sentence boundaries, and entity-chain cues.
   - Penalize fragments, lists, heavy dialogue fillers, lone names, boilerplate, and near duplicates.

3. **C-control: composition-separated matched control**
   - Same source × length-bin quotas and similar easy surface statistics.
   - Lower compositional score than C-high within each bin.
   - This is the key contrast for whether structure, not source/length, drives gains.

4. **S-high: recoverable signal-density selection**
   - Select for contextual recoverability and useful coverage: moderate frequency targets in informative contexts, coherent sentences, relation-bearing local neighborhoods, controlled repetition, and low boilerplate/noise.
   - Penalize pure lexical novelty, repeated templates, isolated rare terms, number strings, and fragments.

If compute forces the first screen to four training jobs rather than five, keep R-a, R-b, C-high, and S-high; then materialize C-control and train it next if C-high looks promising. If compute permits five jobs, train all five so C-high can be interpreted more cleanly.

### Matching and reporting variables

For every arm, save a data card and JSON summary with:

- whitespace words, rows, tokenizer tokens, tokens per word, and expected optimizer updates;
- source-file word counts and row counts;
- length-bin histogram, not only mean length;
- document/book/program/speaker concentration when recoverable from input path metadata;
- unigram distribution summary, type/token ratio, rare-word fraction, punctuation/digit/case/dialogue-marker rates;
- exact duplicate and near-duplicate estimates;
- C-score and S-score distributions, their correlation, and C-high/S-high row overlap;
- selected row IDs or source offsets so the arm can be rebuilt exactly.

### Evaluation for the 4M screen

Fast screen columns should include at least:

- BLiMP;
- BLiMP Supplement;
- EWoK;
- Entity Tracking with current accepted filtered marker;
- COMPS;
- GlobalPIQA parallel and nonparallel mean;
- Reading if the fast path is already validated.

The screen must keep SuperGLUE and AoA as unsolved until full runs can be evaluated. Report a partial-task proxy only as a research summary, never as official Overall. The result pattern matters more than one scalar:

- C-high should improve BLiMP/Supplement/COMPS more than R and C-control to support compositional structure.
- S-high should improve EWoK/COMPS/GlobalPIQA or broad semantic columns without harming syntax to support recoverable signal density.
- If C-high and S-high have the same selected rows, redesign the scoring before scaling.
- If training loss falls but held-out balanced natural loss and broad columns do not improve, treat the route as easier-fitting rather than reusable knowledge formation.
- If a gain appears only in GlobalPIQA or Reading, rerun before interpreting it; earlier analysis showed large variation there.

### Scale path if the 4M signal is strong

If a content arm shows stable broad gains under matched contrasts:

1. Build a 10M version with the same matching logic and full official checkpoint schedule.
2. Evaluate all nine official columns under the current leaderboard processing.
3. Add one factor at a time:
   - structural hard AMLM on the winning data arm;
   - tokenizer factorial: INITIAL_MODEL_STUDIES tokenizer versus 40k SentencePiece/BPE aligned to WWM;
   - Muon/AdaMuon and tail averaging on the winning data+objective recipe;
   - paired simplification/paraphrase local-binding screen.
4. Preserve every config, data manifest, checkpoint, training log, evaluation file, and score computation under `experiments/archive/compact_experience/training` and mirrored research summaries under `notes/`.

## Parallel paired-rewrite reconnaissance

Before large pair training, inspect or reconstruct a legal paired-rewrite pool:

- Prefer publicly visible leader-linked `go76dof/Fineweb_simplification_pairs` only after reading its card and verifying BabyLM legality, word counts, and train split format.
- Compute pair-quality filters: meaning preservation proxies without external-model leakage into the submission model, length ratios, entity retention, negation/quantity/comparison direction retention, duplicate rate, and source diversity.
- Materialize a later four-arm pair screen on the same backbone:
  - Natural-WWM: matched natural text;
  - Pair-Shuffled: original/rewrite multiset globally shuffled;
  - Pair-Coupled: same multiset, original and rewrite local to the same update window;
  - Pair-Coupled + structural hard AMLM, with no explicit representation loss until the locality effect is known.

The strongest mechanistic contrast in that screen is Pair-Coupled minus Pair-Shuffled because content and token counts are identical.

## Experimental status

The materializers, manifests, score functions and training configurations were preparatory assets. Training had not started in this record; no performance result follows from this design alone.

## Immediate next research work

1. Reconstruct the initial reference's official-corpus loading path and row identity representation used by the DeBERTa-v2 WWM trainer.
2. Build C-score and S-score functions that use only legal in-budget text statistics and transparent heuristics.
3. Materialize 4M R-a, R-b, C-high, C-control, and S-high arms with exact source × length-bin matching and JSON summaries.
4. Validate that each arm has 4M whitespace words, comparable tokenizer tokens, and no accidental source or duplicate shift.
5. Prepare training configs for the fixed DeBERTa-v2 WWM 4M screen; completed training measurements remain a separate requirement.
6. In parallel, inspect the leader-linked simplification-pair dataset and prepare the pair-screen design without mixing it into the first content-selection result.
