# clean control trained and representation frontier — Route state: clean control trained, mature vector imminent, representation frontier quantified

## Completed Training Result

The matched clean fixed-tokenizer control **finished training**: `complianttok_cleanqwen_seed43022_80M`, seed43022, vocab 16,384,
34,467,424 params, final MLM loss 2.550567, 80 checkpoints (`chck_1M`..`chck_80M`),
returncode 0, using the spatial repair route status legal tokenizer and the clean-Qwen 100M stream capped at
80M exposure. This is the fixed-tokenizer scientific control (>10M-word union, not a
submission endpoint).

Reinvest reference (same tokenizer/recipe, seed43022) final loss was 2.5526 at 100M; the
clean 80M loss 2.5506 is in the same regime. Loss is not the decision variable — the
official-compatible cheap-column vector is.

The clean 20M/70M/80M cheap-column evaluation remained in progress. After its complete results are available, the comparison uses:
- `python experiments/archive/frontier_consolidation/scripts/compare_legal_treatment_trajectory.py`
- `python experiments/archive/frontier_consolidation/scripts/mature_treatment_subtask_interpreter.py`

Mature reinvest reference (already delivered, s50_t11): 70M mean7 42.6086, 80M mean7
42.9486 (BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading). The reinvest-minus-clean
70M/80M vector is the object that selects the next expensive experiment.

## Legal40k Comparison Status

- The legal40k accumulated-trainer 100M completion summary was stale (`INCOMPLETE_OR_ERROR`, only ~140-191 log records). The seed43022 and seed43122 full official evaluations were still producing collations; no final legal40k Overall was available. `legal40k_endpoint_interpretation.json` still reported WAITING_FOR_LEGAL40K_FULL_COLLATIONS.
- Legal40k uses an accumulated 4x64 microbatch trainer; base 16k used full batch 256. Pre-forward input/mask/RNG identity was verified, but dropout realization and floating-point/kernel order remain different. This establishes objective-weight equivalence, not realized-gradient identity. Legal40k versus legal16k is a package comparison (segmentation, target burden, embedding rows 34.5M->45.8M parameters and trainer), not a pure vocabulary test.

## Training-Only Evidence: representation-allocation frontier

Script `representation_allocation_frontier.py` ->
`data/representation_allocation_frontier/` (JSON+MD+CSV). Uses only the frozen
legal 10M pool (SHA matched) and already-trained legal tokenizers; no eval text, no GPU.
All tok/word measured with truncation disabled (raw segmentation).

Frontier vs legal16k baseline (raw tok/word 1.4669):

| tokenizer | vocab | raw tok/word | over256 | trunc toks | new mass frac | observed-new <50 frac | mass<50 | src-conc low-tail mass | extra emb params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a01_16k | 16384 | 1.4669 | 15143 | 370497 | — | — | 0.00203 | 0.00107 | 0 |
| a01_24k | 24576 | 1.4281 | 13039 | 292086 | 0.0253 | 0.656 | 0.01834 | 0.00234 | 3.9M |
| a01_32k | 32768 | 1.4066 | 11991 | 255416 | 0.0375 | 0.855 | 0.03359 | 0.00340 | 7.9M |
| a01_40k | 40000 | 1.3943 | 11407 | 236482 | 0.0441 | 0.907 | 0.04131 | 0.00404 | 11.3M |
| a01_40k_minfreq25 | 29529 | 1.4139 | 12335 | 267172 | 0.0335 | 0.809 | 0.02864 | 0.00299 | 6.3M |
| a01_40k_minfreq50 | 19609 | 1.4484 | 14096 | 330696 | 0.0124 | 0.031 | 0.00306 | 0.00166 | 1.5M |

### Central quantified tradeoff

40k buys ~5% token compression (1.4669 -> 1.3943 raw tok/word) and cuts over-256 rows
15143 -> 11407, but pays with an **almost entirely low-support tail**: 90.7% of its
new-over-16k observed tokens occur <50 times, and that whole <50 tail is only 4.13% of
token mass. It also adds 11.3M embedding parameters at hidden 480 (34.5M -> 45.8M total),
which is the largest confound in the 40k package.

`minfreq50` is the scientifically clean midpoint: it keeps 185,676 of the 726,632 40k raw
savings (25%) with only a 3.1% low-support new-token fraction and 1.5M extra params. This
means a **support-aware / support-floored inventory is a genuinely distinct route**, not
another size sweep: it separates compression from sparse-embedding estimation and
parameter inflation.

### 40k savings are broadly distributed, not FineWeb-specific

Per-word 40k savings are largest on simple_wiki (0.135), fineweb pairs (0.145), inherited
qwen pairs (0.096), gutenberg (0.074), open_subtitles (0.073); childes lowest (0.025).
Top new 40k tokens are mostly ordinary content words and names (Ġmaintenance, Ġacknowledged,
Ġtiming, Ġembrace, Ġchoices, ĠPAUL, ĠTehran), many source-concentrated. This supports the
verifier reading: the 40k intervention is general segmentation+capacity, not benchmark-cue.

## independent_review synthesis (independent_review01)

- Verifier: safest current statement is NOT "representation/optimization identified" but
  "simple scalar accounts (mean fragmentation, truncation, pooled support/identity) and the
  existing relation prior are poor explanations." The unresolved causal object is the
  interaction of segmentation topology, lexical allocation/support, WWM target burden,
  recipe transfer, embedding capacity, and realized optimization trajectory. The pending
  clean-vs-reinvest vector uses ONE legal tokenizer and cannot separate intrinsic tokenizer
  quality from recipe-mistransfer; legal40k must be read seedwise with clustered family
  uncertainty, not by mean Overall alone.
- Generator: highest-value general mechanism families that are legal and distinct:
  (I) support-aware / compositional embeddings (rare units composed from chars/constituents,
  parameter-matched to 16k) — the ONLY route that cleanly separates compression, inventory
  breadth, and lexical parameter count; (III) equal-word / tokenizer-invariant MLM credit
  (word-mean instead of token-mean, so target weight does not depend on tokenizer
  granularity) — general, benchmark-neutral, and a real optimization mechanism; (IV) more
  iterative contextual computation (12x384 or shared-recurrent depth) if syntax/QA weakness
  persists; (VI/VII) shared-subspace source-view consistency / coverage-consolidation data
  geometry ONLY if reinvest gains knowledge but loses syntax/dialogue.

## Decision map (unchanged spine, sharpened)

- Reinvest broadly beats clean at 70M/80M -> preserve compact-view reinvestment; next
  expensive experiment is legal representation/optimization. The support-floored inventory
  (minfreq50) and equal-word MLM are the two most defensible, distinct candidates; a flat
  40k size sweep is not.
- Reinvest gains knowledge/source-diversity columns but loses syntax/dialogue columns ->
  formulate the coverage-vs-consolidation tradeoff; shared-subspace consistency becomes
  motivated (not full-vector positive MSE, which erases source-only info).
- Reinvest broadly below clean -> the old 42.033 depended on the inherited representation
  coordinate; reopen broader hypotheses; do NOT mine benchmark token classes.
- Do NOT launch relation_only_v1 masking on any "relation" label (Steps56-59: mismatched,
  positive pressure-delta correlation; reinvest is not relation-enriched).

## Discipline

Best fully legal endpoint remains 41.2578 (< 41.8 target). Old 42.033 non-submittable.
No new 100M endpoint, corpus/tokenizer change, or final packaging authorized until the
mature clean vector lands and selects one distinct, general mechanism. One intervention at
a time; do not duplicate active legal40k route while its collations run.
