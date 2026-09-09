# lead factorial route reassessment Lead reassessment: compact views as a three-factor object

## Scope and current boundary

This CPU/file-only analysis evaluates the source-wide skeleton proposal against the existing evidence. No model training or official-compatible evaluation was performed.

Pending DeBERTa comparisons:

- scale1.75 seed43022 protected-reference common 70M--100M DeBERTa grid scoring.
- scale1.25 seed43022 common 70M--100M DeBERTa grid scoring.

Scale1.75 seed43122 common-grid scoring remains prepared but unlaunched. Triangle official-compatible scores are also unavailable.

## Research judgment

The useful scientific object is no longer `compact view` as a single intervention. The evidence now supports treating it as a coupled three-factor object:

1. **Source-position spread:** the second view re-exposes content from late and distributed source positions rather than only the beginning of the source.
2. **Recurrent lexical content:** many compact-view tokens are exact lexical/BPE recurrences from the source, and the MLM uses visible exact recurrence strongly.
3. **Fluent compressed syntax / semantic recoding:** the compact view is a grammatical compressed sentence, not just a content list or fragmented extract.

source wide skeleton recurrence integrated established source-wide geometry and exact-copy lift, but it did not establish downstream causality. The raw copied-token NLL lift is a local reconstruction shortcut; it is not sufficient evidence for a general data-efficient learning principle or for H100 training.

The present source wide skeleton recurrence integrated screen arms are therefore not acceptable as the next expensive route. They vary all three factors and additional exposure variables together.

## Exact DeBERTa MLM-loader exposure audit of source wide skeleton recurrence integrated pools

Script:

- `scripts/mlm_loader_exposure_audit.py`

Output:

- `data/mlm_loader_exposure_audit_pools/mlm_loader_exposure_audit.{json,md}`
- `data/mlm_loader_exposure_audit_pools/changed_row_loader_exposure.csv`
- `data/mlm_loader_exposure_audit_pools/pair_visibility_loader_exposure.csv`

The audit reproduces the actual inherited DeBERTa MLM data loader: each JSONL row is tokenized independently with `add_special_tokens=False`, `truncation=True`, `max_length=256`, `padding=max_length`; WWM groups are inferred by tokenizer word-start flags and the nominal mask rate is 0.15.

Pool-level results per 10M pass under the legal spatial repair route status tokenizer:

| variant | raw tokens | active tokens | truncated tokens | truncated rows | changed active source/view tokens | pair full visible |
|---|---:|---:|---:|---:|---:|---:|
| compact | 14,664,519 | 14,294,893 | 369,626 | 15,117 | 359,905 / 247,289 | 99.55% |
| prefix_repeat | 14,633,230 | 14,264,083 | 369,147 | 15,084 | 359,964 / 216,436 | 99.84% |
| content_spread | 14,665,547 | 14,295,757 | 369,790 | 15,122 | 359,870 / 248,204 | 99.51% |
| scored_source_skeleton | 14,672,401 | 14,302,299 | 370,102 | 15,148 | 359,840 / 254,551 | 99.28% |

Key implications:

- The prefix arm has **30,810 fewer active tokens** and **30,869 fewer changed-view WWM-token mass** than compact. This is not a clean fixed-exposure repeat control.
- `content_spread` is closer to compact in active-token exposure (+864 active tokens and +899 view WWM token mass), but it is still a telegraphic source-only control whose linguistic form differs strongly.
- `scored_source_skeleton` adds **7,406 active tokens** and **7,447 changed-view WWM-token mass** relative to compact and has lower pair full visibility; it is both exposure-shifted and text-shifted.
- Legal word count and row position preservation are not enough. A training design must match active tokens, view-token exposure, WWM target mass, truncation, and pair visibility under the exact loader.

This result supersedes the source-wide skeleton dry H100 plan; the proposed training screen is withdrawn.

## Candidate factorial control audit

Script:

- `scripts/factorial_view_candidate_audit.py`

Output:

- `data/factorial_view_candidate_audit/factorial_view_candidate_audit.{json,md}`
- pair files for inspectability under `data/factorial_view_candidate_audit/*_candidate_pairs.jsonl`
- `data/factorial_view_candidate_audit/factorial_candidate_pair_metrics.csv`
- `data/factorial_view_candidate_audit/sourcewide_onegap_examples.jsonl`

Candidate variants at pair level:

- `compact`: original generated compact view.
- `compact_scrambled`: same compact word multiset, deterministic shuffle; fixes recurrent words and view length while breaking coherent order.
- `prefix_fluent`: first-`k` source words where `k` is compact view length; prefix-local contiguous source text.
- `prefix_scrambled`: same prefix word multiset, deterministic shuffle.
- `sourcewide_onegap`: source-only exact-`k` view produced by deleting one contiguous low-value block, leaving at most two intact source spans; intended as a more sentence-preserving source-wide control than source wide skeleton recurrence integrated's telegraphic skeletons.
- `sourcewide_onegap_scrambled`: same one-gap word multiset, deterministic shuffle.
- `best_contiguous_span`: contiguous source span of length `k` selected for overlap/tail content; useful continuum point, not a clean factorial cell.

Pair-level geometry:

| variant | tail coverage | source width | source runs | function fraction | Jaccard with compact | view tok/word |
|---|---:|---:|---:|---:|---:|---:|
| compact | 70.26% | -- | -- | 30.60% | 1.0000 | 1.5439 |
| compact_scrambled | 70.26% | -- | -- | 30.60% | 1.0000 | 1.5439 |
| prefix_fluent | 4.85% | 60.41% | 1.00 | 47.12% | 0.3592 | 1.3337 |
| prefix_scrambled | 4.85% | 60.41% | 1.00 | 47.12% | 0.3592 | 1.3337 |
| sourcewide_onegap | 86.60% | 100.00% | 2.00 | 38.89% | 0.4965 | 1.4758 |
| sourcewide_onegap_scrambled | 86.60% | 100.00% | 2.00 | 38.89% | 0.4965 | 1.4758 |
| best_contiguous_span | 75.54% | 60.41% | 1.00 | 41.60% | 0.4706 | 1.4021 |

Changed-block loader exposure:

| variant | active tokens | trunc tokens | trunc rows | active source/view tokens | pair full visible |
|---|---:|---:|---:|---:|---:|
| compact | 607,273 | 636 | 52 | 359,905 / 247,289 | 99.55% |
| compact_scrambled | 607,273 | 636 | 52 | 359,905 / 247,124 | 99.55% |
| prefix_fluent | 576,463 | 157 | 19 | 359,964 / 216,436 | 99.84% |
| prefix_scrambled | 576,463 | 157 | 19 | 359,964 / 216,352 | 99.84% |
| sourcewide_onegap | 596,804 | 487 | 34 | 359,919 / 236,822 | 99.70% |
| sourcewide_onegap_scrambled | 596,804 | 487 | 34 | 359,919 / 236,699 | 99.70% |
| best_contiguous_span | 586,049 | 311 | 25 | 359,944 / 225,879 | 99.79% |

Scientific reading:

- The fixed-word-multiset scrambled pairs are strong low-cost probes for **coherence/order** because they match recurrent words and active token counts almost exactly within each base view family.
- `sourcewide_onegap` is a better source-wide control than `content_spread` or `scored_source_skeleton`: it uses at most two contiguous source spans, recovers 86.60% tail content, has function-word fraction 38.89% rather than 21.77%/13.73%, and examples are often sentence-fragment-like rather than pure content lists.
- It is still not a final training arm: relative to compact it has 10,469 fewer changed-block active tokens and 10,467 fewer active view tokens, and it changes recurrent lexical content (Jaccard with compact only 0.4965). Relative to prefix it still changes both position spread and which words recur.
- Therefore the immediate value is design repair, not H100 execution.

## Repaired factorial route if pending evidence leaves this as the best explanation

A scientifically useful design should separate three factors with the least new work:

### A. Frozen-model counterfactual first

Before training, use `chck_82M` and a small fixed set of pairs/masks to score constructed views:

- compact vs compact_scrambled: same recurrent compact word multiset, coherent vs broken order.
- sourcewide_onegap vs sourcewide_onegap_scrambled: same source-wide word multiset, coherent-ish two-span order vs broken order.
- prefix_fluent vs prefix_scrambled: same prefix word multiset, coherent prefix vs broken order.
- compact vs sourcewide_onegap after token/exposure matching: natural generated compression vs source-only two-span recurrence.
- sourcewide_onegap vs prefix_fluent: source-position spread at approximate sentence preservation, but not fixed lexical identity.

Score whole-word/BPE strata separately: unique tail copies, unique prefix copies, common subwords/function words, noncopied compact words, entities/numbers/low-frequency content. Use pair-clustered uncertainty. This does not prove downstream learning, but it tells whether the three-factor decomposition is even visible to the existing model beyond raw copy availability.

### B. Exact-loader matched pool repair

If A is informative and the pending DeBERTa/results do not point elsewhere, build training pools only after matching:

- exact 10M legal words,
- active tokens within a very small tolerance under the spatial repair route status tokenizer,
- changed-view active tokens and WWM candidate token/group mass within a very small tolerance,
- truncation rows/tokens and pair full visibility,
- changed-row positions and filler unchanged,
- same tokenizer coordinate.

Possible repair operations:

- choose pair subsets/row packings that equalize active view tokens while preserving pair atomicity;
- add or remove only source-side/filler top-up rows that do not alter changed-view semantics;
- build a `spread_even`/`sourcewide_onegap` hybrid tuned for function-word fraction and token fertility near compact;
- keep scrambled arms paired with their base arms where word multiset and token exposure are exactly shared.

### C. Minimal real-training screen only after A/B and pending evidence

If this route remains strongest, the first H100 screen should be short and explicitly factorized rather than the source wide skeleton recurrence integrated four-arm plan. A minimal two-GPU wave could compare:

1. compact vs compact_scrambled: linguistic order/semantic recoding at fixed compact lexical recurrence.
2. sourcewide_onegap vs sourcewide_onegap_scrambled or sourcewide_onegap vs prefix_fluent: depending on whether frozen probes show order or position-spread is the critical distinction.

Do not train all candidate arms merely because files exist. The screen should read cheap6/cheap5, relation-state, syntax/COMPS, and volatile-column contribution; a cheap7 bump carried by GlobalPIQA/Reading is not enough. It is mechanism-dissection work, not endpoint chasing.

## Relationship to current global research state

Protected assets remain unchanged:

- public submitted scale1.75 `chck_82M`, displayed 41.94;
- local coherent86 `alpha=0.75`, projected Overall(AoA0) 42.1210, HF-published for submission consideration.

The strongest established data mechanism is still compact semantic second views plus reinvested source diversity under DeBERTa MLM, with pair atomicity and same-window visibility. The GPT2 causal negative bounds architecture-general claims. The seed/scale DeBERTa common grid and triangle official-compatible scores are now more important than another speculative route: they can tell whether the current best explanation should emphasize adapter-scale trajectory dynamics, DeBERTa-specific compact-view learning, or another architecture result.

## Immediate next work

1. The source-wide skeleton recurrence screen is not justified by the present evidence.
2. Scientific interpretation remains conditional on the two pending score results and the prepared seed43122 grid, or an explicit finding that the latter is unnecessary. Completed outputs require `scripts/selected_mlm_integrity_check.py`.
3. Triangle official-compatible scores are also required before committing to a skeleton/factorial training route.
4. If the pending results do not redirect the research, the next proposed test is a frozen-model counterfactual NLL probe using the candidate pair files above, with pair-clustered uncertainty and lexical strata, before any new training.
5. Only if this probe supports a meaningful distinction would exposure-matched trainable pools satisfying the exact-loader constraints justify a short two-arm or two-wave H100 screen.

## lead factorial route reassessment addendum after independent_review: target-level factor alignment table

independent_review integration: `data/external/independent_review01_verifier1_integration.md`.

The memo confirms the Lead judgment: the three-factor decomposition is a useful hypothesis space, but the existing candidates are not a clean \(2^3\) factorial. The cleanest current controls are only the fixed-word-multiset ordered/scrambled pairs; the position-spread and natural-style comparisons remain confounded.

Following that memo, I built the strongest CPU-only artifact identified there:

- script: `scripts/target_level_factor_alignment_audit.py`
- output: `data/target_level_factor_alignment_audit/target_level_factor_alignment_audit.{json,md}`
- full records: `data/target_level_factor_alignment_audit/view_word_target_records.csv`
- strata summary: `data/target_level_factor_alignment_audit/target_strata_summary.csv`
- copy-zone summary: `data/target_level_factor_alignment_audit/copy_zone_summary.csv`
- samples: `data/target_level_factor_alignment_audit/target_alignment_samples.jsonl`

It annotates **1,131,956 view-word target records** across the seven candidate views under exact DeBERTa row truncation and WWM grouping. No model was loaded.

Important aggregate results:

| variant | targets | active target tokens | WWM token mass | full visible | counterpart visible | whole-word copy | complete-BPE copy | seen-BPE fraction | contentlike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 161,708 | 244,738 | 247,333 | 99.81% | 83.47% | 83.49% | 70.21% | 81.45% | 65.82% |
| compact_scrambled | 161,708 | 244,739 | 247,334 | 99.81% | 83.47% | 83.49% | 70.20% | 81.45% | 65.82% |
| prefix_fluent | 161,708 | 214,823 | 216,465 | 99.95% | 99.79% | 99.79% | 99.74% | 100.00% | 50.79% |
| prefix_scrambled | 161,708 | 214,822 | 216,464 | 99.95% | 99.79% | 99.79% | 99.74% | 100.00% | 50.79% |
| sourcewide_onegap | 161,708 | 234,687 | 236,850 | 99.86% | 99.81% | 99.83% | 99.68% | 100.00% | 58.64% |
| sourcewide_onegap_scrambled | 161,708 | 234,688 | 236,851 | 99.86% | 99.81% | 99.83% | 99.68% | 100.00% | 58.64% |
| best_contiguous_span | 161,708 | 224,082 | 226,071 | 99.90% | 99.75% | 99.76% | 99.67% | 100.00% | 55.74% |

Design-relevant contrasts:

- `compact` vs `compact_scrambled`: exactly the same target count, only −1 active token and −1 WWM token mass, and identical counterpart visibility to machine precision. This is a very strong order/coherence-at-fixed-lexical-multiset pair.
- `sourcewide_onegap` vs `sourcewide_onegap_scrambled`: exactly the same target count, only −1 active token and −1 WWM token mass, identical counterpart visibility to machine precision. Also strong for order/coherence at fixed source-wide lexical multiset.
- `sourcewide_onegap` vs `prefix_fluent`: +19,864 active target tokens and +20,385 WWM token mass. Position spread remains confounded with target exposure.
- `compact` vs `sourcewide_onegap`: +10,051 active target tokens and +10,483 WWM token mass, with whole-word copy fraction −16.33 points and complete-BPE copy fraction −29.48 points. Natural compact style remains confounded with lexical identity/composition and exposure.

Copy-zone target opportunity illustrates why raw copy-availability NLL is insufficient:

- `compact`: 47,596 tail-only targets / 77,568 active tokens / 78,447 WWM mass; 26,696 absent targets / 37,625 active tokens.
- `prefix_fluent`: 0 tail-only targets; 144,132 prefix-only targets / 195,331 active tokens.
- `sourcewide_onegap`: 64,584 tail-only targets / 100,272 active tokens; only 282 absent targets.
- These strata differ too much for a naive cross-family NLL or training comparison to identify source-position spread alone.

### Sharpened next experiment boundary

The next useful research asset is a **target-aligned frozen-model probe**, not a training pool. It should:

1. use `chck_82M` and possibly `chck_100M` as a paired trajectory check;
2. sample a fixed set of stable lexical target occurrences from each fixed-word-multiset family (`compact/compact_scrambled`, `prefix_fluent/prefix_scrambled`, `sourcewide_onegap/sourcewide_onegap_scrambled`);
3. mask the identical whole-word occurrence in both ordered and scrambled context, scoring source+view and view-only conditions;
4. compute the source-conditioned interaction
   \[
   I_f = [\mathrm{NLL}(V_{scr}|S)-\mathrm{NLL}(V_{ord}|S)] - [\mathrm{NLL}(V_{scr})-\mathrm{NLL}(V_{ord})],
   \]
   so generic fluency preference does not masquerade as source-use;
5. stratify by tail-only, prefix-only, both, absent, function/common subword, entity/number, low-frequency content, source match multiplicity, and distance, using pair-clustered uncertainty;
6. treat any positive result as local source-use structure only, not downstream causal proof.

Only if this frozen probe survives and the pending DeBERTa/results still leave compact-view mechanism as the best question should an exposure-matched training scaffold be repaired.
