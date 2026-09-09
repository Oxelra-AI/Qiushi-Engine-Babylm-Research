# Corpus-derived structural contrast margin for late capability retention

## Why this route is now the main scientific work

The earlier analysis/144 retention probe failed for a structural reason. It stratified target-token NLL by source and word class, but it still asked only whether a token is predictable from its natural context. That statistic can improve while the model loses sensitivity to entity binding, role assignment, relation polarity, and multi-step state updates. The 82M→100M scale1.75 loss has exactly that shape: legal-corpus masked-token NLL improves to 100M, but official EWoK, Entity, GlobalPIQA, and SuperGLUE decline.

The next measurement should therefore score a minimal-pair margin:

\[
  m_{t,i} = \frac{\mathrm{PLL}_t(S_i; M_i) - \mathrm{PLL}_t(\tilde S_i; M_i)}{|M_i|},
\]

where `S_i` is a coherent legal-corpus example, `\tilde S_i` is a controlled perturbation built only from legal-corpus text and generic English word classes, and `M_i` is a focused span set containing the changed tokens plus anchors such as relation words, compared arguments, event connectives, entity mentions, state words, or query candidates. Whole-sentence PLL is secondary because unchanged easy tokens dilute the structural signal.

## Fixed construction families

All examples must come from the legal 10M compact-view-reinvest pool:
`experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`.
The tokenizer must be the legal spatial repair route status tokenizer:
`experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json` or the tokenizer saved inside the relevant legal model checkpoint.

### 1. Comparative and directed relation binding

Target lost capability: EWoK relation direction and comparative variable binding.

Natural frames:
- `X is more/less ADJ than Y`
- `X is ADJ-er than Y`
- `X has more/fewer N than Y`
- `X likes/hates/wants/prefers A more/less than B`
- `X is above/below/before/after/inside/outside Y`
- `from X to Y`, `into/onto/out of`, and similar directed preposition frames.

Perturbations:
- reverse relation word (`more`↔`less`, `above`↔`below`, `before`↔`after`, `from`↔`to` when syntax permits),
- swap compared arguments,
- apply both reversal and swap as a factorial control to cancel name/order bias,
- preserve coarse argument type and approximate token length.

Primary readout: factorial interaction margin
\[
I = \tfrac12[PLL(O)+PLL(AR)] - \tfrac12[PLL(A)+PLL(R)]
\]
when all four variants are grammatical enough; otherwise use coherent-minus-relation-reversal and coherent-minus-argument-swap separately.

### 2. Belief/report/perspective-holder binding

Target lost capability: speaker, reporter, addressee, and belief-holder binding.

Natural frames:
- `X told/asked/warned/informed/lied to Y that P`,
- `X said to Y, P`,
- `"P," X told Y`,
- `X thinks/believes/knows that P`,
- adjacent dialogue turns with repeated speaker/addressee names.

Perturbations:
- reporter↔addressee swap,
- downstream name/pronoun replacement by the competing participant,
- speaker attribution exchange between adjacent dialogue turns,
- matrix subject↔embedded perspective-holder swap.

Focused spans: both participant mentions, report/belief verb, downstream mention, and complement anchor. Balance by source because CHILDES/OpenSubtitles/Switchboard/Gutenberg provide different discourse registers.

### 3. Multi-operation entity-state tracking

Target lost capability: Entity Tracking high-operation state updates, especially 3–5 update contexts.

Natural frames:
- 2–5 sentence windows with recurring entities and update verbs: `put/move/take/remove/bring/leave/give/get/keep/fill/empty/add/pour/open/close/hide/find/lose`,
- locations or containers introduced with `in/into/on/from/to`,
- repeated object/location terms or repeated named participants.

Perturbations:
- stale state: replace final location/state with an earlier state,
- cross-entity state transplant: assign A the final state of B,
- location swap,
- update argument swap,
- adjacent update permutation only when the two updates affect distinct entities,
- final reference swap.

Preferred scoring form: append a neutral final query made only from corpus-attested words, e.g. `Now, X is in ___`, `After this, X is ___`, or `X now contains ___`, and compare the coherent candidate against stale/cross-entity alternatives. If natural windows are too noisy, use controlled micro-discourses assembled from corpus-attested verbs, nouns, colors, containers, and locations, with balanced permutations so each entity and state appears in each role.

Primary summaries:
- high-depth state margin: update-depth ≥3,
- depth contrast: high-depth margin minus zero/one-update margin.

### 4. Temporal/procedure ordering

Target lost capability: GlobalPIQA-style physical procedure order and preconditions.

Natural frames:
- numbered steps,
- `first/then/next/finally`,
- `before/after/once/until/when`,
- repeated concrete object heads,
- action verbs such as `open/remove/wash/dry/cut/attach/mix/pour/heat/wait/put/take`.

Perturbations:
- adjacent-step reversal,
- `before`↔`after` while holding clause order fixed,
- clause-order reversal with connective adjusted separately,
- premature use before preparation,
- source/goal exchange in `remove from`, `put into`, `pour into`,
- instrument/theme exchange among compatible nouns.

Control for generic event-frequency bias by subtracting an `and` order margin:
\[
m_{temp} = [PLL(a\ then\ b)-PLL(b\ then\ a)] - [PLL(a\ and\ b)-PLL(b\ and\ a)].
\]

### 5. Polarity–relation composition

Target lost capability: NPI/polarity subsets and relation-polarity interaction, secondary weight.

Natural frames:
- `not/never/no/without/cannot` with `any/ever/at all/yet`,
- negated relation/comparative clauses.

Perturbations:
- polarity removal/addition,
- relation reversal under fixed polarity,
- polarity + relation reversal factorial interaction.

Keep this family low weight relative to state/procedure/relation binding because the broad late loss is not only syntactic polarity.

## CPU feasibility evidence

`scripts/contrastive_margin_feasibility_scout.py` scanned 64,739 non-neutral legal rows without official benchmark text or labels. Heuristic frame yields are ample enough to justify a real builder:

| family | candidate events | unique rows | strongest sources |
|---|---:|---:|---|
| belief_report_role_binding | 57,683 | 31,650 | Gutenberg, OpenSubtitles, CHILDES, qwen pairs, BNC |
| comparative_relation | 6,559 | 5,388 | Gutenberg, qwen pairs, OpenSubtitles, FineWeb, SimpleWiki |
| multi_operation_entity_state | 28,349 | 28,349 | CHILDES, Gutenberg, OpenSubtitles, qwen pairs |
| polarity_relation_composition | 35,703 | 23,991 | Gutenberg, qwen pairs, OpenSubtitles, CHILDES |
| spatial_directional_relation | 172,037 | 57,601 | qwen pairs, Gutenberg, OpenSubtitles, CHILDES, SimpleWiki |
| temporal_procedure_order | 64,037 | 36,790 | qwen pairs, Gutenberg, CHILDES, SimpleWiki, OpenSubtitles |

The scout is intentionally loose. The real builder must reject noisy cases and produce fewer but cleaner pairs.

## Fixed scoring and aggregation plan

For every checkpoint, compute per-pair focused PLL margins with `trust_remote_code=True` model loading and isolated writable HF caches. Reuse the batching/model-loading pattern from `scripts/retention_vector_eval.py`, but replace target-token NLL with coherent-vs-perturbed margin.

Primary scalar, frozen before reading scores:

\[
R_t = 0.30 R_{state,d\ge3} + 0.25 R_{temporal} + 0.20 R_{directed/role} + 0.15 R_{belief} + 0.10 R_{comparative} - 0.25 L_t
\]

where each `R` is a family-balanced robust margin (median-of-means or 20% trimmed mean over source×family×head cells) and `L_t` is the matching local/surface margin average: lexical replacements, low-depth state, grammar/easy local corruption, and null event swaps. The point is structural binding relative to local surface improvement, not raw corruption detection.

Report alongside the scalar:
- balanced hard accuracy `Pr(margin > 0)` per family,
- high-depth minus low-depth state margin,
- temporal direction margin after `and` subtraction,
- family-level minimum or 25th percentile to expose a narrow collapse.

Do not choose among these after seeing checkpoint scores. If two summaries are implemented (robust margin and balanced hard accuracy), both must remain visible and neither may be swapped in as primary after the fact.

## Test order

1. Build the pair set and freeze it with hashes, examples, family counts, source balance, token-length statistics, and deterministic seed. Save under `data/structural_contrast_probe/`.
2. Run a small CPU/GPU smoke on two checkpoints only (e.g. scale1.75 chck_82M and chck_100M) to verify scoring correctness and runtime, but do not change transformations or weights from the smoke result.
3. Score scale1.75 late checkpoints `chck_77M..chck_83M,chck_100M` using existing model files under `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/`.
4. Compare the frozen scalar and family summaries against the already known scale1.75 official cheap7 sweep. Treat scale1.75 as the development trajectory because the family choices were motivated by its late losses.
5. Only if the scale1.75 result is coherent, obtain or run official-compatible cheap7 for a genuinely different trajectory such as U256 or spatial repair route status at a small late-window set. A three-point test (`chck_80M`, `chck_82M`, `chck_100M`) can test endpoint direction; a fuller `77M..83M,100M` set is stronger if GPU time is available.
6. Apply the exact frozen probe/readout to the independent trajectory without edits. Success means it selects or rank-aligns with that trajectory's own official late-window surface better than corpus NLL and fixed-step baselines. Failure means this contrast-margin family is not yet a general checkpoint selector and should not be turned into a training objective.

## What not to do

- Do not use official benchmark examples, labels, item IDs, or official dataset text in probe construction.
- Do not tune transformation filters or weights after seeing scale1.75 checkpoint scores.
- Do not launch a new private consolidation training route before this measurement either tracks a real independent trajectory or fails clearly.
- Do not interpret the 86M shuffled private-tail projected 41.9899 as a new scientific principle; it remains a tiny numerical hypothesis driven by SuperGLUE variance and GlobalPIQA flips.

## Relevant support files

- Structural-retention synthesis: `notes/lead_strategic_synthesis.md`
- Feasibility scout script: `scripts/contrastive_margin_feasibility_scout.py`
- Feasibility scout output: `data/contrastive_margin_feasibility/contrastive_margin_feasibility_scout.{json,md}`
- Failed token-NLL retention selector summary: `data/retention_cross_trajectory/retention_cross_trajectory_summary.md`
- Late official item flips: `data/scale1p75_late_window_item_flips/scale1p75_82M_to_100M_late_loss.md`
