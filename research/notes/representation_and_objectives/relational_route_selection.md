# corpus directional pair census — Relational route selection after corpus census

## Changed judgment

The initial design compared three mechanisms: random context-sensitivity IC-MLM, a relational bottleneck adapter (RBA), and late relational masking (PGDC). Independent scientific checking and a 10M-corpus census materially changed the route.

### Corrections

1. **Random IC-MLM is misaligned.** Maximizing KL after deleting random context tokens rewards generic brittleness, not the correct context–outcome ordering. It can make predictions more context-sensitive while leaving the EWoK four-cell interaction negative.
2. **RBA alone supplies no relational supervision.** Differential learning rates do not make an adapter relational; standard MLM may use it for lexical or topical features. It remains useful only if paired with an aligned relational signal.
3. **Original PGDC masks the wrong token.** Masking a relation word teaches relation-word reconstruction. Conditional compatibility instead requires keeping the relational pivot visible and masking its dependent argument, property, state, or consequence.
4. **A four-cell matched/crossed loss is scientifically ideal but lacks substrate.** The compact 10M corpus census found only 150 structural-template directional pairs, and sample inspection showed many are repeated CHILDES utterances or lexical substitutions (`I want it off/on`, `push/pull it on`) rather than valid context–consequence reversals. Treat 150 as a noisy upper bound, not usable pair count. The earlier directed-event miner independently found only 2–3 high-quality reversed-argument pairs across 38,167 FineWeb sources. Natural minimal pairs are too sparse for the main training signal.

## Corpus census

Input: exact 10M compact corpus `data/fw_full_arms/fw_preserved_compact_view_10M.jsonl`.

- 690,663 sentences of at least 5 words.
- 342,159 (49.54%) contain a broad causal/spatial/temporal/comparative marker.
- 341,950 (49.51%) support the cheap proxy pattern “keep a relational pivot visible; mask nearby non-pivot content.”
- 2,587,733 nearby maskable content positions, mean 7.57 per eligible sentence.
- Only 150 noisy structural-template directional pairs across 36 lexical pair types.

The broad marker counts are deliberately permissive and include ambiguous common words (`in`, `on`, `is`, `light`). They establish abundant candidate single-sentence substrate, not 342k clean relational examples. A dependency/quality pass must refine positions before training.

Evidence:
- `data/corpus_directional_census/corpus_directional_pair_census.json`
- `data/corpus_directional_census/directional_pairs.jsonl`
- `scripts/corpus_directional_pair_census.py`

## Selected first mechanism: Pivot-Visible Dependent Masking (PVDM)

PVDM changes the conditional-learning event within ordinary corpus sentences:

- identify a relational pivot `p` (causal/change verb, spatial or temporal relation, comparison, polarity/modality);
- force `p` to remain visible;
- increase masking probability for syntactically linked arguments, states, attributes, and consequences `d`;
- preserve an equal global effective WWM rate by reducing masking on unrelated positions;
- compute ordinary MLM loss on the dependent targets.

The event is therefore `predict d | visible p, context`, rather than randomly masking `p` or `d`. This is distinct from relation-data reallocation: it changes credit assignment while holding the corpus fixed.

### Preprocessing specification

Use dependency parsing plus conservative lexical patterns. Candidate pivot classes:

- causal/change verbs and markers;
- spatial/temporal adpositions and adverbs;
- comparative predicates;
- polarity and modality changes;
- directed transfer/motion verbs.

Candidate dependent positions must be dependency-linked to the pivot (argument, object, complement, attribute, adverbial consequence) or lie in a delimited consequence clause. Exclude punctuation, function-only targets, speaker tags, formatting artifacts, and ambiguous unparsed occurrences. Preserve exact whitespace-word to tokenizer whole-word alignment.

### Matched masking control

PVDM must be compared against **POS/frequency/distance-matched visible-anchor masking**:

- choose a nonrelational visible anchor with matched POS/frequency;
- mask matched nearby dependents at the same positions/count distribution;
- preserve global mask count, masked-token frequency, sequence length, corpus rows, stream order, initialization, optimizer reset, and word exposure.

This separates relational pivot use from generic local-context concentration.

### Minimal causal experiment

Use the shared compact `chck_70M` as common initialization and the exact remaining 30M stream (boundary is exact at row 449,281; 192,549 rows / 30,000,000 words remain). Run two newly materialized continuations with symmetric optimizer reset:

1. PVDM treatment.
2. POS/frequency/distance-matched visible-anchor control.

A new standard continuation with the same reset is preferable if compute allows, because the archived 70M→100M control retained optimizer state and is not fully matched to a reset continuation.

First read at 80M (10M continuation); continue to 100M only if row-level mechanism movement appears without broad damage. This is the lowest-cost reliable stage because early MLM loss alone is not informative, but the saved official-compatible relation readers can evaluate a checkpoint.

Readouts:

- EWoK four-cell on the fixed 7,618 rows: interaction over all rows; fixed 70M-wrong subset; corrected/newly-wrong rows; stable-failure fraction; domain/context-diff movement.
- GlobalPIQA all-option rows: parallel/nonparallel accuracy, hard52 ranks, mean top-minus-correct margin, rows below 0.5 nat.
- broad sentinels: Entity and Supplement first; cheap7 only if those are preserved.
- corpus mechanism readout: masked-dependent NLL by pivot class and visible-anchor control class.

Promising movement requires BOTH:

- EWoK conditional interaction improves on a fixed baseline row set, not merely a changing denominator among wrong rows;
- GlobalPIQA hard52 rank/margin improves without nonparallel deterioration;
- Entity and Supplement remain near the matched control.

Numerical thresholds in the initial note (e.g. 2 percentage points or 0.15 nat) were heuristic and should not be treated as established significance. Use paired row bootstrap intervals and repeated mask seeds.

## Second mechanism if PVDM moves relations but harms broad capability: supervised relation-residual subspace

Add a small residual bottleneck only where PVDM supplies dependent-target gradients:

`h' = h + W2 GELU(W1 LN(h))`

with random `W1`, zero `W2`, fixed residual scale 1. This preserves initial function while permitting immediate gradients into `W2` (unlike zero gate plus zero internal weights). Keep backbone LR low and adapter LR higher. Route PVDM-dependent loss through the residual pathway while ordinary MLM remains on the full hidden state. Log residual norms and gradient norms.

This is stronger than unsupervised RBA because its specialization is defined by PVDM positions. It tests interference only after an aligned single-sentence signal exists.

## Deferred mechanism: paired four-cell interaction loss

For valid pairs `(C_A,T_A),(C_B,T_B)`, the ideal loss is:

`softplus(m - [S_AA + S_BB - S_AB - S_BA])`.

Both contexts can be encoded in one batched forward pass and outcomes cross-scored through the existing decoder. However, current allowed-corpus mining does not provide enough high-quality pairs. Do not launch this objective from the 150 noisy census candidates. Reopen only if a new corpus-derived miner demonstrates thousands of source-group-disjoint, semantically valid pivot/outcome pairs under manual sample reading.

## Role of the transition structure match 12,418-word transition/control set

The transition structure match pair remains a source/style-matched data-mechanism probe, not the selected training substrate. It can test whether PVDM draws more useful signal from transition-rich sentences than matched narrative controls, but its narrative-heavy 12,418 words are too small and narrow to decide the SOTA route alone. The first PVDM experiment should operate broadly on the legal compact stream, with dependency-quality filtering.

## Scientific interpretation

The corpus contains abundant unpaired relational statements but almost no clean natural minimal reversals. The strongest immediately testable principle is therefore:

> sample-efficient conditional knowledge formation can be improved by controlling **which evidence remains visible when a dependent is predicted**, rather than merely increasing relation-text frequency or masking relation words.

This principle directly changes credit assignment while remaining evaluation-independent and compatible with the allowed 10M corpus. The experiment is not yet authorized until the dependency tagger reports yield, class balance, ambiguity samples, and exact whole-word alignment, and the continuation implementation passes K=0/standard-masking parity checks.
