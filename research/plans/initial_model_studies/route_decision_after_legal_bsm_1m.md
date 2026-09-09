# bsm 1m route control — Route decision after legal BSM 1M

## What changed after legal bsm 1m eval

The legal BSM line now has a sharp result but not a route-positive one.

legal bsm 1m eval showed that the three legal 1M arms trained and evaluated correctly:

- `official_control`: 1,000,000 official words, 98 updates.
- `bsm_coherent_20pct`: 999,874 words, 199,234 binding words, 292 updates.
- `bsm_swapped_20pct`: 999,989 words, 198,869 binding words, 291 updates.

The central mechanism measurement was negative: all three arms scored 0.000 pair-level both-correct on every binding-switch probe, including train templates and natural templates. Same-value preference stayed at 1.0 or near 1.0.

The official fast task profile changed, but the changes cannot be read as learned binding:

| arm | BLiMP | Supplement | Entity | EWoK | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|
| official_control | 54.31 | 48.80 | 17.99 | 48.73 | 37.225 | 6.47 |
| bsm_coherent_20pct | 55.29 | 47.60 | 19.85 | 51.00 | 33.265 | 6.13 |
| bsm_swapped_20pct | 53.35 | 50.40 | 19.12 | 50.45 | 36.710 | 6.81 |

Coherent minus swapped gives Entity +0.73 and EWoK +0.55, but Supplement -2.80, GlobalPIQA -3.445, and Reading -0.68, with zero binding-switch formation. Coherent minus official is even less interpretable because row length, update count, official-data replacement, and targeted-mask supervision all changed together.

bsm 1m route control quantified the training-structure mismatch:

- official control mean row length: 160.00 words;
- BSM arms mean row length: about 53.7 words;
- binding rows mean length: 14.62 words;
- official control updates at batch 64: 98;
- BSM arms updates at batch 64: 291–292.

So the current legal BSM recipe changes input structure and learning opportunity as much as it changes binding content.

## Scientific reading

The current result does not end the BSM line, but it blocks scaling the present corpus/trainer recipe as a mechanism-positive result. The hard open question is whether early DeBERTa has any sub-threshold relation-specific learning signal before pair-level both-correct becomes positive.

The pair-level both-correct measure is too coarse for this question. It can stay at zero even if the correct-value margin is beginning to separate from the distractor. The next experiment must track continuous switch-sensitive margins across development, not only final task scores or binary success.

The evidence supports the following comparison: the smallest useful next experiment is a strictly paired coherent/swapped developmental trace, with a matched official short-row reference to estimate update/packing effects on official columns. Direct 10M/20M scaling of the existing recipe would amplify ambiguity. Immediate architecture reconstruction is also premature until we know whether a stricter paired comparison reveals any early relation-specific margin.

## Chosen next experiment: paired early-binding developmental trace to 4M

### Arms

Run four arms from random initialization with the protected DeBERTa-v2 8x480 configuration:

1. `official_standard`: ordinary official rows, ordinary WWM. This is the usual early baseline.
2. `official_short_targeted`: official-only rows repacked to match BSM row length and update count as closely as possible, with target-mask opportunity matched by masking one content token in each short row plus ordinary WWM as needed. This arm estimates how much of the official task movement comes from update count, row packing, and focused target prediction rather than binding content.
3. `bsm_paired_coherent`: generated binding rows plus official rows, with consistent entity-value bindings.
4. `bsm_paired_swapped`: exact paired counterfactual to coherent. Same master rows, same entities, values, templates, row lengths, row order, batches, number of target masks, and learning-rate schedule, but values are swapped in the input text so each arm has internally correct supervision for its own text.

The swapped arm must not be wrong-label training. It should be a text-level swap with labels consistent with the swapped text. The comparison is coherent relation structure versus swapped relation structure under the same lexical inventory and training mechanics.

### Corpus/materialization repair

Build a new materializer, tentatively `scripts/materialize_paired_bsm_trace.py`, that starts from a master table of binding events. For each event, produce both coherent and swapped versions from the same entities, values, templates, query entity, row index, and intended batch order.

Required properties:

- coherent and swapped arms have the same number of rows and nearly identical word counts;
- same entity frequency, value frequency, template frequency, query-entity frequency, context order frequency, and target-position distribution;
- shared row order after shuffling, so batch sequence is matched;
- official rows inserted in the same positions in both BSM arms;
- metadata records `master_id`, `condition`, `query_entity`, `correct_value`, `distractor_value`, target character span, and target token id;
- a companion official-short corpus built from official text split into short rows with a similar row-count/update profile.

The current earlier analysis coherent/swapped corpora were generated independently and therefore are not exact row-matched controls. That must be repaired before more training.

### Training budget and checkpoints

Train to 4M exposure first, not 20M.

Save checkpoints at:

- 250k;
- 500k;
- 1M;
- 2M;
- 4M.

One seed is enough for the mechanism trace. If there is a small but real relation-specific margin, repeat seed before scaling. If the margin remains zero while target losses fall, the current BSM objective is not assigning early credit to entity-value correspondence.

### Measurements

At every checkpoint measure more than pair-level both-correct:

1. **Continuous switch margin** on train, held-out entities, held-out values, held-out templates, order flips, and natural templates.

For a paired context and its swapped counterpart, measure the correct-minus-distractor logit margin in each condition and report:

- mean margin;
- median margin;
- bootstrap interval;
- switch-direction accuracy;
- pair-level both-correct;
- same-value preference;
- correct-value and distractor-value log-probability means.

2. **Training dynamics**:

- BSM target loss;
- ordinary WWM loss;
- number of target masks by row type;
- if cheap, gradient norm by row type or at least loss curves by row type.

3. **Official fast columns**:

- Entity Tracking;
- EWoK with the nested fast path;
- BLiMP;
- Supplement;
- Reading;
- GlobalPIQA parallel and nonparallel separately.

For official columns, the main interpretation should be:

- coherent minus swapped: relation consistency effect under matched structure;
- BSM arms minus official_short_targeted: content effect beyond update/packing/reference target training;
- official_short_targeted minus official_standard: update/packing/reference effect.

Do not use coherent minus official_standard as the relation effect.

## How outcomes change the route

1. If coherent separates from swapped in continuous switch margin by 0.5M–4M and the signal transfers beyond train templates, BSM remains alive as a developmental mechanism. Then extend the same paired design to 10M/20M and check whether official Entity/EWoK/GlobalPIQA move with the margin.

2. If only train templates improve but held-out/order/natural probes remain flat, the route is template learning, not reusable binding. A larger run of the same synthetic rows is not justified.

3. If BSM target loss falls but continuous switch margin stays near zero, the objective is being solved through local token/template statistics rather than entity-conditioned binding. Then redirect toward a credit-assignment reconstruction: paired context-ranking loss, explicit coherent-vs-swapped contrast, simultaneous dual-entity queries, or an encoder mechanism that makes sparse binding evidence addressable earlier.

4. If coherent and swapped both move official columns similarly, official-score movement is training-structure/content distribution rather than consistent binding.

5. If binding margin forms but EWoK, GlobalPIQA, Supplement, or Reading are damaged, the route may teach binding but still fail the SOTA target. Then test lower-dose or official-mined binding rather than increasing synthetic replacement.

6. Only if coherent beats swapped on transferable switch margin and official Entity/EWoK/GlobalPIQA improve without Supplement/Reading damage should BSM receive a larger 10M/20M continuation and second-seed replication.

## Immediate construction task

The paired materializer and measurement script are prerequisites for a longer run.

Concrete files to produce:

- `scripts/materialize_paired_bsm_trace.py`;
- `scripts/eval_binding_switch_margin_trace.py`;
- optionally a patched trainer wrapper if `train_legal_bsm_screen.py` cannot save the 250k/500k checkpoints or cannot accept the official-short corpus cleanly;
- `data/paired_bsm_trace/` containing a tiny smoke corpus and metadata;
- `notes/paired_bsm_trace_smoke.md` proving exact paired row matching, target span correctness, row-order matching, and loadable checkpoint behavior.

Only after those checks does the proposed design proceed to the 4M four-arm trace on H100.
