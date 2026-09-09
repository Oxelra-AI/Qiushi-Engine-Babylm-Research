# raw span identity routing result — Raw-span identity routing: lexical competition, soft null gating, and oracle ceiling

## Scientific question

span matcher construction and design established a nested factorization in a controlled harness: a context-free character-level matcher can bind disjoint held names at supplied `<name>` positions, and the shared relational trunk can then transport an anchor-selected relational gauge. raw span discovery design removed the supplied entity positions by applying a character matcher to every raw token. The interpretation of raw-span identity routing must treat `--no-name-vocab` as causal isolation, not completion of natural span discovery: the high-value question is whether data-efficient learning requires sparse identity evidence to be routed through a reusable equality representation before lexical memorization or other local solutions absorb the problem.

## Engineering repairs made in raw span identity routing result

- Added `--bridge-signs` to `training/scripts/raw_span_discovery_probe.py` so each bridge-sign cell is independently recoverable. This avoids the raw span discovery design two-sign timeout failure.
- Repaired the `--no-name-vocab` bug: `run_one` now receives `no_name_vocab` explicitly instead of referencing out-of-scope `args`. CPU smoke confirmed vocabulary size 60 and train names such as `mira/omar` map to `<unk>`.
- Found that the original `matcher=oracle` mode was not a true oracle: it only altered character tensors while leaving the learned dot-product gate in control. Patched it to hard-route exact raw-token matches: candidate-name tokens receive the candidate embedding, other-name tokens receive the other embedding, and all remaining tokens keep word embeddings. CPU smoke confirmed train/eval match accuracy 1.0.

## GPU results

### 1. Learned raw-span, no name vocabulary, shared trunk

Outputs:
- `data/noname_shared_bsplus_r2/`
- `data/noname_shared_bsminus_r2/`
- analysis: `data/noname_span_analysis/noname_span_analysis.md/json`
- matcher audit: `data/matcher_failure_audit/matcher_failure_audit.md/json`

Key result table from saved predictions:

| run | vocab | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | eval cand>0.5 | eval other>0.5 | cand>other | other>cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical_shared_plus (raw span discovery design) | 76 | 1.000 | 0.958 | 0.375 | 0.672 | 0.500 | 0.492 | 0.344 | 0.344 | 0.781 | 0.781 |
| noname_shared_plus | 60 | 1.000 | 0.542 | 0.500 | 0.906 | 0.500 | 0.492 | 0.438 | 0.438 | 0.875 | 0.875 |
| noname_shared_minus | 60 | 0.778 | 0.917 | 0.594 | 0.906 | 0.578 | 0.488 | 0.062 | 0.062 | 0.594 | 0.594 |

The no-name-vocab isolation did **not** restore the span matcher construction and design/290 fingerprint. It improved unchanged rows (0.906 versus 0.672 in the lexical-available plus run) and the plus cell learned partial relative equality (`cand>other` and `other>cand` both 0.875), but it did not turn equality into actual routing through the candidate/other embeddings. The stricter gate metric is the important one: both name tokens beat the `neither` channel on only 0.250 of eval changed rows in `noname_shared_plus`, and 0.000 in `noname_shared_minus`. The bridge-sign pair also lacks coherent coordinate transport: graph-transfer row-paired sign opposition is only 0.312; unchanged rows remain same-signed (1.000), but graph-transfer minus margins collapse near zero.

Interpretation: the raw span discovery design failure was not merely caused by train-name word embeddings. Removing those embeddings eliminates one shortcut, but the relational objective still fails to learn a robust all-token candidate/other/neither equality router. The bottleneck is upstream of the shared relational gauge coordinate.

### 2. Hard exact raw-token localization oracle, no name vocabulary, shared trunk

Outputs:
- `data/noname_oracle_shared_bsplus/`
- `data/noname_oracle_shared_bsminus/`
- included in `data/noname_span_analysis/`

Patched oracle result:

| run | vocab | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | eval match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard oracle bs+1 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard oracle bs-1 | 60 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |

Row-paired analysis:

| family | opposite sign across bridge signs | same sign |
|---|---:|---:|
| direct_anchor | 1.000 | 0.000 |
| graph_transfer | 1.000 | 0.000 |
| unchanged | 0.000 | 1.000 |

Comparison coordinates are also coherent in this repaired raw-span ceiling: held-held comparison products stay same and correct (1.000/1.000), while mixed held-seen orientation flips with the bridge sign (plus correct 1.000, minus 0.000). This is stronger than span matcher construction and design's learned/oracle mixed discrepancy and shows that raw-token/no-name-vocab downstream architecture can carry the original causal gauge experimental design/291 gauge mechanism when localization is exact.

Interpretation: dedicated train-name lexical entries are not necessary for shared-trunk gauge transport once candidate/other binding is correct. Raw text itself is not incompatible with the gauge mechanism. The learned failure localizes to the identity-routing interface: equality/null calibration, hard versus soft substitution, and joint identity-relational optimization.

## Matcher-only equality pretraining pilot

Output: `data/eq_pretrain_pilot/eq_pretrain_pilot.md/json`.

A cheap CPU-only pilot trained the current CharGRU + neither gate directly on token/query equality labels from train texts, without relational training.

| epoch | train both gate | eval both gate | train cand>other | eval cand>other | train neither(cand) | eval neither(cand) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.599 | 0.585 | 0.738 | 0.731 |
| 25 | 0.625 | 0.189 | 0.844 | 0.842 | 0.192 | 0.362 |
| 50 | 1.000 | 0.407 | 1.000 | 1.000 | 0.078 | 0.339 |
| 100 | 1.000 | 0.654 | 1.000 | 0.971 | 0.010 | 0.227 |

This shows the explicit equality objective can perfectly route training names by epoch 50, and it learns relative equality on held names (held `cand>other` 1.000 at epoch 50, 0.971 at epoch 100), but the all-token null gate still generalizes only partially to held names: held `both_wins_gate` reaches 0.654 at epoch 100, not near 1.0. Therefore an immediate relational equality-aux GPU run would be ambiguous unless the equality router is first made reliably held-generalizing or converted to a hard joint-assignment interface.

## What raw span identity routing result establishes

Supported:
1. **Lexical name embeddings are not the only obstacle.** The no-name-vocab causal isolation removed whole-name lookup for train names but did not restore learned routing or graph transport.
2. **The downstream raw-span shared-trunk architecture is sufficient under correct binding.** Hard exact localization with vocabulary 60 reproduces full anchor-controlled gauge transport, unchanged stability, held-held closure, and mixed held-seen sign behavior.
3. **The failing computation is upstream of canonical role tokens.** The weak point is turning sparse identity evidence into stable candidate/other/neither routing. Relative equality can appear without actual routing through the candidate/other embeddings.
4. **This route strengthens the emerging principle by adding representation competition and timing.** Efficient reuse depends not just on identifiable examples and shared relational coordinates, but on getting low-level identity evidence into the reusable route before local solutions or null/lexical channels absorb credit.

Not supported yet:
1. Natural raw-span discovery is not solved. The learned all-token matcher fails; hard oracle is a ceiling.
2. `--no-name-vocab` does not fully isolate identity. It removes train-name lexical embeddings but leaves template/order cues, the null channel, and soft-mixture calibration.
3. Representation competition is not uniquely proven. The current evidence implicates null calibration, soft routing, and joint identity-relational optimization, with lexical competition as one contributor.
4. The learned sign pair is not a clean converged gauge pair: `noname_shared_plus` has train comparison 0.542 and `noname_shared_minus` has train state 0.778; the negative scientific content is therefore interface failure before the original transport test becomes reachable.

## Next experiment design

Do not launch a broad BabyLM-scale or 100M intervention from this state. The next high-value test should separate four factors using the same fixed substrate, fresh initialization, held names, and bridge-sign intervention:

1. **Equality representation quality:** train/evaluate a context-free equality module until held exact two-name assignment is near-perfect. The current CharGRU+three-way softmax is insufficient; use a joint assignment readout over all tokens or a stronger tied character encoder/contrastive equality loss. Measure train, held, fresh-renamed, and near-name cases.
2. **Hard versus soft routing:** feed the same equality scores either through hard joint assignment (canonical candidate/other replacement) or soft three-way mixture. If hard assignment works but soft mixture fails, the principle is null/mixture calibration, not equality representation.
3. **Protected versus exposed identity route:** freeze the equality module before relational training versus allow relational gradients to update it. If protected works and exposed drifts toward the raw span identity routing result failure, local relational objectives overwrite reusable identity routing.
4. **Lexical competition reintroduction:** after a reliable equality route exists, reintroduce lexical name embeddings in a crossed condition. If lexical-present exposed training fails while frozen/shared character route works, this directly supports the hypothesis that reusable equality must be routed before lexical memorization absorbs sparse identity evidence.

The cleanest next GPU sequence is:
- first CPU/GPU matcher-only improvement until held joint assignment is close to exact;
- then two shared-trunk bridge-sign cells with protected hard-assignment equality;
- then the same with exposed matcher and with lexical vocabulary present.

A new untied relational control is useful only after the routing module is reliable; otherwise untied failure would be uninterpretable.

## Files produced or changed in raw span identity routing result

- Patched raw-span script: `training/scripts/raw_span_discovery_probe.py`
- Repaired no-name learned outputs: `data/noname_shared_bsplus_r2/`, `data/noname_shared_bsminus_r2/`
- Hard-oracle raw-span outputs: `data/noname_oracle_shared_bsplus/`, `data/noname_oracle_shared_bsminus/`
- Analysis script: `scripts/noname_span_competition_analysis.py`
- Analysis outputs: `data/noname_span_analysis/noname_span_analysis.md/json`
- Matcher failure audit: `scripts/matcher_failure_audit.py`, `data/matcher_failure_audit/`
- Equality-aux candidate script: `training/scripts/equality_aux_span_probe.py` (syntax/CPU-smoke checked; not yet scientifically decisive)
- Equality pretraining pilot: `scripts/eq_pretrain_pilot.py`, `data/eq_pretrain_pilot/`
- independent_review support: , 
