# effective input repaired contrast result result: supplied invariance, visible shortcuts, and wrong-constraint degradation

## Why this step changed the experiment

The mechanism macro convergence repetition-vs-variation design was not worth completing as originally framed. It assumed that new name pairs in `VARY_CONTEXT` force the relational learner to abstract over entities. In the actual PosAlign harness, hard matching replaces the two surface names by the same candidate/other parameter vectors before the GRU. When matching succeeds, raw name identity does not reach the relational learner.

This means the old contrast tested mostly object words, active/passive order, and filler tokens after the name coordinate was already installed by the frozen matcher. It did not test whether finite varied experience teaches entity invariance.

The direct evidence is saved at:

- script: `experiments/archive/functional_learning/scripts/effective_input_audit.py`
- result: `research/documents/functional_learning/data/effective_input_audit/effective_input_audit.md`
- json: `experiments/archive/functional_learning/data/effective_input_audit/effective_input_audit.json`

Important counts after replacing names by `<C>/<O>`:

| set | events | unique effective patterns | unique object-abstract patterns | notes |
|---|---:|---:|---:|---|
| fixed train comparison | 384 | 200 | 14 | all comparison rows used by REPEAT |
| `VARY_CONTEXT` epoch 1 | 384 | 161 | 8 | name changes vanish; object/voice remain |
| `VARY_NOISE` epoch 1 | 384 | 383 | 369 | filler changes dominate |
| eval changed state | 384 | 80 | 8 | held names do not make a held effective name pattern |

After object abstraction, fixed train comparison contains all 8 eval changed-state abstract patterns (`intersection=8`, `B_minus_A=0`, Jaccard `0.571` because train also contains comparison-only pair patterns). Thus the original `VARY_CONTEXT` arm did not supply a clean new effective abstraction pressure.

One old mechanism macro convergence cell had already completed before timeout:

- `experiments/archive/functional_learning/data/repeat_vs_variation/repeat_bs+1_e60_seed30000/result.json`
- It saturated: train comparison `1.0`; held h0/h1/h2/h3 all `1.0`; h1 margin `17.44`, h3 margin `18.04`.

This single saturated cell is useful only as confirmation that the task is easy under repeated fixed comparison evidence. It does not answer the variation question.

## Repaired effective-input pilots

I then built a fast effective-input pilot that directly canonicalizes names to `<C>/<O>` and uses a shared GRU with separate state/comparison heads:

- script: `experiments/archive/functional_learning/scripts/repaired_cue_contrast.py`

The purpose was to introduce visible nuisance information that actually reaches the GRU and see whether variation helps acquire invariance to it. Three families were tested.

### 1. Relation cue family

Conditions:

- `clean`: no cue.
- `cue_locked`: relation-specific token prepended to train events.
- `cue_varied_dropout`: cue randomized independently of relation or omitted.
- evaluated on no-cue, matched-cue, and wrong-cue held events.

Output:

- `research/documents/functional_learning/data/repaired_cue_contrast_wrongcue_e20/combined_results.md`
- `research/documents/functional_learning/data/repaired_cue_contrast_plus_e40/combined_results.md`

Result: by epoch 20, all tested conditions reached no-cue graph transport `1.0` and wrong-cue graph transport `1.0`. At epoch 15, no-cue graph was `0.688` for clean and `0.750` for both locked and varied/dropout, but this was a small transient and did not persist. The wrong-cue result shows the cue was not load-bearing after relation learning.

### 2. Packet shortcut family

Conditions:

- `packet_locked`: both events in a positive training comparison row share a row-level packet token; a learner could fit comparison by packet identity.
- `packet_locked_noise`: same packet token plus neutral noise.
- `packet_broken`: equal packet-like material but independent across the two events and changing by epoch.
- direct state anchors remain clean; packet tokens are absent at eval.

Output:

- `research/documents/functional_learning/data/repaired_packet_e40/combined_results.md`
- `experiments/archive/functional_learning/data/repaired_packet_e40/combined_results.json`

Key numbers:

| condition | epoch-15 no-cue graph | epoch-15 train cmp | final no-cue graph |
|---|---:|---:|---:|
| clean | 0.750 | 1.000 | 1.000 |
| packet_locked | 0.771 | 0.995 | 1.000 |
| packet_locked_noise | 0.771 | 0.995 | 1.000 |
| packet_broken | 0.792 | 0.995 | 1.000 |

The packet shortcut did not create a stable separation. Broken packets were slightly ahead at epoch 15, but all conditions saturated by epoch 20.

### 3. Comparison-masked relation family

Conditions:

- direct state anchors remain clean;
- in comparison events, relation verbs are masked with probability `0.75`;
- `comp_cue_locked_masked`, `comp_cue_locked_noise_masked`, and `comp_cue_varied_dropout_masked` manipulate visible cue tokens on the comparison side.

Important correction: the first directory `repaired_compmasked_e40` was produced before fixing the regex in `mask_relation_verb`; it did **not** mask relation verbs and should not be used as evidence. After patching `re.sub(r"\b...\b")`, epoch 1 masked 281/384 comparison event appearances.

Corrected output:

- `research/documents/functional_learning/data/repaired_compmasked_fixed_e40/combined_results.md`
- `experiments/archive/functional_learning/data/repaired_compmasked_fixed_e40/combined_results.json`

Key final numbers at epoch 40:

| condition | final train cmp | no-cue graph | matched-cue graph | wrong-cue graph | no-cue h1 | no-cue h3 |
|---|---:|---:|---:|---:|---:|---:|
| comp_clean_masked | 0.495 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| comp_cue_locked_masked | 1.000 | 0.792 | 0.979 | 0.750 | 0.583 | 1.000 |
| comp_cue_locked_noise_masked | 1.000 | 0.938 | 1.000 | 0.500 | 0.875 | 1.000 |
| comp_cue_varied_dropout_masked | 0.526 | 0.750 | 0.750 | 0.750 | 0.500 | 1.000 |

Budget-curve facts:

- all fixed comp-masked arms stayed near no-cue graph `0.5` through epoch 20;
- `comp_clean_masked` eventually reached no-cue graph `1.0` by epoch 35-40 despite not fitting the masked comparison objective (`train_cmp_current` about `0.5`), so that cell is not clean evidence of comparison-mediated learning;
- `comp_cue_locked_masked` and `comp_cue_locked_noise_masked` fit the comparison objective (`train_cmp_current=1.0`) but remained cue-dependent: matched-cue graph exceeded no-cue graph, and wrong-cue graph was lower, especially for locked-noise (`wrong-cue graph=0.5`, h1=0.0);
- `comp_cue_varied_dropout_masked` did not rescue no-cue h1; it underfit comparison and ended with h1 at chance.

This repaired contrast shows that when true relation evidence is partly hidden, a visible relation-specific cue can become load-bearing for comparison training and can degrade no-cue h1 transport relative to the clean masked condition. It does not show a positive advantage of randomized/dropout cue variation; in this toy pilot, cue variation removed the shortcut but did not supply enough learnable relation evidence to recover h1.

## Scientific interpretation

topology phase1 result remains strong: correct relational constraint content controls coordinate transport, and wrong h1-incident comparison labels actively install wrong h1 orientation while h3 stays correct.

effective input repaired contrast result adds three boundary facts:

1. The old mechanism macro convergence `VARY_CONTEXT` contrast was not a valid test of learning name invariance, because PosAlign hard matching supplies that invariance before the GRU.
2. Once name invariance is supplied and the real relation tokens remain visible, extra surface variation (relation cues, packet tokens, filler/noise) does not produce a robust held-transport advantage; the unmasked cue/packet pilots all saturated by epoch 20.
3. When true relation evidence is obscured, a visible shortcut can become a competing computation: the corrected comp-masked cue-locked conditions fit training comparison and degrade no-cue/wrong-cue h1, while randomized/dropout cue variation does not automatically fix the missing information.

The useful principle is therefore not "variation beats repetition". A sharper formulation is:

> finite experience is valuable when it supplies a constraint or invariance that is not already available to the learner, remains active under the training budget, and enters the target-usable coordinate; exact or wrong recurrence can install a competing relation-specific computation, while variation helps only when it changes the effective learner input in a way that supplies the missing correspondence rather than merely adding surface diversity.

This also bounds the connection to relation_learning. The BabyLM Entity depth dissociation remains important independent evidence, but it is not established by resemblance to controlled h1/h3 transfer. controlled h1/h3 is single-event relation transfer through a comparison graph, not multi-operation state tracking. The repaired synthetic pilots have not yet reproduced the repeat-versus-view depth dissociation; they instead provide a controlled boundary and degradation mechanism that should shape the next bridge experiment.

## Consequence for relation_learning copy/content requests

The proposed cross-substrate readouts were repeated-vs-corrupted source copy gain, source-present-vs-unrelated rewrite/content gain, and stale-initial/update ablations. The current synthetic classifier is not a masked language model and has no token NLL, so it cannot measure the same copy/content quantities faithfully. A superficial score-margin analogue would risk mixing head calibration with token prediction.

A faithful synthetic analogue would require either:

1. a small generative or masked-LM synthetic sequence task with source-repeat packets, rewrite/content variants, and multi-step entity updates; or
2. a raw-input version of the current harness where entity matching is not supplied, so varied name contexts can actually teach the missing invariance.

The second route is closer to the existing coordinate-transport mechanism. It would compare frozen PosAlign matching against learned/soft matching and ask whether name/context variation helps acquire the entity coordinate before relational transport. The first route is closer to the BabyLM Entity/copy readouts and may better bridge to BabyLM.

## Next research work

Do not spend more compute completing the old mechanism macro convergence remaining five long PosAlign cells unless a later design gives them a new role. They would mostly test a contrast whose central manipulation does not reach the relational learner.

The next useful construction should remove or vary the supplied invariance itself. Two strong options:

1. **Raw/learned name-coordinate contrast:** keep the controlled relation graph but do not hard-replace names by `<C>/<O>` with a frozen matcher. Compare repeated fixed name contexts with varied name contexts while tracking matcher accuracy, no-cue h1/h3 transport, and sign reversal across the budget curve. This directly tests whether variation helps acquire the entity-invariance coordinate rather than transport after that coordinate is given.
2. **Synthetic multi-operation sequence task:** build source-repeat vs rewrite/content variants with actual token-prediction or masked-token scoring, plus direct-copy and state-update probes. This would make the BabyLM copy gain, content-conditioning gain, stale-initial ablation, and relevant-update depth quantities comparable in a controlled synthetic setting.

Either route should be designed before execution so the effective input, target competence, and shortcut are explicit.

## Addendum: wrong h1 constraints do degrade held nonidentical content use below full graph

In response to relation_learning evidence revision source trigger test, I analyzed the existing topology phase1 result full-graph and adversarial-h1 prediction files by relation, voice, object, and comparison edge:

- script: `experiments/archive/functional_learning/scripts/degradation_analysis.py`
- result: `research/documents/functional_learning/data/degradation_analysis/degradation_analysis.md`
- json: `experiments/archive/functional_learning/data/degradation_analysis/degradation_analysis.json`

Key result: adversarial h1-incident comparison labels do not merely remove an improvement; they install a competing h1 coordinate that is wrong on held nonidentical renderings.

State readout:

| relation | n | full acc | adversarial acc | full target-signed margin | adversarial margin |
|---|---:|---:|---:|---:|---:|
| h0_dax | 96 | 1.000 | 1.000 | +11.963 | +12.020 |
| h1_mep | 96 | 1.000 | 0.000 | +11.938 | -20.804 |
| h2_norp | 96 | 1.000 | 1.000 | +12.605 | +20.841 |
| h3_ziv | 96 | 1.000 | 1.000 | +15.912 | +0.917 |

The h1 degradation is uniform across active/passive voice and all held h1 objects (`badge`, `booklet`, `drum`, `flute`, `hammer`, `scarf`, `shell`, `ticket`): full h1 accuracy `1.0`, adversarial h1 accuracy `0.0` in every subgroup.

Eval comparison readout:

| edge family | n | full acc | adversarial acc | margin delta |
|---|---:|---:|---:|---:|
| h1 involved | 192 | 1.000 | 0.000 | -19.368 |
| h1 not involved | 448 | 1.000 | 1.000 | +0.055 |

This is the current synthetic analogue of the question about wrong/exact constraints lowering nonidentical content use below a clean condition. It is stronger than a failure-to-improve result: corrupted h1 relation content systematically reverses held h1 use while non-h1 edges remain intact. It still is not equivalent to the BabyLM token-NLL nonoverlap rewrite probe, because the synthetic instrument is a supervised event scorer rather than a masked/generative LM.
