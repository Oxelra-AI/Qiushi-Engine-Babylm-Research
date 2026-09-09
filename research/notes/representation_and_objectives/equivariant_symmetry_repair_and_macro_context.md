# equivariant symmetry repair and macro context — Equivariant symmetry-interface repair and macro context

## Why the symmetry pilot boundary findings random-init pilot was not run as decisive evidence

The symmetry pilot boundary findings learned pilots exposed a deterministic `0.750` mixed-comparison outcome on the symmetric conjoined surface. The strategist note correctly sharpened the interpretation: because the comparison surface itself admitted a name/order solution, a random-init run on that same surface could not distinguish:

1. latent global role-permutation ambiguity from the role coordinate anchor and state probe factorized law,
2. pretrained frame inheritance,
3. architecture/tokenizer bias, and
4. the evaluation surface's deterministic order shortcut.

Therefore equivariant symmetry repair and macro context did **not** treat the already-written symmetry pilot boundary findings random-init script as decisive and did not build a synthesis around frame inheritance. The priority was to make the learned interface actually equivariant under global role exchange and verify transparent surface solvers first.

## Repaired equivariant symmetry repair and macro context surface

Created `experiments/archive/representation_and_objectives/scripts/equivariant_symmetry_substrate.py` and generated the repaired substrate under:

- `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate`
- summary: `research/documents/representation_and_objectives/data/equivariant_symmetry_substrate/equivariant_substrate_summary.md`
- full report: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/equivariant_substrate_report.json`

The construction changes relative to v2/symmetry pilot boundary findings are:

- use active/passive syntactic role orbits rather than order-free conjoined events, preserving a natural reusable argument interface while counterbalancing visible name order;
- balance the held-held graph with two same-orientation and two opposite-orientation relation pairs, removing the 2:1 edge-type imbalance that previously made order-like rules score above chance;
- counterbalance static owner independently of event voice/surface order, so unaffected rows no longer collapse to "first visible participant";
- retain train/eval disjoint names, changed objects, and static objects;
- keep held relation text free of known transfer verbs and to/from cues.

Formal file-only checks after repair:

| arm | satisfying assignments |
|---|---:|
| exposure_only | 16 |
| heldheld_only | 2 |
| aligned_state_bridge | 1 true assignment |
| inverted_state_bridge | 1 inverted assignment |
| neutral_decoupled | 2 |
| mixed_event_bridge | 1 true assignment |

The global checks report:

- held surface transfer-word scan: 0 hits across 5120 held-related fields;
- train/eval names and objects disjoint;
- heldheld_only keeps exactly the global Z2 ambiguity;
- aligned identifies true, inverted identifies inverted, neutral preserves Z2, mixed identifies true;
- mixed-eval order rules at chance;
- changed-state eval order rules at chance.

## Transparent surface baselines

Created and ran:

- `experiments/archive/representation_and_objectives/scripts/surface_baselines.py`
- outputs under `experiments/archive/representation_and_objectives/data/surface_baselines`

The relevant summaries are:

- `surface_baselines_no_common_summary.md`
- `surface_baselines_with_common_summary.md`

Key result after the final static-owner repair:

- `order_only` pattern-majority baselines are 0.500 on held-held eval, mixed held-seen eval, paired changed, paired unchanged, and cross-template changed.
- `order_reltype`/`order_relid` can memorize seen training patterns and, for directly anchored state rows, reflect the intended orientation in aligned versus inverted arms, but they remain 0.500 on mixed held-seen unless mixed evidence is directly present. This is useful: relation identity alone does not leak absolute mixed orientation from the held-held/neutral conditions.

Created and ran a raw lexical surface check:

- `experiments/archive/representation_and_objectives/scripts/bow_surface_baseline.py`
- outputs under `experiments/archive/representation_and_objectives/data/bow_surface_baseline`

Critical BoW results:

- relation-comparison transfer: all arms score 0.500 on held-held and mixed eval, both raw and normalized, despite raw train fits often reaching 1.000;
- state-query transfer: changed, unchanged, and paired-both readouts remain at chance/default across arms and normalizations; after the static-owner repair, unchanged rows are no longer solved by static textual position.

This means the repaired surface now passes the specified cheap tests: transparent order/name/voice and BoW solvers cannot reproduce the previous 0.750 mixed result or the changed-state orientation readout.

## Learned pilot status

Created `experiments/archive/representation_and_objectives/training/scripts/equivariant_model_pilot.py` to compare pretrained and random-init DeBERTa on the repaired surface. The one-arm one-epoch preflight initially found a missing `Counter` import, which was fixed. The repaired preflight completed on `heldheld_only` with one epoch and produced chance eval scores as expected for an undertrained model.

Submitted two managed jobs after the file and transparent baselines passed:

- `s278_t23_tool1`: pretrained DeBERTa, arms `heldheld_only aligned_state_bridge inverted_state_bridge neutral_decoupled`, seeds `27800 27801 27802`, 40 epochs, intended GPU0. This failed with CUDA OOM because GPU0 was already effectively full (`nvidia-smi` later showed both GPUs near 79/81 GB used). This failure is a resource conflict, not evidence about the mechanism.
- `s278_t23_tool2`: random-init DeBERTa, same arms/seeds, 80 epochs, intended GPU1. At last deliberate status check it was still running; no scientific result should be inferred until its authoritative terminal result is delivered.

No further GPU work was launched. The full fine-tuning result remains pending; a lighter model or frozen-encoder comparison is an alternative only if the full comparison cannot be completed.

## Register-Substitution Evidence

Read file-only displaced-block readout:

- `research/documents/frontier_consolidation/data/displaced_clean_block_readout/displaced_clean_block_summary.md`

The important correction is that the sub-dose ladder does **not** remove protected `qwen_pair_packed` synthetic rows. At MAX, `qwen_pair_packed_words_displaced_at_max = 0`. The displaced words are ordinary BabyLM-source heldout rows: chiefly CHILDES (364,916), Gutenberg (263,557), OpenSubtitles (256,165), SimpleWiki (150,554), BNC Spoken (80,511), and Switchboard (2,884). Therefore a plateau at rho≈0.011 would mean that admitting a small amount of authentic FineWeb source+view packet material while replacing ordinary BabyLM heldout text is enough; growth through rho≈0.042 would support amount/depth inside the admitted FineWeb slice. It would not mean direct removal of Qwen paired material.

This macro correction does not reopen the compact/source-correspondence mechanism from Entity aggregate alone. It changes how to interpret sub-dose curve when scores land. In companion analysis, the compact branch remains closed under the paired conservation evidence from earlier analysis unless new score curves show broad ex-Entity V-B or paired conservation patterns strong enough to change that premise.

## Current scientific state

The strongest current route remains the symmetry-identification principle from role coordinate anchor and state probe, but now with a cleaner language-interface test:

- formal Z2 ambiguity survives in the repaired substrate;
- aligned/inverted/neutral bridge conditions have the right assignment algebra;
- transparent surface solvers are at chance on the mechanism-relevant readouts;
- no learned bridge-controlled orientation result has landed yet.

Therefore pretrained frame inheritance is not yet an established principle in this route. It remains a hypothesis that can only be judged after comparing pretrained and random-init models on the repaired surface, with aligned and inverted bridges producing opposite orientations while neutral preserves ambiguity and unchanged facts remain protected.

## Late equivariant symmetry repair and macro context additions: transparent lexical and slot-orbit controls

After the first note was written, two additional CPU-only checks were completed.

### Raw BoW lexical surface learner

Created and ran:

- `experiments/archive/representation_and_objectives/scripts/bow_surface_baseline.py`
- `research/documents/representation_and_objectives/data/bow_surface_baseline/bow_surface_no_common_summary.md`
- `research/documents/representation_and_objectives/data/bow_surface_baseline/bow_surface_with_common_summary.md`

A bag-of-words logistic learner over raw text often fit the training comparison rows, but stayed at 0.500 on held-held eval and mixed held-seen eval. For state queries it also stayed at 0.500 on changed and unchanged readouts after static-owner counterbalancing. Normalizing names and objects did not reveal a hidden lexical route. This strengthens the surface result: simple text statistics cannot reproduce the intended mixed orientation or changed-state readout on the repaired substrate.

### Slot-orbit positive control

Created and ran:

- `experiments/archive/representation_and_objectives/scripts/slot_orbit_solver.py`
- `experiments/archive/representation_and_objectives/scripts/slot_orbit_assignment_analysis.py`
- `research/documents/representation_and_objectives/data/slot_orbit_solver/slot_orbit_assignment_analysis_no_common_summary.md`
- `research/documents/representation_and_objectives/data/slot_orbit_solver/slot_orbit_assignment_analysis_with_common_summary.md`

The slot-orbit control parses the actual event strings into reusable active/passive/give/receive argument roles, then enumerates the held-relation orientation assignments consistent with the training rows. The corrected assignment-set reading is:

| arm | train-consistent assignments | selected orientation | mixed selected | changed selected | unchanged selected |
|---|---:|---|---:|---:|---:|
| exposure_only | 16 | none | n/a | n/a | n/a |
| heldheld_only | 2 | none | n/a, range 0-1 | n/a, range 0-1 | n/a |
| aligned_state_bridge | 1 | true | 1.000 | 1.000 | 1.000 |
| inverted_state_bridge | 1 | inverted | 0.000 versus true labels | 0.000 versus true labels | 1.000 |
| neutral_decoupled | 2 | none | n/a, range 0-1 | n/a, range 0-1 | n/a |
| mixed_event_bridge | 1 | true | 1.000 | 1.000 | 1.000 |

This establishes that the repaired text surface represents the intended role coordinate anchor and state probe identifiability structure when a reusable argument-slot interface is available. It also clarifies the scientific obstacle: the remaining question is whether a data-limited learned model can induce that slot interface and use sparse orientation evidence, not whether the generated rows have the right algebra.

### Learned runs and resource outcome

The full pretrained DeBERTa pilot submitted as `s278_t23_tool1` failed with CUDA OOM because GPU0 had essentially no free memory. This is a resource conflict, not a mechanism result. The random-init full-DeBERTa pilot submitted as `s278_t23_tool2` was deliberately cancelled after it consumed long H100 time and underfit the first two arms: heldheld_only train 0.4926 with all main evals 0.500, aligned_state_bridge train 0.49182 with all main evals 0.500. Continuing the same run was not a minimum-cost route.

A CPU char-GRU raw-text probe timed out before completion and did not supply a bridge-orientation result. Partial stdout showed underfitting and no aligned-vs-inverted orientation separation on mixed readout: heldheld train 0.693/mixed 0.512, aligned train 0.779/mixed 0.492, inverted train 0.775/mixed 0.500. This suggests a generic small sequence learner is not enough under these settings; it does not test pretrained frame inheritance.

## static slot confounded factorial and balanced budget preparation macro update

companion analysis message #348 reports that their breadth/RoBERTa backlog makes broad compact companion form weak: DeBERTa MAX V-B exEntity5 is negative over 10M-80M and near zero late, and RoBERTa view-clean is negative late on exEntity/cheap6. Their live macro variable is now fixed-budget register substitution, not broad compact companion form. They queued position-matched child/subtitle-removal and adult-prose-removal arms that admit the identical low-rho FineWeb source+view block. A flat mixed-quarter result can therefore be cancellation between removed-register effects, while divergence between register arms would support removal-side dependence.

This strengthens decision not to reopen compact/source-correspondence from aggregate Entity movement. The micro route remains focused on coordinate identifiability plus learned induction of a reusable role-slot interface; the macro route, if it revives, must do so through register-substitution evidence rather than through the closed conservation-failing Entity aggregate.
