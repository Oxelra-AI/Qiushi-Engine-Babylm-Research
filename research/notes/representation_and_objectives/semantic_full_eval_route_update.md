# semantic full eval route update semantic-view full evaluation and source-breadth route update

## Selected full evaluation result

The `chck_80M` semantic-view endpoint selected by the completed no-AoA trajectory was completed with official-style SuperGLUE and AoA. The evaluation is submit-ready in the local sense: complete zero-shot/Reading, SuperGLUE, and the 19-step strict-small AoA ladder.

| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_view_treatment__chck_80M | 40.1399 | 66.26 | 61.74 | 49.51 | 18.25 | 52.20 | 37.135 | 68.3595 | 7.805 | 0.0000 |
| original_packet_local__chck_80M | 38.5540 | 65.97 | 63.65 | 48.24 | 16.15 | 51.60 | 38.080 | 68.1796 | 7.370 | -12.2539 |

Same-endpoint treatment minus packet-local: Overall +1.5860, BLiMP +0.29, Supplement -1.91, EWoK +1.27, Entity +2.10, COMPS +0.60, GlobalPIQA -0.945, SuperGLUE +0.1799, Reading +0.435, AoA +12.2539.

## Interpretation

Same-source generated views are mechanistically real, not noise: they improve Entity/EWoK/COMPS and avoid the packet-local control's negative AoA at the selected endpoint. But they are not a SOTA endpoint. The treatment Overall is 40.1399, below the public strict-small leader 41.8 and below the inherited COMPACT_EXPERIENCE clean-Qwen reference 41.3443.

Against the public leader, the selected treatment remains far behind on EWoK (-6.56), Entity (-10.20), COMPS (-1.37), GlobalPIQA (-2.535), and SuperGLUE (-1.43), while better on Supplement (+5.73) and Reading (+2.385). This points back to missing source breadth/entity coverage rather than more same-source rewriting alone.

## Next active experiment

Launched managed task `s14_t33_tool1` using `experiments/archive/representation_and_objectives/training/scripts/wait_train_eval_fineweb_seqsafe96.sh`. It waits for genuinely usable GPUs, trains the audited cached FineWeb seqsafe96 source-breadth treatment/control pair, and then runs the paired no-AoA trajectory. Expected delta summary: `experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json`.

The FineWeb contrast should decide whether broad factual source replacement moves the remaining EWoK/Entity/COMPS/GlobalPIQA deficit. Only after that should relation-preserving rewrites be added as a third arm.

JSON: `experiments/archive/representation_and_objectives/data/semantic_full_eval_route_update/semantic_full_eval_route_update.json`
