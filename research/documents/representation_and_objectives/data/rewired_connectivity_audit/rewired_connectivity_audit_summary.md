# earlier analysis independent audit of rewired connectivity substrate

This audit compares the model-visible connected and rewired training files before any GPU run.

## Main checks

| arm | rows equal | common exact | bridge exact | held-held delex+label | token unigrams | name role/label | all-pair degree sequence | held-held exact text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | True | True | True | True | True | True | False | False |
| inverted_state_bridge | True | True | True | True | True | True | False | False |
| heldheld_only | True | True | True | True | True | True | True | False |

## Topology readout

| arm | connected bridge∩heldheld exact pairs | rewired bridge∩heldheld exact pairs | all-model pair degree sequence connected | all-model pair degree sequence rewired |
|---|---|---|---|---|
| aligned_state_bridge | ['Ava::Jonas', 'Keira::Milo', 'Lena::Pavel', 'Mira::Omar', 'Nia::Felix', 'Noel::Iris', 'Rina::Tomas', 'Sara::Theo'] | [] | [128, 128, 128, 128, 128, 128, 128, 128, 224, 224, 224, 224, 224, 224, 224, 224] | [160, 160, 160, 160, 160, 160, 160, 160, 192, 192, 192, 192, 192, 192, 192, 192] |
| inverted_state_bridge | ['Ava::Jonas', 'Keira::Milo', 'Lena::Pavel', 'Mira::Omar', 'Nia::Felix', 'Noel::Iris', 'Rina::Tomas', 'Sara::Theo'] | [] | [128, 128, 128, 128, 128, 128, 128, 128, 224, 224, 224, 224, 224, 224, 224, 224] | [160, 160, 160, 160, 160, 160, 160, 160, 192, 192, 192, 192, 192, 192, 192, 192] |
| heldheld_only | [] | [] | [128, 128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 160] | [128, 128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 160] |

## Scientific reading

The corrected substrate fixes the connectivity substrate construction and audit defect at the level that matters for the known filler-support confound: bridge supervision is exact-identical; common seen-coordinate exposure is exact-identical; token unigrams, delexicalized row templates, individual-name role/label counts, and pair-degree sequences are matched.  The held-held exact sentence multiset is intentionally not identical because the intervention is an edge rewiring of which name co-occurrences carry held-held constraints.  Thus a future learned comparison can be read as a topology/co-occurrence connectivity test, not as a different bridge dose or different individual-name role exposure.  If an even stricter same-exact-sentence contrast is desired, it would require label rewiring or contradictory decoy rows and should be treated as a different design.

## Files
- full audit JSON: `experiments/archive/representation_and_objectives/data/rewired_connectivity_audit/rewired_connectivity_audit.json`
