# clean init and attention execution plan — clean-init recheck and raw-name attention boundary

## Load-bearing result: central supplied-harness mechanism survives the contamination concern

raw name binding construction and design found a sequential reuse bug in `raw_name_binding_probe.py`: within a condition, `bs=-1` trained from the already-trained `bs=+1` model. Because the gauge-transport result depends on bridge-sign comparison in a nonconvex learner, clean init and attention execution plan reran the minimal supplied-harness sign pair with explicit untouched-initialization auditing rather than assuming that the earlier result was path-independent.

The clean recheck used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/representation_and_objectives/training/scripts/clean_gauge_from_init.py \
  --out experiments/archive/representation_and_objectives/data/clean_gauge_from_init \
  --models shared_trunk untied --bridge-signs 1 -1 --seeds 29000 \
  --epochs 220 --device cuda --print-every 55
```

Key files:
- script: `training/scripts/clean_gauge_from_init.py`
- result summary: `data/clean_gauge_from_init/clean_gauge_from_init_summary.md`
- hash audit: `data/clean_gauge_from_init/clean_init_audit.json`
- combined analysis: `data/clean_attention_analysis/clean_attention_analysis.md`

The initialization audit shows both stored condition initializations stayed unchanged after all bridge-sign cells:

| condition | init hash prefix | untouched after all runs |
|---|---|---:|
| shared_trunk | `ade1a77b34fadeeb` | True |
| untied | `ade1a77b34fadeeb` | True |

Central readout:

| model | bs | train state | train cmp | graph same | pair-both graph | unchanged | held-held closure | mixed acc | mixed margin | graph mean d_e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_trunk | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +13.809 | +2.560 |
| shared_trunk | -1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | -13.809 | -0.105 |
| untied | +1 | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 0.500 | +2.833 | +6.110 |
| untied | -1 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | +2.833 | +7.703 |

The row-paired saved-output analysis confirms the mechanism-relevant distinction:
- shared_trunk graph-transfer changed rows: opposite-sign fraction = 1.0 over 192 h1/h3 rows; direct-anchor changed rows also reverse; unchanged rows remain zero/same;
- untied graph-transfer changed rows: opposite-sign fraction = 0.25 and graph `d_e` remains same-signed on average; direct anchors flip locally but the graph coordinate is not transported to state decisions.

Thus the causal gauge experimental design/291 central supplied-harness mechanism is reestablished under cloned untouched initialization: a shared learned representation permits sparse anchor orientation to propagate through the comparison graph, while an untied comparison pathway can fit the local evidence without carrying the anchor-selected coordinate into state readout.

## Parallel attention pilot: candidate matching becomes more learnable, but raw-name transfer is not solved

The raw-name attention pilot used:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/representation_and_objectives/training/scripts/revision_292b_raw_name_attention_probe.py \
  --out experiments/archive/representation_and_objectives/data/raw_name_attention_primary \
  --conditions tied shared_trunk untied --bridge-signs 1 -1 --seeds 29300 \
  --epochs 220 --device cuda --print-every 55
```

The job timed out after five saved cells: tied ±, shared_trunk ±, and untied +. This is enough to read the immediate boundary because the + sign tied and shared_trunk cells reached full train fit but still did not reproduce held-name behavior.

Key files:
- script: `training/scripts/revision_292b_raw_name_attention_probe.py`
- attention smoke audit: `data/attention_smoke_audit/attention_smoke_audit.md`
- partial primary outputs: `data/raw_name_attention_primary/`
- error analysis: `data/attention_error_structure/attention_error_structure.md`

CPU smoke audit established the mechanics of the attention input:
- exactly one `<QRY>` separator per candidate sequence;
- nonempty event and query masks;
- candidate queries produce different untrained scores;
- comparison scoring runs.

Train/eval overview from saved outputs:

| run | train state | train cmp | graph same | unchanged | held-held closure | mixed acc |
|---|---:|---:|---:|---:|---:|---:|
| tied + | 1.000 | 1.000 | 0.844 | 0.812 | 0.750 | 0.777 |
| tied - | 1.000 | 0.979 | 0.312 | 0.812 | 0.578 | 0.305 |
| shared_trunk + | 1.000 | 1.000 | 0.688 | 0.812 | 0.547 | 0.539 |
| shared_trunk - | 1.000 | 0.948 | 0.531 | 0.812 | 0.422 | 0.492 |
| untied + | 1.000 | 0.500 | 0.469 | 0.812 | 0.500 | 0.500 |

Interpretation:
- Query attention fixes part of the mean-pooled GRU binding bottleneck: tied and shared_trunk no longer stay near 0.5 comparison fit in the + sign cells, and shared_trunk reaches 0.948 on the - sign by 220 epochs.
- This is not raw-name gauge transport. Even when train state and train comparison reach 1.0, held-name state and comparison behavior remains far below the supplied-harness fingerprint. Shared_trunk + has graph_same 0.688, held-held closure 0.547, mixed 0.539, and unchanged 0.812, not 1.0. Tied + is stronger but still far from clean transport and does not preserve unchanged/static rows perfectly.
- Therefore the current failure is not merely incomplete training of the negative sign. It includes failure of variable-like candidate/name generalization to held names under this attention architecture. The raw-name result marks a candidate-binding/generalization boundary, not a successful naturalization of the shared-coordinate mechanism.

## Architectural caveat

In `revision_292b_raw_name_attention_probe.py`, `shared_trunk` shares the GRU trunk but not the attention projections after initialization: separate `QueryAttentionScorer` modules deep-copy the same `W_q/W_k` but then train independently. A future positive result in this architecture would show that a shared recurrent representation can feed learned matching, not that the matcher itself is shared. This did not become the main issue in clean init and attention execution plan because held-name transfer was already poor under full train fit.

## Current scientific state

The controlled principle remains valid but bounded:

> Sparse absolute anchors can orient a graph-connected relative coordinate and transport that orientation to unobserved state decisions when the anchor and comparison tasks share the representation carrying the coordinate. Untied pathways can fit local evidence while failing to transport the oriented coordinate.

clean init and attention execution plan strengthens this by removing the possibility that the central supplied-harness sign-pair was an artifact of sequential reuse.

The boundary is equally important:

> Supplying a query-attention matcher over raw character names is not enough to make the mechanism variable-like over held identities. Learned candidate binding remains a separate bottleneck from gauge transport.

The next research should not spend more GPU merely completing the missing untied negative attention cell. The decisive issue is not that one cell is missing; it is that full-fit tied/shared_trunk positive cells already fail held-name generalization. The next useful work is to design a cleaner learned-binding interface or architecture test that separates (i) identity matching on held names, (ii) comparison-graph closure on held names, and (iii) transport of the anchored coordinate. One promising direction is a symmetric span-pair or pointer-style matcher with shared parameters across state and comparison tasks, plus explicit fresh-renaming and train/eval name-split probes, before attempting any BabyLM-scale intervention.
