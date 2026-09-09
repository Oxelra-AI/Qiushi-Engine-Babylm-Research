# secondary retention and orientation interference — Secondary retention without direct sparse secondary labels

## Scientific Motivation exists

independent orientation cross and context interface established a fit-verified selective-orientation pattern on the familiar ATP interface: event sparse labels mainly control event readout, and focal-ranking sparse labels mainly control focal-ranking readout. However, the apparent `untouched_state` preservation in independent orientation cross and context interface was not genuine conservation because the secondary ranking query was directly trained as true in every sparse compound arm. secondary retention and orientation interference converts that limitation into a direct measurement.

The scientific question is:

> If a model has already learned event, focal ranking, and secondary ranking facts, and a sparse update later changes event and/or focal-ranking evidence while **withholding all secondary-ranking directional labels**, does the secondary relation remain available, or is it overwritten through the shared coordinate/head/interface?

This is a small pretrained-language bridge measurement, not BabyLM-scale training. It is intended to decide whether a stronger principle can speak about consolidation and untouched-fact retention, or only about selective orientation on directly supervised relation families.

## Implemented protocol

Script: `experiments/archive/representation_and_objectives/training/scripts/secondary_retention_no_direct_supervision.py`.

The script imports the independent orientation cross and context interface/contrastive wording cross and grounding ATP and paired-world machinery and uses the same pretrained DeBERTa endpoint:

`experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`.

Training is staged:

1. **Base phase.** Train on true event, true focal ranking, and true secondary ranking over anchor and train context groups. This establishes a high-margin record interface before sparse updates.
2. **Sparse update phase.** Clone the fitted base for each arm. Update rows use train wording and never include secondary-ranking directional hypotheses. Omitted secondary rows are replaced with neutral mention rows so update row counts and labels remain matched.
3. **Evaluation.** Evaluate true facts before and after update with paired AB-vs-BA contrastive accuracy and signed margins. Positive signed margin means the true AB/BA hypothesis is preferred over its swapped alternative.

Main arms:

- `exposure`: neutral mention exposure only.
- `Etrue_Rtrue`: event true, focal-ranking true, secondary withheld.
- `Eflip_Rtrue`: event flipped, focal-ranking true, secondary withheld.
- `Etrue_Rflip`: event true, focal-ranking flipped, secondary withheld.
- `Eflip_Rflip`: event flipped, focal-ranking flipped, secondary withheld.

Single-relation arms:

- `Etrue_only`, `Eflip_only`: event labels only, no focal or secondary directional labels in update.
- `Rtrue_only`, `Rflip_only`: focal-ranking labels only, no event or secondary directional labels in the ATP compound update; event-only paired exposure is neutralized.

## independent orientation cross and context interface replication collected

The pending independent orientation cross and context interface replication completed successfully, writing to:

- summary: `experiments/archive/representation_and_objectives/data/independent_orientation_cross_fitrepair3/independent_orientation_cross_summary.json`
- markdown: `research/documents/representation_and_objectives/data/independent_orientation_cross_fitrepair3/independent_orientation_cross_summary.md`
- factorial effects: `research/documents/representation_and_objectives/data/fitrepair3_factorial_effects/factorial_effects.md`
- fit-filtered effects: `research/documents/representation_and_objectives/data/fitrepair3_factorial_effects/fit_filtered_factorial_effects.md`

All exposure/Etrue_Rtrue/Eflip_Rtrue/Etrue_Rflip seeds fit; the joint-flip arm had one underfit seed (`Eflip_Rflip`, seed 26301, train_acc=0.5), so raw three-seed means mix a real orientation effect with an optimization non-measurement. Fit-filtering at train_acc>=0.99 leaves seeds 26300 and 26302.

Fit-filtered familiar trainTrain/trainHyp effects:

- Event query: `Delta_E = 0.998 ± 0.002`, `Delta_R = 0.000 ± 0.000`.
- Focal-ranking query: `Delta_E = 0.052 ± 0.010`, `Delta_R = 0.754 ± 0.092`.
- Secondary-ranking query: `Delta_E = 0.070 ± 0.055`, `Delta_R = 0.072 ± 0.053`.

Thus the independent orientation cross and context interface selective event/focal-ranking effect replicates on the familiar interface after excluding the underfit joint-flip seed. Context-side held/directional state wording remains weak; fit-filtered tables show no stable rank transfer on held/dir state contexts.

Important context-control facts from independent orientation cross and context interface replication:

- With train event and non-directional state context, event orientation stays controlled (`Delta_E ≈ 1.000`) while ranking queries stay near chance.
- With non-directional event and train state context, event stays near chance while focal ranking responds to rank orientation (`Delta_R ≈ 0.541` after fit filtering), and secondary moves partly with rank orientation (`Delta_R ≈ 0.200`).

This reinforces that the current positive object is a surface-conditioned coordinate/interface effect, not spontaneous natural state updating.

## secondary retention and orientation interference construction checks

Dry construction with `base_ctx_groups=anchor,train` succeeded:

- `experiments/archive/representation_and_objectives/data/secondary_retention_dry_anchortrain/construction_summary.json`
- base rows: 1120 in the 20/4 dry setting, balanced labels.
- update rows: all main arms have equal row count and exactly balanced labels.

Pilot and fit-repair runs used 40 base ATP worlds, 8 sparse ATP worlds, 40 held ATP worlds, 40 paired-world train/eval worlds, batch size 8, base 6 epochs, and update 8 epochs in the fit-repaired run.

Evidence roots:

- first pilot: `research/documents/representation_and_objectives/data/secondary_retention_pilot/secondary_retention_summary.md`
- fit-repaired main arms: `research/documents/representation_and_objectives/data/secondary_retention_fitrepair1/secondary_retention_summary.md`
- single-relation arms: `research/documents/representation_and_objectives/data/secondary_retention_single_relation1/secondary_retention_summary.md`
- comparison analysis: `research/documents/representation_and_objectives/data/secondary_retention_analysis/secondary_retention_analysis.md`

## Main-arm fit-repaired one-seed result

Base phase fit completely and produced very high familiar-interface margins:

- before update at `atp_trainTrain_trainHyp`: event `1.000 / margin 20.057`, focal `1.000 / 20.294`, secondary `1.000 / 20.673`.
- state-only context before update: focal `1.000 / 20.783`, secondary `1.000 / 20.963`.
- event-only context before update: event `1.000 / 20.530`, secondary is not supported (`0.388 / -0.739`), confirming the secondary readout depends on ranking records.

After sparse update:

| arm | update fit | base-after | trainTrain event | trainTrain focal | trainTrain secondary | interpretation |
|---|---:|---:|---:|---:|---:|---|
| exposure | 1.000 | 0.998 | 1.000 / 16.194 | 1.000 / 16.408 | 1.000 / 16.143 | neutral exposure weakens margins but retains all familiar facts |
| Etrue_Rtrue | 1.000 | 1.000 | 1.000 / 21.506 | 1.000 / 21.779 | 1.000 / 21.972 | true relation update reinforces all familiar facts despite no secondary update labels |
| Eflip_Rtrue | 0.982 | 0.622 | 0.106 / -9.328 | 0.975 / 16.090 | 1.000 / 16.485 | event orientation can be mostly inverted while focal and secondary ranking remain true when rank evidence is true |
| Etrue_Rflip | 1.000 | 0.843 | 1.000 / 25.196 | 0.506 / 0.126 | 0.731 / 3.070 | flipped focal-rank evidence damages secondary ranking despite withheld secondary labels |
| Eflip_Rflip | 1.000 | 0.001 | 0.000 / -13.456 | 0.000 / -14.312 | 0.000 / -14.497 | joint flip overwrites the familiar interface globally |

The important selective contrast is not the joint-flip arm. It is:

- `Eflip_Rtrue`: event margin moves from +21.506 to -9.328 while secondary remains +16.485.
- `Etrue_Rflip`: focal margin collapses from +21.779 to +0.126 and secondary drops from +21.972 to +3.070.

This suggests secondary ranking is protected from event-coordinate change when rank evidence remains correctly oriented, but is vulnerable to rank-coordinate reorientation even without direct secondary labels.

## Single-relation one-seed result

The single-relation arms show that absence of a correct relation anchor during the update can make the staged protocol behave like destructive coordinate/head overwriting.

At `atp_trainTrain_trainHyp`:

| arm | update fit | base-after | event | focal | secondary |
|---|---:|---:|---:|---:|---:|
| Etrue_only | 1.000 | 0.998 | 1.000 / 23.882 | 1.000 / 22.098 | 1.000 / 22.457 |
| Eflip_only | 1.000 | 0.001 | 0.000 / -19.621 | 0.000 / -17.880 | 0.000 / -18.303 |
| Rtrue_only | 1.000 | 0.986 | 1.000 / 15.365 | 1.000 / 19.350 | 1.000 / 19.759 |
| Rflip_only | 0.996 | 0.328 | 0.981 / 7.713 | 0.000 / -14.357 | 0.000 / -13.153 |

A flipped event-only update collapses focal and secondary as well as event; a true event-only update reinforces all three. This cannot be interpreted as a relation-specific event operation. It is a staged-update interference effect unless a replay/joint control proves otherwise.

A flipped rank-only update leaves event largely true but inverts focal and secondary. This is more relation-specific: focal and secondary share the same ranking relation coordinate, so flipping the focal-ranking sparse labels reverses the secondary readout even though secondary labels were withheld.

## Current interpretation

The best current statement is:

1. On a familiar argument-record interface, sparse evidence can selectively orient relation coordinates: event labels and ranking labels control different readout families when both are grounded and fit.
2. A secondary fact is not conserved by default. It survives when the sparse update preserves the ranking coordinate (`Etrue_Rtrue`, `Eflip_Rtrue`), and it is damaged or inverted when the update reverses the ranking coordinate (`Etrue_Rflip`, `Rflip_only`, `Eflip_Rflip`).
3. In a staged update without replay, relation labels can also overwrite broader head/interface behavior. The dramatic `Eflip_only` collapse of all readouts is evidence of this interference, not evidence of a clean event relation changing state.
4. Therefore the result supports a more precise candidate principle: data-efficient reusable knowledge is not just formed by sparse anchors; it also requires **coordinate-protecting consolidation**. A relation coordinate can be reused and protected when new evidence is both grounded and consistently anchored to the existing coordinate. Flipped or unanchored sparse evidence can reorient a shared coordinate and erase facts that depend on it.

This is still a small bridge result. It does not establish event-derived temporal updating, held-state wording transfer, BabyLM-scale improvement, or general architecture invariance.

## Running replication

A three-seed staged no-direct-secondary replication is in progress:

- task: `s264_t19_tool1`
- output: `experiments/archive/representation_and_objectives/data/secondary_retention_full3`
- arms: exposure, Etrue_Rtrue, Eflip_Rtrue, Etrue_Rflip, Eflip_Rflip, Etrue_only, Eflip_only, Rtrue_only, Rflip_only
- purpose: determine whether the one-seed secondary-retention/interference pattern is stable, and which arms fit well enough to interpret.

Do not launch any Strict-Small corpus or official evaluation from this bridge result until the full3 result is collected and a sequential-interference control is built.

## Next scientific separator

The next experiment should separate staged-update forgetting from dataset-level sparse orientation. Two cheap variants are natural:

1. **State-replay update control.** During the sparse update, add a small replay buffer of base focal/secondary ranking rows but omit secondary labels for the sparse update worlds. If `Eflip_only` no longer collapses focal/secondary, the previous collapse is ordinary sequential overwriting; if `Rflip_only` still flips secondary despite state replay, that is stronger evidence of shared rank-coordinate reorientation.
2. **Joint no-secondary sparse training.** Train base and update rows together from scratch, still withholding secondary labels on the sparse update worlds. This tests whether the effect persists without a staged update. It is less directly a consolidation test because base true rows and sparse flipped rows conflict across worlds; success may depend on whether the model memorizes worlds versus reorients a coordinate.

The state-replay control is the better immediate next measurement because it keeps the update setting but protects the already learned interface, allowing event-specific and rank-specific update effects to be separated from broad forgetting.

## Replay-control implementation status

The state-replay separator has been added to `experiments/archive/representation_and_objectives/training/scripts/secondary_retention_no_direct_supervision.py` via:

- `--replay_rows_per_query N`
- `--replay_queries focal_state,untouched_state`

Dry construction was checked at:

- `experiments/archive/representation_and_objectives/data/secondary_retention_replay_dry/construction_summary.json`

For the dry 20/4 setting with arms `Etrue_only,Eflip_only,Rtrue_only,Rflip_only` and `--replay_rows_per_query 64`, the script built 1120 base rows, selected 128 balanced replay rows (focal + secondary), and each update arm had 240 total rows. This is only a construction check; no replay-control training has been run yet. Analysis of the pending fit-filtered full3 results is required before deciding whether to run a one-seed replay pilot using the same 40/8 scale.
