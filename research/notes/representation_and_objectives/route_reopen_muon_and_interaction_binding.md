# muon switch 40m globalpiqa margin — route reasoning after fullctx aux budget audit relation update and Muon harvest

## What changed scientifically

fullctx aux budget audit weakened the earlier local relation-objective line. Existing failing checkpoints already prefer attested pivot–consequence pairings and attested targets over matched cross-event alternatives. That means the missing capability is not ordinary local compatibility with a true pivot or a true target. The hard behavior is two-context/two-target binding: the same alternatives must be bound differently under competing contexts, with target priors and lexical fit cancelled.

Therefore another positive-vs-corrupted sentence loss is not scientifically well linked to the open failure. A relation objective remains worth constructing only if it supplies an unsaturated four-cell interaction signal from legal data, e.g.

\[
  \Delta = s(C_a,T_a) + s(C_b,T_b) - s(C_a,T_b) - s(C_b,T_a),
\]

where both targets appear once as positive and once as negative, both contexts appear once with each target, and scoring can be done from the ordinary WWM forward pass whenever target lengths match. This cancels target main effects and focuses on conditional binding rather than attested-sentence fit.

## Why Muon-switch artifacts became the cheapest direct route test

matched-decay Muon experiments are not another data-layout variant. They test whether hidden-matrix update geometry can create broader early representations that AdamW did not reach from the same legal compact substrate. Existing facts:

- 20M matched-decay Muon cheap7 41.0236 vs AdamW/aoa mincontext discrepancy audit 39.6636, delta +1.3600.
- 20M columns: BLiMP +1.46, Supplement +2.65, EWoK -0.03, Entity +2.25, COMPS +0.19, GlobalPIQA +4.41, Reading -1.41.
- Mature continuous Muon at 70M only +0.0657 cheap7 vs AdamW; at 80M -0.1607 cheap7 vs AdamW.
- Mature continuous Muon preserves lower MLM loss and broad hidden spectra, but redistributes behavior: GlobalPIQA improves strongly while Entity and Reading fall, and EWoK relation/order-sensitive slices are damaged.

The immediate question is whether early Muon can be consolidated by AdamW before this mature redistribution appears. This can be tested from already-created artifacts before launching any new training.

## muon switch 40m globalpiqa margin artifact audit

switch directories are named as 80M screens, but current artifacts are not 80M switch endpoints. The audit is saved in:

- `experiments/archive/representation_and_objectives/data/muon_switch_artifact_audit/muon_switch_artifact_audit.json`
- `research/notes/representation_and_objectives/muon_switch_artifact_audit.md`

Key facts at audit time:

| run | max checkpoint | final logged exposure | final step | final loss | metrics file | switch step | immediate switch loss jump |
|---|---:|---:|---:|---:|---|---:|---:|
| AdamW reference | 100M | 100000000 | 2529 | 2.5526 | yes | none | |
| continuous Muon | 80M | 80000000 | 2024 | 2.3978 | yes | none | |
| Muon20M→AdamW | 50M | 50258212 | 1271 | 2.7521 | no | 506 | +1.6046 |
| Muon40M→AdamW | 50M | 57611364 | 1457 | 2.6884 | no | 1012 | +1.2911 |

The switch itself causes a large transient loss shock because the hidden-matrix AdamW moments are first created at the switch rather than carried through from the Muon phase. Muon20M→AdamW: pre-switch 20-step mean loss 3.4925, first 5 post-switch mean 5.0970, post+6..50 mean 3.7524, post+200..300 mean 3.1497. Muon40M→AdamW: pre-switch 2.8470, first 5 post-switch 4.1381, post+6..50 2.9548, post+200..300 2.7368. Thus a negative switch result would close only this discontinuous handoff, not every possible early-Muon consolidation.

## muon switch 40m globalpiqa margin tools prepared or launched

Prepared result reader:

- `experiments/archive/representation_and_objectives/scripts/muon_switch_40m_eval.py`

It reads checkpoints but writes only under companion analysis. It creates small proxy run directories because custom switch runs lack `scientific_metrics.json`; no companion analysis files are modified. The shared 40M comparison includes continuous AdamW, continuous Muon, Muon20M→AdamW, and Muon40M→AdamW. The managed task is running:

- `s126_t12_tool1`: `python3 experiments/archive/representation_and_objectives/scripts/muon_switch_40m_eval.py --run-all --parallel 2 --gpus 0 1 --timeout 2400`
- outputs expected under `experiments/archive/representation_and_objectives/data/muon_switch_40m_eval`
- summary expected at `experiments/archive/representation_and_objectives/data/muon_switch_40m_eval/muon_switch_40m_summary.json`
- note expected at `research/notes/representation_and_objectives/muon_switch_40m_harvest.md`

Prepared and dry-ran relation-surface wrappers:

- `experiments/archive/representation_and_objectives/scripts/muon_switch_40m_ewok_wrapper.py`
- `experiments/archive/representation_and_objectives/scripts/muon_switch_40m_globalpiqa_wrapper.py`

Both dry-runs reported all four 40M checkpoints ready. The GlobalPIQA hard-rank wrapper was launched as managed CPU work:

- `s126_t20_tool1`: `python3 experiments/archive/representation_and_objectives/scripts/muon_switch_40m_globalpiqa_wrapper.py --modes parallel nonparallel --threads 12`
- outputs expected under `experiments/archive/representation_and_objectives/data/muon_switch_40m_globalpiqa_margin`
- note expected at `research/notes/representation_and_objectives/muon_switch_40m_globalpiqa_margin.md`

EWoK four-cell interaction is ready but not launched, because the broad 40M table and GlobalPIQA hard-rank readout should first tell whether the switch arm is worth the heavier full-EWoK pass. If launched, use the prepared wrapper and write under `experiments/archive/representation_and_objectives/data/muon_switch_40m_ewok_interaction`.

## How to read the pending Muon results

The most informative 40M pattern is Muon20M→AdamW compared with both AdamW40 and continuous Muon40:

- If Muon20M→AdamW is broadly above AdamW40 and not merely another GlobalPIQA-for-Entity/Reading tradeoff, then early Muon plus consolidation is alive. The next work should run EWoK four-cell on all four 40M arms, then decide whether a properly repaired switch handoff should continue to 70/80M.
- If Muon20M→AdamW is close to continuous Muon in the same damaged direction, the switch did not avoid mature Muon redistribution even by 40M.
- If Muon20M→AdamW loses broad columns or shows severe Entity/Reading/EWoK weakness, do not extend these exact switch runs just because more exposure exists. The loss shock then becomes the main evidence: this exact handoff is flawed, and any further optimizer route must first repair moment continuity or update blending.

## If optimizer geometry remains promising

A scientifically cleaner repair would not switch from Muon to AdamW with empty hidden AdamW moments. It should keep AdamW moment estimates for hidden matrices alive during the Muon phase without applying their updates, or gradually blend Muon and AdamW hidden updates over a short transition. This would test consolidation rather than a hard optimizer-state discontinuity. Such a repair should only be built if the 40M behavioral table shows enough retained Muon value to justify new training.

## Relation construction retained, but narrowed

The next relation construction, if pursued, should be a legal two-context/two-target interaction objective, not another single-context attested-target objective. A plausible zero-extra-forward version:

1. Use pvdm compliance and control design legal event labels and the exact allowed compact corpus.
2. Pair two events within the same relation family with equal target token length, similar target frequency bin, and similar event geometry.
3. Use ordinary WWM logits when both event targets are masked in their own rows, or use a same-exposure reference if selected target masking is introduced.
4. Score each row's true target and the paired alternative target at the same masked positions.
5. Optimize or first calibrate the four-cell margin `s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)`.

This is the first relation formulation that directly matches the updated failure statement: competing contexts must bind the same alternatives differently. Before training, run a no-update calibration on standard 80M or a 100M FW arm and require evidence that the four-cell margin is not already saturated and that enough physical/spatial/temporal/comparative pairs survive.

## Current route judgment

The immediate highest-value path is to finish harvesting 40M switch artifacts and relation surfaces, because no new training is needed and the result can decide whether optimizer geometry deserves a repaired mature continuation. The relation route should not be abandoned, but it should wait for a cleaner two-context/two-target construction rather than extending positive-versus-corrupted sentence losses.
