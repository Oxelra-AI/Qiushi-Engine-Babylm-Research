# shuf ordinary heldout: State-Update Intervention — Corrected Pre-Result Predictions

Written BEFORE any training or evaluation results.

## Intervention Design

**Name**: STATE_UPDATE — entity-state-update three-part companion packets.

**Construction**: For each accepted Qwen selected original, a Qwen3.5-9B structured
generation produces three same-row parts:
1. **Source**: the original sentence stating an entity's state.
2. **Update**: a plausible event changing one entity's state/location/property/role.
3. **Use**: a sentence requiring current-state knowledge of the target entity.

**Packet types** (approximately equal halves):
- **UPDATED_USE**: update the target entity, then use its NEW state.
  This practices supersession of the queried entity.
- **UNCHANGED_DISTRACTOR_USE**: update a DIFFERENT entity, then use the
  target entity's ORIGINAL state unchanged.
  This practices retaining the queried entity's state while other entities change
  — the training analogue of zero-relevant-update Entity items.

**Natural domain only**: exemplars use natural domains (people/locations/roles/
properties), NOT Entity-benchmark format (no box/basket/marble/container).
Register match amplifies practiced-relation effects (established in the relation-learning study),
so benchmark-format generation would be indistinguishable from teaching to the test.
Transfer from natural training sentences to the benchmark's register is the test.

**Generator**: Qwen3.5-9B, the generator recorded for the inherited Qwen-rewrite submissions (`qwen_rewrites_full`).

**Word budget**: companions replace the Qwen rewrite block at matched word count.
Unconverted originals keep their existing Qwen restatement companions as fallback.
The rest of the 100M stream (common filler, protected rows, reinvest block) is unchanged.

**Stream accounting**: exact compact-view-reinvest 100M base; packed pair-row
topology preserved; legal exposure accounting preserved.

## Comparison Targets

| Baseline | Overall(AoA0) | Entity | Source |
|---|---|---|---|
| chck_82M (public) | ~41.94 | 28.31 | frontier_consolidation challenge submission |
| chck_84M (clean) | 42.0189 | — | frontier_consolidation earlier analysis |
| compact_view_reinvest | 42.0868 | 33.42 | frontier_consolidation earlier analysis seed43022 |
| coherent86 alpha0.75 | 42.1210 | — | frontier_consolidation earlier analysis (adapter-stage) |

Entity deficit at chck_82M vs leader: ~-5.75 points.

## Pre-Training Predictions

### Entity column (primary target)

**With both UPDATED_USE and UNCHANGED_DISTRACTOR_USE packets**:
- **Zero relevant updates (rel_eq0)**: UP. The distractor packets directly train
  retention of the queried entity's state while other entities change. This should
  produce stronger zero-update discrimination than the current restatement practice,
  which does not train state retention under distractor updates.
- **Deep relevant updates (rel_ge3, rel_updates_5)**: UP (higher confidence).
  The updated-use packets directly train supersession of the queried entity's state
  after relevant updates. This is the relation that the relation-learning evidence shows
  exact recurrence harms and restatement helps.
- **Aggregate Entity**: UP, but magnitude is seed-unstable (3-4× compression
  across three seeds in this study). Aggregate is NOT the readout.

**Without distractor packets (hypothetical future arm)**:
- Deep-update: still UP.
- Zero-update: AT RISK, because the model would not practice retaining a queried
  entity's state when only other entities change.

### Other columns
- **Supplement**: FLAT or NON-DEGRADED. The state-update companions preserve
  natural reading-level variation and real-world content diversity.
- **BLiMP**: FLAT. Grammatical competence comes from the broader corpus, not
  from the ~60% of Qwen rows carrying companion packets.
- **SuperGLUE**: FLAT or SLIGHT UP. State-update practice may marginally help
  entailment-like discrimination but is not targeted at it.
- **Reading, EWoK, COMPS**: FLAT within seed noise.

## Decision Rules

1. **Screen with cheap7 + Entity stratification** before full official eval.
2. If Entity rises with Supplement/SuperGLUE non-degraded → launch second seed.
3. If Entity rises but Supplement or SuperGLUE declines → reduce conversion
   fraction (half-dose) rather than launching second seed.
4. If Entity does not improve → revert to compact-view-reinvest baseline;
   the relation-retargeting hypothesis is falsified at this dose/register gap.
5. **Two seeds or matched common grid required** before any benchmark claim.
6. **Compare at base level** to chck_84M/82M under Overall(AoA0), then plan
   coherent-replay stage rerun for any base-level winner.

## Provenance

- Prompt extraction: `scripts/build_9b_state_use_prompts.py`
- Generator: Qwen3.5-9B
- Validation: `scripts/validate_9b_outputs.py`
- Training launcher: `scripts/launch_state_update_training.py`
  (corrected from state update intervention prestate — verified tokenizer path and CLI flags)
- Second-seed base: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M` (already trained)
