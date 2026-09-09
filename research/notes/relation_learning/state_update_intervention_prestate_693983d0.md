# state update intervention prestate: State-Update Intervention — Updated Pre-Result Predictions

Updated from shuf ordinary heldout prestate with three corrections:
1. Plain-use cue removal (earlier analysis): temporal/persistence/change markers rejected
2. Displacement interpretation branch
3. State-type mismatch interpretation for flat Entity (earlier analysis)

Written BEFORE any training or evaluation results.

## Intervention Design

**Name**: STATE_UPDATE_PLAINUSE — cue-controlled entity-state-update three-part
companion packets.

**Construction**: For each accepted Qwen selected original, Qwen3.5-9B generates
three same-row parts:
1. **Source**: the original sentence stating an entity's state (preserved from
   the inherited Qwen pair).
2. **Update**: a plausible event changing one entity's state/location/property/role.
3. **Use**: a sentence using the current state in plain present-tense form, with
   no temporal, persistence, or change markers (still/remains/now/new/etc.).

**Cue removal**: The earlier analysis validator rejects use sentences containing cue families:
still, remain, continue, stay, keep, retain, maintain, unchanged, today, now,
current, new, no longer, again/later, former. This prevents position-cue readout
that would make a null Entity outcome undecidable.

**Packet types** (balanced halves):
- **UPDATED_USE**: update the target entity, then use its NEW state.
- **UNCHANGED_DISTRACTOR_USE**: update a DIFFERENT entity, then use the target
  entity's ORIGINAL state unchanged.

**Validated geometry** (earlier analysis pilot):
- Zero use-source LCS ≥ 6; zero content-Jaccard ≥ 0.8
- Use-source content Jaccard mean ~0.19, median ~0.15
- Zero detected use-sentence temporal/persistence/change cues
- Pilot yield ~24% (48 per type balanced from 256 prompts per type)

**Generator**: Qwen3.5-9B (same provenance as inherited COMPACT_EXPERIENCE Qwen block).

## Stream Materialization

**Materializer**: Qwen-row-aware (`qwen_row_materializer.py`).
- Targets only `qwen_pair_packed` rows by example_id matching to COMPACT_EXPERIENCE metadata.
- Verifies row text reconstruction from pair originals/rewrites.
- Replaces `original + " " + rewrite` with `original + " " + update + " " + use`.
- Tokenizes every converted row; reverts if > 256 tokens.
- Absorbs net word change in common filler at end of each pass.
- Preserves all non-target rows byte-for-byte.
- Records conversion counts, overflow counts, word changes, SHAs.

**Base stream identity**: compact-view-reinvest 100M SHA
`3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`,
647,400 rows (10 passes × 64,740), Qwen pair block 1,656,800 words per 10M.

**Launcher identity**: verified against earlier analysis reference trace with
`lr_total_steps=2529` and 4M-word reproduction (launcher_identity_check.py).

## Comparison Targets

| Baseline | Overall(AoA0) | Entity | Source |
|---|---|---|---|
| chck_82M (public) | ~41.94 | 28.31 | frontier_consolidation challenge submission |
| chck_84M (clean) | 42.0189 | — | frontier_consolidation earlier analysis |
| compact_view_reinvest | 42.0868 | 33.42 | frontier_consolidation earlier analysis seed43022 |
| coherent86 alpha0.75 | 42.1210 | — | frontier_consolidation earlier analysis (adapter-stage) |

Seed43122 base Entity: 25.77 (vs seed43022's 33.42). Within-seed comparison only.

## Pre-Training Predictions

### Entity column (primary target)

**With balanced UPDATED_USE + UNCHANGED_DISTRACTOR_USE packets**:
- **Zero relevant updates (rel_eq0)**: UP. Distractor packets directly train
  retention of the queried entity's state while other entities change.
- **Deep relevant updates (rel_ge3)**: UP. Updated-use packets directly train
  supersession. This is the relation exact recurrence harms and restatement helps.
- **Aggregate Entity**: UP but magnitude is seed-unstable.

### Other columns
- **Supplement**: FLAT or NON-DEGRADED.
- **BLiMP**: FLAT.
- **SuperGLUE**: FLAT or SLIGHT UP.
- **Reading, EWoK, COMPS**: FLAT within seed noise.

## Interpretive Branches

### Branch A: Held-out margins install, Entity rises
The desired outcome. Relation-typed practice transfers to benchmark format.
Proceed to second seed, then coherent replay + official evaluation.

### Branch B: Held-out margins install, Entity FLAT
**Two candidate explanations (not exclusive)**:

**B1. State-type / register mismatch**. The accepted state-update packets
overwhelmingly contain role/identity/affiliation changes of people and places,
while Entity emphasizes containment/location-like states. By the relation-learning
principle, transfer follows the practiced target relation and is amplified by
register match. A flat Entity with installed held-out margins would indicate
state-type mismatch. Follow-up: natural sources whose states are locations and
possessions (narrative, CHILDES text), not benchmark templates.

**B2. Displaced-ALN contribution**. The arm replaces inherited ALN companions
(COMPACT_EXPERIENCE Qwen aligned-restatement pairs), and the packed ALN pairs carry Entity
and Supplement value: dispersing them cost ~2.7 Entity points and ~2.0 Supplement
points at one seed (COMPACT_EXPERIENCE SEP result). So the replacement arm measures:

    net effect = packet gain − displaced-ALN contribution

A flat Entity with installed state margins may mean the packet gain matches or
undercuts the displaced ALN value on the Entity column. The follow-up that
separates displacement from state-type mismatch is **augmentation**: keep the
rewrite, append update and use, absorb the extra words from filler. This preserves
the ALN relation while adding the state-update relation.

### Branch C: Neither margins nor Entity improve
The relation-retargeting hypothesis is falsified at this dose/register gap.
Reduce conversion fraction or revert to compact-view-reinvest baseline.

## Decision Rules

1. Screen with cheap7 + Entity stratification + held-out state margins.
2. If Branch A → second seed → coherent replay → official evaluation.
3. If Branch B → diagnose B1 vs B2:
   - If packet gain is positive on held-out but negative on Entity → try
     augmentation (keep ALN + add state-update) before reducing dose.
   - If state-type labels show location/possession targets install better
     than role/identity → prepare location/possession generation.
4. If Branch C → reduce conversion dose or revert.
5. Two seeds or matched grid required before any benchmark claim.
6. Compare at base level first (chck_84M/82M under Overall AoA0), then
   coherent replay + alpha for any base-level winner.
7. A null Entity result at small dose (~2K pairs, ~1% of stream) is weak
   evidence about the general relation; held-out margins remain interpretable.

## State-Type Taxonomy (for later labeling)

Label UPDATED held-out packets for genuine supersession versus redescriptive shifts:
- **Location/containment**: entity moves to/is placed in a new location
- **Possession**: entity gains/loses/transfers an object
- **Role/identity/affiliation**: entity changes role, title, membership
- **Property/attribute**: entity gains/changes a physical or abstract property
- **Status/condition**: entity changes state of being (alive/broken/closed/etc.)

Stale-state margins are undefined for redescriptive shifts (no competitor state).
Report by subset.

## Provenance

- Prompt build: `scripts/build_plainuse_prompts.py`
  - Prompt SHAs: UPDATED `e5d2a021...`, DISTRACTOR `88a45570...`
- Generator: Qwen3.5-9B (full runs)
- Validation: `scripts/validate_plainuse_outputs.py`
- Materializer: `scripts/qwen_row_materializer.py`
- Identity check: `scripts/launcher_identity_check.py`
- Training launcher: `scripts/launch_state_update_training.py`
- Second-seed base: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M`
