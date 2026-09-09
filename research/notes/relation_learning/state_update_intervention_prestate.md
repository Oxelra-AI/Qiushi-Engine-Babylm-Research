# state update intervention prestate: State-Update Pair Intervention Pre-Result Predictions

Written BEFORE any training or generation results.

## Intervention Design

**Name**: STATE_UPDATE — entity-state-update companion pairs replacing Qwen restatement pairs.

**Rationale**: The established relation-typed readout principle says competence follows the
practiced target relation. The Qwen restatement block practices *corresponding restatement*
(same content, different form). Entity tests *state-update tracking* — after operations
change an entity's state, can the model use the current state rather than the original?
ALN's +3.6 to +4.2 nat advantage on source-recurring changed-form targets is the
largest measured effect; it is entailment-shaped. If the companion block instead practices
entity-state-update (source states a fact, companion supersedes it with a changed state),
the principle predicts Entity improvement under the practiced relation.

**Construction**: Take the same 37,594 selected official originals used in the Qwen block.
For each, generate a state-update companion: a sentence that mentions the same entity
but changes its state, property, or location. Use Qwen3-8B under the same generation 
accounting. Pack into the same ~160-word rows, fill with the same official filler, 
create matched 10M/100M streams. Train the exact same scale1.75 adapter DeBERTa recipe.

**Matched control**: Compare to the existing chck_82M and chck_84M under cheap7/Overall(AoA0),
since the Qwen restatement block already exists as the treatment standard.

## Per-Benchmark Predictions (Signs Only)

| Column | Prediction | Reasoning |
|---|---|---|
| Entity | **UP** at both zero and deep relevant updates | Directly practices state tracking/supersession |
| Supplement | **FLAT or slight DOWN** | Does not practice agreement/semantic-structure relation |
| BLiMP | **FLAT** | Not a source-conditioned task; no targeted relation |
| EWoK | **FLAT or slight UP** | Some entity-state reasoning overlaps with world knowledge |
| COMPS | **FLAT** | Comparative reasoning is not directly practiced |
| GlobalPIQA | **FLAT** | Noisy low-count column; not targeted |
| Reading | **FLAT** | Not a source-conditioned task |
| SuperGLUE | **FLAT or slight UP** | Some entailment tasks use premise→hypothesis state inference |
| Overall(AoA0) | **UP if Entity gain > any Supplement loss** | Entity has largest deficit (-5.75 from leader) |

## Stop Rules

1. If generation quality is too low (<30% acceptance rate after filtering), the
   originals are not entity-state-amenable enough; abort and consider a different
   source selection.
2. If pilot evaluation shows Entity DOWN or no change at one seed: the relation
   is not practiced as predicted; do not run second seed.
3. If Entity UP but Supplement/SuperGLUE significantly DOWN: net column tradeoff;
   report as a relation-type sensitivity result, not an improvement.
4. Two seeds or the matched common grid required before any claim of improvement.

## Legality

- Same official originals as the existing Qwen block (already ruled legal)
- Generated companion words under the same Qwen-model accounting
- Same total 10M/100M word budget
- Same tokenizer, model architecture, training recipe
- No data sources beyond the legal Strict-Small pool + LLM-generated companions

## Comparison Targets

- Base level: chck_82M (Overall 41.94), chck_84M (projected Overall 42.02)
- Full stack: coherent86 alpha0.75 (Overall 42.12) — only if full stack is reproduced
- Qwen restatement: clean-Qwen-aligned (Overall ~41.34 at chck_100M without adapter)

## COMPACT_EXPERIENCE Masking Evidence (from relation arm state margin tu integration)

COMPACT_EXPERIENCE showed masking geometry is NOT the mechanism: concentrating masks on specific
targets did not improve over random WWM. The benefit is data composition. Therefore
this intervention changes the data composition (relation type of companions) rather
than the masking pattern.

## COMPACT_EXPERIENCE Contextual One-Pair Evidence

Contextual one-pair (embedding pairs in natural official context instead of packed
pair rows) lost Entity -2.68 and Supplement -2.03. Therefore this intervention
preserves the packed pair-row topology: source + update companion in the same
packed row, just as the current Qwen restatement pairs are packed.
