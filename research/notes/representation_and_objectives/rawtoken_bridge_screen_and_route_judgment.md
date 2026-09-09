# rawtoken bridge screen and route judgment — Raw-token bridge screen for compositional entity-event-state memory

## Scientific Motivation was needed

compositional update test design repaired the counterbalanced corpus and showed a high integration ceiling: with hard template-derived positions, a small learned EntityMemory module can solve held entity×transition recombination better than vanilla. But that learned arm was still given:

- entity slot positions,
- initial-state token positions,
- event verb and actor token positions,
- query entity coordinates.

Natural EWoK/Entity contexts provide none of these. A frozen-DeBERTa + memory run using those regex coordinates would therefore test whether a handcrafted synthetic interface can be attached to DeBERTa, not whether the mechanism bridges to natural conditional-reversal items. The strategist's constraint is correct: before natural transfer can be interpreted, the interface must learn entity/event/query assignment from raw tokens.

## Work performed

I built three metadata-free raw-token interfaces and ran them as cheap H100 screens before any frozen-DeBERTa natural-transfer training:

1. `rawtoken_entity_memory.py`: soft slot and event discovery over all token positions, using only `input_ids`, `attention_mask`, and the known `<mask>` position.
2. `lexical_raw_entity_memory.py`: lexical-identity grouping of repeated raw token ids as candidate entity cells, with learned state/event aggregation.
3. `lexical_recurrent_memory.py`: left-to-right same-token recurrence, where each raw token occurrence updates all same-id cells in sequence order.

I also built two data supports:

- `data/rawtoken_paraphrase_eval/`: adds 384 held paraphrases of the held-recombination quartets. Proper grouping: 96 groups, 4 rows each, all mixed-answer.
- `data/augmented_rawtoken_data/`: adds 576 non-held multi-event training rows so temporal overwrite is trained rather than eval-only.

The EWoK natural bridge panel was materialized at:

- `data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl`
- 7,618 official EWoK items, including 1,756 items in the correctness transition analysis relational domains: social-properties, physical-dynamics, spatial-relations, physical-relations.

## Main numerical results

Full synthesis: `data/rawtoken_bridge_screen/rawtoken_bridge_screen_summary.json` and `.md`.

| run | held recomb | affected | unaffected | held paraphrase | multi-event | write-permutation delta on held | final loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla_base20 | 0.578 | 0.630 | 0.526 | 0.591 | 0.484 | — | 0.104 |
| rawmem_base20 | 0.555 | 0.714 | 0.396 | 0.482 | 0.516 | -0.003 | 0.096 |
| rawmem_noevent_base20 | 0.531 | 0.516 | 0.547 | 0.495 | 0.500 | 0.000 | 0.128 |
| lexmem_base20 | 0.560 | 0.641 | 0.479 | 0.396 | 0.484 | 0.000 | 0.068 |
| lexmem_noevent_base20 | 0.516 | 0.526 | 0.505 | 0.404 | 0.563 | 0.000 | 0.038 |
| vanilla_aug20 | 0.568 | 0.740 | 0.396 | 0.344 | 0.948 | — | 0.051 |
| lexrec_aug20 | 0.500 | 0.516 | 0.484 | 0.490 | 0.906 | +0.003 | 0.040 |

## Scientific interpretation

### 1. compositional update test design remains an integration ceiling, not a natural-bridge result

The metadata-assisted EntityMemory module had a strong held-recombination result because it received hard occurrence roles from the synthetic generator. Removing those roles changes the problem. None of the raw-token versions reached the compositional update test design level, and none showed write-sensitive held recombination.

### 2. The soft-slot raw interface learned an unselective action channel

`rawmem_base20` lifted affected accuracy to 0.714 but collapsed unaffected preservation to 0.396. That means the module learned something like "apply the action result" rather than "update the acted entity and preserve the other entity." Write permutation changed held accuracy by only -0.003, so the learned write route was not identity-keyed.

### 3. Lexical identity alone is not enough

The lexical grouping and lexical recurrence variants did not produce a write-sensitive improvement. `lexmem_base20` and `lexrec_aug20` both have essentially zero write-permutation sensitivity. This shows that raw token identity persistence by itself does not supply the missing event-to-entity-to-query coordination.

### 4. Temporal overwrite is trainable, but not specific to memory

After adding non-held multi-event training rows, ordinary vanilla reaches 0.948 on held multi-event order and lexrec reaches 0.906. Thus temporal overwrite on this synthetic surface is not uniquely solved by the memory mechanism; it is learnable by ordinary attention when exposed in training. The harder failure remains held entity×transition recombination and surface transfer.

### 5. Paraphrase robustness is still weak

No raw memory arm has a robust held paraphrase result. The best paraphrase number on the base data is vanilla itself (0.591). Lexrec improves paraphrase over augmented vanilla (0.490 vs 0.344) but remains near chance and does not solve held recombination. A natural bridge cannot rest on this.

## Consequence for the immediate Phase 2 plan

The planned frozen-DeBERTa + regex-assisted EntityMemory run should not be interpreted as an EWoK/Entity natural-transfer experiment. At most it can measure whether DeBERTa hidden states can support the synthetic integration ceiling when hard positions are supplied. Because the raw-token bridge precondition failed in three variants, launching official EWoK evaluation for the regex-assisted wrapper would risk another isolated toy mechanism.

The next constructive route should rebuild the interface, not scale this one. A stronger candidate must satisfy all of the following scientifically meaningful properties before a fresh Strict-Small trajectory is justified:

- learn entity/event/query assignment from raw token sequences, not template coordinates;
- solve held entity×transition recombination above 0.70 on the counterbalanced substrate;
- preserve the unaffected entity while updating the affected entity;
- survive held paraphrase surfaces;
- handle multi-event overwrite in a way that is not matched by a same-data vanilla transformer;
- move item-level conditional choices in the official EWoK bridge panel, especially the correctness transition analysis relational domains, not merely aggregate EWoK.

## Recommended next research action

A proposed next comparison requires a stronger raw-token compositional-update interface. The likely missing ingredient is not a larger external memory but an assignment-learning objective or architecture that aligns three latent variables from raw text:

1. which mention denotes which entity across time,
2. which event/action changes which entity,
3. which entity the prediction site queries.

Promising directions include contrastive latent alignment over counterbalanced quartets, slot competition regularized by mention co-reference consistency, self-supervised event-to-mention routing losses derived only from answer correctness, and a natural-EWoK scoring head that reads the same latent assignment variables. The current regex-assisted run can remain as an upper integration reference, but it should not drive a natural-transfer claim.

## Addendum — independent_review-suggested hard-coordinate matched control

independent_review correctly noted that rawtoken bridge screen and route judgment's first raw-token screens reject only the tested raw interfaces, not raw-token assignment learning in general. It recommended a matched hard-versus-inferred coordinate ablation. I therefore ran the original earlier analysis/244 hard-coordinate `shared_key_learned` EntityMemory under the same full-data/20-epoch settings.

Results are decisive:

| run | held recomb | affected | unaffected | quartet-all | held paraphrase | multi-event | write-permutation held | write-permutation paraphrase | write-permutation multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hardcoord_shared_base20 | 0.818 | 1.000 | 0.635 | 0.635 | 0.919 | 0.531 | 0.182 | 0.018 | 0.469 |
| hardcoord_shared_aug20 | 0.849 | 1.000 | 0.698 | 0.698 | 0.911 | 1.000 | 0.151 | 0.099 | 0.000 |

Thus the synthetic integration ceiling persists under the same 20-epoch/full-data/paraphrase setting and collapses when writes are permuted. Compared with the best raw-token base held score (0.560), hard coordinates add +0.258; compared with lexrec on augmented data, hard coordinates add +0.349 held recombination, +0.422 paraphrase, and +0.094 multi-event. This directly identifies the current missing problem as **latent occurrence-role assignment**: the memory update/read mechanism is powerful when supplied roles, but the raw-token interfaces tested here do not discover which mentions are entities, which event changes which entity, and which entity the prediction site queries.

Final synthesis files:

- `experiments/archive/representation_and_objectives/data/final_bridge_synthesis/final_bridge_synthesis.json`
- `research/documents/representation_and_objectives/data/final_bridge_synthesis/final_bridge_synthesis.md`

Updated route decision: the regex-assisted frozen-DeBERTa run remains useful only as an integration ceiling. Natural-transfer interpretation and fresh Strict-Small training should wait until a raw-token latent-assignment mechanism succeeds on held recombination, paraphrase, temporal overwrite beyond vanilla, write/read permutation sensitivity, and the official EWoK bridge panel.
