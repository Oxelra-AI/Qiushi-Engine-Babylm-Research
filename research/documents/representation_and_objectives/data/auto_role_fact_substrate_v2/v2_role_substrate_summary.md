# v2 role substrate result and route — v2 counterfactual role-family substrate

## Purpose

This run repairs the earlier analysis/248 automatic role-label object by asking for counterfactually closed role-exchange families rather than independent easy facts. It uses short teacher inference only; no BabyLM model or student model is trained.

## Main numbers

- Candidate A/B cases: 53
- Parsed structurally valid counterfactual-family cases: 42
- Teacher rows: 1008 (1008 valid)
- Cross-teacher agreement on exact one-sentence prompts: 0.7976
- Both teachers gave expected label: 0.7560
- Source↔bridge retained role families: 11 (0.2619 of structurally valid)
- Full-context retained role families: 7 (0.1667 of structurally valid)

## Retained source↔bridge role types

- cause_effect: 3
- condition_outcome: 3
- cause_effect_agent_patient_entity_state_location_relation_condition_outcome_comparison_beneficiary_instrument_temporal_order_other_role_relation: 1
- agent_patient: 1
- comparison: 1
- entity_state: 1
- cause_effect_entity_state: 1

## Heuristic EWoK bridge-panel domain mapping

- physical-relations: 11
- material-dynamics: 8
- physical-dynamics: 8
- social-properties: 7
- social-relations: 6
- agent-properties: 5
- material-properties: 4
- spatial-relations: 1

## Scientific interpretation

Current collection preserves a real but sparse source-bridge-stable role-family core. It is useful as seed/evaluation material, but too small for distillation as a general principle substrate without broader source-attested transformations.

## Retained source↔bridge families (first 12)

### B010 — cause_effect
Source: Damaged watersheds, massive squatter colonies living in danger zones and the neglect of drainage systems are some of the factors that have made the chaotic city of 15 million people much more vulnerable to enormous floods.
Bridge: Damaged watersheds, massive squatter colonies in danger zones, and neglected drainage systems made the chaotic 15 million people city vulnerable to enormous floods.
Facts:
- p1 ENTAILED (main_relation): Damaged watersheds, massive squatter colonies, and neglected drainage systems made the chaotic city vulnerable to floods.
- p2 ENTAILED (paraphrase_or_untouched_stability): The chaotic city's vulnerability to enormous floods is caused by damaged watersheds, squatter colonies, and neglected drainage.
- n1 NOT_ENTAILED (counterfactual): Damaged watersheds, massive squatter colonies, and neglected drainage systems made the chaotic city immune to enormous floods.
- n2 NOT_ENTAILED (counterfactual): Neglected drainage systems, damaged watersheds, and massive squatter colonies made the chaotic city vulnerable to earthquakes.

### B017 — cause_effect|agent_patient|entity_state|location_relation|condition_outcome|comparison|beneficiary|instrument|temporal_order|other_role_relation
Source: Still, it should help accelerate the rate at which some organizations are able to digitize their book collections, while leaving the books it scans unscathed.
Bridge: It should help accelerate the rate at which organizations digitize books while leaving them unscathed.
Facts:
- p1 ENTAILED (main_relation): Organizations digitize books.
- p2 ENTAILED (untouched_stability): The books remain unscathed.
- n1 NOT_ENTAILED (counterfactual): Books digitize organizations.
- n2 NOT_ENTAILED (counterfactual): Organizations leave the digitized books damaged.

### B019 — cause_effect
Source: In the 1920s, most of Europe was bankrupt due to after effects of WWI.
Bridge: Most of Europe was bankrupt in the 1920s due to WWI effects.
Facts:
- p1 ENTAILED (main_relation): WWI effects caused the bankruptcy of most of Europe in the 1920s.
- p2 ENTAILED (paraphrase_or_untouched_stability): Most of Europe was bankrupt in the 1920s.
- n1 NOT_ENTAILED (counterfactual): The bankruptcy of most of Europe in the 1920s caused WWI effects.
- n2 NOT_ENTAILED (counterfactual): Most of Europe was bankrupt in the 1920s due to the Great Depression.

### B022 — agent_patient
Source: A former San Francisco stock trader who at one point was managing 50 billion dollars for the state of California, Ives found his way to Honduras after Sept.
Bridge: Former San Francisco stock trader Ives, managing 50 billion dollars for California state, found his way to Honduras after Sept.
Facts:
- p1 ENTAILED (main_relation): Ives went to Honduras after Sept.
- p2 ENTAILED (untouched_stability): Ives was managing 50 billion dollars for California.
- n1 NOT_ENTAILED (counterfactual): Honduras found its way to Ives after Sept.
- n2 NOT_ENTAILED (counterfactual): Ives went to California after Sept.

### B023 — condition_outcome
Source: The graph below shows that the total global spending on biologic drugs during the recent years is increasing, and it is expected to further increase in 2017.
Bridge: The graph shows total global spending on biologic drugs during recent years is increasing and expected to further increase in 2017.
Facts:
- p1 ENTAILED (main_relation): Global spending on biologic drugs is increasing during recent years.
- p2 ENTAILED (outcome_stability): Spending on biologic drugs is expected to further increase in 2017.
- n1 NOT_ENTAILED (counterfactual): Global spending on biologic drugs is decreasing during recent years.
- n2 NOT_ENTAILED (counterfactual): Global spending on biologic drugs in 2017 is expected to decrease.

### B024 — cause_effect
Source: Korean households enthusiastically took up the new fuel which _ as they soon discovered _ was so much more convenient than firewood.
Bridge: Korean households enthusiastically took the new fuel, soon discovering it was more convenient than firewood.
Facts:
- p1 ENTAILED (main_relation): The new fuel was more convenient than firewood.
- p2 ENTAILED (paraphrase_or_untouched_stability): Korean households found the new fuel convenient.
- n1 NOT_ENTAILED (counterfactual): Firewood was more convenient than the new fuel.
- n2 NOT_ENTAILED (counterfactual): Korean households found firewood convenient.

### B037 — condition_outcome
Source: However, they are a good type of fat, a monounsaturated fat, and this type of fat is better for you than a saturated or trans fat.
Bridge: They are a good monounsaturated fat type better than saturated or trans fat.
Facts:
- p1 ENTAILED (main_relation): Monounsaturated fat is better for you than saturated or trans fat.
- p2 ENTAILED (paraphrase_or_untouched_stability): They are a good type of fat.
- n1 NOT_ENTAILED (counterfactual): Saturated or trans fat is better for you than monounsaturated fat.
- n2 NOT_ENTAILED (counterfactual): Monounsaturated fat is better for you than unsaturated fat.

### B060 — comparison
Source: These copepods are bigger than our usual ‘local’ species, and pack on even more lipids.
Bridge: These copepods are bigger than usual local species and pack even more lipids.
Facts:
- p1 ENTAILED (main_relation): The copepods are bigger than the local species.
- p2 ENTAILED (untouched_stability): The copepods pack on even more lipids.
- n1 NOT_ENTAILED (counterfactual): The local species are bigger than the copepods.
- n2 NOT_ENTAILED (counterfactual): The copepods are bigger than the local species but have less lipids.

### B061 — condition_outcome
Source: This is why you have better balance when you stand with your feet apart than with your feet together.
Bridge: You have better balance standing with feet apart than together.
Facts:
- p1 ENTAILED (main_relation): Standing with feet apart yields better balance than standing with feet together.
- p2 ENTAILED (paraphrase_or_untouched_stability): The condition of having feet apart results in superior balance compared to having feet together.
- n1 NOT_ENTAILED (counterfactual): Standing with feet together yields better balance than standing with feet apart.
- n2 NOT_ENTAILED (counterfactual): Standing with feet apart yields worse balance than standing with feet together.

### B075 — entity_state
Source: There is also the Freedom Wall, which has 4,048 gold stars, with each star representing 100 Americans who died during the war.
Bridge: The Freedom Wall has 4,048 gold stars, each representing 100 Americans who died during the war.
Facts:
- p1 ENTAILED (main_relation): The Freedom Wall's gold stars represent 100 Americans who died during the war.
- p2 ENTAILED (paraphrase_or_untouched_stability): Each gold star on the Freedom Wall corresponds to 100 war deaths.
- n1 NOT_ENTAILED (counterfactual): Each gold star on the Freedom Wall represents 100 Americans who survived the war.
- n2 NOT_ENTAILED (counterfactual): The Freedom Wall has 4,048 gold stars, each representing 100 soldiers who were injured during the war.

### B098 — cause_effect|entity_state
Source: The oceans that absorb 90% of the excess heat produced by the greenhouse gases now have the highest recorded temperatures.
Bridge: Oceans absorb 90% of excess heat produced by greenhouse gases and now have highest recorded temperatures.
Facts:
- p1 ENTAILED (main_relation): The oceans absorb 90% of the excess heat produced by greenhouse gases.
- p2 ENTAILED (untouched_stability): The oceans now have the highest recorded temperatures.
- n1 NOT_ENTAILED (counterfactual): The greenhouse gases absorb 90% of the excess heat produced by the oceans.
- n2 NOT_ENTAILED (counterfactual): The oceans now have the lowest recorded temperatures.


## Representative family failures

### B010 — cause_effect
Reasons: natural_reference_not_stable
- natural_compact_reference p1 gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Damaged watersheds, massive squatter colonies, and neglected drainage systems made the chaotic city vulnerable to floods.

### B011 — entity_state
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n1 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: In Tulare County, early-season vegetables including eggplant, squash, and cucumbers grew well.
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: In Tulare County, early-season vegetables including eggplant, squash, and cucumbers were planted.
- source_attested_bridge n1 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: In Tulare County, early-season vegetables including eggplant, squash, and cucumbers grew well.
- source_attested_bridge p1 gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: In Tulare County, early-season vegetables including eggplant, squash, and cucumbers were harvested.

### B012 — cause_effect
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Changes in barometric pressure cause migraines.
- source_attested_bridge n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Changes in barometric pressure cause migraines.
- source p1 gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Barometric pressure changes cause headaches.

### B014 — condition_outcome
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n1 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: A damaged tree or shrub generally can tolerate total defoliation without suffering permanent damage.
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: A healthy tree or shrub generally can tolerate total defoliation without suffering immediate damage.
- source_attested_bridge n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: A healthy tree or shrub generally can tolerate total defoliation without suffering immediate damage.
- source n1 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: A damaged tree or shrub generally can tolerate total defoliation without suffering permanent damage.

### B016 — agent_patient
Reasons: source_bridge_not_stable_both_teachers
- source_attested_bridge n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Gabriel Brammer found the dry spell to be extraordinarily clear and beautiful.
- source n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Gabriel Brammer found the dry spell to be extraordinarily clear and beautiful.

### B018 — condition_outcome
Reasons: source_bridge_not_stable_both_teachers
- source_attested_bridge p2 gold=ENTAILED qwen=NOT_ENTAILED llama=NOT_ENTAILED :: The American Bulldog is brave and protective.

### B019 — cause_effect
Reasons: natural_reference_not_stable
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Most of Europe was bankrupt in the 1920s due to the Great Depression.

### B026 — cause_effect
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n1 gold=NOT_ENTAILED qwen=ENTAILED llama=NOT_ENTAILED :: The report needs computer skills because lead pollutes soil.
- source_attested_bridge n1 gold=NOT_ENTAILED qwen=ENTAILED llama=NOT_ENTAILED :: The report needs computer skills because lead pollutes soil.
- source n1 gold=NOT_ENTAILED qwen=ENTAILED llama=NOT_ENTAILED :: The report needs computer skills because lead pollutes soil.

### B028 — condition_outcome
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=ENTAILED llama=ENTAILED :: Learning piano scales and chords helps learning tremendously.
- natural_compact_reference p2 gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Good instructional books exist.
- source_attested_bridge n2 gold=NOT_ENTAILED qwen=ENTAILED llama=ENTAILED :: Learning piano scales and chords helps learning tremendously.
- source n2 gold=NOT_ENTAILED qwen=ENTAILED llama=ENTAILED :: Learning piano scales and chords helps learning tremendously.

### B029 — condition_outcome
Reasons: source_bridge_not_stable_both_teachers
- natural_compact_reference n1 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Teens who are depressed are more likely to have difficulties with sleep within seven years.
- natural_compact_reference n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Teens with sleep difficulties are more likely to be anxious within seven years.
- source_attested_bridge n2 gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Teens with sleep difficulties are more likely to be anxious within seven years.
- source_attested_bridge p2 gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED :: Most teens with sleep difficulties are likely to be depressed within the span of seven years.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_role_substrate_summary.json`
- family table: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/family_retention_summary.jsonl`
- retained source↔bridge families: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/retained_source_bridge_role_families.jsonl`
- retained full-context families: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/retained_full_context_role_families.jsonl`
- failures: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/family_failure_reasons.jsonl`
