# earlier analysis automatic source-attested role-fact substrate probe

## Purpose

The manual panel was positive but small. This probe asks whether A/B source-attested bridge transformations can automatically yield common role facts whose expected labels remain stable when each source, bridge, and natural compact sentence is encoded independently. Qwen generated facts; Qwen and Llama separately labelled the resulting one-sentence entailment probes.

## Aggregate

- teacher label rows: 1032, valid: 1021
- overall expected-label consistency: 0.902
- cross-teacher label agreement: 0.850
- cross-teacher both expected: 0.826
- source↔bridge expected invariance: 0.861
- qwen bridge expected consistency: 0.917
- llama bridge expected consistency: 0.855
- bridge false-role rejection consistency: 0.849
- case all-expected rate: 0.256 (11/43)
- decision: **BORDERLINE_OR_NEGATIVE_ROLE_LABEL_PREMISE_REPAIR_BEFORE_STUDENT**

## By teacher

| teacher | n | expected consistency | invalid |
|---|---:|---:|---:|
| llama3.1-8b-instruct | 516 | 0.872 | 0 |
| qwen3.5-9b | 516 | 0.933 | 11 |

## By context

| context | n | expected consistency | invalid |
|---|---:|---:|---:|
| natural_compact_reference | 344 | 0.891 | 3 |
| source | 344 | 0.929 | 4 |
| source_attested_bridge | 344 | 0.885 | 4 |

## Failure/disagreement sample

### qwen3.5-9b label_B018_source_attested_bridge_e1
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=B_adjudicate_transform role=attribute
- hypothesis: The American Bulldog is brave and protective.
- context: The American Bulldog is best when trained young.

### qwen3.5-9b label_B023_source_e1
- gold=ENTAILED parsed=None raw='ENTAILLED' context=source tier=B_adjudicate_transform role=trend
- hypothesis: Global spending on biologic drugs is increasing.
- context: The graph below shows that the total global spending on biologic drugs during the recent years is increasing, and it is expected to further increase in 2017.

### qwen3.5-9b label_B023_source_e2
- gold=ENTAILED parsed=None raw='ENTAILLED' context=source tier=B_adjudicate_transform role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.
- context: The graph below shows that the total global spending on biologic drugs during the recent years is increasing, and it is expected to further increase in 2017.

### qwen3.5-9b label_B023_source_attested_bridge_e1
- gold=ENTAILED parsed=None raw='ENTAILLED' context=source_attested_bridge tier=B_adjudicate_transform role=trend
- hypothesis: Global spending on biologic drugs is increasing.
- context: The graph shows total global spending on biologic drugs during recent years is increasing and expected to further increase in 2017.

### qwen3.5-9b label_B023_source_attested_bridge_e2
- gold=ENTAILED parsed=None raw='ENTAILLED' context=source_attested_bridge tier=B_adjudicate_transform role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.
- context: The graph shows total global spending on biologic drugs during recent years is increasing and expected to further increase in 2017.

### qwen3.5-9b label_B023_natural_compact_reference_e1
- gold=ENTAILED parsed=None raw='ENTAILLED' context=natural_compact_reference tier=B_adjudicate_transform role=trend
- hypothesis: Global spending on biologic drugs is increasing.
- context: The graph shows global biologic drug spending increased recently and is expected to rise further in 2017.

### qwen3.5-9b label_B023_natural_compact_reference_e2
- gold=ENTAILED parsed=None raw='ENTAILLED' context=natural_compact_reference tier=B_adjudicate_transform role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.
- context: The graph shows global biologic drug spending increased recently and is expected to rise further in 2017.

### qwen3.5-9b label_B028_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=entity_existence
- hypothesis: Good instructional books exist for piano scales and chords.
- context: Good books exist, and learning piano scales and chords helps tremendously.

### qwen3.5-9b label_B029_source_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source tier=B_adjudicate_transform role=cause_effect
- hypothesis: Sleep trouble increases the likelihood of depression in teens over a seven-year span.
- context: Research shows that most of the teens with difficulties in having enough sleep are more likely to be depressed within the span of seven years.

### qwen3.5-9b label_B037_natural_compact_reference_e1
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=entity_attribute
- hypothesis: The fat is monounsaturated.
- context: However, they are good monounsaturated fats, better than saturated or trans fats.

### qwen3.5-9b label_B041_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=purpose
- hypothesis: Gray seals were considered efficient means to retrieve items from the ocean floor.
- context: Gray seals dive 475 feet deep and stay underwater 20 minutes, making them efficient for retrieving ocean floor items.

### qwen3.5-9b label_B043_source_attested_bridge_e1
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=B_adjudicate_transform role=goal
- hypothesis: Wowzers Learning aims to personalize math programs for grades K-8.
- context: Wowzers Learning vision is to breakdown math success barriers for grades K-8.

### qwen3.5-9b label_B043_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=goal
- hypothesis: Wowzers Learning seeks to breakdown barriers to math success.
- context: Wowzers Learning aims to help K-8 math success by personalizing programs.

### qwen3.5-9b label_B045_source_n2
- gold=NOT_ENTAILED parsed=ENTAILED raw='ENTAILED' context=source tier=B_adjudicate_transform role=cause_reversal
- hypothesis: Children learn the material because they want to be competitive in the game.
- context: The aspect of them game allows the children to be competitive, and want to win, therefore want to learn the material.

### qwen3.5-9b label_B045_source_attested_bridge_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=B_adjudicate_transform role=cause
- hypothesis: Children want to win because the game allows them to be competitive.
- context: The aspect of the game allows children to be competitive and want to learn the material.

### qwen3.5-9b label_B045_natural_compact_reference_n2
- gold=NOT_ENTAILED parsed=ENTAILED raw='ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=cause_reversal
- hypothesis: Children learn the material because they want to be competitive in the game.
- context: The game's aspect lets children compete, want to win, and thus learn the material.

### qwen3.5-9b label_B050_source_attested_bridge_e1
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=B_adjudicate_transform role=cause
- hypothesis: Brown's 1836 re-survey of the Sullivan Line caused Missouri to claim its border extended 13 miles into Iowa.
- context: Brown's controversial 1836 re-survey of the Honey War line caused Missouri to claim its border extended 13 miles into Iowa.

### qwen3.5-9b label_B050_source_attested_bridge_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=B_adjudicate_transform role=attribute
- hypothesis: The re-survey of the Sullivan Line was Brown's most controversial survey.
- context: Brown's controversial 1836 re-survey of the Honey War line caused Missouri to claim its border extended 13 miles into Iowa.

### qwen3.5-9b label_B050_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=attribute
- hypothesis: The re-survey of the Sullivan Line was Brown's most controversial survey.
- context: Brown's 1836 Sullivan Line re-survey caused Missouri to claim its border extended 13 miles into Iowa.

### qwen3.5-9b label_B054_source_attested_bridge_e2
- gold=ENTAILED parsed=None raw='ENTAILLED' context=source_attested_bridge tier=B_adjudicate_transform role=outcome
- hypothesis: The fire worm finds a hiding place.
- context: The active fire worm turns around the aquarium for minutes before finding a hiding place.

### qwen3.5-9b label_B054_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=B_adjudicate_transform role=outcome
- hypothesis: The fire worm finds a hiding place.
- context: Fire worms are active, turning around the aquarium for minutes before hiding.

### qwen3.5-9b label_B057_source_attested_bridge_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=source_attested_bridge tier=A_seed_transform role=evaluation
- hypothesis: The finding that HDL may decrease while triglycerides increase on a low-fat diet is noteworthy.
- context: Future research must explore why HDL may decrease while triglycerides increase on a low-fat diet.

### qwen3.5-9b label_B057_natural_compact_reference_e2
- gold=ENTAILED parsed=NOT_ENTAILED raw='NOT_ENTAILED' context=natural_compact_reference tier=A_seed_transform role=evaluation
- hypothesis: The finding that HDL may decrease while triglycerides increase on a low-fat diet is noteworthy.
- context: Future research must explore why HDL may decrease and triglycerides increase on a low-fat diet.

### qwen3.5-9b label_B059_source_n1
- gold=NOT_ENTAILED parsed=ENTAILED raw='ENTAILED' context=source tier=B_adjudicate_transform role=agent_patient_swap
- hypothesis: The single line is performed twice through the form.
- context: This movement is done twice through the form, and it represents a single line.

## Cross-teacher non-both-expected sample

### label_B006_source_attested_bridge_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source_attested_bridge role=action
- hypothesis: The HFP program reduced food security for 5 million people.

### label_B018_source_attested_bridge_e1
- gold=ENTAILED qwen=NOT_ENTAILED llama=NOT_ENTAILED context=source_attested_bridge role=attribute
- hypothesis: The American Bulldog is brave and protective.

### label_B023_natural_compact_reference_e1
- gold=ENTAILED qwen=None llama=ENTAILED context=natural_compact_reference role=trend
- hypothesis: Global spending on biologic drugs is increasing.

### label_B023_natural_compact_reference_e2
- gold=ENTAILED qwen=None llama=ENTAILED context=natural_compact_reference role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.

### label_B023_natural_compact_reference_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=entity_swap
- hypothesis: Spending on generic drugs is expected to rise in 2017.

### label_B023_source_attested_bridge_e1
- gold=ENTAILED qwen=None llama=ENTAILED context=source_attested_bridge role=trend
- hypothesis: Global spending on biologic drugs is increasing.

### label_B023_source_attested_bridge_e2
- gold=ENTAILED qwen=None llama=ENTAILED context=source_attested_bridge role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.

### label_B023_source_e1
- gold=ENTAILED qwen=None llama=ENTAILED context=source role=trend
- hypothesis: Global spending on biologic drugs is increasing.

### label_B023_source_e2
- gold=ENTAILED qwen=None llama=ENTAILED context=source role=prediction
- hypothesis: Global spending on biologic drugs is expected to increase in 2017.

### label_B028_natural_compact_reference_e2
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=entity_existence
- hypothesis: Good instructional books exist for piano scales and chords.

### label_B028_natural_compact_reference_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=entity_swap
- hypothesis: Several good instructional books on guitar scales and chords help tremendously.

### label_B028_source_attested_bridge_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source_attested_bridge role=entity_swap
- hypothesis: Several good instructional books on guitar scales and chords help tremendously.

### label_B029_natural_compact_reference_n1
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=cause_effect
- hypothesis: Depressed teens are more likely to have difficulties sleeping within the span of seven years.

### label_B029_source_e2
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source role=cause_effect
- hypothesis: Sleep trouble increases the likelihood of depression in teens over a seven-year span.

### label_B037_natural_compact_reference_e1
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=entity_attribute
- hypothesis: The fat is monounsaturated.

### label_B041_natural_compact_reference_e2
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=purpose
- hypothesis: Gray seals were considered efficient means to retrieve items from the ocean floor.

### label_B041_natural_compact_reference_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=actor_patient_swap
- hypothesis: The ocean floor dives 475 feet deep and stays underwater 20 minutes.

### label_B041_source_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source role=actor_patient_swap
- hypothesis: The ocean floor dives 475 feet deep and stays underwater 20 minutes.

### label_B043_natural_compact_reference_e2
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=goal
- hypothesis: Wowzers Learning seeks to breakdown barriers to math success.

### label_B043_natural_compact_reference_n1
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=natural_compact_reference role=goal
- hypothesis: Wowzers Learning aims to personalize reading programs for grades K-8.

### label_B043_source_attested_bridge_e1
- gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source_attested_bridge role=goal
- hypothesis: Wowzers Learning aims to personalize math programs for grades K-8.

### label_B043_source_attested_bridge_n1
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source_attested_bridge role=goal
- hypothesis: Wowzers Learning aims to personalize reading programs for grades K-8.

### label_B043_source_attested_bridge_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source_attested_bridge role=goal
- hypothesis: The vision of Wowzers Learning is to breakdown barriers to science success.

### label_B043_source_n2
- gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED context=source role=goal
- hypothesis: The vision of Wowzers Learning is to breakdown barriers to science success.

## Scientific meaning

Inspect disagreement/failure modes and repair fact generation or candidate selection before any student or architecture training.

Boundary: this establishes only a role-label premise for a small student pilot. It does not reopen compact-view tuning or justify 100M BabyLM training by itself.

## Files

- generated fact cases: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate/generated_fact_cases.jsonl`
- all teacher label rows: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate/all_teacher_label_rows.jsonl`
- context invariance rows: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate/context_invariance_rows.jsonl`
- summary JSON: `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate/auto_role_fact_substrate_summary.json`
