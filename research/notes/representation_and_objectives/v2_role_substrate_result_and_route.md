# v2 role substrate result and route — v2 counterfactual role-family substrate result

## Question

After earlier analysis/248, the central risk was selection collapse: retaining only easy lexical entailments would not address the route2 factorial causal review/244/245 binding failure. v2 role substrate result and route therefore rebuilt the automatic source-attested role substrate around counterfactually closed role-exchange families. A usable family must contain two source/bridge-supported positives and two controlled negatives, where actor/query/condition/outcome changes move the corresponding relation, while paraphrase or untouched facts remain stable.

No BabyLM model or student model was trained. This was a short approved-teacher inference experiment deciding whether the current bridge collection is a sufficient learning substrate or only a seed/evaluation object.

## Execution evidence

Script: `experiments/archive/representation_and_objectives/scripts/auto_role_fact_substrate_v2.py`

Inputs:
- A/B high-fidelity bridge candidates: `experiments/archive/frontier_consolidation/data/bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.json`, SHA256 `9c27a71d5ada1bfc1723f69b346da1b588988464d06cb710df430cc6e591eb49`.
- rawtoken bridge screen and route judgment EWoK bridge panel: `experiments/archive/representation_and_objectives/data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl`.

Run sequence:
1. CPU prompt preparation over 53 A/B candidates.
2. Qwen3.5-9B family generation on GPU1 after GPU0 was unavailable due to OOM; 53 prompts, 17,666 generated tokens, 412.7 s.
3. CPU parse: 42 structurally valid counterfactual families, 9 JSON malformed cases, 2 model-empty rejects; 504 one-sentence label prompts per teacher.
4. Qwen3.5-9B and Llama3.1-8B-Instruct one-sentence label passes on GPU1; 504 prompts each.
5. CPU scoring and post-analysis.

Main output files:
- `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_role_substrate_summary.json`
- `research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_role_substrate_summary.md`
- `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/retained_source_bridge_role_families.jsonl`
- `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/retained_full_context_role_families.jsonl`
- `experiments/archive/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_postanalysis_summary.json`
- `research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_postanalysis_summary.md`

## Results

V2 successfully changed the object from independent facts to structured role-exchange families, but it did not solve the sparsity problem.

Core numbers:
- Candidate A/B cases: 53.
- Structurally valid counterfactual-family cases: 42.
- Teacher rows: 1008/1008 valid.
- Cross-teacher agreement on exact one-sentence prompts: 0.7976.
- Both teachers gave expected label: 0.7560.
- Source↔bridge retained role families: 11/42 structurally valid (0.2619), 11/53 original candidates (0.2075).
- Full-context retained role families: 7/42 structurally valid (0.1667), 7/53 original candidates (0.1321).

Retained source↔bridge families: `B010`, `B017`, `B019`, `B022`, `B023`, `B024`, `B037`, `B060`, `B061`, `B075`, `B098`.

Retained full-context families: `B017`, `B022`, `B023`, `B024`, `B061`, `B075`, `B098`.

Diversity is real but small:
- 7 retained source↔bridge role-type normalizations.
- 155 distinct retained hypothesis content words.
- No retained low-value `mention/existence/topic` families by the script's filter.
- Heuristic mapping overlaps all correctness transition analysis relational EWoK domains (`physical-dynamics`, `physical-relations`, `social-properties`, `spatial-relations`), but only as a sparse seed-level mapping, not an item-level bridge.

Failure anatomy:
- 123/504 prompt pairs failed both-teachers-expected.
- Negative overaccepted by at least one teacher: 109.
- Negative accepted by both teachers: 20.
- Positive rejected by at least one teacher: 14.
- Teacher pattern: Llama not expected 86, Qwen not expected 16, both same wrong 21.

This means the main weakness is not parser noise. It is the difficulty of constructing controlled false role-exchange hypotheses that independent teachers reliably reject, plus the small size and narrowness of the bridge-transform collection.

## Scientific interpretation

The current companion analysis bridge collection contains a real source-attested counterfactual role-family core: it is not merely easy lexical entailment, and it covers cause/effect, condition/outcome, agent/patient, comparison, entity/state, and mixed state-cause roles. However, retention remains too low for distillation or BabyLM-scale training. If trained from this collection now, a student would mostly learn teacher-specific handling of a tiny and uneven set of examples, not the general latent occurrence-role operation needed for the route2 factorial causal review/244/245 binding failure.

This result strengthens the live route but changes the next object: the retained v2 families should be used as seed probes and design examples. The immediate next scientific need is a broader source-attested transformation source that can produce many counterfactually closed role-exchange families with stable cross-teacher labels and better coverage of natural conditional-reversal families.

## Consequence for the BabyLM goal

Do not launch BabyLM training, compact-view tuning, or another TinyMLM memory-interface variant from this evidence. The practical 42.12 endpoint remains frozen. The central scientific route remains teacher-supported raw-text role assignment for main-path compositional updating, but the present 11-family core is not enough to instantiate it as a generalizable sample-efficient learning principle.

The next research step should either:
1. construct or locate a broader source-attested transformation source with explicit actor/query/condition/outcome changes, paraphrase stability, and untouched-fact preservation; or
2. use the 11 retained families only as an evaluation/probe set while designing a stronger generator/selection pipeline.

A useful broader source should return hundreds or thousands of stable families, not merely a higher agreement rate on a smaller retained slice. Stability must be reported together with semantic diversity, role-exchange structure, and mapping to the EWoK/Entity conditional-reversal signature.
