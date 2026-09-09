# fineweb slot pairing feasibility FineWeb source/view slot-pairing feasibility

This CPU-only work formalizes a more controlled future FineWeb unit: source `S` plus a second slot. The second slot is official text in `A_natural`, a distinct FineWeb source `D` in `B_breadth`, literal source repetition in `B_repeat`, and a faithful generated view `V` in `C_view`. No view was generated and no training corpus was written.

Input live seed: 12,437 rows / 300,005 words / 3,577 documents from `experiments/archive/representation_and_objectives/data/stratified_live_fineweb_blueprint/live_fineweb_stratified_current_seed_300k.jsonl`.

| view length ratio | D focus match | max D length offset | pairs | C/B_repeat packet words | inventory words S+D | 300k arm reachable | 1M arm coverage |
|---:|---|---:|---:|---:|---:|---|---:|
| 0.75 | True | 0 | 5,801 | 284,142 | 284,142 | False | 0.284 |
| 0.75 | True | 2 | 6,154 | 297,435 | 297,385 | False | 0.297 |
| 0.75 | False | 0 | 5,846 | 287,762 | 287,762 | False | 0.288 |
| 0.75 | False | 2 | 6,188 | 298,873 | 298,853 | False | 0.299 |
| 0.88 | True | 0 | 5,985 | 288,454 | 288,454 | False | 0.288 |
| 0.88 | True | 2 | 6,205 | 299,463 | 299,263 | False | 0.299 |
| 0.88 | False | 0 | 6,093 | 292,887 | 292,887 | False | 0.293 |
| 0.88 | False | 2 | 6,218 | 300,205 | 299,974 | True | 0.300 |
| 0.95 | True | 0 | 5,928 | 286,956 | 286,956 | False | 0.287 |
| 0.95 | True | 2 | 6,213 | 299,863 | 299,649 | False | 0.300 |
| 0.95 | False | 0 | 5,993 | 290,019 | 290,019 | False | 0.290 |
| 0.95 | False | 2 | 6,217 | 300,090 | 299,935 | True | 0.300 |
| 1.05 | True | 0 | 5,965 | 285,377 | 285,377 | False | 0.285 |
| 1.05 | True | 2 | 6,212 | 300,101 | 299,722 | True | 0.300 |
| 1.05 | False | 0 | 6,064 | 290,849 | 290,849 | False | 0.291 |
| 1.05 | False | 2 | 6,217 | 300,207 | 299,925 | True | 0.300 |

Preferred current-seed dry contract: ratio 0.88, same_focus=True, max_length_offset=0; it yields 5,985 pairs and 288,454 changed-block words.

Focus packet words for that contract:

```json
{
  "causal_temporal_process": 154298,
  "physical_spatial_object": 63290,
  "other_expository_relation": 43976,
  "social_entity_state": 22506,
  "definition_taxonomic_fact": 4384
}
```

Scientific use: if cached source breadth is positive, this slot family lets the next experiment distinguish whether the second exposure budget is better spent on new facts, repeated wording, or faithful views while retaining a strong natural arm. If cached source breadth is weak, the same contract can still support a smaller source+view test without repeating the cached 96-word chunk route.

JSON: `experiments/archive/representation_and_objectives/data/fineweb_slot_pairing_feasibility/fineweb_slot_pairing_feasibility.json`
