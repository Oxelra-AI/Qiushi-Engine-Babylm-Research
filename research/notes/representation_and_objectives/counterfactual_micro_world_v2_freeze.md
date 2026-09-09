# counterfactual micro world freeze frozen counterfactual micro-world v2

Status: **FROZEN_V2**

This replaces the first counterfactual micro world freeze object before any checkpoint scoring. The replacement was made because static inspection, not model output, found awkward spatial and transfer frames. v2 uses hand-cleaned compatible templates with slot words attested in the legal 10M pool and one-token alternatives under both legal16k and legal40k tokenizers.

## Files
- Frames: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v2/counterfactual_micro_world_v2_frames.jsonl`
- Renamed controls: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v2/counterfactual_micro_world_v2_renamed_controls.jsonl`
- Manifest: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v2/counterfactual_micro_world_v2_manifest.json`
- Content SHA256: `3fb52fde6d51a05019a0a262294727b9d57f2f2aeafed4e0bfa5d50b24d4dc0b`

## Static quality
- Frames: 400
- Family counts: `{'A_spatial_direct': 100, 'B_transfer_role': 100, 'C_reported_relation': 100, 'D_state_update_order': 100}`
- Depth counts: `{'1.0': 200, '1.5': 100, '2.0': 100}`
- Subtype counts: `{'on_under': 57, 'near_far': 24, 'above_below': 72, 'inside_outside': 47, 'passed': 28, 'handed': 20, 'gave': 26, 'sent': 26, 'open_closed': 51, 'empty_full': 49}`
- All target spans one token under both tokenizers: True
- Target prior ratio ≤4: 202 / 400 (0.505)
- Target prior ratio summary: `{'n': 400, 'median': 3.967019172245892, 'mean': 5.503186844735642, 'max': 29.211267605633804}`

## Examples
### A_spatial_direct
- `A_0001` (on_under): C1=The bag is on the floor. C2=The bag is under the floor. Query=The bag is {target} the floor. Alts=on/under
- `A_0002` (near_far): C1=The hat is near the kitchen. C2=The hat is far from the kitchen. Query=The hat is {target} from the kitchen. Alts=near/far
- `A_0003` (above_below): C1=The key is above the bed. C2=The key is below the bed. Query=The key is {target} the bed. Alts=above/below
- `A_0004` (inside_outside): C1=The key is inside the drawer. C2=The key is outside the drawer. Query=The key is {target} the drawer. Alts=inside/outside
### B_transfer_role
- `B_0001` (passed): C1=John passed the coin to Max. C2=Max passed the coin to John. Query=The recipient of the coin was {target}. Alts=Max/John
- `B_0002` (handed): C1=Sam handed the apple to Amy. C2=Amy handed the apple to Sam. Query=The recipient of the apple was {target}. Alts=Amy/Sam
- `B_0003` (gave): C1=Anna gave the apple to Mary. C2=Mary gave the apple to Anna. Query=The recipient of the apple was {target}. Alts=Mary/Anna
- `B_0004` (handed): C1=Ben handed the letter to Mary. C2=Mary handed the letter to Ben. Query=The recipient of the letter was {target}. Alts=Mary/Ben
### C_reported_relation
- `C_0001` (on_under): C1=John believed that the key is on the chair. C2=John believed that the key is under the chair. Query=According to John, the key is {target} the chair. Alts=on/under
- `C_0002` (above_below): C1=Emma said that the ring is above the desk. C2=Emma said that the ring is below the desk. Query=According to Emma, the ring is {target} the desk. Alts=above/below
- `C_0003` (above_below): C1=Emma said that the box is above the chair. C2=Emma said that the box is below the chair. Query=According to Emma, the box is {target} the chair. Alts=above/below
- `C_0004` (on_under): C1=Lucy said that the glass is on the shelf. C2=Lucy said that the glass is under the shelf. Query=According to Lucy, the glass is {target} the shelf. Alts=on/under
### D_state_update_order
- `D_0001` (open_closed): C1=The drawer was closed. Seth closed the drawer. Then Seth opened the drawer. C2=The drawer was open. Seth opened the drawer. Then Seth closed the drawer. Query=The drawer is now {target}. Alts=open/closed
- `D_0002` (empty_full): C1=The bag was full. Emma filled the bag. Then Emma emptied the bag. C2=The bag was empty. Emma emptied the bag. Then Emma filled the bag. Query=The bag is now {target}. Alts=empty/full
- `D_0003` (open_closed): C1=The drawer was closed. Ben closed the drawer. Then Ben opened the drawer. C2=The drawer was open. Ben opened the drawer. Then Ben closed the drawer. Query=The drawer is now {target}. Alts=open/closed
- `D_0004` (empty_full): C1=The glass was full. Amy filled the glass. Then Amy emptied the glass. C2=The glass was empty. Amy emptied the glass. Then Amy filled the glass. Query=The glass is now {target}. Alts=empty/full

## Scoring interpretation fixed before model runs
For every frame, later scorer computes Δ1=s(C1,Q,a)-s(C1,Q,b), Δ2=s(C2,Q,a)-s(C2,Q,b). Crossed-sign success requires Δ1>0 and Δ2<0. Context-removed and shuffled-context controls use the same frozen alternatives and query; renaming controls are separate frozen records. Candidate-prior-balanced and imbalanced subsets must be reported separately.
