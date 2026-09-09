# counterfactual micro world v3 static repair counterfactual micro-world v3 static repair

Status: **FROZEN_V3_STATIC_REPAIR_BEFORE_SCORING**

This replaces the counterfactual micro world freeze v2 object before any checkpoint scoring. Static inspection found that the v2 `near_far` subtype used a single query frame with `{target} from`, which makes the `near` alternative ungrammatical (`near from`). v3 removes near/far and also makes renamed controls preserve subtype-compatible slot classes, so the measure is less likely to score surface artifacts rather than crossed-sign context-conditioned alternative binding.

## Files
- Frames: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_frames.jsonl`
- Renamed controls: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_renamed_controls.jsonl`
- Manifest: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_manifest.json`
- Content SHA256: `ed60eb1ca541aae7b55dcaf99a66798d42b489ab85dcbd49ac5156a3b0d043bd`
- Frame file SHA256: `88346ddbabfe06ab18e628c1a1f2ba4184ac7dd67f0b8e07220a022f1dbb0817`
- Renamed file SHA256: `3bf8e5d390fdc1078997f0fbe907cdd3fce6b851519dee048f4817c4438aa17f`

## Static quality
- Frames: 400
- Family counts: `{'A_spatial_direct': 100, 'B_transfer_role': 100, 'C_reported_relation': 100, 'D_state_update_order': 100}`
- Depth counts: `{'1.0': 200, '1.5': 100, '2.0': 100}`
- Subtype counts: `{'inside_outside': 67, 'on_under': 64, 'above_below': 69, 'gave': 26, 'passed': 31, 'sent': 21, 'handed': 22, 'open_closed': 51, 'empty_full': 49}`
- All target spans one token under both tokenizers: True
- Target prior ratio ≤4: 198 / 400 (0.495)
- Target prior ratio summary: `{'n': 400, 'median': 4.344827586206897, 'mean': 5.749267932772199, 'max': 29.211267605633804}`
- Surface artifact phrase counts: `{}`

## Examples
### A_spatial_direct
- `A_0001` (inside_outside): C1=The bag is inside the room. C2=The bag is outside the room. Query=The bag is {target} the room. Alts=inside/outside
- `A_0002` (on_under): C1=The hat is on the shelf. C2=The hat is under the shelf. Query=The hat is {target} the shelf. Alts=on/under
- `A_0003` (on_under): C1=The key is on the bed. C2=The key is under the bed. Query=The key is {target} the bed. Alts=on/under
- `A_0004` (above_below): C1=The key is above the table. C2=The key is below the table. Query=The key is {target} the table. Alts=above/below
### B_transfer_role
- `B_0001` (gave): C1=Emma gave the pen to Seth. C2=Seth gave the pen to Emma. Query=The recipient of the pen was {target}. Alts=Seth/Emma
- `B_0002` (passed): C1=Seth passed the hat to Amy. C2=Amy passed the hat to Seth. Query=The recipient of the hat was {target}. Alts=Amy/Seth
- `B_0003` (sent): C1=Rose sent the key to Amy. C2=Amy sent the key to Rose. Query=The recipient of the key was {target}. Alts=Amy/Rose
- `B_0004` (sent): C1=Rose sent the map to Emma. C2=Emma sent the map to Rose. Query=The recipient of the map was {target}. Alts=Emma/Rose
### C_reported_relation
- `C_0001` (on_under): C1=Alice believed that the toy is on the table. C2=Alice believed that the toy is under the table. Query=According to Alice, the toy is {target} the table. Alts=on/under
- `C_0002` (above_below): C1=Seth claimed that the book is above the counter. C2=Seth claimed that the book is below the counter. Query=According to Seth, the book is {target} the counter. Alts=above/below
- `C_0003` (above_below): C1=Mary said that the book is above the shelf. C2=Mary said that the book is below the shelf. Query=According to Mary, the book is {target} the shelf. Alts=above/below
- `C_0004` (inside_outside): C1=Kate believed that the paper is inside the drawer. C2=Kate believed that the paper is outside the drawer. Query=According to Kate, the paper is {target} the drawer. Alts=inside/outside
### D_state_update_order
- `D_0001` (open_closed): C1=The bag was closed. Kate closed the bag. Then Kate opened the bag. C2=The bag was open. Kate opened the bag. Then Kate closed the bag. Query=The bag is now {target}. Alts=open/closed
- `D_0002` (open_closed): C1=The window was closed. Jack closed the window. Then Jack opened the window. C2=The window was open. Jack opened the window. Then Jack closed the window. Query=The window is now {target}. Alts=open/closed
- `D_0003` (empty_full): C1=The glass was full. Anna filled the glass. Then Anna emptied the glass. C2=The glass was empty. Anna emptied the glass. Then Anna filled the glass. Query=The glass is now {target}. Alts=empty/full
- `D_0004` (open_closed): C1=The door was closed. Emma closed the door. Then Emma opened the door. C2=The door was open. Emma opened the door. Then Emma closed the door. Query=The door is now {target}. Alts=open/closed

## Scoring interpretation fixed before model runs
For every frame, the scorer computes Δ1=s(C1,Q,a)-s(C1,Q,b), Δ2=s(C2,Q,a)-s(C2,Q,b). Crossed-sign success requires Δ1>0 and Δ2<0. Context-removed and shuffled-context controls use the same frozen alternatives and query; renamed controls are separate frozen records. Candidate-prior-balanced and imbalanced subsets must be reported separately. Per the counterfactual micro world v3 static repair strategist note, this measure must not be tuned to rank chck_82M highly; the first two-checkpoint run is only a mechanical coherence/control check, and interpretation requires extension to distinct legal trajectories plus the MLM-only/coupled aligned/coupled shuffled panel.
