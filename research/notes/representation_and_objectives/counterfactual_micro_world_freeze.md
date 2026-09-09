# counterfactual micro world freeze frozen counterfactual micro-world

Status: **FROZEN**

This file freezes a no-training diagnostic object before checkpoint scoring. It is designed to measure crossed-sign context-conditioned alternative binding rather than coherent-vs-corrupted text plausibility.

## Construction
- Pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- Pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- Legal16k tokenizer: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
- Legal40k tokenizer: `experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M`
- Frames: **480**
- Renaming controls: **480**
- Family counts: `{'A_spatial_direct': 120, 'B_transfer_role': 120, 'C_reported_relation': 120, 'D_state_update_order': 120}`
- Depth counts: `{'1.0': 240, '1.5': 120, '2.0': 120}`
- Content SHA256: `6795239331ac9c6c718a0cb8562c78fd8213b28796919f62d869b1e321c9f1ed`

## Target-prior balance
- Ratio ≤4: 263 / 480 (0.548)
- Ratio summary: `{'n': 480, 'median': 2.582661290322581, 'mean': 9.113948995233462, 'max': 80.77873254564985}`

## Why v1 is not duplicated
structural contrast v1 scored coherent-vs-perturbed text margins, ranked chck_77M rather than chck_82M, and contained punctuation/template artifacts. This object instead renders two contexts, one query, two alternatives, and requires Δ1 and Δ2 to have opposite correct signs. It includes no model results in construction.

## Example frames
### A_spatial_direct
- `A_0001` C1: The sock is inside the table. C2: The sock is outside the table. Query: The sock is {target} the table. Alts: inside / outside
- `A_0002` C1: The box is on the room. C2: The box is under the room. Query: The box is {target} the room. Alts: on / under
- `A_0003` C1: The paper is inside the shop. C2: The paper is outside the shop. Query: The paper is {target} the shop. Alts: inside / outside
### B_transfer_role
- `B_0001` C1: Rose gave the hat to Amy. C2: Amy gave the hat to Rose. Query: The hat was given to {target}. Alts: Amy / Rose
- `B_0002` C1: Tom showed the card to Kate. C2: Kate showed the card to Tom. Query: The card was given to {target}. Alts: Kate / Tom
- `B_0003` C1: Bob showed the toy to Max. C2: Max showed the toy to Bob. Query: The toy was given to {target}. Alts: Max / Bob
### C_reported_relation
- `C_0001` C1: Anna believed that the shoe is above the closet. C2: Anna believed that the shoe is below the closet. Query: According to Anna, the shoe is {target} the closet. Alts: above / below
- `C_0002` C1: Max thought that the letter is above the closet. C2: Max thought that the letter is below the closet. Query: According to Max, the letter is {target} the closet. Alts: above / below
- `C_0003` C1: Jack thought that the letter is on the kitchen. C2: Jack thought that the letter is under the kitchen. Query: According to Jack, the letter is {target} the kitchen. Alts: on / under
### D_state_update_order
- `D_0001` C1: The card was full. Sam filled the card. Then Sam emptied the card. C2: The card was empty. Sam emptied the card. Then Sam filled the card. Query: The card is now {target}. Alts: empty / full
- `D_0002` C1: The bottle was dirty. Kate dirtied the bottle. Then Kate cleaned the bottle. C2: The bottle was clean. Kate cleaned the bottle. Then Kate dirtied the bottle. Query: The bottle is now {target}. Alts: clean / dirty
- `D_0003` C1: The toy was dirty. Mary dirtied the toy. Then Mary cleaned the toy. C2: The toy was clean. Mary cleaned the toy. Then Mary dirtied the toy. Query: The toy is now {target}. Alts: clean / dirty

## Files
- Frames JSONL: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_frames.jsonl`
- Renamed controls JSONL: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_renamed_controls.jsonl`
- Manifest JSON: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_manifest.json`
