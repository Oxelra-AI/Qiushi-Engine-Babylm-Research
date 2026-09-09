# relation filtered pair pools — full relation four-cell synthesis

This synthesis joins the full no-update four-cell scores with model-free semantic subpool labels. The scientific question is whether the pair pool tracks the known FW relation tradeoff more than matched target/context permutation nulls.

| subset | n pairs | true order | true range | target-perm range | context-perm range | rowblock-compact true | rowblock-compact target-null | rowblock-compact context-null | interleaved-compact true |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| all | 18643 | interleaved > compact > rowblock | 0.035905643807607746 | 0.01603432842202976 | 0.02497467253991384 | -0.01572654131939787 | -0.01603432842202976 | 0.010997130240704912 | 0.020179102488209127 |
| non_generic | 16571 | interleaved > compact > rowblock | 0.030626433862044777 | 0.02005503416957627 | 0.02006076803273632 | -0.020833839655565605 | -0.02005503416957626 | 0.014771555301534192 | 0.009792594206479942 |
| candidate_broad | 5331 | interleaved > compact > rowblock | 0.06325550915250844 | 0.06123551680762168 | 0.0523363269845073 | -0.0009355109842374286 | -0.033090821163188316 | 0.02017353288283391 | 0.06231999816827209 |
| candidate_strict | 1569 | interleaved > compact > rowblock | 0.05044227996407713 | 0.14764022607584243 | 0.029725814398901665 | -0.007678884930951418 | -0.12723088660500018 | -0.01468215523738179 | 0.0427633950331262 |
| non_generic_local10 | 1599 | interleaved > compact > rowblock | 0.0662519935829522 | 0.1201749187716521 | 0.05301753504101586 | -0.009954332226678832 | -0.05737707467855719 | 0.046184359416988834 | 0.056297661356273815 |
| non_generic_local15 | 374 | rowblock > compact > interleaved | 0.08965971731745093 | 0.2298405496602789 | 0.08861240625747524 | 0.009476351427203194 | -0.14589987065702836 | -0.013796505596075346 | -0.08018336589024826 |
| non_generic_char50 | 1219 | interleaved > compact > rowblock | 0.08447058132595231 | 0.14153186715925398 | 0.029525006119653856 | -0.007225577330630856 | -0.14153186715925395 | -0.029525006119653863 | 0.07724500399532135 |
| comparative_all | 77 | rowblock > interleaved > compact | 0.19778388424159665 | 0.37115497325921987 | 0.2679774648957438 | 0.19778388424159646 | -0.37115497325921987 | -0.2679774648957438 | 0.10193745692732273 |

Family tables and exact paired deltas are in the JSON.

JSON: `experiments/archive/representation_and_objectives/data/relation_fourcell_synthesis/relation_fourcell_full_synthesis.json`
CSV: `experiments/archive/representation_and_objectives/data/relation_fourcell_synthesis/relation_fourcell_subset_table.csv`
