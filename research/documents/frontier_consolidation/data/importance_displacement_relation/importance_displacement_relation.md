# token value private readout synthesis — squared-gradient importance vs endpoint displacement

This bounded analysis uses legal-corpus MLM gradients from the frozen spatial repair route status endpoint. It is only evidence about MLM-sensitive displacement.

## Gradient sample
- rows sampled: `769`; batches `33`; mean loss `8.04704559210575`; masked tokens `165071`.
- source rows: `{'gutenberg': 96, 'simple_wiki': 96, 'qwen_pair_packed': 96, 'cleanqwen_fineweb_compact_view_reinvest': 96, 'childes': 96, 'switchboard': 96, 'open_subtitles': 96, 'bnc_spoken': 96, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1}`.

## Element-level normalized grad²/displacement relation
| route | sampled params | pearson(log grad², log delta²) | within-tensor shuffled null |
|---|---:|---:|---:|
| scale1p75_100M_stock_minus_step35_100M | 2796384 | -0.015807920249016314 | -0.00048731006764863606 |
| u256_100M_minus_step35_100M | 2796384 | -0.007357068681453663 | 0.0018614987650851138 |
| reference_100M_minus_step35_80M_normal_late_movement | 2796384 | -0.025253272049077322 | 0.0013712060657484794 |

## Top group movement/intensity

| route | group | delta2_fraction | grad2_fraction | delta/importance ratio | delta2_rel_to_ref2 |
|---|---|---:|---:|---:|---:|
| scale1p75_100M_stock_minus_step35_100M | embeddings | 0.16800856639115339 | 0.05404890317188972 | 3.1084546869867458 | 0.336948739051189 |
| scale1p75_100M_stock_minus_step35_100M | layer6_attention | 0.0757383912333278 | 0.10961371012654089 | 0.6909572821309803 | 0.828163054783411 |
| scale1p75_100M_stock_minus_step35_100M | layer7_attention | 0.06570810823495997 | 0.12166213675623348 | 0.5400867516129118 | 0.6253751680434001 |
| scale1p75_100M_stock_minus_step35_100M | layer5_attention | 0.060439980304085275 | 0.09417028829147184 | 0.641815814739932 | 0.6657944882944246 |
| scale1p75_100M_stock_minus_step35_100M | layer7_ffn_output | 0.054827299517460674 | 0.052640927211084876 | 1.0415336967300872 | 0.6741981170310827 |
| scale1p75_100M_stock_minus_step35_100M | layer6_ffn_output | 0.05404663434856745 | 0.019320739607152426 | 2.7973377545319074 | 0.6790046255839928 |
| scale1p75_100M_stock_minus_step35_100M | layer5_ffn_output | 0.052388429008198664 | 0.022789839901242764 | 2.298762485178399 | 0.6609827933326999 |
| scale1p75_100M_stock_minus_step35_100M | layer1_attention | 0.0518200610441133 | 0.036393874787575775 | 1.423867652086437 | 0.606250879208339 |
| u256_100M_minus_step35_100M | embeddings | 0.22273121821423914 | 0.05404890317188972 | 4.120920224891434 | 0.6014496741836304 |
| u256_100M_minus_step35_100M | layer5_attention | 0.0650064701818391 | 0.09417028829147184 | 0.6903076475738702 | 0.9641803098664212 |
| u256_100M_minus_step35_100M | layer7_attention | 0.06456394443015818 | 0.12166213675623348 | 0.5306823153987569 | 0.8273657065600534 |
| u256_100M_minus_step35_100M | layer2_attention | 0.05962704688309867 | 0.08342771074738266 | 0.7147151270115526 | 0.913195924263958 |
| u256_100M_minus_step35_100M | layer6_attention | 0.057524960263213604 | 0.10961371012654089 | 0.5247971279943476 | 0.8469190383360921 |
| u256_100M_minus_step35_100M | layer4_attention | 0.05141289537146099 | 0.11080543565464528 | 0.4639925385222346 | 0.8208318195005825 |
| u256_100M_minus_step35_100M | layer7_ffn_output | 0.0482656458157668 | 0.052640927211084876 | 0.916884416230481 | 0.7991247718130109 |
| u256_100M_minus_step35_100M | layer1_attention | 0.04702163392137894 | 0.036393874787575775 | 1.292020544551395 | 0.7406925378060039 |
| reference_100M_minus_step35_80M_normal_late_movement | embeddings | 0.2523877549450785 | 0.05404890317188972 | 4.66961844058924 | 0.0005853973370099853 |
| reference_100M_minus_step35_80M_normal_late_movement | layer7_ffn_output | 0.05618762310692839 | 0.052640927211084876 | 1.0673752550296391 | 0.0007985258031689688 |
| reference_100M_minus_step35_80M_normal_late_movement | layer0_ffn_output | 0.054105584571870334 | 0.008111775209452976 | 6.670005414945293 | 0.0009083616675887039 |
| reference_100M_minus_step35_80M_normal_late_movement | layer4_ffn_output | 0.0539650791349742 | 0.02890403244552431 | 1.8670432659070206 | 0.0007937673298001205 |
| reference_100M_minus_step35_80M_normal_late_movement | layer1_ffn_output | 0.05371498819590076 | 0.012909416172718137 | 4.160915371945192 | 0.0008525798973323816 |
| reference_100M_minus_step35_80M_normal_late_movement | layer5_ffn_output | 0.05365346086701458 | 0.022789839901242764 | 2.3542710742820434 | 0.0007823716276880402 |
| reference_100M_minus_step35_80M_normal_late_movement | layer2_ffn_output | 0.053636020618208595 | 0.024742602636681925 | 2.1677598515319882 | 0.0008105133659290364 |
| reference_100M_minus_step35_80M_normal_late_movement | layer3_ffn_output | 0.053298557891043055 | 0.02791793910567893 | 1.9091150564262576 | 0.0007947283851879366 |

## Scientific reading
Compare closed-route displacement to normal spatial repair route status late movement. If a closed route's displacement is not more enriched in high-MLM-importance directions than normal late movement, a simple importance-damping mechanism is weak; if it is enriched, protection may be mechanistically motivated but still needs token-structure and score-sentinel evidence.

Group CSV: `experiments/archive/frontier_consolidation/data/importance_displacement_relation/importance_displacement_group_rows.csv`
Decile CSV: `experiments/archive/frontier_consolidation/data/importance_displacement_relation/importance_displacement_deciles.csv`
JSON: `experiments/archive/frontier_consolidation/data/importance_displacement_relation/importance_displacement_relation.json`
