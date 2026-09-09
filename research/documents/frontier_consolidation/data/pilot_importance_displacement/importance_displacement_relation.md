# token value private readout synthesis — squared-gradient importance vs endpoint displacement

This bounded analysis uses legal-corpus MLM gradients from the frozen spatial repair route status endpoint. It is only evidence about MLM-sensitive displacement.

## Gradient sample
- rows sampled: `17`; batches `5`; mean loss `7.945846271514893`; masked tokens `3511`.
- source rows: `{'qwen_pair_packed': 2, 'childes': 2, 'gutenberg': 2, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1, 'open_subtitles': 2, 'simple_wiki': 2, 'bnc_spoken': 2, 'switchboard': 2, 'cleanqwen_fineweb_compact_view_reinvest': 2}`.

## Element-level normalized grad²/displacement relation
| route | sampled params | pearson(log grad², log delta²) | within-tensor shuffled null |
|---|---:|---:|---:|
| scale1p75_100M_stock_minus_step35_100M | 121640 | -0.013840025919370353 | -0.0017576173991524443 |
| u256_100M_minus_step35_100M | 121640 | -0.00022708471561891993 | 0.0026450432011164296 |
| reference_100M_minus_step35_80M_normal_late_movement | 121640 | -0.020236121113642883 | 0.0025087641696184357 |

## Top group movement/intensity

| route | group | delta2_fraction | grad2_fraction | delta/importance ratio | delta2_rel_to_ref2 |
|---|---|---:|---:|---:|---:|
| scale1p75_100M_stock_minus_step35_100M | embeddings | 0.16800856639115339 | 0.07153507878093296 | 2.3486178984391444 | 0.336948739051189 |
| scale1p75_100M_stock_minus_step35_100M | layer6_attention | 0.0757383912333278 | 0.10594104839252128 | 0.7149107204670104 | 0.828163054783411 |
| scale1p75_100M_stock_minus_step35_100M | layer7_attention | 0.06570810823495997 | 0.10998821253475048 | 0.5974104562722998 | 0.6253751680434001 |
| scale1p75_100M_stock_minus_step35_100M | layer5_attention | 0.060439980304085275 | 0.08992657952309431 | 0.6721036274771632 | 0.6657944882944246 |
| scale1p75_100M_stock_minus_step35_100M | layer7_ffn_output | 0.054827299517460674 | 0.04492482149457203 | 1.2204233137372642 | 0.6741981170310827 |
| scale1p75_100M_stock_minus_step35_100M | layer6_ffn_output | 0.05404663434856745 | 0.01915750756745362 | 2.8211725433629167 | 0.6790046255839928 |
| scale1p75_100M_stock_minus_step35_100M | layer5_ffn_output | 0.052388429008198664 | 0.022750375472773265 | 2.3027500829995105 | 0.6609827933326999 |
| scale1p75_100M_stock_minus_step35_100M | layer1_attention | 0.0518200610441133 | 0.042927930964751504 | 1.2071408959044216 | 0.606250879208339 |
| u256_100M_minus_step35_100M | embeddings | 0.22273121821423914 | 0.07153507878093296 | 3.1135943653088725 | 0.6014496741836304 |
| u256_100M_minus_step35_100M | layer5_attention | 0.0650064701818391 | 0.08992657952309431 | 0.7228838295261146 | 0.9641803098664212 |
| u256_100M_minus_step35_100M | layer7_attention | 0.06456394443015818 | 0.10998821253475048 | 0.5870078524074512 | 0.8273657065600534 |
| u256_100M_minus_step35_100M | layer2_attention | 0.05962704688309867 | 0.08865002923292856 | 0.6726117001769756 | 0.913195924263958 |
| u256_100M_minus_step35_100M | layer6_attention | 0.057524960263213604 | 0.10594104839252128 | 0.5429902869195552 | 0.8469190383360921 |
| u256_100M_minus_step35_100M | layer4_attention | 0.05141289537146099 | 0.10646755030132996 | 0.4828973262364876 | 0.8208318195005825 |
| u256_100M_minus_step35_100M | layer7_ffn_output | 0.0482656458157668 | 0.04492482149457203 | 1.0743647767548372 | 0.7991247718130109 |
| u256_100M_minus_step35_100M | layer1_attention | 0.04702163392137894 | 0.042927930964751504 | 1.0953622237230303 | 0.7406925378060039 |
| reference_100M_minus_step35_80M_normal_late_movement | embeddings | 0.2523877549450785 | 0.07153507878093296 | 3.5281677080133473 | 0.0005853973370099853 |
| reference_100M_minus_step35_80M_normal_late_movement | layer7_ffn_output | 0.05618762310692839 | 0.04492482149457203 | 1.2507033136173322 | 0.0007985258031689688 |
| reference_100M_minus_step35_80M_normal_late_movement | layer0_ffn_output | 0.054105584571870334 | 0.010235238136116322 | 5.286206715694478 | 0.0009083616675887039 |
| reference_100M_minus_step35_80M_normal_late_movement | layer4_ffn_output | 0.0539650791349742 | 0.028742655890590305 | 1.8775258396577452 | 0.0007937673298001205 |
| reference_100M_minus_step35_80M_normal_late_movement | layer1_ffn_output | 0.05371498819590076 | 0.015232925897751716 | 3.52624233561254 | 0.0008525798973323816 |
| reference_100M_minus_step35_80M_normal_late_movement | layer5_ffn_output | 0.05365346086701458 | 0.022750375472773265 | 2.358354961271953 | 0.0007823716276880402 |
| reference_100M_minus_step35_80M_normal_late_movement | layer2_ffn_output | 0.053636020618208595 | 0.026889422585107636 | 1.99468844853196 | 0.0008105133659290364 |
| reference_100M_minus_step35_80M_normal_late_movement | layer3_ffn_output | 0.053298557891043055 | 0.028894688642014756 | 1.8445797617470598 | 0.0007947283851879366 |

## Scientific reading
Compare closed-route displacement to normal spatial repair route status late movement. If a closed route's displacement is not more enriched in high-MLM-importance directions than normal late movement, a simple importance-damping mechanism is weak; if it is enriched, protection may be mechanistically motivated but still needs token-structure and score-sentinel evidence.

Group CSV: `experiments/archive/frontier_consolidation/data/pilot_importance_displacement/importance_displacement_group_rows.csv`
Decile CSV: `experiments/archive/frontier_consolidation/data/pilot_importance_displacement/importance_displacement_deciles.csv`
JSON: `experiments/archive/frontier_consolidation/data/pilot_importance_displacement/importance_displacement_relation.json`
