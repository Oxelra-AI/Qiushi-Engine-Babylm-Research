# official row association order/distance residual table

| target | direct adj R | direct then R | direct distance R | tf adj R | tf distance R | direct adj neg | direct dist neg | tf adj neg | tf dist neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k_base_100M_seed43022 | -1.2293 | -1.2991 | 0.7698 | -0.5544 | -0.4689 | 1.0000 | 0.0000 | 0.6375 | 0.7625 |
| scale1p75_82M | -0.6761 | -0.3751 | 1.0464 | -0.7987 | -0.7896 | 0.8125 | 0.0000 | 1.0000 | 1.0000 |
| scale1p75_100M | -0.7033 | -0.4330 | 0.9781 | -0.7173 | -0.7249 | 0.7375 | 0.0000 | 1.0000 | 1.0000 |
| legal40k_8x480_100M_seed43022 | -0.5857 | -0.5676 | 1.5747 | -0.9618 | -0.6732 | 0.9625 | 0.0000 | 0.8875 | 1.0000 |
| legal40k_depth12_100M_seed43022 | -0.3530 | -0.4057 | 0.9116 | -0.1274 | -0.4013 | 0.6250 | 0.0000 | 0.1875 | 0.7625 |
| fw_rowblock_100M_seed43022 | -0.1862 | -0.2033 | -0.1387 | -0.5909 | -0.5777 | 0.3000 | 0.1375 | 0.6250 | 0.6250 |
| fw_compact_100M_seed43022 | -0.8489 | -0.8741 | 0.3125 | -0.9372 | -1.0128 | 0.6500 | 0.1000 | 0.7000 | 0.6250 |
| mlm_only_20M | 0.0174 | 0.0395 | 0.4401 | -0.1652 | -0.1336 | 0.0000 | 0.0000 | 0.0000 | 0.0750 |
| coupled_shuffled_20M | 0.0474 | 0.0325 | 0.0926 | -0.0026 | -0.2162 | 0.0000 | 0.0000 | 0.0000 | 0.1750 |

JSON: `experiments/archive/representation_and_objectives/data/overwrite_decomposition/order_distance_residual_table.json`
