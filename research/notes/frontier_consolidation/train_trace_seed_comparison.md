# aoa overall update — compact_view_reinvest seed training-trace comparison

CPU/file-only comparison of seed43022 and seed43122 run metadata, checkpoint exposures, training logs, and sparse training dynamics. No model evaluation is performed here.

## Main run-level differences
- loss_first: 43022=9.809807777404785 ; 43122=9.822142601013184
- loss_last: 43022=2.565617084503174 ; 43122=2.6415250301361084

## Loss differences (43122 minus 43022)
- first: +0.0123348
- last: +0.0759079
- mean: -0.00358528
- min: -0.000251532
- max: +0.0110445
- last10_mean: +0.00347264
- last50_mean: -0.000542836

## Selected checkpoint exposure differences (43122 minus 43022)
- chck_100M: 0
- chck_10M: 0
- chck_1M: 0
- chck_40M: 0

## Selected dynamics rows
- chck_16M: entropy diff 0.05393528938293457, loss-band diff {'high': -0.02934741973876953, 'low': 0.047405242919921875, 'mid': 0.02434539794921875}, acc-band diff {'high': -0.007854878902435303, 'low': 0.011821269989013672, 'mid': -0.0032232776284217834}
- chck_24M: entropy diff 0.07203340530395508, loss-band diff {'high': -0.0440521240234375, 'low': 0.2419595718383789, 'mid': -0.05257892608642578}, acc-band diff {'high': 0.01052713394165039, 'low': -0.014931540936231613, 'mid': 0.025619052350521088}
- chck_32M: entropy diff -0.1337580680847168, loss-band diff {'high': -0.16562795639038086, 'low': -0.15405607223510742, 'mid': -0.0751194953918457}, acc-band diff {'high': 0.024367094039916992, 'low': 0.017346903681755066, 'mid': 0.002975448966026306}
- chck_40M: entropy diff 0.009621381759643555, loss-band diff {'high': 0.004838228225708008, 'low': 0.0368199348449707, 'mid': -0.199127197265625}, acc-band diff {'high': 0.0016715526580810547, 'low': 0.020425081253051758, 'mid': 0.026660174131393433}
- chck_64M: entropy diff 0.12165617942810059, loss-band diff {'high': -0.005574345588684082, 'low': 0.027690410614013672, 'mid': 0.09789896011352539}, acc-band diff {'high': -0.004696190357208252, 'low': -0.011759549379348755, 'mid': -0.025879472494125366}
- chck_72M: entropy diff 0.013197898864746094, loss-band diff {'high': -0.0075408220291137695, 'low': 0.24346208572387695, 'mid': -0.3087129592895508}, acc-band diff {'high': -0.0059542059898376465, 'low': -0.02412515878677368, 'mid': 0.03511536121368408}
- chck_80M: entropy diff 0.011376380920410156, loss-band diff {'high': -0.001708984375, 'low': 0.17128562927246094, 'mid': 0.09213399887084961}, acc-band diff {'high': 0.004158377647399902, 'low': -0.023317843675613403, 'mid': 0.015772879123687744}
- chck_87M: entropy diff -0.00974583625793457, loss-band diff {'high': 0.018095016479492188, 'low': 0.12955665588378906, 'mid': -0.010417938232421875}, acc-band diff {'high': -0.003618180751800537, 'low': -0.013414472341537476, 'mid': 0.007174462080001831}
- chck_8M: entropy diff 0.3252086639404297, loss-band diff {'high': 0.14416289329528809, 'low': -0.2775144577026367, 'mid': -0.13516521453857422}, acc-band diff {'high': -0.005290031433105469, 'low': 0.001151952426880598, 'mid': 0.0060509853065013885}
- chck_95M: entropy diff -0.10569548606872559, loss-band diff {'high': -0.07346236705780029, 'low': -0.3408784866333008, 'mid': -0.15899896621704102}, acc-band diff {'high': 0.018156826496124268, 'low': 0.03062725067138672, 'mid': 0.004981577396392822}

## Reading
The two runs share the same checkpoint exposure schedule and corpus path. The final MLM loss is higher for seed43122, but this scalar alone cannot explain which downstream abilities diverged. Use the sparse task-slice temporal probe to determine whether seed43122 is behind early or loses specific abilities later.

Machine-readable output: `experiments/archive/frontier_consolidation/data/train_trace_seed_comparison/train_trace_seed_comparison.json`
