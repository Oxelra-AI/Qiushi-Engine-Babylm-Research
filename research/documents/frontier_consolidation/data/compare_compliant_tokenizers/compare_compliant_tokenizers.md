# bytealphabet tokenizer repair and retrain priority comparison of two legal same-pool compliant tokenizers

A = spatial repair route status current-running legal tokenizer; B = bytealphabet tokenizer repair and retrain priority legal byte-alphabet tokenizer trained on the same 10M pool with `initial_alphabet=ByteLevel.alphabet()`. Evaluation text is used only for coverage/length auditing.

## Tokenizer identity
- current_running: SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, vocab=16384, missing_bytelevel_alphabet=71
- bytealphabet: SHA `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`, vocab=16384, missing_bytelevel_alphabet=0
- Shared token strings: 16313/16384 = 0.995667

## train_pool_full total
strings=64740, A_tokens=14793999, B_tokens=14798756, B/A=1.00032155, A_unk=0, B_unk=0, A_trunc>15967, B_trunc>15999, mean_B_minus_A=0.0735, mean_abs_diff=0.0735

### train_pool_full by family
- train_pool: strings=64740, B/A=1.000322, A_unk=0, B_unk=0, A_trunc=15967, B_trunc=15999, mean_B-A=0.0735

## eval_scored_text total
strings=416634, A_tokens=18223430, B_tokens=18236712, B/A=1.00072884, A_unk=1513, B_unk=0, A_trunc>21917, B_trunc>21921, mean_B_minus_A=0.0319, mean_abs_diff=0.0319

### eval_scored_text by family
- BLiMP: strings=119750, B/A=1.000089, A_unk=0, B_unk=0, A_trunc=0, B_trunc=0, mean_B-A=0.0011
- EWoK: strings=53326, B/A=1.011924, A_unk=0, B_unk=0, A_trunc=0, B_trunc=0, mean_B-A=0.1012
- Reading_AoA: strings=16934, B/A=1.000428, A_unk=44, B_unk=0, A_trunc=1, B_trunc=1, mean_B-A=0.0056
- SuperGLUE: strings=216188, B/A=1.000472, A_unk=523, B_unk=0, A_trunc=21916, B_trunc=21920, mean_B-A=0.0347
- Supplement: strings=10436, B/A=1.000868, A_unk=946, B_unk=0, A_trunc=0, B_trunc=0, mean_B-A=0.0161

## Scientific interpretation
The byte-alphabet tokenizer removes the `<unk>` coverage failure on both the 10M pool and likely scored evaluation strings. Its training-pool token count differs only modestly from the current tokenizer, so it is a legal tokenizer-construction repair rather than a changed corpus route. If the current running retrain performs poorly in Supplement/Reading/SuperGLUE where `<unk>` appears, rerunning the frozen recipe with this byte-alphabet tokenizer is a scientifically justified repair.

Full JSON: `experiments/archive/frontier_consolidation/data/compare_compliant_tokenizers/compare_compliant_tokenizers.json`
