# dense and causal evaluation refined plan dense training trajectory audit

This CPU/file-only audit inspects the completed asset freeze and principles DeBERTa dense runs while the causal GPT arms occupy both H100s. It does not score checkpoints.

## What the two runs actually vary

### scale1p75_seed43122_dense
- adapter_scale=1.75 extra_init_seed=43122 train_rng_seed=43123
- checkpoints=50 final_loss=2.6495566368103027 steps=2529
- selected hashes: {"chck_20M": "c4fed049f7ac4ac142469b3e666e5a55e2c08f24623fc63bb76e04a441694785", "chck_80M": "d1ed917366c87eabc925344be076f630f4a817e94989b9b06f2067010d724063", "chck_82M": "aa7e8adaf9f37057a6ba859e3fcd8e31fcc17a8f6483a009d0b12112b6e64ae3", "chck_100M": "477efde881e93e89769a27494005f2778d057fc3a815fd4c5139affaf3232854"}

### scale1p25_seed43022_dense
- adapter_scale=1.25 extra_init_seed=43022 train_rng_seed=43023
- checkpoints=50 final_loss=2.5401194095611572 steps=2529
- selected hashes: {"chck_20M": "74b83143184c207f981d25c826fa6babb37e9a783aff5f79b6355b9313fc246f", "chck_80M": "3ed7db752f8ab9b4b96b4c4aa18a2ad8de1076310ce4453932a4e5469e1f6cb5", "chck_82M": "3acb66331a6c04c821b2dc3319e1dfaabffcab631a2251a14f05f62ea1c8ec60", "chck_100M": "5460481d3ccd1dfd9ba2642dc0e9889e8af700243620fff9cadc05ed9b160398"}

## Log-stream comparisons

- scale1p25_seed43022_vs_reference_scale1p75_seed43022: same words rows 2529/2529, same mask rows 2529/2529, same loss rows 2/2529, last loss diff -0.004061460494995117
- scale1p75_seed43122_vs_reference_scale1p75_seed43022: same words rows 2529/2529, same mask rows 6/2529, same loss rows 0/2529, last loss diff 0.10537576675415039
- scale1p75_seed43122_vs_scale1p25_seed43022: same words rows 2529/2529, same mask rows 6/2529, same loss rows 0/2529, last loss diff 0.10943722724914551

The key correction is that the scale1.75 cross-seed run changes both initialization and train RNG relative to the protected seed43022 reference, while scale1.25 preserves the protected seed stream and changes only adapter amplitude.

## Loss trajectory snapshots

### scale1p75_seed43022_reference
- 70M: loss=2.539532, roll50=2.597383, lr=0.000231, words=69990941
- 76M: loss=2.533445, roll50=2.539828, lr=0.000152, words=76001011
- 80M: loss=2.533707, roll50=2.518844, lr=0.000108, words=79994979
- 82M: loss=2.511317, roll50=2.487752, lr=8.76e-05, words=82012495
- 86M: loss=2.580469, roll50=2.470851, lr=5.37e-05, words=86006729
- 90M: loss=2.514952, roll50=2.464937, lr=2.77e-05, words=89999404
- 94M: loss=2.478794, roll50=2.456787, lr=1e-05, words=93993033
- 100M: loss=2.544181, roll50=2.453469, lr=0, words=100000000

### scale1p75_seed43122_dense
- 70M: loss=2.593770, roll50=2.582513, lr=0.000231, words=69990941
- 76M: loss=2.571841, roll50=2.541944, lr=0.000152, words=76001011
- 80M: loss=2.407065, roll50=2.526113, lr=0.000108, words=79994979
- 82M: loss=2.553424, roll50=2.487595, lr=8.76e-05, words=82012495
- 86M: loss=2.555208, roll50=2.472047, lr=5.37e-05, words=86006729
- 90M: loss=2.533234, roll50=2.470137, lr=2.77e-05, words=89999404
- 94M: loss=2.491201, roll50=2.466531, lr=1e-05, words=93993033
- 100M: loss=2.649557, roll50=2.456694, lr=0, words=100000000

### scale1p25_seed43022_dense
- 70M: loss=2.535034, roll50=2.599937, lr=0.000231, words=69990941
- 76M: loss=2.537837, roll50=2.542378, lr=0.000152, words=76001011
- 80M: loss=2.530380, roll50=2.522997, lr=0.000108, words=79994979
- 82M: loss=2.521034, roll50=2.492610, lr=8.76e-05, words=82012495
- 86M: loss=2.577450, roll50=2.477062, lr=5.37e-05, words=86006729
- 90M: loss=2.518466, roll50=2.470525, lr=2.77e-05, words=89999404
- 94M: loss=2.486758, roll50=2.462337, lr=1e-05, words=93993033
- 100M: loss=2.540119, roll50=2.460054, lr=0, words=100000000

## Selected GPU scoring plan

### scale1p75_seed43122_dense
- first endpoints: chck_70M, chck_76M, chck_78M, chck_80M, chck_82M, chck_84M, chck_86M, chck_90M, chck_100M
- command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=<GPU> python -B experiments/archive/frontier_consolidation/scripts/batch_trajectory_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M --gpu <GPU> --min-words 70000000 --max-words 100000000 --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense --label scale1p75_seed43122_dense`
- note: The current batch evaluator evaluates every 2M checkpoint in the range; if GPU time is tight, build an endpoint-list wrapper to evaluate only selected_endpoints_first.

### scale1p25_seed43022_dense
- first endpoints: chck_70M, chck_76M, chck_80M, chck_82M, chck_86M, chck_90M, chck_94M, chck_100M
- command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=<GPU> python -B experiments/archive/frontier_consolidation/scripts/batch_trajectory_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p25_seed43022_dense100M --gpu <GPU> --min-words 70000000 --max-words 100000000 --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense --label scale1p25_seed43022_dense`
- note: The current batch evaluator evaluates every 2M checkpoint in the range; if GPU time is tight, build an endpoint-list wrapper to evaluate only selected_endpoints_first.

JSON: `experiments/archive/frontier_consolidation/data/dense_training_trajectory_audit/dense_training_trajectory_audit.json`
