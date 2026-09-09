# u256 endpoint mechanism — U256 endpoint readiness and HF-load smoke

Status: `U256_READINESS_SMOKE_DONE`; ok=True

## Readiness
- run: `experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_100M`
- arm: U256; words: 100000000; steps: 2530
- losses: 9.826857208144903 -> 2.516624725910071
- raw tokens/epoch: 14664519; checkpoints: 100
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- stream SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- endpoint tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- readiness_ok=True; errors=[]

## HF load smoke
- model class: DebertaV2ForMaskedLM; params: 34467424; finite_logits=True; probe_loss=13.028825759887695
- writable caches: `experiments/archive/frontier_consolidation/data/u256_endpoint_readiness_smoke/hf_cache`, `experiments/archive/frontier_consolidation/data/u256_endpoint_readiness_smoke/hf_modules`
- top mask tokens: [{'token_id': 16, 'token': ',', 'prob': 0.13226023316383362}, {'token_id': 527, 'token': 'then', 'prob': 0.12166931480169296}, {'token_id': 21, 'token': '1', 'prob': 0.07338903099298477}, {'token_id': 1960, 'token': 'minutes', 'prob': 0.06989819556474686}, {'token_id': 23, 'token': '3', 'prob': 0.06480786204338074}]

JSON: `experiments/archive/frontier_consolidation/data/u256_endpoint_readiness_smoke/u256_endpoint_readiness_smoke.json`
