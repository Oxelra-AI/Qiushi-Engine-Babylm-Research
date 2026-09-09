# earlier analysis shuffled frozen-tail endpoint validation

Status: **PASS_REAL_LOADABLE_LEGAL_ENDPOINT_HYPOTHESIS**
Route read: `launch_deterministic_replay_and_score_repeat`

## Score arithmetic and legal exposure
- Projected Overall with AoA=0: `41.984020152367975`
- Protected chck_82M Overall: `41.942481167385985`
- Margin vs protected: `0.04153898498199027`
- Cheap7: `44.01285714285714`; SuperGLUE: `69.7661813713118`; AoA scalar: `0.0`
- Total consumed words: `86005413` / `100000000`; within cap: `True`
- Tail charged words: `3992918` (main `3954265`, auxiliary `38653`)

## Function and loading checks
- Trusted class: `FrozenSlowPrivateDebertaV2ForMaskedLM`; parameters: `36458592`
- Protected class: `AdapterDebertaV2ForMaskedLM`; parameters: `35463008`
- Non-private tensor max abs diff vs protected: `0.0` across `35463008` elements
- Private tensors: `48` tensors, `995584` params, RMS `0.06516943556900559`, abs max `1.0363988876342773`
- Private ON vs OFF max logit diff on smoke batch: `10.41353702545166`
- Private OFF vs protected max logit diff on smoke batch: `0.0`
- Non-trust AutoModel load: `{'load_ok': True, 'class': 'DebertaV2ForMaskedLM', 'param_count': 34467424, 'is_scored_function': False}`

## AoA ranking assumption
- Scalar AoA=0 accepted by current validator logic: `True`

Scientific reading: the shuffled tail is a real generic private-tail endpoint hypothesis if and only if trusted load, legal exposure, frozen slow-path equality, active private-path evidence, scalar AoA=0 acceptance, and score arithmetic all pass. It is not evidence for source-correspondence transfer; that route remains closed by aligned<shuffled score and worse source-free NLL.

JSON: `experiments/archive/frontier_consolidation/data/shuffled_tail_endpoint_validation/shuffled_tail_endpoint_validation.json`
