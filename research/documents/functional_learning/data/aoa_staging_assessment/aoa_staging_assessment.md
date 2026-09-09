# evaluation repair synthesis AoA Staging Assessment

## Convention

BabyLM Strict-Small early-stopping: checkpoints up to the amount trained.
Full revisions: 19, Early-stop revisions: 17

## Ancestral Checkpoints

Available: 17/17 from earlier analysis ladder

- ✓ chck_1M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_2M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_3M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_4M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_5M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_6M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_7M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_8M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_9M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_10M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_20M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_30M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_40M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_50M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_60M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_70M: arch=['AdapterDebertaV2ForMaskedLM']
- ✓ chck_80M: arch=['AdapterDebertaV2ForMaskedLM']

## Endpoints

- ✓ coherent86: arch=['FrozenSlowPrivateDebertaV2ForMaskedLM']
- ✓ dense_seed62064: arch=['FrozenSlowPrivateDebertaV2ForMaskedLM']
- ✓ dense_seed62065: arch=['FrozenSlowPrivateDebertaV2ForMaskedLM']

## Findings

- All 17 ancestral checkpoints available from earlier analysis ladder
- Ancestral architectures: {'AdapterDebertaV2ForMaskedLM'}
- Ancestral checkpoints are stock DebertaV2ForMaskedLM + slow adapter
- Final endpoints are FrozenSlowPrivateDebertaV2ForMaskedLM + both adapters
- Architecture change at 82M is genuine training history, not reconstruction
- Coherent86 total words: ~86,005,295 → last 10M milestone: 80M → 17 checkpoints needed
- Dense total words: ~89,168,037 → last 10M milestone: 80M → 17 checkpoints needed
- Both share identical ancestral ladder; only final endpoint differs
- AoA difference between coherent86 and dense will be very small (same trajectory through 80M)

## Next related experiments. Write custom AoA runner that loads each checkpoint from its own directory
2. Stage symlinks to ancestral checkpoints + repaired endpoints
3. Run AoA surprisal computation on all staged checkpoints
4. Compute AoA score using official AoAEvaluator
