# dual mechanism 20m overlap — scale1.75 100M Strict-Small compliance precheck (file-only)

This does not run model inference or produce an official score. It verifies the corpus-budget and tokenizer-provenance facts that would make the scale1.75 endpoint a legally certifiable Strict-Small candidate if the official evaluator later clears the target.

- Prechecks pass: `True`; errors: `[]`.
- Allowed 10M pool: rows 64740, words 10000000, SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23` (expected `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`).
- 100M training stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691` (expected `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`).
- Legal tokenizer trained only on the 10M pool (metadata pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`), vocab 16384.
- Endpoint tokenizer vocab equals training tokenizer vocab: `True` (endpoint 16384, training 16384). HF may reserialize tokenizer.json, so the vocabulary map is compared rather than the file SHA.
- Endpoint custom model package: architecture `['AdapterDebertaV2ForMaskedLM']`, AutoModelForMaskedLM `adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM`.
- Training source-word consumption: total 100000000 == word_exposure 100000000; illegal sources: `[]`.

Scientific consequence: the scale1.75 endpoint is trained on the exact legal 10M-pool-derived 100M stream, with a tokenizer fit only on that legal pool, and packages a custom-but-official-compatible MLM subclass. The endpoint would be submission-legal on provenance grounds; the remaining decisive fact is the official nine-column Overall from the managed evaluator.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_compliance_precheck/scale1p75_compliance_precheck.json`
