
# aoa mincontext discrepancy audit AoA 6560-vs-8005 discrepancy audit

CPU audit JSON: `experiments/archive/representation_and_objectives/data/aoa_discrepancy_audit/aoa_mincontext_discrepancy_audit.json`  
Omitted words/contexts JSONL: `experiments/archive/representation_and_objectives/data/aoa_discrepancy_audit/omitted_by_min_context20_words.jsonl`

## Finding

The local compact reinvest full eval summary AoA helper used `load_eval(word_path, 20, False)`. The official AoA runner default is `--min_context 0`, and the official collator has `AOA_SIZE = 8005`.

Counting the official `cdi_childes.json`:

- official min_context=0: 504 words, 8005 context rows per checkpoint.
- local helper min_context=20: 328 words, 6560 context rows per checkpoint.
- omitted by the helper: 176 words and 1445 context rows per checkpoint.

compact reinvest full eval summary recorded `[6560]` rows per checkpoint and AoA `0.0` with curve record `{'curve_fitness': 0.0, 'n_words': 203}`. That exactly matches the helper-side `min_context=20` count, not the official collator.

## Interpretation

The discrepancy is evaluator configuration, not training-data selection: the official AoA data are present locally and have 8005 context rows. The missing rows are the official low-context CDI words omitted by the helper. Because AoA curve fitness is word-set sensitive and p-gated, the 42.0868 endpoint remains provisional until the official-min_context=0 AoA rerun returns and the result is reproduced through the official collation path.
