# dualview panel readiness and budget dual-view budget substitution audit

Aligned/shuffled main exposure: `19021584`; aux exposure `966720`; charged `19988304`.
mlm_only main/charged exposure: `20000000`; updates `506`.
First `481` / `481` common updates match in batch/masked-token geometry.
Displaced main rows: `6344`; displaced words `978416`.
Displaced sources: `{'childes': 241440, 'gutenberg': 189440, 'open_subtitles': 176960, 'qwen_pair_packed': 169980, 'simple_wiki': 101440, 'bnc_spoken': 54880, 'cleanqwen_fineweb_compact_view_reinvest': 41556, 'switchboard': 2720}`.
Extra mlm_only pair rows: `295`; constituent pairs `1173`.

Dual-view spends ~0.97M legal words on source-conditioned/free auxiliary pair views; under the charged cap this replaces the final ~0.98M main-stream words seen by mlm_only. First 481 main updates share the same batch/mask geometry; mlm_only then receives additional main batches 482-506.
JSON: `experiments/archive/frontier_consolidation/data/dualview_budget_substitution/budget_substitution_audit.json`
