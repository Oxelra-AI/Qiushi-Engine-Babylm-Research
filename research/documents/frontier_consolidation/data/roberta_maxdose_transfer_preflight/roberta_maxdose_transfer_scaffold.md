# earlier analysis RoBERTa MAX-dose transfer scaffold

This is a CPU dry-run scaffold; no RoBERTa MAX-dose GPU training has been launched.

Question: Does the amplified 2.64x compact-dose effect transfer to a second bidirectional absolute-position MLM coordinate under the same fixed tokenizer and text pools?

## Launch rule after DeBERTa dose decomposition
- If MAX-minus-1x growth is carried by `view-repeat`, launch RoBERTa `view` + `repeat`.
- If growth is carried by source/freed-budget admission, launch RoBERTa `view` + `clean` for the total fixed-budget treatment transfer.

## Preflighted streams
- view: rows 653130 words 100000000 sha `11a5cf7281a4b9e4a281fd155aef3103825ade1489f8dead59e3c0225186137f` updates/batch256 2552; preflight `experiments/archive/frontier_consolidation/data/roberta_maxdose_transfer_preflight/view_preflight.json`
  - CPU smoke: returncode 0, finite_loss True, loss 9.758792877197266, masked_tokens 87
- repeat: rows 653130 words 100000000 sha `4cb41afc73bc07869df097050a0e1731a4b5ec90dc61f1c129e9158321cfa980` updates/batch256 2552; preflight `experiments/archive/frontier_consolidation/data/roberta_maxdose_transfer_preflight/repeat_preflight.json`
  - CPU smoke: returncode 0, finite_loss True, loss 9.758792877197266, masked_tokens 87
- clean: rows 653130 words 100000000 sha `64e686d16e0d7d5e81acecc73494d8670a1d6f8ddaffb4ce517277c983a9bed8` updates/batch256 2552; preflight `experiments/archive/frontier_consolidation/data/roberta_maxdose_transfer_preflight/clean_preflight.json`
  - CPU smoke: returncode 0, finite_loss True, loss 9.838220596313477, masked_tokens 55

JSON: `experiments/archive/frontier_consolidation/data/roberta_maxdose_transfer_preflight/roberta_maxdose_transfer_scaffold.json`
