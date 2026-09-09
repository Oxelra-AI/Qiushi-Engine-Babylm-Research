# CPU smoke test: target-mass-matched innovation-biased WWM

Status: **INNOVATION_WWM_CPU_SMOKE_PASS**. CPU only; no model update, evaluation, corpus edit, tokenizer edit, or GPU use.

## Exact mass and targeting

Across 512 row exposures, baseline and treatment selected 11,028 WWM groups / 15,735 tokens. Group delta=0, token delta=0, per-batch mismatches=0/0. Per-row group/token changes=341/341 because donors are deliberately taken elsewhere in the batch.
The mechanism completed 201 exact same-length swaps from 256 proposals; affected eligible changed-row fraction=0.7852. Proposal collision rate=0.2109; candidate-group baseline collision rate=0.1490; no-donor count=1.
Selected innovation groups rose from 351 to 552 (+201), which is 1.8226% of baseline selected-group mass.

## Exclusion, source context, and leakage audit

Forced copyable targets=0; protected-source donors=0; protected-pair donors=0; ordinary-row donor fraction=1.0000; rows whose paired-source selection changed=0; ordinary rows changed=140.
Any paired-source visibility for forced targets=1.0000; fully visible paired source=0.0299.
All-piece selection/label coverage=1.0000. Under the unchanged tokenwise 80/10/10 corruption, 0.1343 of forced groups had at least one intentional gold keep-branch piece, and 0.7164 were entirely mask-branch.

## Exposure accounting

Baseline and treatment used the same 75,702 whitespace-word exposure in this smoke. Rows, words, input ids, attention masks, batch shape, batch-selected groups, and batch-selected tokens are invariant. Hash-based proposal/donor choice consumes no PyTorch masking RNG; the state matched the baseline through WWM selection.

Full JSON and sampled events are in the sibling work artifacts.
