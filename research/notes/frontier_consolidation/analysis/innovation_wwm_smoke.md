# CPU smoke test: target-mass-matched innovation-biased WWM

Status: **INNOVATION_WWM_CPU_SMOKE_PASS**. CPU only; no model update, evaluation, corpus edit, tokenizer edit, or GPU use.

## Exact mass and targeting

Across 24,040 row exposures, baseline and treatment selected 527,935 WWM groups / 764,428 tokens. Group delta=0, token delta=0, per-batch mismatches=0/0. Per-row group/token changes=17079/17079 because donors are deliberately taken elsewhere in the batch.
The mechanism completed 10,183 exact same-length swaps from 12,016 proposals; affected eligible changed-row fraction=0.8475. Proposal collision rate=0.1516; candidate-group baseline collision rate=0.1486; no-donor count=11.
Selected innovation groups rose from 15,637 to 25,820 (+10,183), which is 1.9288% of baseline selected-group mass.

## Exclusion, source context, and leakage audit

Forced copyable targets=0; protected-source donors=0; protected-pair donors=0; ordinary-row donor fraction=1.0000; rows whose paired-source selection changed=0; ordinary rows changed=6896.
Any paired-source visibility for forced targets=1.0000; fully visible paired source=0.0403.
All-piece selection/label coverage=1.0000. Under the unchanged tokenwise 80/10/10 corruption, 0.1298 of forced groups had at least one intentional gold keep-branch piece, and 0.7468 were entirely mask-branch.

## Exposure accounting

Baseline and treatment used the same 3,559,420 whitespace-word exposure in this smoke. Rows, words, input ids, attention masks, batch shape, batch-selected groups, and batch-selected tokens are invariant. Hash-based proposal/donor choice consumes no PyTorch masking RNG; the state matched the baseline through WWM selection.

Full JSON and sampled events are in the sibling work artifacts.
