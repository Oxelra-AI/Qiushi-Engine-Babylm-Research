# triangle oom repair and gc preflight — compact-view triangle OOM repair and activation-checkpoint preflight

## Failure readout from earlier analysis

Both missing matched triangle arms failed before producing `scientific_metrics.json`:

- `compact_repeat_reinvest`: `experiments/archive/representation_and_objectives/training/runs/compact_repeat_reinvest_16k_seed43022/train_stderr.log`
- `adjbreak_reinvest`: `experiments/archive/representation_and_objectives/training/runs/adjbreak_reinvest_16k_seed43022/train_stderr.log`

The error in both files is the same operational CUDA OOM inside `loss.backward()` in `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`, line 878. The failed process had about 58.09 GiB allocated by PyTorch and only ~1.19 GiB free on the visible device at allocation time. The runs had already passed hash/data preflight; the failure is not evidence about the compact-view mechanism.

A live post-failure GPU check showed both H100s with ~19.5 GiB resident memory and no listed compute processes under `nvidia-smi -q -d PIDS,UTILIZATION,MEMORY`; therefore the available free memory is about 61.5 GiB rather than the full 79 GiB. The original batch256 trainer can OOM under this resident-memory condition.

## Repair chosen

Wrote `experiments/archive/representation_and_objectives/scripts/gradient_checkpointed_masking_curriculum_trainer.py`, a narrow wrapper around the COMPACT_EXPERIENCE trainer. It imports `masking_curriculum_trainer.py` and only replaces `build_model()` so the created DeBERTa-v2 MLM calls `gradient_checkpointing_enable()` and disables `use_cache` if present. It inherits the command-line interface, data loading, WWM masking, optimizer, scheduler, seed handling, 256-row batch, checkpoint word positions, and logging from the trusted trainer.

This is a less disruptive memory repair than switching the triangle to the legal40k accum training completion microbatch-accumulation trainer because it preserves the full 256-row forward/update. The legal40k accum training completion/chunk stream preflight evidence remains relevant fallback evidence: masked-token weighted microbatching is algebraically valid for deterministic paths, but it introduces microbatch forward/dropout differences. Activation checkpointing avoids that extra contrast change.

## Preflight evidence

### Batch-256 first update pilot

Command: checkpointed wrapper on the already trained `compact_view_reinvest` stream, same seed triplet, batch 256, seq 256, max exposure 39,370 words (the first full batch), no scheduled million-word checkpoint horizon.

Output run: `experiments/archive/representation_and_objectives/training/runs/gc_pilot_compact_view_firstbatch`

Result:

- loss: `9.809807777404785`
- batch words: `39370`
- masked tokens: `8658`
- effective mask rate: `0.1526`
- word exposure: `39370`
- steps: `1`

The reference trained compact-view run at `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/training_log.jsonl` has the same first update values: loss `9.809807777404785`, batch words `39370`, masked tokens `8658`, effective mask rate `0.1526`. This verifies that checkpointing does not alter the first batch data/masking/loss under the real batch256 shape and that the OOM is bypassed at least for the first update.

### Small-batch original-vs-checkpointed equivalence

The first attempted small-batch pilot with `max_word_exposure=9871` failed because `example_jsonl` selection cannot stop mid-row; row 66 would have crossed the target. I reran at the row-boundary total `9673` words (64 rows), giving two optimizer steps at batch 32.

Original trainer run: `experiments/archive/representation_and_objectives/training/runs/gc_equiv_original_b32`

Checkpointed wrapper run: `experiments/archive/representation_and_objectives/training/runs/gc_equiv_checkpointed_b32`

Both produced exactly the same two logged updates:

| step | loss | batch_words | masked_tokens | effective_mask_rate |
|---:|---:|---:|---:|---:|
| 1 | 9.807024955749512 | 4753 | 1068 | 0.1563 |
| 2 | 9.801679611206055 | 4920 | 1022 | 0.1520 |

This gives a direct local test that checkpointing is not changing the measured trajectory for the first two optimizer steps under a fitting batch size.

## Relaunch rule

The two missing triangle arms should be relaunched in fresh run directories using only the activation-checkpoint wrapper:

- `gc_compact_repeat_reinvest_16k_seed43022`
- `gc_adjbreak_reinvest_16k_seed43022`

Interpretation remains governed by `plans/compact_view_triangle_protocol.md`: do not read a one-seed view-control gap inside the documented seed-spread band as a mechanism verdict, and if the adjacency-broken arm collapses broadly, check whether the control became generically incoherent before attributing the drop to lost source-own-rewrite consolidation.
