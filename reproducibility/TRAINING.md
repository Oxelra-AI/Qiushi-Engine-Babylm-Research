# Training, Data, and Mechanism Programs

The [core programs](../experiments/CORE_PROGRAMS.md) are the original research
implementations with public paths and identifiers. The
[recorded configurations](../experiments/configs/executed_stage3.json) identify
executed conditions. The broader [experiment index](../experiments/entrypoints.json)
also contains historical studies and controls, not just the released recipe.

## Research Platform and Software

The research used two NVIDIA H100 GPUs for model training, evaluation, teacher
text generation and accelerated mechanism experiments. CPU work included text
processing, tokenization, data statistics and result analysis. The platform
configuration is summarized below; GPU hardware is documented in the research records,
while CPU, memory and operating-system values were checked on 9 September 2026.

| Component | Configuration |
| --- | --- |
| GPUs | Two NVIDIA H100 GPUs |
| CPU | Two Intel Xeon Platinum 8473C processors; 104 physical cores and 208 logical threads |
| Memory | Approximately 503 GiB visible to the operating system |
| Operating system | Ubuntu 22.04.5 LTS, x86_64 |

The [paired-run record](../research/notes/compact_experience/execution_state_and_next.md)
documents use of both GPUs. The
[compact-text generation record](../research/notes/representation_and_objectives/compact_provenance_and_overlap_addendum.md)
identifies Qwen/Qwen3.5-9B, H100 execution and the decoding settings. Preserved
research programs cover generation, training, evaluation, representation and
gradient measurements, and interventions on model computations.

Both model packages record a CPU verification environment in
[Frontier ENVIRONMENT.json](../models/frontier/ENVIRONMENT.json) and
[Principle-guided ENVIRONMENT.json](../models/principle_guided/ENVIRONMENT.json):
Python 3.12, PyTorch 2.11.0 (CUDA 12.8 build), Transformers 4.57.6, Tokenizers
0.22.2 and Safetensors 0.8.0. These files describe release verification, not a
single environment for every historical training and generation run. Use an
isolated environment with the dependencies recorded for the selected program;
the archived official evaluator and recursive-model study retain their own
requirements.

Run programs from the repository root. Historical run configurations describe
the experiment, while each program's argument parser defines its CLI; a JSON
record is not automatically an accepted `--config` input. Select a new output
directory when rerunning work so that the archived results remain unchanged.

## First-Generation Model

The primary backbone uses eight width-480 DeBERTa layers and primary residual
adapters. Joint training is followed by frozen-parent replay through a dedicated
995,584-parameter increment. The released model has 36,458,592 parameters and
uses increment scale 0.75. The exact training description is in
[TRAINING.md](../models/frontier/TRAINING.md).

The original backbone trainer is
[adapter_scaled_trainer.py](../experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py).
Frozen-parent replay is implemented in
[frozen82_fastpath_replay_trainer.py](../experiments/archive/frontier_consolidation/scripts/frozen82_fastpath_replay_trainer.py).
Its model dependencies and historical configurations are retained in the same
archive, with source/public SHA-256 identities in the file manifest.

## Second-Generation Model and Controls

The final policy uses the released first-generation model, the existing paired
text stream, 80 word-paced updates and a frozen base. It separates dense masking
of second-view content from sparse focused prediction targets, then adds
ordinary-input preservation. The source remains visible in both conditions.

The complete policy is implemented in
[clean_preservation_train.py](../experiments/archive/functional_learning/scripts/clean_preservation_train.py).
After constructing the exact tail described in the [data guide](../data/RECONSTRUCTION.md),
the corresponding explicit command is:

```bash
python3 experiments/archive/functional_learning/scripts/clean_preservation_train.py \
  --tail-jsonl data/local/reference_tail_unchanged_qwen_segments.jsonl \
  --out-dir build/reruns/principle-guided-62064 \
  --train-seed 62064 --gpu 0 --max-updates 80 --words-per-update 39533 \
  --schedule-total 455 --schedule-offset 101 --warmup 10 --lr 0.00005 \
  --weight-decay 0.01 --max-grad-norm 1.0 --private-scale 0.75 \
  --seq-length 512 --micro-batch 8 --mask-prob 0.15 \
  --focus-prob 0.35 --max-focus-groups-per-row 16 --focus-lambda 0.15 \
  --lambda-pres 1 --kl-temperature 1 --pres-student-mode eval
```

Seed 62065 is a second continuation of the same pretrained parent. It is not an
independently pretrained base. The corrected acquisition-only replay uses this
trainer with `--lambda-pres 0`; the preserved configurations distinguish it from
earlier stochastic variants. The ordinary control uses
[real_stream_train_weighted.py](../experiments/archive/functional_learning/scripts/real_stream_train_weighted.py)
with `--objectives inherited_wwm`. Dense-target controls use the same weighted
trainer with `--objectives correspondence_focus_weighted --focus-prob 1
--max-focus-groups-per-row 128`. Other arguments must follow the recorded run.

Acquisition consumes 3,162,742 words. Preservation adds 517,332 student-input
presentations. Do not attribute the entire last increment to KL alone or call
the full policy compute-matched to acquisition-only training.

Random sampling depends on both the integer seed and the byte sequence used
to distinguish random streams. Configuration keys ending in `_utf8` record
those bytes. Changing them changes the sampled masks or examples even when
the integer seed remains fixed.

## Mechanism Studies

The [research-material index](../research/materials.md) connects each topic to
its code, notes, results and correction history. Main families are source/target
relations and window availability; synthetic binding and held-symbol transfer;
activation replacement, centering, zeroing and rotation; and acquisition versus
preservation conditions. Keep the original target populations, loss denominators,
seed levels and failed controls when reusing the programs.

## Evaluation

The archived [official evaluator](../experiments/archive/initial_model_studies/repos/babylm-eval/strict)
is based on revision `6f825c291e2c4c78ad33b1935fd64d45f52642dc`.
Its license and original requirements accompany it. Local classifier modifications
are retained rather than replaced with a stock encoder loader.

The [evaluation provenance](../evidence/evaluation_provenance.md) describes the
complete-model loading condition. Use the [checkpoint references](checkpoint_references.md)
for AoA trajectories. Fast screens do not replace the nine-component evaluation.
The [final result table](../results/training_strategy_comparison.csv) contains
the authoritative existing comparisons; this packaging task did not rerun them.

## Teacher Generation

`BABYLM_GENERATOR` identifies an externally supplied generator compatible with
the recorded command arguments and JSONL schemas. This repository preserves
the original prompts, model identities, decoding settings and output parsers,
but does not supply that executable. Keep the recorded row order and record
the model revision, tokenizer, chat template and inference environment.
Generation configurations marked as proposed are not completed experiments.
