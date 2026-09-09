# continuation training dynamics — matched legacy-WWM staged control for the PVDM causal split

## Why this branch exists

The current research object remains the persistent context-conditioned world-relation weakness, visible in EWoK conditional-reversal failures and GlobalPIQA_parallel deep-rank errors. pvdm full readout synthesis did not justify leaving that problem; it separated two effects inside the staged 70M→80M PVDM experiment:

1. **Differential pivot effect:** PVDM treatment, which keeps the true relation pivot visible and masks matched surrogate anchors, is worse than the matched control on the relation readouts that motivated the experiment. Treatment-control deltas: Supplement −2.66, Entity +1.05, GlobalPIQA_parallel −1.9417, GlobalPIQA_nonparallel −3.0, hard52 mean top-minus-correct +0.2175 nats, EWoK accuracy −0.003019, EWoK stable-failure fraction among wrong rows +0.009614.
2. **Shared staged-continuation effect:** both PVDM/control staged arms are weaker than the uninterrupted compact 80M reference on EWoK. Control-reference: EWoK accuracy −0.002494, stable-failure fraction +0.005193, wrong-row interaction median −0.2715. Treatment-reference is worse still.

The shared effect currently mixes at least three causes: symmetric optimizer reset, microbatch replay, and target redistribution/anchor swapping. Without a standard WWM staged branch on the same segment, a new objective could be built on the wrong causal interpretation.

## Branch launched

- Task ref: `s121_t3_tool2`
- Command: `CUDA_VISIBLE_DEVICES=0 python3 experiments/archive/representation_and_objectives/scripts/pvdm_continuation_trainer.py --mode standard_legacy --output_dir experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022 --micro_batch_size 8 --checkpoint_name chck_80M --save_trainer_state`
- Run dir: `experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022`
- Init checkpoint: compact `chck_70M`, actual exposure 70,040,037 words
- Segment: exact same tail segment as PVDM/control, tail rows 255..64254, 64,000 rows, 9,971,289 words, target cumulative exposure 80,011,326
- Architecture/tokenizer: same 34,467,424-param DeBERTa-v2 8×480 and shared legal16k tokenizer as pvdm 80m ewok fourcell reader/120 arms
- Execution: same staged trainer and microbatch-8 gradient accumulation used by PVDM/control, but `standard_legacy` uses the inherited WWM masking machinery rather than dependent-target redistribution.

This is the minimum reliable expensive check: one 10M staged branch, not a 100M continuation or full official evaluation.

## Fixed readout prepared

Wrapper: `experiments/archive/representation_and_objectives/scripts/legacy_80m_readout_and_synthesis.py`

It will evaluate the completed `hf_model/chck_80M` using the same fixed surfaces used in pvdm full readout synthesis:

- Supplement and Entity official-compatible sentinel columns
- GlobalPIQA all-option margin reader for parallel and nonparallel rows, including hard52 rank/margin comparison
- EWoK four-cell interaction reader on the 7,618 current rows

Synthesis outputs will go under `experiments/archive/representation_and_objectives/data/legacy_80m_readouts` and `research/notes/representation_and_objectives/legacy_80m_causal_split.md`.

## How to interpret outcomes

Let `standard` denote the staged legacy-WWM branch, `reference` the uninterrupted compact 80M checkpoint, and `control/treatment` the pvdm full readout synthesis PVDM arms.

- If `standard` is close to `reference` on EWoK accuracy, stable-failure fraction, and wrong-row interaction while PVDM/control remain weaker, then the shared EWoK damage is mainly caused by target redistribution/anchor swapping rather than by staged continuation. The next mechanism should add a sparse coupled pivot–consequence signal on top of intact compact MLM, not replace a large fraction of WWM targets.
- If `standard` is also weaker than `reference` on EWoK by a similar amount, then the staged continuation path itself is a causal problem. Before any new relation objective, repair or isolate optimizer reset, scheduler/RNG continuity, and microbatch execution. In that case, a new objective would be uninterpretable unless run on an intact continuation path.
- If `standard` improves relative to `reference`, the optimizer reset/staged replay may be beneficial while PVDM target redistribution is harmful; a future sparse objective should preserve the standard-WWM distribution and use very low-dose auxiliary losses rather than high-density target replacement.
- If `standard` damages broad columns while PVDM/control preserve them differently, inspect Supplement/Entity/GlobalPIQA_nonparallel and training loss before deciding whether the continuation machinery or the WWM distribution is the source of the tradeoff.

In all cases, dense true-pivot-visible PVDM remains closed as a 100M continuation route. The relation problem remains open; the next mechanism should be designed from the causal split, not by switching to an unrelated branch.

## Open Challenge

The open question is whether this decomposition is sufficient and which result would require continuation repair before a new relational objective.
