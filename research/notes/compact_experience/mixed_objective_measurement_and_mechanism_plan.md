# mixed objective measurement and mechanism plan mixed-objective measurement and mechanism-separation plan

## Immediate measurement

The causal attention verification mixed-objective experiment changes the learning signal throughout training on the clean-Qwen corpus while keeping the DeBERTa-v2 8×480 architecture, 16k tokenizer, initialization, data order, word exposure, optimizer, and LR schedule matched to the clean-Qwen pure-MLM reference.

The previous no-AoA screen threshold is not sufficient for this objective. The intervention can alter SuperGLUE or aggregate acquisition behavior independently of BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. Therefore both frozen true-100M mixed-objective arms must be measured completely:

- `mixed_causal50_qwen_seed43022`, causal_fraction = 0.50
- `mixed_causal15_qwen_seed43022`, causal_fraction = 0.15

The full measurement must run the nine official-style columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading, SuperGLUE, and AoA. AoA is only a terminal aggregate readout from the completed checkpoint ladder; it is not used to choose endpoint, causal fraction, curriculum, data, or any training signal.

Scripts prepared:

- `scripts/launch_mixed_objective_full_eval_100M.sh`
- `scripts/score_mixed_objective_full_eval.py`
- Output root: `data/mixed_objective_full_eval/`
- Summary note after completion: `notes/51_mixed_objective_full_eval_summary.md`

## How to interpret a positive or negative result

Any score difference first belongs to the **full mixed-objective package**, not to a purified mechanism. The package includes:

1. causal visibility: lower-triangular attention on causal batches;
2. uncorrupted inputs on causal batches;
3. dense next-token targets on causal batches;
4. fewer MLM updates and fewer masked-word reconstruction targets;
5. different gradient variance and shared-head pressure;
6. a changed masking RNG trajectory because `mask_gen` is consumed only on MLM batches.

Thus a full-eval improvement would be real recipe evidence, but not yet proof that causal objective complementarity rather than target density caused the gain. A full-eval failure would still inform the route: the current dense causal package may damage bidirectional competence, acquisition behavior, or fine-tuning transfer, but other objective mixtures (masked-next-token span, prefix-LM, RTD) may remain scientifically distinct.

## Next separator if mixed objective has any useful signal

Before assigning a deep mechanism, separate **target density** from **objective complementarity** on the same architecture/corpus/init. The most direct separator is a matched-from-scratch `dense-MLM` arm:

- Keep bidirectional attention and `[MASK]` corruption.
- Increase mask probability or repeat masked positions so the expected number of supervised positions per word exposure matches the mixed arm's realized total target count.
- Preserve word exposure, optimizer steps, LR schedule, batch geometry, tokenizer, data order, and initialization.
- Compare true-100M full scores against clean-Qwen and the mixed arm.

Interpretation:

- If dense-MLM reproduces the gain, the active ingredient is mostly target density / supervision count.
- If mixed objective exceeds dense-MLM at matched target count, left-to-right predictive complementarity or uncorrupted causal context is implicated.
- If dense-MLM harms while mixed helps, the causal objective may regularize dense targets by removing mask-token distribution shift.
- If both fail, this route should not be enlarged into bundled GPT-BERT architecture changes without a new source-grounded reason.

Alternative separators if needed:

- `prefix-LM` batches with bidirectional prefix and causal suffix to separate autoregressive signal from fully causal visibility;
- `MNTP-span` objective that predicts future tokens from masked/prefix context without full causal dense targets;
- ELECTRA-style replaced-token detection only after legal-data and teacher constraints are rechecked.

## Current reference values

- Clean-Qwen seed43022 true 100M: Overall `41.34429066479573`, equal7 `43.112857142857145`, AoA raw `0.0`.
- Visible leader: Overall `41.8`.
- Tail/restart/mask-allocation family is closed as a SOTA route because all six complete endpoints scored below `40.0832` with strongly negative AoA.
