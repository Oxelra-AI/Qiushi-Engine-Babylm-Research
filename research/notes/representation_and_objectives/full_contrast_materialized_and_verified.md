# full contrast materialized and verified — capped semantic-view contrast materialized, audited, and made launch-ready

## Why the packet local semantic view control design changed

packet local semantic view control correctly repaired the main causal control from `original_stream_matched`
to `original_packet_local`, but two final implementation facts still needed
execution evidence before using H100 time:

1. The late independent_review-driven edits (sentence-boundary packet-local repetition and deep
   actual-string audit) had not been rerun.
2. frontier_consolidation reported a mechanism-preservation failure in a different curriculum
   route: row chunking can hide paired views under seq limits. This made it
   necessary to audit whether appended generated views are actually visible to
   the protected seq256 MLM trainer.

full contrast materialized and verified resolved both issues and produced the preferred training corpus.

## Preferred corpus for training

Use this directory, not the uncapped first full materialization:

`experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1`

Scripts:

- Materializer: `experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py`
- Auditor: `experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py`
- Seq256 visibility audit: `experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_visibility.py`
- Launcher now points to capped corpus: `experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_packet_local_training.sh`

## Materialization command

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py \
  --out-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1 \
  --note research/notes/representation_and_objectives/semantic_view_contrast_materialization_capped1.md \
  --total-words 10000000 --passes 10 --max-semantic-words 2500000 \
  --max-per-type-per-source 1 --write-training
```

Result:

- Available accepted views before source/type cap: 16,394.
- Used accepted views after cap: **16,368** (simplification 7,065, paraphrase 9,303).
- Dropped duplicate exact-source views: 26 (12 simplification, 14 paraphrase).
- Semantic packet rows: **10,840**.
- Views per packet after cap: **5,312 rows with one view, 5,528 rows with two views; max=2**.
- Semantic-view words: **820,187** = **8.2019%** of the 10M pool.
- Identical official filler: **9,179,813** words.
- Pool rows per arm: 68,214.

Scientific reading: the cap removes the few many-view duplicate-source outliers
(e.g. a 14-view row) and restores the intended packet structure: original source
+ at most one simplification + at most one paraphrase. The corpus remains a
semantic-view mechanism over ~0.82M SimpleWiki words, not a broad factual-source
composition intervention.

## Full contrast audit

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py \
  --corpus-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1 \
  --total-words 10000000 --passes 10
```

Audit result:

- Status: `SEMANTIC_VIEW_CONTRAST_AUDITED`.
- 10M exact word accounting per pool.
- Row-length sequence identical across treatment, stream control, packet-local control, and packet-exact artifact.
- Official filler identical after semantic prefix for both stream and packet-local controls.
- Packet-local identity true: source key/article/row fields match treatment packet rows.
- Deep prefix construction true: 10,840/10,840 rows reconstructed with 0 mismatches; treatment text = source + used accepted views; packet-local control uses only source-vocab words and starts with the exact source.
- Token delta packet-local minus treatment: mean -0.2922 tokens, over-256 row delta -8, excess-token delta +9.
- All four 100M training files exist and have exact 100,000,000-word exposure.

Audit JSON:
`experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/semantic_view_contrast_audit.json`

## Seq256 visibility audit

Trainer code in `masking_curriculum_trainer.py` tokenizes each row with
`add_special_tokens=False, truncation=True, max_length=256`. I wrote and ran a
visibility audit that reconstructs source/view token spans under the same
baseline16k tokenizer:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_visibility.py \
  --corpus-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1 --max-len 256
```

Result:

- Accepted generated views audited: **16,368**.
- Fully visible views: **16,218**.
- Partially visible views: **150**.
- Not visible views: **0**.
- Overall generated-view token visibility: **0.996583**.
- Rows with any hidden/partial view: 150.
- Rows with zero visible view tokens: 0.
- By type: simplification fully visible 7,065/7,065, token visibility 1.0000; paraphrase fully visible 9,153/9,303, partial 150, none 0, token visibility 0.9943.

Visibility JSON:
`experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/semantic_view_seq256_visibility_audit.json`

This directly answers the concern raised by frontier_consolidation's row-chunk finding: in this
packet corpus, generated views are actually available to the protected seq256
trainer. Any treatment/control score movement will not be a hidden-view artifact.

## Launch readiness

`launch_semantic_view_packet_local_training.sh` now points to the capped
corpus and its internal pre-flight gate checks:

- identical filler after prefix
- packet-local identity
- deep prefix construction `ok`
- exact 100M exposure for treatment
- exact 100M exposure for packet-local control

All pass on the capped corpus. Syntax/pre-flight checks passed:

```bash
bash -n experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_packet_local_training.sh
bash -n experiments/archive/representation_and_objectives/training/scripts/eval_semantic_view_contrast_noaoa.sh
PYTHONDONTWRITEBYTECODE=1 python -B -c "import ast,pathlib; [ast.parse(pathlib.Path(p).read_text()) for p in ['experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py','experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py','experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_visibility.py']]; print('syntax_ok_all_step006')"
```

## GPU scheduling state

The fixed-data WWM-to-token masking comparison remained in progress. Read-only artifact inspection showed:

- run dir: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022`
- checkpoints present through `chck_60M`
- latest visible training log line: earlier analysis, loss 2.6569, lr 0.000316, seq_len 256, mask_mode `wwm`, effective_mask_rate 0.1539
- token masking should begin near 70M by `switch_frac=0.7`; no result should be inferred until the runtime delivers completion and official-compatible evaluation is run.

The capped semantic-view treatment versus packet-local control had not been launched. The proposed comparison remained conditional on the fixed-data WWM-to-token result and completion of the corpus checks.

## Split-arm scheduling update

After the capped corpus passed its audit and visibility checks, semantic-view treatment training was launched:

- task ref: `s6_t43_tool1`
- run dir: `experiments/archive/representation_and_objectives/training/runs/semantic_view_treatment_8x480_16k_wwm_seed43022`
- data: `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/semantic_view_treatment_100M.jsonl`
- recipe: protected 8x480/baseline16k/WWM fixed/AdamW, seed 43, extra_init_seed 43022, train_rng_seed 43023, 1M checkpoints, 100M exposure.

Because the treatment arm is now separate, I wrote a packet-local-control-only launcher:

- `experiments/archive/representation_and_objectives/training/scripts/launch_packet_local_control_only.sh`

Use it when a GPU is free, e.g. `GPU=0 bash experiments/archive/representation_and_objectives/training/scripts/launch_packet_local_control_only.sh`, or submit it as a managed task with write target `experiments/archive/representation_and_objectives/training/runs/original_packet_local_8x480_16k_wwm_seed43022`. It trains the exact matched control with the same seeds/recipe/checkpoint spacing and gates on the capped corpus audit. Do not use the original two-arm launcher while the treatment run directory exists; it is designed to refuse overwriting.
