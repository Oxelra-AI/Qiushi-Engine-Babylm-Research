# changed block overlap ancestry current official — GlobalPIQA official generation lineage

## Purpose
official collator coordinate seed43022 reproduced the seed43022 compact_view_reinvest endpoint through one unmodified BabyLM strict collator, but GlobalPIQA scoring used inherited INITIAL_MODEL_STUDIES generated `eng_latn.jsonl` files because the `BabyLM-2026-Strict-Evals` snapshot does not ship them. This note records the direct reproduction of those files through the current official procedure.

## Procedure
Script: `experiments/archive/representation_and_objectives/scripts/globalpiqa_official_lineage.py`

Output summary: `experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/globalpiqa_official_lineage.json`

The script ran the unmodified current official generator:
`experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/global_piqa/dl.py`

It first failed because Hugging Face hub writes still targeted the shared read-only cache. The rerun set `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `XDG_CACHE_HOME` into an isolated writable output directory. The second run returned 0.

## Official source identities
- Official generator sha256: `be1a0bb84fc4efee4278d140279523cd1da0962c9598298cc3cef98f1f24da8c`
- The same `dl.py` file in the inherited INITIAL_MODEL_STUDIES checkout is byte-identical to the official coordinate and seed43122 pristine checkout.
- Upstream `mrlbenchmarks/global-piqa-parallel` dataset revision: `b0b18516a8bc2cb1106bce3dd4db32848ca715ea`, last modified 2026-06-02 16:48:48 UTC.
- Upstream `mrlbenchmarks/global-piqa-nonparallel` dataset revision: `6777742fa3634c0583cda3b7f8a482ea7b1b0937`, last modified 2026-06-02 16:48:02 UTC.

## Reproduced file identities
The current official generator writes identical data under both `full_eval` and `fast_eval`.

- GlobalPIQA parallel: 103 rows, sha256 `cb9513ed5096ad8115becbe42fdfc97711980f519ada87808e3ddc5f8d5e4453`
- GlobalPIQA nonparallel: 100 rows, sha256 `aeec831d3adb09bf268ce1b847a854d105b3922c440a164a2c5893b96b228a8f`

Both files are byte-identical to the inherited INITIAL_MODEL_STUDIES files used in official collator coordinate seed43022 scoring. Therefore the seed43022 official-coordinate GlobalPIQA column is tied to the current official generation procedure and current upstream dataset revisions, not an unstated stale local evaluation asset.

## Scientific effect on the endpoint
No numerical score changes. This strengthens the official collator coordinate seed43022 official-coordinate result:
`compact_view_reinvest` seed43022 remains Overall 42.0331347900748 through the unmodified collator coordinate, with GlobalPIQA 35.62135922330097 computed on files now reproduced from the current official GlobalPIQA source lineage.
