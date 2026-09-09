# BabyLM 2026 Official Evaluation Surface and First Experiment Plan (2026 eval and experiment plan)

## Evaluation source and hardware

- Cloned `babylm-org/babylm-eval` to `experiments/archive/initial_model_studies/repos/babylm-eval` at commit `6f825c2` (`Fix fast eval collation for EWoK`).
- Recorded hardware visibility: 2 GPUs.
- 2026 leaderboard Space is `https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026`; backing public config was parsed from its Gradio `/config` endpoint.

## Live leaderboard extraction

- Leaderboard components: [{'id': 5, 'n_rows': 54, 'n_headers': 52}, {'id': 7, 'n_rows': 122, 'n_headers': 52}, {'id': 9, 'n_rows': 49, 'n_headers': 97}].
- Track counts: {'multilingual': 49, 'strict': 54, 'strict-small': 122}.
- Current live Strict-Small top row: wwm_curriculum_simplification_40k — Overall 41.8, NLP 52.97, Human-like 2.71.
- Parsed JSON saved to `experiments/archive/initial_model_studies/data/leaderboard_probe/leaderboard_config_parsed.json`.

### Top Strict-Small rows from live config

| Rank | Model | Overall | NLP | Human-like | BLiMP | EWoK | Entity | COMPS | GlobalPIQA | (Super)GLUE | Reading | AoA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wwm_curriculum_simplification_40k | 41.8 | 52.97 | 2.71 | 67.2 | 56.07 | 28.45 | 53.57 | 39.67 | 69.79 | 5.42 | 0.0 |
| 2 | RecGPT-10M | 41.53 | 52.4 | 3.46 | 73.11 | 52.62 | 16.59 | 55.43 | 40.68 | 66.64 | 6.92 | 0.0 |
| 3 | Wordpiece-24-4 | 41.31 | 52.85 | 0.92 | 70.2 | 52.1 | 21.82 | 54.04 | 36.08 | 69.18 | 1.84 | 0.0 |
| 4 | Bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 | 40.94 | 51.81 | 2.92 | 67.85 | 53.0 | 21.47 | 52.24 | 37.59 | 65.29 | 5.84 | 0.0 |
| 5 | bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 | 40.93 | 51.79 | 2.92 | 67.85 | 53.0 | 19.9 | 52.24 | 38.65 | 65.66 | 5.84 | 0.0 |
| 6 | BabySteps_MurphysLaw-10M-mixed | 40.86 | 53.59 | -3.72 | 71.6 | 51.94 | 27.95 | 53.13 | 36.15 | 70.44 | 7.67 | -15.1 |
| 7 | instanton-hybrid | 40.78 | 51.53 | 3.18 | 72.13 | 50.15 | 19.87 | 52.96 | 37.14 | 67.58 | 6.35 | 0.0 |
| 8 | bb26_claim_agg_synonly_s0 | 40.67 | 51.46 | 2.91 | 67.13 | 52.28 | 19.17 | 52.22 | 40.11 | 66.99 | 5.82 | 0.0 |
| 9 | deberta-base-75k-sam_ext-s1 | 40.62 | 48.74 | 12.19 | 67.95 | 51.25 | 19.66 | 51.77 | 32.64 | 64.95 | 1.49 | 22.9 |
| 10 | bb26_claim_compagg_clean_union_s1 | 40.59 | 51.3 | 3.09 | 68.35 | 51.98 | 21.26 | 52.37 | 35.12 | 65.98 | 6.18 | 0.0 |

## 2026 Strict/Strict-Small command surface

From `strict/README.md`: final models require full zero-shot, full fine-tuning, AoA, and fast zero-shot evaluation over intermediate checkpoints.

```bash
cd experiments/archive/initial_model_studies/repos/babylm-eval/strict
bash scripts/eval_zero_shot.sh <model> <causal|mntp|mlm|enc_dec_mask|enc_dec_prefix> [evaluation_data/full_eval]
bash scripts/eval_zero_shot_global_piqa.sh <model> <backend> <strict|strict-small> [full_eval] [fast_eval]
bash scripts/eval_aoa.sh <model> <backend> <strict|strict-small> [cdi_childes.json] [results]
bash scripts/eval_finetuning.sh <model> [lr=3e-5] [batch=32] [epochs=10] [seed=42]
bash scripts/eval_zero_shot_fast_all_revisions.sh <model> <backend> <track> [evaluation_data/fast_eval]
bash scripts/collate_preds.sh NAME_OF_YOUR_MODEL BACKEND SUBMISSION_TRACK  # includes --fast in current script
```

Required checkpoint naming convention remains `chck_1M` ... `chck_10M`, then `chck_20M` ... `chck_100M` for Strict-Small if full 100M exposure is used.

## Scientific implication for the first decisive experiment

The live 2026 baseline table shifts the target surface relative to the 2025 Findings: official 2026 Strict-Small baselines are GPT-2-style 98.4M causal models with 16,384-token BPE, not only GPT-BERT. The leaderboard still reports `Overall Average`, `NLP Average`, and `Human-like Average`, so the MoEP-vs-AMLM tradeoff remains the right scientific object, but the executable route must evaluate with the 2026 scripts and include GlobalPIQA plus the 2026 hidden/server-side treatment.

The next experiment should not be a vague fusion. It should isolate three mechanisms under identical corpus/tokenizer/checkpoint schedule:

1. **Causal sparse-routing backbone**: MoEP-style routing with a 2026-compatible HF causal model interface, to test whether path diversity raises human-likeness and early AoA/reading without sacrificing the new 2026 NLP columns.
2. **Difficulty-aware auxiliary prediction**: add AMLM-style token-difficulty weighting as an auxiliary masked/span prediction head or scheduled denoising objective while retaining causal scoring, to test whether hard-token learning recovers BLiMP/GLUE/COMPS/GlobalPIQA without destroying AoA curves.
3. **Multi-granularity morphology signal**: a small character/morpheme side encoder tied to the token embedding, constrained to learn only from the same 10M words, to attack morphology/entity/COMPS while avoiding AMLM n-hot's syntax loss.

A pilot should first validate pipeline feasibility on a tiny 1M-exposure run and a fast-eval subset, then scale the best two variants to the full 100M-exposure schedule with the required checkpoints.
