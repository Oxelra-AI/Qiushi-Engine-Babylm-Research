# Leader Analysis and Route Pivot: Research Plan and Outstanding Work

## What Was Accomplished

1. **Complete Leader Method Analysis**: The SOTA leader (`wwm_curriculum_simplification_40k`, Overall 41.8) was fully reverse-engineered from its HuggingFace model card. Key finding: it replaces the entire BabyLM official corpus with FineWeb-Edu simplification pairs (9,999,969 words). Architecture: 12×384 DeBERTa-v2, LAMB lr=0.007, 40k tokenizer, masking curriculum (WWM→token), seq length curriculum (64→256).

2. **Gap Root Cause Identified**: Our EWoK/Entity/COMPS/GlobalPIQA deficit comes from the official BabyLM corpus being 75% conversational/literary (CHILDES, Gutenberg, OpenSubtitles) vs the leader's factual encyclopedic FineWeb-Edu content.

3. **Data Generation Pipeline**: Qwen3.5-9B is used to generate factual content from SimpleWiki seeds (31k paragraphs extracted, 50k prompts prepared). Three generation types: expansions (20k), simplifications (15k), paraphrases (15k).

4. **Generation Status**: Qwen3.5-9B generation with 50k prompts was in progress when this note was written. If successful, outputs go to `training/runs/gen_expand_v4/generations.jsonl` and `training/runs/gen_simpara_v4/generations.jsonl`.

## Critical Files

| File | Purpose |
|------|---------|
| `notes/leader_analysis_and_route_pivot.md` | Complete leader analysis + gap breakdown |
| `plans/training_plan_and_experimental_schedule.md` | Full experimental plan |
| `training/scripts/prepare_factual_generation_prompts.py` | Prompt preparation |
| `training/scripts/launch_factual_generation.sh` | Generation launcher |
| `training/scripts/assemble_factual_corpus.py` | Corpus assembly (run after generation) |
| `training/data/factual_prompts_shard0_expand.jsonl` | 20k expansion prompts |
| `training/data/factual_prompts_shard1_simpara.jsonl` | 30k simp/para prompts |

## Proposed Experiments and Prerequisites

1. **Generation prerequisite**: Successful generation must be established before corpus assembly; completion was not established in this note.

2. **If generation succeeded**: Run `assemble_factual_corpus.py` to create the training corpus.

3. **Train 40k tokenizer**: Use `sentencepiece` to train a 40k BPE tokenizer on the assembled corpus.

4. **Start Experiment 1** (quick test): Apply leader's curriculum innovations (LAMB lr=0.007, seq 64→256, WWM→token) to EXISTING clean-Qwen data. Uses existing masking_curriculum_trainer.py with modifications for LAMB and seq curriculum.

5. **Start Experiment 2** (main attempt): Train on factual-enriched corpus with full leader recipe + our paraphrase innovation.

## Source Dependencies

- Masking curriculum trainer: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py` (999 lines, supports wwm_to_token mode)
- Clean-Qwen data: `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`
- Official corpus: `experiments/archive/initial_model_studies/data/mixture_revision_61/official_raw/*.train.txt`
- Full eval infrastructure: `experiments/archive/compact_experience/scripts/custom_endpoint_full_eval.py`

## Unresolved Issues

- Generation completion remained unverified in this note.
- The 40k tokenizer needs to be trained (SentencePiece BPE on the new corpus). 
- LAMB optimizer needs to be imported (`pytorch_optimizer` or torch built-in).
- Seq length curriculum needs implementation in the trainer (batch construction logic).
