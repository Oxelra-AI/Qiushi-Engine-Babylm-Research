# clean qwen control eval summary control wave: rule evidence and mechanism state

## Training wave launched

The recorded design is the clean-Qwen control and replication training comparison.

Script: `scripts/launch_control_replication_wave.sh`.

Preflight summary: `data/control_preflight_summary.json`.

Preflight validated:

- Clean-Qwen selected-pair fraction: `0.16568`, `37,594` complete original--Qwen-rewrite pairs.
- Shuffled control: same pair-word fraction `0.16568`; all `37,594` pairs changed within source/length bins; same selected originals and same Qwen rewrite multiset.
- Source-matched official control: official-only, row-length sequence matched to Qwen treatment, effective source word counts exactly matched after assigning Qwen pair words back to source. It requires official stream reuse: bnc_spoken `6,617`, gutenberg `105,829`, simple_wiki `47,539` words.
- Original-duplication control: official-only original+original rows, pair-word fraction `0.16568`, `36,687` selected duplicate pairs.

Training schedule in the launcher:

1. Wave 1: `qwen_shuffled_control_16k_seed43022` on GPU0 + `official_lengthmatched_16k_seed43122` on GPU1.
2. Wave 2: `qwen_clean_aligned_16k_seed43122` on GPU0 + `official_sourcematched_16k_seed43022` on GPU1.
3. Wave 3: `official_original_dup_16k_seed43022` on GPU0.

All use DeBERTa-v2 `8x480`, inherited baseline16k tokenizer, WWM `0.15`, batch256, seq256, LR `1e-3`, exact 100M whitespace-word exposure, and 1M-spaced checkpoints through 100M.

## First positive result being tested

Canonical result: `data/full_eval/full_eval_summary.json` and `notes/clean_qwen_full_eval_and_control_wave.md`.

Seed 43022 clean-Qwen treatment vs row-length-matched official control:

- Overall `41.34429066479573` vs `40.66789913207705`, Δ `+0.676392`.
- Gains: Supplement `+3.80`, Entity `+2.45`, SuperGLUE `+2.340479`, AoA `+0.157045`.
- Losses: BLiMP `-0.45`, EWoK `-0.61`, COMPS `-1.35`, GlobalPIQA `-0.06`, Reading `-0.19`.
- This passes the internal continuation threshold but is below visible 2026 Strict-Small leader `41.8` and is not a mechanism result by itself.

## Current official-rule evidence

Read sources:

- `Knowledge/objects/papers/BabyLM-Turns-4-and-Goes-Multilingual-Call-for-Papers-for-the-2026-BabyLM--c029b1e13fb4--f8f389e8f710/object.md`, lines 134--156 and 209--223.
- `Knowledge/objects/papers/BabyLM-Turns-3-Call-for-papers-for-the-2025-BabyLM-workshop--11bc426639c1--f8eb47c999e1/object.md`, lines 236--245.
- Official website scrape: `data/external/content.md`, lines 106--113 and 176--182.

Key 2026 text:

- Strict/Strict-small need not use official corpus, but must stay within corpus size: Strict `100M`; Strict-small `10M`.
- The 2026 call merges interaction/multimodal into Strict/Strict-Small and says external models may generate synthetic data or feedback for the submission model.
- External models must come from a predetermined list on the BabyLM website.
- For Strict-small, the `100M word/reward` limits in the Strict external-model paragraph become `10M word/reward` limits.
- Distillation remains prohibited: external model tokenizer, weights, hidden states, or output distribution cannot be revealed to the submission model; if using output distributions, the external model's training word count would count toward the submission limit.
- Training duration: at most `100M` whitespace-separated input words for Strict-small; intermediate checkpoints every `1M` until `10M` and every `10M` until `100M`.
- Submissions need HF links to model and intermediate checkpoints, predictions, a custom-data datasheet and download link if not using BabyLM-provided corpus, and when using an external model, any generated data and external-model fine-tuning/distillation data if any.

Implication for our Qwen route:

- Quantity/accounting: current clean corpora replace official words inside a 10M pool and train for 100M whitespace-word exposure, so the internal word accounting matches the competition exposure limit.
- Non-distillation: the submitted model uses only generated text, not Qwen tokenizer, weights, hidden states, or output distributions, so it is not distillation in the prohibited sense.
- Remaining rule risk: Qwen3.5-9B must be on the predetermined BabyLM external-model list (not yet verified). If it is absent, the route may be valuable internal science but not directly submission-compliant without regenerating with an allowed external model.

## Mechanistic interpretation


Main scientific points:

1. `qwen_clean_aligned - qwen_shuffled_control` is the cleanest current test of original--rewrite correspondence because it fixes selected originals, rewrite multiset, pair-word budget, and global Qwen register. If this contrast vanishes, the first positive result is likely generated text/style/source/selection rather than aligned correspondence. If it remains positive, it still reflects meaningful same-window related adjacency, not abstract semantic alignment alone.
2. The source-matched official control is necessary but imperfect: it matches effective source word totals and row lengths using official-only words, but achieves this by flattening source streams and re-chunking, which may introduce unnatural boundaries and official-word reuse.
3. Original-duplication tests whether exact repetition/redundancy is enough, but is not exact: it uses `36,687` selected pairs rather than all `37,594`, changes lengths and selected-ID composition, and should be interpreted as a redundancy control, not a fully matched mechanism isolate.
4. Second seed must be interpreted as paired deltas (`qwen_seed43122 - official_seed43122`), not just a treatment score. Two seeds are a sign/shape check, not a full variance estimate.
5. Tokenizer/context truncation remains a real confound. The audit showed Qwen treatment has `17,549` rows over seq256 and `451,433` excess tokens per 10M, versus official control `17,239` rows and `433,082` excess tokens. Because pair rows are ordered original then rewrite, right truncation could systematically reduce rewrite exposure.
6. High-value missing controls after this wave: selected-original exposure control, separated-pair/non-adjacent control, and pair-order balance or order-randomized control. Also needed: training-time token-exposure audit by pair side and held-out representation/conditional-prediction alignment probes.

## Post-training evaluation route

Required checks after training:

1. Verify every new run has `scientific_metrics.json`, `word_exposure=100000000`, `chck_100M`, and ideally all `chck_1M..chck_100M`.
2. Run full official-style local evaluation with `scripts/launch_control_eval.sh`.
3. Summarize with `scripts/summarize_control_eval.py`, producing `data/control_eval_summary.json` and `notes/clean_qwen_control_eval_summary.md`.
4. Interpret contrasts in this order:
   - `qwen_clean_aligned_seed43122 - official_lengthmatched_seed43122`: seed replication.
   - `qwen_clean_aligned - qwen_shuffled_control`: contribution of preserving original--rewrite correspondence under identical selected originals and rewrite multiset.
   - `qwen_clean_aligned - official_sourcematched`: improvement beyond official-only source/domain mixture.
   - `qwen_clean_aligned - official_original_dup`: generated second view beyond exact original repetition.

Do not turn any single positive control result into a final learning principle. If second seed or shuffled control removes the advantage, redirect toward the stronger mechanism exposed by the data. If controls support pair correspondence, still add selected-original exposure, separated-pair/non-adjacent, and token/truncation audits before mechanism claims or SOTA-facing work.
