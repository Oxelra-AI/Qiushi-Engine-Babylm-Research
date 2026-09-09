# clean qwen experiment state — Clean-Qwen matched experiment: materialization verified, training launched, controls prepared

## What is now established (real evidence, this step)
- Both extra Qwen shards complete: `training/runs/extra_qwen2_shard0/outputs.jsonl` and `.../shard1/outputs.jsonl`, 32,804 records each. With the original initial rewrite set, 110,608 generation records total.
- Fixed a syntax error (bad indentation, lines 870–878) in `scripts/clean_materialize_qwen_pairs.py` left from clean qwen compliance and validity line-edits; all clean qwen compliance and validity scripts now compile.
- Ran clean materialization. Result (`data/qwen_clean_aligned/clean_materialization_metadata.json`):
  - clean_pairs_available = 37,704; selected_pairs = 37,594; selected_pair_words = 1,656,800 = **16.568%** of the 10M pool (below 25% cap, above 10% floor → meaningful matched test).
  - Rejections dominated by copy_overlap (44,347) and entity_recall (20,379): the filter removes near-copies and entity-dropping rewrites, as intended.
  - Exact word totals: official_only_10M/100M and qwen_aligned_10M/100M all exactly 10M/100M. Treatment/control row-length sequence matched. No pair truncation; no word-fragment padding in Qwen pair rows.
  - Selected pair sources: gutenberg 704,709 w; simple_wiki 463,379; open_subtitles 249,118; bnc_spoken 162,137; childes 75,334; switchboard 2,123.
- Ran preflight audit (`data/qwen_clean_aligned/pretrain_audit.json`): hashes match metadata; baseline16k tokenizer; token-length: official mean 233.38, qwen mean 231.96 (Δ −1.43 tokens/row); rows over seq256 official 17,239 vs qwen 17,549 (+310); excess tokens above 256 official 433,082 vs qwen 451,433 (+18,351).
- Built selected-pair stratified spotcheck: `notes/selected_pairs_stratified_spotcheck.md` (220 items). First-page inspection: mostly meaning-preserving; residual child-language/dialogue speaker-label artifacts and a few register shifts.

## Training configuration; completion pending in this record
- GPU0: `official_lengthmatched_16k_seed43022` (official-only, row-length-matched control).
- GPU1: `qwen_clean_aligned_16k_seed43022` (treatment: 16.568% complete Qwen pairs + official filler).
- Both: DeBERTa-v2 8×480, baseline16k inherited tokenizer, WWM 0.15, batch256, seq256, extra_init_seed=43022, train_rng_seed=43023, 100M exposure, checkpoint every 1M words (AoA-capable ladder).

## Scientific interpretation
Confirms clean qwen compliance and validity removes the two initial rewrite mechanical defects (pair truncation, fragment padding). Sharpens interpretation: a single Δ Overall combines at least five confounded mechanisms —
1. paired semantic alignment (intended); 1a. within-row propositional redundancy;
2. generated-register injection; 3. source/domain reweighting (treatment replaces uneven official words with a gutenberg/simple_wiki-heavy pair mix); 4. token-exposure/truncation asymmetry (trainer truncates at 256; +18,351 excess tokens/10M in Qwen arm).
Residual, unquantified fraction of pairs are style-distorted or occasionally meaning-changed (entity/overlap filters miss relation/tense inversion).

## Decision gate (unchanged)
- Complete nine-column ΔOverall (qwen_clean_aligned − official_lengthmatched) < +0.25 → terminate clean-Qwen route, return to stronger directions.
- ΔOverall ≥ +0.25 → run controls before any mechanism/SOTA claim.

## Prepared controls (ordered by isolating power; scripts ready)
1. **Shuffled-pair control** (most decisive for adjacency): `scripts/materialize_shuffled_pair_control.py` + new `scripts/launch_shuffled_control_training.sh`. Same selected originals and Qwen rewrite multiset, rewrites permuted within source/length bins to break correspondence. Trains `qwen_shuffled_control_16k_seed43022` under identical recipe/seeds. Treatment>shuffled ⇒ adjacency matters; treatment≈shuffled ⇒ style/domain.
2. **Original-duplication control** (isolates generalization vs redundancy): to build — rows placing each original twice (no Qwen text), same word budget.
3. **Source-matched control** (isolates domain reweighting): to build — rebuild official control so per-source word counts equal `source_word_counts_qwen_treatment`.
4. **Token-matched control** (if truncation-at-256 matters): to build — match subword tokens, not just whitespace words.
5. **Second seed** (43122/43123) + **per-task delta profile** (syntax-concentrated ⇒ register; broad ⇒ alignment).
6. Complete manual relabel of the 220-item spotcheck to quantify meaning-change/corruption rate.

## Compliance (partial; internal validity unaffected)
`staging/acquired/BabyLM_Challenge_613440df8812e194/content.md`: 2026 Strict/Strict-Small allows teacher-model feedback, releases a 10M detoxified set, keeps ≤10 epochs. Generated-text/tokenizer-provenance for submission not yet fully settled. This experiment is currently an internal scientific test, not a submission claim.

## Full evaluation after training
`bash scripts/launch_full_eval.sh` → `full_eval_runner.py` (delegates to `full_overall_eval_runner`): full zero-shot + 7-task SuperGLUE + AoA (checkpoint ladder now saved every 1M so AoA is computable). Summary + gate in `summarize_full_eval.py`.
