# clean qwen compliance and validity clean-Qwen route: compliance and validity record

## Why initial rewrite materialization was stopped

The initial rewrite materializer was not a clean test of semantic alignment: it could truncate original--rewrite pairs and fill incomplete rows with unrelated official word fragments. A spotcheck of Qwen rewrites also contained incomplete sources, residual fragments, and some meaning changes. Training that corpus would confound semantic alignment with generation noise and packing artifacts.

## Replacement materialization design

`scripts/clean_materialize_qwen_pairs.py` now treats clean corpus construction as a prerequisite to training:

- validates Qwen outputs with explicit surface filters for length ratio, entities, numbers, repetition, duplicates, content overlap, and source/rewrite completeness;
- rejects incomplete source or rewrite fragments, unbalanced quotes/brackets, ellipses, wiki/CHILDES path fragments, bad terminal fragments, and common prompt/meta artifacts;
- preserves complete original--rewrite pair boundaries in `pack_pairs()`;
- forbids pair truncation and word-fragment padding in pair rows;
- fills the remaining treatment budget only with complete 160-word official rows;
- constructs an official-only control whose row-length sequence exactly matches the treatment, so the trainer sees the same whitespace-word batch shape;
- repeats each 10M pool for 10 shuffled passes using the same per-pass index orders, giving exact 100M training files;
- saves `clean_pairs.jsonl`, `selected_pairs.jsonl`, packed-row metadata, spotcheck files, and SHA256 hashes.

`scripts/audit_clean_corpus.py` is a pretraining audit that verifies word totals, SHA256s, row-length matching, source and cohort composition, reuse of selected official examples in filler, and baseline16k tokenizer length/truncation statistics before the long run can begin.

`scripts/launch_clean_training.sh` now refuses to start if run directories are non-empty, if metadata fields disagree with the intended clean construction, if corpus hashes do not match metadata, or if the selected pair fraction is below 10%.

## Extra generation

The first initial rewrite generation yielded only about 0.825M selected clean pair words under the initial stricter filter, below the 10% minimum for a meaningful matched training run. Therefore `scripts/extract_extra_clean_prompts.py` created additional complete-sentence prompts and sharded them for two-H100 Qwen generation. The prompt asks Qwen to preserve names, speaker labels, numbers, dates, quantities, and quoted words exactly.

Additional generation used the scientific output identifiers `extra_qwen2_shard0` and `extra_qwen2_shard1`; completed acceptance counts are separate measurements.

## Interpretation limits

The corrected materializer fixes the two original errors, but the result should be interpreted as a filtered-Qwen augmentation pipeline unless additional controls are run. Main points to preserve:

- surface filters do not prove semantic equivalence;
- accepted samples still need stratified spotcheck after final selection;
- treatment changes Qwen-generated style, semantic adjacency, official-content replacement/repetition, source mixture, and line structure;
- whitespace-word matching is not tokenizer/truncation matching, hence the preflight tokenizer audit is required;
- a shuffled-pair control and ideally an original-duplication control are needed before claiming the mechanism is paired semantic alignment;
- official-rule evidence for Qwen-generated text remains incomplete until the 2026 guidelines/call text is fully verified.

## Current official-rule evidence

A fresh `knowledge_request` acquired `data/external/content.md` from `https://babylm.github.io` fetched 2026-08-26. It states:

- Strict/Strict-Small now subsume former Multimodal/Interaction tracks and allow paired image-text data and teacher-model feedback;
- a detoxified 10M-word Strict-Small dataset is released;
- competition entries may not conduct more than 10 epochs over their training data.

This is useful but not complete. The acquired page points to guidelines and the updated call for papers for detailed rules, but the retrieved content does not fully specify generated text, tokenizer provenance, external-model use, or distillation policy. Older official calls/findings support custom datasets and synthetic/teacher-related methods under budgeted constraints, but the 2026-specific final text still needs direct confirmation before any official submission. Therefore clean qwen compliance and validity training, if launched, is an internal scientific test; it is not yet a submission-ready compliance conclusion.

## Interpretation before result

If the clean qwen compliance and validity matched training later gives `qwen_clean_aligned - official_lengthmatched` complete nine-column Overall below +0.25, terminate this clean-Qwen route. If it is at least +0.25, proceed to:

1. second-seed replication;
2. shuffled-rewrite-pair control with identical generated text but broken original--rewrite adjacency;
3. original-duplication/generated-style controls if needed;
4. direct 2026 guideline/call confirmation before submission language.

Only replicated causal improvement plus official-server score above the current leader can support a SOTA claim.
