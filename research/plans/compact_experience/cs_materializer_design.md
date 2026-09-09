# earlier analysis — C/S Materializer Design for 4M Matched-Content Screen

## Purpose

Build the first COMPACT_EXPERIENCE experiment: 4–5 arms of 4,000,000 whitespace words each,
selected from the official BabyLM Strict-Small 10M-word pool using compositional
structure density (C-score) and recoverable signal density (S-score), with
source-matched random references and a composition-separated control. Every arm
feeds directly into the full-cycle DeBERTa-v2 WWM trainer via `--example_jsonl`.

This document specifies every data structure, scoring function, selection logic,
output format, and validation so the implementation script can be written from
it without ambiguity.

## Pool Construction

### Source files

Read from the official BabyLM 2026 Strict-Small dataset. If already downloaded
under an INITIAL_MODEL_STUDIES run directory, reuse the raw files; otherwise download via the
HuggingFace `BabyLM-community/BabyLM-2026-Strict-Small` repo.

| Source file              | Words (epoch 1) | Proportion |
|--------------------------|-----------------|------------|
| childes.train.txt        | 2,841,120       | 28.41%     |
| gutenberg.train.txt      | 2,557,760       | 25.58%     |
| open_subtitles.train.txt | 2,282,880       | 22.83%     |
| simple_wiki.train.txt    | 1,531,520       | 15.32%     |
| bnc_spoken.train.txt     |   762,080       |  7.62%     |
| switchboard.train.txt    |    24,640       |  0.25%     |
| **TOTAL**                | **10,000,000**  | **100%**   |

### Row packing

Reproduce the exact `iter_examples()` logic from `babylm_masked_train_fullcycle.py`:
- Read files in the fixed order: bnc_spoken, childes, gutenberg, open_subtitles,
  simple_wiki, switchboard (this is the `TRAIN_FILES` order).
- Pack words sequentially into 160-word rows (`words_per_example=160`).
- Each row records: `text` (space-joined words), `words` (exactly 160 or tail),
  `example_id` (sequential integer), `source` (filename of the first word).
- Pool cap: `max_words=10,000,000` to read the full unique corpus once.
- Result: ~62,500 rows.

**Important**: the `iter_examples` packing reads files in `TRAIN_FILES` order and
does NOT shuffle. This means rows at the beginning come from bnc_spoken, then
childes, etc. Cross-source boundary rows (where a source file ends mid-row) get
the source of the first word. The materializer must reproduce this exactly to
give every row a stable identity.

### Row schema (internal, pre-scoring)

```python
@dataclass
class PoolRow:
    example_id: int       # sequential, 0-based
    text: str             # space-joined 160 words
    words: int            # whitespace word count (usually 160)
    source: str           # filename of the first word
    # Added by scoring:
    c_score: float        # compositional structure density
    s_score: float        # recoverable signal density
    token_count: int      # tokens under the training tokenizer
```

## Scoring Functions

### C-score: Compositional Structure Density

Parse each row with `spacy.load("en_core_web_sm")` and compute per-row:

```
c_raw = (
    n_finite_verbs           # POS=VERB with TAG in {VBZ, VBP, VBD, VBN, VBG, MD}
  + n_clausal_deps           # dep in {advcl, ccomp, xcomp, relcl, acl}
  + n_core_args              # dep in {nsubj, nsubjpass, dobj, obj, iobj, attr}
  + 0.5 * n_modifiers        # dep in {amod, advmod, prep, pobj, det}
  + n_complete_sents          # rows parsed as having ≥1 complete sentence (root + nsubj/expl)
  + 0.5 * n_entity_mentions  # NER entity spans
)

c_penalty = (
    2.0 * n_fragment_sents   # sentences with no finite verb and no clausal head
  + 1.0 * n_filler_tokens    # tokens matching filler/backchannel patterns
  + 1.5 * n_boilerplate_lines  # lines that are section headers, bullet markers, etc.
)

c_score = (c_raw - c_penalty) / words
```

Where:
- `n_finite_verbs`: tokens where `token.pos_ == "VERB"` and `token.tag_` in
  `{"VBZ", "VBP", "VBD", "VBN", "VBG", "MD"}`.
- `n_clausal_deps`: tokens whose `token.dep_` is in `{"advcl", "ccomp", "xcomp",
  "relcl", "acl"}`.
- `n_core_args`: tokens whose `token.dep_` is in `{"nsubj", "nsubjpass", "dobj",
  "obj", "iobj", "attr"}`.
- `n_modifiers`: tokens whose `token.dep_` is in `{"amod", "advmod", "prep",
  "pobj", "det"}`.
- `n_complete_sents`: count of sentences in `doc.sents` where at least one token
  has `dep_ == "ROOT"` and another token in the same sentence has `dep_` in
  `{"nsubj", "nsubjpass", "expl"}`.
- `n_entity_mentions`: count of `doc.ents`.
- `n_fragment_sents`: count of sentences in `doc.sents` that have no token with
  `tag_` in `{"VBZ", "VBP", "VBD", "VBN", "VBG", "MD"}`.
- `n_filler_tokens`: tokens whose lowered text matches filler regex:
  `^(uh|um|erm|hmm|mhm|huh|yeah|yep|yup|nah|ok|okay|oh|ah|wow|hm|mm)$`.
- `n_boilerplate_lines`: lines in the raw text matching patterns like
  `^= =`, `^#`, `^\*[A-Z]+:`, `^[A-Z]:[\t]` (CHILDES/switchboard speaker tags,
  wiki section headers). These carry structural formatting but minimal compositional
  content for MLM.

The C-score is **per word** so short and long rows are comparable.

### S-score: Recoverable Signal Density

This does NOT require spacy parsing. It uses the training tokenizer and
lightweight text statistics.

Load the portable INITIAL_MODEL_STUDIES baseline16k tokenizer from `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model` (the exact tokenizer used by the 40.7028 run). This local path avoids the read-only shared HuggingFace cache.

```
# Compute per-row:
tokens = tokenizer.encode(text, add_special_tokens=False)
token_count = len(tokens)
unique_tokens = len(set(tokens))
token_freqs = [global_unigram_freq.get(t, 0) for t in tokens]

# Moderate-frequency ratio: fraction of tokens whose corpus unigram frequency
# is in [50th, 95th] percentile of the global distribution
moderate_freq_ratio = sum(1 for f in token_freqs if p50 <= f <= p95) / token_count

# Local type-token ratio (bounded diversity without rewarding extreme rarity)
local_ttr = unique_tokens / token_count

# Sentence completeness: fraction of spacy sentences with a finite verb
sent_completeness = n_complete_sents / max(n_total_sents, 1)

# Repetition control: penalize near-duplicate bigrams within the row
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
unique_bigrams = len(set(bigrams))
bigram_diversity = unique_bigrams / max(len(bigrams), 1)

# Noise penalty: fraction of tokens that are digits, punctuation-only, or
# single-character non-letter
noise_ratio = sum(1 for t in text.split() if is_noise_word(t)) / words

s_score = (
    0.35 * moderate_freq_ratio
  + 0.20 * local_ttr
  + 0.25 * sent_completeness
  + 0.15 * bigram_diversity
  - 0.30 * noise_ratio
)
```

Where `is_noise_word(w)` returns True if:
- `w` is all digits
- `w` is all punctuation (no letters)
- `w` is a single non-letter character

The S-score is already normalized to [0, ~1] range.

### Global unigram frequency table

Before scoring, build a global unigram frequency table by tokenizing the entire
10M-word pool with the training tokenizer. Store `global_unigram_freq[token_id] = count`.
Compute `p50` and `p95` as the 50th and 95th percentile count values across all
token types that appear at least once.

## Arm Selection

### Source quotas for 4M words

Select 40% of each source's rows to maintain source proportions:

| Source           | Pool rows | 4M quota (rows) | 4M quota (words) |
|------------------|-----------|------------------|-------------------|
| bnc_spoken       | ~4,763    | 1,905            | 304,800           |
| childes          | ~17,757   | 7,103            | 1,136,480         |
| gutenberg        | ~15,986   | 6,394            | 1,023,040         |
| open_subtitles   | ~14,268   | 5,707            | 913,120           |
| simple_wiki      | ~9,572    | 3,829            | 612,640           |
| switchboard      | ~154      | 62               | 9,920             |
| **TOTAL**        | ~62,500   | **25,000**       | **4,000,000**     |

Note: exact quotas will be adjusted at materialization time to hit exactly
4,000,000 words, since some rows may have <160 words (tail rows at source
boundaries). The materializer must verify `sum(words) == 4,000,000` for each arm.

If exact 4M is not achievable with integer rows, allow the last row per source
to be a partial row (truncated text with adjusted word count), or adjust the
largest source quota by ±1 row. The JSONL loader in the full-cycle trainer
validates exact word count match.

### Arm definitions

**R-a (random reference A)**:
- For each source, draw `quota` rows uniformly at random (seed=100).
- No score-based selection.

**R-b (random reference B)**:
- Same as R-a but with seed=200.
- Independent draw — measures run-to-run and order variation.

**C-high (compositional density)**:
- For each source, sort rows by C-score descending, take top `quota`.

**C-control (composition-separated control)**:
- For each source, sort rows by C-score ascending (lowest first), take top `quota`.
- Same source proportions, similar surface statistics, but systematically lower
  compositional density.
- **Key contrast**: C-high minus C-control isolates compositional structure.

**S-high (signal density)**:
- For each source, sort rows by S-score descending, take top `quota`.

### Overlap analysis

After selection, compute and report:
- Row overlap between C-high and S-high (count and fraction of shared `example_id`s).
- C-score distribution for each arm (mean, std, p10, p50, p90).
- S-score distribution for each arm (mean, std, p10, p50, p90).
- If C-high ∩ S-high overlap > 60% of either arm, the scores are not separating
  different mechanisms and the score functions must be redesigned before training.

## Output Format

### Per-arm JSONL file

Path: `experiments/archive/compact_experience/data/cs_4m_screen/{arm_name}.jsonl`

Each line is a JSON object:
```json
{"text": "space-joined words...", "words": 160, "example_id": 42, "source": "gutenberg.train.txt", "kind": "official", "c_score": 0.234, "s_score": 0.567}
```

Fields:
- `text`: exactly `words` whitespace-separated words
- `words`: integer, validated against `len(text.split())`
- `example_id`: integer from the pool, stable across arms
- `source`: original filename
- `kind`: always `"official"` for BabyLM legality
- `c_score`, `s_score`: float, for provenance (trainer ignores them)

Row order within each arm: sorted by `example_id` ascending (deterministic,
source-interleaved, no curriculum order).

### Per-arm metadata JSON

Path: `experiments/archive/compact_experience/data/cs_4m_screen/{arm_name}_meta.json`

Contents:
```json
{
  "arm_name": "C-high",
  "description": "Top compositional-density rows per source, 4M words",
  "total_words": 4000000,
  "total_rows": 25000,
  "source_quotas": {"childes.train.txt": {"rows": 7103, "words": 1136480}, ...},
  "selection_method": "top c_score per source",
  "selection_seed": null,
  "c_score_stats": {"mean": 0.234, "std": 0.05, "p10": ..., "p50": ..., "p90": ...},
  "s_score_stats": {"mean": ..., ...},
  "token_count": 54321,
  "tokens_per_word": 1.23,
  "type_token_ratio": 0.45,
  "rare_word_fraction": 0.12,
  "punct_digit_rate": 0.03,
  "dialogue_marker_rate": 0.08,
  "duplicate_estimate": {"exact_dup_rows": 0, "near_dup_fraction": 0.01},
  "example_ids": [0, 3, 7, ...],
  "text_multiset_hash": "sha256hex..."
}
```

### Screen-level summary JSON

Path: `experiments/archive/compact_experience/data/cs_4m_screen/screen_summary.json`

```json
{
  "screen_name": "cs_4m_matched_content",
  "pool_total_words": 10000000,
  "pool_total_rows": 62500,
  "arms": ["R-a", "R-b", "C-high", "C-control", "S-high"],
  "words_per_arm": 4000000,
  "source_matching": "proportional to official corpus",
  "c_high_s_high_overlap": {"shared_ids": 1234, "fraction_of_c_high": 0.05, "fraction_of_s_high": 0.06},
  "global_unigram_freq_p50": 1234,
  "global_unigram_freq_p95": 56789,
  "scoring_notes": "spacy en_core_web_sm for C-score, baseline16k tokenizer for S-score",
  "pool_construction": "iter_examples(TRAIN_FILES, max_words=10M, words_per_example=160)",
  "timestamp_utc": "2026-08-24T...",
  "trainer_interface": "--example_jsonl path/to/arm.jsonl --max_word_exposure 4000000"
}
```

## Training Configuration (prepared, not launched)

For each arm, the fixed DeBERTa-v2 WWM training command will be:

```bash
python experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py \
    --output_dir experiments/archive/compact_experience/training/runs/cs_4m_{arm_name} \
    --example_jsonl experiments/archive/compact_experience/data/cs_4m_screen/{arm_name}.jsonl \
    --example_jsonl_label "{arm_name}" \
    --example_jsonl_meta experiments/archive/compact_experience/data/cs_4m_screen/{arm_name}_meta.json \
    --tokenizer_path experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model \
    --tokenizer_label baseline16k_step263 \
    --max_word_exposure 4000000 \
    --example_pool_words 4000000 \
    --words_per_example 160 \
    --model_type deberta_v2 \
    --hidden_size 480 \
    --n_layer 8 \
    --n_head 8 \
    --ffn_mult 4 \
    --mask_mode wwm \
    --mask_prob 0.15 \
    --seq_length 256 \
    --max_seq_length 256 \
    --max_position_embeddings 512 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --warmup_fraction 0.05 \
    --checkpoint_words 1000000 \
    --seed 43 \
    --log_every 50
```

Key parameters match the INITIAL_MODEL_STUDIES 40.7028 recipe: DeBERTa-v2 8×480, WWM 0.15,
seq256, seed43, batch64, LR 1e-3. The only experimental variable is the
content of `--example_jsonl`.

Checkpoints saved: chck_1M, chck_2M, chck_3M, chck_4M.

### Evaluation plan (prepared, not launched)

Use the INITIAL_MODEL_STUDIES evaluation pipeline for fast-screen columns:
- BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS, GlobalPIQA
- Evaluate each arm's chck_4M checkpoint
- Report per-column scores and partial-task proxy
- Pattern analysis: which columns separate C-high from C-control? Does S-high
  improve different columns than C-high?

## Validation Checks

Before any training, the materializer must verify:
1. Each arm has exactly 4,000,000 whitespace words
2. Each arm's source proportions match the planned quotas (±1 row tolerance)
3. No row appears twice within any arm
4. C-high and C-control have no shared rows
5. R-a and R-b have different row sets (with measured overlap from random draws)
6. C-high/S-high overlap is reported and < 60%
7. Each arm's JSONL passes a dry-run through `load_examples_jsonl()` with the
   exact `selected_words=4000000` target

## Implementation Notes

- The spacy scoring loop over ~62,500 rows with `en_core_web_sm` should take
  ~5-15 minutes on CPU. Use `nlp.pipe(texts, batch_size=256)` for efficiency.
- The tokenizer frequency table computation over 10M words should take ~2 min.
- The entire materializer should be one self-contained script that:
  1. Reads official corpus files (from INITIAL_MODEL_STUDIES cached path or fresh download)
  2. Packs into the pool
  3. Scores all rows
  4. Selects arms
  5. Writes JSONL + metadata + summary
  6. Runs all validation checks
  7. Prints a single JSON status line

Path: `experiments/archive/compact_experience/scripts/materialize_cs_4m_screen.py`

## Connection to Training Path

The full-cycle trainer's `load_examples_jsonl()` function:
- Reads JSONL lines in file order (no shuffle)
- Validates `words == len(text.split())` per row
- Raises RuntimeError if partial example would be needed
- Raises RuntimeError if `selected != selected_words`
- Returns `(examples, total_file_words, total_rows, sample_rows)`

The materializer MUST produce JSONL files that pass this validation with
`selected_words=4000000`.

## Remaining Questions

1. **Validation status**: The materializer is CPU-based. Training configs were
   prepared but training had not started in this record.
2. **Tokenizer choice**: The first screen uses baseline16k to isolate the
   content-selection variable. A later 40k SentencePiece arm can test the
   leader's tokenizer choice.
3. **LR schedule**: The 4M run has fewer steps than the 100M run. The cosine
   schedule automatically adapts via `lr_total_steps` default (= actual steps).
   This is intentional for a fast screen but means the LR trajectory differs
   from the 100M coordinate. If a content arm wins at 4M, the 10M/100M scaling
   run will use the full schedule.
