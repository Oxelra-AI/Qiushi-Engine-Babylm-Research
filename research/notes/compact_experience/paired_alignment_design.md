# paired alignment design — Paired alignment experiment: design and rationale

## Attention-visibility constraint

The original route decision mechanism plan would have placed paired text as independent JSONL rows. The fundamental flaw is: if both sides of a pair are independent MLM training examples, the model processes them in separate forward passes. Self-attention cannot see across examples. Therefore, what the experiment would actually measure is:

1. **Semantic repetition**: the model sees the same *meaning* twice (in different surface forms)
2. **Lexical diversity**: the paired corpus has more diverse vocabulary than the official corpus

Neither of these is "alignment" in any mechanistically meaningful sense. The model has no mechanism to detect that two sentences separated by many training steps express the same meaning.

## How alignment is made visible

To test whether the model benefits from *seeing aligned alternative expressions*, both sides of a pair must appear within the **same input sequence** (same 160-word training example). This way:

- Self-attention directly attends across both the original and simplified versions
- When a word in the simplified version is masked, the model can use the adjacent original (same meaning, different wording) as context for prediction
- The MLM gradient signal from masked tokens in one view is **informed by** the other view

## Experimental design (four arms + baseline)

All arms use: DeBERTa-v2 8×480, baseline16k tokenizer, WWM 0.15, seq 256, batch 256, LR 1e-3, weight_decay 0.01, seed 43, 100M word exposure.

| Arm | Structure within 160-word examples | Word budget | What it tests |
|-----|-----------------------------------|-------------|---------------|
| **ALIGNED** | [orig_1][simp_1][orig_2][simp_2]... | 9,999,840 | Paired alignment visible to attention |
| **MISMATCHED** | [orig_i][simp_j][orig_k][simp_l]... (i≠j) | 9,999,840 | Same text, same structure, no alignment |
| **SINGLE_REPEAT** | originals from pairs, each 2× in different contexts | 9,999,840 | Repetition consolidation, no diversity |
| **SINGLE_ORIG** | diverse unique originals only | 9,551,200 | Max semantic coverage, no pairing |
| **INITIAL_MODEL_STUDIES baseline** | Official BabyLM 10M corpus | 10,000,000 | Reference (inherited 40.7028 coordinate) |

## Critical contrasts

1. **ALIGNED vs MISMATCHED** (most important): Same sentences, same within-sequence alternating structure, same word counts. ONLY difference is whether semantically related text is adjacent. If ALIGNED wins, the model genuinely benefits from within-sequence semantic alignment.

2. **ALIGNED vs SINGLE_REPEAT**: Does multi-view diversity (seeing the same meaning in different surface forms) outperform repetition consolidation (seeing the same surface form twice)?

3. **ALIGNED vs SINGLE_ORIG**: Does depth per meaning (aligned pairs, fewer unique meanings) beat breadth (more unique meanings, one view each)?

4. **MISMATCHED vs SINGLE_ORIG**: Does having simplified-style text in the corpus help even without alignment (domain/style effect)?

## Data sources

- **WikiLarge** (Nechba/wikilarge-text-simplification): Apache-2.0 license, 148,795 aligned Normal/Simple sentence pairs
- **SynCSE-partial-NLI** (hkust-nlp/SynCSE-partial-NLI): MIT license, 261,641 sentence/paraphrase pairs

Combined pool: 410,436 pairs, ~14.2M pair-words available. Selected 288,098 pairs (9,999,943 pair-words) for the 10M budget.

## BabyLM legality

- Custom data is explicitly allowed under 2026 rules
- All text seen by training counts toward the ≤10M word budget: ✓ (each arm ≤10M)
- Total exposure ≤100M word exposures: ✓ (exactly 100M per arm)
- No distillation: ✓ (published text pairs, no model-generated or model-distilled)
- No external model weights/hidden states/output exposed to submission model: ✓

## Validation results

- ALIGNED and MISMATCHED are exactly matched: 62,499 examples, 9,999,840 words each
- Word multiset between ALIGNED and MISMATCHED differs by only 84 tokens (0.00084% boundary artifact from sentence-order-dependent packing — negligible)
- SINGLE_REPEAT exactly matched at 9,999,840 words
- SINGLE_ORIG slightly smaller at 9,551,200 words (95.5% of budget — all available diverse originals exhausted)
- Alignment structure verified by manual inspection of example text

## Training configuration

Each arm trains for 100M word exposure (10 shuffled epochs over ~10M pool):
- ~2,442 optimization steps per arm
- 10 checkpoints saved (every 10M words)
- Estimated ~65 min per arm on H100
- Wave 1 (ALIGNED + MISMATCHED): both H100s in parallel
- Wave 2 (SINGLE_REPEAT + SINGLE_ORIG): both H100s in parallel

## Files

- Pool JSONLs: `experiments/archive/compact_experience/data/paired_alignment/{arm}_pool.jsonl`
- Expanded training files: `.../training_expanded/{arm}_100M.jsonl` (625K rows, 100M words each)
- Screen summary: `.../screen_summary.json`
- Materializer script: `experiments/archive/compact_experience/scripts/paired_alignment_materializer.py`
- Expansion/launch script: `.../expand_and_train.py`
- Wave 1 training: `.../wave1_train.sh`
- Wave 2 training: `.../wave2_train.sh`
- Evaluation: `.../eval_paired_alignment.py`

## What a positive result means

If ALIGNED significantly beats MISMATCHED on the official evaluation columns, then:
- Within-sequence semantic alignment is a genuine learning signal for small masked language models
- The leader's advantage is at least partly attributable to its paired simplification data
- This provides a data-efficiency principle: **meaning-preserving multi-view exposure within context windows enables more efficient learning of generalizable language knowledge**

If ALIGNED does NOT beat MISMATCHED, then:
- The leader's advantage must come from its tokenizer (40k), capacity (12 layers), optimizer (LAMB), or data domain (FineWeb), not from alignment per se
- The next route should focus on those factors
