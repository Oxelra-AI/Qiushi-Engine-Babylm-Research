# earlier analysis — Structured Experience Distribution Route

## Scientific Hypothesis

The protected DeBERTa-v2 8×480 WWM model (Overall 40.53) has its largest gaps to the visible leader (41.80) in Entity (-5.83), GlobalPIQA (-4.03), EWoK (-3.88), and SuperGLUE (-1.77), while it LEADS on Supplement (+3.84) and Reading (+2.195). The hypothesis is that replacing low-density conversational/child-directed text with structured sentence-level contrasts (paraphrases, entailment pairs, contradictions) will close the Entity/EWoK/SuperGLUE gaps while adding AMLM (adaptive masking) improves learning efficiency, AND that preserving enough developmental/narrative text will protect Supplement and Reading.

## Grounded Evidence Supporting This Route

### 1. AMLM (Edman et al., 2025) — Direct DeBERTa Entity/SuperGLUE breakthrough

- Architecture: DeBERTa-V2, hidden 384, intermediate 1280, 40k BPE, LAMB optimizer
- Data modification: Replaced child-directed speech (CHILDES) with synthetic triplets from Zhang et al. (2023a) — sentences, paraphrases, and contradictions
- Results (Table 5 ablation, Hard AMLM, 40k vocab):
  - With "New dataset": Entity 43.8, BLiMP 70.0, COMPS 53.1, SuperGLUE (finetune avg) 69.0
  - With "Original dataset": Entity 35.6, BLiMP 67.0, COMPS 51.9, SuperGLUE 64.1
  - Delta from data alone: Entity +8.2, BLiMP +3.0, COMPS +1.2, SuperGLUE +4.9
- This is the SAME DeBERTa architecture family as our protected model
- Entity 44.2 (best submitted) far exceeds our 22.62 and leader's 28.45 (different evaluation year but same task type)
- Citation: `\cite{edman2025maskd}`

### 2. RecGPT differential — Experience distribution is load-bearing

- Same architecture/optimizer/NextLat on official text vs custom corpus → BLiMP drops 18.07, Supplement drops 19.01
- Tokenizer NOT the explanation (recgpt differential route update measurement: official tokenizer is slightly better on probe words)
- The custom corpus (Baby-COSMO-Fine / ClimbMix lineage) provides high-density, web-quality sentence experience
- Citation: RecGPT-10M model card

### 3. Visible leader clues

- `wwm_curriculum_simplification_40k`: DeBERTa-v2, 12×384, WWM, SentencePiece 40k, custom data with "simplification" in name
- Entity 28.45, EWoK 56.07 — much higher than our protected model
- The "simplification" may refer to data reformulation/paraphrasing, not just Simple English Wikipedia

### 4. Our prior negative results (what NOT to do)

- Custom corpus 35% SimpleWiki/GEM mix (earlier analysis): Entity -0.06, GlobalPIQA -2.50, Reading -0.855 → NEGATIVE
- WikiAuto aligned rewrite (related experiments): Entity -0.42, EWoK -1.75 → NEGATIVE  
- Structure-density selection (structure density pretrain validation): Entity ±0, EWoK ±0 → NULL
- These all added simplified/entity-dense text WITHOUT structured contrasts; the AMLM finding shows that structured contrastive experience (paraphrases + contradictions), not just "cleaner text," is the key mechanism

## Mechanism Theory

The official BabyLM corpus is ~58% conversational/dialogue text (CHILDES, BNC dialogue, OpenSubtitles, Switchboard) that is:
- Low in informational density per word
- Repetitive in patterns (short turns, backchannel, disfluency)
- High in pronouns, discourse markers, and grounding cues → protects Reading and some Supplement
- Low in sentence-level semantic contrasts that teach entity state, entailment, and world knowledge

Structured sentence contrasts (paraphrases, contradictions, NLI triplets) provide:
- Dense per-word learning signal about meaning boundaries
- Explicit entailment/contradiction relationships → teaches SuperGLUE-relevant reasoning
- Varied entity references in controlled semantic frames → teaches Entity Tracking
- Diverse lexical/syntactic alternatives → teaches BLiMP-relevant grammaticality judgments

The critical balance: keep enough developmental/narrative text (Gutenberg stories, some CHILDES) to protect Reading and Supplement, while replacing the lowest-density conversational text with structured contrasts.

## Concrete Construction Plan

### Step A: Identify legal sentence-contrast data sources (< 1 hour)

Candidate sources (all publicly available, verifiable licenses):
1. **AllNLI** (SNLI + MultiNLI): ~1M sentence pairs with entailment/contradiction/neutral labels; CC-licensed premise/hypothesis pairs
2. **QQP** (Quora Question Pairs): ~400K question pairs with paraphrase/non-paraphrase labels
3. **PAWS** (Paraphrase Adversaries from Word Scrambling): ~108K high-quality paraphrase pairs
4. **ParaNMT-50M**: Large-scale back-translated paraphrases (CC-licensed subset available)
5. **Zhang et al. (2023a)** triplets: The exact source used by AMLM — check availability
6. **ASSET/TurkCorpus simplified targets**: Already materialized in the earlier experiments but failed in naive mix

Priority: AllNLI premise+hypothesis text is most directly analogous to what AMLM used (sentence-level contrasts). Word count and exposure must fit within 10M total.

### Step B: Characterize official corpus by source/type (< 30 min)

Measure: word count per source file, sentence length, pronoun/dialogue rate, entity density, lexical diversity. Identify the lowest-density conversational portions for replacement.

### Step C: Build two candidate 10M-word experience distributions

**Candidate 1: Structured-Contrast-Heavy (targets Entity/EWoK/SuperGLUE)**
- Keep: Gutenberg children's stories (~2.6M), Simple Wikipedia (~1.5M), some CHILDES/BNC (~1M) = ~5.1M words of official developmental/informational text
- Replace: ~4.9M words with NLI premise/hypothesis pairs, QQP questions, and/or Zhang et al. triplets as running text (sentence pairs formatted as adjacent sentences)
- Total: exactly 10M words

**Candidate 2: Balanced Mix (targets Entity/EWoK while preserving Reading/Supplement)**
- Keep: ~7M words of official text (all sources but reduced conversational)
- Replace: ~3M words with structured contrasts
- Total: exactly 10M words

### Step D: Training configuration

- Architecture: Protected DeBERTa-v2 8×480 (same as our best model)
- Tokenizer: baseline16k (preserves Reading, proven safe)
- Objective: AMLM hard method with decaying mask (40%→15%) — implements Edman et al.'s best configuration
- Optimizer: AdamW with cosine schedule (our proven setup) OR LAMB (AMLM's setup)
- Exposure: 10M words × 10 epochs = 100M word budget
- Checkpoints: chck_1M through chck_9M (AoA), then epoch marks

### Step E: Evaluation (fast sharded pipeline from earlier analysis)

- Use `score_freeze_sharded_eval.py` for batched AoA
- Use patched SuperGLUE classifier for fine-tuning evaluation
- Full 9/9 coordinate required before any claim

### Step F: Artifact contract (verified BEFORE training)

- [ ] Exact 10M word count with source ledger
- [ ] Tokenizer and data packed correctly
- [ ] Checkpoint trajectory: chck_1M–chck_9M at correct token-clock marks
- [ ] AoA sharded evaluation path verified
- [ ] SuperGLUE patched evaluator path verified
- [ ] Aggregation script ready

## What This Route Does NOT Do

- Does not change the architecture (stays on proven DeBERTa-v2)
- Does not use external language models for data generation (legal under BabyLM 2026)
- Does not add new parameters or modules
- Does not repeat closed routes (WikiAuto, structure-density, morphology adapter)
- Does not assume the RecGPT recipe is needed

## Success Criteria

- Entity Tracking ≥ 28.45 (match leader, close protected gap)
- EWoK ≥ 52.19 (preserve or improve)
- SuperGLUE ≥ 68.02 (preserve or improve)  
- GlobalPIQA ≥ 35.635 (preserve or improve)
- Supplement ≥ 56.04 (at least match leader level; protected leads at 59.88)
- Reading ≥ 5.42 (at least match leader level; protected leads at 7.62)
- Overall > 41.80 (beat visible leader)

## Files

- AMLM source: `data/external/content.md`
- RecGPT differential note: `notes/recgpt_differential_route_update.md`
- Previous custom corpus screen: `data/custom_mix_screen_scores.json`
- Protected DeBERTa 9/9: `data/debertav2_b256_true_9of9_coordinate.json`
- Leader local coordinate: `data/public_leader_available_coordinate.json`
