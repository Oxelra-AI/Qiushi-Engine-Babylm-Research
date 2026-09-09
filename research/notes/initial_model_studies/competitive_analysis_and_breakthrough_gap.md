# BabyLM Strict-Small — Competitive Analysis & Breakthrough Gap

Sources: 2025 Findings (official) Table 3, MoEP paper, AMLM paper.
Literature review date: 2026-08-13.

## 1. Official 2025 Strict-Small Scores (Table 3, Findings paper)

### Best Models
| Model | Human-likeness | NLP score | Macro Avg |
|---|---|---|---|
| MoEP* (human-likeness winner) | 31.5 | 53.2 | **42.3** |
| AMLM-Hard-Decay† (NLP winner) | 8.4 | 58.3 | 33.3 |
| GPT-BERT-ACLM-6k-MNTP | 21.5 | 50.3 | 35.9 |
| **AMLM-Hard** | **26.7** | **58.0** | **42.3** |

(* = human-likeness winner, † = NLP winner)

### Baselines
| Model | Human-likeness | NLP score | Macro Avg |
|---|---|---|---|
| GPT-BERT-causal MNTP | 17.4 | 56.7 | 37.1 |
| GPT-BERT-causal AR | 18.7 | 56.1 | 37.4 |
| GPT-BERT-mixed MNTP | 13.4 | 57.3 | 35.4 |
| GPT-BERT-mixed AR | 16.6 | 54.3 | 35.4 |
| GPT-BERT-masked MNTP | 9.5 | 57.3 | 33.4 |
| GPT-BERT-masked AR | 18.9 | 54.0 | 36.5 |
| GPT2 | 19.8 | 49.1 | 34.5 |

## 2. The Central Discovery: Strict-Small Uniquely Has Negative Correlation

The Findings paper (Section 7) explicitly states:
> "We found a positive correlation between linguistic and cognitive modeling task performance, **except for the Strict-Small track**."

This means:
- In Strict (100M words): improving NLP also improves human-likeness → synergy.
- In Strict-Small (10M words): improving NLP may hurt human-likeness → **tradeoff**.
- Under 10M words, representational capacity is so constrained that the model must specialize. It cannot simultaneously capture both word-level usage trajectories (needed for AoA, reading time, entity tracking) AND abstract syntactic/semantic patterns (needed for BLiMP, GLUE).

This is the **root scientific problem** this research must solve.

## 3. The Two Leading Strategies (MoEP vs AMLM) Are Complementary

### MoEP (Tapaninaho, 2025)
- **Architecture**: Sparse decoder-only. 2 Large Layers (d=384) → MoE shrink (4 experts, top-2) → 10 Parallel Layers with top-2 routing over 4 blocks at d=192 → MoE grow → Large Layer. 28M params.
- **Objective**: Standard CLM (next-token prediction).
- **Tokenization**: 16K BPE, GPT-2 style.
- **Strength**: Human-likeness 31.5 (AoA task 53.7, Entity Tracking 35.65).
- **Weakness**: NLP 53.2 (below baseline GPT-BERT-causal 56.7).
- **Training**: Peaks at 30M words, then degrades (overfitting). Fast early learning but unstable.
- **Why it works for human-likeness**: Sparse routing creates diverse computational pathways. Different tokens use different expert subsets → captures word-level distributional variation that matches AoA/reading-time patterns.

### AMLM-Hard (Edman & Fraser, 2025)
- **Architecture**: DeBERTa-V2 (encoder-only, disentangled attention). Similar to GPT-BERT backbone.
- **Objective**: Adaptive Masked LM. Tracks per-token accuracy. High-accuracy tokens get masked less; low-accuracy get masked more. Hard metric = smoothed accuracy per token ID. Decaying mask prob 40%→15%.
- **Tokenization**: 40K BPE with byte-fallback.
- **Data**: Custom (replaced CHILDES with synthetic paraphrase triplets).
- **Strength**: NLP 58.0 (far above MoEP's 53.2).
- **Weakness**: Human-likeness 26.7 (below MoEP's 31.5).
- **Why it works for NLP**: Adaptive masking focuses compute on difficult tokens, forcing the model to learn harder linguistic patterns → better syntactic/semantic generalization.

### The Net Result
Both achieve **42.3 Macro Average** through opposite specializations:
- MoEP: +4.8 Human-likeness, -4.8 NLP (vs AMLM-Hard)
- AMLM-Hard: -4.8 Human-likeness, +4.8 NLP (vs MoEP)

**The gap is exactly 4.8 points on each sub-score.** A model that bridges this gap would reach ~47 Macro.

### Additional Observations
- AMLM-Hard-Decay achieves the highest single-task NLP score (58.3) but human-likeness collapses to 8.4 → Macro 33.3. The "decay" mechanism (scheduled reduction of mask rate) amplifies NLP specialization at catastrophic human-likeness cost.
- AMLM N-Hot (Table 2 of AMLM paper): Gets +30 on adjective nominalization (64.0 vs baseline 2.7), but BLiMP drops from ~70 to 65.6, Past-tense also drops. Overall "Final Score" 41.9 (different metric than Findings macro avg). N-hot provides morphological signal but disrupts syntax.
- GPT-BERT causal AR baseline: Surprisingly competitive (37.4 Macro) with just CLM objective.

## 4. Tokenization: The Underexplored Lever

2025 Findings Appendix D.4:
> "Tokenization choices exert disproportionately large effects in our constrained setups, often rivaling objective or architectural modifications. Models using morphology-aware tokenizers demonstrated substantial gains in entity tracking and world knowledge tasks. One submission reported ~20% improvement on EWoK and ~40% on entity tracking with morpheme segmentation."

Key table:
| Tokenizer | Strongest on |
|---|---|
| BPE | Syntactic acceptability (BLiMP) |
| Morphology/syllable-aware | Semantic generalization, discourse tracking |
| BPE (English-trained on morphologically rich) | Token inflation reduces usable budget |

**Critical rule**: "Tokenizer counts toward the 10M word budget" (2026 CfP §5 FAQ). This is strategically underexplored — most submissions use a stock BPE tokenizer trained on the corpus. A better tokenizer that uses the budget more efficiently could give disproportionate gains.

## 5. MoEP Technical Architecture (what to inherit/improve)

From MoEP paper Table 2-3:
```
Hyperparameter       GPT-2    MoEP      MoEP-SwiGLU
Vocabulary size      16K      16K       16K
d_model              384      384/192   384/192
Layers               12       2/10      2/10
Parallel blocks      -        4         4
Heads                6        6/3       6/3
Head dimension       64       64        64
FF multiplier        4        4         4
FF type              MLP      MLP       SwiGLU
MoE FF type          -        Linear    SwiGLU
N experts            -        4         4
Top k                -        2         2
Total params         28M      28M       38M
```

Training: AdamW, lr=3e-4, batch=16, cosine schedule, 800 warmup, weight decay 0.1.
Best checkpoint at 30M words (out of 100M budget). Sequencing: 512 tokens.

**MoEP's key failure mode**: After 30M words, performance degrades. The load-balancing regularizer may force uniform expert usage, limiting specialization. Entity tracking collapses dramatically after early gains.

## 6. AMLM Technical Details

From AMLM paper:
- Adaptive masking update: every 200 batches. λ=0.2 EMA.
- Token probabilities per type i: w_t,i = λ·w_{t-1,i} + (1-λ)·p_mlm·(1-score_{t-1,i})
- Hard score = smoothed accuracy (correct+0.5)/(total+1) per type
- Soft score = 1 - norm(loss) per type
- Hard method masks more frequent tokens MORE at start, LESS later
- POS analysis: adverbs, subordinating conjunctions, numbers, verbs get higher mask weights
- 40K BPE vs MoEP's 16K. Larger vocab lets fewer tokens cover same text → lower sequence length → less compute per batch, but vocabulary sparsity.

**AMLM's key failure mode**: Adaptive masking hurts human-likeness (AoA goes to -15.0 for hard variant). Focusing compute on difficult tokens may prevent the model from learning the natural usage trajectories needed for word-acquisition tasks.

## 7. Open Questions Driving Research

1. **Why the Strict-Small tradeoff?** Is it a competition for model capacity, optimization conflict, or data insufficiency? Understanding this determines the intervention strategy.

2. **Can a single mechanism improve BOTH human-likeness and NLP?** Candidates:
   a. **Learnable tokenization that encodes morphological structure without sacrificing syntax** — hybrid or multi-granularity tokenization.
   b. **Dual-objective sparse routing architecture** — combine MoEP's routing with AMLM's adaptive difficulty, but with routing conditioned on both token difficulty AND task-family signal.
   c. **Memory-augmented representation** — augment the model with an external memory that separately tracks word-level trajectories (for AoA/human-likeness) and abstract patterns (for syntax/semantics).

3. **What is the optimal checkpoint trajectory?** The Findings show different models peak at different word counts for different tasks. Can we design early stopping or sequential fine-tuning that captures the best of both?

4. **Can the "tokenizer counts toward budget" rule be exploited strategically?** Most submissions spend the same 10M words on both tokenizer and LM training. A model that learns its token representations jointly with the main objective could be more efficient.

## 8. Research Direction Selection (for 2026 eval and experiment plan decision)

Comparing the candidate directions:

**Direction A: "Dual-objective Sparse Architecture"** (highest promise)
- Start from MoEP's sparse routing backbone (proven human-likeness gains from path diversity)
- Replace CLM objective with a dual loss: (1) standard next-token prediction for fluency, (2) adaptive difficulty prediction (AMLM-style) for hard-token learning
- Route tokens to different objectives based on difficulty — easy tokens get CLM, hard tokens get extra masked prediction
- Risk: May not resolve the capacity tradeoff; complexity of routing+dual objective is high.

**Direction B: "Learnable Tokenization / Multi-granularity Input"** (highest leverage per budget)
- Instead of fixed BPE, learn token representations dynamically within the LM
- Use a character/byte-level encoder with learned soft-pooling to token-like units
- Provide both character-level and token-level pathways to the model
- This bypasses the "tokenizer counts toward budget" rule inefficiency
- Risk: Previous character-level BabyLM attempts (Edman 2023, Goriely 2024) didn't improve scores; but those used fixed character representations, not learned ones.

**Direction C: "Checkpoint Trajectory Ensemble"** (lowest risk)
- Train one model, select checkpoints per task based on fast-eval scores
- Combine predictions at inference time
- Risk: Adds inference complexity; leaderboard may require single checkpoint.

**Direction D: "Cognitive-NLP Multi-Task Objective"**
- Design pretraining with a joint objective: LM loss + human-likeness proxy loss (e.g., AoA curve fitting, reading time correlation) during the same training.
- Risk: Human-likeness tasks require evaluation data that may not be usable as training signal without contamination.

**Recommendation**: Pursue Direction A (dual-objective sparse architecture) as the primary line, with Direction B (tokenization innovation) as a rapidly prototyped parallel sub-question. Direction C is a good safety net. Direction D requires careful contamination analysis.

The decisive experiment: compare three 10M-word variants:
1. MoEP baseline (CLM only) → replicate ~42.3
2. MoEP + AMLM-style adaptive masking → test if dual objective raises both sub-scores
3. MoEP + AMLM + n-hot sub-token embeddings → test if morphological signal closes the gap

This will determine whether the tradeoff is architectural (MoEP architecture favors human-likeness) or objective-level (CLM vs MLM) or representational (tokenization).
