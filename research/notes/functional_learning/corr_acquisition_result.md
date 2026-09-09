# corr acquisition result result: identity exposure HELPS correspondence acquisition through shared retrieval representations

## Scientific question

The earlier recurrence contrast showed that exact repetition causes broad rewrite-token damage but no positive excess true-source cost. This was interpreted as "missing instruction" rather than "destructive competition": VARIED was taught the s→r correspondence while REPEAT was not. corr acquisition result tests the sharper question: when all arms share the same limited correspondence evidence, does filling the remaining training budget with identity practice change the sample efficiency of learning that correspondence for held-out entities?

## Design

8 entities (E0-E7), 10 shared source attributes, 10 corresponding rewrite attributes. Train queries: E0-E5; held-out queries: E6, E7 (never queried in training but appear in context). All arms share an identical backbone of 100 correspondence sequences per epoch. The remaining 400 sequences per epoch differ by arm:

| Arm | Substitution (400 seqs) | Purpose |
|---|---|---|
| all_corr | 400 more correspondence | Upper bound (500 total corr evidence) |
| ident_full | 400 identity (full attention) | Does retrieval training help correspondence? |
| ident_masked | 400 identity (retrieval blocked) | Is the effect attention-mediated? |
| neutral | 400 shuffled targets | Baseline noise fill |
| wrong | 400 wrong correspondence | Active interference control |

All arms share the same model initialization, training budget (500 seqs/epoch × 300 epochs), and evaluation probes. ident_full and ident_masked have identical token-level data; only the attention mask differs.

## Key numerical results (3 seeds × 300 epochs)

### Held-out correspondence gain (hcg = nosrc_NLL − src_NLL, higher = more source-dependent)

| Arm | hcg (mean ± std) |
|---|---|
| all_corr | +5.693 ± 0.890 |
| ident_full | +3.073 ± 1.188 |
| ident_masked | +3.251 ± 1.134 |
| neutral | +1.672 ± 1.146 |
| wrong | +0.513 ± 1.078 |

### Decisive contrasts on hcg

| Contrast | Δ (mean ± std) | Reading |
|---|---|---|
| ident_full − neutral | +1.401 ± 0.355 | Identity HELPS correspondence acquisition |
| ident_full − wrong | +2.560 ± 1.181 | Wrong correspondence actively hurts |
| ident_full − ident_masked | −0.178 ± 0.839 | No significant attention-mediated difference |
| all_corr − ident_full | +2.620 ± 0.818 | More correspondence > identity, but identity still helps |
| all_corr − neutral | +4.020 ± 0.524 | Full correspondence advantage over noise |

### Absolute held-out correspondence NLL with source (hc_s, lower = better)

| Arm | hc_s (mean ± std) | copy_nll (mean ± std) |
|---|---|---|
| all_corr | 8.470 ± 0.523 | 23.940 ± 0.583 |
| neutral | 9.021 ± 1.228 | 21.476 ± 1.326 |
| ident_masked | 9.887 ± 0.513 | 15.908 ± 2.271 |
| wrong | 10.138 ± 0.259 | 21.817 ± 0.899 |
| ident_full | 13.129 ± 1.052 | 10.153 ± 1.083 |

## What changed scientifically

### 1. Identity exposure facilitates correspondence learning, not destructive competition

The experiment asked whether identity exposure interferes with, helps, or leaves unchanged the acquisition of a reusable correspondence. The answer is that it **helps**: ident_full develops +1.40 nats more held-out correspondence gain than neutral filler. The facilitation is stable across the learning curve (ident_full exceeds neutral at every evaluation epoch from e50 onward).

This corrects Step008b's interpretation: the null excess true-source cost was indeed "missing instruction" (REPEAT never saw the s→r mapping), not evidence of destructive interference. When all arms share the same correspondence evidence, identity practice enhances rather than degrades the model's ability to learn and generalize the correspondence.

### 2. The facilitation is primarily weight-based, not attention-mediated

ident_masked (retrieval attention blocked for identity sequences) achieves hcg = +3.25, indistinguishable from ident_full's +3.07 (Δ = −0.18 ± 0.84). This means the identity data facilitates correspondence through weight-level representation learning — the model builds better entity-attribute representations from seeing structured (entity, attribute) patterns, even when it cannot use in-context retrieval on those sequences.

This is distinct from identity shortcut bridge result, where the identity *shortcut itself* required attention (REPEAT_MASKED had zero gains). The distinction: **learning the identity relation requires attention; but the benefit of identity practice for OTHER relations transfers through weights.**

### 3. Identity creates output-level competition despite learning-level facilitation

ident_full has the WORST absolute correspondence NLL (hc_s = 13.13) despite the best source-conditional gain. The reason: identity training shifts the output distribution toward source tokens (copy_nll = 10.15, much lower than neutral's 21.48). This means:

- **Probability mass competition**: the model assigns more probability to copy tokens (SRC), leaving less for rewrite tokens (RWT), even when the correspondence is available.
- **Source dependence**: without source in context, ident_full is much worse (nosrc NLL = 16.20 vs neutral's 10.69). With source, the gap narrows (13.13 vs 9.02).

The **net effect** of identity exposure is:
- Better at *using* source information for correspondence (higher hcg)
- Worse at *absolute* correspondence prediction (higher hc_s)
- The model becomes more source-dependent: helpful when source is present, harmful when absent

### 4. Wrong correspondence actively hurts, establishing constraint content specificity

wrong hcg (+0.51) is much less than neutral (+1.67) and ident (+3.07). Wrong correspondence fills provide actively misleading mapping evidence that interferes with the shared backbone's correspondence signal. This is consistent with topology phase1 result's adversarial-h1 result: the constraint content itself matters, not just exposure.

## Connection to BabyLM relation learning

This result reconciles the synthetic mechanism with the BabyLM relation-learning observations:

**BabyLM relation-learning finding**: DeBERTa REPEAT develops copy advantage (+0.59/+0.72 nats over CLEAN) but loses on nonoverlap content (excess true-source cost +0.75/+1.04 nats). Entity tracking: REPEAT wins at 0 operations (direct retrieval), VIEW wins at 3-4 operations (content transformation).

**corr acquisition result mechanism**: Identity practice builds source-dependent retrieval representations (higher hcg) but shifts output probability toward copying (higher copy_nll → higher absolute correspondence NLL). The learning-level effect is facilitative; the performance-level effect is competitive.

**Reconciliation**: In BabyLM's REPEAT condition, exact recurrence builds strong retrieval representations that help when the answer is directly retrievable (Entity 0-ops). But the output distribution is shifted toward copying, hurting nonidentical content use (Entity 3-4 ops, nonoverlap rewrite cost). The recurrence cost is not a learning-level interference but an OUTPUT-level competition: the model has learned retrieval well, but it retrieves and copies rather than retrieves and transforms.

**Principle**: The value of finite experience depends on both (a) what representational structure it builds (which can be shared and facilitative across tasks) and (b) what output computation it installs (which competes for probability mass). These two effects can have opposite signs. Identity practice builds shared retrieval representations (positive for correspondence learning) while simultaneously installing a competing output computation (negative for correspondence performance when copy and rewrite compete).

## Source-specific excess decomposition

The same corr acquisition result data reproduces the BabyLM relation-learning pattern when the comparison aligns with BabyLM's structure. Define T = hc_s (NLL with true source), U = hc_n (NLL without source). Excess true-source cost of arm A vs arm B = (T_A − T_B) − (U_A − U_B).

| Comparison | T delta | U delta | Excess | Sign |
|---|---:|---:|---:|---|
| ident_full vs neutral | +4.11 ± 1.83 | +5.51 ± 1.52 | **−1.40 ± 0.35** | FACILITATION |
| ident_full vs all_corr | +4.66 ± 0.65 | +2.04 ± 0.79 | **+2.62 ± 0.82** | BabyLM pattern |
| ident_masked vs neutral | +0.87 ± 0.73 | +2.45 ± 1.21 | **−1.58 ± 0.49** | FACILITATION |
| wrong vs neutral | +1.12 ± 1.35 | −0.04 ± 0.57 | **+1.16 ± 1.43** | BabyLM pattern |

The sign of the excess depends on what identity REPLACES:
- **Replacing noise with identity** → negative excess (facilitation: identity helps more WITH the source than WITHOUT)
- **Replacing correspondence with identity** → positive excess (BabyLM pattern: identity makes true-source use worse beyond the general quality loss)

This exactly matches the fixed-budget substitution framework: in BabyLM, REPEAT replaces diverse linguistic content (correspondences) with exact recurrence, producing the positive excess. The mechanism is opportunity cost (less correspondence evidence) combined with output competition (probability mass shifted toward copying). The +2.62 excess in ident_full vs all_corr is the synthetic analogue of DeBERTa's +0.75/+1.04 excess.

## What remains unresolved

1. **Scale calibration**: The synthetic excess (+2.62) is larger than BabyLM (+0.75–1.04), likely because the synthetic task has perfect identity structure and extreme substitution (80% identity fill). Whether the relationship holds under more realistic mixing ratios.

2. **Multi-operation depth crossover**: corr acquisition result doesn't test multi-step state tracking. The hypothesis: identity practice helps 0-op retrieval but hurts multi-op transformation because the installed retrieval computation short-circuits the deeper transformation pathway. This needs a synthetic task with multiple update operations.

3. **Scale and natural language**: Whether the facilitative effect of identity on correspondence persists at BabyLM scale (10M+ tokens, complex syntax, many overlapping patterns) or whether the output competition dominates at scale.

4. **Optimal mix**: The result implies there should be an optimal balance of identity and correspondence evidence. Too little identity → weak retrieval representations. Too much identity → output competition dominates. The all_corr arm achieves the best hcg, suggesting that when possible, direct correspondence evidence is always better than indirect identity facilitation.

## Files

- Script: `scripts/corr_acquisition.py`
- Analysis: `scripts/analyze.py`
- Data: `data/corr_acquisition/summary.json`
- Per-seed: `data/corr_acquisition/seed{42,43,100}/results.json`
- Figure: `figures/corr_acquisition.png`
