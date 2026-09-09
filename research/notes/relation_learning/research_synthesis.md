# research synthesis research synthesis: principle state after third seed and synthetic bridge

## The principle

Finite experience under a fixed budget does not merely add examples — it practices
relations between spans. The relation structure within each training context determines
which cross-span computation is installed: exact in-window recurrence practices an
identity-matching computation, while content-preserving nonidentical restatement
practices a content-conditioned use of earlier spans.

The identity matcher can actively degrade nonidentical content use below a
no-companion baseline. This active cost is the strongest architecture-spanning
result and now has a synthetic causal proof.

## Three evidence pillars

### 1. Active cost of exact recurrence (pair level relation robustness center)

On held-out compact rewrite pairs with tokenizer-nonoverlap targets:
- DeBERTa seed43022: REPEAT−CLEAN gain Δ = −0.752, T excess = +0.448, U excess = −0.304
- DeBERTa seed43122: REPEAT−CLEAN gain Δ = −1.043, T excess = +0.693, U excess = −0.350
- RoBERTa seed43022: REPEAT−CLEAN gain Δ = −0.398, T excess = +0.492, U excess = +0.094

DeBERTa shows content-specific competition: REPEAT is *better* than CLEAN with an
unrelated source but *worse* with the true source. RoBERTa shows partial baseline
degradation but the true-source penalty is 5× the unrelated-source penalty.

Pair-level: positive in 66–70% of pairs. Strict word-level filter: cost persists
(+0.588, +0.830, +0.252).

**Seed43222 probes completed: all directions replicate.**
Three-seed V−R nonoverlap rewrite gain: +1.44, +1.94, +1.69 (mean 1.687, sd 0.251).
Three-seed R−V held-out copy gain: +0.17, +0.36, +0.24 (mean 0.258, sd 0.094).
Seed43222 T/U decomposition: V−R gain = +1.69, T delta = −1.78, U delta = −0.09.
The entire content-conditioning gap is content-specific (unrelated-source matched).
Entity cue ablation directions also replicate at seed43222.
R−C active cost awaits CLEAN seed43222 (still training, ~78M words).

### 2. Synthetic causal proof (functional_learning earlier analysis)

A 161K-param causal-LM Transformer on randomized-position event sequences:
- REPEAT_FULL: copy_gain +3.39 nats, content_nll +2.19 above UNIQUE
- VARIED: content_gain +3.55 nats, copy_nll +2.16 above UNIQUE
- REPEAT_MASKED (IDENTICAL data, attention blocked): copy_gain 0.001, content degradation 0.001
- WRONG: zero gains

The attention mask is the causal variable. Without in-context access to the identical
first occurrence, the identity matcher cannot form and neither benefit nor cost appears.
This proves the active cost is an attention-mediated installed computation, not a
weight-level artifact. The +2.19 nat synthetic content degradation maps to our
+0.45–0.69 nat natural-language excess true-source cost.

### 3. Entity depth crossover (three seeds)

| group | prior V−R (2-seed) | seed43222 V−R | direction match |
|---|---:|---:|---|
| rel_updates_0 | −9.42 | −2.68 | ✓ |
| rel_updates_1 | +2.16 | +0.83 | ✓ |
| rel_updates_2 | +3.19 | +2.47 | ✓ |
| rel_updates_3 | +7.54 | +2.14 | ✓ |
| rel_updates_4 | +8.74 | +2.04 | ✓ |
| rel_updates_5 | +7.41 | +2.38 | ✓ |

Direction replicates at all depths (three seeds). Magnitude attenuated 3–4×; all
pre-registered bands miss. The compression is asymmetric: REPEAT lost its zero-update
dominance while rising at deep updates. Slope: +0.817 points/update vs +3.092 prior.

Without CLEAN at seed43222, we cannot determine whether both arms dropped or the
gap narrowed. CLEAN is still training.

## Positioning against Chan et al. 2022 (\cite{chan2022data})

Chan et al. showed that distributional properties of experience — burstiness, class
rarity, dynamic meaning, within-class variation — determine whether transformers develop
in-context versus in-weights computation. Burstiness (items in clusters) promotes ICL;
within-class variation promotes ICL; Zipfian frequency enables both ICL and IWL.

Our result is the natural-language, fixed-budget counterpart:
- **Exact recurrence is extreme burstiness** — identical spans in the same context
  window. It installs identity-matching, which is a form of in-context computation.
- **Varied restatement provides within-class variation** — same content, different
  surface form. It installs content-conditioned use, a richer in-context computation.
- **We additionally show the active cost**: practiced exact recurrence does not just
  promote identity matching — it actively degrades content-conditioned prediction of
  nonidentical targets below a no-companion baseline. This goes beyond Chan et al.'s
  tradeoff (ICL vs IWL) to an installed competing computation.

Key differences:
1. Chan et al. use synthetic Omniglot classification; we use natural language MLM
2. Chan et al. measure ICL/IWL as evaluation conditions; we measure specific held-out
   behavioral tendencies (copy gain, content conditioning, state discrimination)
3. Chan et al. show a tradeoff; we show an active cost (below baseline, not just
   below the alternative)
4. The synthetic bridge provides the causal variable (attention mask) that
   Chan et al.'s design does not separate

The combined framing: experience structure determines not just whether in-context
computation emerges, but which specific cross-span relation it encodes, and a relation
practiced under one structure can actively impair performance on a different relation
structure at evaluation time.

## Immediate open work

1. **Seed43222 held-out probes** (running): Do V-R and V-C/R-C copy/rewrite gains
   replicate at the third DeBERTa seed? This is the direct test of the pair level relation robustness center.

2. **Natural re-mention probe** (258 records ready): Score on established DeBERTa arms
   first. Tests the principle outside Entity and outside compact rewrites, on register-
   free natural text with distance-matched verbatim/nonidentical pairs.

3. **CLEAN seed43222** (training): Needed for the full three-arm comparison and to
   determine whether the Entity crossover attenuation is from the CLEAN baseline
   moving or from V-R compression.

4. **Dose-1.82x probes**: earlier analysis VIEW+REPEAT exist at dose-1.82x without CLEAN.
   V-R comparison would test dose response of the cost.

5. **Multi-operation synthetic bridge**: Tests whether the Entity depth crossover emerges
   from the same attention mechanism in the synthetic task.
