# relation practice principle decisive results: locality confirmed and mechanism specified

## Summary of two simultaneous results

### 1. REPEAT_SPLIT eliminates both copy benefit and recurrence cost

The split control removes same-window source/companion co-occurrence while preserving text exposure and 100M budget. REPEAT_SPLIT (D_RS_43022) vs CLEAN (D_C_43022), late 80M/90M/100M:

| quantity | original R−C range (3 seeds) | REPEAT_SPLIT−CLEAN | prestatement reading |
|---|---:|---:|---|
| token-nonoverlap rewrite gain | [−1.043, −0.752] | **−0.031** | near zero → in-window practice necessary |
| held-out natural copy gain | [+0.500, +0.668] | **−0.201** | below zero → in-window practice necessary |
| ALL rewrite gain | — | −0.431 | — |
| token-overlap rewrite gain | — | −0.777 | — |

**The recurrence cost** (token-nonoverlap rewrite gain) drops from a magnitude of 0.75–1.04 to 0.03. This is well within the ±0.25 near-zero band. Removing in-window co-occurrence nearly eliminates the active cost.

**The raw copy-gain contrast** drops from the original +0.50–0.67 to −0.20, but this raw nat difference must be read with the arm's own unrepeated-control NLL. compact mixture model and predictions normalization shows RS has lower control NLL than CLEAN, leaving less room for a raw gain; normalized RS−C is only −0.0307, while normalized RS−R is −0.0804. The strong conclusion is not that split repetition destroys copy ability. It is that the large original REPEAT copy advantage is an in-window relation-practice effect, whereas split-row exact-token exposure mainly improves ordinary control fit and does not install a large local copy gain.

**This is Branch A for the recurrence-cost mechanism and the original copy advantage**: same-window relation practice is necessary for the source-specific T/U sign reversal and for the large copy advantage; spaced repetition of the same tokens can preserve or improve ordinary content fit without the local shortcut.

### 2. Source-specificity confirms recognition-triggered in-window copy mechanism

The 2×2 design measures content-token mass under true-source (T) and unrelated-source (U) window conditions. R−C true_src_content_mass delta across three DeBERTa seeds:

| seed | T (true source in window) | U (unrelated source in window) | T/U ratio |
|---:|---:|---:|---:|
| 43022 | **+0.0727** | +0.0054 | 13.5× |
| 43122 | **+0.0972** | +0.0083 | 11.7× |
| 43222 | **+0.0550** | +0.0050 | 11.0× |

R−C true_src_content_mass is elevated by +0.055 to +0.097 when the true source is in-window, but only +0.005 to +0.008 when an unrelated source occupies the window. **The elevation is 11–14× larger under T than U.**

This rules out frequency hedging: the mechanism is specifically triggered by the recognized true source in-window.

R−C unrel_src_content_mass delta under both conditions is near zero (−0.004 to +0.003), confirming the copy mechanism does not fire indiscriminately on whatever text is in the window.

Target probability confirms the sign reversal mechanism:

| seed | R−C target_prob under T | R−C target_prob under U |
|---:|---:|---:|
| 43022 | **−0.0069** | +0.0000 |
| 43122 | **−0.0068** | −0.0001 |
| 43222 | **−0.0106** | −0.0007 |

REPEAT suppresses the correct nonoverlap target only when the true source is present. Under the unrelated-source condition, REPEAT and CLEAN perform identically.

### 3. VIEW's mechanism is content-conditioned reading, not identity misfire

V−C true_src_content_mass delta is also elevated under T (+0.060 to +0.071) but near zero under U (+0.006 to +0.008). However, VIEW SIMULTANEOUSLY improves target probability:

| seed | V−C target_prob under T | V−C target_prob under U |
|---:|---:|---:|
| 43022 | **+0.0574** | +0.0106 |
| 43122 | **+0.0628** | +0.0124 |
| 43222 | **+0.0537** | +0.0103 |

Both REPEAT and VIEW route attention through source content tokens when the true source is present. The critical difference:
- **REPEAT**: source mass elevated → target probability suppressed = identity misfire
- **VIEW**: source mass elevated → target probability improved = content-conditioned reading

This mechanistic dissociation explains the T/U sign reversal at the output level and is consistent with the installed-computation hypothesis: REPEAT practices identity between source and companion, so at test time it routes mass toward source tokens and AWAY from nonidentical targets; VIEW practices content correspondence, so it routes THROUGH source content tokens to REACH the correct nonidentical target.

## Integrated scientific conclusion

1. **Locality is established for the source-specific recurrence cost and for the large original copy advantage**: the T/U sign reversal and original copy gain require in-window co-occurrence. Duplicated-token budget without local relation practice instead gives generic improvements on control/ordinary NLL, so raw copy-gain contrasts must be normalized before interpretation.

2. **The mechanism is recognition-triggered**: the copy mechanism fires specifically when the model recognizes related source content in-window, with an 11–14× specificity ratio over unrelated sources.

3. **The mechanism operates through competing output computations**: identity practice installs a source-token routing that competes with correct nonidentical target prediction; varied restatement installs a content-conditioned routing that uses source tokens to support correct target prediction.

4. **The Relation-Practice Principle (Branch A) is supported**: Under a fixed experience budget, the relation between spans inside a training window determines which cross-span computation is installed. Exact identity practices copy routing that misfires on nonidentical targets. Partial-overlap restatement practices content-conditioned routing that serves both copy and nonidentical use.

## Remaining work for the principle

1. **VIEW_SPLIT** (still training at this stage): Does VIEW's content-conditioning benefit also require in-window co-occurrence? If yes, both sides of the principle are local. If no, the positive side operates through cross-row paraphrase exposure and only the cost side is window-local.

2. **Natural variation-set probe** (research, in progress): Tests whether the competence pattern transfers from synthetic compact rewrites to natural caregiver speech.

3. **Architecture and objective generality**: RoBERTa and causal-LM tests for the locality and mechanism results.

## Data files

- REPEAT_SPLIT probes: `data/split_repeat_split_probes/`
- Source-specificity 2×2: `data/source_specificity_misfire/`
- Principle note: `notes/relation_practice_principle.md`
- Split prestatement: `notes/split_control_numeric_prestatement.md`


## Addendum: REPEAT_SPLIT T/U decomposition eliminates the sign reversal

The most informative result from the REPEAT_SPLIT scoring is the term decomposition on token-nonoverlap rewrite targets:

| arm | T NLL (true src) | U NLL (unrel src) | gain (U−T) |
|---|---:|---:|---:|
| CLEAN (C_43022) | 7.0073 | 8.2192 | 1.2119 |
| REPEAT_SPLIT (RS_43022) | 6.4604 | 7.6410 | 1.1806 |
| RS−C delta | **−0.5469** | **−0.5782** | **−0.0313** |
| Original REPEAT−C (seed43022) | +0.4480 | −0.3037 | −0.7517 |

The original REPEAT showed the isolating sign reversal: T worse (+0.45), U better (−0.30). This reversal is **completely eliminated** in REPEAT_SPLIT: both T and U improve by approximately −0.55 nats.

REPEAT_SPLIT is actually a better general language model than CLEAN on these targets (both T and U lower), likely because the source paragraphs placed in separate rows are higher-quality training content than CLEAN's filler. But the source-specific asymmetry — the +0.45 true-source cost that defines the active recurrence mechanism — requires same-window co-occurrence and vanishes when source and companion are placed in different rows.

**This is the clearest locality evidence**: the entire T/U sign reversal that defines the active recurrence cost is an in-window phenomenon. Cross-row token exposure produces a generic LM quality change; only in-window relation practice produces the source-specific copy/misfire computation.
