# identity shortcut bridge result identity-shortcut bridge: complete result

## v2 randomized-position result (decisive)

### Design
Same as v1 except: MODIFY_IDX randomized per sequence (forces content-based entity matching, not position shortcuts). Per-example 3D attention masks for REPEAT_MASKED. Eval probes use random context positions.

### Epoch 300 summary

| condition | copy_gain | content_gain | copy_nll | content_nll | base_has_nll | base_is_nll | mod_attr | train_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unique | +0.000 | +0.000 | 3.466 | 3.467 | 3.466 | 3.468 | 3.466 | 1.831 |
| repeat_full | **+3.386** | **+0.642** | **2.764** | **5.660** | 6.150 | 6.302 | **0.959** | 1.591 |
| repeat_masked | +0.001 | +0.001 | 3.466 | 3.468 | 3.466 | 3.469 | 3.470 | 1.700 |
| varied | **+0.609** | **+3.552** | **5.634** | **2.677** | 6.243 | 6.230 | **0.945** | 1.592 |
| wrong | −0.001 | −0.001 | 3.473 | 3.473 | 3.472 | 3.472 | 3.468 | 1.734 |

### Key findings

#### 1. Relation-specific installation
REPEAT_FULL trains the model to find identical content across context → develops massive copy_gain (+3.39) but only moderate content_gain (+0.64). VARIED trains the model to find equivalent content across templates → develops massive content_gain (+3.55) but only moderate copy_gain (+0.61). Cross-relation transfer is limited to about 17% of same-relation gain.

#### 2. Active degradation of the unpracticed relation
**This is the central result.** Compared to UNIQUE (which stays at the uniform prior 3.47 for all probe NLLs):
- **REPEAT_FULL content_nll = 5.66** vs UNIQUE content_nll = 3.47 → **+2.19 nat degradation on nonidentical content**
- **REPEAT_FULL copy_nll = 2.76** vs UNIQUE copy_nll = 3.47 → −0.71 nat improvement on copy
- **VARIED copy_nll = 5.63** vs UNIQUE copy_nll = 3.47 → +2.16 nat degradation on copy
- **VARIED content_nll = 2.68** vs UNIQUE content_nll = 3.47 → −0.79 nat improvement on content

The practiced relation improves by ~0.7-0.8 nats, but the unpracticed relation degrades by ~2.2 nats. The degradation is roughly 3× the improvement. This is a genuine active cost, not just failure to help.

#### 3. Attention access is the mechanism (REPEAT_MASKED = UNIQUE ≈ 0)
REPEAT_MASKED has identical training data to REPEAT_FULL but with attention blocked from second→first occurrence. It shows zero gains on both copy and content (within 0.001 of UNIQUE). This proves:
- Weight-based learning from repetition without attention access does NOT produce in-context retrieval
- The identity shortcut is not just a shortcut — it is the required training mechanism
- Blocking within-context attention prevents both the benefit AND the cost

#### 4. Wrong content also shows zero gains (WRONG ≈ UNIQUE)
Contradictory second occurrences produce conflicting gradient signals that prevent learning any in-context retrieval. This parallels the adversarial relation-constraint result but through a different mechanism: here the model receives correct AND wrong attributes for the same entity, preventing commitment to either.

#### 5. Baseline degradation at modification positions
For both REPEAT_FULL and VARIED, the baseline NLLs at the probe position (base_has 6.15/6.24, base_is 6.30/6.23) are far above the uniform prior (3.47). The model has learned to expect matching entities in context; when they're absent (baseline), prediction degrades. This explains why the "gains" (baseline minus probe) are large even though absolute probe NLLs are degraded: both baseline and probe are affected, but the practiced relation's probe is much better.

### Connection to BabyLM relation learning

relation_learning source trigger experiment design found:
- REPEAT minus CLEAN excess true-source cost = +0.75/+1.04 nats (DeBERTa) and +0.40 (RoBERTa) on tokenizer-nonoverlap rewrite tokens
- VIEW minus CLEAN content-conditioning advantage large for DeBERTa but small for RoBERTa

Synthetic result:
- REPEAT_FULL content_nll = 5.66 vs UNIQUE content_nll = 3.47 → **+2.19 nat active content degradation** (the synthetic analog of the BabyLM excess true-source cost)
- VARIED content_nll = 2.68 vs UNIQUE content_nll = 3.47 → −0.79 nat content improvement (the synthetic analog of VIEW's content advantage)

The mechanism is now identified: within-context attention to identical content installs a cross-span identity-matching computation. When the evaluation requires nonidentical content use, the identity-matching computation competes with content prediction and produces worse results than a model with no in-context training.

### What this establishes vs what it doesn't

**Establishes:**
- Within-context attention access is causally necessary for learning in-context retrieval (FULL vs MASKED)
- The practiced within-context relation determines which ability develops (copy vs content)
- Cross-relation transfer is limited (17% of same-relation gain)
- The practiced relation actively degrades the unpracticed relation (~2.2 nats vs UNIQUE)
- The mechanism is attention-mediated, not weight-based repetition

**Does not establish:**
- Whether the same mechanism operates in natural language / BabyLM (the synthetic task has 70 tokens, no syntax, no multi-operation tracking)
- Whether multi-operation state tracking shows the Entity depth crossover (0-ops favoring REPEAT, 3-4 ops favoring VARIED)
- Whether the cross-relation transfer rate depends on model capacity, task complexity, or training budget
- Whether the active degradation manifests as negative conditioning gain (it does as absolute NLL degradation, but the relative gain is still positive)

## v1 position-confounded pilot (archived)

v1 used fixed MODIFY_IDX=[0,3], creating position-specific copy shortcuts. The model learned slot→slot position mapping rather than content-based matching. Eval with mismatched positions showed catastrophic NLL divergence (9.6 vs 3.47). This confound was diagnosed and fixed in v2. v1 results are retained only as a demonstration that position-locked repetition creates compulsive copying.

## Files
- v2 script: `scripts/v2_randomized.py`
- v2 results: `data/v2_randomized_pilot/`
- v1 script: `scripts/identity_shortcut_bridge.py`
- v1 results: `data/identity_shortcut_pilot/`
