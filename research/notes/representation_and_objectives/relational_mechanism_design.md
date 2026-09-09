# corpus directional pair census — Three Distinct Mechanisms for Context-Conditioned Relation Learning

## Diagnostic Foundation

### What fails
The DeBERTa-v2 lineage has a stable **context-conditioned world-relation weakness** that persists across tokenizers (inherited 16k, legal 16k, legal 40k, shared 16k), architectures (8×480, 12×384), and data views (compact, row-block breadth, interleaved breadth).

**EWoK conditional reversals**: ~67-69% of wrong rows are "stable failures" where the model consistently prefers the wrong context-target combination. The four-cell interaction is deeply negative (compact wrong-row median interaction -0.956; row-block -0.505). Worst context-diff types: "variable swap" 81.3% stable failure rate among wrong, "antonym" 62.2%, "physical-relations" domain 78.9%, "spatial-relations" 80.6%. Only ~3% of wrong rows have the RIGHT interaction direction.

**GlobalPIQA parallel deep-rank**: On 52 hard rows (wrong for all endpoints), correct answers sit at rank 3-4 among 4 choices with mean top-minus-correct 1.4-1.7 nats. Only 8/52 hard rows have margin ≤0.50 nats. Three-arm oracle union 40.78% on parallel but 61/103 rows wrong for all three FW arms.

### Why standard MLM does not fix this
Standard MLM trains P(token|visible_context) independently per masked position. The model minimizes loss by predicting the locally most likely token, which is dominated by lexical co-occurrence and topic statistics. There is **no explicit training signal** that says "when context word X changes to X', the prediction should change from A to B." The conditional-reversal structure exists implicitly in the corpus (sentences about ice melting vs. water freezing both appear) but the learning objective never forces the model to connect them.

### Why prior relation-data routes failed
INITIAL_MODEL_STUDIES earlier analysis/345: relation-focused material moved EWoK (+0.55 to +3.09) but consistently hurt GlobalPIQA (-1.95 to -3.89 parallel). The binding mechanism probes showed 0.000 both-correct: the model never learned the intended conditional structure, only absorbed relational topic statistics that biased predictions. Simply adding more relational text repeats this failure because the learning signal (MLM loss) is unchanged.

### What must change
The route forward requires changing the **learning signal**, not just the data distribution. The new signal must:
1. Explicitly reward predictions that change when context changes (directly targets conditional reversal)
2. OR create a separated representational pathway that specializes in relational processing without interfering with entity/lexical learning
3. OR restructure masking to force the model to use relational context tokens for predictions, combined with a schedule that protects broad capability

## Three Mechanisms

### Mechanism A: Interaction-Contrast MLM (IC-MLM)

**Core idea**: Add an auxiliary loss that explicitly encourages context-sensitive predictions by penalizing the model when removing a context word does not change the prediction for nearby masked tokens.

**Learning signal**:
1. Standard forward pass with WWM mask → logits L1, labels → standard MLM loss L_mlm
2. For a fraction f_aux (e.g., 0.20) of training steps:
   a. Select k random non-masked, non-special content tokens per sequence ("context probes")
   b. Create a perturbed input: replace each context probe with [MASK]
   c. Second forward pass → logits L2
   d. For each originally-masked position within distance d_window of a context probe:
      Compute KL(softmax(L1[pos]) || softmax(L2[pos]))
   e. Auxiliary loss = λ · mean(max(0, margin_min − KL_per_position))
      This penalizes low sensitivity: if removing a content word doesn't change the target prediction, the loss increases.

**Parameters**: f_aux = 0.20, k = 3 probes per sequence, d_window = 32 tokens, margin_min = 0.10 nats, λ = 0.50

**Why this targets the failure**: The model currently assigns similar P(target|context) regardless of which context words are present, leading to conditional-reversal failures. IC-MLM explicitly trains the model to make predictions that depend on visible context. On sentences with relational structure, this forces the model to use relational pivot words when predicting consequence words.

**Compute overhead**: ~1.20x (second forward pass for 20% of steps, no second backward pass — only the gradient through the auxiliary loss on the standard parameters)

**Expected signal**: EWoK wrong-row interaction sum should become less negative (median moves toward 0 from -0.96); GlobalPIQA hard52 mean top-minus-correct should decrease (from 1.7 toward 1.0) because the model better uses physical-consequence context words. Risk: if λ is too high, broad MLM accuracy drops, damaging BLiMP/Supplement/Entity.

**Pass criterion for minimum experiment**: EWoK stable-failure fraction decreases by ≥2pp AND GlobalPIQA hard52 mean margin decreases by ≥0.15 nats, with Entity change ≤ -1.5 points.

**Fail indicators**: Entity drops >3 points (interference not controlled), or EWoK improves but GlobalPIQA doesn't change (signal too weak/noisy), or loss diverges (λ too high).

### Mechanism B: Relational Bottleneck Adapter (RBA)

**Core idea**: Add small, zero-initialized adapter layers to the model that specialize in relational processing while protecting the main parameters through differential learning rates. The adapter naturally captures whatever the main model's reduced learning rate leaves underfit.

**Architecture change**:
- After the attention+FFN block in layers 4, 5, 6, 7, add:
  `x → x + α · FFN_adapter(LayerNorm(x))`
- FFN_adapter: Linear(480 → d_rel) → GELU → Linear(d_rel → 480)
- d_rel = 32 (bottleneck dimension)
- α = learnable scalar per layer, initialized to 0.0 (adapter starts transparent)
- Total added parameters: 4 layers × (480×32 + 32 + 32×480 + 480 + 1) ≈ 125,572

**Training modification**:
- Main model parameters: LR × 0.60
- Adapter parameters (including α): LR × 2.0
- Same MLM loss, single forward pass
- No new objective needed — the differential LR naturally causes the adapter to specialize

**Why this targets the failure**: The INITIAL_MODEL_STUDIES and fw weight space sweep evidence shows that relational learning and entity/lexical learning interfere when sharing the same parameters. By creating a dedicated adapter pathway with higher LR, relational patterns can be captured without overwriting entity knowledge in the main parameters. The adapter's small size (125k vs 34.5M main) prevents it from memorizing noise.

**Compute overhead**: ~1.03x (additional small FFN computation per layer)

**Expected signal**: Entity and Supplement should be preserved (main parameters move slowly). EWoK should improve if relational patterns are learnable but currently suppressed by the entity-protecting equilibrium. GlobalPIQA improvement depends on whether the adapter can capture the conditional structure from standard MLM loss alone.

**Pass criterion**: Entity change ≤ -0.5 points AND (EWoK stable-failure fraction decreases by ≥1pp OR GlobalPIQA hard52 mean margin decreases by ≥0.10 nats).

**Fail indicators**: Entity drops despite protected LR (adapter α grows too fast, interfering with main residual), or no improvement in either relational metric (the signal isn't in the standard MLM loss, so the adapter has nothing to specialize on — this would confirm that a changed learning signal, not just capacity, is needed).

### Mechanism C: Phase-Gated Directed Consolidation (PGDC)

**Core idea**: A two-phase training schedule that shifts masking emphasis from broad language learning to relational consolidation while reducing the main learning rate. This is a curriculum over the training objective, not the data.

**Phase design (starting from compact chck_70M)**:
- **Phase 2a (70M → 85M)**: Mixed masking
  - Offline: tag tokens in relational positions using spaCy POS tags + pattern rules
    (relational = VERB tokens in causal/spatial/temporal patterns, ADP, ADJ in comparative/superlative)
  - Masking: relational tokens masked at p_rel = 0.25, non-relational at p_std = 0.12
  - Loss weight: relational positions × 1.5, non-relational × 0.8
  - LR: 0.70× Phase 1 (= 0.0007)
  
- **Phase 2b (85M → 100M)**: Strong consolidation
  - Masking: relational tokens at p_rel = 0.30, non-relational at p_std = 0.08
  - Loss weight: relational positions × 2.0, non-relational × 0.5
  - LR: 0.30× Phase 1 (= 0.0003)

**Why this targets the failure**: By masking relational tokens at higher rates, the model is forced to reconstruct relational structure more frequently. The higher loss weight ensures gradient signal is concentrated on relational predictions. The reduced LR protects already-learned broad knowledge. Late-phase emphasis leverages the model's existing language competence to build relational understanding more efficiently.

**Compute overhead**: ~1.0x (only change is masking distribution and loss weighting; offline tagging is one-time preprocessing)

**Expected signal**: EWoK should improve (model sees more relational masking targets). Risk of repeating the EWoK+/PIQA− pattern from INITIAL_MODEL_STUDIES is HIGHER than for IC-MLM and RBA because this mechanism changes WHAT is masked rather than HOW the loss works. The key difference from INITIAL_MODEL_STUDIES: here we upweight relational tokens within the SAME data rather than adding separate relational data, and we protect broad knowledge via LR reduction. Whether this is enough to break the pattern is the central test.

**Pass criterion**: EWoK stable-failure fraction decreases by ≥1pp AND GlobalPIQA hard52 mean margin decreases by ≥0.08 nats AND Entity change ≤ -2.0 points.

**Fail indicators**: EWoK improves but GlobalPIQA worsens (INITIAL_MODEL_STUDIES pattern repeated — confirms that masking redistribution alone doesn't fix the signal problem), or Entity drops >3 points (LR protection insufficient).

## Comparative Analysis

| Criterion | IC-MLM | RBA | PGDC |
|---|---|---|---|
| Directly targets conditional reversal | YES (contrastive loss) | No (capacity only) | Partially (masking bias) |
| Addresses interference | Partially (λ tuning) | Directly (separate params) | Partially (LR schedule) |
| Prior-failure-mode risk | LOW | MEDIUM | HIGH (most similar to INITIAL_MODEL_STUDIES) |
| Implementation complexity | HIGH (two-pass, new loss) | MEDIUM (adapter layers) | LOW (masking/weights) |
| Compute overhead | ~1.20x | ~1.03x | ~1.0x |
| Novel learning signal | YES | NO (same MLM) | NO (same MLM, different weighting) |
| Entity preservation | Depends on λ | Good (frozen main) | Depends on LR decay |

**Key distinction**: IC-MLM is the only mechanism that introduces a genuinely new learning signal (context-sensitivity penalty). RBA and PGDC both use standard MLM loss with modifications to capacity or weighting. If the root cause is "missing signal" (model never sees contrastive context), only IC-MLM directly addresses it. If the root cause is "interference" (model can learn relations but overwrites them with entity learning), RBA is best. If it's "underweighting" (relational tokens get too little gradient), PGDC is best.

The prior evidence points toward "missing signal" as the dominant cause (INITIAL_MODEL_STUDIES showed relation data moved EWoK without learning conditional structure; binding probes were always 0.000). This favors IC-MLM.

## Minimum-Cost Experimental Design

### Shared infrastructure
- **Starting checkpoint**: compact chck_70M (`experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_70M`)
- **Continuation**: 30M more words (70M → 100M exposure) using the compact-view JSONL stream, selecting words 70,000,001–100,000,000
- **Shared tokenizer**: legal 16k SHA `e70d167f...`
- **Architecture**: DeBERTa-v2 8×480 (except RBA adds adapters)
- **Control**: compact 70M→100M trajectory (already exists as chck_70M to chck_100M; difference in EWoK/GlobalPIQA readouts gives the null effect)
- **Evaluation**: EWoK four-cell (7,618 rows), GlobalPIQA all-option margins (103 parallel + 100 nonparallel), cheap7 surface
- **Hardware**: One H100, ~25-30 min per 30M arm

### Implementation priority
1. **IC-MLM** (highest theoretical alignment with failure mode; if it fails, confirms the needed signal is even harder to extract)
2. **RBA** (if IC-MLM succeeds but needs Entity protection; or if IC-MLM fails, tests whether capacity separation alone helps)
3. **PGDC** (cheapest to implement; most likely to repeat INITIAL_MODEL_STUDIES pattern; useful as negative control to confirm signal > weighting)

### Continuation trainer requirements
A new warm-start trainer is needed because the COMPACT_EXPERIENCE `masking_curriculum_trainer.py` does not support resume. The continuation trainer must:
1. Load model from checkpoint path (DebertaV2ForMaskedLM.from_pretrained)
2. Load the same compact-view JSONL stream, skip first 70M words
3. Create optimizer with new LR (for RBA: parameter groups with differential LR)
4. Apply the mechanism-specific masking/loss modification
5. Train for exactly 30M words with checkpoints at 80M, 90M, 100M
6. Save identical HF checkpoint format for existing evaluation scripts

### Decision tree after results
- IC-MLM improves BOTH EWoK and GlobalPIQA without Entity loss → scale test from 0→100M with IC-MLM
- IC-MLM improves EWoK only (not GlobalPIQA) → add RBA on top of IC-MLM
- IC-MLM improves neither → the conditional signal is too dilute with random context probes; try targeted probes using relational tokens
- RBA improves Entity preservation but not relational metrics → confirms signal problem, not capacity problem
- PGDC repeats EWoK+/PIQA− → confirms INITIAL_MODEL_STUDIES pattern, closes masking-redistribution alone

## Offline Preprocessing: Relational Token Tagger

Both IC-MLM (for context probe selection) and PGDC (for masking distribution) benefit from knowing which tokens are in "relational positions." This can be done offline:

1. Run spaCy `en_core_web_sm` on the compact-view JSONL
2. Tag each whitespace word with relational indicators:
   - `verb_causal`: VERB with causal/change semantics (lemma in {cause, make, let, help, force, prevent, lead, result, create, destroy, melt, freeze, heat, cool, break, fix, open, close, push, pull, lift, drop, fall, rise, grow, shrink, ...})
   - `verb_motion`: VERB with spatial/temporal motion (go, come, move, run, walk, fly, turn, enter, leave, arrive, ...)
   - `prep_spatial`: ADP with spatial meaning (on, in, under, above, below, behind, beside, between, ...)
   - `prep_temporal`: ADP/SCONJ with temporal meaning (before, after, during, when, while, until, ...)
   - `adj_comparative`: ADJ in comparative/superlative form
   - `consequence_marker`: SCONJ/ADV marking consequence (because, so, therefore, if, unless, although, ...)
3. Store as parallel arrays in the JSONL or as a separate tag file
4. Count: what fraction of corpus tokens are relational? (Expected: 15-25%)

This is a one-time CPU cost (~5-10 min for 10M words).

## Complete Three-Arm EWoK Four-Cell Summary

| metric | compact | row-block | interleaved |
|---|---|---|---|
| accuracy | 0.5043 | 0.4957 | 0.5112 |
| wrong_rows | 3776 | 3842 | 3724 |
| stable_failure_frac_wrong | 0.6891 | 0.6458 | 0.6821 |
| interaction_sum_wrong mean | -3.651 | -3.267 | -3.378 |
| interaction_sum_wrong median | -0.956 | -0.505 | -0.666 |
| within_both_positive_wrong_frac | 0.031 | 0.038 | 0.024 |

Row-block breadth has the strongest conditional-reversal improvement (lowest stable failure rate 0.646, least negative median -0.505) but at lower overall accuracy. Interleaved is intermediate on conditional metrics but has the highest accuracy. No arm substantially fixes the conditional-reversal failure: even row-block still has 64.6% stable failure among wrong rows. This completes the FW family closure and confirms that data allocation changes can weakly touch the conditional-reversal structure but cannot overcome the fundamental MLM signal limitation.

## Files
- This note: `research/notes/representation_and_objectives/relational_mechanism_design.md`
- Evidence sources:
  - EWoK four-cell: `data/fw_ewok_interaction_reader/fw_ewok_interaction_reader_summary.json`
  - GlobalPIQA hard52: `data/globalpiqa_margin_synthesis/globalpiqa_margin_synthesis.json`
  - INITIAL_MODEL_STUDIES meta-synthesis: `notes/existing_smallroute_meta_synthesis.md`
  - FW closure: `notes/weight_space_and_ewok_synthesis.md`
  - full ewok interaction synthesis EWoK interaction: `data/ewok_interaction_synthesis/ewok_interaction_synthesis.json`
