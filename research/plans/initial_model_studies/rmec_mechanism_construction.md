# new best verification and assessment — RMEC Mechanism Construction

## Evidence-Based Route Selection

### What we now know
1. **40k tokenizer alone doesn't fix Entity.** official40k oom and accum repair official40k on DeBERTa-v2 8×480 at chck_100M gives Entity 22.16, BELOW the 16k-protected 22.62. The leader's Entity 28.45 is NOT from the tokenizer.
2. **The leader's advantage is in data processing/organization**, not tokenizer or architecture. Both our model and the leader use DeBERTa-v2 + WWM + ~35M params. The leader's +6.25 Entity and +5.63 EWoK come from how training data is presented.
3. **AMLM-40M shows broad representations CAN form** under difficulty pressure, but they decay because WWM loss minimum doesn't require them.
4. **Entity-density selection alone failed** (wikiauto pair signal analysis): high-structure windows didn't improve Entity.
5. **Generic curriculum alone failed** (true s2 100m available coordinate): length ordering caused regressions.
6. **The gap is concentrated:** Entity -6.25, EWoK -5.63, GlobalPIQA -2.07, SuperGLUE -1.53.

### Why prior entity-targeted approaches failed
- Entity-density SELECTION without masking change: the model still uses shortcuts because uniform masking doesn't REQUIRE entity-tracking for the loss
- Curriculum without relational focus: length ordering changes difficulty but not what the loss TARGETS
- AMLM without persistence: difficulty weighting is generic (not entity-targeted) and decays

### What must be different this time
The mechanism must make the loss minimum REQUIRE entity/relational tracking AT CONVERGENCE, not transiently. This means:
1. The masking must CONTINUOUSLY target entity-relevant positions (not decay like AMLM)
2. The data ordering must present entity signal EARLY and MAINTAIN it (not dilute it in random shuffle)
3. Both must operate simultaneously so neither alone can be satisfied by narrow shortcuts

---

## Mechanism: Relational Masking with Entity-Curriculum (RMEC)

### Component 1: Relational Token Priority Masking (RTPM)

**What:** Instead of masking all word groups with uniform p=0.15, assign masking probability based on relational role:
- **Entity re-mentions** (second+ occurrence of a noun phrase within a 256-token window): p=0.35
- **Pronouns** (he/she/it/they/his/her/its/their/him/them/who/which/that-referring): p=0.30
- **State/relation verbs at entity positions** (is/was/has/had/went/moved/gave/took/put): p=0.25
- **All other word groups**: adjusted to maintain overall ~15% mask rate per example

**Why this isn't AMLM:** AMLM targets tokens the model finds HARD overall (and decays as the model improves). RTPM targets tokens whose prediction requires RELATIONAL context regardless of difficulty. An entity re-mention might be EASY (predictable from context) but still requires tracking who is being discussed. This prevents the model from ever converging to a solution that doesn't track entities.

**Implementation:** Before masking, tag each word group using simple deterministic rules:
1. Build a set of noun phrases (capitalized words + "the X" patterns) seen so far in the example
2. Tag subsequent occurrences as "entity re-mention"
3. Tag pronouns by POS (deterministic from closed list)
4. Tag state verbs by lemma (deterministic from closed list of ~50 verbs)
5. Assign masking probability by tag; normalize so total masked tokens ≈ 15% of sequence

### Component 2: Entity-Dense Curriculum (EDC)

**What:** Order training examples from MOST entity-dense to LEAST entity-dense within each epoch:
- Entity density = (count of entity re-mention tokens) / (total tokens in example)
- Epoch 1-3: highest-density examples first (strong entity signal)
- Epoch 4-7: medium density
- Epoch 8-10: lowest density (extends to harder cases)

**Why this isn't wikiauto pair signal analysis-161:** That experiment SELECTED only high-density windows and discarded the rest. EDC uses ALL official text (same 10M words, same 10 epochs) — it only changes the ORDER, not the content. The total word exposure is identical.

**Why this isn't the generic curriculum (true s2 100m available coordinate):** That ordered by sentence LENGTH. EDC orders by ENTITY DENSITY, which directly targets the deficit.

### Component 3: Existing 16k Tokenizer (NOT 40k)

Based on the official40k oom and accum repair evidence (40k hurts Entity), we use the protected 16k baseline tokenizer. The leader's 40k advantage is NOT from the tokenizer itself but from their data processing; our RMEC mechanism aims to replicate that advantage through a different means.

---

## Experiment Design

### Full-budget 100M, 2 arms on 2 GPUs

| Arm | GPU | Tokenizer | Masking | Ordering | Purpose |
|---|---|---|---|---|---|
| A | 0 | baseline16k | RTPM (relational) | EDC (entity-dense first) | FULL RMEC |
| B | 1 | baseline16k | Standard WWM p=0.15 | EDC (entity-dense first) | EDC-only control |

### Why these specific arms
- Arm A vs protected reference: tests whether RMEC improves Entity/EWoK/GlobalPIQA
- Arm A vs Arm B: isolates RTPM contribution beyond curriculum alone
- Both vs protected: both use EDC, so any shared gain is from ordering; any A-only gain is from relational masking

### Architecture and hyperparameters (identical to protected reference)
- DeBERTa-v2 8×480, 34,467,424 parameters
- Baseline16k tokenizer
- Batch 256, seq_len 256, LR 1e-3, weight_decay 0.01, warmup 0.05
- 100,000,000 word exposure (10 epochs), checkpoint every 1M words
- Seed 42, extra_init_seed 456, train_rng_seed 789

### Evaluation plan
- Primary: 7-column zero-shot/Reading at chck_60M, chck_70M, chck_80M, chck_90M, chck_100M
- Secondary: SuperGLUE at best endpoint checkpoint
- Tertiary: AoA over full trajectory
- Success criterion: Entity ≥ 25.0 AND EWoK ≥ 53.0 at best single endpoint, while Supplement ≥ 57.0 and Reading ≥ 6.0. These thresholds, combined with our existing BLiMP/COMPS/SuperGLUE levels, would put Overall above 41.80.

### Implementation requirements
1. Modify `babylm_masked_train_fullcycle.py` to accept `--mask_mode rtpm` with entity/pronoun/verb tagging
2. Add `--curriculum entity_dense` option that computes entity density per example and sorts by decreasing density within each epoch
3. The entity tagging must be DETERMINISTIC (same text → same tags → reproducible)
4. Word accounting must be IDENTICAL to protected reference (same total words, same epochs)

---

## Arithmetic check

If RMEC achieves Entity 25.0 (halfway to leader), EWoK 53.0, GlobalPIQA 37.5, while keeping BLiMP 66.5, Supplement 59.0, COMPS 53.0, Reading 7.0, SuperGLUE 68.0, AoA 0.0:
- Sum = 25+53+37.5+66.5+59+53+7+68+0 = 369.0
- Overall = 369.0/9 = 41.0

Not yet SOTA (41.80). We'd need Entity ~28+ or broader gains.

If RMEC achieves Entity 27.0, EWoK 55.0, GlobalPIQA 38.5, with BLiMP 67.0, Supplement 58.0, COMPS 53.0, Reading 6.5, SuperGLUE 68.5, AoA 0.0:
- Sum = 27+55+38.5+67+58+53+6.5+68.5+0 = 373.5
- Overall = 373.5/9 = 41.5

Close! And if early stopping at 70-80M helps (our 80M finding):
- Entity might peak earlier, similar to AMLM dynamics
- Best single endpoint might push Overall above 41.80

---

## Files
- Evidence: `data/current_best_internal_coordinate.json`
- official40k oom and accum repair 40k scores: Entity 22.16, BLiMP 66.65, Supplement 59.27, COMPS 52.32, GlobalPIQA 36.605
- Route plan: this file
- Implementation: to be written as `scripts/rmec_train.py`
