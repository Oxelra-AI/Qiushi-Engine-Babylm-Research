# anti source and composition plan: Anti-source diagnostic + principle-guided composition plan

## Anti-source-state diagnostic (corrects earlier analysis operation-content interpretation)

The analysis identified that gold_item_hits is always 0 on rel_eq0 (queried box untouched),
so "no negative cases" is structural. The decisive test: **over half of flips have ZERO operation-mentioned
items in the predicted option**, confirming the primary failure is NOT operation-content pull but an
**anti-source policy**: "operations present → answer is not the source state."

alpha0.75 flips (n=292):
- 164 flips (56.2%) with ZERO operation-item hits in the predicted option
- 128 flips (43.8%) with nonzero operation-item hits

Zero-content flips by irrelevant_ops: 42/80/24/18 for bins 0/1-3/4-6/7+

Full rel_eq0 accuracy by bin (alpha0.75):
- irrelevant_ops 0: base 35.2% → 29.9% (Δ -5.3%, n=304)
- irrelevant_ops 1-3: base 39.0% → 31.8% (Δ -7.2%, n=641)  
- irrelevant_ops 4-6: base 43.0% → 21.7% (Δ -21.4%, n=323)
- irrelevant_ops 7+: base 29.7% → 14.3% (Δ -15.4%, n=273)

Mechanism: The binding training family ALWAYS has operations present, and B/update rows
(where the answer is the new state) are easier to learn than A/unchanged rows. The optimizer
establishes "operations present → not source state" as a cheap prior. On rel_eq0 (where gold
IS the source state), this prior systematically displaces correct answers, especially when many
operations are present. Operation-content pull is a real but SECONDARY landing attractor.

## Principle-guided composition plan

Instead of designing another binding objective (4th attempt), use the measured relation→column
map to COMPOSE the private phase from different relation types. The data IS the principle.

### Measured relation→column map

From the combined study evidence:
- **Correct correspondence** (ALN, dose): shifts Entity/Supplement/EWoK up, BLiMP/Reading slightly down
- **Wrong correspondence** (SHUF): shifts BLiMP/Reading up, Entity/Supplement/EWoK down
- **Coherent replay**: preserves all columns near base
- **Context isolation** evidence: ALN +0.35 context gain, SHUF ~0 context gain, SEP -0.33

### Overall(AoA0) arithmetic

Overall = sum(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, 0) / 9
Each column = 1/9 = 11.1% of Overall

Notation correction: the historical shorthand wrote `mean(...)/9`; the expression
above uses the intended arithmetic mean, dividing the sum by 9 only once. This
corrects the description, not any recorded model score.

Columns grouped by relation benefit:
- Context-supported (Entity + Supplement + EWoK) → aligned restatement = 3/9 of Overall
- Isolation-supported (BLiMP + Reading) → isolated/coherent practice = 2/9 of Overall
- Broad (COMPS + GlobalPIQA + SuperGLUE + AoA) → coherent replay = 4/9 of Overall

### Composition Arms

**Arm A (primary)**: Coherent replay + aligned restatement pairs (dose25 fraction)
- ~3,150K ordinary text words (coherent replay)
- ~843K aligned pair words (dose25 probe-clean)
- Total: ~3,993K words (matching coherent86)
- Objective: standard CE + KL on ALL rows (no special answer targeting)
- The pairs practice correct-correspondence relation through standard masked prediction

**Arm B (hypothesis)**: Coherent replay + aligned pairs + isolated sentences
- ~1.8M coherent replay + ~0.84M aligned pairs + ~1.35M spanbreak/isolated
- Tests whether BLiMP/Reading benefit can be added without wrong-correspondence cost
- earlier analysis already has spanbreak_replay mode

### Word accounting
- chck_82M: 82,012,495 initial words
- coherent86 private: 3,992,800 charged words  
- Total: 86,005,295
- composition private: ~3,993,000 charged words (matched)
- Total: ~86,005,495

### Available data
- Coherent replay stream: cleanqwen_fineweb_compact_view_reinvest_100M.jsonl (skip_rows=530,944)
- Aligned pair rows: dose25 superset (6,312 packed rows, 843,200 pair words) — probe-clean, zero eval overlap
- Pair row format: `text_pairs_only` key → rename to `text`
