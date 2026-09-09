# cross data binding comparison result: Cross-Data Entity-Event-State Binding Comparison

## Decisive Result: Binding Weakness Is Generic to DeBERTa at This Scale, Not Compact-Specific

### Finding

Entity-event-state binding accuracy (EEBF composite) varies by only **±0.008** across all four stock DeBERTa data treatments at 80M exposure:

| Treatment | Affected | Unaffected | EEBF | Aff margin |
|-----------|----------|------------|------|------------|
| clean_qwen | 0.2656 | 0.7500 | 0.5078 | -1.2657 |
| compact_reinvest | 0.3958 | 0.6250 | 0.5104 | -0.3459 |
| extractive_balanced | 0.4062 | 0.6250 | 0.5156 | -0.6506 |
| extractive_wide | 0.3385 | 0.6927 | 0.5156 | -0.8182 |

At 100M the pattern persists: EEBF 0.5234 (compact), 0.5312 (ext_balanced), 0.5104 (ext_wide).

### Interpretation

1. **No compact-specific binding advantage.** EEBF deltas are negligible: compact-minus-clean = +0.0026, compact-minus-ext_balanced = -0.0052, compact-minus-ext_wide = -0.0052.

2. **Prediction redistribution, not binding improvement.** Compact models show higher affected accuracy (+0.13 vs clean) but compensate with lower unaffected accuracy (-0.125 vs clean). The net binding composite is unchanged. Compact training shifts the answer distribution toward "changed" without genuinely improving entity-state tracking.

3. **Extractive-balanced matches compact on affected accuracy** (0.4062 vs 0.3958 at 80M). This further weakens any claim that compact views specifically strengthen binding.

4. **Adapter models shift toward "unchanged" predictions.** Scale1.75 adapters show lower affected accuracy (0.3177/0.2865/0.2969 for 82M/84M/100M) but higher unaffected accuracy (0.7552/0.7708/0.7813), giving slightly higher EEBF (0.5365/0.5286/0.5391). The adapters don't improve binding—they shift predictions toward "unchanged."

5. **Affected margin is most informative.** Compact has the least negative affected margin (-0.346 vs clean -1.266). Compact models are "less wrong" on affected entities, but this doesn't translate to binary accuracy gains because unaffected predictions weaken proportionally.

### Scientific Consequence

- **Binding and compact views are orthogonal.** binding/memory architecture work addresses a generic DeBERTa weakness, not one caused by or ameliorated by compact views. The two mechanisms could be complementary if both work.
- **Compact views help through a different mechanism** (proven downstream on official-compatible stable families), not through improved entity tracking per se.
- **No layer×position hidden-state probe is needed** to determine compact-specificity of binding—output behavior already shows there isn't one.

### Data and Scripts

- Script: `experiments/archive/frontier_consolidation/scripts/cross_data_binding_comparison.py`
- Results: `experiments/archive/frontier_consolidation/data/cross_data_binding_comparison`
- Individual model results: `..._binding.json` files in same directory
- Eval substrate: counterbalanced corpus, 1,088 items, 192 affected + 192 unaffected in held_recomb

### What This Means for the Research Route

The scientific question is whether compact training specifically strengthens result-state information or entity-query binding. Such an effect would connect the data mechanism with the architecture route; without it, linear decodability remains a secondary observation rather than an explanation of structured semantic compression.

**There is no compact-specific binding advantage in this comparison.**
1. Binding remains a separate scientific question; the tested memory route was also closed.
2. Structured semantic compression remains the surviving data mechanism.
3. The mechanisms could be additive rather than unified: a successful binding repair could improve all data treatments.

## Updated Research State After cross data binding comparison result

The memory route was closed after its fourth failure, and this binding comparison found no compact-specific effect. The remaining scientific alternatives are:

1. **Score improvement**: Current best local endpoints are coherent86 α=0.75 (42.12 projected) and chck84 (42.02 projected). The original 41.8-42.0 target has been exceeded. Pushing higher requires a genuinely new intervention, not more of the same.

2. **Structured semantic compression**: Established as a real DeBERTa-specific data effect. Not extractive keyword packing. Not entity binding improvement. The mechanism is a coupled natural data system that doesn't transfer to GPT2 or RoBERTa in tested coordinates.

3. **Remaining open directions**:
   - binding fix, if ever achieved, could be combined with compact-reinvest data (the two mechanisms appear orthogonal)
   - Score improvement through other architecture/data combinations not yet tried
   - Deeper understanding of why compact works specifically in DeBERTa bidirectional MLM
   - Write-up of findings as a data-efficiency study

## What Is Closed (Updated)

- Cross-data binding specificity: binding weakness is generic, not compact-specific
- Source-attested fluent bridge: construction yields too low for clean 100M training
- Extractive comparison: source-only selection fails on stable families
- Coherent88: GlobalPIQA-only gain, stable families worsen
- memory route: 4 variants failed at chance on raw-token binding

---
