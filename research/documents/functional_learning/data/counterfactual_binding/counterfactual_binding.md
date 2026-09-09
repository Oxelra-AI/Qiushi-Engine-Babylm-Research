# credit allocation binding pilot counterfactual entity-binding diagnostic

Model: `models/frontier` (private_adapter)

## Summary

- Both correct: 7/8
- Mean counterfactual shift: +11.826
- Mean UPDATE margin (new > source): +11.826
- Mean RETAIN margin (source > new): +12.238
- Mean NEUTRAL margin (source > new): +17.148

## Per-pair conditions

| pair | entity | UPDATE | RETAIN | NEUTRAL | CROSS | CF shift | both |
|---|---|---:|---:|---:|---:|---:|:---:|
| tp_001 | Alice | +2.8 | +4.4 | +4.1 | -2.8 | +2.8 | ✓ |
| tp_002 | Professor Chen | +16.4 | +16.9 | +31.7 | +11.2 | +16.4 | ✓ |
| tp_003 | Tom | +16.6 | +3.4 | +6.4 | -2.9 | +16.6 | ✓ |
| tp_004 | Maria | -4.7 | +29.0 | +29.4 | +22.8 | -4.7 | ✗ |
| tp_005 | Elena | +17.3 | +13.0 | +17.6 | +6.3 | +17.3 | ✓ |
| tp_006 | Chef Kim | +6.0 | +10.1 | +11.5 | +1.4 | +6.0 | ✓ |
| tp_007 | David | +19.0 | +6.5 | +22.0 | +14.0 | +19.0 | ✓ |
| tp_008 | Rebecca | +21.2 | +14.7 | +14.6 | +11.4 | +21.2 | ✓ |

## Interpretation

**Counterfactual shift** measures how much the model's preference for new_state
changes between UPDATE context (where target entity was updated) and RETAIN context
(where a distractor entity was updated). A large positive shift means the model
tracks which entity was updated and adjusts its prediction accordingly.

**NEUTRAL** shows the baseline preference without any update. If RETAIN ≈ NEUTRAL,
the model may be ignoring the distractor update rather than actively binding.
If UPDATE >> NEUTRAL, the model actively incorporates the update information.
