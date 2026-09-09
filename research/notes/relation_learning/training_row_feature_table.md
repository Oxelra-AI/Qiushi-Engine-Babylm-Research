# anti source and composition plan: Training-row feature table — why binding absorbed cheap credit

## Decisive finding

The earlier analysis frame-varied binding training rows (16,660 rows, balanced A/B)
contain two dominant cheap features that predict the answer without identity tracking:

| Feature | P(source|T) | P(source|F) | Δ | Prevalence |
|---|---:|---:|---:|---|
| Answer overlaps update-sentence content | 0.1832 | 0.9878 | -0.8046 | B: 99.0% overlap, A: 22.2% |
| Query entity mentioned before update | 0.6185 | 0.0875 | +0.5310 | 77.6% of rows query-first |
| Frame | 0.5000 | 0.5000 | 0 | perfectly balanced |
| Identity: query IS updated entity | 0.0000 | 1.0000 | -1.0000 | 50.0% (balanced) |

## Interpretation

Identity (query_is_updated) IS the strongest feature (delta=1.0), but the optimizer
takes the cheapest sufficient feature first. Since:

1. B/updated answers contain update-sentence words 99% of the time, while A/source
   answers contain them only 22%, a model that picks "answer with more update-text
   overlap" is correct ~82% without any identity tracking.

2. When the query entity appears before the update sentence (77.6% of rows),
   there's a 62% chance the answer is the source state — entity position partially
   predicts identity.

These features captured credit and transferred to official Entity as:
- **Operation-content pull**: selecting options with operation-mentioned content (earlier analysis result)
- **Anti-source policy**: "operations present → not source state" (anti source and composition plan anti-source test)

## Implication for the principle-guided composition

The composition approach (Arm A: coherent replay + aligned restatement pairs)
avoids the binding trap entirely because:
- It uses **standard CE** on all text, not answer-only CE on binding rows
- There are no special answer tokens that can attract cheap features
- Aligned pairs practice correct correspondence through standard masked prediction
- The data composition IS the principle — no objective design needed
