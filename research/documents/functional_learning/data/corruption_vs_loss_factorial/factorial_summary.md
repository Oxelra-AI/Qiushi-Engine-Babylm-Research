# corruption vs loss design corruption-vs-loss factorial

## Design

| Arm | Input corruption | Loss | Answer coefficient |
|---|---|---|---|
| 1. clean_answer_only | None | Answer only | Full |
| 2. corrupted_answer_only | 15% WWM | Answer only | Full |
| 3. corrupted_answer_plus_bg | 15% WWM | Answer + bg | Answer full, bg separate |

## Results

| Arm | Held flip | Held U | Held R | Held N | Held R-N |
|---|---|---|---|---|---|
| clean_answer_only | 2/12 | +0.446 | +0.039 | +1.673 | -1.634 |
| corrupted_answer_only | 6/12 | +1.728 | -0.722 | +2.296 | -3.018 |
| corrupted_answer_plus_bg | 0/12 | +2.018 | -1.385 | +0.866 | -2.251 |

Baseline: held flip 0/12

## Interpretation

Arm 1 (clean, answer-only): held flip 2/12, U=+0.446, R=+0.039, N=+1.673
Arm 2 (corrupted, answer-only): held flip 6/12, U=+1.728, R=-0.722, N=+2.296
Arm 3 (corrupted, answer+bg): held flip 0/12, U=+2.018, R=-1.385, N=+0.866
→ Evidence corruption does NOT destroy usable relational information under 15% WWM.
→ Background GRADIENTS are the destructive factor, not input corruption.
→ BabyLM design: preserve relational packet content, isolate answer credit from bg loss.
