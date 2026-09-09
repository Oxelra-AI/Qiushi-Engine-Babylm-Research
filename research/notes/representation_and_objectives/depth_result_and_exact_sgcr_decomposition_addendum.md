# Depth Result and Exact SGCR Decomposition

Status: completed depth evaluation and corrected SGCR implementation checks; full SGCR outcome not yet measured.

The legal-40k 12x384 depth endpoint scored:

| Metric | Score |
|---|---:|
| BLiMP | 67.47509961688662 |
| Supplement | 60.13954967329228 |
| EWoK | 50.54719004691065 |
| Entity | 27.171427053823486 |
| COMPS | 52.70234321016695 |
| SuperGLUE | 68.22750825670133 |
| GlobalPIQA | 35.63592233009709 |
| Reading | 7.349322294299417 |
| AoA | 0.0 |
| Overall | 41.02759583135309 |

GlobalPIQA parallel/nonparallel were 24.271844660194176 and 47.0. Overall was below the matched 8x480 value 41.140577774478444, so depth alone did not justify second-seed continuation.

The initial SGCR decomposition was invalid: fixed padding made every token a 256-slot component row, with 99.2976% padding. Removing padding alone was insufficient because isolated decode/re-encode disagreed with exact shared BPE ancestry for 10,910/40,000 tokens.

Recursive BPE-prefix decomposition yielded length counts {1: 16384, 2: 19333, 3: 3629, 4: 571, 5: 63, 6: 16, 7: 4}, or 68,660 slots instead of 10,240,000. Corpus counts covered 13,942,644 tokens and 39,320 used types. Special IDs 0-4 had zero corpus count and residual gate 1. For K=50, mass-weighted residual was 0.0642800704; the uniform control matched within 2.24e-08. Of 24,854 low-support used types, 23,089 had every exact component supported at least 50 times, a fraction 0.928985274.

Cold table/logit difference was 0.0; projection gradients were active on the first backward pass and component gradients after one update. The repaired model had 38,421,952 base plus 1,073,536 new parameters, total 39,495,488, with a 40,000-by-7 decomposition. Checkpoint saving restored the live base table and retained base embeddings in the sidecar. This justified a matched full-horizon representation test, not a claim of improved competence from the short checks.
