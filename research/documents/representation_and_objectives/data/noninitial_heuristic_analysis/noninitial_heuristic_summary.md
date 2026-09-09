# matched state and initial owner results non-initial-owner heuristic analysis

This CPU-only file compares saved counterfactual model choices with the deterministic state heuristic: for changed objects, choose the participant who did **not** initially own the changed object; for unchanged objects, preserve the static owner. In ordinary equivariant symmetry repair and macro context state rows the changed object always starts with the complement of the true final owner, so this heuristic is indistinguishable from true role-based state inference unless the initial owner is counterfactually varied.

## Changed-object choices

| arm | ordinary true acc | ordinary heuristic agree | true-final-init true acc | true-final-init heuristic agree | opposite-init true acc | opposite-init heuristic agree |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.609 | 0.609 | 0.354 | 0.646 | 0.643 | 0.643 |
| aligned_matched | 0.818 | 0.818 | 0.203 | 0.797 | 0.805 | 0.805 |
| inverted_matched | 0.753 | 0.753 | 0.250 | 0.750 | 0.768 | 0.768 |
| neutral_matched | 0.557 | 0.557 | 0.471 | 0.529 | 0.609 | 0.609 |

## Deterministic heuristic accuracy against the true target

For changed rows this baseline is 1.0 on ordinary/opposite-initial rows and 0.0 on true-final-initial rows by construction. The important comparison is whether model choices agree with the heuristic even when the heuristic is false.

## Scientific reading

Aligned and inverted state-trained arms agree with the non-initial-owner heuristic on the changed object at roughly the same level whether the heuristic is true (ordinary/opposite-initial rows) or false (true-final-initial rows). This explains the complete orientation probe results/281 dissociation: state-query performance can improve without role-coordinate orientation because the corpus lets the model learn a transition-away rule tied to initial ownership. The unchanged half remains much easier, so pair-both can look high while changed-role semantics is wrong on the counterfactual rows.

## Files

- choice rows: `experiments/archive/representation_and_objectives/data/noninitial_heuristic_analysis/choice_rows.jsonl`
- JSON summary: `experiments/archive/representation_and_objectives/data/noninitial_heuristic_analysis/noninitial_heuristic_summary.json`
- CSV summary: `experiments/archive/representation_and_objectives/data/noninitial_heuristic_analysis/noninitial_heuristic_summary.csv`
