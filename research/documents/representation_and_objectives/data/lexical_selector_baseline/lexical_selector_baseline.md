# selector reader bridge synthesis and macro challenge lexical selector baseline

A deterministic matcher extracts the tag from the inline-role context line whose role words match the query role. This tests whether exact selector reader bridge synthesis and macro challenge selector success exceeds local surface correspondence.

| eval | exact matcher top1 | exact none | synonym-map matcher top1 | synonym-map none |
|---|---:|---:|---:|---:|
| held_exact_nsA | 1.000 | 0 | 1.000 | 0 |
| held_exact_nsB | 1.000 | 0 | 1.000 | 0 |
| held_para_nsA | 0.000 | 640 | 1.000 | 0 |
| held_role_swap_nsA | 1.000 | 0 | 1.000 | 0 |
| trainChanged_exact_nsA | 1.000 | 0 | 1.000 | 0 |

For exact role wording and role-swap contexts, surface matching is sufficient. For held_para_nsA, exact matching finds no background/update line because the query uses prior/revised; a hand-supplied synonym map restores 1.0, isolating the missing temporal-role paraphrase map.
