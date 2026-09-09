# mechanism preservation summary — aligned-pair joint visibility audit

JSON: `experiments/archive/frontier_consolidation/data/pair_joint_visibility/pair_joint_visibility_audit.json`

This audit measures whether a Qwen original/rewrite pair is jointly present in one visible training example, not merely whether counted words are visible.

## Overall pair visibility

| tokenizer | arm | pairs | complete joint | partial joint | split no joint | mean examples touching | mean visible pair frac |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline16k | fixed_seq256_row_atomic | 37594 | 0.997446 | 0.999628 | 0.0 | 1.0 | 0.999391 |
| baseline16k | stagewise_row_chunked | 37594 | 0.570916 | 0.986673 | 0.01314 | 1.507927 | 0.863726 |
| baseline16k | candidate_pair_atomic | 37594 | 0.672528 | 0.984146 | 0.0 | 1.0 | 0.92742 |
| leader40k | fixed_seq256_row_atomic | 37594 | 0.999282 | 0.999894 | 0.0 | 1.0 | 0.999841 |
| leader40k | stagewise_row_chunked | 37594 | 0.572006 | 0.987099 | 0.012848 | 1.507927 | 0.864133 |
| leader40k | candidate_pair_atomic | 37594 | 0.73948 | 0.992073 | 0.0 | 1.0 | 0.945717 |

## Stagewise row-chunked rates by stage (40k tokenizer)

| stage key | pairs | complete joint | partial joint | split no joint | mean visible pair frac |
|---|---:|---:|---:|---:|---:|
| 1 | 11077 | 0.083867 | 0.971202 | 0.028798 | 0.683229 |
| 2 | 11010 | 0.461308 | 0.985104 | 0.014896 | 0.855017 |
| 3 | 15507 | 0.999291 | 0.999871 | 0.0 | 0.99983 |

## Scientific reading

- Inherited fixed-seq row-atomic clean-Qwen preserves complete joint visibility for 0.999 of pairs and partial joint visibility for 1.000.
- repaired pilot eval row chunking preserves complete joint visibility for only 0.572 of pairs, although counted-word visibility was near-unity; partial joint visibility is 0.987.
- Candidate pair-atomic stagewise data would preserve complete joint visibility for 0.739 of pairs and partial joint visibility for 0.992 under the leader 40k tokenizer.
- Therefore a fast score from the in-flight row-chunked model cannot by itself decide whether leader-style curriculum helps or hurts the paired-view mechanism; a pair-boundary-aware arm is needed if the row-chunked model is weak or ambiguous.
