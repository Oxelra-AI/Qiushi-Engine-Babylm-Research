# commoncopy and paired world design paired-world shortcut pilot

Created UTC: `2026-09-02T14:06:44Z`

Families: `149`; context packets: `596`

## score_ablated
- context_packets: `298`
- template_counts: `{'active_winner_first': 75, 'passive_loser_first': 75, 'lost_to_loser_first': 74, 'not_loser_winner_first': 74}`
- winner_first_rate: `0.5`
- exact_query_verb_defeated_rate: `0.0`
- team_count_equal_rate: `1.0`
- winner_loser_count_equal_rate: `1.0`
- hypothesis_bow_collision_rate: `1.0`
- score_digit_sentence_rate: `0.0`
- mean_text_token_count: `41.56711409395973`
- mean_context_pair_bow_jaccard_names_numbers_normalized: `0.8371314891020731`
- min_context_pair_bow_jaccard_names_numbers_normalized: `0.7872340425531915`

## score_visible
- context_packets: `298`
- template_counts: `{'active_winner_first': 75, 'passive_loser_first': 75, 'lost_to_loser_first': 74, 'not_loser_winner_first': 74}`
- winner_first_rate: `0.5`
- exact_query_verb_defeated_rate: `0.0`
- team_count_equal_rate: `1.0`
- winner_loser_count_equal_rate: `1.0`
- hypothesis_bow_collision_rate: `1.0`
- score_digit_sentence_rate: `1.0`
- mean_text_token_count: `38.97651006711409`
- mean_context_pair_bow_jaccard_names_numbers_normalized: `0.8335041157429713`
- min_context_pair_bow_jaccard_names_numbers_normalized: `0.782608695652174`

## Scientific use
This is not a training corpus. It is a CPU construction probe for a later, shortcut-resistant paired-world test after the common-copy DeBERTa adjudication is complete.
The score-visible version intentionally preserves the numeric shortcut; the score-ablated version removes it and keeps entity counts/directed-query bags matched so order-sensitive role binding is needed.

## Boundary
CPU construction/readout only; no teacher generation, BabyLM training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.
