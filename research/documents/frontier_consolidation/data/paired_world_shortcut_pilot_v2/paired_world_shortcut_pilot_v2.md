# commoncopy and paired world design paired-world shortcut pilot v2

Created UTC: `2026-09-02T14:08:09Z`

Every family/context is crossed with all templates; score_ablated omits numeric score sentence entirely. Template identity is no longer a label proxy because each template appears equally with team_a and team_b as winner.

Families: `149`; context packets: `2384`

## score_ablated
- context_packets: `1192`
- families: `149`
- template_counts: `{'active_winner_first': 298, 'passive_loser_first': 298, 'lost_to_loser_first': 298, 'not_loser_winner_first': 298}`
- template_by_entailed_direction: `{"active_winner_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "lost_to_loser_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "not_loser_winner_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "passive_loser_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}}`
- max_template_direction_abs_delta: `0`
- winner_first_rate: `0.5`
- team_a_entailed_rate: `0.5`
- exact_query_verb_defeated_rate: `0.0`
- team_count_equal_rate: `1.0`
- winner_loser_count_equal_rate: `1.0`
- hypothesis_bow_collision_rate: `1.0`
- score_digit_sentence_rate: `0.0`
- any_digit_rate: `1.0`
- mean_text_token_count: `29.568791946308725`
- mean_same_template_context_pair_bow_jaccard_names_numbers_normalized: `1.0`
- min_same_template_context_pair_bow_jaccard_names_numbers_normalized: `1.0`

## score_visible
- context_packets: `1192`
- families: `149`
- template_counts: `{'active_winner_first': 298, 'passive_loser_first': 298, 'lost_to_loser_first': 298, 'not_loser_winner_first': 298}`
- template_by_entailed_direction: `{"active_winner_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "lost_to_loser_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "not_loser_winner_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}, "passive_loser_first": {"absolute_delta": 0, "team_a_over_team_b": 149, "team_b_over_team_a": 149}}`
- max_template_direction_abs_delta: `0`
- winner_first_rate: `0.5`
- team_a_entailed_rate: `0.5`
- exact_query_verb_defeated_rate: `0.0`
- team_count_equal_rate: `1.0`
- winner_loser_count_equal_rate: `1.0`
- hypothesis_bow_collision_rate: `1.0`
- score_digit_sentence_rate: `1.0`
- any_digit_rate: `1.0`
- mean_text_token_count: `37.97818791946309`
- mean_same_template_context_pair_bow_jaccard_names_numbers_normalized: `1.0`
- min_same_template_context_pair_bow_jaccard_names_numbers_normalized: `1.0`

## Remaining scientific limits
Still one relation family (sports outcome) and deterministic templates; future use needs mixed sources/relations, held-out family splits, paraphrase checks, and a bag-of-words/position-light baseline before model training.

## Boundary
CPU construction/readout only; no teacher generation, BabyLM training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.
