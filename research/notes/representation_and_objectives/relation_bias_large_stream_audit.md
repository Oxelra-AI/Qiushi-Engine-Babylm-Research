# relation bias execution replay relation-biased masking large-pool audit

CPU-only audit on the exact allowed 10M compact_view_reinvest pool with the legal 40k tokenizer. No model training, no GPU work, no official evaluation text.

## Integrity

- Pool SHA matched expected: `True`; tokenizer SHA matched expected: `True`.
- Rows/words audited: `64740` / `10000000`.

## Global visible relation substrate

- Visible groups: `9871336`; visible tokens: `13706162`.
- Relation-bearing groups: `1079828` (`0.109390`); relation-bearing tokens: `1316350` (`0.096041`).
- Row relation-group fraction p50/p90/p99: `0.1074` / `0.1625` / `0.2125`.

## Candidate boosts

- Boost 2.0: relation selected-group fraction `0.218781`, relation target-token fraction `0.194706`, selected-group multiplier `1.000000`, target-token multiplier `0.986522`, clipped rows `0` (`0.000000`).
- Boost 3.0: relation selected-group fraction `0.328166`, relation target-token fraction `0.296100`, selected-group multiplier `1.000000`, target-token multiplier `0.973046`, clipped rows `5` (`0.000077`).
- Boost 4.0: relation selected-group fraction `0.437356`, relation target-token fraction `0.400149`, selected-group multiplier `1.000000`, target-token multiplier `0.959609`, clipped rows `120` (`0.001854`).

## Interpretation

- Boost 2.0 gives relation selected-group fraction 0.2188 and relation target-token fraction 0.1947; selected-group multiplier 1.000000, target-token multiplier 0.986522.
- Boost 2.0 row clipping fraction is 0.000000; low clipping supports it as the cleanest relation-pressure intervention if legal40k endpoints motivate this route.
- Boost 3.0 raises relation selected-group fraction to 0.3282 but clips 0.000077 of rows and lowers total target tokens to multiplier 0.973046.
- Boost 4.0 raises relation selected-group fraction to 0.4374 with row clipping 0.001854 and stronger target-token reduction 0.959609; it is less clean as a first route.
- Across source classes under boost 2.0, relation selected-group fraction ranges from 0.1221 (simple_wiki) to 0.2637 (bnc_spoken), so source composition changes the dose but the intervention remains broad rather than confined to the compact block.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/relation_bias_large_stream_audit/relation_bias_large_stream_audit.json`
- Source CSV: `experiments/archive/representation_and_objectives/data/relation_bias_large_stream_audit/boost_budget_by_source.csv`
- Category CSV: `experiments/archive/representation_and_objectives/data/relation_bias_large_stream_audit/boost_budget_by_category.csv`
- Row quantiles CSV: `experiments/archive/representation_and_objectives/data/relation_bias_large_stream_audit/row_relation_density_quantiles.csv`
