# relational xspan materialization v2 — relational XSpan v2 status and remaining repair

## What changed from v1 to v2

The v1 rule-based XSpan materializer produced 7,611 rows but the formal audit flagged 1,077 targets (14.15%), dominated by dangling target endings such as `in`, `until`, `according`, and discourse starts such as `because`/`however`.

A repaired v2 materializer was written at:

- `scripts/materialize_relational_xspan_v2.py`

It keeps the core experimental constraint: **no protected-model score filtering**. counterfactual propagation closed model scores were used only to infer a corpus-intrinsic rule family. v2 still uses official raw Simple-Wiki same-line adjacent pairs and rule-based target selection.

v2 output:

- JSONL: `data/xspan_revision_116/relational_xspan_v2_seed116_target200000_actual.jsonl`
- Summary: `data/xspan_revision_116/relational_xspan_v2_seed116_target200000_actual.summary.json`
- Rows: 7,555
- True-context counted words: 199,980
- Target types:
  - `location_spatial_phrase`: 4,139
  - `definition_property_complement`: 1,546
  - `semantic_content_continuation`: 1,169
  - `action_object_result_phrase`: 701

## Inline v2 quality snapshot

A direct audit of the v2 JSONL showed:

- rows: 7,555
- flagged rows: 161
- flagged fraction: 2.13%
- reason counts: `too_long: 161`

This is a major improvement over v1, especially for dangling preposition/complementizer endings, but sample inspection still found remaining quality problems that make v2 **not yet ready for trainer use**.

Representative v2 samples:

- good: `east of Thurso` / `It is 5 miles east of Thurso.`
- good: `youngest Councillor of World Future Council` / `It is the youngest Councillor of World Future Council.`
- good-ish: `opposed the New Deal, the Marshall Plan, social security` / needs phrase completion to include the list endpoint.
- questionable: `released` / target is a bare verb, too weak as a semantic span.
- questionable: `lives` / bare verb with missing location/object.
- questionable: `in the unification process after` / still dangling on `after` because `after` was not in the v2 dangling list.
- questionable: `in the slums of Kolkata (Calcutta, India) with Mother` / likely truncated before `Teresa`.
- questionable: `species which mate whenever conditions allow it` / target contains relative/discourse clause; not a clean short content span.
- questionable: `home to the Tampa Bay Buccaneers of the National` / incomplete proper-noun complement.

## v3 repair rules before training

Before implementing any XSpan trainer or spending H100 time, make v3 materialization stricter:

1. Require at least two substantive content tokens in the target span. This rejects bare verbs such as `released` and `lives`.
2. Add `after`, `before`, `next`, `during`, `whenever`, and similar subordinators/prepositions to the dangling-end set.
3. Reject targets containing mid-span clause triggers (`which`, `who`, `whenever`, `where`, `when`) unless the span ends before the trigger and still has enough content.
4. Reject or extend targets with unmatched parentheses/quotes or incomplete proper-name endings.
5. Increase phrase-completion capacity slightly for proper-noun complements, but reject if the completed phrase exceeds a compact semantic span length.
6. Re-run a formal audit on the v3 output, not a hardcoded v1 audit path.

## Scientific status

The rule-based relational XSpan route remains scientifically stronger than the closed auxiliary-head routes because it changes the primary MLM masking target rather than adding a discriminative shortcut label. But data quality is now the bottleneck. v2 proves that model-independent rule construction can produce a large official-data pool, yet v3 cleanup is required before trainer construction.
