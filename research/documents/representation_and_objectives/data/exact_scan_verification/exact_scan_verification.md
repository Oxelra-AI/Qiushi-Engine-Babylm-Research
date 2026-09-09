# raw span discovery design exact-scan tensor verification

Verdict: **PASS**

- Total substrate rows: 4832
- Unique event+candidate checks: 1951
- State checks passed: 831
- Comparison checks passed: 1120
- Mismatches: 0

## Name statistics
- Unique names across substrate: 32
- Name list: Arun, Ava, Ben, Caleb, Eli, Felix, Hugo, Iris, Jonas, June, Keira, Leah, Lena, Luca, Mateo, Maya, Milo, Mira, Nia, Noel, Nora, Omar, Pavel, Rina, Sara, Simon, Tara, Theo, Tomas, Vera, Yara, Zara
- Substring overlaps: [('Eli', 'Felix')]

## Position statistics
- cand_missing: 0
- cand_multi: 0
- cand_single: 384
- nonoverlap: 384
- other_missing: 0
- other_multi: 0
- other_single: 384
- overlap: 0

## Interpretation

If verdict=PASS, exact raw-substring scan produces identical token sequences
to causal gauge experimental design's normalize_event_for_candidate on every substrate row.
This confirms that a deterministic scan recovers the supplied-coordinate
representation from raw text + name strings. GPU replay with exact scan
would therefore produce identical results to causal gauge experimental design.

The learned span discovery experiment can proceed: its comparison baseline
is the exact-scan (=causal gauge experimental design) representation, and any difference in gauge
transport is attributable to the learned localization interface.
