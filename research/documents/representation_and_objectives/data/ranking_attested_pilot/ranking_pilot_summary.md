# equivariance protocol Ranking-Attested Event-to-State Pilot

## Independent attestation
- **Event source**: ATP match files (winner, loser, date, tournament, round)
- **State source**: ATP ranking files (weekly snapshots: date, rank, points)
- **Cross-verification**: match-embedded ranks checked against ranking file

## Data
- Matches loaded: 73140
- Cross-verification: 2000 checked, 0.9505 match rate
- Reversed pairs with ranking: 5685
- Both-consistent (winner ranked higher): 975
- Families built: 300 (240 train, 60 held)
- NLI rows: 3600

## Shortcut test
- TF-IDF logistic: train=0.5000, held=0.5000

## Sample family
- A: Daniil Medvedev, B: Borna Coric
- C1: On February 27, 2023, Daniil Medvedev defeated Borna Coric in the quarterfinal at Dubai. Daniil Medvedev was ranked #7 and Borna Coric was ranked #20 in the official ATP rankings.
- C2: On November 06, 2017, Borna Coric defeated Daniil Medvedev in the round robin at NextGen Finals. Borna Coric was ranked #48 and Daniil Medvedev was ranked #65 in the official ATP rankings.
- Ranking C1: A=#7, B=#20
- Ranking C2: B=#48, A=#65
- Both consistent: True

## Scientific meaning
This pilot demonstrates that independently attested event-to-state
data exists at scale in the tennis archive. The ranking state is
recorded in a separate weekly snapshot file, not derived from the
match result record. Cross-verification confirms the independence.
However, this remains sports-derived and the ranking-consistent
filter biases toward expected outcomes (higher-ranked player wins).
