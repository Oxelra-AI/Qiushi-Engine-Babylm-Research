# extractive view pool preflight and training plan extractive view pool preflight

Pairs: 12155, changed block rows: 3006

Word count mismatches: {'extractive_wide': 0, 'extractive_balanced': 0}


## Content density comparison (mean content_fraction)

- compact: 0.6474
- repeat: 0.4884
- extractive_wide: 0.8107
- extractive_balanced: 0.6321

## Source content coverage (mean)

- compact: 0.6570
- repeat: 0.5981
- extractive_wide: 0.9821
- extractive_balanced: 0.7923

## Tail content coverage (mean)

- compact: 0.7024
- repeat: 0.0474
- extractive_wide: 0.9999
- extractive_balanced: 0.8136

## Source-absent content words (total)

- compact: 17891
- repeat: 0
- extractive_wide: 0
- extractive_balanced: 0

## Source position decile coverage

| Decile | compact | repeat | ext_wide | ext_balanced |
|--------|---------|--------|----------|--------------|
| 0 | 0.6828 | 1.0000 | 0.8777 | 0.8308 |
| 1 | 0.6263 | 1.0000 | 0.9518 | 0.7266 |
| 2 | 0.6206 | 1.0000 | 0.9870 | 0.7750 |
| 3 | 0.6318 | 0.9969 | 0.9972 | 0.7740 |
| 4 | 0.6352 | 0.9484 | 0.9998 | 0.7565 |
| 5 | 0.6408 | 0.7100 | 1.0000 | 0.7665 |
| 6 | 0.6390 | 0.3525 | 1.0000 | 0.7923 |
| 7 | 0.6596 | 0.1215 | 1.0000 | 0.7657 |
| 8 | 0.7088 | 0.0567 | 1.0000 | 0.7153 |
| 9 | 0.7632 | 0.0585 | 1.0000 | 0.8748 |

## Stream tokens (10M pool)

- compact: 14664519
- repeat: 14640121
- extractive_wide: 14671110
- extractive_balanced: 14654691

## Changed block active tokens

- compact: 607909
- repeat: 583511
- extractive_wide: 614500
- extractive_balanced: 598081

## Examples


### compact:frontier_consolidation_fwcompact_medium_046028
- **Source**: Based on the home's plans, the Home Energy Rater uses an energy efficiency software package to perform an energy analysis of the home's design.
- **Compact**: The Home Energy Rater uses software to analyze the home's design based on its plans.
- **Repeat**: Based on the home's plans, the Home Energy Rater uses an energy efficiency software package
- **Extractive wide**: Based home's plans, Home Energy Rater energy efficiency software package perform energy analysis home's design.
- **Extractive balanced**: Based on plans, the Home Rater an energy software package an energy analysis the design.

### compact:frontier_consolidation_fwcompact_medium_030624
- **Source**: Just like the physical environment, the virtual environment has become polluted in some way shape or form - some areas are worse than others.
- **Compact**: Like the physical environment, the virtual environment is polluted, with some areas worse than others.
- **Repeat**: Just like the physical environment, the virtual environment has become polluted in some way shape
- **Extractive wide**: like physical environment, virtual environment polluted shape form - some areas are worse than others.
- **Extractive balanced**: like the physical environment, virtual environment has some shape form - areas are worse others.

### compact:frontier_consolidation_fwcompact_medium_035003
- **Source**: The polar regions become much colder than they are presently and there are noticeably large differences in temperature from the equator There have actually been a number of ice ages in the history of the Earth.
- **Compact**: Polar regions become much colder than now, with large temperature differences from the equator, and Earth has had many ice ages.
- **Repeat**: The polar regions become much colder than they are presently and there are noticeably large differences in temperature from the equator
- **Extractive wide**: polar regions colder presently noticeably large differences temperature equator actually a number of ice ages in the history of the Earth.
- **Extractive balanced**: polar regions become colder they presently there noticeably large differences temperature the equator actually been number ice ages history of Earth.


JSON: `experiments/archive/frontier_consolidation/data/extractive_view_pool_preflight/extractive_view_pool_preflight.json`
