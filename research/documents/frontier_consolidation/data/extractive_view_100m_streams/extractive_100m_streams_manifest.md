# earlier analysis extractive 100M streams

Created UTC: `2026-09-02T05:57:45Z`
All status ok: `True`

These streams replace only example_id 950000--953004 rows in the compact 100M stream; every other line is copied byte-for-byte.

| variant | rows | words | changed rows | filler rows | sha256 | ok |
|---|---:|---:|---:|---:|---|---:|
| extractive_balanced | 647400 | 100000000 | 30050 | 617350 | `8a4e77af3ae7049b889e5d8f174616606a930bcd8371188ea35db3197175776c` | True |
| extractive_wide | 647400 | 100000000 | 30050 | 617350 | `b9e4666844f966a2fedbe6a37eb61aa3b1bcd5fe08cdc738cd0a15fda4258753` | True |

## Token-mass summary (extractive view pool preflight and training plan exact 10M counts ×10)

| arm | active tokens 100M | changed-block active tokens 100M | delta vs compact active | delta vs compact changed |
|---|---:|---:|---:|---:|
| compact | 146645190 | 6079090 |  |  |
| repeat | 146401210 | 5835110 |  |  |
| extractive_balanced | 146546910 | 5980810 | -98280 | -98280 |
| extractive_wide | 146711100 | 6145000 | 65910 | 65910 |

The paired balanced+wide comparison is required because the extractive construction cannot match compact simultaneously on density, source coverage, source-absent content, fluency, and token mass.
JSON: `experiments/archive/frontier_consolidation/data/extractive_view_100m_streams/extractive_100m_streams_manifest.json`
