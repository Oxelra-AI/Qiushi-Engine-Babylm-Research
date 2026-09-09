# preservation controls and corrected interpretation acquisition stream audit

Status: `ACQUISITION_STREAM_AUDIT_DONE`
Audited updates: `5`
Acquisition digest: `b6be50e4b0f9e3aaf8e5fdb19d004d0779905c616c247d9e15855b102d48bbea`

## Seed audit
Original earlier analysis/earlier analysis acquisition global seed: `135392207`
preservation experiment design and evidence seeds by lambda_pres: `{'0.0': 940740990, '1.0': 253484362, '3.0': 228406199}`

Interpretation: acquisition masks/labels are deterministic from row keys, but preservation experiment design and evidence did not hold the model dropout/RNG stream fixed across preservation strengths.

## Per update
- u0001: rows 252, words 39652, focus targets 247, ordinary targets 7894, sha `7839c75f01838ff78e9a8603ef44225bd8c9eeeb961cd933f4825d11324ed2a6`
- u0002: rows 255, words 39663, focus targets 375, ordinary targets 7576, sha `c53ec103ba188d0826e459dc8701eb1ad5049153962dec4b31cee4eae3e21a36`
- u0003: rows 256, words 39583, focus targets 363, ordinary targets 7802, sha `ffb828efe1d2a16cd7779ca167e7539bad4eddd49394456d71bd24e1415bc6fa`
- u0004: rows 257, words 39592, focus targets 456, ordinary targets 7056, sha `ae29571cdbd2d9ca1d56e00faf369e30bc805b9a409428d0b74030b96fe4c428`
- u0005: rows 256, words 39299, focus targets 442, ordinary targets 6895, sha `c87f837d10f8c35c97e19cbd8e18503fc42436b3662206c1e8608b5dc849c6b5`
