# muon switch 40m globalpiqa margin — Muon switch artifact audit

This is a read-only audit of what optimizer-switch artifacts actually contain before using them for route decisions.

| run | max checkpoint | final logged exposure | final step | final loss | metrics file | switch step | immediate switch loss jump |
|---|---:|---:|---:|---:|---|---:|---:|
| adamw_reference | 100M | 100000000 | 2529 | 2.5525617599487305 | True | None |  |
| continuous_muon | 80M | 80000000 | 2024 | 2.397839307785034 | True | None |  |
| muon20toadamw | 50M | 50258212 | 1271 | 2.7521181106567383 | False | 506 | 1.6045744600750154 |
| muon40toadamw | 50M | 57611364 | 1457 | 2.6884374618530273 | False | 1012 | 1.2911111695425856 |

Notes:
- `muon20toadamw`: pre-switch 20-step mean loss 3.4925; first 5 post-switch mean 5.0970; post+6..50 mean 3.7524; post+200..300 mean 3.149679932263818. This reflects zero-initialized AdamW moments for hidden matrices after Muon, not just a smooth optimizer schedule change.
- `muon40toadamw`: pre-switch 20-step mean loss 2.8470; first 5 post-switch mean 4.1381; post+6..50 mean 2.9548; post+200..300 mean 2.7367950642463006. This reflects zero-initialized AdamW moments for hidden matrices after Muon, not just a smooth optimizer schedule change.

JSON: `experiments/archive/representation_and_objectives/data/muon_switch_artifact_audit/muon_switch_artifact_audit.json`
