# strict innovation target readout before scores strict row-unique innovation target probe

Fixed CPU readout of repaired row-unique content-innovation target prediction for token-mean and, once available, earlier analysis checkpoints.

- examples loaded: `128` / requested `128`
- selected targets: `64`; selected rows: `48`; token labels: `108`
- target manifest: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe_manifest_smoke/strict_innovation_target_manifest.jsonl`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Results
| checkpoint | status | n | true loss | source help | same-row decoy advantage | cross-row decoy advantage | masked-same | source help bootstrap 5-95 | same decoy bootstrap 5-95 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|

## Reading
- This readout is not a BabyLM score; it measures whether the exact target class selected for innovation-biased masking becomes easier under true source context.
- For earlier analysis, a useful mechanism signature is lower true_loss and/or higher source_help on these strict targets versus tokenmean at the same exposure, interpreted together with the official-compatible cheap-column trajectory.
- If broad transfer weakens and this target readout does not improve, the innovation-masking family should not be carried forward just because exact-swap is mechanically cleaner.

Full JSON: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe_manifest_smoke/strict_innovation_target_probe.json`
