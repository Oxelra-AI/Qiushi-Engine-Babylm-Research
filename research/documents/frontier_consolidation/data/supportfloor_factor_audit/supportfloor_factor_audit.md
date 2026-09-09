# earlier analysis support-floor factor audit

CPU-only audit before any support-floor H100 launch. No official evaluation text or model training used.

## Token/target accounting on the allowed 10M pool

Pool SHA matched expected: `True`. Rows/words scanned: `64740` / `10000000`.

| tokenizer | vocab | raw tok/word | visible tok/word | visible groups/word | over256 rows | trunc toks | expected selected groups/word | expected target toks/word |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 16384 | 1.466452 | 1.429489 | 0.980219 | 15117 | 369626 | 0.147033 | 0.214423 |
| minfreq50_supportfloor | 19609 | 1.448360 | 1.415290 | 0.982206 | 14096 | 330696 | 0.147331 | 0.212294 |

### Deltas versus spatial repair route status legal16k

- `minfreq50_supportfloor_minus_step35_legal16k`: Δ raw tok/word `-0.018092`; Δ visible tok/word `-0.014199`; Δ visible groups/word `0.001986`; relative expected target tokens `0.990067`; relative selected groups `1.002026`.

## Initialization topology

Parameter counts: spatial repair route status `34467424`, minfreq50 `36018649`, delta `1551225`.

### Standard same-seed build (what the existing dormant launcher would do)

- Same-shape tensors exact: `101/168`; same-shape exact numel fraction `0.002257`.
- Random-like same-shape tensors exact: `0/67`; random-like exact numel fraction `0.000000`.
- Shape-mismatch tensors (vocab-dependent): `4`.

First standard-build mismatches:
- `cls.predictions.transform.dense.weight` shape=[480, 480] max_abs=0.12363766878843307
- `deberta.embeddings.position_embeddings.weight` shape=[512, 480] max_abs=0.13531669974327087
- `deberta.encoder.layer.0.attention.output.dense.weight` shape=[480, 480] max_abs=0.14618220925331116
- `deberta.encoder.layer.0.attention.self.key_proj.weight` shape=[480, 480] max_abs=0.13278301060199738
- `deberta.encoder.layer.0.attention.self.pos_key_proj.weight` shape=[480, 480] max_abs=0.13321343064308167
- `deberta.encoder.layer.0.attention.self.pos_query_proj.weight` shape=[480, 480] max_abs=0.1297319531440735
- `deberta.encoder.layer.0.attention.self.query_proj.weight` shape=[480, 480] max_abs=0.16034045815467834
- `deberta.encoder.layer.0.attention.self.value_proj.weight` shape=[480, 480] max_abs=0.12410816550254822

### Init-matched copy simulation

- Copied same-shape tensors: `168`; skipped vocab-shaped tensors: `4`.
- After copying, random-like same-shape exact numel fraction `1.000000`.

## Scientific reading

- Plain minfreq50 training would not isolate tokenizer/support-floor alone: it also changes the contextual random initialization because the larger embedding/output tensors consume a different random stream.
- A better fallback launcher should use an init-matched wrapper that copies all same-shape tensors from a spatial repair route status 16k random reference into the minfreq50 model, leaving only vocab-shaped tensors independently initialized.
- Minfreq50 should still be read as a representation package: it changes segmentation, expected target-token burden, visible groups, and embedding/output rows; this audit quantifies those changes before any score is interpreted.

Full JSON: `experiments/archive/frontier_consolidation/data/supportfloor_factor_audit/supportfloor_factor_audit.json`
