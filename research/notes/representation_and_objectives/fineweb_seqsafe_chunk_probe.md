# cached fineweb quality audit FineWeb seq256-safe chunk probe

Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`; seq_len=256. Source: cached INITIAL_MODEL_STUDIES FineWeb-Edu single-document rows.

| chunk words | FineWeb rows | FineWeb trunc frac | FineWeb visible word frac | control trunc frac | control visible word frac | Δvisible word frac (FW-control) | FW token p95 | control token p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 22064 | 0.0015 | 0.9998 | 0.0010 | 0.9998 | -0.0000 | 147.0 | 151.0 |
| 96 | 22064 | 0.0021 | 0.9997 | 0.0014 | 0.9997 | -0.0000 | 163.0 | 172.0 |
| 112 | 22064 | 0.0040 | 0.9992 | 0.0020 | 0.9994 | -0.0002 | 189.0 | 200.0 |
| 128 | 22064 | 0.0101 | 0.9979 | 0.0084 | 0.9986 | -0.0007 | 215.0 | 229.0 |
| 144 | 22064 | 0.0263 | 0.9945 | 0.0503 | 0.9936 | +0.0009 | 242.0 | 257.0 |
| 160 | 11032 | 0.1643 | 0.9842 | 0.3147 | 0.9727 | +0.0116 | 287.0 | 296.0 |

Interpretation: the 160-word cached fallback imported an avoidable visibility mismatch. A seqsafe variant should use a shorter split such as 96 or 112 words if it preserves enough context while bringing both blocks near-full visibility.

JSON: `experiments/archive/representation_and_objectives/training/data/fineweb_seqsafe/seqsafe_chunk_probe.json`
