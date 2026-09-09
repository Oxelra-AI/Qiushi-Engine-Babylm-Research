# sgcr zero sharing parity SGCR zero-sharing training parity

Status: `SGCR_ZERO_SHARING_PARITY_PASSED`

This test compared a standard 12x384 DeBERTa-v2 two-step training prefix with SGCR at K=0 on the same first 78,887 words of the real 100M stream. It uses identical seeds, WWM masks, masked-token-weighted microbatch accumulation, AdamW/cosine schedule, and gradient clipping.

- Actual words: 78887
- Effective batches: 2
- Standard losses: [10.648108288969965, 10.644323993480894]
- SGCR(K=0) losses: [10.648108288969965, 10.644323993480894]
- Loss absolute differences: [0.0, 0.0]
- Max standard-vs-SGCR state diff after two steps: 0.0
- Probe logits max diff after two steps: 0.0
- JSON: `experiments/archive/representation_and_objectives/data/sgcr_zero_sharing_parity/sgcr_zero_sharing_parity.json`
