# factorial construction report tensor parity smoke test

JSON: `experiments/archive/representation_and_objectives/data/tensor_parity_smoke/tensor_parity_smoke.json`

## Parity checks
- Init SHA match: True (5397c302c18ee1a9...)
- Param count match: True (30528064)
- Pool word count match: True (2279 per arm)
- LR schedule identical: True
- All finite: True
- Params diverge after training: True

## Verdict: READY

### babylm2026 live surface
- Losses: {'HS': 9.769025802612305, 'LS': 9.740862846374512, 'HD': 9.757328987121582, 'LD': 9.832427978515625}
- Masked tokens: {'HS': 120, 'LS': 110, 'HD': 123, 'LD': 121}
- LR parity: True

### leader analysis and route pivot
- Losses: {'HS': 9.763880729675293, 'LS': 9.781519889831543, 'HD': 9.754597663879395, 'LD': 9.783050537109375}
- Masked tokens: {'HS': 119, 'LS': 135, 'HD': 130, 'LD': 120}
- LR parity: True

### generation slice quality
- Losses: {'HS': 9.807605743408203, 'LS': 9.812556266784668, 'HD': 9.765483856201172, 'LD': 9.813552856445312}
- Masked tokens: {'HS': 113, 'LS': 92, 'HD': 109, 'LD': 103}
- LR parity: True
