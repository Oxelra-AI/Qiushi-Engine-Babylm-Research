# sgcr followup assets and entry logic — SGCR uniform-control preflight

CPU-only; no training and no evaluation. This prepares the mass-matched nonsharing comparison for the pending exact-prefix SGCR endpoint.

Status: `PASSED`

## Why this control would matter
If SGCR improves the measured support-aligned columns (especially COMPS/GlobalPIQA) but remains below the visible leader, the full endpoint alone cannot distinguish support-dependent routing from generic auxiliary embedding capacity. The uniform control keeps the same component table/projection and the same training-token residual mass, but sets one ordinary-token rho value for all tokens.

## Preflight numbers
- Treatment residual mass: `0.06428007036447525`
- Uniform residual mass: `0.06428009271621704`
- Abs diff: `2.2351741790771484e-08`
- Treatment low/common residual type-mean ratio: `2.5294293292828436`
- Uniform low/common residual type-mean ratio: `1.0`
- Ordinary used uniform rho unique count (rounded 1e-12): `1`
- Params: base `38421952`, new `1073536`, total `39495488`
- Cold effective-table max diff vs standard: `0.0`
- Decomposition SHA: `b450cc45f7d66564a894fb8cc12e0b0ab8e3ba7db0339cad5eec6f30c6edfd8b`; histogram `{1: 16384, 2: 19333, 3: 3629, 4: 571, 5: 63, 6: 16, 7: 4}`

## Use condition
Run the full uniform-control training only if the pending SGCR endpoint is sub-frontier but improves COMPS/GlobalPIQA or Overall over matched depth enough that distinguishing support routing from generic auxiliary capacity will change the next route.

## Files
- JSON: `experiments/archive/representation_and_objectives/data/sgcr_uniform_control_preflight/sgcr_uniform_control_preflight.json`
