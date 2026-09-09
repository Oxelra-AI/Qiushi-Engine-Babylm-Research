# frozen anchor fastpath disruption design fast-path replay mechanical check

Status: **PASS**

## Spanbreak target preservation

Rows checked: 256; spanbreak rows changed: 256; target mismatches: 0.
Selected target counts coherent/spanbreak: 8595 / 8595.

## Smoke model preservation

| smoke | nonprivate max diff | private rms | private OFF vs chck82 logit maxdiff | private ON vs OFF logit maxdiff |
|---|---:|---:|---:|---:|
| coherent_smoke | 0.0 | 0.06481105 | 0.0 | 0.03886080 |
| spanbreak_smoke | 0.0 | 0.06481105 | 0.0 | 0.03359646 |

The spanbreak control preserves row length, token multiset, selected WWM group IDs, and target-token multiset while breaking cross-span order; both smoke models preserve all non-private chck82 tensors and recover chck82 exactly with private adapters disabled.

JSON: `experiments/archive/frontier_consolidation/data/fastpath_replay_mech_check/fastpath_replay_mech_check.json`
