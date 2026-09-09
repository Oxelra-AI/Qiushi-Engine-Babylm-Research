# edit state probe chck82 synthesis: Edit-state decodability probe on chck_82M — synthesis and closure

## Probe result

Parameterized token value private readout synthesis source-absent edit-state probe on the score-bearing chck_82M
(AdapterDebertaV2ForMaskedLM, 35,463,008 params, trust_remote_code=True).
Items: 3072, exactly reproducing token value private readout synthesis (item_id match 3072/3072).

### Key numbers

| metric | spatial repair route status | chck_82M | delta |
|---|---:|---:|---:|
| raw true-vs-decoy NLL advantage | -0.3122 | -0.1756 | +0.1366 |
| private true-vs-decoy held-out | +0.8365 | +0.7464 | -0.0901 |
| private true-vs-decoy high-error | +1.2397 | +0.9506 | -0.2891 |
| base NLL (no context) | 7.3289 | 7.6183 | +0.2894 |

### Token-kind breakdown

| kind | n | spatial repair route status raw | chck_82M raw | delta |
|---|---|---:|---:|---:|
| content | 2651 | -0.4434 | -0.2895 | +0.1539 |
| function | 227 | -0.0842 | -0.1526 | -0.0684 |
| pronoun | 117 | +1.6172 | +1.9714 | +0.3542 |
| connective | 77 | +0.5999 | +0.4174 | -0.1825 |

Overall negative raw advantage dominated by content words (86% of items).
Pronouns/connectives (6.3% of items) show positive raw advantage.

### Changed-fraction breakdown

| changed_frac_bin | n | spatial repair route status raw | chck_82M raw | delta |
|---|---|---:|---:|---:|
| lt0p2 | 217 | -0.8870 | -0.6852 | +0.2018 |
| 0p2_0p4 | 1724 | -0.3792 | -0.2403 | +0.1389 |
| 0p4_0p6 | 899 | -0.1705 | -0.0788 | +0.0917 |
| ge0p6 | 232 | +0.1748 | +0.4073 | +0.2325 |

Heavily edited items (ge0p6) improved most. Pattern consistent with general
text-editing distributional signal, not relation/state specificity.

## Scientific interpretation

1. **Decodability persists**: The detached private readout can still extract true-source
   information from chck_82M hidden states (+0.7464 held-out, CI 0.57–0.91). The
   information exists but is slightly weaker than at spatial repair route status (-0.09 held-out, -0.29
   high-error).

2. **Information is generic changed-token distributional signal**: Dominated by content
   words where true source conditioning HURTS raw predictions. The adapter model
   improved slightly at using true source for highly edited items, but this is
   distributional text-editing information, not relation/state specificity.

3. **Raw source conditioning still negative**: Neither spatial repair route status nor chck_82M benefits
   from receiving the true source text for these masked-token predictions (-0.31 and
   -0.18 respectively). The models predict WORSE with the true source.

4. **Source-free transfer already falsified**: static decoy contingency validation showed that at the same frozen-82M
   checkpoint, aligned training improved conditioned NLL by +0.149 but WORSENED
   source-free NLL by -0.018 compared to shuffled. The decodable information doesn't
   transfer to source-free inference.

5. **Scientific criterion not met**: "the true-source teacher signal must survive source
   removal on document-disjoint examples and outperform matched decoy or shuffled
   teachers in structure-relevant residual prediction." Prior evidence (static decoy contingency validation) already
   shows failure, and the current probe shows the signal is generic, not
   structure-relevant.

## Route closure

**The edit-state correspondence family is closed.** The complete evidence chain:

- token value private readout synthesis: decodability established on spatial repair route status (+0.84 held-out)
- source free transfer synthesis: source-free transfer showed early promise (shared readout improved NLL -0.895)
- dualview panel readiness and budget-129: broad dual-view training harmful (-1.02 cheap7)
- frozen82 short tail plan-133: separated sparse training showed early aligned > shuffled, but mature
  frozen-82M aligned tail failed source-free transfer (aligned worse than shuffled
  by +0.018 NLL source-free)
- edit state probe chck82 synthesis: decodability persists at chck_82M (+0.75 held-out) but is generic
  changed-token information dominated by content words, not relation/state specificity

No further edit-state correspondence training should be launched from current evidence.

## Canonical artifacts

- Probe script: `scripts/parameterized_edit_state_probe.py`
- chck_82M result: `data/edit_state_probe_chck82/edit_state_probe.{json,md}`
- Smoke verification: `data/edit_state_probe_smoke/`
- Item verification: `data/edit_state_probe_verify/`
