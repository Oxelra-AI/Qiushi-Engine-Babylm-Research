# three mechanism 1m profile comparison — Three-Mechanism 1M Word Official Profile Comparison

Purpose: provide official-compatible capability evidence for three mechanism-distinct candidates at identical 1M word exposure, with explicit caveats about parameter/capacity confounds.

## Official profile suite

All scores from official BabyLM 2026 evaluation scripts under `experiments/archive/initial_model_studies/repos/babylm-eval/strict`, using `revision_name chck_1M` and `--backend causal`.

Combined JSON: `experiments/archive/initial_model_studies/data/profile_all3_1m.json`

### Model summary

| alias | run_id | architecture | params | layers | embd | heads | LM head | morph dim |
|---|---:|---|---:|---:|---:|---:|---:|
| dense6x256 | `babylm_compare_dense6x256_1M` | GPT2LMHeadModel (tied) | 8,998,912 | 6 | 256 | 4 | tied | — |
| sparse4x256 | `babylm_compare_sparse4x256_1M` | BabyLMSparseForCausalLM (untied) | 17,924,624 | 4 | 256 | 4 | untied | — |
| morphside4x256 | `babylm_compare_morphside4x256_1M_fix` | BabyLMMorphForCausalLM (untied) | 11,761,408 | 4 | 256 | 4 | untied | 64 |

**Critical caveat:** Dense and sparse/morph-side differ in parameter count, layer count, and whether the output head is tied. Score differences cannot be directly attributed to routing or morphology mechanisms. If sparse shows a profile advantage, a parameter-matched dense control with tied output head must be added before attributing gains to path diversity.

### Official fast/local scores at 1M word exposure

| benchmark | dense6x256 | sparse4x256 | morphside4x256 | V0 (7.42M, 4L, tied) |
|---|---:|---:|---:|---:|
| BLiMP fast | 55.23 | 55.33 | 53.41 | 53.63 |
| BLiMP Supplement fast | 47.60 | **50.00** | 46.40 | 48.00 |
| EWoK fast | 50.55 | 47.09 | **51.55** | 48.36 |
| Entity Tracking fast | 17.45 | 16.20 | **18.25** | 17.83 |
| COMPS | 50.07 | 49.83 | 49.90 | 50.38 |
| Reading eye-tracking | **9.67** | 8.79 | 8.94 | 9.64 |
| Reading self-paced | 2.54 | 2.52 | 2.56 | 2.65 |

**Bold** = best among the three three mechanism 1m profile comparison candidates.

### Observations (not causal conclusions)

1. **Sparse (17.9M params, untied) vs dense (9.0M params, tied):** Sparse has the same BLiMP (~55.2), notably better Supplement (50.0 vs 47.6), but worse on EWoK (47.1 vs 50.6), Entity (16.2 vs 17.5), and Reading (8.8 vs 9.7). The Supplement gain is the most interesting signal for further investigation, but the EWoK/Reading loss and the large parameter gap prevent attributing it to routing alone.

2. **Morph-side (11.8M params, untied) vs dense (9.0M, tied):** Morph-side has the best EWoK (51.6) and Entity (18.3) among the three candidates, but the lowest BLiMP (53.4) and Supplement (46.4). The EWoK improvement is the strongest positive signal for the persistent representation approach, but BLiMP loss and parameter mismatch require a controlled follow-up.

3. **All three candidates** have extremely weak Entity Tracking (16-18%, near chance for multi-choice), consistent with the V0 profile. This is not resolved by any mechanism at 1M words.

4. **Reading scores** are nearly flat across all candidates (eye-tracking 8.8-9.7, self-paced 2.5-2.6). No candidate shows a clear developmental trajectory advantage at 1M exposure.

5. **COMPS** is at or near chance (49.8-50.4) for all candidates at 1M. Morphology-sensitive tasks do not show improvement from the morph-side channel at this exposure.

### What this enables

The comparison provides an evidence baseline for deciding which route to scale to 10M/100M exposure:

- **If sparse Supplement gain holds with a matched dense control** (matching parameter count and output-head tying), the path-diversity mechanism becomes a strong candidate for the full 2026 evaluation.
- **If morph-side EWoK/Entity improvement holds with a matched dense control**, the persistent representation route deserves deeper investigation with learned token-form projections rather than fixed hashed features.
- **If neither advantage survives a controlled dense comparison**, the 1M profiles suggest that simple dense scale (6 layers vs 4, tied head) already produces competitive BLiMP and Reading, and the main bottleneck may be exposure/data rather than architecture.

### Next work

The next step should not expand the candidate set. It should:

1. Run a **parameter-matched dense control** (same ~17.9M params, 4 layers, untied output head) so sparse gains can be attributed to routing rather than capacity.
2. Optionally match the morph-side parameter count similarly.
3. If a clear advantage survives, scale the best candidate to 10M exposure with the full official checkpoint schedule and evaluate at intermediate checkpoints.
4. Do **not** launch 100M training or final expression until a controlled profile advantage is confirmed.
