# twoseed synthesis and provenance: Two-Seed Deterministic Signature and Mechanism Provenance

This synthesis records the deterministic attribution already established at two seeds
and the provenance chain from Stage-II understanding to Stage-III decisions, written
from verified source files rather than approximate recollection.

## 1. Two-Seed Deterministic Signature

### Seed64 complete ladder (from earlier analysis JSON)

| rung | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86 | 42.0240 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 68.9457 | 38.565 | 8.165 | 0.0 |
| ordinary (O64) | 42.0926 | 68.47 | 63.63 | 49.83 | 28.16 | 52.00 | 68.9885 | 39.535 | 8.22 | 0.0 |
| exact (M,S)64 | 42.2025 | 68.12 | 63.08 | 49.95 | 29.39 | 52.15 | 68.8878 | 40.05 | 8.195 | 0.0 |
| clean64 (v5) | 42.2464 | 68.26 | 63.28 | 49.82 | 29.40 | 52.16 | 69.0478 | 40.05 | 8.20 | 0.0 |

### Seed65 landed values

| rung | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | AoA | SuperGLUE | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O65 | 68.52 | 63.62 | 49.81 | 28.09 | 52.00 | 39.535 | 8.195 | 0.0 | 69.2733 | 42.1159 |
| (M,S)65 | 68.09 | 63.09 | 49.82 | 29.29 | 52.14 | 40.05 | 8.195 | 0.0 | 68.9349 | 42.1789 |
| clean65 | 68.22 | 63.29 | 49.72 | 29.45 | 52.14 | 40.05 | 8.195 | 0.0 | 69.0206 | 42.2317 |

### Deterministic signature replicates across seeds

**O→(M,S) rung (acquisition installs source-responsive behavior, pays lexical cost):**
- Seed64: Entity +1.23, Supplement −0.55, BLiMP −0.35
- Seed65: Entity +1.20, Supplement −0.53, BLiMP −0.43

**clean−(M,S) rung (preservation partially recovers costs, keeps Entity):**
- Seed64: BLiMP +0.14, Supplement +0.20, EWoK −0.13, Entity +0.01 → Overall +0.044
- Seed65: BLiMP +0.13, Supplement +0.20, EWoK −0.10, Entity +0.16 → Overall +0.053

Same signs, near-identical magnitudes on three of four columns. **The preservation
increment is now two-seed replicated** (seed64 +0.044, seed65 +0.053).
MS65 Overall = 42.1789, clean65 Overall = 42.2317.

### Interpretation limits

Clean differs from (M,S) in two ways: it adds the coherent86-teacher KL constraint
*and* it exposes the model to additional ordinary-mask presentations with source evidence
visible. A no-teacher control was never run and exploration is now closed, so the report
should present preservation as a combined effect and state this as a limit.

## 2. Stage-II → Stage-III Prospective Provenance

For each Stage-III design choice, what was known before the outcome:

### 2a. "The intended variable must become the cheapest sufficient explanation"

**Before:** Descriptor packets (calibrated route options) and literal-binding maps (related experiments)
showed that organizing examples around an intended operation does not ensure that
operation is learned when candidate presence, operation count, position, recency,
or format provide sufficient cheaper explanations. The 480-map family proved that
literal-value binding is learnable in principle but transfers family priors to official
Entity, not clean variable tracking.

**Design decision:** The exact `(M,S)` construction uses packed Qwen-pair rows with the first view/source text still visible, densely masks detected second-view content groups, and keeps labels sparse on a deterministic subset of those second-view groups. Ordinary non-Qwen rows continue ordinary 15% WWM. earlier analysis records the causal change explicitly: labels follow the same sparse policy as earlier analysis (`focus_prob`, `max_focus_groups_per_row`), while masks cover all detected Qwen second-view content groups up to the cap and always include the labelled groups. At seed64 this expanded candidate groups from the selected 21,479 labelled groups to 132,283 masked candidate groups while keeping 28,590 focus targets alongside 592,858 ordinary targets; at seed65 the corresponding total target counts are 28,476 focus and 592,836 ordinary. Dense corruption removes within-view completion evidence in the second view; because the first/source view remains visible in the same row, the sparse focus positions can be explained by source-conditioned relation use rather than by second-view local completion.

### 2b. "Same-window relation structure has causal locality"

**Before:** Split controls (seed43222 threearm probe decomposition) demonstrated that REPEAT's large source-
conditioned cost disappeared when source and companion were moved to different rows.
The ALIGNED/SHUFFLED family (related experiments) showed that correct correspondence installed
usable source-conditioned prediction while wrong correspondence discounted it.

**Design decision:** Dense corruption of the second-view content in packed Qwen-pair
rows, where source and rewrite co-occur in the same local window. This is a principled
choice, not an arbitrary masking pattern: the Stage-II evidence showed that local
co-occurrence under a relationship is what installs the computation.

### 2c. "Dense acquisition damage requires geometry-matched preservation"

**Before:** Earlier dense-corruption acquisition endpoints showed a source-responsive trade relative to coherent86, but the exact `(M,S)` row must be kept separate from the denser-label dense endpoint. In the matched ladder, exact `(M,S)`64 improves Entity over coherent86 by about +1.07 and over O64 by about +1.23; exact `(M,S)`65 improves Entity over coherent86/clean65-family coordinates by about +0.97 and over O65 by about +1.20. At seed64, exact `(M,S)` raises Qwen source specificity by +0.146789 Tfit/+66.97 rank relative to the coherent86 parent, while still damaging BLiMP/Supplement and evidence-absent lexical behavior. The damage pattern motivated preservation as selective retention rather than broad shrinkage.

**Design decision:** Match the coherent86 starting function (not chck82) on ordinary
15% WWM positions in full packed Qwen-pair rows, where source evidence is visible and
the parent's prediction is well-evidenced. This leaves the dense-corruption positions
free for acquisition while constraining ordinary-mask positions near the parent.

### 2d. "Input × corruption geometry determines which computation can move"

**Before:** The prediction was that matching the coherent86 parent
under the dense-corruption view instead of the ordinary-mask view would cause
source-responsive acquisition to collapse.

**After:** the follow-up trained the dense-corruption leash with the same Qwen rows,
same coherent86 teacher, same λ=1, and same 80-update schedule. Measured at seed62064:

| endpoint | Qwen source spec. (ΔTfit) | Qwen source spec. (Δrank) | Dense-common CE gain | Dense-common rank gain |
|---|---:|---:|---:|---:|
| ordinary64 | +0.000075 | −1.24 | −0.005 | +0.70 |
| exact (M,S)64 | +0.146789 | +66.97 | +0.757 | +132.85 |
| clean64 | +0.100746 | +42.70 | +0.682 | +128.34 |
| densecorr64 | +0.003914 | +1.58 | +0.159 | +34.82 |

The dense-corruption leash matched the parent at non-label positions, yet it suppressed
source-following at the label positions where acquisition was credited. This means the
acquired behavior is a shared computation over the corrupted view, not label-local
memorization. The ordinary-mask leash does not pin these evidence-poor positions, so the
acquired source-responsive computation survives. This is the strongest form of
"a computation was installed" the program has.

**Limits:** One seed, λ=1, source-readout level only. Entity was not scored on the
dense-corruption leash endpoint, so its benchmark-face prediction remains unverified.

## 3. Parameter Count Resolution

The research skeleton uses 34,219,200 for the stripped stock `AutoModel`. This is correct:

| class | parameters | adapter | private | source |
|---|---:|---:|---:|---|
| Stock DebertaV2Model (AutoModel) | 34,219,200 | 0 | 0 | stage3 execution synthesis audit |
| Stock DebertaV2ForMaskedLM | 34,467,424 | 0 | 0 | = 34,219,200 + 248,224 MLM head |
| Repaired chck82 encoder | 35,214,784 | 995,584 | 0 | stage3 execution synthesis audit |
| Repaired chck82 MLM | 35,463,008 | 995,584 | 0 | stage3 execution synthesis audit |
| Repaired coherent86 encoder | 36,210,368 | 995,584 | 995,584 | stage3 execution synthesis audit |
| Repaired coherent86 MLM | 36,458,592 | 995,584 | 995,584 | stage3 execution synthesis audit |

The skeleton's claim about the historical SuperGLUE path is about `AutoModel`, so
34,219,200 is the correct number. The MLM scoring path would use 34,467,424 for the
stock class, but MLM scoring (BLiMP, Supplement, etc.) was not affected by the
AutoModel stripping bug because those paths use `AutoModelForMaskedLM` with trusted
remote code. Only the SuperGLUE downstream evaluation used `AutoModel` and thus
silently dropped adapter parameters.

The difference: 248,224 = MLM prediction head (dense + bias + layer norm).

## 4. Figures Produced

All figures saved in `experiments/archive/relation_learning/figures/mechanism`:

1. `fig1_twoseed_component_ladder.png` — Seed64 full 4-rung bar + seed65 Entity replication
2. `fig2_source_specificity_geometry.png` — The central Stage-III mechanism figure
3. `fig3_dense_common_geometry_probe.png` — Geometry prediction confirmation (3-panel)
4. `fig4_tun_relation_reversal.png` — Stage-II three-seed T/U/N relation reversal
5. `fig5_aoa_30m_calibration.png` — AoA timing result with common-subset shift
6. `fig6_overall_ladder.png` — Overall scores horizontal bar chart

## 5. What the Report Should Not Lose

### Failed routes as knowledge

The report must include the failed routes because they are where the principle earned
its discriminating power:

- **State-update replacement** (related experiments): showed that organizing rows around an
  intended state-change operation is insufficient when recency, position, and format
  provide cheaper prediction cues.
- **Descriptor credit** (calibrated route options, 91): the initially favorable binding result
  measured mention-conditioned summary readout, not literal state assignment.
- **480-map literal binding** (related experiments): proved identity-conditioned assignment
  is learnable but transfers family-level priors to official Entity rather than clean
  relational variables.
- **Marginal relation dose** (state update route synthesis): additional aligned-restatement dose above
  the already relation-practicing substrate reached a practical knee at dose25.
- **Format alignment** (related experiments): yielded the geometry law — private branches are
  controlled only on input geometries represented in training/preservation — but did
  not produce a v5 lever.
- **Exposure audits** (related experiments): 35,341 leaked rows and 33,431 inherited blocking
  hits taught that corpus re-admission must be checked before any generalization claim.
- **Loader identity** (related experiments, 99-110): historical stock-AutoModel SuperGLUE
  silently dropped 995,584 adapter parameters; this changed the comparison coordinate.
- **AoA timing** (related experiments): occurrence timing can move acquisition order at 30M,
  but one seed, fewer fitted words, and higher loss make it a bounded scientific result
  rather than a current SOTA lever.

### Frontier positioning (what is new vs. what was known)

Prior work established that data repetition, paraphrase augmentation, curriculum
ordering, and copy/induction circuits affect language model learning. The new claims
from this program are narrower and stronger:

- In-window relation structure under a fixed budget determines which source-conditioned
  computation is installed and where its liabilities travel (not merely "data quality")
- Corruption geometry is a credit lever: making evidence available through dense input
  corruption while restricting gradient to sparse labels changes which positions can
  acquire new computation
- Leash geometry is a selective retention lever: matching the parent function on
  ordinary-mask positions retains source-responsive behavior because it leaves the
  evidence-poor dense-corruption positions free to move

The withdrawn claims (scalar data value, generic context improvement, universal state
tracking, broad commonsense reasoning from GlobalPIQA) define the boundary of what
the method learned to not claim.


## 6. Release Gate Status (Updated twoseed synthesis and provenance)

### Official fine-tuning entry check: PASSED
- The entry check ran `python -m evaluation_pipeline.finetune.run`
  directly from each exported package directory for WSC task
- v4 package: `all_valid = true`, returncode 0, WSC accuracy 61.5385%
- v5 package: `all_valid = true`, returncode 0, WSC accuracy 63.4615%
- These WSC values are export-entry compatibility measurements, not repaired-coordinate Overall components.
- Both packages load `FrozenSlowPrivateDebertaV2Model` (36,210,368 params)
  and `FrozenSlowPrivateDebertaV2ForMaskedLM` (36,458,592 params) from the export paths. The entry probe reports `private_params=995,584` but `slow_params=0` and empty scale arrays because its helper does not separately count the inherited slow adapter. The public documentation should take the scales from `config.json`/modeling code: `adapter_scale=1.75` for the inherited slow path and `private_adapter_scale=0.75` for the private path.

### Mask-token smoke: PASSED
- v4 and v5 both pass tokenizer, MLM load/forward, encoder load/forward,
  and private-adapter gradient smoke
- Mask prediction token: `<mask>` (id=4)
- Source: `experiments/archive/functional_learning/data/export_path_smoke_mask_repaired`

### Model weight provenance: VERIFIED
- v4: SHA256 `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
- v5: SHA256 `1c6268a37a7774a78d725307f99bbaf1f5225b4315f9bfd9ddb2a5b47c680e4a`
- Training, repaired evaluation, and export copies match for both

### Two-seed preservation increment: REPLICATED
- clean64 − MS64 = +0.04387 Overall
- clean65 − MS65 = +0.05286 Overall
- Same sign, similar magnitude

### Remaining evaluation items for full two-seed ladder:
- All O65, MS65, and clean65 components are now complete from earlier analysis admission.
- O65 Overall: 42.1159198215161
- MS65 Overall: 42.17887384024569
- clean65 Overall: 42.23173113265801
