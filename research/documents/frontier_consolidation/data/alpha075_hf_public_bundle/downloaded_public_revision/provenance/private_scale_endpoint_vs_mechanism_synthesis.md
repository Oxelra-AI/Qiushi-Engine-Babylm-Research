# private scale endpoint vs mechanism synthesis private-scale endpoint vs mechanism synthesis

## Research context

The frozen-anchor private-scale line began from the verified 82M scale1.75 endpoint: freeze the public rank-1 protected function and allow only a reversible private residual after additional legal replay. Coherent86 alpha1 is a faithful public-HF candidate, and alpha0.5/alpha0.75 showed higher cheap7 in no-training interpolation screens. The unresolved question is whether this is a real mechanism for protected complementary learning or only endpoint redistribution.

No new model training, evaluation or submission was performed for this analysis. It added CPU/payload-only analyses while the alpha SuperGLUE evaluations and anchor-margin census remained unresolved.

## Durable artifacts produced in private scale endpoint vs mechanism synthesis

1. `scripts/anchor_margin_gate_analysis.py`
   - Consumes `anchor_margin_scored_items.jsonl` from the anchor margin census validation anchor-margin census.
   - Reports per-column activation/damage margin separability, not only pooled COMPS/BLiMP-dominated AUCs.
   - Simulates an optimistic decision-level label-free gate: use an alpha endpoint only when `|anchor margin| <= threshold`, otherwise keep the protected anchor decision.
   - Smoke-tested successfully on `data/anchor_margin_alpha_census_cpu_smoke/anchor_margin_scored_items.jsonl`; output `data/anchor_margin_gate_analysis_cpu_smoke/anchor_margin_gate_analysis.{json,md}`.
   - This script is ready to run on `data/anchor_margin_alpha_census_full/anchor_margin_scored_items.jsonl` once the full census results are available.

2. `scripts/globalpiqa_alpha_microaudit.py`
   - Output: `data/globalpiqa_alpha_microaudit/globalpiqa_alpha_microaudit.{json,md}`.
   - GlobalPIQA has 203 common examples; only 5 change across alpha0/0.5/0.75/1.
   - Alpha0.5 gains exactly 3 net examples = +1.477833 GlobalPIQA score points = +0.211119 cheap7 points.
   - Alpha0.75 gains exactly 2 net examples = +0.985222 score points = +0.140746 cheap7 points.
   - Therefore the GlobalPIQA-heavy alpha0.5 endpoint edge is a few-example endpoint effect, not broad commonsense acquisition.

3. `scripts/relation_state_alpha_audit.py`
   - Output: `data/relation_state_alpha_audit/relation_state_alpha_audit.{json,md}`.
   - EWoK / Entity vs protected chck82:
     - alpha0.5: EWoK score delta -0.059339 despite +10 net items; Entity -0.083845 with -5 net items.
     - alpha0.75: EWoK -0.035837 with +4 net items; Entity +0.008177 with -4 net items.
     - alpha1: EWoK -0.149427 with -2 net items; Entity +0.125905 with +7 net items.
   - The relation/state movement is small and mixed. It is much smaller than COMPS/BLiMP churn and smaller than the apparent GlobalPIQA swing from five items.

## Scientific reading

The no-training alpha interpolation remains useful endpoint engineering: alpha0.5/0.75 reduce anchor erosion relative to alpha1 and increase cheap7. But the mechanism reading has weakened. The alpha-sensitive set is dominated by COMPS and BLiMP, GlobalPIQA movement is only five examples, and the motivating relation/state families show small mixed changes. Unless the full anchor-margin census shows a strong, per-column separability pattern enabling a real label-free gate, the private-scale direction should not be treated as a general data-efficient learning principle.

The practical endpoint question is separate: once the pending SuperGLUE results are available, compute Overall(AoA0) using the earlier analysis official identity and, if an alpha remains above protected chck82, materialize a truthful scalar-AoA/no-fast carrier with `scripts/materialize_truthful_private_scale_carrier.py`.

## Next use of prepared analysis

After the anchor-margin census results become available, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/anchor_margin_gate_analysis.py \
  --census-dir experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_full \
  --out-dir experiments/archive/frontier_consolidation/data/anchor_margin_gate_analysis_full
```

Then read `data/anchor_margin_gate_analysis_full/anchor_margin_gate_analysis.md` before proposing any gated training mechanism. If no threshold has positive cheap7 while preserving nonnegative EWoK and Entity, close anchor-confidence gating as a mechanism and seek a different route for relation/state retention.

## Full Anchor-Margin Census Result

Full census artifacts:
- Raw census: `data/anchor_margin_alpha_census_full/anchor_margin_alpha_census.{json,md}`
- Post-census gate analysis: `data/anchor_margin_gate_analysis_full/anchor_margin_gate_analysis.{json,md}`

The raw signed margin nearly perfectly separates activation from damage, but this is not usable as a label-free mechanism because the sign is defined by the gold answer: monotonic activations are anchor-wrong and therefore have negative gold-vs-best-other margins; monotonic damages are anchor-correct and therefore have positive margins. The usable confidence magnitude `|anchor margin|` does not separate the populations: pooled AUC(damage by low |margin|) is 0.5113; per-column values are BLiMP 0.5084, COMPS 0.5105, EWoK 0.4604, Entity 0.5890, GlobalPIQA 0.3333 on only five changed items, Supplement 0.4606 after excluding four scorer-mismatch rows.

Four Supplement QA-congruence rows had disagreement between the lightweight margin rescore and saved official anchor decisions (6359/6363 agreement). The gate analysis explicitly excludes those four rows and records them in the JSON; they should not be used for high-confidence claims.

Optimistic decision-level gating (use alpha decision only when `|anchor margin| <= threshold`, otherwise keep anchor) does not rescue alpha0.5 or alpha0.75 as a relation/state protection mechanism:
- alpha0.5 best cheap7 gate is threshold 0.35 with +0.2295 cheap7 points, but Entity remains negative (EWoK +13 items / +0.1706 pts; Entity -7 items / -0.1032 pts), so no threshold gives positive cheap7 while keeping both EWoK and Entity nonnegative.
- alpha0.75 best cheap7 gate is threshold 0.2 with +0.1278 cheap7 points, but Entity remains negative (EWoK +11 / +0.1444; Entity -12 / -0.1770); no threshold gives positive cheap7 with nonnegative EWoK and Entity.
- alpha1 has a weak threshold satisfying the relation/state condition at |margin|<=1.0 (+0.0464 cheap7, EWoK +1, Entity +8), but this is smaller than the existing endpoint gains and still has net -120 items overall with COMPS -143 and Supplement -12.

Scientific consequence: anchor-confidence magnitude is not a useful label-free separator for private-residual benefit versus damage. The private-scale line remains a practical endpoint knob pending SuperGLUE/carrier materialization, but it does not currently support a general slow-fast relation/state retention principle. New training such as retention-KL or anchor-confidence-gated fast-path should not be launched merely from this evidence; the next mechanism route should look for a different measurable object than scalar anchor confidence.

## Managed-task status and execution repair

The alpha0.75 SuperGLUE evaluation had not produced a model result. The same SuperGLUE-only endpoint evaluation was retried with separate output roots:
- eval root: `data/private_scale_superglue_eval/coherent86_private_scale_0p75/`
- collate root: `data/private_scale_superglue_collate/coherent86_private_scale_0p75/`
- summary root: `data/private_scale_superglue_summary_alpha0p75/`

The alpha0.5 SuperGLUE evaluation remained in progress. No SuperGLUE result for alpha0.5 or alpha0.75 is inferred in this note.

## Alpha-scale self-consistency headroom

Additional payload-only artifact: `data/alpha_self_consistency_probe/alpha_self_consistency_probe.{json,md}`.

Simple label-free winner rules over alpha0/0.5/0.75/1 have modest six-discrete headroom:
- best single is alpha0.75 at discrete6 +0.2560 vs anchor, with 2,332 gains / 2,418 losses (net -86).
- `anchor_plus_scaled_majority_tie_anchor` and `private_unanimous_else_anchor` both reach discrete6 +0.2546 with fewer changed items (1,553 gains / 1,575 losses; net -22).
- However these rules still lower EWoK/Entity versus anchor (e.g. private-unanimous EWoK 50.0147 vs 50.0555, Entity 28.2302 vs 28.3140) while improving Supplement and GlobalPIQA. This is not a relation/state protection mechanism.

This suggests that a multi-alpha custom inference wrapper has at most endpoint-engineering value and would be post-hoc unless frozen before a fresh evaluation coordinate. It should not displace the main scientific conclusion: the private-scale family currently lacks a general label-free separator of useful residual decisions from damaging ones.
