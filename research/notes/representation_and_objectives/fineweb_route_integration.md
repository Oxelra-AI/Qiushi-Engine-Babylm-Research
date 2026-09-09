# FineWeb source breadth and generated-view utility

This historical note compares two different interventions: adding source content
and replacing source repetition with generated views. Unfinished comparisons
below are experimental proposals, not measured results.

## Generated-view comparison

The FineWeb core-fact comparison tested the **generated-view value** mechanism at the inherited
clean-Qwen 41.3443 base, using a small row-holdout overlay (changed block =
**423,520 words**, ~4.2% of the 10M corpus; inherited clean-Qwen pair rows preserved;
one coherent official/non-Qwen slice held out and replaced).

Near-length pair result (`near_view` − `near_repeat`), no-AoA task-family screen:

| target      | BLiMP | Supp  | EWoK  | Entity full | COMPS | GPIQA  | Reading | equal7 fast | equal7 fullEnt |
|-------------|------:|------:|------:|------------:|------:|-------:|--------:|------------:|---------------:|
| near_repeat | 67.190| 62.400| 49.450| 23.560      | 51.780| 34.150 | 7.745   | 42.281      | 42.325         |
| near_view   | 66.820| 63.200| 49.180| 27.130      | 51.170| 34.135 | 8.520   | 42.999      | 42.879         |

Contrast `near_view − near_repeat`: **Entity full +3.57**, Entity fast +4.72,
Reading +0.775, Supplement +0.80, equal7 fast +0.719, equal7 fullEnt +0.554;
BLiMP -0.37, EWoK -0.27, COMPS -0.61, GlobalPIQA ~flat. Training losses tied
(2.4082 vs 2.4065), so this is not a simple MLM-loss effect.

Scientific reading: FineWeb same-source generated near-views transfer to **Entity
Tracking**, one of the reference model's largest deficits in the historical
public comparison (~-10.2 Entity, ~-6.56 EWoK). The Entity gain is consistent
with the separate SimpleWiki semantic-view comparison (+2.10 full-eval),
but neither comparison establishes a general representation mechanism.

## Source-breadth comparison (result pending in this note)

The seqsafe96 cached FineWeb contrast uses:
- FineWeb block = **1,753,280 words** (~17.5% of corpus), about four times the generated-view comparison's changed block.
- Treatment = clean-Qwen pairs + FineWeb sources (source repetition, no views) + identical official tail.
- Control = clean-Qwen pairs + identical official tail (no FineWeb) + matched row lengths.

This isolates **broad-source content value at larger scale**, i.e. the B−A
(source breadth) term, whereas the near/compact contrasts isolate the C−B
(rewrite/view utility) term at small scale. The two runs are complementary along
the source-breadth × rewrite-utility decomposition.

## Planned interpretation of the source-breadth result

A small positive effect at 17.5% corpus fraction would leave source scale and
quality unresolved. The historical decision criteria were:

- **Strong positive on Entity/EWoK/COMPS/GlobalPIQA (e.g. summed knowledge-cluster
  gain ≳ +2 with protected Supplement/Reading/SuperGLUE):** source breadth is a
  real lever; next test source breadth at a materially larger corpus fraction
  (e.g. 30–50%) and compare generated views in a joint arm.
- **Small positive (< +1 summed knowledge cluster):** likely under-scaled; the
  candidate lever is view/rewrite utility plus larger source fraction,
  not raw repetition of 17.5% FineWeb.
- **Flat/negative:** cached-sample FineWeb source repetition alone is not the
  lever; pivot to combined source+view at scale, or to a non-data mechanism.

## Strongest combined next experiment (candidate)

Given the near_view Entity +3.57 and the larger-scale source-breadth test, the
highest-value joint arm is: **larger FineWeb source fraction (≥25%) paired with
faithful near-length generated views**, versus (a) same fraction source-repeat and
(b) protected clean-Qwen slice control. This directly stacks the two confirmed/
tested levers against the leader's full-10M FineWeb factual-pair phenotype, while
protecting the reference model's stronger columns (Supplement +6.83, Reading +2.34).

## Files
- Generated-view result: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json`
- Overlay: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json`
- Source-breadth candidate: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate`
- Pending result: `experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json`
