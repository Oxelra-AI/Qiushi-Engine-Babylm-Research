# semantic repair frontier and refill plan — semantic-force repair frontier and refill plan

## Current research role of this work

The frozen current-official endpoint remains `compact_view_reinvest` seed43022 at Overall 42.0331347900748. The unresolved issue is not whether this single checkpoint is above the visible 41.8 leader, but whether the density principle can be made stable enough to support a stronger reproducible route. The exact clean-Qwen sparse temporal control was still running when this analysis began; no result was inferred from unfinished or absent files.

This analysis therefore used only existing generated-row and selected-corpus metadata to prepare the lowest-cost semantic-repair fork. The work answers: if the exact temporal treatment comparison later points to treatment-specific Supplement/EWoK fragility caused by assertion-force or relation damage in compact views, can a safer compact block be built without new teacher generation while preserving the 423.5k-word changed-block scale?

## CPU results produced

1. Ran the earlier analysis refill feasibility script:
   - Script: `experiments/archive/frontier_consolidation/scripts/filter_refill_feasibility.py`
   - Output: `experiments/archive/frontier_consolidation/data/filter_refill_feasibility/filter_refill_feasibility.json`
   - Note: `research/documents/frontier_consolidation/data/filter_refill_feasibility/filter_refill_feasibility.md`
   - Accepted medium compact rows: 18,682; selected reinvest rows: 12,155 / 423,511 pair words; unused accepted rows: 6,527 / 221,722 pair words.
   - `semantic_force_minimal` removes 113,387 pair words and has 156,174 unused passing pair words (available:needed 1.38).
   - `semantic_force_plus_pronoun` removes 124,339 pair words and has 148,447 unused passing pair words (available:needed 1.19).
   - Content floors are not refillable from the unused accepted pool: even force + content >=0.45 has available:needed 0.94; content >=0.60 has 0.38.

2. Built and ran the broader rule frontier:
   - Script: `experiments/archive/frontier_consolidation/scripts/semantic_repair_frontier.py`
   - Output: `experiments/archive/frontier_consolidation/data/semantic_repair_frontier/semantic_repair_frontier.json`
   - Table: `experiments/archive/frontier_consolidation/data/semantic_repair_frontier/semantic_repair_frontier.csv`
   - Note: `research/documents/frontier_consolidation/data/semantic_repair_frontier/semantic_repair_frontier.md`
   - Refillable total-word rules: `force_only`, `force_plus_pronoun`, and `force_surface_content_ge_none`.
   - All rules adding content-recall floors were non-refillable with the existing unused accepted pool. This is important because broad content thresholds would turn the repair into a different, much less compact data regime rather than preserving the rate--distortion idea.

3. Built and ran a domain-aware replacement planner:
   - Script: `experiments/archive/frontier_consolidation/scripts/domain_aware_refill_plan.py`
   - Output: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/domain_aware_refill_plan.json`
   - Note: `research/documents/frontier_consolidation/data/domain_aware_refill_plan/domain_aware_refill_plan.md`
   - Replacement row lists:
     - `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_only_replacement_rows.csv`
     - `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_plus_pronoun_replacement_rows.csv`
     - `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_surface_replacement_rows.csv`
     - `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_content_ge_0p45_replacement_rows.csv`
   - `force_only`: removes 113,387 pair words, chooses 113,394 replacement words, final changed block +7 words, same-domain shortfall before global fill 1,055 words.
   - `force_plus_pronoun`: removes 124,339 pair words, chooses 124,340 replacement words, final changed block +1 word, same-domain shortfall 3,602 words.
   - `force_surface`: removes 121,567 pair words, chooses 121,572 replacement words, final changed block +5 words, same-domain shortfall 2,919 words.
   - `force_content_ge_0p45`: cannot refill: final block -7,145 words and same-domain shortfall 13,744 words.

## Scientific interpretation

The low-cost corpus-repair fork is real but conditional. If later treatment evidence selects semantic-view damage as a load-bearing cause of seed43122 weakness, the strongest repair is not a broad content-recall threshold. It is a targeted assertion-force repair: remove rows with question-force flips, lost modals/hedges, lost attribution, lost or gained negation, and new causal markers not present in the source; optionally remove unresolved starting-pronoun rows and entity/number surface losses. These rules are mechanically refillable from already generated unused compact rows with almost exact changed-block word accounting.

Domain preservation matters. Even refillable rules have small primary-domain shortfalls in science_physical, quant_numeric, or media_culture before global fill. Because the current EWoK weakness involves physical/spatial/material relations, any future repair corpus should protect science_physical and causal_relational exposure instead of filling solely by a scalar quality rank.

This evidence does not justify a new training run. The next decisive evidence is still the exact clean sparse temporal comparison and the missing full official seed43122 vector. If those results do not show treatment-specific semantic fragility, this repair plan should remain unused; the route should turn toward optimization, initialization, or consolidation dynamics instead.
