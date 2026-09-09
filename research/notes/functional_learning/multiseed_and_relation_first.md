# multiseed and relation first multi-seed template result and relation-first construction

## Multi-seed template robustness: definitive closure

The 5-seed comparison is complete. All 10 runs (5 seeds × 2 arms) 
converge to exactly **3/12 held flips**. The paired-context term has zero effect
on held joint correctness.

| Seed  | CE held β | Comb held β | CE held R  | Comb held R |
|-------|-----------|-------------|------------|-------------|
| 43033 | +2.104    | +2.174      | -4.267     | -4.125      |
| 43034 | +1.954    | +1.979      | -4.788     | -4.755      |
| 43035 | +1.851    | +1.973      | -5.215     | -4.998      |
| 43036 | +1.977    | +2.099      | -4.743     | -4.516      |
| 43037 | +1.829    | +1.958      | -5.092     | -4.784      |
| **Mean** | **1.943** | **2.037** | **-4.821** | **-4.636** |

CE-only mean held flips: 3.0/12 (all seeds identical)
Combined mean held flips: 3.0/12 (all seeds identical)

Held |alpha| is 6.3-7.1 in all runs, so gamma is deeply negative.
The 3/12 ceiling is a stable template attractor that neither CE nor the paired
term can break. The +0.094 mean beta gain is negligible relative to |alpha|~6.7.

**Conclusion:** On the template substrate, the paired-context objective is closed.
The template task itself is limited: a small fixed entity vocabulary in a rigid
frame produces a degenerate held-out geometry where shared preference dominates.
The template served its purpose: establishing the beta/alpha/gamma decomposition
and showing that ordinary CE already rewards the correct geometry. It cannot
demonstrate or reject transferable recipient-sensitive selection because its
structure is too constrained.

## Relation-first construction: the semantic repair

### Why entity-first failed

The earlier state-update pipeline and the curated source-grounded probe both started from 
entity mentions and tried to invent a shared mutable attribute. This produced:
- McAfee/Intel: successive names of one product, not two independent entities
- Military example: incompatible value types (positions vs. bombers vs. artillery)
- Production example: grammatically different roles forced into nominal slots
- earlier analysis pilot: only 55/512 accepted, mean gamma=-3.33, joint 3/55

The fundamental error: the relation must be **independently grounded** before
the update/retain context is meaningful. Starting from entities and inventing
relations produces ambiguous or self-certifying packets.

### Relation-first principle

Reverse the dependency:
1. Start from an actually supported relation with compatible value types
2. Find two independently addressable instances in BabyLM source text
3. Compose source context from actual BabyLM sentences (real text, not generated)
4. Add counterfactual update as labeled augmentation
5. Query tests whether the model tracks which entity was affected

### Current extraction: 120 pairs from regex patterns

Extracted 229 explicit triples from Simple Wikipedia source sentences:
- death_place: 172 (explicit "X died in Y" patterns)
- birthplace: 42 (explicit "X was born in Y" patterns)  
- located_in: 7
- founded_year: 8

After pair matching (different values, no cross-sentence entity/value leakage):
- 120 valid pairs (90 train, 30 held)
- All values verified as exact raw-source spans
- Update sentences labeled as controlled counterfactual augmentation
- Replacement values from curated city/year pool

### Next: LLM-assisted extraction for scale

Regex-only captures narrow patterns. For the BabyLM bridge I need 200-500 pairs.
Plan: use Qwen to extract broader (entity, relation, value) triples from all 
simple_wiki sentences, with validation that value is an exact source substring
and relation type is a recognized mutable attribute.

### Key files
- Data builder: `scripts/relation_first_constructor.py`
- Pairs: `data/relation_first_packets/relation_first_pairs.jsonl`
- Scoring rows: `data/relation_first_packets/scoring_rows.jsonl`
- Scoring results: pending at the time

## Coherent86 baseline on relation-first pairs (step039b robust scorer)

Scored 120 pairs (720 rows), 0 NaN. Zero recipient dependence confirmed:

| Split | n | mean U | mean R | mean β | mean |α| | mean γ | UPDATE | RETAIN | Joint |
|-------|---|--------|--------|--------|----------|--------|--------|--------|-------|
| All   |120| +1.475 | -1.476 | -0.000 | 1.982    | -1.983 | 41/120 | 13/120 | 0/120 |
| Train | 90| +1.359 | -1.351 | +0.004 | 1.876    | -1.872 | 29/90  | 9/90   | 0/90  |
| Held  | 30| +1.826 | -1.852 | -0.013 | 2.300    | -2.313 | 12/30  | 4/30   | 0/30  |

Comparison with the earlier state-update pilot (55 noisy pairs):
- Noisy state-update pilot: β=-0.009, |α|=3.324, γ=-3.333, joint=3/55
- Relation-first pilot: β=-0.000, |α|=1.982, γ=-1.983, joint=0/120

Both substrates show zero recipient dependence. The relation-first pairs have
smaller |α| because replacement cities are not present in the source context.
Only 3/120 pairs have γ>0 and those are marginal (max 0.218).

## Answer-only training pilot (in progress at the time)

Training coherent86 on 90 train pairs (360 UPDATE/RETAIN rows) with answer-only
masking. Evaluating on 30 held pairs every 50 epochs for 500 epochs.
Script: scripts/revision_039d_answer_only_training.py
Output: data/answer_only_training/

## Figures
- figures/relation_first_alpha_beta.png
- figures/relation_first_U_R_distribution.png
