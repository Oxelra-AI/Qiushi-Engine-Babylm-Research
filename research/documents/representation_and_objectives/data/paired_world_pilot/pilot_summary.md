# paired world pilot result — Role-Equivariant Paired-World Pilot

**Total families**: 500 (400 train, 100 held)  
**Sports**: {'tennis': 350, 'football': 50, 'badminton': 100}  
**Unique participants**: 841  
**A-wins-in-C1 fraction**: 0.4620 (target ~0.50)  
**Label-consistency errors**: 0  
**Total labeled items**: 3000 (2000 four-cell + 1000 invariant)  

## Template Balance

First-mention distribution across all contexts: {'m': 322, 'l': 350, 'w': 328}  
(w = winner-first, l = loser-first, m = mixed)  

## BoW Shortcut Test

- Group-CV accuracy: **0.525** ± 0.0385  
- Stratified-CV accuracy: 0.275 ± 0.0292  
- Chance level: 0.50  
- Shortcut detected: **False**  
- Within-family Jaccard: mean 0.3611, range [0.1613, 0.8095]  

## Sample Context (score-ablated)

**C1**: During the round of 16 at Trnava CH on September 16, 2013, Aljaz Bedene prevailed against Kyle Edmund.  
**C2**: The round of 32 of Vercelli CH on April 21, 2014 ended with Kyle Edmund victorious over Aljaz Bedene.  
**Invariant**: Both Aljaz Bedene and Kyle Edmund are professional tennis players.  
**Four cells**: `{'defeated_ab_c1': True, 'defeated_ba_c1': False, 'defeated_ab_c2': False, 'defeated_ba_c2': True}`  

## Files

- `train`: `experiments/archive/representation_and_objectives/data/paired_world_pilot/families_train.jsonl`  
- `held`: `experiments/archive/representation_and_objectives/data/paired_world_pilot/families_held.jsonl`  
- `shortcut`: `experiments/archive/representation_and_objectives/data/paired_world_pilot/shortcut_test_result.json`  
- `summary`: `experiments/archive/representation_and_objectives/data/paired_world_pilot/pilot_summary.json`  
- `summary_md`: `research/documents/representation_and_objectives/data/paired_world_pilot/pilot_summary.md`  
- `teacher_check_design`: `research/documents/representation_and_objectives/data/paired_world_pilot/teacher_check_design.md`  
