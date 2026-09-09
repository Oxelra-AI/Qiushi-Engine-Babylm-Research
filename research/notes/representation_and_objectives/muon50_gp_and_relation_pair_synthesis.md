# muon50 gp and relation pair synthesis — 50M Muon-switch GlobalPIQA and active relation-pair feasibility

## Scientific Motivation changed the immediate evidence target

The muon switch 40m globalpiqa margin 40M comparison was not sufficient to decide the optimizer-consolidation route. `muon40toadamw` at `chck_40M` is exactly the switch boundary, not an AdamW-recovered checkpoint, and both switch runs already expose `chck_50M`. muon50 gp and relation pair synthesis therefore prepared the deepest common existing-weight 50M comparison before any new training.

Scripts prepared and checked:

- broad cheap-column wrapper: `experiments/archive/representation_and_objectives/scripts/muon_switch_50m_eval.py`
- GlobalPIQA hard-rank wrapper: `experiments/archive/representation_and_objectives/scripts/muon_switch_50m_globalpiqa_wrapper.py`
- EWoK four-cell wrapper: `experiments/archive/representation_and_objectives/scripts/muon_switch_50m_ewok_wrapper.py`

GlobalPIQA 50M scoring completed on CPU. Broad 50M scoring was not launched, the 40M broad table remains pending, and EWoK 50M is prepared but not yet evaluated.

## 50M GlobalPIQA hard-rank evidence

Source files:

- `experiments/archive/representation_and_objectives/data/muon_switch_50m_globalpiqa_margin/globalpiqa_margin_reader_results.json`
- `research/notes/representation_and_objectives/muon_switch_50m_globalpiqa_margin.md`

| arm | GP-parallel | GP-nonparallel | hard52 acc | hard52 mean top-minus-correct | interpretation |
|---|---:|---:|---:|---:|---|
| AdamW50 | 28.16 | 45.00 | 3.85 | 1.762 | legal compact AdamW reference improves parallel vs AdamW40 but loses nonparallel |
| continuous Muon50 | 27.18 | 48.00 | 5.77 | 1.849 | recovers some parallel from Muon40 but still below AdamW50 and margins remain deep |
| Muon20→AdamW50 | 26.21 | 51.00 | 5.77 | 1.699 | retains nonparallel breadth relative to AdamW50 but is worse on parallel; hard rows mostly still wrong |
| Muon40→AdamW50 | 23.30 | 45.00 | 7.69 | 1.680 | after ~10M AdamW recovery it remains poor on parallel and nonparallel is not improved |

Comparison with muon switch 40m globalpiqa margin 40M GlobalPIQA:

- `Muon20→AdamW` moved from 20.39/56.0 parallel/nonparallel at 40M to 26.21/51.0 at 50M. It recovered parallel but gave back nonparallel; it remains worse than AdamW50 on parallel by 1.94 points and only ties/equals AdamW40's 26.21 parallel.
- `Muon40→AdamW` moved from 22.33/45.0 at boundary to 23.30/45.0 after 50M. That is not meaningful repair of the conditional hard-rank weakness.
- All arms remain deeply wrong on the 52 cross-endpoint hard rows: even the best hard52 accuracy here is 7.69% (4/52), and mean top-minus-correct remains 1.68–1.85 nats.

Scientific reading: these existing 50M weights do **not** show that abrupt empty-moment Muon→AdamW handoff solves GlobalPIQA_parallel or the hard52 conditional-ranking problem. The only attractive signal is `Muon20→AdamW50` retaining higher nonparallel than AdamW50, but this is precisely the old tradeoff direction unless broad and EWoK surfaces say otherwise. This does not close every possible Muon-to-AdamW consolidation scheme; it only weakens the already-created abrupt handoff artifacts.

## Relation-pair feasibility: corrected active-token funnel

The first structural pair-funnel pilot (`relation_pair_funnel.py`) produced a large pair count, but I invalidated it as route evidence because it paired raw pvdm compliance and control design text events without verifying that each target group survives the actual 256-token training view. A full managed pass from that script (`s127_t19_tool1`) failed after partial progress and should not be used.

The repaired script is:

- `experiments/archive/representation_and_objectives/scripts/relation_active_pair_funnel.py`

It uses `MaskedChunkDataset` and `pvdm_masking_lib.collect_active_events`, the same active word-group mapping used by the PVDM trainers. Every retained event has real forwarded-token target positions and token ids from the 256-token training view.

Full active-token funnel output:

- `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_funnel_summary.json`
- `experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_pool.jsonl`
- `research/notes/representation_and_objectives/relation_active_pair_funnel.md`

Key facts:

- Segment: tail rows 255–64254, 64,000 rows / 9,971,289 words, matching the 70M→80M continuation segment.
- Active-token/control/token/leak filters kept 50,865 events.
- Greedy pair construction formed 18,651 two-context/two-target pairs from 37,302 events; event-use fraction 0.7334.
- Pair families: causal_connector 5,340; spatial 4,585; temporal 3,826; negation 3,538; physical_change 1,285; comparative 77.
- Matching levels: 17,465 exact strict-stratum pairs (level 0) and 1,186 one-relaxed-level pairs (level 1); pair-cost median 0, p95 2.1.
- Ordinary WWM both-target natural masking is estimated at ~419.6 pairs/epoch and ~4,196.5 over 10 epochs, but exact mask replay is still needed.

Scientific reading: the active legal corpus does contain a nontrivial structural pair pool for a two-context/two-target interaction calibration. It is large enough for no-update scoring and destructive controls in the main families, but comparative remains sparse (77 pairs) and the pair pool is not yet a training target: cross-cell plausibility has not been proven, frequent targets still hit the cap, and exact natural-mask realized inclusion has not been replayed. The next relation action should be a no-update four-cell margin calibration on this frozen active pair pool, including both row margins (`m_a`, `m_b`), target/context swap sign tests, matched permutation controls, pivot deletion/window controls if feasible, and a comparison across standard80/FW100 checkpoints to test saturation.

## Current route interpretation

- Optimizer route: GP50 alone weakens the abrupt Muon→AdamW artifacts but does not fully decide them. The pending 40M broad harvest and 50M broad/EWoK surfaces are needed before extending or closing these exact runs. If broad/EWoK show the same nonparallel-for-conditional tradeoff, stop the abrupt empty-moment handoff; any future optimizer route must repair moment continuity or blend updates before spending training.
- Relation route: active-token pair feasibility is real enough to proceed to calibration, but only as a calibration/probe, not as a training launch. The object to test is conditional binding with main effects cancelled, not local attested-target likelihood.

No new training, official full evaluation, or compliant SOTA endpoint was produced in muon50 gp and relation pair synthesis.
