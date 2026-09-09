# pvdm compliance and control design — PVDM legality + strict label/control readiness

## Resolved Questions

The PVDM route (corpus directional pair census) required a supervision source before any GPU launch. The compliance concern is that a pretrained spaCy parser imports outside language knowledge and would recreate the compliance failure that invalidated the 42.03 tokenizer coordinate.

### 1. Rule/provenance resolution (compliance)
- Official BabyLM FAQ (`data/external/FAQs.md`, lines 116/128): ancillary language-learned models count toward the 100M budget; explicit example — an off-the-shelf POS tagger cannot be used in the pipeline; "you can train ... a parser ... on parts of the 100M" but the sum of all learned text counts.
- Installed `en_core_web_sm` v3.8.0 meta reports OntoNotes 5, ClearNLP, WordNet sources → external language-learned parser/tagger.
- Decision: the corpus directional pair census spaCy pilot is a non-training feasibility probe only. It is excluded from any training-facing labels.

### 2. Deterministic replacement extractor
- `scripts/pvdm_strict_label_builder.py`: hand-coded lexical/window rules + frequency bins computed only from the allowed compact 10M pool (SHA `ee08a1f8d974...`). No model outputs.
- Pivot classes: physical_change verbs, strong spatial adpositions, temporal/causal connectors, negation, comparatives (than-gated).
- Targets: nearest content dependent in a short deterministic window; excludes speaker tags (`*CHI/*MOT`), annotation artifacts (`[...]`, `%...`, `=`), stopwords, and pivots.
- Strict tail labels: `data/pvdm_strict_labels/` (SHA in summary). Permissive first pass `pvdm_deterministic_labels` is retained only as a documented reject.

### 3. Strict label facts (70M→100M tail of compact stream, SHA `c8d7f24b...`)
- 192,549 rows / 30,000,000 words; eligible rows 184,689 (0.959).
- accepted events 628,344; unique dependent targets 628,344; ~20.94 target positions / 1000 words (~1.4% of words; well under the ~15% effective mask rate, so PVDM can bias without saturating).
- category balance: physical_change 183,300; negation 119,919; causal_connector 110,307; temporal 108,522; spatial 92,955; comparative 13,341.
- 80,538 candidate events dropped for lack of a matched control (kept honest, not forced).

### 4. Matched-control quality (the strategist's second requirement)
- treatment and control share identical dependent-target index sequence: SHA `fd78f6c82db80dcd...` → dependent-target distribution and count are equal by construction.
- anchor class always matched: verbish→verbish 183,300; function→function 431,703; adjectival→adjectival 13,341.
- frequency-bin abs diff 0 for 491,226/628,344 (78.2%), ≤1 for 90.3%.
- distance abs diff 0/1 for 266,205 (42.4%); the remainder are the closest same-class same-freq anchor available in the row.
- The trainer must additionally normalize masking so PVDM and control have equal expected AND realized masked mass per batch (target positions are already identical; only the protected-anchor exclusion differs by one visible token).

## What remains before GPU (implementation spec for the continuation trainer)
Warm start from compact `chck_70M` (`experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_70M`), same shared 16k tokenizer, DeBERTa-v2 8×480, exact remaining 30M words = `compact_tail_70M_100M.jsonl` (192,549 rows), symmetric optimizer reset, checkpoints 80M/90M/100M.

Two masking modes over WWM 0.15 baseline geometry:
- PVDM treatment: force pivot word-group visible (never masked); raise selection weight on dependent-target word-groups; renormalize non-target selection down so realized masked-word count per row equals the WWM-0.15 expectation.
- Matched control: force the surrogate control word-group visible; raise selection weight on the same dependent-target word-groups by the same amount; identical renormalization.
Both keep the standard 80/10/10 mask/random/keep split and standard MLM loss.

Parity gate (must pass before 30M continuation): with PVDM/control weighting disabled the continuation from chck_70M must reproduce plain WWM-0.15 continuation loss trajectory (a few hundred steps), confirming the warm-start + masking machinery is neutral at K=0.

First readout at chck_80M (cheap, no SuperGLUE/AoA/collation): EWoK four-cell on fixed 7,618 rows, GlobalPIQA all-option margins (103 parallel + 100 nonparallel), Entity + Supplement sentinels. Continue to 100M only if EWoK conditional interaction improves on a fixed baseline row set AND GlobalPIQA hard52 rank/margin improves without nonparallel loss AND Entity/Supplement stay near control.

## Files
- strict labels: `data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl`
- continuation tail: `data/pvdm_strict_labels/compact_tail_70M_100M.jsonl`
- strict summary: `data/pvdm_strict_labels/pvdm_strict_label_summary.json`
- quality note: `notes/pvdm_strict_label_quality.md`
- compliance/design note: `notes/pvdm_compliance_and_control_design.md`
