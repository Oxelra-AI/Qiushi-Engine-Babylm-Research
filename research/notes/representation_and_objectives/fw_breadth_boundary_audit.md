# fw breadth boundary audit — FW source-breadth sentence-boundary audit

Current fw comparison mechanical audit source-breadth arm uses 13,836 selected independent FineWeb sentences (318,851 companion words) sliced into 5,785 compact-row companion budgets.

## Boundary damage in current arm

- Rows with any split source sentence: 5,780 / 5,785 (0.999).
- Rows starting mid-source: 5,548; rows ending mid-source: 5,548.
- Source sentences spanning multiple rows: 5,546 / 13,836 (0.401).
- Sources per row: mean 3.351, max 6.

## Repair feasibility clue

Candidate pool after excluding compact-pair sources: 38,790 sentences / 904,011 words. Candidate lengths span 10–48 words. Unbounded length expressibility failures among the 5,785 row companion budgets: 0.

## Scientific consequence

Before launching H100 training, repair the source-breadth arm if exact whole-sentence per-row packing is feasible. The expensive compact-vs-breadth result should not be vulnerable to the objection that the breadth arm was made weaker by arbitrary cross-row sentence slicing.

JSON: `experiments/archive/representation_and_objectives/data/fw_breadth_boundary_audit/fw_breadth_boundary_audit.json`
