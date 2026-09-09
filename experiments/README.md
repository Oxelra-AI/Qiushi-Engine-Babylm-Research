# Experiments

Experiments are named by scientific question, with shared numerical results in `results/`.

| Experiment family | Methods | Included materials |
| --- | --- | --- |
| First-generation model construction | Compact restatements, joint pretraining and incremental continuation | Model implementation, published training description, budget and late-stage controls |
| Relation organization | Repetition/restatement, shared/split windows and different target relations | Core results and detailed methods |
| Functional access | Familiar/unseen queries, supervision allocation and internal signal interventions | Shared mechanism tables and intervention interpretations |
| Second-generation model improvement | Input masking, supervised targets and ordinary-input preservation | [Configuration record](configs/stage3.json), full nine-component controls and both model generations |

The [experiment index](index.csv) connects methods, results and report chapters without storing a separate set of scores.

For the rationale behind experimental designs, see the [original plans](../research/plans/README.md) and [research notes](../research/notes/README.md). [Selected reading paths](../research/reading_paths.md) connect major controls with subsequent corrections. The [topic material catalog](../research/materials.md) locates files by scientific question without requiring a directory-by-directory search through historical outputs.

Original trainers, data constructors, mechanism experiments, historical configurations and results are retained in `archive/`. Research notes, experiment plans and technical descriptions are maintained in `research/`, separately from program outputs; corresponding read/write paths and material references were updated during relocation.

The [core program entry points](CORE_PROGRAMS.md) connect both model generations and the main mechanism experiments. The [training and data guide](../reproducibility/TRAINING.md) records actual parameters, construction chains and conditions of use. The [full program index](entrypoints.json) preserves dependencies and file references among original implementations; the [executed-configuration index](configs/executed_stage3.json) locates original records for the major Stage III controls.

`configs/stage3.json` summarizes research conditions; it is not a one-command trainer. Source organization does not establish that training has been rerun. The [material manifest](../evidence/materials_manifest.json) records historical file status, summaries and hashes. Default output locations in archived code reflect the historical layout; specify new output directories when reusing it.

Usable model implementations are available in [models/frontier](../models/frontier/modeling_frozen_slow_private_debertav2.py) and [models/principle_guided](../models/principle_guided/modeling_frozen_slow_private_debertav2.py). Rebuilding existing numerical comparisons and reports does not require retraining.
