# density eval repair density no-AoA evaluation repair

The first near-pair no-AoA evaluation failed before producing scores. This failure is not BabyLM evidence about the trained models.

## What failed

The wrapper `experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py` originally constructed subprocess commands using `pathlib.Path(sys.executable).resolve()`. This selected an interpreter without the required package environment. All official evaluation subprocesses therefore failed immediately with:

`ModuleNotFoundError: No module named 'transformers'`

## Repair

The evaluator was changed to keep `PYTHON_EXE = sys.executable` so subprocesses use the same Python environment, and to write the Markdown evaluation note inside `out_root` as `density_noaoa_eval_note.md`.

Syntax check after repair succeeded:

`python -B -c "import ast,pathlib; ast.parse(pathlib.Path('experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py').read_text()); print('SYNTAX_OK')"`

## Retry

A repaired retry was submitted:

`python -B experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py --targets near_repeat near_view --gpu 0 --work_tag near_pair_noaoa_retry --out-root experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry --force`

Expected output:

`experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json`

The active scientific question remains `near_view_minus_near_repeat` over BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading. The failed first attempt only diagnosed infrastructure.
