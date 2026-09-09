#!/usr/bin/env python3
"""Print aggregate markdown tables of strict-track results from results/.

Zero-shot results are read from:
  results/{model}/main/{eval_type}/causal/{task}/{subtask}/best_temperature_report.txt

Finetune results are read from:
  results/{model}/main/finetune/{task}/results.txt
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


from pathlib import Path

# Primary metric per GLUE task, matching the leaderboard definition in about.py.
FINETUNE_METRIC = {
    "boolq":   "accuracy",
    "mnli":    "accuracy",
    "mrpc":    "f1",
    "multirc": "accuracy",
    "qqp":     "f1",
    "rte":     "accuracy",
    "wsc":     "accuracy",
}


def parse_average_score(report_path: Path) -> float | None:
    """Return the average score from a best_temperature_report.txt file.

    Handles both '### AVERAGE ACCURACY' and '### AVERAGE SPEARMAN'S RHO' headers.
    """
    lines = report_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("### AVERAGE"):
            if i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
    return None


def parse_finetune_score(results_path: Path, metric: str) -> float | None:
    """Return the value of `metric` from a finetune results.txt file."""
    for line in results_path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == metric:
            try:
                return float(value.strip()) * 100
            except ValueError:
                return None
    return None


def fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def print_table(
    all_models: dict[str, dict[str, float]],
    row_order: list[str],
    models: list[str],
) -> None:
    print(fmt_row(["task"] + models))
    print(fmt_row(["---"] + ["---"] * len(models)))
    for row_key in row_order:
        vals = [all_models[model].get(row_key) for model in models]
        best = max((v for v in vals if v is not None), default=None)
        row = [row_key]
        for val in vals:
            if val is None:
                row.append("")
            elif val == best:
                row.append(f"**{val:.2f}**")
            else:
                row.append(f"{val:.2f}")
        print(fmt_row(row))


def main():
    results_dir = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/results')

    # --- Zero-shot ---
    zeroshot_models: dict[str, dict[str, float]] = {}
    zeroshot_row_order: list[str] = []

    for report in sorted(results_dir.glob("*/main/*/causal/*/*/best_temperature_report.txt")):
        # parts: results / model / main / eval_type / causal / task / subtask / filename
        parts = report.parts
        model     = parts[-7]
        eval_type = parts[-5]
        task      = parts[-3]
        subtask   = parts[-2]

        score = parse_average_score(report)
        if score is None:
            continue

        row_key = f"{eval_type}/{task}/{subtask}"
        if row_key not in zeroshot_row_order:
            zeroshot_row_order.append(row_key)
        zeroshot_models.setdefault(model, {})[row_key] = score

    # --- Finetune ---
    finetune_models: dict[str, dict[str, float]] = {}
    finetune_row_order: list[str] = []

    for results_file in sorted(results_dir.glob("*/main/finetune/*/results.txt")):
        # parts: results / model / main / finetune / task / filename
        parts = results_file.parts
        model = parts[-5]
        task  = parts[-2]

        metric = FINETUNE_METRIC.get(task, "accuracy")
        score = parse_finetune_score(results_file, metric)
        if score is None:
            continue

        row_key = f"{task} ({metric})"
        if row_key not in finetune_row_order:
            finetune_row_order.append(row_key)
        finetune_models.setdefault(model, {})[row_key] = score

    if not zeroshot_models and not finetune_models:
        print("No results found.")
        return

    all_models = sorted(set(list(zeroshot_models) + list(finetune_models)))

    if zeroshot_models:
        print("## Zero-shot\n")
        print_table(zeroshot_models, zeroshot_row_order, all_models)

    if finetune_models:
        print("\n## Finetune (GLUE)\n")
        print_table(finetune_models, finetune_row_order, all_models)


if __name__ == "__main__":
    main()
