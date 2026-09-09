#!/usr/bin/env python3
"""research CPU preflight for seed43122 robustness decision.

This script does not read partial outputs from the running seed43122 full evaluation.
It records, from already-completed code/evidence, why the inherited full wrapper will
need official-coordinate replacement of EWoK and AoA, fills in known nonfinal ledger
values from research/research, and writes an exact post-delivery command plan for
placing seed43122 on the same coordinate as seed43022.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WORKSPACE = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WORKSPACE / "data/seed43122_robustness_decision_preflight"
NOTE = WORKSPACE / "notes/seed43122_robustness_decision_preflight.md"

PATHS = {
    "seed43022_official_summary": WORKSPACE / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json",
    "seed_variance_repaired": WORKSPACE / "data/seed_variance_fast_and_dynamics_repaired/seed_variance_fast_and_dynamics.json",
    "addendum": WORKSPACE / "data/compact_provenance_and_overlap_addendum/compact_provenance_and_overlap_addendum.json",
    "ledger": WORKSPACE / "data/submission_readiness_compliance_ledger/submission_readiness_compliance_ledger.json",
    "seed43122_full_wrapper": WORKSPACE / "training/scripts/full_eval_seed43122.py",
    "inherited_full_runner": ROOT / "experiments/archive/compact_experience/scripts/full_overall_eval_runner.py",
    "compact_experience_aoa_helper": ROOT / "experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py",
    "ewok_repair_wrapper": WORKSPACE / "training/scripts/official_ewok_reeval_seed43122.py",
    "aoa_repair_wrapper": WORKSPACE / "training/scripts/official_aoa_min0_seed43122.py",
    "repair_controller": WORKSPACE / "training/scripts/seed43122_official_coordinate_repairs_controller.py",
    "staging_script": WORKSPACE / "scripts/stage_pristine_collate_seed43122.py",
    "dryrun_aoa": WORKSPACE / "data/official_aoa_min0_seed43122/official_aoa_min0_seed43122_dryrun.json",
    "repair_dryrun": WORKSPACE / "data/seed43122_official_coordinate_repairs/seed43122_official_coordinate_repairs_summary.json",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def ast_literal_names(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        names[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return names


def find_lines(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        if any(p in line for p in patterns):
            out.append({"line": i, "text": line})
    return out


def extract_runner_facts() -> dict[str, Any]:
    full_wrapper = PATHS["seed43122_full_wrapper"].read_text(encoding="utf-8")
    inherited = PATHS["inherited_full_runner"].read_text(encoding="utf-8")
    helper = PATHS["compact_experience_aoa_helper"].read_text(encoding="utf-8")
    return {
        "seed43122_wrapper_imports_compact_experience_runner": "import full_overall_eval_runner as base" in full_wrapper,
        "seed43122_wrapper_overrides_workspace_or_strict": bool(re.search(r"base\.(WORKSPACE|STRICT)\s*=", full_wrapper)),
        "seed43122_wrapper_overrides_outroot_targets_only": all(x in full_wrapper for x in ["base.OUT_ROOT", "base.PER_TARGET_DIR", "base.TARGETS"]),
        "inherited_strict_points_to_initial_model_studies": 'ROOT_INITIAL_MODEL_STUDIES / "repos" / "babylm-eval" / "strict"' in inherited,
        "inherited_zero_shot_ewok_data_path": "evaluation_data/full_eval/ewok_filtered" if '"data_path": "evaluation_data/full_eval/ewok_filtered"' in inherited else None,
        "compact_experience_aoa_helper_uses_min_context20": "load_eval(word_path, 20, False)" in helper,
        "relevant_lines": {
            "seed43122_wrapper": find_lines(PATHS["seed43122_full_wrapper"], ["COMPACT_EXPERIENCE_SCRIPTS", "import full_overall_eval_runner", "base.OUT_ROOT", "base.TARGETS"]),
            "inherited_full_runner": find_lines(PATHS["inherited_full_runner"], ["STRICT =", "ewok_filtered", "local_runner =", "aoa", "ROOT_INITIAL_MODEL_STUDIES"]),
            "compact_experience_aoa_helper": find_lines(PATHS["compact_experience_aoa_helper"], ["load_eval(word_path, 20, False)"]),
        },
        "coordinate_consequence": "The pending inherited full task should be treated as authoritative for non-EWoK zero-shot, Reading, and SuperGLUE outputs, but its EWoK and AoA must be replaced before seed43122 robustness is judged on the seed43022 official coordinate.",
    }


def planned_staging_command(ewok_pred: str, aoa_dir: str) -> str:
    return (
        "python -B experiments/archive/representation_and_objectives/scripts/stage_pristine_collate_seed43122.py "
        f"--pristine-ewok-predictions {ewok_pred} --aoa-dir {aoa_dir}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    seed430 = load_json(PATHS["seed43022_official_summary"])
    sv = load_json(PATHS["seed_variance_repaired"])
    add41 = load_json(PATHS["addendum"])
    ledger42 = load_json(PATHS["ledger"])
    dry_aoa = load_json(PATHS["dryrun_aoa"])
    repair_dry = load_json(PATHS["repair_dryrun"])

    scores430 = safe_get(seed430, "score_summary.official_overall.scores", {})
    overall430 = safe_get(seed430, "score_summary.official_overall.Overall")
    sv_fast = safe_get(sv, "fast_endpoint_delta_431_minus_430", {})
    sv_recipe = safe_get(sv, "recipe_and_order", {})
    overlap = safe_get(add41, "overlap_ancestry.recomputed_from_rows", {})
    reinvest_summary = safe_get(add41, "source_and_training_lineage.source_chain", [])
    changed_block_words = None
    for entry in reinvest_summary:
        if isinstance(entry, dict):
            summary = entry.get("compact_reinvest_summary") or safe_get(entry, "selection", {})
            if isinstance(summary, dict) and summary.get("changed_block_budget_words") is not None:
                changed_block_words = summary.get("changed_block_budget_words")
                break

    repair_expectations = {
        "official_ewok_required_rows": 7618,
        "official_aoa_required_rows_per_checkpoint": 8005,
        "aoa_dryrun_min_context0": safe_get(dry_aoa, "min_context0_loaded", {}),
        "aoa_dryrun_min_context20": safe_get(dry_aoa, "min_context20_loaded", {}),
        "repair_controller_preflight": safe_get(repair_dry, "preflight", {}),
        "repair_task_ref": "s43_t12_tool1",
        "pending_full_task_ref": "s37_t46_tool1",
    }
    post_delivery = {
        "do_not_use_partial_full_outputs": True,
        "expected_seed43122_full_root_after_delivery": "experiments/archive/representation_and_objectives/data/seed43122_full_eval",
        "official_ewok_json_after_repair": "experiments/archive/representation_and_objectives/data/official_ewok_reeval_seed43122/official_ewok_reeval_seed43122.json",
        "official_aoa_json_after_repair": "experiments/archive/representation_and_objectives/data/official_aoa_min0_seed43122/official_aoa_min0_seed43122.json",
        "stage_command_template": planned_staging_command(
            "<ewok_predictions_path_from_official_ewok_reeval_seed43122.json>",
            "experiments/archive/representation_and_objectives/data/official_aoa_min0_seed43122",
        ),
        "staged_summary_expected": "experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json",
        "interpretation_rule_if_overall_ge_41p8": "Seed43122 clears the visible leader on the same official coordinate; freeze seed43022+seed43122 endpoint coordinates, then run the prepared adjacency-broken mechanism experiment before packaging if mechanism evidence is still scientifically needed.",
        "interpretation_rule_if_overall_lt_41p8": "Do not launch ad hoc corpus/recipe variants; first compare the full component vector against the fast variance localization and separate pretraining RNG sensitivity from downstream finetuning/checkpoint effects.",
    }

    filled_from_prior = {
        "nulls_now_filled_in_step043_addendum": {
            "compact_reinvest_changed_block_words": changed_block_words,
            "overlap_ngram_occurrences_in_original_source": overlap.get("matched_ngram_occurrences_in_original_source_text"),
            "overlap_ngram_occurrences_in_generated_rewrite": overlap.get("matched_ngram_occurrences_in_generated_rewrite_text"),
            "fast_equal7_delta_431_minus_430": sv_fast.get("equal7_delta"),
            "fast_delta_not_explained_by_last100_loss": safe_get(sv, "training_dynamics.interpretation.last100_loss_delta_not_broad_loss_disadvantage", None) or "Last-100 matched-step mean loss delta was +0.000336 seed43122-minus-seed43022; lower fast surface is not a broad late loss disadvantage.",
        },
        "seed43022_current_official_coordinate": {
            "overall": overall430,
            "scores": scores430,
            "collated_json_sha256": safe_get(seed430, "collated_artifact.sha256") or safe_get(add41, "seed43022_official_coordinate.collated_json.sha256"),
            "unmodified_collator_accepted": safe_get(ledger42, "compliance_closure.official_score_coordinate_seed43022.unmodified_collator_accepted"),
        },
        "seed43122_identical_training_stream_except_rng": {
            "same_train_stream_hash": safe_get(sv_recipe, "train_file_hashes.same_train_stream_hash"),
            "stream_hash": safe_get(sv_recipe, "train_file_hashes.seed431_actual_sha256"),
            "recipe_diffs_except_init_and_training_rng": safe_get(sv_recipe, "recipe_diffs_except_init_and_training_rng"),
            "order_manifest_diffs": safe_get(sv_recipe, "order_manifest_diffs"),
            "different_seeds": safe_get(sv_recipe, "different_seeds"),
        },
    }

    payload = {
        "status": "SEED43122_ROBUSTNESS_DECISION_PREFLIGHT",
        "created_utc": now_utc(),
        "scientific_purpose": "Prepare an exact, official-coordinate seed43122 robustness decision while managed full evaluation and coordinate repairs run asynchronously; no partial managed outputs are read here.",
        "input_artifacts": {name: file_record(path) for name, path in PATHS.items()},
        "inherited_full_wrapper_coordinate_facts": extract_runner_facts(),
        "repair_expectations": repair_expectations,
        "filled_from_prior_ledgers": filled_from_prior,
        "post_delivery_decision_procedure": post_delivery,
        "nonfinal_judgment": "The current SOTA-facing evidence remains single-seed seed43022 Overall 42.0331. Seed43122 must be collated after the full evaluation and coordinate repair finish before robustness or mechanism-packaging work is judged.",
    }
    out_json = OUT_DIR / "seed43122_robustness_decision_preflight.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — seed43122 robustness decision preflight",
        "",
        "This is a CPU-only preflight/addendum. It does not read partial outputs from the running managed seed43122 full evaluation.",
        "",
        "## Coordinate finding",
        "",
        "The seed43122 full wrapper imports the COMPACT_EXPERIENCE full-eval runner and only overrides OUT_ROOT/PER_TARGET/TARGETS. It does not override the runner's WORKSPACE or STRICT path. The inherited runner therefore uses the INITIAL_MODEL_STUDIES strict checkout for zero-shot data and the COMPACT_EXPERIENCE AoA helper. The helper calls `load_eval(word_path, 20, False)`, so its AoA is the old 6,560-row subset, not the official 8,005-row coordinate.",
        "",
        "The inherited full evaluation remains useful for non-EWoK zero-shot, Reading, and SuperGLUE, but seed43122 must use the official EWoK repair and official min_context=0 AoA repair before comparison with seed43022.",
        "",
        "## Already established values now made explicit",
        "",
        f"- Seed43022 official Overall: `{overall430}`.",
        f"- Seed43122 fast equal7 delta: `{sv_fast.get('equal7_delta')}`.",
        f"- Compact reinvest changed-block budget: `{changed_block_words}` words.",
        f"- Exact-overlap n-gram occurrences: original source `{overlap.get('matched_ngram_occurrences_in_original_source_text')}`, generated rewrite `{overlap.get('matched_ngram_occurrences_in_generated_rewrite_text')}`.",
        f"- AoA dry run: min_context=0 `{safe_get(dry_aoa, 'min_context0_loaded.total_contexts_loaded')}` contexts; min_context=20 `{safe_get(dry_aoa, 'min_context20_loaded.total_contexts_loaded')}` contexts.",
        "",
        "## After managed delivery",
        "",
        "After the seed43122 full evaluation and coordinate repair complete, read the result files and run:",
        "",
        "```bash",
        post_delivery["stage_command_template"],
        "```",
        "",
        "Use the actual EWoK predictions path from the research EWoK repair JSON. The staged summary should be `experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json`.",
        "",
        f"JSON: `{out_json}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "seed43022_overall": overall430,
        "fast_equal7_delta_431_minus_430": sv_fast.get("equal7_delta"),
        "full_wrapper_uses_old_aoa_helper": payload["inherited_full_wrapper_coordinate_facts"].get("compact_experience_aoa_helper_uses_min_context20"),
        "aoa_min_context0_contexts": safe_get(dry_aoa, "min_context0_loaded.total_contexts_loaded"),
        "aoa_min_context20_contexts": safe_get(dry_aoa, "min_context20_loaded.total_contexts_loaded"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
