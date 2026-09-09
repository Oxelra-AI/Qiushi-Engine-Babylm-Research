#!/usr/bin/env python3
"""research endpoint decision ledger for compliant BabyLM Strict-Small endpoints.

Static only: reads completed summaries, tokenizer files, and dry-run manifests. It does
not inspect active training run directories, query task state, or run evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
WORK = STUDY
OUT_DIR = WORK / "data" / "endpoint_decision_ledger"

VISIBLE_LEADER = 41.8
OLD_REF_PATH = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_collate_seed43022" / "pristine_collate_seed43022_summary.json"
SEED43122_PATH = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_collate_seed43122" / "pristine_collate_seed43122_summary.json"
TWOBY2_PATH = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "consistent_official_2x2" / "consistent_official_2x2.json"
READY_PATH = WORK / "data" / "postdelivery_static_readiness" / "static_postdelivery_readiness.json"
UNK_PATH = WORK / "data" / "unk_scored_row_exposure" / "unk_scored_row_exposure.json"
SYM_PATH = WORK / "data" / "supplement_newline_symmetry" / "supplement_newline_symmetry.json"
COMP_PATH = WORK / "data" / "supplement_prediction_slices" / "existing_oldtok_reinvest_vs_clean_comparison.json"

ENDPOINTS = {
    "complianttok_reinvest_seed43022": {
        "task_ref": "s36_t25_tool1",
        "target": "complianttok_reinvest_seed43022",
        "run_dir": "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2",
        "tokenizer_dir": "experiments/archive/frontier_consolidation/data/compliant_tokenizer",
        "tokenizer_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
        "postdelivery_manifest": "experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json",
        "role": "legal same-pool tokenizer endpoint with known newline unknown-token exposure",
        "priority": "evaluate if terminal and this does not delay byte-alphabet priority evaluation",
        "driver_gpu": 0,
    },
    "bytealphatok_reinvest_seed43022": {
        "task_ref": "s41_t38_tool1",
        "target": "bytealphatok_reinvest_seed43022",
        "run_dir": "experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022",
        "tokenizer_dir": "experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet",
        "tokenizer_sha256": "b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf",
        "postdelivery_manifest": "experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json",
        "role": "resource-priority legal endpoint using standard full ByteLevel alphabet",
        "priority": "evaluate first after terminal delivery",
        "driver_gpu": 1,
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path, required: bool = True) -> Optional[Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(rel(path))
        return None
    return json.loads(path.read_text())


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_old_reference() -> Dict[str, Any]:
    old = load_json(OLD_REF_PATH)
    seed2 = load_json(SEED43122_PATH)
    two = load_json(TWOBY2_PATH)
    scores = old["score_summary"]["official_overall"]["scores"]
    overall = old["score_summary"]["official_overall"]["Overall"]
    cheap7 = sum(scores[k] for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"])
    req_sg_aoa_for_leader = 9 * VISIBLE_LEADER - cheap7
    req_sg_aoa_for_old = 9 * overall - cheap7
    seed2_scores = seed2["score_summary"]["scores"]
    return {
        "non_submittable_scientific_reference": {
            "path": rel(OLD_REF_PATH),
            "overall": overall,
            "columns": scores,
            "margin_over_visible_leader": overall - VISIBLE_LEADER,
            "cheap7_sum": cheap7,
            "required_superglue_plus_aoa_to_match_visible_leader_given_same_cheap7": req_sg_aoa_for_leader,
            "actual_superglue_plus_aoa": scores["SuperGLUE"] + scores["AoA"],
            "not_submission_reason": "the tokenizer used by this inherited endpoint was trained on the 100M Strict corpus, outside the Strict-Small 10M tokenizer-training budget",
        },
        "second_seed_reference": {
            "path": rel(SEED43122_PATH),
            "overall": seed2["score_summary"]["Overall"],
            "columns": seed2_scores,
            "margin_over_visible_leader": seed2["score_summary"]["Overall"] - VISIBLE_LEADER,
        },
        "two_seed_treatment_evidence": {
            "path": rel(TWOBY2_PATH),
            "overalls": two["overalls"],
            "within_seed_treatment_effect": two["treatment_effect_within_seed"],
            "treatment_x_seed_difference": two["DiD_treatment_x_seed"],
            "meaning": "the redundancy-reduced compact-view plus reinvested-source intervention improved Overall in both old-tokenizer seeds; the unresolved part is compliant-tokenizer recovery and relation stability, not whether the old-tokenizer treatment moved both seeds positively",
        },
    }


def endpoint_static_records() -> Dict[str, Any]:
    ready = load_json(READY_PATH)
    out: Dict[str, Any] = {}
    for target, spec in ENDPOINTS.items():
        tok_json = ROOT / spec["tokenizer_dir"] / "tokenizer.json"
        manifest_path = ROOT / spec["postdelivery_manifest"]
        manifest = load_json(manifest_path)
        tok_sha = sha256_file(tok_json)
        stages = [e.get("stage") for e in manifest.get("events", [])]
        driver_cmd = [
            "PYTHONDONTWRITEBYTECODE=1",
            "python",
            "-B",
            "experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py",
            "--arm", "reinvest",
            "--target", target,
            "--run-dir", spec["run_dir"],
            "--expected-tokenizer", spec["tokenizer_dir"],
            "--expected-tokenizer-sha256", spec["tokenizer_sha256"],
            "--gpu", str(spec["driver_gpu"]),
        ]
        out[target] = {
            **spec,
            "tokenizer_json": rel(tok_json),
            "tokenizer_sha256_matches_spec": tok_sha == spec["tokenizer_sha256"],
            "tokenizer_actual_sha256": tok_sha,
            "dry_run_manifest_path": rel(manifest_path),
            "dry_run_status": manifest.get("status"),
            "dry_run_stages": stages,
            "collate_summary_json": manifest.get("collate_summary_json"),
            "static_ready_manifest_ok": ready.get("dry_run_manifests", {}).get(target, {}).get("all_static_manifest_checks_ok"),
            "postdelivery_command": driver_cmd,
        }
    return out


def surface_interpretation() -> Dict[str, Any]:
    unk = load_json(UNK_PATH, required=False)
    sym = load_json(SYM_PATH, required=False)
    comp = load_json(COMP_PATH, required=False)

    result: Dict[str, Any] = {
        "source_files": {
            "row_exposure": rel(UNK_PATH),
            "newline_symmetry": rel(SYM_PATH),
            "oldtok_slice_comparison": rel(COMP_PATH),
        },
        "what_this_can_decide": "only interpretation after official predictions exist; it cannot choose an endpoint before scoring",
    }
    if unk:
        # Preserve the stable family-level numbers without depending on exact nested names.
        families = unk.get("families", unk.get("family_summary", {}))
        result["unknown_surface"] = families if families else "see source JSON; family schema not summarized here"
    if sym:
        keys = [
            "official_supplement_rows",
            "target_unk_rows",
            "target_unk_tokens",
            "affected_rows_both_good_bad",
            "affected_same_unk_count",
            "affected_same_span_multiset",
            "affected_same_span_offset",
            "subtasks",
        ]
        result["supplement_newline_symmetry"] = {k: sym.get(k) for k in keys if k in sym}
    if comp:
        # The comparison file has a compact table-like record in most runs.
        result["oldtok_reinvest_vs_clean_supplement_slices"] = comp
    return result


def decision_logic() -> Dict[str, Any]:
    return {
        "expensive_work_entry": {
            "terminal_training_result_required_before_evaluation": True,
            "what_full_official_evaluation_decides": [
                "whether a legal 10M-tokenizer compact-view-reinvest endpoint exceeds the visible 41.8 Strict-Small Overall score",
                "whether the old-tokenizer 42.0331 scientific result survives under an end-to-end compliant tokenizer coordinate",
                "whether the byte-alphabet construction improves, preserves, or hurts the legal endpoint relative to the research same-pool tokenizer",
            ],
            "lowest_cost_reliable_sequence": [
                "collect authoritative terminal task result only after runtime delivery",
                "inspect delivered run metadata for 100M exposure, 10-pass corpus use, full checkpoint ladder, frozen recipe, `hf_model/chck_100M`, and exact tokenizer SHA",
                "run zero-shot and Reading columns first, including GlobalPIQA parallel/nonparallel combined as the official GlobalPIQA column",
                "run the posthoc Supplement slice join only to interpret the already-produced Supplement predictions",
                "apply the existing hard upper-bound continuation arithmetic before SuperGLUE and AoA",
                "run SuperGLUE, official min_context=0 AoA, and pristine collation only when the endpoint remains mathematically viable for the target being tested",
            ],
            "not_allowed_as_shortcuts": [
                "partial columns cannot stand in for the complete official-coordinate Overall",
                "old-tokenizer dynamics cannot predict the corrected-tokenizer endpoint",
                "newline unknown-token analysis cannot determine the sign of the research endpoint before official predictions",
                "tokenizer design is frozen; no vocabulary changes may be derived from benchmark strings",
            ],
        },
        "route_after_results": [
            {
                "condition": "bytealphatok_reinvest_seed43022 completes and inspection verifies the frozen recipe",
                "action": "run the research post-delivery driver for bytealphatok first and preserve pristine nine-column collation",
                "next_scientific_use": "if Overall exceeds 41.8, independently review compliance, evidence, official coordinate, contamination safeguards and reproducibility; if below 41.8, compare column movements against old 42.0331 and other completed endpoints before a new training comparison",
            },
            {
                "condition": "complianttok_reinvest_seed43022 completes before or alongside bytealphatok",
                "action": "evaluate it only when this does not delay bytealphatok priority; use its isolated target name and pristine collation path",
                "next_scientific_use": "treat as a legal endpoint, not a discarded contrast; use Supplement slice output only after predictions to separate newline-surface movement from broader training dynamics",
            },
            {
                "condition": "a terminal task fails before producing a complete chck_100M endpoint because of resource/OOM interruption",
                "action": "preserve the authoritative failure record and relaunch the same frozen command only if the failure is resource interference; do not change corpus, tokenizer, seeds, batch, sequence length, optimizer, WWM, or exposure",
                "next_scientific_use": "maintain the load-bearing endpoint experiment without turning resource failure into evidence about the learning principle",
            },
            {
                "condition": "no legal endpoint exceeds the visible leader after full official-coordinate scoring",
                "action": "use the exact compliant-tokenizer score vector and posthoc interpretation to choose a new evidence-supported comparison; renaming the same tokenizer or spatial-lengthening variant does not create a new comparison",
                "next_scientific_use": "decide whether the next route should address relation stability, GlobalPIQA weakness, optimizer dynamics, or a deeper data/representation principle through a cheaper discriminating screen before any new 100M run",
            },
        ],
    }


def write_markdown(ledger: Dict[str, Any], path: Path) -> None:
    old = ledger["official_references"]["non_submittable_scientific_reference"]
    two = ledger["official_references"]["two_seed_treatment_evidence"]
    endpoints = ledger["endpoints"]
    decision = ledger["decision_logic"]["expensive_work_entry"]
    lines: List[str] = []
    lines.append("# research endpoint decision ledger")
    lines.append("")
    lines.append("This file records how the pending legal endpoints should be interpreted once the runtime delivers a terminal retrain result. It reads completed artifacts only and does not inspect active training run directories or managed-task state.")
    lines.append("")
    lines.append("## Fixed references")
    lines.append("")
    lines.append(f"- Visible leaderboard target used here: **{VISIBLE_LEADER:.3f}** Overall.")
    lines.append(f"- Old-tokenizer compact-view-reinvest seed43022: **{old['overall']:.12f}** Overall, margin **{old['margin_over_visible_leader']:.6f}** over 41.8, but not submittable because its tokenizer was trained on 100M Strict text.")
    lines.append(f"- For the old endpoint's cheap seven-column surface, SuperGLUE+AoA needed to match 41.8 would be **{old['required_superglue_plus_aoa_to_match_visible_leader_given_same_cheap7']:.6f}**; actual SuperGLUE+AoA was **{old['actual_superglue_plus_aoa']:.6f}**.")
    te = two["within_seed_treatment_effect"]
    lines.append(f"- Two old-tokenizer seeds both improved under compact-view reinvestment: treatment effects **{te['TE_43022_overall']:.6f}** and **{te['TE_43122_overall']:.6f}** Overall; the two-seed average is **{te['ATE_overall_two_seeds']:.6f}**.")
    lines.append("")
    lines.append("## Pending legal endpoints")
    lines.append("")
    for target, rec in endpoints.items():
        lines.append(f"### `{target}`")
        lines.append("")
        lines.append(f"- Task: `{rec['task_ref']}`.")
        lines.append(f"- Role: {rec['role']}.")
        lines.append(f"- Priority: {rec['priority']}.")
        lines.append(f"- Tokenizer: `{rec['tokenizer_dir']}`; SHA match: `{rec['tokenizer_sha256_matches_spec']}`.")
        lines.append(f"- Run directory to inspect after terminal delivery only: `{rec['run_dir']}`.")
        lines.append(f"- Dry-run stages: {', '.join(rec['dry_run_stages'])}.")
        lines.append(f"- Isolated pristine collation summary: `{rec['collate_summary_json']}`.")
        lines.append("- Post-delivery command:")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(rec["postdelivery_command"]))
        lines.append("```")
        lines.append("")
    lines.append("## Why the next GPU evaluation is justified only after terminal training")
    lines.append("")
    lines.append("The next expensive evaluation resolves:")
    for item in decision["what_full_official_evaluation_decides"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Use the lowest-cost reliable sequence:")
    for item in decision["lowest_cost_reliable_sequence"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Do not use these shortcuts:")
    for item in decision["not_allowed_as_shortcuts"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Result-dependent route")
    lines.append("")
    for row in ledger["decision_logic"]["route_after_results"]:
        lines.append(f"- If {row['condition']}: {row['action']} Scientific use: {row['next_scientific_use']}")
    lines.append("")
    lines.append("## Interpretation assets")
    lines.append("")
    lines.append("The research and research files explain possible score movement only after official predictions exist. They cannot choose the endpoint before scoring. The research tokenizer unknown-token exposure is concentrated in Supplement QA/turn-taking newlines and is structurally symmetric in good and bad alternatives, while old-tokenizer reinvestment's Supplement gain over clean came mainly from unaffected rows.")
    lines.append("")
    lines.append(f"Full JSON: `{rel(OUT_DIR / 'endpoint_decision_ledger.json')}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = {
        "status": "ENDPOINT_DECISION_LEDGER",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "completed summaries, tokenizer artifacts, and post-training dry-run manifests only; active run directories are not inspected",
        "active_tasks_not_polled": ["s36_t25_tool1", "s41_t38_tool1"],
        "visible_leader_overall": VISIBLE_LEADER,
        "official_references": get_old_reference(),
        "static_readiness": load_json(READY_PATH),
        "endpoints": endpoint_static_records(),
        "surface_interpretation": surface_interpretation(),
        "decision_logic": decision_logic(),
    }
    out_json = OUT_DIR / "endpoint_decision_ledger.json"
    out_md = OUT_DIR / "endpoint_decision_ledger.md"
    out_json.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    write_markdown(ledger, out_md)
    print(json.dumps({
        "status": ledger["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "endpoints": list(ledger["endpoints"].keys()),
        "old_reference_overall": ledger["official_references"]["non_submittable_scientific_reference"]["overall"],
        "two_seed_ate_overall": ledger["official_references"]["two_seed_treatment_evidence"]["within_seed_treatment_effect"]["ATE_overall_two_seeds"],
        "active_tasks_not_polled": ledger["active_tasks_not_polled"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
