#!/usr/bin/env python3
"""research: static readiness check for compliant endpoint post-delivery evaluation.

This script checks only static scripts, tokenizer artifacts, and dry-run evaluation manifests. It does not inspect active training directories.
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
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
DATA = WORKSPACE / "data"
OUT_DIR = DATA / "postdelivery_static_readiness"

TOK = DATA / "compliant_tokenizer"
TOK = DATA / "compliant_tokenizer_bytealphabet"
EXPECTED = {
    "complianttok_reinvest_seed43022": {
        "tokenizer_dir": TOK,
        "sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
        "run_dir": "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2",
        "priority": "evaluate_if_delivered_without_delaying_bytealphabet_priority",
    },
    "bytealphatok_reinvest_seed43022": {
        "tokenizer_dir": TOK,
        "sha256": "b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf",
        "run_dir": "experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022",
        "priority": "resource_priority_submission_relevant_endpoint",
    },
}

STATIC_SCRIPTS = [
    SCRIPTS / "compliant_postdelivery_driver.py",
    SCRIPTS / "inspect_compliant_retrain.py",
    SCRIPTS / "evaluate_compliant_endpoint.py",
    SCRIPTS / "project_compliant_eval_continuation.py",
    SCRIPTS / "supplement_prediction_slices.py",
]

DRY_RUN_MANIFESTS = {
    target: DATA / "compliant_postdelivery_driver" / f"{target}_postdelivery_driver.json"
    for target in EXPECTED
}

REQUIRED_STAGES = [
    "inspect",
    "cheap_columns",
    "supplement_prediction_slices",
    "continuation_projection",
    "superglue_aoa",
    "pristine_collate",
]

CHEAP_COLUMNS = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading",
]
FINAL_COLUMNS = ["SuperGLUE", "AoA"]
FROZEN_REINVEST_POOL_10M = "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
FROZEN_REINVEST_TRAIN_100M = "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
FROZEN_REINVEST_META = "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
FROZEN_REINVEST_POOL_SHA256 = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
FROZEN_REINVEST_TRAIN_SHA256 = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
FROZEN_REINVEST_META_SHA256 = "92aa4c00b09d201a09da27e7e16677643bc7cda321471e1ec46332600339b3e4"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_script(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": rel(path), "exists": path.is_file()}
    if path.is_file():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            rec["ast_ok"] = True
        except SyntaxError as e:
            rec["ast_ok"] = False
            rec["syntax_error"] = str(e)
    return rec


def check_tokenizer(target: str, spec: dict[str, Any]) -> dict[str, Any]:
    tok_json = Path(spec["tokenizer_dir"]) / "tokenizer.json"
    rec: dict[str, Any] = {
        "target": target,
        "tokenizer_dir": rel(Path(spec["tokenizer_dir"])),
        "tokenizer_json": rel(tok_json),
        "exists": tok_json.is_file(),
        "expected_sha256": spec["sha256"],
    }
    if tok_json.is_file():
        actual = sha256_file(tok_json)
        rec["actual_sha256"] = actual
        rec["sha256_ok"] = actual == spec["sha256"]
        try:
            payload = json.loads(tok_json.read_text(encoding="utf-8"))
            rec["model_type"] = payload.get("model", {}).get("type")
            rec["vocab_size"] = len(payload.get("model", {}).get("vocab", {}))
            rec["has_added_tokens"] = len(payload.get("added_tokens", []))
        except Exception as e:
            rec["json_error"] = str(e)
    return rec


def cmd_contains_sequence(cmd: list[str], seq: list[str]) -> bool:
    if not seq:
        return True
    n = len(seq)
    return any(cmd[i : i + n] == seq for i in range(0, len(cmd) - n + 1))


def value_after(cmd: list[str], flag: str) -> str | None:
    try:
        i = cmd.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(cmd):
        return None
    return cmd[i + 1]


def check_manifest(target: str, path: Path) -> dict[str, Any]:
    spec = EXPECTED[target]
    rec: dict[str, Any] = {
        "target": target,
        "manifest_path": rel(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return rec
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    by_stage = {ev.get("stage"): ev for ev in events}
    rec.update({
        "status": payload.get("status"),
        "target_field": payload.get("target"),
        "run_dir_field": payload.get("run_dir"),
        "collate_summary_json": payload.get("collate_summary_json"),
        "stage_names": [ev.get("stage") for ev in events],
        "required_stages_present": all(stage in by_stage for stage in REQUIRED_STAGES),
        "target_ok": payload.get("target") == target,
        "run_dir_ok": payload.get("run_dir") == spec["run_dir"],
        "dry_run_status_ok": payload.get("status") == "POSTDELIVERY_DRY_RUN",
    })

    inspect_cmd = by_stage.get("inspect", {}).get("cmd", [])
    cheap_cmd = by_stage.get("cheap_columns", {}).get("cmd", [])
    slice_cmd = by_stage.get("supplement_prediction_slices", {}).get("cmd", [])
    proj_cmd = by_stage.get("continuation_projection", {}).get("cmd", [])
    final_cmd = by_stage.get("superglue_aoa", {}).get("cmd", [])
    collate_cmd = by_stage.get("pristine_collate", {}).get("cmd", [])

    rec["inspect_expected_sha_ok"] = cmd_contains_sequence(
        inspect_cmd, ["--expected-tokenizer-sha256", spec["sha256"]]
    )
    rec["inspect_tokenizer_dir_ok"] = cmd_contains_sequence(
        inspect_cmd, ["--expected-tokenizer", rel(Path(spec["tokenizer_dir"]))]
    )
    rec["inspect_expected_train_file_ok"] = value_after(inspect_cmd, "--expected-train-file") == FROZEN_REINVEST_TRAIN_100M
    rec["inspect_expected_train_sha_ok"] = value_after(inspect_cmd, "--expected-train-sha256") == FROZEN_REINVEST_TRAIN_SHA256
    rec["inspect_expected_pool_file_ok"] = value_after(inspect_cmd, "--expected-pool-10m") == FROZEN_REINVEST_POOL_10M
    rec["inspect_expected_pool_sha_ok"] = value_after(inspect_cmd, "--expected-pool-sha256") == FROZEN_REINVEST_POOL_SHA256
    rec["inspect_expected_meta_file_ok"] = value_after(inspect_cmd, "--expected-meta") == FROZEN_REINVEST_META
    rec["inspect_expected_meta_sha_ok"] = value_after(inspect_cmd, "--expected-meta-sha256") == FROZEN_REINVEST_META_SHA256
    rec["inspect_expected_pool_words_ok"] = value_after(inspect_cmd, "--expected-pool-words") == "10000000"
    rec["inspect_expected_train_words_ok"] = value_after(inspect_cmd, "--expected-train-words") == "100000000"
    rec["inspect_expected_pool_rows_ok"] = value_after(inspect_cmd, "--expected-pool-rows") == "64740"
    rec["inspect_expected_train_rows_ok"] = value_after(inspect_cmd, "--expected-train-rows") == "647400"
    rec["inspect_expected_passes_ok"] = value_after(inspect_cmd, "--expected-passes") == "10"
    rec["inspect_verify_corpus_content_ok"] = "--verify-corpus-content" in inspect_cmd
    rec["cheap_columns_ok"] = cmd_contains_sequence(cheap_cmd, ["--columns"] + CHEAP_COLUMNS)
    rec["final_columns_ok"] = cmd_contains_sequence(final_cmd, ["--columns"] + FINAL_COLUMNS)
    rec["projection_target_ok"] = value_after(proj_cmd, "--decision-target") == "41.8"
    rec["slice_target_ok"] = value_after(slice_cmd, "--tag") == target
    rec["collate_target_ok"] = value_after(collate_cmd, "--target") == target
    rec["collate_endpoint_ok"] = value_after(collate_cmd, "--endpoint") == "chck_100M"
    rec["collate_model_root_ok"] = value_after(collate_cmd, "--model-root") == f"{spec['run_dir']}/hf_model"
    rec["collate_output_isolated_ok"] = target in str(payload.get("collate_summary_json", ""))

    boolean_keys = [k for k, v in rec.items() if k.endswith("_ok") or k == "required_stages_present"]
    rec["all_static_manifest_checks_ok"] = all(bool(rec.get(k)) for k in boolean_keys)
    rec["boolean_checks"] = {k: rec.get(k) for k in boolean_keys}
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts = [parse_script(p) for p in STATIC_SCRIPTS]
    tokenizers = {target: check_tokenizer(target, spec) for target, spec in EXPECTED.items()}
    manifests = {target: check_manifest(target, path) for target, path in DRY_RUN_MANIFESTS.items()}

    output_paths = [m.get("collate_summary_json") for m in manifests.values() if m.get("collate_summary_json")]
    output_paths_are_distinct = len(output_paths) == len(set(output_paths))
    no_deliverables_targets = all("/deliverables/" not in str(p) and not str(p).startswith("deliverables/") for p in output_paths)

    status_ok = (
        all(s.get("exists") and s.get("ast_ok") for s in scripts)
        and all(t.get("exists") and t.get("sha256_ok") and t.get("vocab_size") == 16384 for t in tokenizers.values())
        and all(m.get("exists") and m.get("all_static_manifest_checks_ok") for m in manifests.values())
        and output_paths_are_distinct
        and no_deliverables_targets
    )

    result = {
        "status": "STATIC_POSTDELIVERY_READY" if status_ok else "STATIC_POSTDELIVERY_NOT_READY",
        "scope": "static scripts, tokenizer artifacts, and dry-run manifests only; active training run directories were not inspected",
        "active_tasks_not_polled": ["s36_t25_tool1", "s41_t38_tool1"],
        "endpoint_policy": {
            "bytealphatok_reinvest_seed43022": "resource-priority compliant endpoint; evaluate first after terminal delivery",
            "complianttok_reinvest_seed43022": "legal research-tokenizer endpoint; evaluate if terminal delivery is available and it does not delay bytealphabet priority",
            "comparison_basis": "isolated pristine official nine-column collation only",
            "no_more_preresult_subset_mining": True,
        },
        "scripts": scripts,
        "tokenizers": tokenizers,
        "dry_run_manifests": manifests,
        "output_paths_are_distinct": output_paths_are_distinct,
        "no_deliverables_targets": no_deliverables_targets,
    }

    out_json = OUT_DIR / "static_postdelivery_readiness.json"
    out_md = OUT_DIR / "static_postdelivery_readiness.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# research static post-delivery readiness check",
        "",
        f"Status: `{result['status']}`",
        "",
        "Scope: checked only static scripts, tokenizer artifacts, and existing dry-run manifests. Active retrain run directories were not inspected.",
        "",
        "## Endpoint policy preserved",
        "",
        "- `bytealphatok_reinvest_seed43022`: resource-priority compliant endpoint; after terminal delivery, inspect and evaluate first.",
        "- `complianttok_reinvest_seed43022`: legal research-tokenizer endpoint; evaluate if delivered and doing so does not delay the byte-alphabet priority path.",
        "- Compare endpoints only through isolated pristine official nine-column collation. Do not add more pre-result subset mining or tokenizer-design changes.",
        "",
        "## Static checks",
        "",
    ]
    lines.append(f"- Scripts AST/pass: {sum(1 for s in scripts if s.get('exists') and s.get('ast_ok'))}/{len(scripts)}")
    lines.append("- Post-delivery inspector requires frozen reinvest corpus provenance before evaluation: 10M pool SHA, 100M training SHA, metadata SHA, 64,740/647,400 rows, 10M/100M words, and 10 passes, with JSONL word counting enabled.")
    for s in scripts:
        lines.append(f"  - `{s['path']}`: exists={s.get('exists')} ast_ok={s.get('ast_ok')}")
    lines.append("")
    lines.append("## Tokenizer artifacts")
    for target, t in tokenizers.items():
        lines.append(
            f"- `{target}`: tokenizer_json=`{t.get('tokenizer_json')}`, vocab_size={t.get('vocab_size')}, sha256_ok={t.get('sha256_ok')}"
        )
    lines.append("")
    lines.append("## Dry-run post-delivery manifests")
    for target, m in manifests.items():
        lines.append(
            f"- `{target}`: exists={m.get('exists')}, stages_present={m.get('required_stages_present')}, all_static_manifest_checks_ok={m.get('all_static_manifest_checks_ok')}"
        )
        lines.append(f"  - collate_summary_json=`{m.get('collate_summary_json')}`")
        failed = [k for k, v in m.get("boolean_checks", {}).items() if not v]
        if failed:
            lines.append(f"  - failed_checks={failed}")
    lines.append("")
    lines.append(f"Distinct collate outputs: {output_paths_are_distinct}")
    lines.append(f"No deliverables targets: {no_deliverables_targets}")
    lines.append("")
    lines.append(f"Full JSON: `{rel(out_json)}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "scripts_ok": sum(1 for s in scripts if s.get('exists') and s.get('ast_ok')),
        "tokenizers_ok": {k: v.get("sha256_ok") for k, v in tokenizers.items()},
        "manifests_ok": {k: v.get("all_static_manifest_checks_ok") for k, v in manifests.items()},
        "output_paths_are_distinct": output_paths_are_distinct,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
