#!/usr/bin/env python3
"""research: verify content identity and provenance of composite fast-eval inputs.

The research full --fast materializer uses a composite fast_eval_data root:
  * BLiMP/Supplement/Entity/Reading from the current pristine BabyLM-2026 Strict-Evals snapshot.
  * GlobalPIQA folders generated in research by the current official global_piqa/dl.py from mrlbenchmarks datasets.
  * EWoK fast plaintext linked from the expanded INITIAL_MODEL_STUDIES strict repo because the current official ewok_fast.zip is encrypted.

This script verifies content-level identity of that composite root to those sources and ties each source back to current official/revision evidence.  It is intentionally CPU-only and does not read or alter model predictions.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import binascii
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01 = ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = A01
OUT_DIR = WORKSPACE / "data/fast_eval_input_provenance"
OUT_JSON = OUT_DIR / "fast_eval_input_provenance.json"
OUT_MD = OUT_DIR / "fast_eval_input_provenance.md"
CURRENT_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
CURRENT_FAST = CURRENT_STRICT / "evaluation_data/fast_eval"
CURRENT_HF_DL_CACHE = CURRENT_STRICT / ".cache/huggingface/download"
LINEAGE = WORKSPACE / "data/globalpiqa_official_lineage/globalpiqa_official_lineage.json"
FAST = WORKSPACE / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/fast_eval"
COMPOSITE_FAST = WORKSPACE / "data/chck82_fast_submission_materialization/fast_eval_data"
INITIAL_MODEL_STUDIES_EWOK_FAST = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"
CURRENT_EWOK_ZIP = CURRENT_FAST / "ewok_fast.zip"
EXPECTED_FAST_REVISION = "8d52da9424a9ff30b9e8266c4f751aba9c504233"
EXPECTED_GLOBALPIQA_SHAS = {
    "parallel": "b0b18516a8bc2cb1106bce3dd4db32848ca715ea",
    "nonparallel": "6777742fa3634c0583cda3b7f8a482ea7b1b0937",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str((p if p.is_absolute() else ROOT / p).relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def crc32_file(path: Path) -> int:
    c = 0
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            c = binascii.crc32(b, c)
    return c & 0xFFFFFFFF


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def count_csv_rows_no_header(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    return max(0, len(rows) - 1)


def file_head_tail(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f]
    return {
        "first_line": lines[0] if lines else None,
        "last_line": lines[-1] if lines else None,
        "line_count": len([x for x in lines if x.strip()]) if path.suffix.lower() != ".csv" else len(lines),
    }


def file_record(path: Path, *, include_sample: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        rec["symlink_target"] = os.readlink(path)
        try:
            rec["resolved"] = rel(path.resolve())
        except Exception as exc:
            rec["resolve_error"] = repr(exc)
    if path.exists() and path.is_file():
        rec.update({
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "crc32_hex": f"{crc32_file(path):08x}",
        })
        if path.suffix.lower() == ".jsonl":
            rec["jsonl_rows"] = count_jsonl(path)
        elif path.suffix.lower() == ".csv":
            rec["csv_rows_excluding_header"] = count_csv_rows_no_header(path)
        if include_sample:
            rec.update(file_head_tail(path))
    return rec


def manifest_dir(path: Path, *, include_sample: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    records = []
    combined = hashlib.sha256()
    for p in files:
        r = file_record(p, include_sample=include_sample)
        r["relative_to_dir"] = str(p.relative_to(path))
        records.append(r)
        combined.update(str(p.relative_to(path)).encode("utf-8") + b"\0")
        combined.update(bytes.fromhex(r["sha256"]))
    return {
        "path": rel(path),
        "exists": True,
        "is_symlink": path.is_symlink(),
        "resolved": rel(path.resolve()) if path.exists() else None,
        "file_count": len(files),
        "total_size_bytes": sum(p.stat().st_size for p in files),
        "tree_sha256_over_names_and_file_sha256": combined.hexdigest(),
        "files": records,
    }


def same_manifest(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def trim(m: dict[str, Any]) -> list[tuple[str, int, str]]:
        return sorted((f.get("relative_to_dir"), int(f.get("size_bytes", -1)), str(f.get("sha256"))) for f in m.get("files", []))
    return trim(a) == trim(b)


def hf_metadata_for_current(rel_path: str) -> dict[str, Any]:
    p = CURRENT_HF_DL_CACHE / rel_path
    mp = Path(str(p) + ".metadata")
    rec = {"metadata_path": rel(mp), "exists": mp.exists()}
    if mp.exists():
        lines = mp.read_text(encoding="utf-8", errors="replace").splitlines()
        rec.update({
            "revision_or_commit": lines[0] if len(lines) > 0 else None,
            "etag_or_blob_sha": lines[1] if len(lines) > 1 else None,
            "timestamp": lines[2] if len(lines) > 2 else None,
            "line_count": len(lines),
        })
    return rec


def group_file_records(base_dir: Path, rels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for r in rels:
        p = base_dir / r
        rec = file_record(p, include_sample=True)
        rec["relative_path"] = r
        rows.append(rec)
    return rows


def verify_current_group(name: str, rel_dir: str, row_expectation: int | dict[str, int] | None) -> dict[str, Any]:
    source = CURRENT_FAST / rel_dir
    comp = COMPOSITE_FAST / rel_dir
    src_manifest = manifest_dir(source, include_sample=False)
    comp_manifest = manifest_dir(comp, include_sample=False)
    errors: list[str] = []
    if not same_manifest(src_manifest, comp_manifest):
        errors.append("composite_files_not_byte_identical_to_current_official_snapshot")
    metadata_revisions = []
    for f in src_manifest.get("files", []):
        rp = f"evaluation_data/fast_eval/{rel_dir}/{f['relative_to_dir']}"
        md = hf_metadata_for_current(rp)
        metadata_revisions.append({"relative_to_source": f["relative_to_dir"], "metadata": md})
        if md.get("revision_or_commit") != EXPECTED_FAST_REVISION:
            errors.append(f"metadata_revision_mismatch_{f['relative_to_dir']}_{md.get('revision_or_commit')}")
    counts: dict[str, int] = {}
    for f in comp_manifest.get("files", []):
        p = comp / f["relative_to_dir"]
        if str(p).endswith(".jsonl"):
            counts[Path(f["relative_to_dir"]).stem] = count_jsonl(p)
    if rel_dir == "reading":
        p = comp / "reading_data.csv"
        if p.exists():
            counts["reading_rows_excluding_header"] = count_csv_rows_no_header(p)
    if isinstance(row_expectation, int):
        bad = {k: v for k, v in counts.items() if v != row_expectation}
        if bad:
            errors.append(f"row_count_mismatch_{bad}")
    elif isinstance(row_expectation, dict):
        if counts != row_expectation:
            errors.append(f"row_count_dict_mismatch_{counts}")
    return {
        "name": name,
        "source_kind": "current_official_strict_eval_snapshot",
        "source_root": rel(source),
        "composite_root": rel(comp),
        "source_manifest": src_manifest,
        "composite_manifest": comp_manifest,
        "byte_identical_source_to_composite": same_manifest(src_manifest, comp_manifest),
        "hf_metadata_revision_records": metadata_revisions,
        "counts": counts,
        "errors": errors,
    }


def verify_globalpiqa() -> dict[str, Any]:
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = lineage.get("checks", {})
    required_checks = [
        "initial_model_studies_dl_identical_to_pristine_dl",
        "official_dl_returncode",
        "all_generated_files_exist",
        "all_generated_counts_match_collator_expectation",
        "all_generated_match_inherited_bytes",
        "generated_full_equals_fast",
        "inherited_full_equals_fast",
    ]
    for k in required_checks:
        if checks.get(k) is not True and not (k == "official_dl_returncode" and checks.get(k) == 0):
            errors.append(f"lineage_check_not_true_{k}_{checks.get(k)}")
    if lineage.get("official_procedure", {}).get("official_global_piqa_dl_sha256") != checks.get("official_dl_sha256"):
        errors.append("official_dl_sha_not_self_consistent")
    for side, expected_sha in EXPECTED_GLOBALPIQA_SHAS.items():
        got = lineage.get("upstream_dataset_revisions", {}).get(side, {}).get("sha")
        if got != expected_sha:
            errors.append(f"globalpiqa_{side}_dataset_sha_{got}_not_{expected_sha}")
    out_sides = {}
    for name in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        source = FAST / name
        comp = COMPOSITE_FAST / name
        src_manifest = manifest_dir(source, include_sample=True)
        comp_manifest = manifest_dir(comp, include_sample=True)
        if not same_manifest(src_manifest, comp_manifest):
            errors.append(f"{name}_composite_not_identical_to_step038_current_official_generation")
        rows = None
        f = comp / "eng_latn.jsonl"
        if f.exists():
            rows = count_jsonl(f)
        out_sides[name] = {
            "source_manifest": src_manifest,
            "composite_manifest": comp_manifest,
            "byte_identical_source_to_composite": same_manifest(src_manifest, comp_manifest),
            "rows": rows,
        }
        expected_rows = 103 if name.endswith("parallel") and not name.endswith("nonparallel") else 100
        if rows != expected_rows:
            errors.append(f"{name}_rows_{rows}_not_{expected_rows}")
    return {
        "name": "GlobalPIQA fast parallel/nonparallel",
        "source_kind": "generated_by_current_official_global_piqa_dl_from_recorded_mrlbenchmarks_revisions",
        "lineage_path": rel(LINEAGE),
        "official_procedure": lineage.get("official_procedure"),
        "upstream_dataset_revisions": lineage.get("upstream_dataset_revisions"),
        "lineage_checks": checks,
        "sides": out_sides,
        "errors": errors,
    }


def zip_ewok_entries(zip_path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if "ewok_fast/" in name:
                key = name.split("ewok_fast/", 1)[1]
            else:
                key = Path(name).name
            if not key:
                continue
            entries[key] = {
                "zip_filename": name,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc32_hex": f"{info.CRC:08x}",
                "encrypted_flag": bool(info.flag_bits & 0x1),
            }
    return entries


def verify_ewok() -> dict[str, Any]:
    errors: list[str] = []
    comp = COMPOSITE_FAST / "ewok_fast"
    source = INITIAL_MODEL_STUDIES_EWOK_FAST
    src_manifest = manifest_dir(source, include_sample=True)
    comp_manifest = manifest_dir(comp, include_sample=True)
    if not same_manifest(src_manifest, comp_manifest):
        errors.append("composite_ewok_not_byte_identical_to_initial_model_studies_expanded_tree")
    zip_md = hf_metadata_for_current("evaluation_data/fast_eval/ewok_fast.zip")
    if zip_md.get("revision_or_commit") != EXPECTED_FAST_REVISION:
        errors.append(f"ewok_zip_revision_{zip_md.get('revision_or_commit')}_not_{EXPECTED_FAST_REVISION}")
    if not CURRENT_EWOK_ZIP.exists():
        errors.append("current_official_ewok_fast_zip_missing")
        entries = {}
    else:
        entries = zip_ewok_entries(CURRENT_EWOK_ZIP)
    file_to_zip_matches = []
    for f in sorted(comp.rglob("*.jsonl")):
        key = str(f.relative_to(comp))
        z = entries.get(key)
        frec = file_record(f, include_sample=True)
        match = bool(z) and z.get("file_size") == f.stat().st_size and z.get("crc32_hex") == frec.get("crc32_hex")
        file_to_zip_matches.append({"relative_to_ewok_fast": key, "file": frec, "zip_entry": z, "matches_zip_uncompressed_size_and_crc32": match})
        if not match:
            errors.append(f"ewok_file_zip_crc_or_size_mismatch_{key}")
        if frec.get("jsonl_rows") != 100:
            errors.append(f"ewok_file_rows_{key}_{frec.get('jsonl_rows')}_not_100")
    comp_keys = sorted(str(p.relative_to(comp)) for p in comp.rglob("*.jsonl"))
    zip_keys = sorted(entries)
    if comp_keys != zip_keys:
        errors.append(f"ewok_zip_file_list_mismatch_comp={comp_keys}_zip={zip_keys}")
    return {
        "name": "EWoK fast",
        "source_kind": "initial_model_studies_expanded_plaintext_tree_matched_to_current_official_encrypted_zip_by_filename_size_crc32",
        "current_official_zip": file_record(CURRENT_EWOK_ZIP),
        "current_official_zip_hf_metadata": zip_md,
        "source_manifest": src_manifest,
        "composite_manifest": comp_manifest,
        "byte_identical_initial_model_studies_source_to_composite": same_manifest(src_manifest, comp_manifest),
        "zip_entry_count": len(entries),
        "zip_entries": entries,
        "file_to_zip_matches": file_to_zip_matches,
        "all_files_match_zip_uncompressed_size_and_crc32": all(x["matches_zip_uncompressed_size_and_crc32"] for x in file_to_zip_matches) and comp_keys == zip_keys,
        "note": "The current official ewok_fast.zip is encrypted, so direct extraction without the password is not available here; zip central-directory CRC32 and uncompressed sizes are nevertheless compared against every plaintext file used by the composite fast root.",
        "errors": errors,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The entity-tracking fast input is one raw `regular.jsonl` file with 3,152
    # rows.  The official runner drops/partitions examples while writing
    # predictions, so the post-evaluation prediction-file expectation is the
    # 2,238-row regular_0_ops...regular_5_ops dictionary checked separately by
    # `fast_submission_verifier.py`; it is not the raw input row count.
    groups = {
        "blimp_fast": verify_current_group("BLiMP fast", "blimp_fast", 200),
        "supplement_fast": verify_current_group("Supplement fast", "supplement_fast", 50),
        "entity_tracking_fast": verify_current_group("Entity Tracking fast", "entity_tracking_fast", {"regular": 3152}),
        "reading": verify_current_group("Reading fast", "reading", {"reading_rows_excluding_header": 1726}),
        "global_piqa": verify_globalpiqa(),
        "ewok_fast": verify_ewok(),
    }
    errors = []
    for key, value in groups.items():
        for e in value.get("errors", []):
            errors.append(f"{key}:{e}")
    composite_manifest = manifest_dir(COMPOSITE_FAST, include_sample=False)
    out = {
        "status": "PASS" if not errors else "NEEDS_REPAIR",
        "created_utc": now(),
        "purpose": "Establish content-level identity and revision provenance for the composite fast_eval_data root used by the research chck_82M full --fast submission materializer before any final collated file is treated as a submission carrier.",
        "current_official_coordinate": {
            "strict_root": rel(CURRENT_STRICT),
            "strict_evals_revision_from_current_metadata": EXPECTED_FAST_REVISION,
            "download_script": rel(CURRENT_STRICT / "scripts/download_evals.py"),
            "ewok_fast_zip_metadata": hf_metadata_for_current("evaluation_data/fast_eval/ewok_fast.zip"),
        },
        "composite_fast_root": {
            "path": rel(COMPOSITE_FAST),
            "exists": COMPOSITE_FAST.exists(),
            "manifest": composite_manifest,
        },
        "groups": groups,
        "errors": errors,
        "ready_to_use_composite_fast_inputs_if_full_materialization_passes": not errors,
        "remaining_endpoint_open_work_not_resolved_by_this_file": [
            "research full fast prediction materialization task must finish and the research verifier must pass on prediction files and collated fast_eval_results.",
            "research from-corpus training reproduction task must finish and be compared against the original chck_82M ladder before training reproducibility is claimed.",
            "This provenance file does not change the endpoint score or solve the separate representation-forming mechanism problem.",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research fast-eval input provenance",
        "",
        f"Status: {out['status']}",
        f"Composite fast root: `{rel(COMPOSITE_FAST)}`",
        f"Current Strict-Evals revision from metadata: `{EXPECTED_FAST_REVISION}`",
        "",
        "## Source identity",
        f"- BLiMP, Supplement, Entity Tracking, and Reading: byte-identical to the current pristine Strict-Evals snapshot under `{rel(CURRENT_FAST)}`; per-file HF metadata is checked against revision `{EXPECTED_FAST_REVISION}`.",
        f"- GlobalPIQA: byte-identical to research files generated by current official `global_piqa/dl.py`; upstream dataset revisions are `{EXPECTED_GLOBALPIQA_SHAS['parallel']}` for parallel and `{EXPECTED_GLOBALPIQA_SHAS['nonparallel']}` for nonparallel.",
        f"- EWoK: byte-identical to the expanded INITIAL_MODEL_STUDIES tree and every plaintext `.jsonl` matches the current official encrypted `ewok_fast.zip` central-directory filename, uncompressed size, and CRC32; zip metadata revision is `{EXPECTED_FAST_REVISION}`.",
        "",
        f"Errors: {len(errors)}",
    ]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    lines += ["", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": rel(OUT_JSON), "md": rel(OUT_MD), "errors": len(errors)}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
