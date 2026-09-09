#!/usr/bin/env python3
"""research: audit evaluation_data coordinate used by inherited full wrapper.

The evaluation controller intentionally repairs EWoK and AoA by using the
pristine current official coordinate, while the inherited COMPACT_EXPERIENCE full-eval runner
still executes non-EWoK zero-shot, Reading, and SuperGLUE tasks from the older
INITIAL_MODEL_STUDIES checkout. This CPU-only audit compares file inventories, hashes, and line
counts between:

- old runner checkout: experiments/archive/initial_model_studies/repos/babylm-eval/strict
- pristine current checkout: experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict

It flags any dataset differences beyond the already-known EWoK coordinate and
AoA context-loading behavior before corrected-tokenizer official evaluation is
launched.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/eval_dataset_coordinate_audit.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OLD_STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
PRISTINE_STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
OUT = _public_path('experiments/archive/representation_and_objectives/data/eval_dataset_coordinate_audit/eval_dataset_coordinate_audit.json')
NOTE = _public_path('research/notes/representation_and_objectives/eval_dataset_coordinate_audit.md')

# Task directories used by the evaluation controllers.
TASK_DIRS = {
    "BLiMP": "evaluation_data/full_eval/blimp_filtered",
    "Supplement": "evaluation_data/full_eval/supplement_filtered",
    "EWoK": "evaluation_data/full_eval/ewok_filtered",
    "Entity": "evaluation_data/full_eval/entity_tracking",
    "COMPS": "evaluation_data/full_eval/comps",
    "GlobalPIQA_parallel": "evaluation_data/full_eval/global_piqa_parallel",
    "GlobalPIQA_nonparallel": "evaluation_data/full_eval/global_piqa_nonparallel",
    "Reading": "evaluation_data/full_eval/reading",
    "SuperGLUE": "evaluation_data/full_eval/glue_filtered",
    "AoA": "evaluation_data/full_eval/aoa",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix.lower() not in {".jsonl", ".csv", ".txt", ".tsv", ".json"}:
        return None
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "files": {}}
    files = {}
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rp = str(p.relative_to(root))
        files[rp] = {"size_bytes": p.stat().st_size, "sha256": sha256(p), "line_count": line_count(p)}
    return {"exists": True, "file_count": len(files), "files": files}


def compare_task(name: str, rel_dir: str) -> dict[str, Any]:
    old_root = OLD_STRICT / rel_dir
    new_root = PRISTINE_STRICT / rel_dir
    old = inventory(old_root)
    new = inventory(new_root)
    old_files = set(old.get("files", {}))
    new_files = set(new.get("files", {}))
    shared = sorted(old_files & new_files)
    only_old = sorted(old_files - new_files)
    only_new = sorted(new_files - old_files)
    diff_hash = []
    diff_size = []
    diff_lines = []
    for rp in shared:
        of = old["files"][rp]
        nf = new["files"][rp]
        if of["sha256"] != nf["sha256"]:
            diff_hash.append(rp)
        if of["size_bytes"] != nf["size_bytes"]:
            diff_size.append(rp)
        if of.get("line_count") != nf.get("line_count"):
            diff_lines.append(rp)
    return {
        "task": name,
        "relative_dir": rel_dir,
        "old_root": rel(old_root),
        "pristine_root": rel(new_root),
        "old_exists": old.get("exists"),
        "pristine_exists": new.get("exists"),
        "old_file_count": old.get("file_count", 0),
        "pristine_file_count": new.get("file_count", 0),
        "only_old": only_old,
        "only_pristine": only_new,
        "shared_file_count": len(shared),
        "diff_hash_files": diff_hash,
        "diff_size_files": diff_size,
        "diff_line_count_files": diff_lines,
        "identical_inventory_and_hashes": bool(old.get("exists") and new.get("exists") and not only_old and not only_new and not diff_hash),
        "old_total_lines_jsonl_csv_txt": sum(v.get("line_count") or 0 for v in old.get("files", {}).values()),
        "pristine_total_lines_jsonl_csv_txt": sum(v.get("line_count") or 0 for v in new.get("files", {}).values()),
    }


def main() -> None:
    tasks = {name: compare_task(name, rel_dir) for name, rel_dir in TASK_DIRS.items()}
    known_repaired = {"EWoK", "AoA"}
    unrepaired_diffs = {
        name: rec for name, rec in tasks.items()
        if name not in known_repaired and not rec["identical_inventory_and_hashes"]
    }
    payload = {
        "status": "EVAL_DATASET_COORDINATE_AUDIT",
        "created_utc": now_utc(),
        "old_strict": rel(OLD_STRICT),
        "pristine_strict": rel(PRISTINE_STRICT),
        "tasks": tasks,
        "known_repaired_coordinate_diffs": sorted(known_repaired),
        "unrepaired_task_differences": unrepaired_diffs,
        "safe_to_use_inherited_wrapper_for_non_ewok_non_aoa": len(unrepaired_diffs) == 0,
        "interpretation": "If safe_to_use is true, the inherited wrapper's non-EWoK/non-AoA datasets match the current pristine official checkout byte-for-byte; EWoK and AoA remain separately repaired by research. If false, patch the evaluation controller before launching corrected-tokenizer full evaluation.",
    }
    _public_path('experiments/archive/representation_and_objectives/data/eval_dataset_coordinate_audit').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research evaluation dataset coordinate audit",
        "",
        f"Old runner checkout: `{rel(OLD_STRICT)}`",
        f"Pristine current checkout: `{rel(PRISTINE_STRICT)}`",
        "",
        f"Safe for inherited wrapper on non-EWoK/non-AoA: `{payload['safe_to_use_inherited_wrapper_for_non_ewok_non_aoa']}`",
        "",
        "| Task | identical hashes | old files | pristine files | old lines | pristine lines | diff hashes | only old | only pristine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rec in tasks.items():
        lines.append(
            f"| {name} | {rec['identical_inventory_and_hashes']} | {rec['old_file_count']} | {rec['pristine_file_count']} | {rec['old_total_lines_jsonl_csv_txt']} | {rec['pristine_total_lines_jsonl_csv_txt']} | {len(rec['diff_hash_files'])} | {len(rec['only_old'])} | {len(rec['only_pristine'])} |"
        )
    if unrepaired_diffs:
        lines.extend(["", "Unrepaired differences were found; inspect the JSON before evaluation."])
    else:
        lines.extend(["", "No unrepaired dataset-coordinate differences were found. EWoK/AoA differences remain handled by the research repair scripts."])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT),
        "note": rel(NOTE),
        "safe_to_use_inherited_wrapper_for_non_ewok_non_aoa": payload["safe_to_use_inherited_wrapper_for_non_ewok_non_aoa"],
        "unrepaired_task_differences": sorted(unrepaired_diffs),
        "known_repaired": sorted(known_repaired),
    }, indent=2), flush=True)
    if unrepaired_diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
