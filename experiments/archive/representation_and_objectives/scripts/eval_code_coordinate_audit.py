#!/usr/bin/env python3
"""research: compare evaluation code coordinate between INITIAL_MODEL_STUDIES runner and pristine current checkout.

The corrected-tokenizer full evaluation wrappers inherited COMPACT_EXPERIENCE's runner, which
runs zero-shot, Reading and SuperGLUE commands from the INITIAL_MODEL_STUDIES babylm-eval
checkout, while the corrected evaluation uses a staged pristine current checkout. EWoK data and AoA
are separately repaired, but if the runner code differs for non-EWoK tasks then
corrected-tokenizer results would not be on the current official coordinate.

This CPU-only audit compares relevant .py files by hash and writes a small note.
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

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/eval_code_coordinate_audit.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OLD_STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
PRISTINE_STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
OUT = _public_path('experiments/archive/representation_and_objectives/data/eval_code_coordinate_audit/eval_code_coordinate_audit.json')
NOTE = _public_path('research/notes/representation_and_objectives/eval_code_coordinate_audit.md')

CODE_GROUPS = {
    "sentence_zero_shot": "evaluation_pipeline/sentence_zero_shot",
    "reading": "evaluation_pipeline/reading",
    "finetune": "evaluation_pipeline/finetune",
    "global_piqa": "evaluation_pipeline/global_piqa",
    "collator_scoring": "evaluation_pipeline/collate_preds.py",
    "calculate_results": "evaluation_pipeline/calculate_results_from_pred.py",
    "utils": "evaluation_pipeline/utils.py",
    "aoa_word": "evaluation_pipeline/AoA_word",
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


def inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": {}}
    files = {}
    if path.is_file():
        targets = [path]
        root = path.parent
    else:
        targets = sorted(p for p in path.rglob("*.py") if p.is_file())
        root = path
    for p in targets:
        rp = str(p.relative_to(root)) if path.is_dir() else p.name
        txt = p.read_text(encoding="utf-8", errors="replace")
        files[rp] = {"size_bytes": p.stat().st_size, "sha256": sha256(p), "num_lines": txt.count("\n") + (0 if txt.endswith("\n") or txt == "" else 1)}
    return {"exists": True, "file_count": len(files), "files": files}


def compare_group(name: str, rel_path: str) -> dict[str, Any]:
    a_path = OLD_STRICT / rel_path
    b_path = PRISTINE_STRICT / rel_path
    a = inventory(a_path)
    b = inventory(b_path)
    af = set(a.get("files", {}))
    bf = set(b.get("files", {}))
    only_a = sorted(af - bf)
    only_b = sorted(bf - af)
    diff = []
    line_diff = []
    for rp in sorted(af & bf):
        if a["files"][rp]["sha256"] != b["files"][rp]["sha256"]:
            diff.append(rp)
        if a["files"][rp].get("num_lines") != b["files"][rp].get("num_lines"):
            line_diff.append(rp)
    return {
        "group": name,
        "relative_path": rel_path,
        "old_path": rel(a_path),
        "pristine_path": rel(b_path),
        "old_exists": a.get("exists"),
        "pristine_exists": b.get("exists"),
        "old_file_count": a.get("file_count", 0),
        "pristine_file_count": b.get("file_count", 0),
        "only_old": only_a,
        "only_pristine": only_b,
        "diff_hash_files": diff,
        "diff_line_count_files": line_diff,
        "identical_py_hashes": bool(a.get("exists") and b.get("exists") and not only_a and not only_b and not diff),
    }


def main() -> None:
    groups = {name: compare_group(name, rp) for name, rp in CODE_GROUPS.items()}
    runner_relevant = ["sentence_zero_shot", "reading", "finetune", "global_piqa"]
    runner_diffs = {name: groups[name] for name in runner_relevant if not groups[name]["identical_py_hashes"]}
    scoring_diffs = {name: groups[name] for name in ["collator_scoring", "calculate_results", "utils", "aoa_word"] if not groups[name]["identical_py_hashes"]}
    payload = {
        "status": "EVAL_CODE_COORDINATE_AUDIT",
        "created_utc": now_utc(),
        "old_strict": rel(OLD_STRICT),
        "pristine_strict": rel(PRISTINE_STRICT),
        "groups": groups,
        "runner_relevant_diffs": runner_diffs,
        "scoring_or_repair_diffs": scoring_diffs,
        "safe_to_use_initial_model_studies_runner_code_for_non_ewok_non_aoa": len(runner_diffs) == 0,
        "interpretation": "If runner_relevant_diffs is empty, non-EWoK/non-AoA predictions from the inherited wrapper use code byte-identical to the current pristine checkout. Scoring/collator differences are still resolved by research pristine collation and official AoA repair.",
    }
    _public_path('experiments/archive/representation_and_objectives/data/eval_code_coordinate_audit').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research evaluation code coordinate audit",
        "",
        f"Old runner checkout: `{rel(OLD_STRICT)}`",
        f"Pristine current checkout: `{rel(PRISTINE_STRICT)}`",
        "",
        f"Safe to use INITIAL_MODEL_STUDIES runner code for non-EWoK/non-AoA: `{payload['safe_to_use_initial_model_studies_runner_code_for_non_ewok_non_aoa']}`",
        "",
        "| group | identical py hashes | old files | pristine files | diff hashes | only old | only pristine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rec in groups.items():
        lines.append(f"| {name} | {rec['identical_py_hashes']} | {rec['old_file_count']} | {rec['pristine_file_count']} | {len(rec['diff_hash_files'])} | {len(rec['only_old'])} | {len(rec['only_pristine'])} |")
    if runner_diffs:
        lines.extend(["", "Runner-relevant code differences found; inspect JSON and patch evaluation before corrected full eval."])
    else:
        lines.extend(["", "Runner-relevant code for sentence zero-shot, Reading, finetune, and GlobalPIQA is byte-identical between the inherited and pristine checkouts."])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT),
        "note": rel(NOTE),
        "safe_to_use_initial_model_studies_runner_code_for_non_ewok_non_aoa": payload["safe_to_use_initial_model_studies_runner_code_for_non_ewok_non_aoa"],
        "runner_relevant_diffs": sorted(runner_diffs),
        "scoring_or_repair_diffs": sorted(scoring_diffs),
    }, indent=2), flush=True)
    if runner_diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
