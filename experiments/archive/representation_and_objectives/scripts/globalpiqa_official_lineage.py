#!/usr/bin/env python3
"""research: reproduce GlobalPIQA official generated files and compare with inherited INITIAL_MODEL_STUDIES assets.

The BabyLM Strict-Evals HF snapshot does not ship generated GlobalPIQA eng_latn.jsonl
files. The official strict/evaluation_pipeline/global_piqa/dl.py generates them from
mrlbenchmarks/global-piqa-parallel and mrlbenchmarks/global-piqa-nonparallel. This script
runs that unmodified official generator in a scratch directory with local caches,
then compares the output byte/line-for-line against the inherited INITIAL_MODEL_STUDIES-generated files
that were used for seed43022 scoring.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from huggingface_hub import HfApi

ROOT = pathlib.Path(".").resolve()
STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
OUT = STUDY / "data/globalpiqa_official_lineage"
SCRATCH = OUT / "official_dl_scratch"
GEN_ROOT = SCRATCH / "generated_by_current_official_dl"
PRISTINE_STRICT = STUDY / "data/pristine_official_coordinate/babylm-eval/strict"
DL_PY = PRISTINE_STRICT / "evaluation_pipeline/global_piqa/dl.py"
COLLATOR = PRISTINE_STRICT / "evaluation_pipeline/collate_preds.py"
INITIAL_MODEL_STUDIES_STRICT = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
DATASET_REPOS = {
    "parallel": "mrlbenchmarks/global-piqa-parallel",
    "nonparallel": "mrlbenchmarks/global-piqa-nonparallel",
}
TASK_DIRS = {
    "parallel": "global_piqa_parallel",
    "nonparallel": "global_piqa_nonparallel",
}
BASE_DIRS = ["full_eval", "fast_eval"]
EXPECTED_COUNTS = {"parallel": 103, "nonparallel": 100}


@dataclass
class FileInfo:
    path: str
    exists: bool
    size: int | None = None
    sha256: str | None = None
    num_lines: int | None = None
    first_line: str | None = None
    last_line: str | None = None


def sha256_bytes(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: pathlib.Path) -> FileInfo:
    if not path.exists():
        return FileInfo(path=str(path), exists=False)
    data = path.read_text(encoding="utf-8").splitlines()
    return FileInfo(
        path=str(path),
        exists=True,
        size=path.stat().st_size,
        sha256=sha256_bytes(path),
        num_lines=len(data),
        first_line=data[0] if data else None,
        last_line=data[-1] if data else None,
    )


def short_repo_info(api: HfApi, repo_id: str) -> dict[str, Any]:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    siblings = sorted(s.rfilename for s in info.siblings)
    eng_candidates = [s for s in siblings if "eng_latn" in s or "eng" in s.lower()]
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "sha": getattr(info, "sha", None),
        "last_modified": str(getattr(info, "lastModified", None)),
        "private": getattr(info, "private", None),
        "gated": getattr(info, "gated", None),
        "num_files": len(siblings),
        "eng_candidate_files": eng_candidates,
        "first_20_files": siblings[:20],
    }


def compare_files(a: pathlib.Path, b: pathlib.Path) -> dict[str, Any]:
    ai = file_info(a)
    bi = file_info(b)
    out: dict[str, Any] = {"a": asdict(ai), "b": asdict(bi)}
    if not (a.exists() and b.exists()):
        out["identical_bytes"] = False
        out["identical_lines"] = False
        return out
    ab = a.read_bytes()
    bb = b.read_bytes()
    al = a.read_text(encoding="utf-8").splitlines()
    bl = b.read_text(encoding="utf-8").splitlines()
    out["identical_bytes"] = ab == bb
    out["identical_lines"] = al == bl
    out["line_count_delta"] = len(al) - len(bl)
    if al != bl:
        first_diff = None
        for i, (x, y) in enumerate(zip(al, bl), start=1):
            if x != y:
                first_diff = {"line": i, "a": x, "b": y}
                break
        if first_diff is None and len(al) != len(bl):
            first_diff = {"line": min(len(al), len(bl)) + 1, "a": al[min(len(al), len(bl))] if len(al) > len(bl) else None, "b": bl[min(len(al), len(bl))] if len(bl) > len(al) else None}
        out["first_diff"] = first_diff
    return out


def collect_jsonl_keys(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    ids = []
    for r in rows:
        ids.append(r.get("example_id") or r.get("id") or r.get("idx") or r.get("question"))
    return {
        "exists": True,
        "num_rows": len(rows),
        "keys": keys,
        "first_row": rows[0] if rows else None,
        "last_row": rows[-1] if rows else None,
        "example_id_count": sum(1 for x in ids if x is not None),
        "first_5_ids": ids[:5],
        "last_5_ids": ids[-5:],
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    GEN_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cache_root = OUT / "hf_cache"
    env.update({
        "HF_HOME": str(cache_root / "hf_home"),
        "HF_HUB_CACHE": str(cache_root / "hf_hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hf_hub"),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    for key in ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_infos = {kind: short_repo_info(api, repo) for kind, repo in DATASET_REPOS.items()}

    dl_hash = sha256_bytes(DL_PY)
    collator_hash = sha256_bytes(COLLATOR)
    initial_model_studies_dl = INITIAL_MODEL_STUDIES_STRICT / "evaluation_pipeline/global_piqa/dl.py"
    official_dl_compare = compare_files(DL_PY, initial_model_studies_dl)

    cmd = [sys.executable, "-B", str((ROOT / DL_PY).resolve())]
    proc = subprocess.run(
        cmd,
        cwd=str((ROOT / GEN_ROOT).resolve()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )

    generated: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    row_views: dict[str, Any] = {}
    inherited_full_fast_equivalence: dict[str, Any] = {}
    generated_full_fast_equivalence: dict[str, Any] = {}
    for kind, task_dir in TASK_DIRS.items():
        generated[kind] = {}
        comparisons[kind] = {}
        row_views[kind] = {}
        for base in BASE_DIRS:
            gen_file = GEN_ROOT / "evaluation_data" / base / task_dir / "eng_latn.jsonl"
            old_file = INITIAL_MODEL_STUDIES_STRICT / "evaluation_data" / base / task_dir / "eng_latn.jsonl"
            generated[kind][base] = asdict(file_info(gen_file))
            comparisons[kind][base] = compare_files(gen_file, old_file)
            row_views[kind][base] = collect_jsonl_keys(gen_file)
        inherited_full_fast_equivalence[kind] = compare_files(
            INITIAL_MODEL_STUDIES_STRICT / "evaluation_data/full_eval" / task_dir / "eng_latn.jsonl",
            INITIAL_MODEL_STUDIES_STRICT / "evaluation_data/fast_eval" / task_dir / "eng_latn.jsonl",
        )
        generated_full_fast_equivalence[kind] = compare_files(
            GEN_ROOT / "evaluation_data/full_eval" / task_dir / "eng_latn.jsonl",
            GEN_ROOT / "evaluation_data/fast_eval" / task_dir / "eng_latn.jsonl",
        )

    checks = {
        "official_dl_exists": DL_PY.exists(),
        "official_dl_sha256": dl_hash,
        "collator_sha256": collator_hash,
        "initial_model_studies_dl_identical_to_pristine_dl": official_dl_compare.get("identical_bytes"),
        "official_dl_returncode": proc.returncode,
        "all_generated_files_exist": all(generated[k][b]["exists"] for k in generated for b in generated[k]),
        "all_generated_counts_match_collator_expectation": all(generated[k][b]["num_lines"] == EXPECTED_COUNTS[k] for k in generated for b in generated[k]),
        "all_generated_match_inherited_bytes": all(comparisons[k][b].get("identical_bytes") for k in comparisons for b in comparisons[k]),
        "generated_full_equals_fast": all(generated_full_fast_equivalence[k].get("identical_bytes") for k in generated_full_fast_equivalence),
        "inherited_full_equals_fast": all(inherited_full_fast_equivalence[k].get("identical_bytes") for k in inherited_full_fast_equivalence),
    }

    summary = {
        "status": "GLOBALPIQA_OFFICIAL_LINEAGE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Tie GlobalPIQA eng_latn evaluation files used in seed43022 official-coordinate scoring to the current official generator and upstream mrlbenchmarks dataset revisions.",
        "official_procedure": {
            "pristine_strict_root": str(PRISTINE_STRICT),
            "official_global_piqa_dl_py": str(DL_PY),
            "official_global_piqa_dl_sha256": dl_hash,
            "collator": str(COLLATOR),
            "collator_sha256": collator_hash,
            "official_dl_writes_base_dirs": BASE_DIRS,
            "official_dl_subsets": DATASET_REPOS,
            "expected_counts_from_collator": EXPECTED_COUNTS,
            "official_dl_identical_between_pristine_and_initial_model_studies": official_dl_compare,
        },
        "upstream_dataset_revisions": repo_infos,
        "generation_run": {
            "cmd": cmd,
            "cwd": str(GEN_ROOT),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-8000:],
            "scratch_root": str(GEN_ROOT),
            "cache_root": str(cache_root),
        },
        "generated_files": generated,
        "row_views": row_views,
        "comparisons_to_inherited_initial_model_studies_files": comparisons,
        "generated_full_fast_equivalence": generated_full_fast_equivalence,
        "inherited_full_fast_equivalence": inherited_full_fast_equivalence,
        "checks": checks,
        "interpretation": "If all_generated_match_inherited_bytes is true, the inherited INITIAL_MODEL_STUDIES GlobalPIQA files used in research scoring are byte-identical to files generated now by the current unmodified official dl.py from the current mrlbenchmarks/global-piqa-* dataset revisions; the official-coordinate margin does not rest on an untracked stale GlobalPIQA asset.",
        "elapsed_sec": time.time() - t0,
    }
    out_json = OUT / "globalpiqa_official_lineage.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "returncode": proc.returncode,
        "checks": checks,
        "out_json": str(out_json),
    }, indent=2), flush=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    if not checks["all_generated_files_exist"] or not checks["all_generated_counts_match_collator_expectation"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
