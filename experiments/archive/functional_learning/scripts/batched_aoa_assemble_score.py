#!/usr/bin/env python3
"""research: assemble and score AoA trajectories from batched raw-surprisal files.

research's original assembly code assumes its own extraction directories.  research
repaired the extraction bottleneck by producing raw `surprisal.json` files with the
same schema, first for endpoints and now for shared ancestry.  This script connects
those batched raw files back to the platform AoA scorer without changing scorer
semantics:

  * validate shared and endpoint extraction manifests;
  * assemble true early-stop trajectory = shared chck_1M..chck_80M plus exact endpoint;
  * score with research/research platform AoAEvaluator provenance;
  * write an explicit manifest distinguishing measured zero from missing evidence.

It can also run on smoke paths to validate the path plumbing without waiting for the
full shared ancestry extraction.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import time
from collections import Counter
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
research = _public_path('experiments/archive/functional_learning/scripts/shared_aoa.py')
DEFAULT_ENDPOINT_ROOT = _public_path('experiments/archive/functional_learning/data/batched_aoa_full_endpoints/endpoints')
DEFAULT_SHARED_DIR = _public_path('experiments/archive/functional_learning/data/batched_aoa_shared/full_retry')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/functional_learning/data/batched_aoa_measured')

TARGETS = ["coherent86", "dense_seed62064", "dense_seed62065"]


def import_step081():
    spec = importlib.util.spec_from_file_location("shared_aoa", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S81 = import_step081()
EARLY_STOP_NAMES = list(S81.EARLY_STOP_NAMES)
EARLY_STOP_WORDS = list(S81.EARLY_STOP_WORDS)
ENDPOINTS = S81.ENDPOINTS


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: pathlib.Path, block_size: int = 1 << 20) -> str | None:
    if not path.is_file():
        return None
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def endpoint_step_name(target: str) -> str:
    return f"endpoint_{int(ENDPOINTS[target]['words']) / 1_000_000:.6f}M"


def endpoint_dir_for(root: pathlib.Path, target: str, suffix: str) -> pathlib.Path:
    # research full wrapper writes endpoints/<target>/<suffix>.  Earlier smoke tests
    # wrote a single endpoint directory directly as <root>/<suffix>; accept both so
    # validation can reuse the exact smoke artifacts without copying them.
    primary = root / target / suffix
    if (primary / "manifest.json").is_file() or (primary / "surprisal.json").is_file():
        return primary
    fallback = root / suffix
    if (fallback / "manifest.json").is_file() or (fallback / "surprisal.json").is_file():
        return fallback
    return primary


def result_summary(rows: list[dict[str, Any]], expected_steps: list[str], expected_contexts: int) -> dict[str, Any]:
    by_step = Counter(r.get("step") for r in rows)
    finite = 0
    nonfinite = 0
    for r in rows:
        try:
            v = float(r.get("surprisal"))
            if math.isfinite(v):
                finite += 1
            else:
                nonfinite += 1
        except Exception:
            nonfinite += 1
    return {
        "n_results": len(rows),
        "n_finite": finite,
        "n_nan_or_nonfinite": nonfinite,
        "counts_by_step": dict(by_step),
        "missing_steps": [s for s in expected_steps if by_step.get(s, 0) == 0],
        "steps_with_wrong_counts": {s: by_step.get(s, 0) for s in expected_steps if by_step.get(s, 0) != expected_contexts},
        "first_result": rows[0] if rows else None,
        "last_result": rows[-1] if rows else None,
    }


def validate_manifest(path: pathlib.Path, expected_mode: str, expected_contexts: int | None = None) -> dict[str, Any]:
    man_path = path / "manifest.json"
    surp_path = path / "surprisal.json"
    out = {
        "dir": rel(path),
        "manifest_path": rel(man_path),
        "surprisal_path": rel(surp_path),
        "manifest_exists": man_path.is_file(),
        "surprisal_exists": surp_path.is_file(),
        "manifest_sha256": sha256_file(man_path),
        "surprisal_sha256": sha256_file(surp_path),
    }
    if not man_path.is_file() or not surp_path.is_file():
        out["ok"] = False
        out["reason"] = "missing_manifest_or_surprisal"
        return out
    man = load_json(man_path)
    out["status"] = man.get("status")
    out["mode"] = man.get("mode")
    out["target"] = man.get("target")
    out["expected_results"] = man.get("expected_results")
    out["summary"] = man.get("summary")
    out["eval_info"] = man.get("eval_info")
    out["steps"] = man.get("steps")
    out["word_counts"] = man.get("word_counts")
    wrong = (man.get("summary") or {}).get("steps_with_wrong_counts")
    missing = (man.get("summary") or {}).get("missing_steps")
    finite = (man.get("summary") or {}).get("n_finite")
    n_results = (man.get("summary") or {}).get("n_results")
    ok = (
        man.get("status") == "BATCHED_AOA_EXTRACTION_COMPLETE"
        and man.get("mode") == expected_mode
        and wrong == {}
        and missing == []
        and finite == n_results
        and (expected_contexts is None or int(man.get("eval_info", {}).get("contexts_evaluated", -1)) == int(expected_contexts))
    )
    out["ok"] = bool(ok)
    if not ok:
        out["reason"] = "manifest_not_complete_or_unexpected"
    return out


def assemble_target(shared_dir: pathlib.Path, endpoint_root: pathlib.Path, endpoint_suffix: str, out_root: pathlib.Path, target: str, allow_incomplete: bool = False) -> dict[str, Any]:
    shared_man = validate_manifest(shared_dir, "shared")
    ep_dir = endpoint_dir_for(endpoint_root, target, endpoint_suffix)
    endpoint_man = validate_manifest(ep_dir, "endpoint", expected_contexts=(shared_man.get("eval_info") or {}).get("contexts_evaluated") if shared_man.get("ok") else None)
    if not shared_man.get("ok") or not endpoint_man.get("ok"):
        if not allow_incomplete:
            raise RuntimeError({"shared": shared_man, "endpoint": endpoint_man})
    shared_data = load_json(shared_dir / "surprisal.json") if (shared_dir / "surprisal.json").is_file() else {"results": []}
    endpoint_data = load_json(ep_dir / "surprisal.json") if (ep_dir / "surprisal.json").is_file() else {"results": []}
    shared_rows = list(shared_data.get("results", []))
    endpoint_rows = [dict(r) for r in endpoint_data.get("results", [])]

    shared_steps = list(shared_man.get("steps") or [])
    # Keep only the prefix represented in the shared raw file: useful for smoke tests as well as full 17-step runs.
    n = len(shared_steps)
    shared_words = EARLY_STOP_WORDS[:n]
    final_step = endpoint_step_name(target)
    for r in endpoint_rows:
        r["step"] = final_step
        r["word_count"] = int(ENDPOINTS[target]["words"])
    all_steps = shared_steps + [final_step]
    all_words = shared_words + [int(ENDPOINTS[target]["words"])]
    rows = shared_rows + endpoint_rows
    expected_contexts = int((shared_man.get("eval_info") or {}).get("contexts_evaluated", 0))
    summary = result_summary(rows, all_steps, expected_contexts)

    out_dir = out_root / target / ("full" if endpoint_suffix == "full" and n == len(EARLY_STOP_NAMES) else f"{endpoint_suffix}_a{n}")
    aoa_word_dir = out_dir / "AoA_word"
    aoa_word_dir.mkdir(parents=True, exist_ok=True)
    assembled = {
        "metadata": {
            "model_name": target,
            "use_bos_only": bool((shared_data.get("metadata") or {}).get("use_bos_only", False)),
            "total_steps": len(all_steps),
            "completed_steps": len(set(r.get("step") for r in rows)),
            "assembled_by": rel(_public_path('experiments/archive/functional_learning/scripts/batched_aoa_assemble_score.py')),
            "raw_shared_source": rel(shared_dir / "surprisal.json"),
            "raw_endpoint_source": rel(ep_dir / "surprisal.json"),
            "early_stop_convention": "shared true ancestry through available chck_* steps plus exact candidate endpoint exposure; no borrowed post-branch states",
        },
        "results": rows,
    }
    assembled_surp = aoa_word_dir / "surprisal.json"
    S81.JsonProcessor.save_json(assembled, assembled_surp)
    score = S81.score_trajectory(assembled, ENDPOINTS[target]["path"], aoa_word_dir)
    complete = (
        bool(shared_man.get("ok"))
        and bool(endpoint_man.get("ok"))
        and summary["missing_steps"] == []
        and summary["steps_with_wrong_counts"] == {}
        and summary["n_nan_or_nonfinite"] == 0
        and score.get("measured") is True
    )
    manifest = {
        "status": "BATCHED_AOA_MEASURED" if complete else "BATCHED_AOA_INCOMPLETE",
        "created_utc": now(),
        "target": target,
        "out_dir": rel(out_dir),
        "early_stop_convention": "Use the true shared research slow-adapter ancestry through the shared batched raw file and the target's exact repaired endpoint. Dense endpoints stop at 89.168037M; coherent86 stops at 86.005295M.",
        "shared_raw": shared_man,
        "endpoint_raw": endpoint_man,
        "assembled": {
            "surprisal_path": rel(assembled_surp),
            "surprisal_sha256": sha256_file(assembled_surp),
            "steps": all_steps,
            "word_counts": all_words,
            "expected_results": len(all_steps) * expected_contexts,
            "summary": summary,
        },
        "aoa_estimator": S81.platform_aoa_evidence(),
        "score": score,
        "complete_measured_evidence": bool(complete),
        "aoa_payload_for_comparison": {
            "column": "AoA",
            "status": "measured" if score.get("measured") else "not_measured",
            "measured": bool(score.get("measured")),
            "estimator_id": S81.PLATFORM_AOA_ESTIMATOR_ID,
            "estimator_variant": S81.PLATFORM_AOA_VARIANT,
            "estimator_provenance_path": rel(S81.PLATFORM_AOA_PROVENANCE),
            "leaderboard_snapshot_path": rel(S81.PLATFORM_LEADERBOARD_SNAPSHOT),
            "aoa_raw_correlation": score.get("aoa_raw_correlation"),
            "aoa_leaderboard_score": score.get("aoa_leaderboard_score"),
            "legitimate_zero": bool(score.get("legitimate_zero")),
            "surprisal_json": rel(assembled_surp),
            "aoa_score_json": score.get("aoa_score_json"),
            "aoa_score_manifest": rel(aoa_word_dir / "aoa_score_manifest.json"),
            "n_results": summary["n_results"],
            "n_steps": len(all_steps),
            "n_contexts_evaluated": expected_contexts,
            "interpretation": "This AoA is measured only if raw shared and endpoint extractions are complete and platform AoAEvaluator scoring completed. A zero value is a completed scorer output, not a missing-checkpoint placeholder.",
        },
    }
    (out_dir / "aoa_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_dir / "aoa_manifest.md", manifest)
    return manifest


def write_md(path: pathlib.Path, man: dict[str, Any]) -> None:
    score = man.get("score", {})
    summ = man.get("assembled", {}).get("summary", {})
    lines = [
        f"# research batched AoA measured trajectory: {man.get('target')}\n\n",
        f"Status: `{man.get('status')}`\n\n",
        f"Complete measured evidence: `{man.get('complete_measured_evidence')}`\n\n",
        f"Estimator: `{man.get('aoa_estimator', {}).get('estimator_id')}`\n\n",
        f"Steps: `{', '.join(man.get('assembled', {}).get('steps', []))}`\n\n",
        f"Results: `{summ.get('n_results')}` finite `{summ.get('n_finite')}` nonfinite `{summ.get('n_nan_or_nonfinite')}`; missing `{summ.get('missing_steps')}` wrong-count `{summ.get('steps_with_wrong_counts')}`.\n\n",
        f"AoA raw correlation: `{score.get('aoa_raw_correlation')}`; leaderboard score: `{score.get('aoa_leaderboard_score')}`; measured `{score.get('measured')}`; legitimate zero `{score.get('legitimate_zero')}`; valid words `{score.get('n_valid_words')}`.\n\n",
        f"Assembled surprisal: `{man.get('assembled', {}).get('surprisal_path')}`\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shared-dir", type=pathlib.Path, default=DEFAULT_SHARED_DIR)
    ap.add_argument("--endpoint-root", type=pathlib.Path, default=DEFAULT_ENDPOINT_ROOT)
    ap.add_argument("--endpoint-suffix", default="full")
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for target in args.targets:
        manifests.append(assemble_target(args.shared_dir, args.endpoint_root, args.endpoint_suffix, args.out_root, target, allow_incomplete=args.allow_incomplete))
    report = {
        "status": "BATCHED_AOA_ASSEMBLY_DONE" if all(m.get("complete_measured_evidence") for m in manifests) else "BATCHED_AOA_ASSEMBLY_INCOMPLETE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/batched_aoa_assemble_score.py')),
        "shared_dir": rel(args.shared_dir),
        "endpoint_root": rel(args.endpoint_root),
        "endpoint_suffix": args.endpoint_suffix,
        "targets": args.targets,
        "target_manifests": [rel(args.out_root / m["target"] / ("full" if args.endpoint_suffix == "full" and len(m["assembled"]["steps"]) == len(EARLY_STOP_NAMES)+1 else f"{args.endpoint_suffix}_a{len(m['assembled']['steps'])-1}") / "aoa_manifest.json") for m in manifests],
        "scores": {m["target"]: {"complete": m.get("complete_measured_evidence"), "aoa": m.get("score", {}).get("aoa_leaderboard_score"), "raw": m.get("score", {}).get("aoa_raw_correlation"), "n_results": m.get("assembled", {}).get("summary", {}).get("n_results"), "n_steps": len(m.get("assembled", {}).get("steps", []))} for m in manifests},
    }
    out = args.out_root / "assembly_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": rel(out), "scores": report["scores"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
