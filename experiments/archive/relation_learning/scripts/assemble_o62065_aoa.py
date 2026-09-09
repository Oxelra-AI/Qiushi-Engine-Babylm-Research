#!/usr/bin/env python3
"""Assemble and score O62065 AoA trajectory from shared ancestry plus endpoint extraction.

This is a narrow closing-step utility.  It reuses the established research assembly
semantics but makes the O62065 endpoint explicit because it lives in relation_learning, not
in the original endpoint table.  The output distinguishes completed measured
AoA zero from missing evidence and also records the raw official-semantics Pearson
fit before leaderboard clipping.
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
import sys
import time
from collections import Counter
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

ROOT = _public_path('.')
DEFAULT_SHARED_DIR = _public_path('experiments/archive/functional_learning/data/batched_aoa_shared/full_retry')
DEFAULT_ENDPOINT_DIR = _public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract')
DEFAULT_CHECKPOINT = _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/o62065_aoa_measured')
research = _public_path('experiments/archive/functional_learning/scripts/shared_aoa.py')
FIT_OFFICIAL = _public_path('experiments/archive/relation_learning/scripts/refit_raw_aoa_endpoints_official.py')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_manifest(path: pathlib.Path, expected_mode: str) -> dict[str, Any]:
    man_path = path / "manifest.json"
    surp_path = path / "surprisal.json"
    out: dict[str, Any] = {
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
    summ = man.get("summary") or {}
    out.update({
        "status": man.get("status"),
        "mode": man.get("mode"),
        "target": man.get("target"),
        "steps": man.get("steps"),
        "word_counts": man.get("word_counts"),
        "expected_results": man.get("expected_results"),
        "eval_info": man.get("eval_info"),
        "summary": summ,
    })
    out["ok"] = bool(
        man.get("status") == "BATCHED_AOA_EXTRACTION_COMPLETE"
        and man.get("mode") == expected_mode
        and summ.get("missing_steps") == []
        and summ.get("steps_with_wrong_counts") == {}
        and summ.get("n_nan_or_nonfinite") == 0
        and summ.get("n_finite") == summ.get("n_results")
    )
    if not out["ok"]:
        out["reason"] = "manifest_not_complete_or_unexpected"
    return out


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


def write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    official = payload.get("official_score", {})
    raw = payload.get("raw_fit_summary", {})
    assembled = payload.get("assembled", {})
    lines = [
        "# research O62065 AoA measured trajectory", "",
        "This file assembles the already-measured shared 1M..80M ancestry with the O62065 endpoint extraction. It is a measurement artifact, not a new training run.", "",
        "## Result", "",
        f"- Complete measured evidence: `{payload.get('complete_measured_evidence')}`",
        f"- Assembled records: `{assembled.get('summary', {}).get('n_results')}` across `{len(assembled.get('steps', []))}` steps",
        f"- Official raw curve value: `{official.get('aoa_raw_correlation')}`",
        f"- Official leaderboard AoA score: `{official.get('aoa_leaderboard_score')}`",
        f"- Official legitimate zero: `{official.get('legitimate_zero')}`",
        f"- Raw official-semantics Pearson r: `{raw.get('raw_pearson_r')}`",
        f"- Raw official-semantics Pearson p: `{raw.get('raw_pearson_p')}`",
        f"- Fitted words: `{raw.get('fitted_word_count')}`", "",
        "## Inputs", "",
        f"- Shared ancestry: `{payload.get('shared_raw', {}).get('dir')}`",
        f"- Endpoint extraction: `{payload.get('endpoint_raw', {}).get('dir')}`",
        f"- O62065 checkpoint: `{payload.get('checkpoint')}`", "",
        "## Outputs", "",
        f"- Assembled surprisal: `{assembled.get('surprisal_path')}`",
        f"- AoA score manifest: `{official.get('aoa_score_manifest')}`",
        f"- Raw word fits: `{payload.get('outputs', {}).get('raw_word_fits_csv')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-dir", type=pathlib.Path, default=DEFAULT_SHARED_DIR)
    ap.add_argument("--endpoint-dir", type=pathlib.Path, default=DEFAULT_ENDPOINT_DIR)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--label", default="o62065")
    ap.add_argument("--words", type=int, default=89_168_037)
    args = ap.parse_args()

    shared_dir = args.shared_dir if args.shared_dir.is_absolute() else ROOT / args.shared_dir
    endpoint_dir = args.endpoint_dir if args.endpoint_dir.is_absolute() else ROOT / args.endpoint_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    aoa_word_dir = out_dir / "AoA_word"
    aoa_word_dir.mkdir(parents=True, exist_ok=True)

    shared_raw = validate_manifest(shared_dir, "shared")
    endpoint_raw = validate_manifest(endpoint_dir, "endpoint")
    if not shared_raw.get("ok") or not endpoint_raw.get("ok"):
        raise RuntimeError(json.dumps({"shared": shared_raw, "endpoint": endpoint_raw}, indent=2))
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(checkpoint / "config.json")

    shared_data = load_json(shared_dir / "surprisal.json")
    endpoint_data = load_json(endpoint_dir / "surprisal.json")
    shared_steps = list(shared_raw.get("steps") or [])
    shared_words = list(shared_raw.get("word_counts") or [])
    final_step = f"endpoint_{args.words / 1_000_000:.6f}M"
    endpoint_rows = [dict(r) for r in endpoint_data.get("results", [])]
    for r in endpoint_rows:
        r["step"] = final_step
        r["word_count"] = int(args.words)
    rows = list(shared_data.get("results", [])) + endpoint_rows
    all_steps = shared_steps + [final_step]
    all_words = shared_words + [int(args.words)]
    expected_contexts = int((shared_raw.get("eval_info") or {}).get("contexts_evaluated", 8005))
    summary = result_summary(rows, all_steps, expected_contexts)
    assembled = {
        "metadata": {
            "model_name": args.label,
            "use_bos_only": False,
            "total_steps": len(all_steps),
            "completed_steps": len(set(r.get("step") for r in rows)),
            "assembled_by": rel(_public_path('experiments/archive/relation_learning/scripts/assemble_o62065_aoa.py')),
            "shared_ancestry_source": rel(shared_dir / "surprisal.json"),
            "endpoint_source": rel(endpoint_dir / "surprisal.json"),
            "early_stop_convention": "Shared true ancestry through chck_1M..chck_80M plus exact O62065 endpoint; no borrowed post-branch states.",
        },
        "results": rows,
    }
    assembled_surp = aoa_word_dir / "surprisal.json"
    # Reuse official JsonProcessor for format stability.
    S81 = import_module(research, "shared_aoa_for_step130")
    S81.JsonProcessor.save_json(assembled, assembled_surp)
    official_score = S81.score_trajectory(assembled, checkpoint, aoa_word_dir)
    official_score["aoa_score_manifest"] = rel(aoa_word_dir / "aoa_score_manifest.json")

    FIT = import_module(FIT_OFFICIAL, "official_fit_for_step130")
    child_aoas = FIT.load_cdi_child_aoas(FIT.CDI_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    raw_fit_summary, raw_rows = FIT.fit_endpoint(args.label, assembled_surp, child_aoas, tokenizer)
    raw_word_fits_csv = out_dir / f"{args.label}_raw_word_fits.csv"
    pd.DataFrame(raw_rows).to_csv(raw_word_fits_csv, index=False)

    complete = bool(
        summary["missing_steps"] == []
        and summary["steps_with_wrong_counts"] == {}
        and summary["n_nan_or_nonfinite"] == 0
        and official_score.get("measured") is True
    )
    payload = {
        "status": "O62065_AOA_MEASURED" if complete else "O62065_AOA_INCOMPLETE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/relation_learning/scripts/assemble_o62065_aoa.py')),
        "checkpoint": rel(checkpoint),
        "label": args.label,
        "endpoint_words": int(args.words),
        "shared_raw": shared_raw,
        "endpoint_raw": endpoint_raw,
        "assembled": {
            "surprisal_path": rel(assembled_surp),
            "surprisal_sha256": sha256_file(assembled_surp),
            "steps": all_steps,
            "word_counts": all_words,
            "summary": summary,
        },
        "official_score": official_score,
        "raw_fit_summary": raw_fit_summary,
        "complete_measured_evidence": complete,
        "outputs": {
            "summary_json": rel(out_dir / "summary.json"),
            "summary_md": rel(out_dir / "summary.md"),
            "raw_word_fits_csv": rel(raw_word_fits_csv),
            "assembled_surprisal": rel(assembled_surp),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_dir / "summary.md", payload)
    print(json.dumps({"status": payload["status"], "complete": complete, "official_aoa": official_score.get("aoa_leaderboard_score"), "raw_r": raw_fit_summary.get("raw_pearson_r"), "raw_p": raw_fit_summary.get("raw_pearson_p"), "n_results": summary.get("n_results"), "out": rel(out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
