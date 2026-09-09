#!/usr/bin/env python3
"""research: save interim dense official-payload provenance checks.

The dense official jobs may write payload fields incrementally.  This script records
which checkpoint each staged model symlink resolves to, full model.safetensors SHA256,
and whether SuperGLUE is complete or still only an interim subset.  It does not
claim terminal evaluation status.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
import pathlib
import time
from typing import Any, Dict

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT = _public_path('experiments/archive/functional_learning/data/dense_interim_payload_provenance')
SEEDS = {
    "seed62064": {
        "expected_source": "experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080",
        "staged": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/staged_model_roots/dense_focus_seed62064_u0080/hf_model/final",
        "payload": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
    },
    "seed62065": {
        "expected_source": "experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080",
        "staged": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/staged_model_roots/dense_focus_seed62065_u0080/hf_model/final",
        "payload": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
    },
}
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def as_path(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def payload_score(rec: Dict[str, Any], col: str) -> Any:
    tasks = rec.get("tasks") or {}
    if col == "Reading":
        return (tasks.get("Reading") or {}).get("scores", {}).get("Reading")
    if col == "GlobalPIQA":
        vals = [(tasks.get(k) or {}).get("score") for k in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]]
        return None if any(v is None for v in vals) else sum(float(v) for v in vals) / 2.0
    if col == "SuperGLUE":
        return (tasks.get("SuperGLUE") or {}).get("superglue_mean")
    if col == "AoA":
        aoa = tasks.get("AoA") or {}
        return aoa.get("aoa_leaderboard_score") if aoa.get("aoa_leaderboard_score") is not None else aoa.get("aoa_for_provisional_overall")
    return (tasks.get(col) or {}).get("score")


def inspect_seed(seed: str, spec: Dict[str, str]) -> Dict[str, Any]:
    src = as_path(spec["expected_source"])
    staged = as_path(spec["staged"])
    payload_path = as_path(spec["payload"])
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.exists() else {}
    sg = (payload.get("tasks") or {}).get("SuperGLUE") or {}
    sg_entries = sg.get("tasks") or []
    sg_done = [str(x.get("task")) for x in sg_entries if isinstance(x, dict) and x.get("returncode") == 0 and x.get("predictions")]
    zero_present = []
    for col in ZERO_COLS:
        t = (payload.get("tasks") or {}).get(col) or {}
        if col == "Reading":
            if (t.get("scores") or {}).get("Reading") is not None and t.get("predictions"):
                zero_present.append(col)
        elif t.get("score") is not None and t.get("predictions"):
            zero_present.append(col)
    return {
        "expected_source": rel(src),
        "staged": rel(staged),
        "staged_is_symlink": staged.is_symlink(),
        "staged_symlink_target": os.readlink(staged) if staged.is_symlink() else None,
        "source_model_sha256": sha256_file(src / "model.safetensors"),
        "staged_model_sha256": sha256_file(staged / "model.safetensors"),
        "payload": rel(payload_path),
        "payload_target": payload.get("target"),
        "payload_model_path": payload.get("model_path"),
        "zero_reading_columns_present": zero_present,
        "zero_reading_complete": set(zero_present) == set(ZERO_COLS),
        "superglue_entries_done": sg_done,
        "superglue_complete": set(sg_done) == set(SUPERGLUE_TASKS) and len(sg.get("superglue_primary_metric_details") or []) == len(SUPERGLUE_TASKS),
        "superglue_current_field": sg.get("superglue_mean"),
        "superglue_primary_metric_details_count": len(sg.get("superglue_primary_metric_details") or []),
        "aoa_present": "AoA" in (payload.get("tasks") or {}),
        "aoa_value": payload_score(payload, "AoA") if payload else None,
        "scores_current_payload": {col: payload_score(payload, col) for col in ZERO_COLS + ["GlobalPIQA", "SuperGLUE", "AoA"]},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = {seed: inspect_seed(seed, spec) for seed, spec in SEEDS.items()}
    result = {
        "status": "DENSE_INTERIM_PAYLOAD_PROVENANCE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "interpretation": "Payloads are being written incrementally by official evaluations. Treat zero-shot/Reading columns with prediction files as completed-current columns, but do not treat SuperGLUE as complete until all seven tasks and primary-metric details are present, and do not treat Overall/AoA as final until AoA is recorded by the evaluator and the evaluation is complete.",
    }
    out_json = _public_path('experiments/archive/functional_learning/data/dense_interim_payload_provenance/dense_interim_payload_provenance.json')
    out_md = _public_path('research/documents/functional_learning/data/dense_interim_payload_provenance/dense_interim_payload_provenance.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dense interim payload provenance\n"]
    for seed, rec in seeds.items():
        lines.append(f"## {seed}")
        lines.append(f"- staged symlink: `{rec['staged']}` -> `{rec['staged_symlink_target']}`; symlink `{rec['staged_is_symlink']}`.")
        lines.append(f"- source SHA256: `{rec['source_model_sha256']}`; staged SHA256: `{rec['staged_model_sha256']}`.")
        lines.append(f"- zero-shot/Reading complete-current columns: `{rec['zero_reading_columns_present']}`.")
        lines.append(f"- SuperGLUE tasks currently done: `{rec['superglue_entries_done']}`; complete `{rec['superglue_complete']}`; current field `{rec['superglue_current_field']}`; primary detail count `{rec['superglue_primary_metric_details_count']}`.")
        lines.append(f"- AoA present `{rec['aoa_present']}`, AoA value `{rec['aoa_value']}`.\n")
    lines.append("Interpretation: do not use current SuperGLUE fields as completed official means; current BoolQ-only equality across seeds is not evidence of full SuperGLUE equality.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
