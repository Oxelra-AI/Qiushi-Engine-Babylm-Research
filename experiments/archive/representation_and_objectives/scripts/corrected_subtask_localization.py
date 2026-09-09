#!/usr/bin/env python3
"""research: post-result localization for corrected-tokenizer endpoints.

CPU-only. Before corrected collations exist it writes a waiting record. After they
exist, compares corrected-tokenizer seed43022/43122 against inherited-tokenizer
references at the column and available subtask/detail levels. This is for
scientific interpretation and route repair, not final expression.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/corrected_subtask_localization.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/corrected_subtask_localization')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/corrected_subtask_localization/corrected_subtask_localization.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/corrected_subtask_localization.md')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
OLD = {
    "43022": _public_path('experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json'),
}
CORRECTED = {
    "43022": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json'),
}
TOKEN_SURFACE = _public_path('experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def extract_scores(obj: Any) -> dict[str, float]:
    candidates = []
    if isinstance(obj, dict):
        try:
            candidates.append(obj["score_summary"]["official_overall"]["scores"])
        except Exception:
            pass
        try:
            candidates.append(obj["score_summary"]["scores"])
        except Exception:
            pass
        ss = obj.get("score_summary")
        if isinstance(ss, dict):
            candidates.append(ss)
        candidates.append(obj.get("scores", obj))
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        out: dict[str, float] = {}
        for k in COLUMNS + ["Overall"]:
            x = finite(cand.get(k))
            if x is not None:
                out[k] = x
        if all(k in out for k in COLUMNS):
            if "Overall" not in out:
                out["Overall"] = sum(out[k] for k in COLUMNS) / 9.0
            return out
    return {}


def extract_details(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    ss = obj.get("score_summary")
    if not isinstance(ss, dict):
        return {}
    if isinstance(ss.get("details"), dict):
        return ss["details"]
    # Newer research score_collated may return details at the top level of score_summary.
    return {k: v for k, v in ss.items() if isinstance(v, dict)}


def load_summary(path: Path) -> dict[str, Any]:
    obj = load(path)
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "scores": extract_scores(obj),
        "details": extract_details(obj),
        "collated_sha256": obj.get("collated_sha256") if isinstance(obj, dict) else None,
        "status": obj.get("status") if isinstance(obj, dict) else None,
    }


def diff_dict(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in sorted(set(a) & set(b)) if finite(a[k]) is not None and finite(b[k]) is not None}


def top_abs(d: dict[str, float], n: int = 12) -> list[dict[str, Any]]:
    return [{"key": k, "delta": d[k]} for k in sorted(d, key=lambda k: abs(d[k]), reverse=True)[:n]]


def entity_scores(details: dict[str, Any]) -> dict[str, float]:
    e = details.get("Entity") or {}
    fs = e.get("subtask_fraction_scores") if isinstance(e, dict) else {}
    return {k: float(v) * 100.0 for k, v in (fs or {}).items() if finite(v) is not None}


def superglue_scores(details: dict[str, Any]) -> dict[str, float]:
    sg = details.get("SuperGLUE") or {}
    out: dict[str, float] = {}
    if isinstance(sg, dict):
        for task, rec in sg.items():
            if isinstance(rec, dict) and finite(rec.get("score")) is not None:
                out[task] = float(rec["score"])
    return out


def globalpiqa_scores(details: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = details.get(k) or {}
        if isinstance(r, dict) and finite(r.get("score")) is not None:
            out[k] = float(r["score"])
    return out


def reading_scores(details: dict[str, Any]) -> dict[str, float]:
    r = details.get("Reading") or {}
    out: dict[str, float] = {}
    if isinstance(r, dict):
        for k in ["self_paced_fraction", "eye_tracking_fraction", "leaderboard_score"]:
            if finite(r.get(k)) is not None:
                out[k] = float(r[k]) * (100.0 if k.endswith("fraction") else 1.0)
    return out


def detail_deltas(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in [("Entity", entity_scores), ("SuperGLUE", superglue_scores), ("GlobalPIQA", globalpiqa_scores), ("Reading", reading_scores)]:
        nd = fn(new)
        od = fn(old)
        dd = diff_dict(nd, od)
        out[name] = {
            "new": nd,
            "old": od,
            "delta_corrected_minus_inherited": dd,
            "largest_abs_delta": top_abs(dd, 10),
        }
    return out


def surface_rows() -> dict[str, Any]:
    ts = load(TOKEN_SURFACE)
    if not isinstance(ts, dict):
        return {}
    rows = ts.get("eval_occurrence_rows") or []
    by = {r.get("group"): r for r in rows if isinstance(r, dict)}
    return {
        k: {
            "new_over_old_token_ratio": (by.get(k) or {}).get("new_over_old_token_ratio"),
            "old_only_occ_pct": 100.0 * float((by.get(k) or {}).get("old_only_occ_fraction", 0.0)),
            "mean_token_delta": ((by.get(k) or {}).get("delta_len") or {}).get("mean"),
        }
        for k in ["family::EWoK", "family::COMPS", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::SuperGLUE", "family::BLiMP", "family::Supplement", "family::Entity"]
        if by.get(k)
    }


def write_md(payload: dict[str, Any]) -> None:
    lines = ["# research corrected-tokenizer subtask localization\n\n"]
    if payload["status"].endswith("WAITING"):
        lines.append("Corrected full official collations are not both present yet. This script is ready to rerun after both corrected-tokenizer evaluations complete.\n")
    else:
        lines.append("## Column movements corrected minus inherited\n\n")
        for seed, rec in payload["per_seed"].items():
            lines.append(f"### seed {seed}\n\n")
            for x in rec.get("column_largest_abs_movement", []):
                lines.append(f"- {x['key']}: {x['delta']:+.6f}\n")
            lines.append("\n")
        lines.append("## Available detail movements\n\n")
        for seed, rec in payload["per_seed"].items():
            lines.append(f"### seed {seed}\n\n")
            for detail, d in rec.get("detail_deltas", {}).items():
                lines.append(f"{detail}: " + ", ".join(f"{x['key']} {x['delta']:+.3f}" for x in d.get("largest_abs_delta", [])[:6]) + "\n\n")
    lines.append("\n## Files\n\n")
    for k, v in payload.get("files", {}).items():
        lines.append(f"- {k}: `{v}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = {s: load_summary(p) for s, p in OLD.items()}
    corr = {s: load_summary(p) for s, p in CORRECTED.items()}
    complete = [s for s, r in corr.items() if all(k in r["scores"] for k in COLUMNS)]
    per_seed: dict[str, Any] = {}
    for s in ["43022", "43122"]:
        col_delta = diff_dict(corr[s]["scores"], old[s]["scores"])
        per_seed[s] = {
            "old_path": old[s]["path"],
            "corrected_path": corr[s]["path"],
            "corrected_exists": corr[s]["exists"],
            "old_scores": old[s]["scores"],
            "corrected_scores": corr[s]["scores"],
            "column_delta_corrected_minus_inherited": col_delta,
            "column_largest_abs_movement": top_abs(col_delta, 12),
            "detail_deltas": detail_deltas(corr[s]["details"], old[s]["details"]) if corr[s]["details"] else {},
        }
    corrected_seed_delta = diff_dict(corr["43022"]["scores"], corr["43122"]["scores"]) if len(complete) == 2 else {}
    inherited_seed_delta = diff_dict(old["43022"]["scores"], old["43122"]["scores"])
    status = "CORRECTED_SUBTASK_LOCALIZATION_READY" if len(complete) == 2 else "CORRECTED_SUBTASK_LOCALIZATION_WAITING"
    payload = {
        "status": status,
        "created_utc": now_utc(),
        "complete_corrected_seeds": complete,
        "per_seed": per_seed,
        "corrected_seed43022_minus_seed43122_column_delta": corrected_seed_delta,
        "inherited_seed43022_minus_seed43122_column_delta": inherited_seed_delta,
        "a01_tokenizer_surface_eval_rows": surface_rows(),
        "files": {
            "old_seed43022": rel(OLD["43022"]),
            "old_seed43122": rel(OLD["43122"]),
            "corrected_seed43022": rel(CORRECTED["43022"]),
            "corrected_seed43122": rel(CORRECTED["43122"]),
            "a01_tokenizer_surface": rel(TOKEN_SURFACE),
            "note": rel(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload)
    print(json.dumps({
        "status": status,
        "out_json": rel(OUT_JSON),
        "note": rel(OUT_MD),
        "complete_corrected_seeds": complete,
        "seed43022_corrected_overall": corr["43022"]["scores"].get("Overall"),
        "seed43122_corrected_overall": corr["43122"]["scores"].get("Overall"),
        "top_column_moves_seed43022": per_seed["43022"]["column_largest_abs_movement"][:5],
        "top_column_moves_seed43122": per_seed["43122"]["column_largest_abs_movement"][:5],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
