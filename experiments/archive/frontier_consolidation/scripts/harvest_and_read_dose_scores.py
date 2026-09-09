#!/usr/bin/env python3
"""research: harvest completed dose/RoBERTa score files and read the dose profile.

This script deliberately avoids waiting for managed jobs.  It assembles only
already-written official-compatible per-target JSONs and already-written CSVs:
  * 1x DeBERTa view/repeat from research seed43022 rows,
  * 1.82x DeBERTa view/repeat from research rows,
  * 2.64x MAX DeBERTa view/repeat from research per-target JSONs,
  * clean0 DeBERTa from research/external clean per-target JSONs,
  * optional RoBERTa view/clean from research per-target JSONs.

It writes partial-coverage CSVs plus pre-result-order readouts.  Missing rows
remain missing and are reported in the coverage table; no score is filled in.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import glob
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/score_harvest_readout"

STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
COMPOSITES = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
ALL_METRICS = STABLE + COMPOSITES
CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]

DOSE_META = {
    "clean0": {"dose": 0.0, "rho": 0.0, "data_arm": "clean", "dose_name": "clean0", "expected": CKS_10_80},
    "dose1_view": {"dose": 1.0, "rho": 0.042352, "data_arm": "view", "dose_name": "dose1", "expected": CKS_10_100},
    "dose1_repeat": {"dose": 1.0, "rho": 0.042352, "data_arm": "repeat", "dose_name": "dose1", "expected": CKS_10_100},
    "dose1p82_view": {"dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "view", "dose_name": "dose1p82", "expected": CKS_10_100},
    "dose1p82_repeat": {"dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "repeat", "dose_name": "dose1p82", "expected": CKS_10_100},
    "max_view": {"dose": 2.641480921798262, "rho": 0.111872, "data_arm": "view", "dose_name": "dose2p64", "expected": CKS_10_100},
    "max_repeat": {"dose": 2.641480921798262, "rho": 0.111872, "data_arm": "repeat", "dose_name": "dose2p64", "expected": CKS_10_100},
}

ROBERTA_META = {
    "roberta_view": {"data_arm": "view", "expected": CKS_10_100},
    "roberta_clean": {"data_arm": "clean", "expected": CKS_10_100},
}

ROWS = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_stable_rows.csv"
ROWS = WS / "data/intermediate_dose_ladder_stable_eval/intermediate_dose_ladder_rows.csv"
SEED_SPREAD = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"

PT = WS / "data/dose_ladder_stable_eval/eval/per_target"
ROBERTA_PT = WS / "data/roberta_viewclean_total_stable_eval/eval/per_target"

CLEAN_EXTERNAL = {
    "chck_20M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json",
    "chck_70M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json",
    "chck_80M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def words_from_ck(ck: str) -> int:
    m = re.match(r"chck_(\d+)M$", ck)
    if not m:
        raise ValueError(f"bad checkpoint {ck}")
    return int(m.group(1)) * 1_000_000


def finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def f(v: Any) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def read_csv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ["words", "dose", "rho"] + ALL_METRICS:
            if k in r:
                val = f(r[k])
                if val is not None:
                    r[k] = val
    return rows


def extract_scores_from_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tasks = payload.get("tasks") or {}
    out: dict[str, Any] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col) or {}
        val = f(rec.get("score"))
        if val is None:
            return None
        if rec.get("returncode") not in (None, 0):
            return None
        out[col] = val
    rec = tasks.get("Reading") or {}
    scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
    val = f(scores.get("Reading") if scores else rec.get("score"))
    if val is None:
        return None
    if rec.get("returncode") not in (None, 0):
        return None
    out["Reading"] = val
    return out


def add_composites(row: dict[str, Any]) -> dict[str, Any]:
    vals = {c: f(row.get(c)) for c in STABLE}
    if all(vals[c] is not None for c in STABLE):
        row["cheap6_no_GlobalPIQA"] = sum(float(vals[c]) for c in STABLE) / 6.0
    if all(vals[c] is not None for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]):
        row["cheap5_no_GlobalPIQA_Reading"] = sum(float(vals[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]) / 5.0
    if vals.get("EWoK") is not None and vals.get("Entity") is not None:
        row["EWoK_plus_Entity_sum"] = float(vals["EWoK"]) + float(vals["Entity"])
    return row


def row_from_payload(arm: str, ck: str, path: pathlib.Path, source_type: str) -> dict[str, Any] | None:
    scores = extract_scores_from_payload(path)
    if scores is None:
        return None
    meta = DOSE_META[arm]
    row = {
        "arm": arm,
        "dose": meta["dose"],
        "rho": meta["rho"],
        "data_arm": meta["data_arm"],
        "dose_name": meta["dose_name"],
        "checkpoint": ck,
        "words": words_from_ck(ck),
        "source_type": source_type,
        "per_target": rel(path),
    }
    row.update(scores)
    return add_composites(row)


def roberta_row_from_payload(arm: str, ck: str, path: pathlib.Path, source_type: str) -> dict[str, Any] | None:
    scores = extract_scores_from_payload(path)
    if scores is None:
        return None
    meta = ROBERTA_META[arm]
    row = {
        "arm": arm,
        "data_arm": meta["data_arm"],
        "checkpoint": ck,
        "words": words_from_ck(ck),
        "source_type": source_type,
        "per_target": rel(path),
    }
    row.update(scores)
    return add_composites(row)


def collect_deberta_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # 1x seed43022 authoritative rows from research.
    for r in read_csv_rows(ROWS):
        if r.get("seed") not in ("43022", 43022, 43022.0):
            continue
        arm = None
        if r.get("arm") == "seed43022_compact":
            arm = "dose1_view"
        elif r.get("arm") == "seed43022_repeat":
            arm = "dose1_repeat"
        if arm is None:
            continue
        meta = DOSE_META[arm]
        out = {k: r.get(k) for k in ["checkpoint", "words"] + ALL_METRICS if k in r}
        out.update({"arm": arm, "dose": meta["dose"], "rho": meta["rho"], "data_arm": meta["data_arm"], "dose_name": meta["dose_name"], "source_type": "seed43022_csv", "per_target": r.get("per_target", "")})
        rows.append(out)

    # 1.82x authoritative rows from research.
    for r in read_csv_rows(ROWS):
        arm_in = str(r.get("arm"))
        if arm_in not in {"dose1p82_view", "dose1p82_repeat"}:
            continue
        meta = DOSE_META[arm_in]
        out = {k: r.get(k) for k in ["checkpoint", "words"] + ALL_METRICS if k in r}
        out.update({"arm": arm_in, "dose": meta["dose"], "rho": meta["rho"], "data_arm": meta["data_arm"], "dose_name": meta["dose_name"], "source_type": r.get("source_type", "csv"), "per_target": r.get("per_target", "")})
        rows.append(out)

    # clean0 and MAX from research/external per-target JSONs.
    for ck in CKS_10_80:
        p = PT / f"dose_clean0_{ck}.json"
        src = "clean_json"
        if not p.exists() and ck in CLEAN_EXTERNAL:
            p = CLEAN_EXTERNAL[ck]
            src = "external_clean_json"
        row = row_from_payload("clean0", ck, p, src)
        if row is not None:
            rows.append(row)

    for arm in ["max_view", "max_repeat"]:
        for ck in CKS_10_100:
            p = PT / f"dose_{arm}_{ck}.json"
            row = row_from_payload(arm, ck, p, "max_json")
            if row is not None:
                rows.append(row)

    # Deduplicate by arm/checkpoint with preference order: CSVs > JSONs.  Later
    # duplicates should not occur except accidental reruns.
    priority = {"seed43022_csv": 10, "new_eval": 9, "csv": 9, "external_clean_json": 8, "clean_json": 8, "max_json": 8}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("arm")), str(r.get("checkpoint")))
        old = best.get(key)
        if old is None or priority.get(str(r.get("source_type")), 0) >= priority.get(str(old.get("source_type")), 0):
            best[key] = r
    return sorted(best.values(), key=lambda r: (str(r.get("arm")), int(r.get("words") or 0)))


def collect_roberta_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, prefix in [("roberta_view", "view"), ("roberta_clean", "clean")]:
        for ck in CKS_10_100:
            p = ROBERTA_PT / f"roberta_viewclean_total_{prefix}_{ck}.json"
            row = roberta_row_from_payload(arm, ck, p, "roberta_json")
            if row is not None:
                rows.append(row)
    return sorted(rows, key=lambda r: (str(r.get("arm")), int(r.get("words") or 0)))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for preferred in ["arm", "dose", "rho", "dose_name", "data_arm", "checkpoint", "words"] + ALL_METRICS + ["source_type", "per_target"]:
        fields.append(preferred)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def lookup(rows: list[dict[str, Any]], arm: str, ck: str) -> dict[str, Any] | None:
    for r in rows:
        if r.get("arm") == arm and r.get("checkpoint") == ck:
            return r
    return None


def delta(a: dict[str, Any], b: dict[str, Any], label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"label": label, "checkpoint": a.get("checkpoint"), "words": a.get("words")}
    if extra:
        out.update(extra)
    for m in ALL_METRICS:
        va, vb = f(a.get(m)), f(b.get(m))
        out[m] = va - vb if va is not None and vb is not None else None
    return out


def make_profiles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sem: list[dict[str, Any]] = []
    growth: list[dict[str, Any]] = []
    total: list[dict[str, Any]] = []
    repeat_clean: list[dict[str, Any]] = []
    for name, view, rep, dose in [("dose1", "dose1_view", "dose1_repeat", 1.0), ("dose1p82", "dose1p82_view", "dose1p82_repeat", 1.8209293539856442), ("dose2p64", "max_view", "max_repeat", 2.641480921798262)]:
        for ck in CKS_10_100:
            v = lookup(rows, view, ck)
            r = lookup(rows, rep, ck)
            if v and r:
                sem.append(delta(v, r, f"{name}_view_minus_repeat", {"dose": dose, "dose_name": name}))
    for name, view, dose in [("dose1p82", "dose1p82_view", 1.8209293539856442), ("dose2p64", "max_view", 2.641480921798262)]:
        for ck in CKS_10_100:
            vd = lookup(rows, view, ck)
            v1 = lookup(rows, "dose1_view", ck)
            if vd and v1:
                growth.append(delta(vd, v1, f"{name}_view_minus_dose1_view", {"dose": dose, "dose_name": name}))
    for name, view, rep, dose in [("dose1", "dose1_view", "dose1_repeat", 1.0), ("dose1p82", "dose1p82_view", "dose1p82_repeat", 1.8209293539856442), ("dose2p64", "max_view", "max_repeat", 2.641480921798262)]:
        for ck in CKS_10_80:
            v = lookup(rows, view, ck)
            r = lookup(rows, rep, ck)
            c = lookup(rows, "clean0", ck)
            if v and c:
                total.append(delta(v, c, f"{name}_view_minus_clean", {"dose": dose, "dose_name": name}))
            if r and c:
                repeat_clean.append(delta(r, c, f"{name}_repeat_minus_clean", {"dose": dose, "dose_name": name}))
    return {"semantic_VR": sem, "cleanfree_growth_Vd_minus_V1": growth, "total_VC": total, "repeat_clean_RC": repeat_clean}


def summarize_profile(profile: list[dict[str, Any]], by: str = "dose_name") -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in profile:
        groups.setdefault(str(r.get(by)), []).append(r)
    out: dict[str, Any] = {}
    for g, rs in sorted(groups.items()):
        d: dict[str, Any] = {"n": len(rs), "checkpoints": [r.get("checkpoint") for r in sorted(rs, key=lambda x: int(x.get("words") or 0))]}
        for m in ALL_METRICS:
            vals = [f(r.get(m)) for r in rs]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                d[m] = {
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "stdev": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
                    "at_80M": next((f(r.get(m)) for r in rs if r.get("checkpoint") == "chck_80M"), None),
                    "at_100M": next((f(r.get(m)) for r in rs if r.get("checkpoint") == "chck_100M"), None),
                }
        out[g] = d
    return out


def coverage(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, spec in meta.items():
        got = sorted([str(r.get("checkpoint")) for r in rows if r.get("arm") == arm], key=words_from_ck)
        exp = spec["expected"]
        out[arm] = {"expected": len(exp), "got": len(got), "checkpoints": got, "missing": [ck for ck in exp if ck not in got]}
    return out


def seed_spread_summary() -> dict[str, Any]:
    rows = read_csv_rows(SEED_SPREAD)
    out: dict[str, Any] = {}
    for m in ALL_METRICS:
        vals: list[float] = []
        for r in rows:
            key = f"treatment_delta_seed43122_minus_seed43022_{m}"
            if key in r and f(r.get(key)) is not None:
                vals.append(abs(float(r[key])))
        if vals:
            out[m] = {"n": len(vals), "mean_abs": statistics.mean(vals), "median_abs": statistics.median(vals), "max_abs": max(vals)}
    return out


def roberta_contrast(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ck in CKS_10_100:
        v = lookup(rows, "roberta_view", ck)
        c = lookup(rows, "roberta_clean", ck)
        if v and c:
            out.append(delta(v, c, "roberta_max_view_minus_clean", {"contrast": "roberta_total_transfer"}))
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# research score harvest and dose readout")
    lines.append("")
    lines.append("This is a file-only readout from already completed official-compatible stable-family evaluations. Missing rows remain missing; no score is imputed.")
    lines.append("")
    lines.append("## Coverage")
    for arm, c in summary["coverage"].items():
        miss = ", ".join(c["missing"]) if c["missing"] else "none"
        lines.append(f"- {arm}: {c['got']}/{c['expected']} rows; missing: {miss}")
    lines.append("")
    lines.append("## Seed-spread scale from research")
    for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
        s = summary["seed_spread_context"].get(m, {})
        if s:
            lines.append(f"- {m}: mean |seed spread| {s['mean_abs']:.4f}, median {s['median_abs']:.4f}, max {s['max_abs']:.4f}")
    lines.append("")
    lines.append("## Semantic leg V-R (view minus repeat)")
    for dose_name, d in summary["profile_summaries"]["semantic_VR"].items():
        lines.append(f"### {dose_name} ({d['n']} checkpoints: {', '.join(d['checkpoints'])})")
        for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
            x = d.get(m)
            if x:
                e80 = x.get("at_80M")
                e100 = x.get("at_100M")
                lines.append(f"- {m}: mean {x['mean']:.4f}, median {x['median']:.4f}, min {x['min']:.4f}, max {x['max']:.4f}, @80M {e80}, @100M {e100}")
        lines.append("")
    lines.append("## Clean-free view growth V_d - V_1")
    for dose_name, d in summary["profile_summaries"]["cleanfree_growth_Vd_minus_V1"].items():
        lines.append(f"### {dose_name} ({d['n']} checkpoints: {', '.join(d['checkpoints'])})")
        for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
            x = d.get(m)
            if x:
                lines.append(f"- {m}: mean {x['mean']:.4f}, median {x['median']:.4f}, @80M {x.get('at_80M')}, @100M {x.get('at_100M')}")
        lines.append("")
    lines.append("## Total leg V-C (view minus clean, 10M-80M when clean exists)")
    for dose_name, d in summary["profile_summaries"]["total_VC"].items():
        lines.append(f"### {dose_name} ({d['n']} checkpoints: {', '.join(d['checkpoints'])})")
        for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
            x = d.get(m)
            if x:
                lines.append(f"- {m}: mean {x['mean']:.4f}, median {x['median']:.4f}, @80M {x.get('at_80M')}")
        lines.append("")
    lines.append("## RoBERTa MAX view-clean available contrast")
    rob = summary.get("roberta_contrast_summary", {})
    if not rob:
        lines.append("No matched RoBERTa view-clean rows available yet.")
    else:
        for key, d in rob.items():
            lines.append(f"### {key} ({d['n']} checkpoints: {', '.join(d['checkpoints'])})")
            for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
                x = d.get(m)
                if x:
                    lines.append(f"- {m}: mean {x['mean']:.4f}, median {x['median']:.4f}, @80M {x.get('at_80M')}, @100M {x.get('at_100M')}")
    lines.append("")
    lines.append("## Scientific reading so far")
    lines.append(summary.get("scientific_reading", ""))
    lines.append("")
    lines.append("## Files")
    for k, v in summary.get("files", {}).items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    deberta_rows = collect_deberta_rows()
    roberta_rows = collect_roberta_rows()
    profiles = make_profiles(deberta_rows)
    rob_con = roberta_contrast(roberta_rows)

    write_csv(out_dir / "harvested_deberta_stable_rows.csv", deberta_rows)
    write_csv(out_dir / "harvested_roberta_stable_rows.csv", roberta_rows)
    for name, rows in profiles.items():
        write_csv(out_dir / f"{name}.csv", rows)
    write_csv(out_dir / "roberta_max_view_clean_contrast.csv", rob_con)

    profile_summaries = {name: summarize_profile(rows) for name, rows in profiles.items()}
    rob_summary = summarize_profile(rob_con, by="contrast") if rob_con else {}
    cov = coverage(deberta_rows, DOSE_META)
    rob_cov = coverage(roberta_rows, ROBERTA_META)
    spread = seed_spread_summary()

    # Direct compact scientific reading: keep it conservative but substantive,
    # based only on rows present now.
    sem = profile_summaries.get("semantic_VR", {})
    growth = profile_summaries.get("cleanfree_growth_Vd_minus_V1", {})
    total = profile_summaries.get("total_VC", {})
    reading_parts: list[str] = []
    for dose in ["dose1", "dose1p82", "dose2p64"]:
        d = sem.get(dose, {})
        c6 = (d.get("cheap6_no_GlobalPIQA") or {}).get("mean")
        c5 = (d.get("cheap5_no_GlobalPIQA_Reading") or {}).get("mean")
        n = d.get("n", 0)
        if c6 is not None and c5 is not None:
            reading_parts.append(f"V-R {dose}: cheap6 mean {c6:.4f}, cheap5 mean {c5:.4f} over {n} rows.")
    for dose in ["dose1p82", "dose2p64"]:
        d = growth.get(dose, {})
        c6 = (d.get("cheap6_no_GlobalPIQA") or {}).get("mean")
        c5 = (d.get("cheap5_no_GlobalPIQA_Reading") or {}).get("mean")
        n = d.get("n", 0)
        if c6 is not None and c5 is not None:
            reading_parts.append(f"V_d-V_1 {dose}: cheap6 mean {c6:.4f}, cheap5 mean {c5:.4f} over {n} rows.")
    for dose in ["dose1", "dose1p82", "dose2p64"]:
        d = total.get(dose, {})
        c6 = (d.get("cheap6_no_GlobalPIQA") or {}).get("mean")
        c5 = (d.get("cheap5_no_GlobalPIQA_Reading") or {}).get("mean")
        n = d.get("n", 0)
        if c6 is not None and c5 is not None:
            reading_parts.append(f"V-C {dose}: cheap6 mean {c6:.4f}, cheap5 mean {c5:.4f} over {n} rows.")
    if not reading_parts:
        reading_parts.append("No matched dose contrasts are complete enough to read yet.")
    scientific_reading = " ".join(reading_parts) + " EWoK+Entity remains secondary because research measured much larger seed spread there than on cheap6/cheap5. Any incomplete MAX or RoBERTa rows must be reread when their evaluators finish; the second-basin MAX pair launched in research is required before a mechanism-level synthesis."

    summary = {
        "status": "SCORE_HARVEST_READOUT_COMPLETE",
        "created_utc": now(),
        "deberta_row_count": len(deberta_rows),
        "roberta_row_count": len(roberta_rows),
        "coverage": cov,
        "roberta_coverage": rob_cov,
        "profile_summaries": profile_summaries,
        "roberta_contrast_summary": rob_summary,
        "seed_spread_context": spread,
        "scientific_reading": scientific_reading,
        "boundary": "Stable selected families only. No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action. Missing score rows are not imputed.",
        "files": {
            "deberta_rows": rel(out_dir / "harvested_deberta_stable_rows.csv"),
            "roberta_rows": rel(out_dir / "harvested_roberta_stable_rows.csv"),
            "semantic_VR": rel(out_dir / "semantic_VR.csv"),
            "cleanfree_growth": rel(out_dir / "cleanfree_growth_Vd_minus_V1.csv"),
            "total_VC": rel(out_dir / "total_VC.csv"),
            "repeat_clean_RC": rel(out_dir / "repeat_clean_RC.csv"),
            "roberta_contrast": rel(out_dir / "roberta_max_view_clean_contrast.csv"),
            "summary_json": rel(out_dir / "score_harvest_readout_summary.json"),
            "summary_md": rel(out_dir / "score_harvest_readout_summary.md"),
        },
    }
    (out_dir / "score_harvest_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "score_harvest_readout_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
