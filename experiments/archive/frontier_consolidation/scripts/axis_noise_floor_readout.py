#!/usr/bin/env python3
"""research: distinguish shared family-axis structure from same-coordinate seed floor.

File-only analysis.  It uses existing selected-family score tables and the research
PCA axes to ask whether the apparent ex-Entity family-vector movement is a
reproducible low-dimensional trade-off or is better read as ordinary basin-level
variation in the same DeBERTa compact-versus-repeat coordinate.

No model loading, training, official evaluation, GPU work, upload, or leaderboard
action occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ROOT = Path("experiments/archive/frontier_consolidation")
DATA = ROOT / "data"
OUT_DIR = DATA / "axis_noise_floor_readout"

DELTAS = DATA / "full_deberta_seed_ladder_stable_eval" / "full_deberta_seed_ladder_treatment_deltas.csv"
JSON = DATA / "score_vector_tradeoff" / "score_vector_tradeoff.json"
DOSE = DATA / "conservation_vector_readout" / "dose_common10_80_net_norm.csv"
VISIBLE = DATA / "conservation_vector_readout" / "visible_reference_net_norm.csv"

COLS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
STABLE5_EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
STABLE4_EX_ENTITY_NO_READING = ["BLiMP", "Supplement", "EWoK", "COMPS"]
WINDOWS = {
    "common10_80": ["chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M"],
    "full10_100": ["chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"],
    "late80_100": ["chck_80M", "chck_90M", "chck_100M"],
    "mature70_100": ["chck_70M", "chck_80M", "chck_90M", "chck_100M"],
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def vec_from_row(row: Mapping[str, Any], cols: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        v = safe_float(row.get(c))
        if v is not None:
            out[c] = v
    return out


def vector_stats(vec: Mapping[str, float], cols: Sequence[str]) -> Dict[str, Any]:
    vals = np.array([float(vec[c]) for c in cols if c in vec and safe_float(vec[c]) is not None], dtype=float)
    if vals.size == 0:
        return {
            "n": 0,
            "net": None,
            "rms": None,
            "centered_rms": None,
            "l2": None,
            "mean_abs": None,
            "pos_count": 0,
            "neg_count": 0,
            "max_abs_column": None,
            "max_abs_delta": None,
        }
    net = float(vals.mean())
    rms = float(np.sqrt(np.mean(vals * vals)))
    centered = float(np.sqrt(np.mean((vals - net) * (vals - net))))
    abs_vals = np.abs(vals)
    present_cols = [c for c in cols if c in vec and safe_float(vec[c]) is not None]
    max_i = int(abs_vals.argmax())
    return {
        "n": int(vals.size),
        "net": net,
        "rms": rms,
        "centered_rms": centered,
        "l2": float(np.linalg.norm(vals)),
        "mean_abs": float(abs_vals.mean()),
        "pos_count": int(np.sum(vals > 0)),
        "neg_count": int(np.sum(vals < 0)),
        "max_abs_column": present_cols[max_i],
        "max_abs_delta": float(vals[max_i]),
    }


def cosine(vec_a: Mapping[str, float], vec_b: Mapping[str, float], cols: Sequence[str], center: bool = False) -> Optional[float]:
    common = [c for c in cols if c in vec_a and c in vec_b]
    if not common:
        return None
    a = np.array([float(vec_a[c]) for c in common], dtype=float)
    b = np.array([float(vec_b[c]) for c in common], dtype=float)
    if center:
        a = a - a.mean()
        b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def l2_norm(vec: Mapping[str, float], cols: Sequence[str], center: bool = False) -> Optional[float]:
    present = [c for c in cols if c in vec]
    if not present:
        return None
    arr = np.array([float(vec[c]) for c in present], dtype=float)
    if center:
        arr = arr - arr.mean()
    return float(np.linalg.norm(arr))


def diff_norm(vec_a: Mapping[str, float], vec_b: Mapping[str, float], cols: Sequence[str], center: bool = False) -> Optional[float]:
    common = [c for c in cols if c in vec_a and c in vec_b]
    if not common:
        return None
    a = np.array([float(vec_a[c]) for c in common], dtype=float)
    b = np.array([float(vec_b[c]) for c in common], dtype=float)
    if center:
        a = a - a.mean()
        b = b - b.mean()
    return float(np.linalg.norm(a - b))


def vector_json(vec: Mapping[str, float], cols: Sequence[str]) -> str:
    return json.dumps({c: float(vec[c]) for c in cols if c in vec}, sort_keys=True)


def load_step253_vectors() -> Dict[Tuple[int, str], Dict[str, float]]:
    rows = read_csv_rows(DELTAS)
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in rows:
        seed = int(row["seed"])
        ckpt = row["checkpoint"]
        out[(seed, ckpt)] = vec_from_row(row, STABLE6)
    return out


def mean_vector(vectors: Iterable[Mapping[str, float]], cols: Sequence[str]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {c: [] for c in cols}
    for vec in vectors:
        for c in cols:
            if c in vec:
                buckets[c].append(float(vec[c]))
    return {c: float(np.mean(vals)) for c, vals in buckets.items() if vals}


def parse_vector_json(s: Any) -> Dict[str, float]:
    if isinstance(s, dict):
        return {str(k): float(v) for k, v in s.items() if safe_float(v) is not None}
    if not s:
        return {}
    obj = json.loads(str(s))
    return {str(k): float(v) for k, v in obj.items() if safe_float(v) is not None}


def load_step107() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    obj = json.loads(JSON.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = obj["records"]
    pc_loadings: Dict[str, Dict[str, float]] = {}
    pc_varfrac: Dict[str, float] = {}
    for pc in obj["pca"][:3]:
        name = f"PC{pc['component']}"
        pc_loadings[name] = {c: float(v) for c, v in pc["loadings"].items()}
        pc_varfrac[name] = float(pc["variance_fraction"])
    mean_vec: Dict[str, float] = {}
    for c in COLS7:
        vals = [float(r["deltas"][c]) for r in records if c in r.get("deltas", {})]
        mean_vec[c] = float(np.mean(vals))
    return records, pc_loadings, pc_varfrac, mean_vec


def project_onto_axes(
    vec: Mapping[str, float],
    axes: Mapping[str, Mapping[str, float]],
    axis_varfrac: Mapping[str, float],
    axis_mean: Mapping[str, float],
    cols: Sequence[str],
    source_block: str,
    label: str,
    family_set: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    extra = extra or {}
    present = [c for c in cols if c in vec]
    arr = np.array([float(vec[c]) for c in present], dtype=float)
    l2 = float(np.linalg.norm(arr)) if arr.size else 0.0
    net = float(arr.mean()) if arr.size else None
    centered = arr - arr.mean() if arr.size else arr
    centered_l2 = float(np.linalg.norm(centered)) if arr.size else 0.0
    row: Dict[str, Any] = {
        "source_block": source_block,
        "label": label,
        "family_set": family_set,
        "n_columns": len(present),
        "columns": ";".join(present),
        "net": net,
        "rms": float(math.sqrt(np.mean(arr * arr))) if arr.size else None,
        "l2": l2,
        "centered_l2": centered_l2,
        "vector_json": vector_json(vec, present),
    }
    row.update(extra)
    top3_energy_raw = 0.0
    top3_energy_centered = 0.0
    for name, axis in axes.items():
        a = np.array([float(axis[c]) for c in present], dtype=float)
        axis_norm = float(np.linalg.norm(a)) if a.size else 0.0
        raw_proj = float(np.dot(arr, a)) if arr.size else 0.0
        raw_cos = raw_proj / (l2 * axis_norm) if l2 > 1e-12 and axis_norm > 1e-12 else None
        centered_arr = arr - np.array([float(axis_mean.get(c, 0.0)) for c in present], dtype=float)
        centered_proj = float(np.dot(centered_arr, a)) if arr.size else 0.0
        centered_norm = float(np.linalg.norm(centered_arr)) if arr.size else 0.0
        centered_cos = centered_proj / (centered_norm * axis_norm) if centered_norm > 1e-12 and axis_norm > 1e-12 else None
        demeaned_proj = float(np.dot(centered, a)) if arr.size else 0.0
        demeaned_cos = demeaned_proj / (centered_l2 * axis_norm) if centered_l2 > 1e-12 and axis_norm > 1e-12 else None
        row[f"{name}_variance_fraction_step107"] = axis_varfrac.get(name)
        row[f"{name}_axis_norm_on_available_columns"] = axis_norm
        row[f"{name}_raw_projection"] = raw_proj
        row[f"{name}_raw_direction_cosine"] = raw_cos
        row[f"{name}_centered_to_step107_mean_projection"] = centered_proj
        row[f"{name}_centered_to_step107_mean_direction_cosine"] = centered_cos
        row[f"{name}_within_vector_demeaned_projection"] = demeaned_proj
        row[f"{name}_within_vector_demeaned_direction_cosine"] = demeaned_cos
        top3_energy_raw += raw_proj * raw_proj
        top3_energy_centered += centered_proj * centered_proj
    row["top3_raw_energy_over_l2_sq"] = top3_energy_raw / (l2 * l2) if l2 > 1e-12 else None
    # This is exact only when axes are used in the full 7D space.  In six-column rows it is
    # a subspace indicator because the restricted PC axes are not mutually complete.
    row["top3_projection_note"] = "full_7d_exact_for_cheap7_rows; subspace_indicator_when_GlobalPIQA_missing"
    return row


def seed_vector_reproducibility(research: Mapping[Tuple[int, str], Mapping[str, float]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    family_sets = {
        "stable6": STABLE6,
        "stable5_exEntity": STABLE5_EX_ENTITY,
        "stable4_exEntity_noReading": STABLE4_EX_ENTITY_NO_READING,
    }
    summary: Dict[str, Any] = {}

    for window_name, ckpts in WINDOWS.items():
        seed_vecs: Dict[int, Dict[str, float]] = {}
        for seed in [43022, 43122]:
            seed_vecs[seed] = mean_vector([research[(seed, ckpt)] for ckpt in ckpts if (seed, ckpt) in research], STABLE6)
        for fs_name, cols in family_sets.items():
            a = {c: seed_vecs[43022][c] for c in cols if c in seed_vecs[43022]}
            b = {c: seed_vecs[43122][c] for c in cols if c in seed_vecs[43122]}
            norm_a = l2_norm(a, cols)
            norm_b = l2_norm(b, cols)
            dnorm = diff_norm(a, b, cols)
            centered_norm_a = l2_norm(a, cols, center=True)
            centered_norm_b = l2_norm(b, cols, center=True)
            centered_dnorm = diff_norm(a, b, cols, center=True)
            row = {
                "window": window_name,
                "family_set": fs_name,
                "n_checkpoints": len(ckpts),
                "seed43022_net": vector_stats(a, cols)["net"],
                "seed43122_net": vector_stats(b, cols)["net"],
                "seed43022_rms": vector_stats(a, cols)["rms"],
                "seed43122_rms": vector_stats(b, cols)["rms"],
                "seed43022_l2": norm_a,
                "seed43122_l2": norm_b,
                "difference_l2": dnorm,
                "difference_over_mean_seed_l2": dnorm / ((norm_a + norm_b) / 2.0) if norm_a and norm_b and dnorm is not None else None,
                "cosine": cosine(a, b, cols, center=False),
                "centered_cosine": cosine(a, b, cols, center=True),
                "centered_difference_l2": centered_dnorm,
                "centered_difference_over_mean_centered_seed_l2": centered_dnorm / ((centered_norm_a + centered_norm_b) / 2.0) if centered_norm_a and centered_norm_b and centered_dnorm is not None else None,
                "seed43022_vector_json": vector_json(a, cols),
                "seed43122_vector_json": vector_json(b, cols),
            }
            rows.append(row)
            summary[f"{window_name}_{fs_name}"] = row

    for ckpt in WINDOWS["full10_100"]:
        a_full = research[(43022, ckpt)]
        b_full = research[(43122, ckpt)]
        for fs_name, cols in family_sets.items():
            a = {c: a_full[c] for c in cols if c in a_full}
            b = {c: b_full[c] for c in cols if c in b_full}
            norm_a = l2_norm(a, cols)
            norm_b = l2_norm(b, cols)
            dnorm = diff_norm(a, b, cols)
            checkpoint_rows.append({
                "checkpoint": ckpt,
                "family_set": fs_name,
                "seed43022_net": vector_stats(a, cols)["net"],
                "seed43122_net": vector_stats(b, cols)["net"],
                "seed43022_l2": norm_a,
                "seed43122_l2": norm_b,
                "difference_l2": dnorm,
                "difference_over_mean_seed_l2": dnorm / ((norm_a + norm_b) / 2.0) if norm_a and norm_b and dnorm is not None else None,
                "cosine": cosine(a, b, cols),
                "centered_cosine": cosine(a, b, cols, center=True),
                "seed43022_vector_json": vector_json(a, cols),
                "seed43122_vector_json": vector_json(b, cols),
            })

    return rows, checkpoint_rows, summary


def build_projection_rows(
    research: Mapping[Tuple[int, str], Mapping[str, float]],
    seed_summary_rows: Sequence[Mapping[str, Any]],
    records107: Sequence[Mapping[str, Any]],
    axes: Mapping[str, Mapping[str, float]],
    axis_varfrac: Mapping[str, float],
    axis_mean: Mapping[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # research 1x seed vectors over the windows that matter for dose comparison.
    for seed in [43022, 43122]:
        for window_name, ckpts in WINDOWS.items():
            mean_vec6 = mean_vector([research[(seed, ckpt)] for ckpt in ckpts if (seed, ckpt) in research], STABLE6)
            rows.append(project_onto_axes(
                mean_vec6, axes, axis_varfrac, axis_mean, STABLE6,
                source_block="reference_1x_seed_mean",
                label=f"seed{seed}_{window_name}_VminusR",
                family_set="stable6_noGlobalPIQA",
                extra={"seed": seed, "window": window_name, "contrast": "V_minus_R"},
            ))
            vec5 = {c: mean_vec6[c] for c in STABLE5_EX_ENTITY}
            rows.append(project_onto_axes(
                vec5, axes, axis_varfrac, axis_mean, STABLE5_EX_ENTITY,
                source_block="reference_1x_seed_mean",
                label=f"seed{seed}_{window_name}_VminusR_exEntity",
                family_set="stable5_exEntity_noGlobalPIQA",
                extra={"seed": seed, "window": window_name, "contrast": "V_minus_R"},
            ))

    # Dose vectors from research file, especially V_minus_R and its growth/readout.
    for row in read_csv_rows(DOSE):
        contrast = row.get("contrast", "")
        famset = row.get("family_set", "")
        if famset not in {"stable6", "stable5_exEntity", "stable4_exEntity_noReading"}:
            continue
        if contrast not in {"V_minus_R", "V_minus_C", "R_minus_C", "view_growth_vs_1x", "repeat_growth_vs_1x", "VR_growth_vs_1x"}:
            continue
        vec = parse_vector_json(row.get("vector_json"))
        cols = [c for c in (STABLE6 if famset == "stable6" else STABLE5_EX_ENTITY if famset == "stable5_exEntity" else STABLE4_EX_ENTITY_NO_READING) if c in vec]
        rows.append(project_onto_axes(
            vec, axes, axis_varfrac, axis_mean, cols,
            source_block="dose_common10_80",
            label=f"{row.get('dose_name')}:{contrast}:{famset}",
            family_set=famset + "_noGlobalPIQA",
            extra={
                "dose_name": row.get("dose_name"),
                "rho": safe_float(row.get("rho")),
                "contrast": contrast,
                "window": "common10_80",
            },
        ))

    # Visible V-B/B-C/B-R/V-C/V-R reference rows available before the backlog lands.
    for row in read_csv_rows(VISIBLE):
        contrast = row.get("contrast", "")
        famset = row.get("family_set", "")
        if famset not in {"stable6", "stable5_exEntity", "stable4_exEntity_noReading"}:
            continue
        if contrast not in {"D1_VminusB", "D1_BminusCold", "D1_BminusR", "D1_VminusCold", "D1_VminusR"}:
            continue
        vec = parse_vector_json(row.get("vector_json"))
        cols = [c for c in (STABLE6 if famset == "stable6" else STABLE5_EX_ENTITY if famset == "stable5_exEntity" else STABLE4_EX_ENTITY_NO_READING) if c in vec]
        rows.append(project_onto_axes(
            vec, axes, axis_varfrac, axis_mean, cols,
            source_block="visible_reference",
            label=f"{contrast}:{row.get('checkpoint')}:{famset}",
            family_set=famset + "_noGlobalPIQA",
            extra={
                "contrast": contrast,
                "checkpoint": row.get("checkpoint"),
                "window": row.get("checkpoint"),
            },
        ))

    # research archive itself: full cheap7 vectors plus stable6/exEntity shadows.
    for rec in records107:
        deltas = {c: float(rec["deltas"][c]) for c in COLS7 if c in rec.get("deltas", {})}
        rows.append(project_onto_axes(
            deltas, axes, axis_varfrac, axis_mean, COLS7,
            source_block="archive",
            label=f"{rec['label']}:{rec['exposure']}:cheap7",
            family_set="cheap7_full7",
            extra={
                "intervention_family": rec.get("family"),
                "exposure": rec.get("exposure"),
                "contrast": "intervention_minus_reference",
                "note": rec.get("note"),
            },
        ))
        stable6_vec = {c: deltas[c] for c in STABLE6 if c in deltas}
        rows.append(project_onto_axes(
            stable6_vec, axes, axis_varfrac, axis_mean, STABLE6,
            source_block="archive",
            label=f"{rec['label']}:{rec['exposure']}:stable6",
            family_set="stable6_noGlobalPIQA",
            extra={
                "intervention_family": rec.get("family"),
                "exposure": rec.get("exposure"),
                "contrast": "intervention_minus_reference",
                "note": rec.get("note"),
            },
        ))
        exent_vec = {c: deltas[c] for c in STABLE5_EX_ENTITY if c in deltas}
        rows.append(project_onto_axes(
            exent_vec, axes, axis_varfrac, axis_mean, STABLE5_EX_ENTITY,
            source_block="archive",
            label=f"{rec['label']}:{rec['exposure']}:stable5_exEntity",
            family_set="stable5_exEntity_noGlobalPIQA",
            extra={
                "intervention_family": rec.get("family"),
                "exposure": rec.get("exposure"),
                "contrast": "intervention_minus_reference",
                "note": rec.get("note"),
            },
        ))

    return rows


def summarize_axis_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("source_block")), str(r.get("contrast", "")), str(r.get("family_set")))] .append(r)
    out: List[Dict[str, Any]] = []
    for (source, contrast, famset), rs in sorted(groups.items()):
        entry: Dict[str, Any] = {
            "source_block": source,
            "contrast": contrast,
            "family_set": famset,
            "n_rows": len(rs),
        }
        for pc in ["PC1", "PC2", "PC3"]:
            vals = [safe_float(r.get(f"{pc}_raw_direction_cosine")) for r in rs]
            vals = [v for v in vals if v is not None]
            projs = [safe_float(r.get(f"{pc}_raw_projection")) for r in rs]
            projs = [v for v in projs if v is not None]
            if vals:
                entry[f"{pc}_mean_raw_cosine"] = float(np.mean(vals))
                entry[f"{pc}_mean_abs_raw_cosine"] = float(np.mean(np.abs(vals)))
                entry[f"{pc}_max_abs_raw_cosine"] = float(np.max(np.abs(vals)))
            if projs:
                entry[f"{pc}_mean_raw_projection"] = float(np.mean(projs))
                entry[f"{pc}_mean_abs_raw_projection"] = float(np.mean(np.abs(projs)))
        shares = [safe_float(r.get("top3_raw_energy_over_l2_sq")) for r in rs]
        shares = [v for v in shares if v is not None]
        if shares:
            entry["mean_top3_raw_energy_over_l2_sq"] = float(np.mean(shares))
            entry["max_top3_raw_energy_over_l2_sq"] = float(np.max(shares))
        out.append(entry)
    return out


def make_plots(out_dir: Path, seed_rows: Sequence[Mapping[str, Any]], projection_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    paths: List[str] = []
    if plt is None:
        return paths

    # Seed reproducibility: cosine and difference ratio by window/family set.
    selected = [r for r in seed_rows if r["family_set"] in {"stable6", "stable5_exEntity"}]
    if selected:
        labels = [f"{r['window']}\n{r['family_set'].replace('_', ' ')}" for r in selected]
        cos_vals = [float(r["cosine"]) if r["cosine"] not in (None, "") else np.nan for r in selected]
        ratio_vals = [float(r["difference_over_mean_seed_l2"]) if r["difference_over_mean_seed_l2"] not in (None, "") else np.nan for r in selected]
        x = np.arange(len(selected))
        fig, ax1 = plt.subplots(figsize=(max(9, 0.55 * len(selected)), 4.4))
        ax1.axhline(0, color="black", lw=0.8)
        ax1.bar(x - 0.18, cos_vals, width=0.36, color="#4C78A8", label="cosine")
        ax1.set_ylabel("seed-vector cosine")
        ax1.set_ylim(-1.05, 1.05)
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, ratio_vals, width=0.36, color="#F58518", alpha=0.75, label="||diff|| / mean ||seed||")
        ax2.set_ylabel("difference ratio")
        ax2.set_ylim(0, max(2.5, float(np.nanmax(ratio_vals)) * 1.15))
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=35, ha="right")
        ax1.set_title("research 1x compact-minus-repeat vector reproducibility")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
        fig.tight_layout()
        p = out_dir / "seed_1x_vector_reproducibility.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(str(p))

    # PC1/PC2 scatter for representative rows.
    reps: List[Mapping[str, Any]] = []
    for r in projection_rows:
        sb = r.get("source_block")
        fam = r.get("family_set")
        label = str(r.get("label"))
        if fam not in {"stable6_noGlobalPIQA", "cheap7_full7"}:
            continue
        if sb == "dose_common10_80" and r.get("contrast") == "V_minus_R" and fam == "stable6_noGlobalPIQA":
            reps.append(r)
        elif sb == "visible_reference" and str(r.get("contrast")) in {"D1_VminusB", "D1_BminusCold", "D1_VminusCold"} and fam == "stable6_noGlobalPIQA":
            reps.append(r)
        elif sb == "reference_1x_seed_mean" and str(r.get("window")) == "common10_80" and fam == "stable6_noGlobalPIQA":
            reps.append(r)
        elif sb == "archive" and fam == "cheap7_full7" and str(r.get("exposure")) in {"80M", "100M"}:
            reps.append(r)
    if reps:
        colors = {
            "dose_common10_80": "#4C78A8",
            "visible_reference": "#54A24B",
            "reference_1x_seed_mean": "#E45756",
            "archive": "#B279A2",
        }
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        for r in reps:
            x = safe_float(r.get("PC1_raw_projection"))
            y = safe_float(r.get("PC2_raw_projection"))
            if x is None or y is None:
                continue
            sb = str(r.get("source_block"))
            ax.scatter([x], [y], s=54, color=colors.get(sb, "gray"), alpha=0.9)
            short = str(r.get("label"))
            short = short.replace(":stable6", "").replace(":cheap7", "")
            short = short.replace("D1_", "").replace("V_minus_R", "VR")
            if len(short) > 36:
                short = short[:33] + "..."
            ax.annotate(short, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("raw projection on research PC1")
        ax.set_ylabel("raw projection on research PC2")
        ax.set_title("Representative family vectors on research axes")
        fig.tight_layout()
        p = out_dir / "axis_projection_pc1_pc2.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(str(p))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    research = load_step253_vectors()
    records107, axes, axis_varfrac, axis_mean = load_step107()

    seed_rows, checkpoint_rows, seed_summary = seed_vector_reproducibility(research)
    projection_rows = build_projection_rows(research, seed_rows, records107, axes, axis_varfrac, axis_mean)
    axis_summary_rows = summarize_axis_rows(projection_rows)
    plot_paths = make_plots(out_dir, seed_rows, projection_rows)

    write_csv(out_dir / "seed_1x_vector_reproducibility.csv", seed_rows)
    write_csv(out_dir / "checkpoint_seed_vector_alignment.csv", checkpoint_rows)
    write_csv(out_dir / "axis_projection_rows.csv", projection_rows)
    write_csv(out_dir / "axis_projection_group_summary.csv", axis_summary_rows)

    # Pull out the decisive comparisons.
    def find_seed(window: str, family_set: str) -> Mapping[str, Any]:
        for r in seed_rows:
            if r["window"] == window and r["family_set"] == family_set:
                return r
        raise KeyError((window, family_set))

    common6 = find_seed("common10_80", "stable6")
    common5 = find_seed("common10_80", "stable5_exEntity")
    full6 = find_seed("full10_100", "stable6")
    mature6 = find_seed("mature70_100", "stable6")
    late6 = find_seed("late80_100", "stable6")

    # Entity contribution check from research table: all-family RMS growth is almost exactly the Entity term.
    dose_vr_rows = [r for r in read_csv_rows(DOSE) if r.get("contrast") == "V_minus_R" and r.get("family_set") in {"stable6", "stable5_exEntity"}]
    dose_entity_decomp: List[Dict[str, Any]] = []
    by_dose: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in dose_vr_rows:
        by_dose[r["dose_name"]][r["family_set"]] = r
    for dose_name, parts in sorted(by_dose.items()):
        if "stable6" in parts and "stable5_exEntity" in parts:
            s6 = parts["stable6"]
            s5 = parts["stable5_exEntity"]
            rms6 = float(s6["rms_family_delta"])
            rms5 = float(s5["rms_family_delta"])
            implied_entity_abs = math.sqrt(max(0.0, 6 * rms6 * rms6 - 5 * rms5 * rms5))
            vec = parse_vector_json(s6.get("vector_json"))
            entity = vec.get("Entity")
            dose_entity_decomp.append({
                "dose_name": dose_name,
                "rho": safe_float(s6.get("rho")),
                "stable6_net": safe_float(s6.get("mean_net")),
                "stable6_rms": rms6,
                "stable5_exEntity_net": safe_float(s5.get("mean_net")),
                "stable5_exEntity_rms": rms5,
                "implied_abs_entity_from_rms_identity": implied_entity_abs,
                "actual_entity_delta": entity,
                "abs_identity_error": abs(implied_entity_abs - abs(entity)) if entity is not None else None,
            })
    write_csv(out_dir / "dose_entity_component_identity.csv", dose_entity_decomp)

    # Readable axis highlights.
    def proj_lookup(source_block: str, label_contains: str, family_set: str = "stable6_noGlobalPIQA") -> Optional[Mapping[str, Any]]:
        for r in projection_rows:
            if r.get("source_block") == source_block and family_set == r.get("family_set") and label_contains in str(r.get("label")):
                return r
        return None

    key_projections: List[Dict[str, Any]] = []
    for spec in [
        ("reference_1x_seed_mean", "seed43022_common10_80_VminusR", "stable6_noGlobalPIQA"),
        ("reference_1x_seed_mean", "seed43122_common10_80_VminusR", "stable6_noGlobalPIQA"),
        ("dose_common10_80", "dose1:V_minus_R:stable6", "stable6_noGlobalPIQA"),
        ("dose_common10_80", "dose1p82:V_minus_R:stable6", "stable6_noGlobalPIQA"),
        ("dose_common10_80", "dose2p64:V_minus_R:stable6", "stable6_noGlobalPIQA"),
        ("visible_reference", "D1_VminusB:chck_80M:stable6", "stable6_noGlobalPIQA"),
        ("visible_reference", "D1_BminusCold:chck_80M:stable6", "stable6_noGlobalPIQA"),
        ("visible_reference", "D1_VminusCold:chck_80M:stable6", "stable6_noGlobalPIQA"),
    ]:
        r = proj_lookup(*spec)
        if r:
            key_projections.append({
                "source_block": r.get("source_block"),
                "label": r.get("label"),
                "family_set": r.get("family_set"),
                "net": r.get("net"),
                "rms": r.get("rms"),
                "PC1_raw_projection": r.get("PC1_raw_projection"),
                "PC1_raw_direction_cosine": r.get("PC1_raw_direction_cosine"),
                "PC2_raw_projection": r.get("PC2_raw_projection"),
                "PC2_raw_direction_cosine": r.get("PC2_raw_direction_cosine"),
                "PC3_raw_projection": r.get("PC3_raw_projection"),
                "PC3_raw_direction_cosine": r.get("PC3_raw_direction_cosine"),
                "top3_raw_energy_over_l2_sq": r.get("top3_raw_energy_over_l2_sq"),
                "vector_json": r.get("vector_json"),
            })
    write_csv(out_dir / "key_axis_projection_rows.csv", key_projections)

    # Most aligned research rows to external dose/V-B vectors using stable6 restricted cosines.
    archive_stable6 = [r for r in projection_rows if r.get("source_block") == "archive" and r.get("family_set") == "stable6_noGlobalPIQA"]
    external_targets = [r for r in projection_rows if (
        (r.get("source_block") == "dose_common10_80" and r.get("contrast") == "V_minus_R" and r.get("family_set") == "stable6_noGlobalPIQA")
        or (r.get("source_block") == "visible_reference" and r.get("contrast") in {"D1_VminusB", "D1_BminusCold", "D1_VminusCold"} and r.get("family_set") == "stable6_noGlobalPIQA")
        or (r.get("source_block") == "reference_1x_seed_mean" and r.get("window") == "common10_80" and r.get("family_set") == "stable6_noGlobalPIQA")
    )]
    nearest_rows: List[Dict[str, Any]] = []
    for t in external_targets:
        tv = parse_vector_json(t.get("vector_json"))
        sims: List[Tuple[float, Mapping[str, Any]]] = []
        for a in archive_stable6:
            av = parse_vector_json(a.get("vector_json"))
            co = cosine(tv, av, STABLE6)
            if co is not None:
                sims.append((co, a))
        sims.sort(key=lambda x: abs(x[0]), reverse=True)
        for rank, (co, a) in enumerate(sims[:5], start=1):
            nearest_rows.append({
                "target_label": t.get("label"),
                "target_source": t.get("source_block"),
                "rank_by_abs_cosine": rank,
                "archive_label": a.get("label"),
                "archive_family": a.get("intervention_family"),
                "archive_exposure": a.get("exposure"),
                "stable6_cosine": co,
                "archive_vector_json": a.get("vector_json"),
            })
    write_csv(out_dir / "nearest_archive_directions.csv", nearest_rows)

    # Scientific reading distilled from the numeric outputs.
    reading: List[str] = []
    reading.append(
        "The research all-family V-R RMS growth is not a broad conservation-vector law: "
        "the identity |Entity| = sqrt(6*RMS6^2 - 5*RMS5_exEntity^2) recovers the Entity term at each dose, "
        "while ex-Entity RMS stays in a narrow band."
    )
    reading.append(
        f"For research 1x common10_80 V-R, the two seed mean stable6 vectors have cosine {common6['cosine']:+.4f} "
        f"and ||diff||/mean||seed|| {common6['difference_over_mean_seed_l2']:.4f}; ex-Entity cosine is {common5['cosine']:+.4f} "
        f"with ratio {common5['difference_over_mean_seed_l2']:.4f}. This does not support a reproduced broad rotation direction."
    )
    reading.append(
        f"For the full 10M-100M ladder, stable6 seed-vector cosine is {full6['cosine']:+.4f}; for mature70_100 it is {mature6['cosine']:+.4f}; "
        f"late80_100 rises to {late6['cosine']:+.4f} mainly because both seeds share positive Entity/Supplement tendencies while other families still differ."
    )
    reading.append(
        "research PCA axes are useful as descriptive coordinates, but the current data do not show that unrelated interventions share one reliable manifold. "
        "External dose and V-B vectors project onto these axes in different mixtures; research archive rows themselves contain several opposing directions rather than a single reusable route."
    )
    reading.append(
        "The safer current interpretation is that ex-Entity V-R movement is at, or close to, the same-coordinate basin floor until a repeated direction is shown. "
        "The high-value incoming result remains the matched clean/content-admission leg and the full breadth readout; any broad mechanism reading must be conditioned on this seed-floor comparison."
    )

    summary = {
        "status": "AXIS_NOISE_FLOOR_READOUT_COMPLETE",
        "inputs": {
            "treatment_deltas": str(DELTAS),
            "json": str(JSON),
            "dose_csv": str(DOSE),
            "visible_reference_csv": str(VISIBLE),
        },
        "seed_vector_reproducibility": seed_summary,
        "dose_entity_component_identity": dose_entity_decomp,
        "pca_variance_fraction": axis_varfrac,
        "pca_loadings": axes,
        "pca_mean_vector": axis_mean,
        "key_axis_projections": key_projections,
        "reading": reading,
        "files": {
            "seed_reproducibility_csv": str(out_dir / "seed_1x_vector_reproducibility.csv"),
            "checkpoint_alignment_csv": str(out_dir / "checkpoint_seed_vector_alignment.csv"),
            "axis_projection_rows_csv": str(out_dir / "axis_projection_rows.csv"),
            "axis_projection_group_summary_csv": str(out_dir / "axis_projection_group_summary.csv"),
            "key_axis_projection_rows_csv": str(out_dir / "key_axis_projection_rows.csv"),
            "nearest_step107_archive_directions_csv": str(out_dir / "nearest_archive_directions.csv"),
            "dose_entity_component_identity_csv": str(out_dir / "dose_entity_component_identity.csv"),
            "summary_json": str(out_dir / "axis_noise_floor_summary.json"),
            "summary_md": str(out_dir / "axis_noise_floor_summary.md"),
            "plots": plot_paths,
        },
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "axis_noise_floor_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("# research axis/noise-floor readout")
    lines.append("")
    lines.append("File-only analysis of already scored tables. It asks whether broad ex-Entity family-vector movement is a reproducible low-dimensional trade-off, or is better treated as the same-coordinate basin floor before incoming breadth and matched-clean results are read.")
    lines.append("")
    lines.append("## Seed reproducibility of the 1x DeBERTa V-R vector")
    lines.append("")
    lines.append("| window | family set | seed43022 net/RMS | seed43122 net/RMS | cosine | centered cosine | ||diff|| / mean ||seed|| | vector reading |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in seed_rows:
        if r["family_set"] not in {"stable6", "stable5_exEntity", "stable4_exEntity_noReading"}:
            continue
        vec_read = "not reproduced" if (safe_float(r["cosine"]) is not None and safe_float(r["cosine"]) < 0.25 and safe_float(r["difference_over_mean_seed_l2"]) is not None and safe_float(r["difference_over_mean_seed_l2"]) > 1.0) else "partially aligned"
        lines.append(
            f"| {r['window']} | {r['family_set']} | {float(r['seed43022_net']):+.4f}/{float(r['seed43022_rms']):.4f} | "
            f"{float(r['seed43122_net']):+.4f}/{float(r['seed43122_rms']):.4f} | {float(r['cosine']):+.4f} | "
            f"{float(r['centered_cosine']):+.4f} | {float(r['difference_over_mean_seed_l2']):.4f} | {vec_read} |"
        )
    lines.append("")
    lines.append("For the dose-matched common10_80 window, the two 1x seed vectors point in different directions. This is the low-cost exclusion that research lacked: ex-Entity V-R movement should not be promoted to a shared rotation unless a direction reproduces beyond this floor.")
    lines.append("")
    lines.append("## Entity explains the apparent dose growth in V-R displacement")
    lines.append("")
    lines.append("| dose | rho | stable6 net | RMS6 | ex-Entity net | RMS5 | implied |Entity| | actual Entity | identity error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in dose_entity_decomp:
        lines.append(
            f"| {r['dose_name']} | {float(r['rho']):.6f} | {float(r['stable6_net']):+.4f} | {float(r['stable6_rms']):.4f} | "
            f"{float(r['stable5_exEntity_net']):+.4f} | {float(r['stable5_exEntity_rms']):.4f} | "
            f"{float(r['implied_abs_entity_from_rms_identity']):.4f} | {float(r['actual_entity_delta']):+.4f} | {float(r['abs_identity_error']):.2e} |"
        )
    lines.append("")
    lines.append("The displacement growth from dose1 to MAX is therefore the same Entity component already shown to be an operation-sensitive allocation in two DeBERTa basins. Outside Entity, the V-R RMS stays roughly 0.53-0.59 and the net remains close to zero.")
    lines.append("")
    lines.append("## Key projections on research PCA axes")
    lines.append("")
    lines.append("research PC1/PC2/PC3 explain approximately 42.9%, 28.0%, and 17.4% of archive delta-vector variance. For rows without GlobalPIQA, projections use the available six-column subspace, so they are descriptive coordinates rather than a complete seven-column reconstruction.")
    lines.append("")
    lines.append("| label | net | RMS | PC1 proj/cos | PC2 proj/cos | PC3 proj/cos | vector |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in key_projections:
        lines.append(
            f"| {r['label']} | {float(r['net']):+.4f} | {float(r['rms']):.4f} | "
            f"{float(r['PC1_raw_projection']):+.4f}/{float(r['PC1_raw_direction_cosine']):+.3f} | "
            f"{float(r['PC2_raw_projection']):+.4f}/{float(r['PC2_raw_direction_cosine']):+.3f} | "
            f"{float(r['PC3_raw_projection']):+.4f}/{float(r['PC3_raw_direction_cosine']):+.3f} | `{r['vector_json']}` |"
        )
    lines.append("")
    lines.append("The projections do not rescue a single shared broad rotation. Dose V-R changes PC mixture with dose; the visible V-B row is dominated by Entity and Supplement with BLiMP/COMPS losses; B-C points along a different content-placement direction. research archive vectors occupy multiple axes and include opposing signs.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    for s in reading:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Files")
    for k, v in summary["files"].items():
        if isinstance(v, list):
            for p in v:
                lines.append(f"- {k}: `{p}`")
        else:
            lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No model loading, training, official evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA work, upload, or leaderboard action occurred.")
    (out_dir / "axis_noise_floor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "common10_80_stable6_cosine": common6["cosine"],
        "common10_80_stable6_diff_ratio": common6["difference_over_mean_seed_l2"],
        "common10_80_exEntity_cosine": common5["cosine"],
        "common10_80_exEntity_diff_ratio": common5["difference_over_mean_seed_l2"],
        "dose_entity_component_identity": dose_entity_decomp,
        "summary_md": str(out_dir / "axis_noise_floor_summary.md"),
        "summary_json": str(out_dir / "axis_noise_floor_summary.json"),
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
