#!/usr/bin/env python3
"""research: centered research-axis correction and contribution table.

The first research readout saved both raw axis dots and PCA-centered projections, but
its human-readable table emphasized raw projections.  This file-only supplement
makes the centered coordinates explicit and decomposes selected projections by
family, so the result is not misread as a shared manifold.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path("experiments/archive/frontier_consolidation")
DATA = ROOT / "data"
OUT_DIR = DATA / "axis_noise_floor_readout"
AXIS_ROWS = OUT_DIR / "axis_projection_rows.csv"
JSON = DATA / "score_vector_tradeoff" / "score_vector_tradeoff.json"
SEED_REPRO = OUT_DIR / "seed_1x_vector_reproducibility.csv"
DELTAS = DATA / "full_deberta_seed_ladder_stable_eval" / "full_deberta_seed_ladder_treatment_deltas.csv"

COLS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
STABLE5_EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
KEY_LABELS = [
    "seed43022_common10_80_VminusR",
    "seed43122_common10_80_VminusR",
    "seed43022_common10_80_VminusR_exEntity",
    "seed43122_common10_80_VminusR_exEntity",
    "dose1:V_minus_R:stable6",
    "dose1p82:V_minus_R:stable6",
    "dose2p64:V_minus_R:stable6",
    "D1_VminusB:chck_80M:stable6",
    "D1_BminusCold:chck_80M:stable6",
    "D1_VminusCold:chck_80M:stable6",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def parse_vec(s: str) -> Dict[str, float]:
    return {str(k): float(v) for k, v in json.loads(s).items()}


def fl(x: Any) -> float:
    return float(x)


def load_step107() -> tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    obj = json.loads(JSON.read_text(encoding="utf-8"))
    axes: Dict[str, Dict[str, float]] = {}
    varfrac: Dict[str, float] = {}
    for pc in obj["pca"][:3]:
        name = f"PC{pc['component']}"
        axes[name] = {c: float(v) for c, v in pc["loadings"].items()}
        varfrac[name] = float(pc["variance_fraction"])
    mean: Dict[str, float] = {}
    for c in COLS7:
        vals = [float(r["deltas"][c]) for r in obj["records"]]
        mean[c] = float(np.mean(vals))
    return axes, varfrac, mean


def dot_contrib(vec: Mapping[str, float], axis: Mapping[str, float], mean: Mapping[str, float], cols: Sequence[str]) -> Dict[str, float]:
    return {c: (float(vec[c]) - float(mean.get(c, 0.0))) * float(axis[c]) for c in cols if c in vec and c in axis}


def norm(vec: Mapping[str, float], cols: Sequence[str]) -> float:
    return float(np.linalg.norm([float(vec[c]) for c in cols if c in vec]))


def cosine(a: Mapping[str, float], b: Mapping[str, float], cols: Sequence[str]) -> float | None:
    xs = np.array([float(a[c]) for c in cols if c in a and c in b], dtype=float)
    ys = np.array([float(b[c]) for c in cols if c in a and c in b], dtype=float)
    if xs.size == 0:
        return None
    nx = float(np.linalg.norm(xs)); ny = float(np.linalg.norm(ys))
    if nx <= 1e-12 or ny <= 1e-12:
        return None
    return float(np.dot(xs, ys) / (nx * ny))


def main() -> None:
    axes, varfrac, mean = load_step107()
    rows = read_csv(AXIS_ROWS)
    selected = [r for r in rows if r.get("label") in KEY_LABELS]

    centered_rows: List[Dict[str, Any]] = []
    contrib_rows: List[Dict[str, Any]] = []
    for r in selected:
        vec = parse_vec(r["vector_json"])
        cols = [c for c in COLS7 if c in vec]
        centered_arr = np.array([vec[c] - mean.get(c, 0.0) for c in cols], dtype=float)
        centered_norm = float(np.linalg.norm(centered_arr))
        out = {
            "label": r["label"],
            "family_set": r["family_set"],
            "net": r["net"],
            "rms": r["rms"],
            "centered_to_step107_mean_l2": centered_norm,
            "columns": ";".join(cols),
            "vector_json": r["vector_json"],
        }
        for pc in ["PC1", "PC2", "PC3"]:
            contrib = dot_contrib(vec, axes[pc], mean, cols)
            proj = sum(contrib.values())
            axis_norm = norm(axes[pc], cols)
            cos = proj / (centered_norm * axis_norm) if centered_norm > 1e-12 and axis_norm > 1e-12 else None
            out[f"{pc}_variance_fraction_step107"] = varfrac[pc]
            out[f"{pc}_centered_projection"] = proj
            out[f"{pc}_centered_direction_cosine"] = cos
            abs_sorted = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
            out[f"{pc}_top_contributions"] = json.dumps(abs_sorted[:4], ensure_ascii=False)
            for fam, val in contrib.items():
                contrib_rows.append({
                    "label": r["label"],
                    "family_set": r["family_set"],
                    "pc": pc,
                    "family": fam,
                    "centered_coordinate": vec[fam] - mean.get(fam, 0.0),
                    "loading": axes[pc][fam],
                    "projection_contribution": val,
                    "projection_total": proj,
                    "fraction_of_total_if_nonzero": val / proj if abs(proj) > 1e-12 else None,
                })
        centered_rows.append(out)

    write_csv(OUT_DIR / "centered_axis_key_rows.csv", centered_rows)
    write_csv(OUT_DIR / "centered_axis_family_contributions.csv", contrib_rows)

    # Extra sign agreement for the common10_80 vectors.
    seed_delta_rows = read_csv(DELTAS)
    common = [r for r in seed_delta_rows if r["checkpoint"] in {f"chck_{i}M" for i in range(10, 81, 10)}]
    by_seed: Dict[int, Dict[str, List[float]]] = {43022: {c: [] for c in STABLE6}, 43122: {c: [] for c in STABLE6}}
    for r in common:
        seed = int(r["seed"])
        if seed in by_seed:
            for c in STABLE6:
                by_seed[seed][c].append(float(r[c]))
    mean_vec = {seed: {c: float(np.mean(vals)) for c, vals in fams.items()} for seed, fams in by_seed.items()}
    sign_rows: List[Dict[str, Any]] = []
    for famset, cols in [("stable6", STABLE6), ("stable5_exEntity", STABLE5_EX_ENTITY)]:
        agree = 0; disagree = 0; zeros = 0
        for c in cols:
            a = mean_vec[43022][c]
            b = mean_vec[43122][c]
            sa = 0 if abs(a) < 1e-12 else (1 if a > 0 else -1)
            sb = 0 if abs(b) < 1e-12 else (1 if b > 0 else -1)
            if sa == 0 or sb == 0:
                zeros += 1
            elif sa == sb:
                agree += 1
            else:
                disagree += 1
            sign_rows.append({"family_set": famset, "family": c, "seed43022_delta": a, "seed43122_delta": b, "sign_relation": "agree" if sa == sb and sa != 0 else "disagree" if sa != 0 and sb != 0 else "zero"})
        sign_rows.append({"family_set": famset, "family": "__summary__", "seed43022_delta": "", "seed43122_delta": "", "sign_relation": f"agree={agree};disagree={disagree};zero={zeros}"})
    write_csv(OUT_DIR / "common10_80_seed_sign_agreement.csv", sign_rows)

    def get(label: str) -> Mapping[str, Any]:
        for r in centered_rows:
            if r["label"] == label:
                return r
        raise KeyError(label)

    s43022 = get("seed43022_common10_80_VminusR")
    s43122 = get("seed43122_common10_80_VminusR")
    s43022_ex = get("seed43022_common10_80_VminusR_exEntity")
    s43122_ex = get("seed43122_common10_80_VminusR_exEntity")
    vmb = get("D1_VminusB:chck_80M:stable6")
    bmc = get("D1_BminusCold:chck_80M:stable6")
    maxvr = get("dose2p64:V_minus_R:stable6")

    lines: List[str] = []
    lines.append("# research centered-axis correction")
    lines.append("")
    lines.append("This supplement corrects the first research summary by separating raw dot products from the actual PCA-centered coordinates. research fitted PCA after subtracting the research archive mean, so a new vector's comparable coordinate is `(delta - mean) dot PC`, not the raw `delta dot PC`.")
    lines.append("")
    lines.append("## Common10_80 1x seed directions")
    lines.append("")
    lines.append(f"- Stable6 raw family-vector cosine remains {float(next(r for r in read_csv(SEED_REPRO) if r['window']=='common10_80' and r['family_set']=='stable6')['cosine']):+.4f}; ex-Entity cosine remains {float(next(r for r in read_csv(SEED_REPRO) if r['window']=='common10_80' and r['family_set']=='stable5_exEntity')['cosine']):+.4f}.")
    lines.append("- The stable6 centered research coordinates have a shared PC2-negative / PC3-positive component, but the ex-Entity centered coordinates are small and sign-unstable. The shared stable6 axis component is therefore mainly the already-known positive Entity coordinate, not a reproduced broad ex-Entity direction.")
    lines.append("")
    lines.append("| label | centered L2 | PC1 centered proj/cos | PC2 centered proj/cos | PC3 centered proj/cos | PC2 top contributions | PC3 top contributions |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for r in [s43022, s43122, s43022_ex, s43122_ex, maxvr, vmb, bmc]:
        lines.append(
            f"| {r['label']} | {float(r['centered_to_step107_mean_l2']):.4f} | "
            f"{float(r['PC1_centered_projection']):+.4f}/{float(r['PC1_centered_direction_cosine']):+.3f} | "
            f"{float(r['PC2_centered_projection']):+.4f}/{float(r['PC2_centered_direction_cosine']):+.3f} | "
            f"{float(r['PC3_centered_projection']):+.4f}/{float(r['PC3_centered_direction_cosine']):+.3f} | "
            f"`{r['PC2_top_contributions']}` | `{r['PC3_top_contributions']}` |"
        )
    lines.append("")
    lines.append("## Scientific correction")
    lines.append("")
    lines.append("- The research statement that the intervention norm grows while broad net stays flat is too broad. For V-R dose, all-family radius growth is overwhelmingly the Entity component; ex-Entity radius is nearly dose-invariant.")
    lines.append("- Ex-Entity V-R is not absent: across dose it can reorient tangentially, and from 1x to MAX its endpoint-to-endpoint ex-Entity displacement is nontrivial. But with only seed43022 at higher doses and with the two 1x seed directions anti-aligned over common10_80, that movement is currently indistinguishable from same-coordinate basin variation rather than a reproducible trade-off direction.")
    lines.append("- research axes remain useful descriptors. They do not establish a shared intervention manifold here because the stable6 PC2/PC3 similarity is mainly Entity, ex-Entity projections are small/unstable, and rows omitting GlobalPIQA use restricted axes with large missing PC loadings.")
    lines.append("- A01 research adds an independent check on the Entity V-B surface: official macro V-B is real and positive across zero/nonzero operations, but paired binding flow does not move examples into both-correct retained-state form; it shifts affected-only upward while eroding unaffected-only. That further weakens an Entity-driven argument for launching the permuted companion before broad ex-Entity V-B or content-admission evidence changes.")
    lines.append("- The incoming matched-clean scores should therefore be read primarily as a content-placement test, not as a rescue of the broad V-R/conservation-vector idea. The full breadth table should be read family-by-family and ex-Entity before any mechanism-bearing V-B claim.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- centered_axis_key_rows_csv: `{OUT_DIR / 'centered_axis_key_rows.csv'}`")
    lines.append(f"- centered_axis_family_contributions_csv: `{OUT_DIR / 'centered_axis_family_contributions.csv'}`")
    lines.append(f"- common10_80_seed_sign_agreement_csv: `{OUT_DIR / 'common10_80_seed_sign_agreement.csv'}`")
    lines.append(f"- summary_md: `{(OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_correction.md')}`")
    lines.append("")
    lines.append("No model loading, training, official evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA work, upload, or leaderboard action occurred.")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_correction.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "CENTERED_AXIS_CORRECTION_COMPLETE",
        "centered_axis_key_rows_csv": str(OUT_DIR / "centered_axis_key_rows.csv"),
        "centered_axis_family_contributions_csv": str(OUT_DIR / "centered_axis_family_contributions.csv"),
        "common10_80_seed_sign_agreement_csv": str(OUT_DIR / "common10_80_seed_sign_agreement.csv"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_correction.md')),
        "stable6_seed43022_PC2_PC3_centered": [s43022["PC2_centered_projection"], s43022["PC3_centered_projection"]],
        "stable6_seed43122_PC2_PC3_centered": [s43122["PC2_centered_projection"], s43122["PC3_centered_projection"]],
        "exEntity_seed43022_PC1_PC2_PC3_centered": [s43022_ex["PC1_centered_projection"], s43022_ex["PC2_centered_projection"], s43022_ex["PC3_centered_projection"]],
        "exEntity_seed43122_PC1_PC2_PC3_centered": [s43122_ex["PC1_centered_projection"], s43122_ex["PC2_centered_projection"], s43122_ex["PC3_centered_projection"]],
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
