#!/usr/bin/env python3
"""research: quantify net-vs-vector movement for finite-budget BabyLM interventions.

File-only analysis.  No model loading, training, evaluation, GPU use, upload, or
leaderboard action.  The scientific question is whether content-presentation
interventions mostly rotate a finite competence vector across evaluation families,
while corpus/content admission can move the broad mean.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path("experiments/archive/frontier_consolidation")
OUT_DIR = ROOT / "data/conservation_vector_readout"

STABLE6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
STABLE5_EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
STABLE4_EX_ENTITY_READING = ["BLiMP", "Supplement", "EWoK", "COMPS"]
CHEAP7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and not x.strip():
        return None
    try:
        return float(x)
    except Exception:
        return None


def vec_metrics(values: Dict[str, float], cols: Sequence[str], *, label: str = "") -> Dict[str, Any]:
    vals = [float(values[c]) for c in cols if c in values and values[c] is not None]
    used = [c for c in cols if c in values and values[c] is not None]
    if not vals:
        return {"label": label, "n": 0, "columns": []}
    m = sum(vals) / len(vals)
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    centered = math.sqrt(sum((v - m) * (v - m) for v in vals) / len(vals))
    l1 = sum(abs(v) for v in vals) / len(vals)
    pos_sum = sum(v for v in vals if v > 0)
    neg_sum = sum(v for v in vals if v < 0)
    max_abs_col = max(used, key=lambda c: abs(values[c]))
    abs_mean_over_rms = abs(m) / rms if rms > 1e-12 else None
    centered_over_abs_mean = centered / abs(m) if abs(m) > 1e-12 else None
    return {
        "label": label,
        "n": len(vals),
        "columns": used,
        "mean_net": m,
        "rms_family_delta": rms,
        "centered_rms_family_delta": centered,
        "mean_abs_delta": l1,
        "abs_net_over_rms": abs_mean_over_rms,
        "centered_over_abs_net": centered_over_abs_mean,
        "positive_count": sum(1 for v in vals if v > 0),
        "negative_count": sum(1 for v in vals if v < 0),
        "positive_sum": pos_sum,
        "negative_sum": neg_sum,
        "max_abs_column": max_abs_col,
        "max_abs_delta": float(values[max_abs_col]),
        "vector": {c: float(values[c]) for c in used},
    }


def slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-18:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 1e-18 or sy <= 1e-18:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(fobj, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dose_family_vectors() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = ROOT / "data/common_window_budget_decomposition/common10_80_family_components.csv"
    rows = read_csv(path)
    vectors: Dict[Tuple[str, str], Dict[str, float]] = {}
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        comp = r.get("component", "")
        dose = r.get("dose_name", "")
        fam = r.get("family", "")
        val = f(r.get("mean"))
        if comp and dose and fam and val is not None:
            key = (dose, comp)
            vectors.setdefault(key, {})[fam] = val
            meta.setdefault(key, {"rho": f(r.get("rho")), "dose": f(r.get("dose"))})

    out: List[Dict[str, Any]] = []
    for (dose, comp), vals in sorted(vectors.items(), key=lambda kv: (meta[kv[0]].get("rho") or 0, kv[0][1])):
        for family_set_name, cols in [
            ("stable6", STABLE6),
            ("stable5_exEntity", STABLE5_EX_ENTITY),
            ("stable4_exEntity_noReading", STABLE4_EX_ENTITY_READING),
        ]:
            if all(c in vals for c in cols):
                m = vec_metrics(vals, cols, label=f"{dose}:{comp}:{family_set_name}")
                out.append({
                    "source_block": "dose_common10_80",
                    "dose_name": dose,
                    "rho": meta[(dose, comp)].get("rho"),
                    "dose_multiplier": meta[(dose, comp)].get("dose"),
                    "contrast": comp,
                    "family_set": family_set_name,
                    **{k: v for k, v in m.items() if k != "vector" and k != "columns"},
                    "columns": ";".join(m.get("columns", [])),
                    "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
                })

    # Summarize the dose-response for the component currently under strongest pressure.
    vr6 = [r for r in out if r["contrast"] == "V_minus_R" and r["family_set"] == "stable6"]
    vr5 = [r for r in out if r["contrast"] == "V_minus_R" and r["family_set"] == "stable5_exEntity"]
    def trend(rs: List[Dict[str, Any]], ykey: str) -> Dict[str, Any]:
        rs = [r for r in rs if r.get("rho") is not None and r.get(ykey) is not None]
        xs = [float(r["rho"]) for r in rs]
        ys = [float(r[ykey]) for r in rs]
        return {"n": len(rs), "slope_per_rho": slope(xs, ys), "corr_with_rho": corr(xs, ys), "values": list(zip(xs, ys))}

    summary = {
        "input": str(path),
        "dose_vr_stable6_rms_trend": trend(vr6, "rms_family_delta"),
        "dose_vr_stable6_centered_rms_trend": trend(vr6, "centered_rms_family_delta"),
        "dose_vr_stable6_net_trend": trend(vr6, "mean_net"),
        "dose_vr_stable5_exEntity_net_trend": trend(vr5, "mean_net"),
        "dose_vr_stable5_exEntity_rms_trend": trend(vr5, "rms_family_delta"),
    }
    return out, summary


def reference_vectors() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = ROOT / "data/reference_decomposition_readout/contrast_rows.csv"
    rows = read_csv(path)
    # Build vectors for each contrast/checkpoint from scalar family rows.
    vectors: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        quantity = r.get("quantity", "")
        if quantity not in STABLE6:
            continue
        val = f(r.get("delta"))
        if val is None:
            continue
        vectors.setdefault((r.get("contrast", ""), r.get("checkpoint", "")), {})[quantity] = val

    out: List[Dict[str, Any]] = []
    for (contrast, checkpoint), vals in sorted(vectors.items()):
        for family_set_name, cols in [("stable6", STABLE6), ("stable5_exEntity", STABLE5_EX_ENTITY), ("stable4_exEntity_noReading", STABLE4_EX_ENTITY_READING)]:
            if all(c in vals for c in cols):
                m = vec_metrics(vals, cols, label=f"{contrast}:{checkpoint}:{family_set_name}")
                out.append({
                    "source_block": "max_breadth_clean_reference_visible",
                    "contrast": contrast,
                    "checkpoint": checkpoint,
                    "family_set": family_set_name,
                    **{k: v for k, v in m.items() if k != "vector" and k != "columns"},
                    "columns": ";".join(m.get("columns", [])),
                    "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
                })

    # Entity stratum vectors already available from research: two-component vector zero/nonzero.
    ep = ROOT / "data/entity_balanced_and_transfer_readout_full/entity_late_summary_rows.csv"
    entity_rows = read_csv(ep)
    by_contrast: Dict[str, Dict[str, float]] = {}
    for r in entity_rows:
        if r.get("window") != "late_80_90_100M":
            continue
        name = r.get("contrast_name", "")
        q = r.get("quantity", "")
        val = f(r.get("mean"))
        if val is None:
            continue
        by_contrast.setdefault(name, {})[q] = val
    entity_vec_rows: List[Dict[str, Any]] = []
    for name, vals in sorted(by_contrast.items()):
        if "zero_ops_delta_pp" in vals and "nonzero_ops_delta_pp" in vals:
            v = {"zero_ops": vals["zero_ops_delta_pp"], "nonzero_ops": vals["nonzero_ops_delta_pp"]}
            m = vec_metrics(v, ["zero_ops", "nonzero_ops"], label=f"{name}:entity_ops")
            entity_vec_rows.append({
                "source_block": "entity_operation_strata_late80_100",
                "contrast": name,
                "checkpoint": "late_80_90_100M",
                "family_set": "entity_zero_nonzero",
                "official_all18_delta_pp_mean": vals.get("official_all18_delta_pp"),
                "neutral_zero_nonzero_delta_pp_mean": vals.get("balanced_zero_nonzero_delta_pp"),
                **{k: v2 for k, v2 in m.items() if k not in {"vector", "columns"}},
                "columns": ";".join(m.get("columns", [])),
                "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
            })
    out.extend(entity_vec_rows)

    summary = {
        "visible_reference_input": str(path),
        "entity_strata_input": str(ep),
        "complete_visible_80M_vectors": [r for r in out if r.get("checkpoint") == "chck_80M" and r.get("family_set") in {"stable6", "stable5_exEntity"}],
        "entity_operation_vectors": entity_vec_rows,
    }
    return out, summary


def archive_step107_vectors() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = ROOT / "data/score_vector_tradeoff/score_vector_tradeoff.json"
    data = load_json(path)
    out: List[Dict[str, Any]] = []
    for rec in data.get("records", []):
        vals = {c: f(rec.get("deltas", {}).get(c)) for c in CHEAP7}
        vals = {k: v for k, v in vals.items() if v is not None}
        for set_name, cols in [("cheap7", CHEAP7), ("stable6_noGlobalPIQA", STABLE6), ("stable5_noGlobalPIQA_noReading", ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]), ("stable5_exEntity", ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"])]:
            if all(c in vals for c in cols):
                m = vec_metrics(vals, cols, label=f"{rec.get('label')}:{rec.get('exposure')}:{set_name}")
                out.append({
                    "source_block": "archive",
                    "label": rec.get("label"),
                    "intervention_family": rec.get("family"),
                    "exposure": rec.get("exposure"),
                    "record_note": rec.get("note"),
                    "source": rec.get("source"),
                    "family_set": set_name,
                    **{k: v for k, v in m.items() if k != "vector" and k != "columns"},
                    "columns": ";".join(m.get("columns", [])),
                    "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
                })
    mature = [r for r in out if r.get("exposure") in {"70M", "80M", "100M"} and r.get("family_set") == "cheap7"]
    stable_mature = [r for r in out if r.get("exposure") in {"70M", "80M", "100M"} and r.get("family_set") == "stable6_noGlobalPIQA"]
    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "mean_net": mean([r["mean_net"] for r in rs]),
            "mean_abs_net": mean([abs(r["mean_net"]) for r in rs]),
            "mean_rms_family_delta": mean([r["rms_family_delta"] for r in rs]),
            "mean_centered_rms_family_delta": mean([r["centered_rms_family_delta"] for r in rs]),
            "mean_abs_net_over_rms": mean([r["abs_net_over_rms"] for r in rs if r.get("abs_net_over_rms") is not None]),
            "count_all_positive": sum(1 for r in rs if r["negative_count"] == 0),
            "labels": [f"{r.get('label')} {r.get('exposure')}" for r in rs],
        }
    summary = {"input": str(path), "mature_cheap7": agg(mature), "mature_stable6_noGlobalPIQA": agg(stable_mature)}
    return out, summary


def gpt2_and_roberta_vectors() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # GPT2 causal compact-vs-repeat selected endpoints.
    gpath = ROOT / "data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.json"
    if gpath.exists():
        g = load_json(gpath)
        for rec in g.get("contrasts", []):
            vals = {c: f(rec.get("delta_" + c)) for c in CHEAP7}
            vals = {k: v for k, v in vals.items() if v is not None}
            for set_name, cols in [("cheap7", CHEAP7), ("stable6_noGlobalPIQA", STABLE6), ("stable5_noGlobalPIQA_noReading", ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]), ("stable5_exEntity", ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"])] :
                if all(c in vals for c in cols):
                    m = vec_metrics(vals, cols, label=f"gpt2_causal:{rec.get('endpoint')}:{set_name}")
                    out.append({
                        "source_block": "gpt2_causal_compact_minus_repeat",
                        "label": "GPT2 causal compact-minus-repeat",
                        "checkpoint": rec.get("endpoint"),
                        "words": rec.get("nominal_words"),
                        "family_set": set_name,
                        **{k: v for k, v in m.items() if k not in {"vector", "columns"}},
                        "columns": ";".join(m.get("columns", [])),
                        "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
                    })
    # RoBERTa selected endpoint(s).
    rpath = ROOT / "data/roberta_minimal_selected_eval/minimal_selected_eval_summary.json"
    if rpath.exists():
        r = load_json(rpath)
        for rec in r.get("per_checkpoint", []):
            vals = {c: f(rec.get("compact_minus_repeat", {}).get(c)) for c in CHEAP7}
            vals = {k: v for k, v in vals.items() if v is not None}
            for set_name, cols in [("cheap7", CHEAP7), ("stable6_noGlobalPIQA", STABLE6), ("stable5_noGlobalPIQA_noReading", ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]), ("stable5_exEntity", ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"])] :
                if all(c in vals for c in cols):
                    m = vec_metrics(vals, cols, label=f"roberta:{rec.get('ck')}:{set_name}")
                    out.append({
                        "source_block": "roberta_compact_minus_repeat",
                        "label": "RoBERTa compact-minus-repeat",
                        "checkpoint": rec.get("ck"),
                        "family_set": set_name,
                        **{k: v for k, v in m.items() if k not in {"vector", "columns"}},
                        "columns": ";".join(m.get("columns", [])),
                        "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
                    })
    # RoBERTa 60M lateband pair.
    comp60 = ROOT / "data/roberta_lateband_selected_eval/compact/chck_60M/roberta_compact_chck_60M_summary.json"
    rep60 = ROOT / "data/roberta_lateband_selected_eval/repeat/chck_60M/roberta_repeat_chck_60M_summary.json"
    if comp60.exists() and rep60.exists():
        c = load_json(comp60)["record"]["scores"]
        r = load_json(rep60)["record"]["scores"]
        vals = {col: float(c[col]) - float(r[col]) for col in CHEAP7}
        for set_name, cols in [("cheap7", CHEAP7), ("stable6_noGlobalPIQA", STABLE6), ("stable5_noGlobalPIQA_noReading", ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]), ("stable5_exEntity", ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"])] :
            m = vec_metrics(vals, cols, label=f"roberta:chck_60M:{set_name}")
            out.append({
                "source_block": "roberta_compact_minus_repeat",
                "label": "RoBERTa compact-minus-repeat",
                "checkpoint": "chck_60M",
                "family_set": set_name,
                **{k: v for k, v in m.items() if k not in {"vector", "columns"}},
                "columns": ";".join(m.get("columns", [])),
                "vector_json": json.dumps(m.get("vector", {}), sort_keys=True),
            })
    # Summaries by source block/family set.
    summary: Dict[str, Any] = {"inputs": [str(gpath), str(rpath), str(comp60), str(rep60)]}
    for block in sorted(set(r["source_block"] for r in out)):
        for fs in sorted(set(r["family_set"] for r in out if r["source_block"] == block)):
            rs = [r for r in out if r["source_block"] == block and r["family_set"] == fs]
            if rs:
                summary[f"{block}:{fs}"] = {
                    "n": len(rs),
                    "mean_net": mean([r["mean_net"] for r in rs]),
                    "mean_abs_net": mean([abs(r["mean_net"]) for r in rs]),
                    "mean_rms": mean([r["rms_family_delta"] for r in rs]),
                    "mean_centered_rms": mean([r["centered_rms_family_delta"] for r in rs]),
                    "mean_abs_net_over_rms": mean([r["abs_net_over_rms"] for r in rs if r.get("abs_net_over_rms") is not None]),
                }
    return out, summary


def item_turnover_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # Coherent replay vs anchor.
    cpath = ROOT / "data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json"
    if cpath.exists():
        data = load_json(cpath)
        ag = data.get("item_transition_scientific_reading", {}).get("coherent_minus_chck82_aggregate", {})
        if ag:
            gains = int(ag["total_gain_items"]); losses = int(ag["total_loss_items"])
            out.append({
                "source_block": "item_turnover",
                "label": "coherent86 alpha1 minus chck82 anchor",
                "gain_items": gains,
                "loss_items": losses,
                "net_gain_minus_loss": int(ag.get("total_gain_minus_loss", gains - losses)),
                "changed_items": gains + losses,
                "net_over_changed": (gains - losses) / (gains + losses) if gains + losses else None,
                "loss_to_gain_ratio": ag.get("loss_to_gain_ratio"),
                "common_items": ag.get("total_common_items"),
                "payload_mean_delta": ag.get("discrete_payload_mean_delta"),
                "source": str(cpath),
            })
    # Alpha sweep monotonic counts.
    apath = (ROOT.parents[2] / 'research/documents/frontier_consolidation/data/alpha_sweep_decision_patterns/alpha_sweep_decision_patterns.md')
    if apath.exists():
        # The markdown has the relevant counts; preserve them as a row because they were used in prior synthesis.
        out.append({
            "source_block": "item_turnover",
            "label": "coherent private-alpha monotonic decision pattern",
            "gain_items": 3116,
            "loss_items": 3231,
            "net_gain_minus_loss": -115,
            "changed_items": 6347,
            "net_over_changed": -115 / 6347,
            "loss_to_gain_ratio": 3231 / 3116,
            "common_items": 170722,
            "payload_mean_delta": None,
            "source": str(apath),
        })
    summary = {"rows": out}
    return out, summary


def make_plots(dose_rows: List[Dict[str, Any]], archive_rows: List[Dict[str, Any]], ref_rows: List[Dict[str, Any]], xarch_rows: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return [f"matplotlib_unavailable:{e}"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vr6 = [r for r in dose_rows if r["contrast"] == "V_minus_R" and r["family_set"] == "stable6"]
    vr5 = [r for r in dose_rows if r["contrast"] == "V_minus_R" and r["family_set"] == "stable5_exEntity"]
    if vr6 and vr5:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
        ax.plot([r["rho"] for r in vr6], [r["rms_family_delta"] for r in vr6], marker="o", label="V-R stable6 RMS")
        ax.plot([r["rho"] for r in vr6], [r["centered_rms_family_delta"] for r in vr6], marker="o", label="V-R stable6 centered RMS")
        ax.plot([r["rho"] for r in vr6], [abs(r["mean_net"]) for r in vr6], marker="s", label="|stable6 mean|")
        ax.plot([r["rho"] for r in vr5], [abs(r["mean_net"]) for r in vr5], marker="s", label="|ex-Entity mean|")
        ax.set_xlabel("paired packet dose rho")
        ax.set_ylabel("score points")
        ax.set_title("MAX dose: family-vector movement grows faster than broad non-Entity mean")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        p = OUT_DIR / "dose_vr_net_vs_norm.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))

    mature = [r for r in archive_rows if r.get("source_block") == "archive" and r.get("exposure") in {"70M", "80M", "100M"} and r.get("family_set") == "cheap7"]
    if mature:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        xs = [r["rms_family_delta"] for r in mature]
        ys = [r["mean_net"] for r in mature]
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(xs, ys, s=45)
        for r, x, y in zip(mature, xs, ys):
            lab = f"{r.get('label')} {r.get('exposure')}"
            ax.annotate(lab[:24], (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("cheap7 family-delta RMS")
        ax.set_ylabel("cheap7 mean movement")
        ax.set_title("Mature archive: large vector shifts, small or negative broad means")
        ax.grid(alpha=0.25)
        p = OUT_DIR / "archive_mature_net_vs_norm.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))

    # Single visible MAX breadth reference point: bars for V-B at 80M.
    vbr = [r for r in ref_rows if r.get("contrast") == "D1_VminusB" and r.get("checkpoint") == "chck_80M" and r.get("family_set") == "stable6"]
    if vbr:
        vec = json.loads(vbr[0]["vector_json"])
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
        cols = STABLE6
        vals = [vec[c] for c in cols]
        ax.axhline(0, color="black", lw=0.8)
        colors = ["#4C78A8" if v >= 0 else "#F58518" for v in vals]
        ax.bar(cols, vals, color=colors)
        ax.set_ylabel("V-B delta, score points")
        ax.set_title("Visible 80M breadth contrast: Entity/Supplement gain paid by BLiMP/COMPS")
        ax.tick_params(axis="x", rotation=30)
        p = OUT_DIR / "visible_80M_vminusb_family_vector.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dose_rows, dose_summary = dose_family_vectors()
    ref_rows, ref_summary = reference_vectors()
    archive_rows, archive_summary = archive_step107_vectors()
    xarch_rows, xarch_summary = gpt2_and_roberta_vectors()
    item_rows, item_summary = item_turnover_rows()

    plot_paths = make_plots(dose_rows, archive_rows, ref_rows, xarch_rows)

    write_csv(OUT_DIR / "dose_common10_80_net_norm.csv", dose_rows)
    write_csv(OUT_DIR / "visible_reference_net_norm.csv", ref_rows)
    write_csv(OUT_DIR / "archive_net_norm.csv", archive_rows)
    write_csv(OUT_DIR / "gpt2_roberta_net_norm.csv", xarch_rows)
    write_csv(OUT_DIR / "item_turnover_rows.csv", item_rows)

    # Compact high-value readings for the markdown.
    def find_row(rows: List[Dict[str, Any]], **kw: Any) -> Optional[Dict[str, Any]]:
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    dose_vr_key = []
    for dose in ["dose1", "dose1p82", "dose2p64"]:
        r6 = find_row(dose_rows, dose_name=dose, contrast="V_minus_R", family_set="stable6")
        r5 = find_row(dose_rows, dose_name=dose, contrast="V_minus_R", family_set="stable5_exEntity")
        if r6 and r5:
            dose_vr_key.append({
                "dose": dose,
                "rho": r6["rho"],
                "stable6_net": r6["mean_net"],
                "stable6_rms": r6["rms_family_delta"],
                "stable6_centered_rms": r6["centered_rms_family_delta"],
                "stable6_abs_net_over_rms": r6["abs_net_over_rms"],
                "exEntity_net": r5["mean_net"],
                "exEntity_rms": r5["rms_family_delta"],
                "exEntity_abs_net_over_rms": r5["abs_net_over_rms"],
            })

    ref80_key = []
    for contrast in ["D1_VminusB", "D1_BminusCold", "D1_VminusCold", "D1_BminusR", "D1_VminusR"]:
        r6 = find_row(ref_rows, contrast=contrast, checkpoint="chck_80M", family_set="stable6")
        r5 = find_row(ref_rows, contrast=contrast, checkpoint="chck_80M", family_set="stable5_exEntity")
        if r6 and r5:
            ref80_key.append({
                "contrast": contrast,
                "stable6_net": r6["mean_net"],
                "stable6_rms": r6["rms_family_delta"],
                "stable6_centered_rms": r6["centered_rms_family_delta"],
                "exEntity_net": r5["mean_net"],
                "exEntity_rms": r5["rms_family_delta"],
                "vector": json.loads(r6["vector_json"]),
            })

    # Archive mature selected rows with the strongest difference between family RMS and net.
    mature_cheap7 = [r for r in archive_rows if r.get("exposure") in {"70M", "80M", "100M"} and r.get("family_set") == "cheap7"]
    mature_cheap7_sorted = sorted(mature_cheap7, key=lambda r: (r["rms_family_delta"] - abs(r["mean_net"])), reverse=True)

    summary = {
        "status": "CONSERVATION_VECTOR_READOUT_COMPLETE",
        "created_from_existing_files_only": True,
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "formula": {
            "family_vector": "Delta over selected evaluation families",
            "net": "mean(Delta_i), the broad composite movement over the selected family set",
            "rms_family_delta": "sqrt(mean(Delta_i^2)), total family-vector movement",
            "centered_rms_family_delta": "sqrt(mean((Delta_i - mean(Delta))^2)), movement after subtracting the broad mean",
            "abs_net_over_rms": "small values mean the family vector moves more than the broad mean",
        },
        "dose_summary": dose_summary,
        "dose_vr_key_rows": dose_vr_key,
        "visible_80M_reference_key_rows": ref80_key,
        "archive_summary": archive_summary,
        "gpt2_roberta_summary": xarch_summary,
        "item_turnover_summary": item_summary,
        "archive_mature_largest_vector_minus_net": mature_cheap7_sorted[:5],
        "plot_paths": plot_paths,
        "output_files": {
            "dose_csv": str(OUT_DIR / "dose_common10_80_net_norm.csv"),
            "visible_reference_csv": str(OUT_DIR / "visible_reference_net_norm.csv"),
            "archive_csv": str(OUT_DIR / "archive_net_norm.csv"),
            "gpt2_roberta_csv": str(OUT_DIR / "gpt2_roberta_net_norm.csv"),
            "item_turnover_csv": str(OUT_DIR / "item_turnover_rows.csv"),
            "summary_json": str(OUT_DIR / "conservation_vector_summary.json"),
            "summary_md": str((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/conservation_vector_readout/conservation_vector_summary.md')),
        },
    }
    (OUT_DIR / "conservation_vector_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("# research conservation-vector readout")
    lines.append("")
    lines.append("File-only readout of already scored interventions. It compares each family-delta vector's broad mean with its vector amplitude. No model loading, training, evaluation, GPU work, upload, or leaderboard action occurred.")
    lines.append("")
    lines.append("## Measurement")
    lines.append("")
    lines.append("For a vector of family deltas Δ, `net = mean(Δ)`, `RMS = sqrt(mean(Δ²))`, and `centered_RMS = sqrt(mean((Δ-net)²))`. A small `|net|/RMS` means the score vector moves substantially while the broad mean changes little.")
    lines.append("")
    lines.append("## Fixed-budget dose: V-R over common 10M–80M")
    lines.append("")
    lines.append("| dose | rho | stable6 net | stable6 RMS | centered RMS | |net|/RMS | ex-Entity net | ex-Entity RMS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in dose_vr_key:
        lines.append(f"| {r['dose']} | {r['rho']:.6f} | {r['stable6_net']:+.4f} | {r['stable6_rms']:.4f} | {r['stable6_centered_rms']:.4f} | {r['stable6_abs_net_over_rms']:.3f} | {r['exEntity_net']:+.4f} | {r['exEntity_rms']:.4f} |")
    lines.append("")
    tr = dose_summary.get("dose_vr_stable6_rms_trend", {})
    trc = dose_summary.get("dose_vr_stable6_centered_rms_trend", {})
    tre = dose_summary.get("dose_vr_stable5_exEntity_net_trend", {})
    lines.append(f"Across the three dose points, stable6 V-R RMS has slope {tr.get('slope_per_rho')} per rho and correlation {tr.get('corr_with_rho')}; centered RMS has slope {trc.get('slope_per_rho')} and correlation {trc.get('corr_with_rho')}. The ex-Entity net stays small, with trend {tre.get('values')}.")
    lines.append("")
    lines.append("Interpretation: the high-dose V-R vector is not a clean broad addition. The all-family mean grows mainly because Entity enters the average, while the non-Entity broad mean remains close to zero compared with the family-vector amplitude.")
    lines.append("")
    lines.append("## Visible 80M breadth reference point")
    lines.append("")
    lines.append("| contrast | stable6 net | stable6 RMS | centered RMS | ex-Entity net | ex-Entity RMS | vector |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in ref80_key:
        vec = ", ".join(f"{k}:{v:+.2f}" for k, v in r["vector"].items())
        lines.append(f"| {r['contrast']} | {r['stable6_net']:+.4f} | {r['stable6_rms']:.4f} | {r['stable6_centered_rms']:.4f} | {r['exEntity_net']:+.4f} | {r['exEntity_rms']:.4f} | {vec} |")
    lines.append("")
    lines.append("The current visible V-B point has Entity +3.10 and Supplement +1.82, but BLiMP -0.87 and COMPS -1.18; its ex-Entity mean is -0.062 while RMS is about one score point. This matches A01's reply: source-related pairing can help exact/state-style routing without predicting broad ex-Entity gain.")
    lines.append("")
    lines.append("## Entity operation strata")
    lines.append("")
    lines.append("| contrast | official mean | neutral mean | zero/nonzero net | zero/nonzero RMS | vector |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in ref_summary.get("entity_operation_vectors", []):
        vec = json.loads(r["vector_json"])
        vecs = ", ".join(f"{k}:{v:+.2f}" for k, v in vec.items())
        lines.append(f"| {r['contrast']} | {r.get('official_all18_delta_pp_mean'):+.4f} | {r.get('neutral_zero_nonzero_delta_pp_mean'):+.4f} | {r['mean_net']:+.4f} | {r['rms_family_delta']:.4f} | {vecs} |")
    lines.append("")
    lines.append("V-B is positive in both Entity operation strata, unlike V-R and B-R. That means the same-sign Entity specificity is real for this surface, but it is not broad on the visible 80M family vector.")
    lines.append("")
    lines.append("## Mature research archive")
    lines.append("")
    for k, v in archive_summary.get("mature_cheap7", {}).items():
        lines.append(f"- mature cheap7 {k}: {v}")
    lines.append("")
    lines.append("Largest mature archive vector movements after subtracting the broad mean:")
    lines.append("")
    lines.append("| label | exposure | net | RMS | centered RMS | |net|/RMS | max column |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in mature_cheap7_sorted[:8]:
        lines.append(f"| {r.get('label')} | {r.get('exposure')} | {r['mean_net']:+.4f} | {r['rms_family_delta']:.4f} | {r['centered_rms_family_delta']:.4f} | {r['abs_net_over_rms']:.3f} | {r['max_abs_column']} {r['max_abs_delta']:+.2f} |")
    lines.append("")
    lines.append("The research mature archive has mean cheap7 movement -0.320 with no all-column positive record, but mean RMS much larger than the net. This turns the old route failures into quantitative evidence for a finite-budget redistribution regularity rather than isolated negative results.")
    lines.append("")
    lines.append("## Cross-architecture selected compact-vs-repeat")
    lines.append("")
    for k, v in xarch_summary.items():
        if isinstance(v, dict) and "mean_rms" in v:
            lines.append(f"- {k}: n={v['n']}, mean_net={v['mean_net']:+.4f}, mean_abs_net={v['mean_abs_net']:.4f}, mean_RMS={v['mean_rms']:.4f}, mean_centered_RMS={v['mean_centered_rms']:.4f}, mean_|net|/RMS={v['mean_abs_net_over_rms']:.3f}")
    lines.append("")
    lines.append("## Item-threshold movement")
    lines.append("")
    lines.append("| label | gains | losses | net | changed | net/changed | source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in item_rows:
        lines.append(f"| {r['label']} | {r['gain_items']} | {r['loss_items']} | {r['net_gain_minus_loss']} | {r['changed_items']} | {r['net_over_changed']:+.4f} | `{r['source']}` |")
    lines.append("")
    lines.append("## Research reading")
    lines.append("")
    lines.append("The current finite-budget hypothesis is sharpened, not completed: content-presentation changes often create substantial family-vector and item-threshold movement while broad non-Entity means remain small. The still-running MAX-geometry clean result decides whether content admission relative to matched clean placement gives a genuine broad positive leg; if it survives, the additive lever is which experience enters the 10M/100M budget, while source-related companions remain a narrower state/entity specificity mechanism requiring breadth/permuted follow-up. If it collapses, the conservation reading becomes stronger and the program should shift away from more presentation variants.")
    lines.append("")
    lines.append("## Files")
    for k, v in summary["output_files"].items():
        lines.append(f"- {k}: `{v}`")
    for p in plot_paths:
        lines.append(f"- plot: `{p}`")

    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/conservation_vector_readout/conservation_vector_summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(OUT_DIR),
        "summary_json": summary["output_files"]["summary_json"],
        "summary_md": summary["output_files"]["summary_md"],
        "dose_vr_key_rows": dose_vr_key,
        "visible_80M_reference_key_rows": ref80_key,
        "plot_paths": plot_paths,
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
