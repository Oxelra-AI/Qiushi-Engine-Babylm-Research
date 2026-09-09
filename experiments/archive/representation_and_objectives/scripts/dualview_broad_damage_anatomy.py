#!/usr/bin/env python3
"""research: CPU-only anatomy of existing dual-view broad damage and hard-surface repair.

This script does not train or score any model. It parses existing official-compatible
20M dual-view evaluation reports and existing research hard-surface CSVs to preserve
what a future broad-preserving coupled variant must protect if the pending matched
coupled-shuffled control confirms true-correspondence specificity.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
WS1 = _public_path('experiments/archive/representation_and_objectives')
WS2 = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/representation_and_objectives/data/dualview_broad_damage_anatomy')
NOTE = _public_path('research/notes/representation_and_objectives/dualview_broad_damage_anatomy.md')

BROAD_PAYLOADS = {
    "mlm_only_20M": _public_path('experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json'),
    "sep_sparse20_aligned_20M": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_aligned_20m_eval/per_target/sep_sparse20_aligned_20M.json'),
    "sep_sparse20_shuffled_20M": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_shuffled_20m_eval/per_target/sep_sparse20_shuffled_20M.json'),
    "coupled_sparse20_aligned_20M": _public_path('experiments/archive/frontier_consolidation/data/coupled_sparse20_aligned_20m_eval/per_target/coupled_sparse20_aligned_20M.json'),
}

EWOK_ROOTS = {
    "mlm_only_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset/ewok_stable_subset/mlm_only_20M'),
    "sep_sparse20_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset/ewok_stable_subset/sep_sparse20_aligned_20M'),
    "sep_sparse20_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining/ewok_stable_subset/sep_sparse20_shuffled_20M'),
    "coupled_sparse20_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining/ewok_stable_subset/coupled_sparse20_aligned_20M'),
}
GPIQA_ROW_DIR = _public_path('experiments/archive/representation_and_objectives/data/detached_private_hard_surface_readout/globalpiqa/rows')
GPIQA_HARD_JSON = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json')
CONSOLIDATION = _public_path('experiments/archive/representation_and_objectives/data/endpoint_and_dualview_consolidation/endpoint_and_dualview_consolidation.json')

SCORE_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def parse_report(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "sections": {}}
    if not path.exists():
        return out
    cur: str | None = None
    pending_average = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            cur = line.replace("###", "", 1).strip()
            out["sections"].setdefault(cur, {})
            pending_average = (cur == "AVERAGE ACCURACY")
            continue
        if pending_average and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", line):
            out["sections"].setdefault("AVERAGE ACCURACY", {})["average"] = float(line)
            pending_average = False
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            val = fnum(v)
            if val is None:
                continue
            section = cur or "UNSECTIONED"
            out["sections"].setdefault(section, {})[k.strip()] = val
    return out


def extract_scores(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    ov = payload.get("official_overall")
    if isinstance(ov, dict) and isinstance(ov.get("scores"), dict):
        for k, v in ov["scores"].items():
            fv = fnum(v)
            if fv is not None:
                scores[k] = fv
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {}
    for name, rec in tasks.items():
        if not isinstance(rec, dict):
            continue
        fv = fnum(rec.get("score"))
        if fv is not None:
            scores[name] = fv
        if name == "Reading" and isinstance(rec.get("scores"), dict):
            fv2 = fnum(rec["scores"].get("Reading"))
            if fv2 is not None:
                scores["Reading"] = fv2
    if "GlobalPIQA" not in scores and fnum(scores.get("GlobalPIQA_parallel")) is not None and fnum(scores.get("GlobalPIQA_nonparallel")) is not None:
        scores["GlobalPIQA"] = (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2.0
    if all(k in scores for k in SCORE_COLS):
        scores["cheap7"] = sum(scores[k] for k in SCORE_COLS) / 7.0
    return scores


def load_payloads() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in BROAD_PAYLOADS.items():
        p = read_json(path)
        tasks = p.get("tasks") if isinstance(p.get("tasks"), dict) else {}
        reports: dict[str, Any] = {}
        for task, rec in tasks.items():
            if not isinstance(rec, dict):
                continue
            rpath = rec.get("report")
            if isinstance(rpath, str):
                reports[task] = parse_report(ROOT / rpath)
        out[name] = {
            "payload_path": rel(path),
            "run_dir": p.get("run_dir"),
            "model_path": p.get("model_path"),
            "scores": extract_scores(p),
            "reports": reports,
        }
    return out


def numeric_leaf_deltas(base_report: dict[str, Any], cand_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bsecs = base_report.get("sections", {}) if isinstance(base_report.get("sections"), dict) else {}
    csecs = cand_report.get("sections", {}) if isinstance(cand_report.get("sections"), dict) else {}
    for sec in sorted(set(bsecs) | set(csecs)):
        bm = bsecs.get(sec, {}) if isinstance(bsecs.get(sec), dict) else {}
        cm = csecs.get(sec, {}) if isinstance(csecs.get(sec), dict) else {}
        for key in sorted(set(bm) & set(cm)):
            bv, cv = fnum(bm.get(key)), fnum(cm.get(key))
            if bv is None or cv is None:
                continue
            rows.append({"section": sec, "metric": key, "base": bv, "candidate": cv, "delta": cv - bv})
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def keyed_csv(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(r[key]): r for r in read_csv_rows(path)}


def group_deltas(base_rows: dict[str, dict[str, Any]], cand_rows: dict[str, dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    metrics = ["accuracy", "stable_failure", "stable_failure_frac_all", "stable_failure_frac_wrong", "wrong", "interaction_sum_median"]
    out: list[dict[str, Any]] = []
    for k in sorted(set(base_rows) & set(cand_rows)):
        b, c = base_rows[k], cand_rows[k]
        row: dict[str, Any] = {group_key: k, "n": fnum(c.get("n")) or fnum(b.get("n"))}
        for m in metrics:
            bv, cv = fnum(b.get(m)), fnum(c.get(m))
            if bv is not None and cv is not None:
                row[f"{m}_base"] = bv
                row[f"{m}_cand"] = cv
                row[f"{m}_delta"] = cv - bv
        out.append(row)
    # default: strongest stable-failure repair first (negative delta), then largest accuracy gains
    return sorted(out, key=lambda r: (r.get("stable_failure_delta", 0.0), -r.get("accuracy_delta", 0.0)))


def load_ewok_groups() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, root in EWOK_ROOTS.items():
        out[name] = {
            "by_domain": keyed_csv(root / "ewok_stable_subset_by_domain.csv", "domain"),
            "by_context_diff": keyed_csv(root / "ewok_stable_subset_by_context_diff.csv", "ContextDiff"),
            "summary_path": rel(root / "ewok_stable_subset_summary.json"),
        }
    return out


def load_hard_ids() -> set[str]:
    d = read_json(GPIQA_HARD_JSON)
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    return {r["example_id"] for r in rows if r.get("n_ok") == 0}


def bool_from_csv(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_gpiqa_rows(target: str) -> dict[str, dict[str, Any]]:
    path = GPIQA_ROW_DIR / f"{target}_parallel_rows.csv"
    rows = read_csv_rows(path)
    return {r["example_id"]: r for r in rows}


def gpiqa_pair_transition(base_rows: dict[str, dict[str, Any]], cand_rows: dict[str, dict[str, Any]], hard_ids: set[str]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for eid in sorted(hard_ids):
        b = base_rows.get(eid); c = cand_rows.get(eid)
        if not b or not c:
            continue
        br = int(float(b["correct_rank"])); cr = int(float(c["correct_rank"]))
        bm = float(b["top_minus_correct"]); cm = float(c["top_minus_correct"])
        bc = bool_from_csv(b["correct"]); cc = bool_from_csv(c["correct"])
        details.append({
            "example_id": eid,
            "base_correct": bc,
            "cand_correct": cc,
            "base_rank": br,
            "cand_rank": cr,
            "rank_delta": cr - br,
            "base_top_minus_correct": bm,
            "cand_top_minus_correct": cm,
            "margin_delta": cm - bm,
            "prompt": b.get("prompt"),
            "completions": b.get("completions"),
        })
    improved_rank = sum(1 for r in details if r["rank_delta"] < 0)
    worsened_rank = sum(1 for r in details if r["rank_delta"] > 0)
    improved_margin = sum(1 for r in details if r["margin_delta"] < -0.05)
    worsened_margin = sum(1 for r in details if r["margin_delta"] > 0.05)
    flipped_to_correct = sum(1 for r in details if (not r["base_correct"]) and r["cand_correct"])
    lost_correct = sum(1 for r in details if r["base_correct"] and (not r["cand_correct"]))
    n = len(details)
    mean_margin_delta = sum(r["margin_delta"] for r in details) / n if n else None
    mean_rank_delta = sum(r["rank_delta"] for r in details) / n if n else None
    return {
        "n": n,
        "improved_rank_count": improved_rank,
        "worsened_rank_count": worsened_rank,
        "same_rank_count": n - improved_rank - worsened_rank,
        "improved_margin_lt_minus_0p05_count": improved_margin,
        "worsened_margin_gt_0p05_count": worsened_margin,
        "flipped_to_correct": flipped_to_correct,
        "lost_correct": lost_correct,
        "mean_rank_delta": mean_rank_delta,
        "mean_margin_delta": mean_margin_delta,
        "top_margin_improvements": sorted(details, key=lambda r: r["margin_delta"])[:10],
        "top_margin_losses": sorted(details, key=lambda r: -r["margin_delta"])[:10],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads()
    mlm = payloads["mlm_only_20M"]

    score_deltas: dict[str, Any] = {}
    report_deltas: dict[str, Any] = {}
    broad_loss_tables: dict[str, Any] = {}
    broad_gain_tables: dict[str, Any] = {}
    for cand_name, cand in payloads.items():
        if cand_name == "mlm_only_20M":
            continue
        sd: dict[str, float] = {}
        for k in sorted(set(mlm["scores"]) & set(cand["scores"])):
            sd[k] = cand["scores"][k] - mlm["scores"][k]
        score_deltas[f"{cand_name}_minus_mlm_only"] = sd
        task_rows: list[dict[str, Any]] = []
        for task in sorted(set(mlm["reports"]) & set(cand["reports"])):
            for row in numeric_leaf_deltas(mlm["reports"][task], cand["reports"][task]):
                row["task"] = task
                task_rows.append(row)
        report_deltas[f"{cand_name}_minus_mlm_only"] = task_rows
        broad_rows = [r for r in task_rows if not str(r["task"]).startswith("GlobalPIQA") and r["section"] != "AVERAGE ACCURACY"]
        broad_loss_tables[f"{cand_name}_minus_mlm_only"] = sorted(broad_rows, key=lambda r: r["delta"])[:25]
        broad_gain_tables[f"{cand_name}_minus_mlm_only"] = sorted(broad_rows, key=lambda r: -r["delta"])[:25]

    ew = load_ewok_groups()
    ewok_group_deltas: dict[str, Any] = {}
    for cand_name in ["sep_sparse20_aligned_20M", "sep_sparse20_shuffled_20M", "coupled_sparse20_aligned_20M"]:
        ewok_group_deltas[f"{cand_name}_minus_mlm_only"] = {
            "by_domain": group_deltas(ew["mlm_only_20M"]["by_domain"], ew[cand_name]["by_domain"], "domain"),
            "by_context_diff": group_deltas(ew["mlm_only_20M"]["by_context_diff"], ew[cand_name]["by_context_diff"], "ContextDiff"),
        }

    hard_ids = load_hard_ids()
    grows = {name: load_gpiqa_rows(name) for name in EWOK_ROOTS.keys()}
    gpiqa_transitions = {
        f"{cand_name}_minus_mlm_only": gpiqa_pair_transition(grows["mlm_only_20M"], grows[cand_name], hard_ids)
        for cand_name in ["sep_sparse20_aligned_20M", "sep_sparse20_shuffled_20M", "coupled_sparse20_aligned_20M"]
    }

    consolidation = read_json(CONSOLIDATION) if CONSOLIDATION.exists() else None
    out = {
        "status": "PASS",
        "created_utc": now(),
        "boundary": "CPU-only parsing of existing reports and research row CSVs; no model scoring/training; chck_82M untouched",
        "payloads": {k: {"payload_path": v["payload_path"], "run_dir": v["run_dir"], "model_path": v["model_path"], "scores": v["scores"]} for k, v in payloads.items()},
        "score_deltas_vs_mlm_only": score_deltas,
        "report_deltas_vs_mlm_only": report_deltas,
        "top_broad_losses_vs_mlm_only": broad_loss_tables,
        "top_broad_gains_vs_mlm_only": broad_gain_tables,
        "ewok_hard_subset_group_deltas_vs_mlm_only": ewok_group_deltas,
        "globalpiqa_hard52_rank_margin_transitions_vs_mlm_only": gpiqa_transitions,
        "consolidation_source": rel(CONSOLIDATION),
        "source_files": {
            "payload_jsons": {k: rel(v) for k, v in BROAD_PAYLOADS.items()},
            "ewok_roots": {k: rel(v) for k, v in EWOK_ROOTS.items()},
            "globalpiqa_rows": rel(GPIQA_ROW_DIR),
            "hard_ids_source": rel(GPIQA_HARD_JSON),
        },
        "interpretation": [
            "Coupled aligned is the only existing arm with a large EWoK hard-row repair, but its broad losses are not uniform: strongest broad damage is BLiMP morphology/agreement/quantifier families plus Supplement turn-taking and QA congruence, with Reading and COMPS also down.",
            "The hard EWoK repair is concentrated in the largest load-bearing strata: agent-properties, physical-relations, variable swap, physical-interactions, social-relations, and spatial-relations. This is the surface a broad-preserving coupled variant must keep if the pending shuffled control confirms true-correspondence specificity.",
            "Separated aligned preserves broad score but fails the main hard EWoK surface; it is useful as a broad-preservation clue, not as the mechanism successor.",
            "This analysis deliberately does not decide true-correspondence specificity; that depends on the pending matched coupled_sparse20_shuffled_20M control read by the upgraded research script.",
        ],
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/dualview_broad_damage_anatomy/dualview_broad_damage_anatomy.json')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any, nd: int = 4) -> str:
        return "NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines: list[str] = []
    lines.append("# research dual-view broad-damage and hard-surface anatomy")
    lines.append("")
    lines.append(f"Status: **{out['status']}**")
    lines.append("")
    lines.append("CPU-only parse of existing 20M dual-view reports and research row CSVs. No training, no model scoring, and `chck_82M` untouched.")
    lines.append("")
    lines.append("## Broad score deltas vs `mlm_only_20M`")
    lines.append("| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, sd in score_deltas.items():
        lines.append("| " + label.replace("_minus_mlm_only", "") + " | " + " | ".join(fmt(sd.get(k), 4) for k in ["cheap7", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]) + " |")
    lines.append("")
    ca = "coupled_sparse20_aligned_20M_minus_mlm_only"
    lines.append("## Coupled aligned broad damage/gains: largest report-level deltas")
    lines.append("Losses (excluding GlobalPIQA item IDs):")
    for r in broad_loss_tables.get(ca, [])[:15]:
        lines.append(f"- {r['task']} / {r['section']} / {r['metric']}: {fmt(r['base'],2)} -> {fmt(r['candidate'],2)} (delta {fmt(r['delta'],2)})")
    lines.append("")
    lines.append("Gains (excluding GlobalPIQA item IDs):")
    for r in broad_gain_tables.get(ca, [])[:12]:
        lines.append(f"- {r['task']} / {r['section']} / {r['metric']}: {fmt(r['base'],2)} -> {fmt(r['candidate'],2)} (delta {fmt(r['delta'],2)})")
    lines.append("")
    lines.append("## Coupled aligned EWoK hard-subset repair by domain")
    lines.append("Negative stable-failure delta means fewer stable reversals than `mlm_only_20M`.")
    lines.append("| domain | n | acc_delta | stable_failure_delta | median_interaction_delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in ewok_group_deltas[ca]["by_domain"]:
        lines.append(f"| {r['domain']} | {fmt(r.get('n'),0)} | {fmt(r.get('accuracy_delta'),4)} | {fmt(r.get('stable_failure_delta'),0)} | {fmt(r.get('interaction_sum_median_delta'),4)} |")
    lines.append("")
    lines.append("## Coupled aligned EWoK hard-subset repair by ContextDiff")
    lines.append("| ContextDiff | n | acc_delta | stable_failure_delta | median_interaction_delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in ewok_group_deltas[ca]["by_context_diff"]:
        lines.append(f"| {r['ContextDiff']} | {fmt(r.get('n'),0)} | {fmt(r.get('accuracy_delta'),4)} | {fmt(r.get('stable_failure_delta'),0)} | {fmt(r.get('interaction_sum_median_delta'),4)} |")
    lines.append("")
    lines.append("## GlobalPIQA hard52 rank/margin transitions vs `mlm_only_20M`")
    lines.append("| arm | n | improved_rank | worsened_rank | margin_improved | margin_worsened | flips_to_correct | lost_correct | mean_rank_delta | mean_margin_delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, rec in gpiqa_transitions.items():
        lines.append(f"| {label.replace('_minus_mlm_only','')} | {rec['n']} | {rec['improved_rank_count']} | {rec['worsened_rank_count']} | {rec['improved_margin_lt_minus_0p05_count']} | {rec['worsened_margin_gt_0p05_count']} | {rec['flipped_to_correct']} | {rec['lost_correct']} | {fmt(rec['mean_rank_delta'],4)} | {fmt(rec['mean_margin_delta'],4)} |")
    lines.append("")
    lines.append("## Scientific reading")
    for item in out["interpretation"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": rel(out_json), "note": rel(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
