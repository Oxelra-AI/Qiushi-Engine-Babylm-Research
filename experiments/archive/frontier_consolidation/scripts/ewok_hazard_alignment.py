#!/usr/bin/env python3
"""research: align compact-view semantic hazard burden with official EWoK DiD.

This is a CPU-only analysis over the already-selected compact_view_reinvest
packet rows. It asks whether domains with negative current-official EWoK
seed-treatment interactions are directly enriched for the surface semantic hazards
used by the proposed force repair. If not, a repaired-corpus training run should
not be authorized just because the repair is mechanically refillable.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import math
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
FRONTIER_SCRIPT = STUDY / "scripts/semantic_repair_frontier.py"
DID_PATH = STUDY / "data/official_ewok_2x2_domain_did/official_ewok_2x2_domain_did.json"
OUT_DIR = STUDY / "data/ewok_hazard_alignment"

spec = importlib.util.spec_from_file_location("frontier", FRONTIER_SCRIPT)
frontier = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(frontier)  # type: ignore[arg-type]

FORCE_FLAGS = list(frontier.FORCE_FLAGS)
ALL_FLAGS = FORCE_FLAGS + [frontier.PRONOUN_FLAG] + list(frontier.SURFACE_FLAGS)

# Map heuristic compact-row domains onto EWoK relation domains. This is deliberately
# many-to-one and weak; the script also reports the raw compact primary-domain view.
EWOK_TO_COMPACT_DOMAINS = {
    "material-dynamics": ["science_physical", "causal_relational"],
    "material-properties": ["science_physical"],
    "physical-dynamics": ["science_physical", "causal_relational"],
    "physical-interactions": ["science_physical", "causal_relational"],
    "physical-relations": ["science_physical", "causal_relational"],
    "spatial-relations": ["geography_places", "science_physical"],
    "quantitative-properties": ["quant_numeric"],
    "social-interactions": ["institutions_society", "people_history", "causal_relational"],
    "social-relations": ["institutions_society", "people_history"],
    "social-properties": ["institutions_society", "people_history"],
    "agent-properties": ["institutions_society", "people_history", "no_domain"],
}


def pair_words(r: dict[str, Any]) -> int:
    return frontier.pair_words(r)


def primary_domain(r: dict[str, Any]) -> str:
    return frontier.primary_domain(r)


def all_domains(r: dict[str, Any]) -> list[str]:
    return frontier.all_domains(r)


def feature_flags(r: dict[str, Any]) -> dict[str, bool]:
    return frontier.feature_flags(r)


def safe_mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs); my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(vals: list[float]) -> list[float]:
    # Average ranks for ties.
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": len(rows),
        "pair_words": sum(pair_words(r) for r in rows),
    }
    if not rows:
        for fl in ALL_FLAGS:
            out[f"{fl}_row_fraction"] = None
            out[f"{fl}_pair_word_fraction"] = None
        out["force_any_row_fraction"] = None
        out["force_any_pair_word_fraction"] = None
        out["pronoun_or_force_row_fraction"] = None
        out["pronoun_or_force_pair_word_fraction"] = None
        out["mean_content_recall"] = None
        out["mean_length_ratio"] = None
        return out
    flags_list = [feature_flags(r) for r in rows]
    total_pw = out["pair_words"]
    for fl in ALL_FLAGS:
        hit_rows = [r for r, f in zip(rows, flags_list) if f.get(fl, False)]
        out[f"{fl}_rows"] = len(hit_rows)
        out[f"{fl}_row_fraction"] = len(hit_rows) / len(rows)
        out[f"{fl}_pair_words"] = sum(pair_words(r) for r in hit_rows)
        out[f"{fl}_pair_word_fraction"] = (out[f"{fl}_pair_words"] / total_pw) if total_pw else None
    force_rows = [r for r, f in zip(rows, flags_list) if any(f.get(fl, False) for fl in FORCE_FLAGS)]
    pron_force_rows = [r for r, f in zip(rows, flags_list) if any(f.get(fl, False) for fl in FORCE_FLAGS) or f.get(frontier.PRONOUN_FLAG, False)]
    out["force_any_rows"] = len(force_rows)
    out["force_any_row_fraction"] = len(force_rows) / len(rows)
    out["force_any_pair_words"] = sum(pair_words(r) for r in force_rows)
    out["force_any_pair_word_fraction"] = out["force_any_pair_words"] / total_pw if total_pw else None
    out["pronoun_or_force_rows"] = len(pron_force_rows)
    out["pronoun_or_force_row_fraction"] = len(pron_force_rows) / len(rows)
    out["pronoun_or_force_pair_words"] = sum(pair_words(r) for r in pron_force_rows)
    out["pronoun_or_force_pair_word_fraction"] = out["pronoun_or_force_pair_words"] / total_pw if total_pw else None
    out["mean_content_recall"] = safe_mean([float(r.get("content_recall", 0.0)) for r in rows])
    out["mean_length_ratio"] = safe_mean([float(r.get("length_ratio", r.get("rewrite_words", 0) / max(1, r.get("source_words", 1)))) for r in rows])
    return out


def weighted_log_odds(k: int, n: int, k0: int, n0: int) -> float:
    # Smoothed log odds of a flag within subset relative to complement.
    return math.log((k + 0.5) / (n - k + 0.5)) - math.log((k0 + 0.5) / (n0 - k0 + 0.5))


def hazard_enrichment(subset: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    subset_ids = {id(r) for r in subset}
    comp = [r for r in all_rows if id(r) not in subset_ids]
    sub_flags = [feature_flags(r) for r in subset]
    comp_flags = [feature_flags(r) for r in comp]
    out = {}
    for fl in ALL_FLAGS + ["force_any", "pronoun_or_force"]:
        if fl == "force_any":
            ks = sum(1 for f in sub_flags if any(f.get(x, False) for x in FORCE_FLAGS))
            kc = sum(1 for f in comp_flags if any(f.get(x, False) for x in FORCE_FLAGS))
        elif fl == "pronoun_or_force":
            ks = sum(1 for f in sub_flags if any(f.get(x, False) for x in FORCE_FLAGS) or f.get(frontier.PRONOUN_FLAG, False))
            kc = sum(1 for f in comp_flags if any(f.get(x, False) for x in FORCE_FLAGS) or f.get(frontier.PRONOUN_FLAG, False))
        else:
            ks = sum(1 for f in sub_flags if f.get(fl, False))
            kc = sum(1 for f in comp_flags if f.get(fl, False))
        ns = len(subset); nc = len(comp)
        fs = ks / ns if ns else None
        fc = kc / nc if nc else None
        out[fl] = {
            "subset_rows": ns,
            "subset_flag_rows": ks,
            "subset_fraction": fs,
            "complement_rows": nc,
            "complement_flag_rows": kc,
            "complement_fraction": fc,
            "fraction_delta": (fs - fc) if fs is not None and fc is not None else None,
            "smoothed_log_odds_vs_complement": weighted_log_odds(ks, ns, kc, nc) if ns and nc else None,
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = frontier.load_jsonl(frontier.SELECTED_REINVEST)
    did = json.loads(DID_PATH.read_text(encoding="utf-8"))
    did_domains = {r["domain"]: r for r in did["domain_summary_sorted_by_DiD"]}

    # Raw compact primary-domain burden. This tells which coarse compact labels carry hazards.
    by_primary: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_multi: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in selected:
        by_primary[primary_domain(r)].append(r)
        for d in all_domains(r):
            by_multi[d].append(r)
    primary_summary = {d: {**summarize_rows(rows), "enrichment": hazard_enrichment(rows, selected)} for d, rows in sorted(by_primary.items())}
    multi_summary = {d: {**summarize_rows(rows), "enrichment": hazard_enrichment(rows, selected)} for d, rows in sorted(by_multi.items())}

    # EWoK domain aligned burden via the weak map above.
    ewok_rows: dict[str, list[dict[str, Any]]] = {}
    aligned_rows = []
    for ewok_domain, compact_domains in EWOK_TO_COMPACT_DOMAINS.items():
        subset = [r for r in selected if any(d in all_domains(r) for d in compact_domains)]
        ewok_rows[ewok_domain] = subset
        summ = summarize_rows(subset)
        enrich = hazard_enrichment(subset, selected)
        di = did_domains.get(ewok_domain, {})
        rec = {
            "ewok_domain": ewok_domain,
            "mapped_compact_domains": compact_domains,
            "official_EWoK_DiD": di.get("DiD_TE43122_minus_TE43022"),
            "TE43022": di.get("TE43022_reinvest_minus_clean"),
            "TE43122": di.get("TE43122_reinvest_minus_clean"),
            "rows": summ["rows"],
            "pair_words": summ["pair_words"],
            "force_any_row_fraction": summ["force_any_row_fraction"],
            "force_any_pair_word_fraction": summ["force_any_pair_word_fraction"],
            "pronoun_or_force_row_fraction": summ["pronoun_or_force_row_fraction"],
            "mean_content_recall": summ["mean_content_recall"],
            "mean_length_ratio": summ["mean_length_ratio"],
            "flag_row_fractions": {fl: summ.get(f"{fl}_row_fraction") for fl in ALL_FLAGS},
            "flag_pair_word_fractions": {fl: summ.get(f"{fl}_pair_word_fraction") for fl in ALL_FLAGS},
            "enrichment_force_any": enrich["force_any"],
            "enrichment_by_flag": {fl: enrich[fl] for fl in FORCE_FLAGS + [frontier.PRONOUN_FLAG]},
        }
        aligned_rows.append(rec)
    aligned_rows.sort(key=lambda r: (r["official_EWoK_DiD"] if r["official_EWoK_DiD"] is not None else 999))

    # Correlations across EWoK domains: more negative DiD should correspond to higher hazard if force repair is directly selected.
    corr_metrics = ["force_any_row_fraction", "pronoun_or_force_row_fraction", "mean_content_recall", "mean_length_ratio"]
    correlations = {}
    did_vals = [float(r["official_EWoK_DiD"]) for r in aligned_rows if r["official_EWoK_DiD"] is not None]
    for metric in corr_metrics:
        xs = []
        ys = []
        for r in aligned_rows:
            if r["official_EWoK_DiD"] is None or r.get(metric) is None:
                continue
            xs.append(float(r[metric]))
            ys.append(float(r["official_EWoK_DiD"]))
        correlations[metric + "_vs_DiD"] = {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys), "n": len(xs)}
    for fl in FORCE_FLAGS + [frontier.PRONOUN_FLAG]:
        xs = []
        ys = []
        for r in aligned_rows:
            v = r["flag_row_fractions"].get(fl)
            if r["official_EWoK_DiD"] is None or v is None:
                continue
            xs.append(float(v)); ys.append(float(r["official_EWoK_DiD"]))
        correlations[f"{fl}_row_fraction_vs_DiD"] = {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys), "n": len(xs)}

    # Compare named negative domains against the rest as a more robust small-N contrast.
    negative_domains = ["material-dynamics", "physical-dynamics", "spatial-relations", "physical-interactions"]
    negative_subset = []
    for d in negative_domains:
        negative_subset.extend(ewok_rows[d])
    # unique by key, not object id, because one row may satisfy multiple mapped domains.
    seen = set(); neg_unique = []
    for r in negative_subset:
        k = frontier.key_of(r)
        if k not in seen:
            seen.add(k); neg_unique.append(r)
    negative_enrichment = hazard_enrichment(neg_unique, selected)
    negative_summary = summarize_rows(neg_unique)

    result = {
        "status": "EWOK_HAZARD_ALIGNMENT",
        "purpose": "Test whether current-official EWoK negative treatment interactions align with compact-view semantic hazard flags before authorizing force-repair training.",
        "inputs": {"selected_reinvest_pairs": str(frontier.SELECTED_REINVEST), "domain_did": str(DID_PATH), "frontier_script": str(FRONTIER_SCRIPT)},
        "selected_total": summarize_rows(selected),
        "primary_compact_domain_summary": primary_summary,
        "multi_compact_domain_summary": multi_summary,
        "ewok_domain_alignment": aligned_rows,
        "correlations_across_ewok_domains": correlations,
        "negative_domain_group": {
            "domains": negative_domains,
            "mapped_unique_rows": negative_summary,
            "enrichment": negative_enrichment,
        },
        "scientific_read": {
            "directness": "This is heuristic: compact corpus domain tags are broad and do not prove that a particular EWoK item was learned from a particular row. It only tests whether the proposed repair's flags are enriched in the domains where official EWoK DiD is negative.",
            "use": "A strong positive alignment between hazard burden and negative DiD would support materializing a force-repair corpus; weak or opposite alignment would favor checkpoint/optimization/consolidation work or a different relation-domain data idea.",
        },
    }
    out_json = OUT_DIR / "ewok_hazard_alignment.json"
    out_md = OUT_DIR / "ewok_hazard_alignment.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — EWoK semantic-hazard alignment\n\n"]
    lines.append("CPU-only analysis over selected compact_view_reinvest packets. It tests whether the EWoK domains with negative current-official treatment interaction are enriched for the force/modal/attribution/negation/causal/coreference hazards used by the candidate repair.\n\n")
    total = result["selected_total"]
    lines.append(f"Selected block: {total['rows']} rows / {total['pair_words']} pair words; force-any row fraction {total['force_any_row_fraction']:.3f}; pronoun-or-force row fraction {total['pronoun_or_force_row_fraction']:.3f}.\n\n")
    lines.append("## EWoK-domain alignment (most negative official DiD first)\n")
    lines.append("| EWoK domain | DiD | TE43022 | TE43122 | rows mapped | force-any frac | pronoun+force frac | modal loss | new causal | lost neg | content recall |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in aligned_rows:
        ff = r["flag_row_fractions"]
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {r['ewok_domain']} | {fmt(r['official_EWoK_DiD'])} | {fmt(r['TE43022'])} | {fmt(r['TE43122'])} | {r['rows']} | "
            f"{fmt(r['force_any_row_fraction'])} | {fmt(r['pronoun_or_force_row_fraction'])} | {fmt(ff.get('lost_modal_or_hedge'))} | {fmt(ff.get('new_causal_marker_without_source'))} | {fmt(ff.get('lost_negation'))} | {fmt(r['mean_content_recall'])} |\n"
        )
    lines.append("\n## Correlations across EWoK domains\n")
    for k, v in correlations.items():
        lines.append(f"- {k}: pearson={v['pearson']}, spearman={v['spearman']}, n={v['n']}\n")
    lines.append("\n## Negative domain group\n")
    neg = result["negative_domain_group"]
    ns = neg["mapped_unique_rows"]
    lines.append(f"Domains {negative_domains}: unique mapped rows {ns['rows']} / pair words {ns['pair_words']}; force-any frac {ns['force_any_row_fraction']:.3f}; pronoun+force frac {ns['pronoun_or_force_row_fraction']:.3f}.\n")
    lines.append("Top enrichment deltas vs complement:\n")
    enrich_items = sorted(neg["enrichment"].items(), key=lambda kv: (kv[1]["fraction_delta"] if kv[1]["fraction_delta"] is not None else -999), reverse=True)
    for fl, rec in enrich_items[:8]:
        lines.append(f"- {fl}: subset {rec['subset_fraction']:.3f}, complement {rec['complement_fraction']:.3f}, delta {rec['fraction_delta']:+.3f}, log_odds {rec['smoothed_log_odds_vs_complement']:+.3f}\n")
    lines.append("\n## Scientific read\n")
    for v in result["scientific_read"].values():
        lines.append(f"- {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "total_force_any_fraction": total["force_any_row_fraction"],
        "negative_group_force_any_fraction": ns["force_any_row_fraction"],
        "correlations": correlations,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
