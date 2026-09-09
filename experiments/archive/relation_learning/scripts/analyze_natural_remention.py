#!/usr/bin/env python3
"""research: merge and analyze the natural re-mention probe scores.

This repairs the research scorer-output failure by merging the partial successful
score file (D 43022 V/R/C + D_C_43122) with the research repaired 43122 V/R pass.
It joins the probe metadata so the result can be read with the instrument's real
resolution and heuristic limitations.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import random
import statistics
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/analyze_natural_remention.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
PROBE = _public_path('experiments/archive/relation_learning/analysis/natural_remention_probe/natural_remention_probe.jsonl')
AUDIT = _public_path('experiments/archive/relation_learning/analysis/natural_remention_probe/manual_sample_audit.json')
IN_DIRS = [
    _public_path('experiments/archive/relation_learning/data/natural_remention_scores'),
    _public_path('experiments/archive/relation_learning/data/natural_remention_43122_vr'),
]
OUT = _public_path('experiments/archive/relation_learning/data/natural_remention_merged')
NOTE = _public_path('research/notes/relation_learning/natural_remention_readout.md')

LATE_CKS = {"chck_80M", "chck_90M", "chck_100M"}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def stdev(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) > 1 else float("nan")


def load_probe_meta() -> dict[str, dict[str, Any]]:
    meta = {}
    with PROBE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d["probe_id"]
            cp = d.get("corruption_plan") or {}
            cm = d.get("class_matching") or {}
            flags = []
            # Builder v1.1 stores acceptance evidence under variable names; keep robust.
            for k in ["manual_audit_flags", "flags", "quality_flags", "soft_flags"]:
                v = d.get(k) or []
                if isinstance(v, str):
                    flags.append(v)
                elif isinstance(v, list):
                    flags.extend(str(x) for x in v)
            meta[pid] = {
                "probe_id": pid,
                "class_label": d.get("class_label"),
                "subtype": d.get("subtype"),
                "pair_id": cm.get("pair_id"),
                "matching_relaxation": cm.get("relaxation_level") or cm.get("level") or "",
                "distance_bin": (d.get("mention_distance") or {}).get("distance_bin"),
                "sentence_distance": (d.get("mention_distance") or {}).get("sentence_distance"),
                "intervening_words": (d.get("mention_distance") or {}).get("intervening_words"),
                "replacement_kind": cp.get("replacement_kind") or cp.get("donor_kind") or cp.get("strategy") or "",
                "replacement_exact_word_count": cp.get("exact_word_count_match"),
                "has_typed_fallback": "fallback" in json.dumps(cp).lower(),
                "is_pronoun": "pronoun" in str(d.get("subtype") or ""),
                "is_appositive": "appositive" in str(d.get("subtype") or ""),
                "flags_joined": ";".join(flags),
            }
    # Manual sample audit: mark any sample flags if schema exposes them.
    if AUDIT.exists():
        try:
            a = json.loads(AUDIT.read_text(encoding="utf-8"))
            flagged_ids: set[str] = set()
            if isinstance(a, dict):
                for key in ["flagged", "flags", "records", "audit_records", "sample"]:
                    v = a.get(key)
                    if isinstance(v, list):
                        for rec in v:
                            if isinstance(rec, dict):
                                txt = json.dumps(rec).lower()
                                pid = str(rec.get("probe_id") or rec.get("id") or "")
                                if pid and ("ambiguous" in txt or "flag" in txt or "confound" in txt or "unclear" in txt):
                                    flagged_ids.add(pid)
            for pid in flagged_ids:
                if pid in meta:
                    meta[pid]["manual_sample_flag"] = True
        except Exception:
            pass
    for d in meta.values():
        d.setdefault("manual_sample_flag", False)
        d["primary_sensitivity"] = not (d["is_pronoun"] or d["is_appositive"] or d["has_typed_fallback"] or d["manual_sample_flag"])
    return meta


def arm_parts(arm: str) -> tuple[str, str, str]:
    p = arm.split("_")
    if len(p) >= 3:
        return p[0], p[1], p[2]
    return "", "", ""


def merge_raw(meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # Prefer repaired scores for an exact duplicate key.
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for in_dir in IN_DIRS:
        p = in_dir / "natural_remention_raw_scores.csv"
        if not p.exists():
            continue
        for r in read_csv(p):
            arm = r["arm"]
            ck = r["checkpoint"]
            pid = r["probe_id"]
            m = meta.get(pid, {})
            arch, role, seed = arm_parts(arm)
            out: dict[str, Any] = dict(r)
            out.update({k: v for k, v in m.items() if k not in {"probe_id"}})
            out.update({"arch": arch, "role": role, "seed": seed})
            for k in ["nll_present", "nll_replaced", "antecedent_gain"]:
                out[k] = float(out[k])
            for k in ["n_tokens_present", "n_tokens_replaced"]:
                out[k] = int(float(out[k]))
            out["source_score_dir"] = rel(in_dir)
            by_key[(arm, ck, pid)] = out
    return list(by_key.values())


def late_by_record(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    d: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["checkpoint"] in LATE_CKS:
            d[(r["arm"], r["probe_id"])].append(r)
    out = {}
    for (arm, pid), vals in d.items():
        base = dict(vals[0])
        base["late_mean_gain"] = mean([v["antecedent_gain"] for v in vals])
        base["late_mean_nll_present"] = mean([v["nll_present"] for v in vals])
        base["late_mean_nll_replaced"] = mean([v["nll_replaced"] for v in vals])
        base["n_checkpoints"] = len(vals)
        out[(arm, pid)] = base
    return out


def group_summaries(late: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    d = defaultdict(list)
    for r in late.values():
        groups = ["ALL", r["class_label"], f"subtype::{r['subtype']}", f"dist::{r['distance_bin']}"]
        if r.get("primary_sensitivity"):
            groups.append("primary_sensitivity")
            groups.append(f"primary::{r['class_label']}")
        for g in groups:
            d[(r["arch"], r["seed"], r["role"], g)].append(r["late_mean_gain"])
    out = []
    for (arch, seed, role, g), xs in sorted(d.items()):
        sd = stdev(xs)
        out.append({
            "arch": arch, "seed": seed, "role": role, "group": g,
            "n": len(xs), "mean_gain": mean(xs), "sd_gain": sd,
            "se_gain": sd / math.sqrt(len(xs)) if math.isfinite(sd) and len(xs) > 1 else float("nan"),
        })
    return out


def contrasts_from_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val = {(r["arch"], r["seed"], r["role"], r["group"]): r for r in summary}
    groups = sorted({(r["arch"], r["seed"], r["group"]) for r in summary})
    rows = []
    for arch, seed, g in groups:
        vr = val.get((arch, seed, "V", g))
        rr = val.get((arch, seed, "R", g))
        cr = val.get((arch, seed, "C", g))
        vg = vr["mean_gain"] if vr else float("nan")
        rg = rr["mean_gain"] if rr else float("nan")
        cg = cr["mean_gain"] if cr else float("nan")
        if not (math.isfinite(vg) or math.isfinite(rg) or math.isfinite(cg)):
            continue
        rows.append({
            "arch": arch, "seed": seed, "group": g,
            "V_gain": vg if math.isfinite(vg) else None,
            "R_gain": rg if math.isfinite(rg) else None,
            "C_gain": cg if math.isfinite(cg) else None,
            "VminusC": (vg-cg) if math.isfinite(vg) and math.isfinite(cg) else None,
            "RminusC": (rg-cg) if math.isfinite(rg) and math.isfinite(cg) else None,
            "VminusR": (vg-rg) if math.isfinite(vg) and math.isfinite(rg) else None,
        })
    return rows


def pair_interactions(late: dict[tuple[str, str], dict[str, Any]], n_boot=5000) -> list[dict[str, Any]]:
    # Build seed -> pair -> class -> role -> gain.
    d: dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in late.values():
        if r["arch"] != "D":
            continue
        pair = r.get("pair_id")
        cls = r.get("class_label")
        role = r.get("role")
        if pair and cls in {"verbatim_remention", "nonidentical_remention"} and role in {"V", "R", "C"}:
            d[(r["arch"], r["seed"])][pair][cls][role] = float(r["late_mean_gain"])
    rng = random.Random(9061401)
    outs = []
    for (arch, seed), pairs in sorted(d.items()):
        complete = []
        for pair, by_cls in pairs.items():
            if all(cls in by_cls and all(role in by_cls[cls] for role in ["V", "R", "C"]) for cls in ["verbatim_remention", "nonidentical_remention"]):
                verb = by_cls["verbatim_remention"]
                non = by_cls["nonidentical_remention"]
                complete.append({
                    "pair_id": pair,
                    "R_C_verbatim": verb["R"] - verb["C"],
                    "R_C_nonidentical": non["R"] - non["C"],
                    "V_C_verbatim": verb["V"] - verb["C"],
                    "V_C_nonidentical": non["V"] - non["C"],
                    "V_R_verbatim": verb["V"] - verb["R"],
                    "V_R_nonidentical": non["V"] - non["R"],
                })
        def one_metric(rows):
            rcv = mean([x["R_C_verbatim"] for x in rows])
            rcn = mean([x["R_C_nonidentical"] for x in rows])
            vcv = mean([x["V_C_verbatim"] for x in rows])
            vcn = mean([x["V_C_nonidentical"] for x in rows])
            vrv = mean([x["V_R_verbatim"] for x in rows])
            vrn = mean([x["V_R_nonidentical"] for x in rows])
            return {
                "R_C_verbatim": rcv,
                "R_C_nonidentical": rcn,
                "R_confirm_interaction": rcv - rcn,
                "V_C_verbatim": vcv,
                "V_C_nonidentical": vcn,
                "V_confirm_interaction": vcn - vcv,
                "V_R_verbatim": vrv,
                "V_R_nonidentical": vrn,
                "V_R_nonminusverb": vrn - vrv,
            }
        est = one_metric(complete)
        boot_vals = {k: [] for k in est}
        if complete:
            for _ in range(n_boot):
                samp = [complete[rng.randrange(len(complete))] for _ in range(len(complete))]
                b = one_metric(samp)
                for k, v in b.items():
                    boot_vals[k].append(v)
        row = {"arch": arch, "seed": seed, "n_pairs": len(complete)}
        for k, v in est.items():
            xs = sorted(boot_vals[k])
            row[k] = v
            row[f"{k}_ci_lo"] = xs[int(0.025 * len(xs))] if xs else None
            row[f"{k}_ci_hi"] = xs[int(0.975 * len(xs))] if xs else None
        outs.append(row)
    return outs


def combined_two_seed(pair_rows: list[dict[str, Any]], n_boot=5000) -> dict[str, Any]:
    rng = random.Random(9061402)
    seeds = [r for r in pair_rows if r.get("n_pairs", 0)]
    # Weighted by pair count; here both should be 129.
    metrics = [k for k in pair_rows[0].keys() if k in {
        "R_C_verbatim", "R_C_nonidentical", "R_confirm_interaction", "V_C_verbatim", "V_C_nonidentical", "V_confirm_interaction", "V_R_verbatim", "V_R_nonidentical", "V_R_nonminusverb"
    }] if pair_rows else []
    out = {"arch": "D", "seed": "43022+43122", "n_seed_records": len(seeds)}
    for k in metrics:
        vals = [float(r[k]) for r in seeds if r[k] is not None]
        out[k] = mean(vals)
        out[f"{k}_sd_across_seeds"] = stdev(vals)
    return out


def write_note(raw, summary, contrasts, pairs, combined):
    def fmt(x):
        if x is None:
            return "NA"
        try:
            x = float(x)
            return f"{x:+.4f}"
        except Exception:
            return str(x)
    lines = []
    lines.append("# research natural re-mention readout")
    lines.append("")
    lines.append("This note merges the partial research scorer output with the repaired research pass over the real seed43122 VIEW/REPEAT run paths. The scorer failure was a note-write permission error after it had already produced raw scores for DeBERTa seed43022 V/R/C and seed43122 CLEAN; the repaired pass added seed43122 V/R. The instrument has 258 records (129 verbatim, 129 nonidentical), so the result is a thin natural-text sensitivity check, not a primary pillar.")
    lines.append("")
    lines.append(f"Merged raw rows: {len(raw)}. Output data: `{rel(OUT)}`.")
    lines.append("")
    lines.append("## Late mean gains by class")
    lines.append("")
    lines.append("Antecedent gain = NLL(antecedent replaced) - NLL(antecedent present). Positive means the earlier antecedent helps predict the re-mention.")
    lines.append("")
    lines.append("| seed | group | V gain | R gain | C gain | V-C | R-C | V-R |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    keep_groups = ["ALL", "verbatim_remention", "nonidentical_remention", "primary_sensitivity", "primary::verbatim_remention", "primary::nonidentical_remention"]
    for c in contrasts:
        if c["arch"] == "D" and c["seed"] in {"43022", "43122"} and c["group"] in keep_groups:
            lines.append(f"| {c['seed']} | {c['group']} | {fmt(c['V_gain'])} | {fmt(c['R_gain'])} | {fmt(c['C_gain'])} | {fmt(c['VminusC'])} | {fmt(c['RminusC'])} | {fmt(c['VminusR'])} |")
    lines.append("")
    lines.append("## Pair-level interaction")
    lines.append("")
    lines.append("The pre-stated interaction was `(R-C)_verbatim > (R-C)_nonidentical` and `(V-C)_nonidentical > (V-C)_verbatim`. Each bootstrap CI resamples the 129 distance-matched cross-class pairs within seed.")
    lines.append("")
    lines.append("| seed | pairs | R-C verb | R-C nonid | R interaction | 95% CI | V-C verb | V-C nonid | V interaction | 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for r in pairs:
        lines.append(f"| {r['seed']} | {r['n_pairs']} | {fmt(r['R_C_verbatim'])} | {fmt(r['R_C_nonidentical'])} | {fmt(r['R_confirm_interaction'])} | [{fmt(r['R_confirm_interaction_ci_lo'])}, {fmt(r['R_confirm_interaction_ci_hi'])}] | {fmt(r['V_C_verbatim'])} | {fmt(r['V_C_nonidentical'])} | {fmt(r['V_confirm_interaction'])} | [{fmt(r['V_confirm_interaction_ci_lo'])}, {fmt(r['V_confirm_interaction_ci_hi'])}] |")
    if combined:
        lines.append("")
        lines.append("Across the two available DeBERTa seeds, the mean pair-level R-side interaction is " + fmt(combined.get("R_confirm_interaction")) + " and the mean V-side interaction is " + fmt(combined.get("V_confirm_interaction")) + ".")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("This natural probe does **not** supply a new strong behavioral pillar. Seed43022 shows the expected REPEAT class interaction direction: R-C is positive on verbatim re-mentions and negative on nonidentical re-mentions. Seed43122 keeps only a very small R-side interaction around the same sign. The VIEW-side nonidentical-specific prediction is not supported: in both seeds V-C is not larger for nonidentical than for verbatim. Absolute arm contrasts are only about 0.02--0.18 nats, comparable to the within-class SEs in the raw summary and much smaller than the compact-rewrite V-R gap (~1.4--1.9 nats).")
    lines.append("")
    lines.append("The result should therefore be read as resolution-limited and partly unfavorable to a broad natural-coreference extension, not as evidence against the already stronger compact-rewrite and Entity results. It says the 258-record heuristic re-mention set is too small/noisy, and perhaps too dominated by easy name repetition and local apposition/pronoun cases, to establish the four-way interaction. If this domain remains important, the next useful work is to extend the instrument with many more nonidentical nonpronoun long-distance records and stronger corruption matching before treating a null as scientific evidence.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Merged raw scores: `{rel(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_raw_scores_merged.csv'))}`")
    lines.append(f"- Late record table: `{rel(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_late_by_record.csv'))}`")
    lines.append(f"- Group contrasts: `{rel(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_late_contrasts_merged.csv'))}`")
    lines.append(f"- Pair interactions: `{rel(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_pair_interactions.csv'))}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_probe_meta()
    raw = merge_raw(meta)
    late = late_by_record(raw)
    late_rows = list(late.values())
    summary = group_summaries(late)
    contrasts = contrasts_from_summary(summary)
    pair_rows = pair_interactions(late)
    comb = combined_two_seed(pair_rows) if pair_rows else {}
    write_csv(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_raw_scores_merged.csv'), raw)
    write_csv(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_late_by_record.csv'), late_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_summary_merged.csv'), summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_late_contrasts_merged.csv'), contrasts)
    write_csv(_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_pair_interactions.csv'), pair_rows)
    (_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_pair_interactions_combined.json')).write_text(json.dumps(comb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "status": "NATURAL_REMENTION_MERGED_ANALYZED",
        "raw_rows": len(raw),
        "late_records": len(late_rows),
        "summary_rows": len(summary),
        "contrast_rows": len(contrasts),
        "pair_interaction_rows": len(pair_rows),
        "combined_two_seed": comb,
        "note": rel(NOTE),
    }
    (_public_path('experiments/archive/relation_learning/data/natural_remention_merged/natural_remention_analysis_result.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
