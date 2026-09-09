#!/usr/bin/env python3
"""research: feasibility audit for exact-loader-matched factorial training contrasts.

This script is CPU-only.  It reads the research target-level records and asks a narrow
question before any H100 spend: can the existing candidate view families support one
scientifically interpretable, exact-loader-matched training contrast?

It aggregates per-pair active/WWM/copy-zone target mass under the real DeBERTa row
loader, then evaluates:
  * fixed-word-multiset ordered-vs-scrambled controls (should be nearly exact);
  * compact vs sourcewide_onegap feasibility after same-pair subset selection;
  * sourcewide_onegap vs prefix_fluent feasibility for source-position spread.

The output is intended to constrain future route choice, not to authorize training by
itself.  Pending DeBERTa trajectory and triangle/fork evidence must still be read.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

VARIANTS = [
    "compact",
    "compact_scrambled",
    "prefix_fluent",
    "prefix_scrambled",
    "sourcewide_onegap",
    "sourcewide_onegap_scrambled",
    "best_contiguous_span",
]

COPY_ZONES = ["absent", "prefix_only", "tail_only", "both_prefix_tail"]


def root_dir() -> Path:
    # __file__ = <user-root>/experiments/archive/frontier_consolidation/scripts/this_file.py
    # parents[4] is the user root; parents[3] would be <user-root>/Sessions.
    return _public_path('.')


def safe_float(x: str) -> float:
    if x is None or x == "" or x.lower() == "nan":
        return float("nan")
    return float(x)


def quantiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    ys = sorted(xs)
    n = len(ys)
    def q(p: float) -> float:
        if n == 1:
            return float(ys[0])
        pos = (n - 1) * p
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(ys[lo])
        return float(ys[lo] * (hi - pos) + ys[hi] * (pos - lo))
    return {"min": q(0), "p05": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": q(1)}


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def pct(x: float) -> float:
    return 100.0 * float(x)


def load_pair_metrics(records_csv: Path, max_pairs: int | None = None) -> Dict[Tuple[str, str], dict]:
    """Aggregate research target records by (variant, pair_id)."""
    metrics: Dict[Tuple[str, str], dict] = {}
    # Per variant deterministic pair order for optional quick runs.
    seen_pairs_by_variant: Dict[str, set] = defaultdict(set)
    with records_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variant"]
            pid = row["pair_id"]
            if max_pairs is not None:
                # Keep first max_pairs per variant by first encounter.
                s = seen_pairs_by_variant[var]
                if pid not in s and len(s) >= max_pairs:
                    continue
                s.add(pid)
            key = (var, pid)
            m = metrics.get(key)
            if m is None:
                m = {
                    "variant": var,
                    "pair_id": pid,
                    "targets": 0,
                    "active_tokens": 0.0,
                    "wwm_mass": 0.0,
                    "full_visible": 0,
                    "counterpart_any_visible": 0,
                    "counterpart_full_visible": 0,
                    "whole_word_copy": 0,
                    "complete_bpe_copy": 0,
                    "seen_bpe_sum": 0.0,
                    "contentlike": 0,
                    "copy_zone_targets": defaultdict(int),
                    "copy_zone_active": defaultdict(float),
                    "lex_targets": defaultdict(int),
                    "source_match_targets": defaultdict(int),
                }
                metrics[key] = m
            m["targets"] += 1
            active = safe_float(row["view_active_tokens"])
            wwm = safe_float(row["wwm_token_mass"])
            m["active_tokens"] += 0.0 if math.isnan(active) else active
            m["wwm_mass"] += 0.0 if math.isnan(wwm) else wwm
            m["full_visible"] += 1 if row["view_full_visible"] in ("True", "true", "1", "1.0") else 0
            m["counterpart_any_visible"] += 1 if row["counterpart_any_visible"] in ("True", "true", "1", "1.0") else 0
            m["counterpart_full_visible"] += 1 if row["counterpart_full_visible"] in ("True", "true", "1", "1.0") else 0
            try:
                source_match_count = int(float(row.get("source_match_count", "0") or 0))
            except ValueError:
                source_match_count = 0
            m["whole_word_copy"] += 1 if source_match_count > 0 else 0
            m["complete_bpe_copy"] += 1 if row["complete_bpe_copy_any_source"] in ("True", "true", "1", "1.0") else 0
            seen = safe_float(row.get("frac_view_bpes_seen_in_source", "nan"))
            m["seen_bpe_sum"] += 0.0 if math.isnan(seen) else seen
            lex = row["lex_class"]
            copy_zone = row["copy_zone"]
            source_bin = row.get("source_match_bin", "")
            contentlike = lex in {"content", "capitalized_content", "number"}
            m["contentlike"] += 1 if contentlike else 0
            m["copy_zone_targets"][copy_zone] += 1
            m["copy_zone_active"][copy_zone] += 0.0 if math.isnan(active) else active
            m["lex_targets"][lex] += 1
            m["source_match_targets"][source_bin] += 1
    # Convert defaultdicts and fractions.
    for m in metrics.values():
        n = m["targets"] or 1
        for k in ["copy_zone_targets", "copy_zone_active", "lex_targets", "source_match_targets"]:
            m[k] = dict(m[k])
        m["full_visible_frac"] = m["full_visible"] / n
        m["counterpart_any_visible_frac"] = m["counterpart_any_visible"] / n
        m["counterpart_full_visible_frac"] = m["counterpart_full_visible"] / n
        m["whole_word_copy_frac"] = m["whole_word_copy"] / n
        m["complete_bpe_copy_frac"] = m["complete_bpe_copy"] / n
        m["seen_bpe_frac_mean"] = m["seen_bpe_sum"] / n
        m["contentlike_frac"] = m["contentlike"] / n
    return metrics


def aggregate_variant(metrics: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    for (var, _pid), m in metrics.items():
        a = agg.setdefault(var, {
            "pairs": 0, "targets": 0, "active_tokens": 0.0, "wwm_mass": 0.0,
            "full_visible": 0, "counterpart_any_visible": 0, "counterpart_full_visible": 0,
            "whole_word_copy": 0, "complete_bpe_copy": 0, "seen_bpe_sum_weighted": 0.0,
            "contentlike": 0, "copy_zone_targets": defaultdict(int), "copy_zone_active": defaultdict(float),
        })
        a["pairs"] += 1
        n = m["targets"]
        a["targets"] += n
        for k in ["active_tokens", "wwm_mass", "full_visible", "counterpart_any_visible", "counterpart_full_visible", "whole_word_copy", "complete_bpe_copy", "contentlike"]:
            a[k] += m[k]
        a["seen_bpe_sum_weighted"] += m["seen_bpe_frac_mean"] * n
        for z, v in m["copy_zone_targets"].items():
            a["copy_zone_targets"][z] += v
        for z, v in m["copy_zone_active"].items():
            a["copy_zone_active"][z] += v
    for a in agg.values():
        n = a["targets"] or 1
        a["full_visible_frac"] = a["full_visible"] / n
        a["counterpart_any_visible_frac"] = a["counterpart_any_visible"] / n
        a["counterpart_full_visible_frac"] = a["counterpart_full_visible"] / n
        a["whole_word_copy_frac"] = a["whole_word_copy"] / n
        a["complete_bpe_copy_frac"] = a["complete_bpe_copy"] / n
        a["seen_bpe_frac_mean"] = a["seen_bpe_sum_weighted"] / n
        a["contentlike_frac"] = a["contentlike"] / n
        a["copy_zone_targets"] = dict(a["copy_zone_targets"])
        a["copy_zone_active"] = dict(a["copy_zone_active"])
    return agg


def common_pair_ids(metrics: Dict[Tuple[str, str], dict], a: str, b: str) -> List[str]:
    aa = {pid for (var, pid) in metrics if var == a}
    bb = {pid for (var, pid) in metrics if var == b}
    return sorted(aa & bb)


def compare_pairwise(metrics: Dict[Tuple[str, str], dict], a: str, b: str, pair_ids: List[str]) -> dict:
    fields = ["active_tokens", "wwm_mass", "whole_word_copy_frac", "complete_bpe_copy_frac", "contentlike_frac", "counterpart_any_visible_frac"]
    zone_fields = [f"active_{z}" for z in COPY_ZONES]
    rows = []
    for pid in pair_ids:
        ma = metrics[(a, pid)]; mb = metrics[(b, pid)]
        d = {"pair_id": pid}
        for fld in fields:
            d[f"delta_{fld}"] = float(ma[fld] - mb[fld])
        for z in COPY_ZONES:
            d[f"delta_active_{z}"] = float(ma["copy_zone_active"].get(z, 0.0) - mb["copy_zone_active"].get(z, 0.0))
            d[f"delta_targets_{z}"] = float(ma["copy_zone_targets"].get(z, 0) - mb["copy_zone_targets"].get(z, 0))
        rows.append(d)
    summary = {"contrast": f"{a}_minus_{b}", "pairs": len(rows)}
    all_fields = fields + zone_fields
    for fld in fields:
        vals = [r[f"delta_{fld}"] for r in rows]
        summary[f"delta_{fld}_sum"] = float(sum(vals))
        summary[f"delta_{fld}_mean"] = mean(vals)
        summary[f"delta_{fld}_quantiles"] = quantiles(vals)
        summary[f"delta_{fld}_positive_frac"] = mean([v > 0 for v in vals])
    for z in COPY_ZONES:
        vals = [r[f"delta_active_{z}"] for r in rows]
        summary[f"delta_active_{z}_sum"] = float(sum(vals))
        summary[f"delta_active_{z}_mean"] = mean(vals)
        summary[f"delta_active_{z}_quantiles"] = quantiles(vals)
        summary[f"delta_active_{z}_positive_frac"] = mean([v > 0 for v in vals])
    return {"summary": summary, "rows": rows}


def standardize(rows: List[dict], field: str) -> Tuple[float, float]:
    vals = [float(r[field]) for r in rows]
    mu = mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if sd == 0.0:
        sd = 1.0
    return mu, sd


def select_balanced_subset(rows: List[dict], n_select: int, objective_fields: List[str], seed: int, n_iter: int) -> dict:
    """Heuristic subset selection minimizing absolute standardized aggregate deltas.

    We need a practical feasibility signal, not an exact combinatorial proof.  The algorithm
    initializes random subsets and greedily accepts swaps that reduce L1 aggregate imbalance.
    """
    rng = random.Random(seed)
    n = len(rows)
    if n_select >= n:
        selected = list(range(n))
        return score_subset(rows, selected, objective_fields, label="all_pairs")
    # Standardization uses per-pair delta scale.  Aggregate deltas are divided by sd*sqrt(n).
    scales = {}
    for fld in objective_fields:
        mu, sd = standardize(rows, fld)
        # Centering not used in subset sum: absolute raw aggregate delta matters.
        scales[fld] = sd * math.sqrt(n_select)
    def obj(sel_set: set[int]) -> Tuple[float, Dict[str, float]]:
        sums = {fld: 0.0 for fld in objective_fields}
        for i in sel_set:
            r = rows[i]
            for fld in objective_fields:
                sums[fld] += float(r[fld])
        val = sum(abs(sums[fld]) / (scales[fld] if scales[fld] > 0 else 1.0) for fld in objective_fields)
        return val, sums
    best = None
    best_set: set[int] | None = None
    # Mix random and sign-stratified initializations.
    for restart in range(max(8, min(64, n_iter // 200))):
        if restart == 0:
            # sort by first field and take centered around zero when possible
            fld0 = objective_fields[0]
            order = sorted(range(n), key=lambda i: rows[i][fld0])
            # alternate low/high to balance active delta
            cand = []
            lo, hi = 0, n - 1
            while len(cand) < n_select and lo <= hi:
                cand.append(order[lo]); lo += 1
                if len(cand) < n_select and lo <= hi:
                    cand.append(order[hi]); hi -= 1
            sel = set(cand[:n_select])
        else:
            sel = set(rng.sample(range(n), n_select))
        cur, _ = obj(sel)
        non = [i for i in range(n) if i not in sel]
        # Greedy stochastic swaps.
        tries = max(200, n_iter // max(1, min(64, n_select // 128 + 1)))
        for _ in range(tries):
            out_i = rng.choice(tuple(sel))
            in_i = rng.choice(non)
            sel.remove(out_i); non.append(out_i)
            non.remove(in_i); sel.add(in_i)
            new, _ = obj(sel)
            if new <= cur or rng.random() < math.exp((cur - new) / 0.2):
                cur = new
            else:
                sel.remove(in_i); non.append(in_i)
                non.remove(out_i); sel.add(out_i)
        if best is None or cur < best:
            best = cur
            best_set = set(sel)
    assert best_set is not None
    result = score_subset(rows, sorted(best_set), objective_fields, label=f"heuristic_n{n_select}")
    result["objective_value"] = best
    return result


def score_subset(rows: List[dict], selected: List[int] | set[int], objective_fields: List[str], label: str) -> dict:
    idxs = sorted(selected)
    out = {"label": label, "n_pairs": len(idxs), "objective_fields": objective_fields, "aggregate_deltas": {}, "mean_deltas": {}, "abs_mean_deltas": {}}
    for fld in sorted({k for r in rows[:1] for k in r.keys() if k.startswith("delta_")}):
        vals = [float(rows[i][fld]) for i in idxs]
        out["aggregate_deltas"][fld] = float(sum(vals))
        out["mean_deltas"][fld] = mean(vals)
        out["abs_mean_deltas"][fld] = mean([abs(v) for v in vals])
    out["pair_ids"] = [rows[i]["pair_id"] for i in idxs[:200]]  # preview only
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_int(x: float | int) -> str:
    return f"{int(round(float(x))):,}"


def fmt_float(x: float, nd: int = 4) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def make_markdown(result: dict) -> str:
    lines: List[str] = []
    lines.append("# research factorial training feasibility audit")
    lines.append("")
    lines.append("CPU-only audit.  It does not authorize H100 training; it constrains whether a future exact-loader-matched factorial screen is possible after the DeBERTa trajectory and A01 evidence are read.")
    lines.append("")
    lines.append("## Variant aggregate target mass under exact DeBERTa loader")
    lines.append("| variant | pairs | targets | active | WWM mass | whole-word copy | complete-BPE copy | contentlike | tail active | absent active |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for var, a in result["variant_aggregate"].items():
        lines.append("| {var} | {pairs} | {targets} | {active} | {wwm} | {ww:.2f}% | {cb:.2f}% | {cont:.2f}% | {tail} | {absent} |".format(
            var=var,
            pairs=a["pairs"],
            targets=fmt_int(a["targets"]),
            active=fmt_int(a["active_tokens"]),
            wwm=fmt_int(a["wwm_mass"]),
            ww=pct(a["whole_word_copy_frac"]),
            cb=pct(a["complete_bpe_copy_frac"]),
            cont=pct(a["contentlike_frac"]),
            tail=fmt_int(a["copy_zone_active"].get("tail_only", 0)),
            absent=fmt_int(a["copy_zone_active"].get("absent", 0)),
        ))
    lines.append("")
    lines.append("## Clean single-factor controls already supported")
    for name in ["compact_minus_compact_scrambled", "sourcewide_onegap_minus_sourcewide_onegap_scrambled", "prefix_fluent_minus_prefix_scrambled"]:
        c = result["pairwise_contrasts"][name]["summary"]
        lines.append(f"- `{name}`: active delta {fmt_int(c['delta_active_tokens_sum'])}, WWM delta {fmt_int(c['delta_wwm_mass_sum'])}, complete-BPE-copy delta {fmt_float(c['delta_complete_bpe_copy_frac_mean'], 6)} mean fraction. This remains the cleanest ordered-vs-scrambled contrast within a fixed lexical multiset.")
    lines.append("")
    lines.append("## Cross-family feasibility: compact vs sourcewide_onegap")
    c = result["pairwise_contrasts"]["compact_minus_sourcewide_onegap"]["summary"]
    lines.append(f"All 12,155 pairs: compact-sourcewide active delta {fmt_int(c['delta_active_tokens_sum'])}, WWM delta {fmt_int(c['delta_wwm_mass_sum'])}, tail-active delta {fmt_int(c['delta_active_tail_only_sum'])}, absent-active delta {fmt_int(c['delta_active_absent_sum'])}, complete-BPE-copy mean delta {fmt_float(c['delta_complete_bpe_copy_frac_mean'], 4)}.")
    lines.append("")
    lines.append("Pair-level sign structure:")
    lines.append(f"- active-token delta positive fraction {fmt_float(c['delta_active_tokens_positive_frac'], 3)}; quantiles {c['delta_active_tokens_quantiles']}")
    lines.append(f"- tail-active delta positive fraction {fmt_float(c['delta_active_tail_only_positive_frac'], 3)}; quantiles {c['delta_active_tail_only_quantiles']}")
    lines.append("")
    lines.append("Heuristic subset selection shows whether exposure can be numerically balanced by reducing pair dose.  This does **not** fix lexical-copy composition; it only tells whether active/WWM/tail mass can be made comparable.")
    lines.append("| contrast | subset | n pairs | Δ active | Δ WWM | Δ tail active | Δ absent active | Δ complete-BPE-copy mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, selections in result["subset_feasibility"].items():
        for sel in selections:
            d = sel["aggregate_deltas"]
            md = sel["mean_deltas"]
            lines.append("| {key} | {label} | {n} | {da} | {dw} | {dt} | {dab} | {dcb} |".format(
                key=key,
                label=sel["label"],
                n=sel["n_pairs"],
                da=fmt_int(d.get("delta_active_tokens", 0)),
                dw=fmt_int(d.get("delta_wwm_mass", 0)),
                dt=fmt_int(d.get("delta_active_tail_only", 0)),
                dab=fmt_int(d.get("delta_active_absent", 0)),
                dcb=fmt_float(md.get("delta_complete_bpe_copy_frac", float('nan')), 4),
            ))
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("- The fixed-word-multiset ordered/scrambled arms are mechanically clean and are the only immediately credible exact-loader training contrast from the present candidates.")
    lines.append("- Compact vs sourcewide_onegap can be partially balanced on active/WWM/tail exposure by pair subset selection, but it cannot remove the central lexical-copy/semantic-recoding difference: compact contains many absent/non-source words and much lower complete-BPE copy fraction. Training such a contrast would test a coupled natural-compact-vs-extractive-sourcewide object, not a pure tail-coverage or pure semantic-transformation factor.")
    lines.append("- Therefore the next possible H100 factorial screen, if pending trajectory/A01 evidence leaves this route strongest, should be a minimal ordered-vs-scrambled contrast at fixed compact lexical multiset (compact vs compact_scrambled), optionally paired with sourcewide_onegap vs sourcewide_onegap_scrambled as a second wave. It should not be represented as proving tail coverage or budget efficiency by itself.")
    lines.append("")
    lines.append(f"JSON: `{result['json_path']}`")
    lines.append(f"Pairwise CSVs: `{result['out_dir']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=184043)
    ap.add_argument("--n-iter", type=int, default=2500)
    args = ap.parse_args()

    root = root_dir()
    records_csv = Path(args.records_csv) if args.records_csv else root / "experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/view_word_target_records.csv"
    out_dir = Path(args.out_dir) if args.out_dir else root / "experiments/archive/frontier_consolidation/data/factorial_training_feasibility_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_pair_metrics(records_csv, max_pairs=args.max_pairs)
    variant_agg = aggregate_variant(metrics)

    pairwise = {}
    contrast_defs = [
        ("compact", "compact_scrambled"),
        ("sourcewide_onegap", "sourcewide_onegap_scrambled"),
        ("prefix_fluent", "prefix_scrambled"),
        ("compact", "sourcewide_onegap"),
        ("sourcewide_onegap", "prefix_fluent"),
        ("compact", "prefix_fluent"),
    ]
    for a, b in contrast_defs:
        pids = common_pair_ids(metrics, a, b)
        comp = compare_pairwise(metrics, a, b, pids)
        key = f"{a}_minus_{b}"
        pairwise[key] = comp
        write_csv(out_dir / f"pairwise_{key}.csv", comp["rows"])

    subset_feasibility = {}
    # Compact-sourcewide: balance changed-view active tokens, WWM mass, tail mass.
    cs_rows = pairwise["compact_minus_sourcewide_onegap"]["rows"]
    cs_fields = ["delta_active_tokens", "delta_wwm_mass", "delta_active_tail_only"]
    subset_feasibility["compact_minus_sourcewide_onegap"] = [
        score_subset(cs_rows, list(range(len(cs_rows))), cs_fields, "all_pairs"),
    ]
    for nsel in [9000, 6000, 3000, 1500]:
        subset_feasibility["compact_minus_sourcewide_onegap"].append(
            select_balanced_subset(cs_rows, nsel, cs_fields, args.seed + nsel, args.n_iter)
        )

    # Sourcewide-prefix: balance active/WWM; source-position spread remains not fixed lexical identity.
    sp_rows = pairwise["sourcewide_onegap_minus_prefix_fluent"]["rows"]
    sp_fields = ["delta_active_tokens", "delta_wwm_mass", "delta_active_tail_only"]
    subset_feasibility["sourcewide_onegap_minus_prefix_fluent"] = [
        score_subset(sp_rows, list(range(len(sp_rows))), sp_fields, "all_pairs"),
    ]
    for nsel in [9000, 6000, 3000, 1500]:
        subset_feasibility["sourcewide_onegap_minus_prefix_fluent"].append(
            select_balanced_subset(sp_rows, nsel, sp_fields, args.seed + 17 + nsel, args.n_iter)
        )

    # Strip rows from JSON summaries, keep CSVs as durable detailed records.
    pairwise_summary = {k: {"summary": v["summary"], "csv": str(out_dir / f"pairwise_{k}.csv")} for k, v in pairwise.items()}
    result = {
        "status": "FACTORIAL_TRAINING_FEASIBILITY_AUDIT",
        "records_csv": str(records_csv),
        "out_dir": str(out_dir),
        "variant_aggregate": variant_agg,
        "pairwise_contrasts": pairwise_summary,
        "subset_feasibility": subset_feasibility,
        "interpretation_boundary": "CPU feasibility only; no H100 training is authorized until DeBERTa common-grid and A01 evidence are collected/read.",
    }
    json_path = out_dir / "factorial_training_feasibility_audit.json"
    result["json_path"] = str(json_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = make_markdown(result)
    (out_dir / "factorial_training_feasibility_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(out_dir), "variants": len(variant_agg), "contrasts": len(pairwise_summary)}, indent=2))


if __name__ == "__main__":
    main()
