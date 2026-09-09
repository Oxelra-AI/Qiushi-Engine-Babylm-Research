#!/usr/bin/env python3
"""research: analyze raw-name binding probe results.

Reads saved result.json files from primary and comparison-only runs.
Produces:
  1. Bridge-sign paired comparison (same as research analysis)
  2. Fresh rename vs original comparison
  3. Binding diagnostic: does train fit reach 1.0?
  4. Comparison with research <cand>/<other> baseline
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("experiments/archive/representation_and_objectives")
PRIMARY_DIR = PROJECT / "data/primary_raw_name"
CMPONLY_DIR = PROJECT / "data/comparison_only_raw_name"
DIR = PROJECT / "data/primary_gauge"
OUTPUT_DIR = PROJECT / "data/raw_name_analysis"


def load_results(base_dir: Path):
    results = []
    for rj in sorted(base_dir.rglob("result.json")):
        try:
            results.append(json.loads(rj.read_text()))
        except Exception as e:
            print(f"WARN: {rj}: {e}", file=sys.stderr)
    return results


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    primary = load_results(PRIMARY_DIR)
    cmponly = load_results(CMPONLY_DIR)

    lines = ["# research raw-name binding analysis\n\n"]

    # 1. Training convergence diagnostic
    lines.append("## Training convergence\n\n")
    lines.append("| experiment | condition | bridge_sign | seed | train_state | train_cmp | epochs |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for label, results in [("primary", primary), ("cmp_only", cmponly)]:
        for r in results:
            ce = r.get("central_eval", {})
            lines.append(f"| {label} | {r['condition']} | {r['bridge_sign']:+d} | {r['seed']} | "
                         f"{ce.get('train_state_acc', float('nan')):.3f} | "
                         f"{ce.get('train_cmp_acc', float('nan')):.3f} | {r.get('epochs', '?')} |\n")

    # 2. Primary central readout
    lines.append("\n## Primary readout (full graph + anchors)\n\n")
    lines.append("| condition | bs | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc | mixed_margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in primary:
        ce = r.get("central_eval", {})
        lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | "
                     f"{ce.get('direct_same', float('nan')):.3f} | "
                     f"{ce.get('graph_same', float('nan')):.3f} | "
                     f"{ce.get('pair_both_graph_same', float('nan')):.3f} | "
                     f"{ce.get('unchanged', float('nan')):.3f} | "
                     f"{ce.get('hh_closure', float('nan')):.3f} | "
                     f"{ce.get('mixed_acc', float('nan')):.3f} | "
                     f"{ce.get('mixed_margin', float('nan')):.1f} |\n")

    # 3. Fresh rename comparison
    has_rename = any(r.get("fresh_rename") for r in primary)
    if has_rename:
        lines.append("\n## Fresh rename comparison\n\n")
        lines.append("| condition | bs | orig_graph_same | rename_graph_same | orig_hh | rename_hh | orig_mixed | rename_mixed |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in primary:
            ce = r.get("central_eval", {})
            lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | "
                         f"{ce.get('graph_same', float('nan')):.3f} | "
                         f"{ce.get('rename_graph_same', float('nan')):.3f} | "
                         f"{ce.get('hh_closure', float('nan')):.3f} | "
                         f"{ce.get('rename_hh_closure', float('nan')):.3f} | "
                         f"{ce.get('mixed_acc', float('nan')):.3f} | "
                         f"{ce.get('rename_mixed_acc', float('nan')):.3f} |\n")

    # 4. Comparison-only readout
    if cmponly:
        lines.append("\n## Comparison-only readout (narrow filter: no changed bridge anchors)\n\n")
        lines.append("| condition | bs | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in cmponly:
            ce = r.get("central_eval", {})
            lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | "
                         f"{ce.get('direct_same', float('nan')):.3f} | "
                         f"{ce.get('graph_same', float('nan')):.3f} | "
                         f"{ce.get('pair_both_graph_same', float('nan')):.3f} | "
                         f"{ce.get('unchanged', float('nan')):.3f} | "
                         f"{ce.get('hh_closure', float('nan')):.3f} | "
                         f"{ce.get('mixed_acc', float('nan')):.3f} |\n")

    # 5. Bridge-sign paired analysis for primary
    lines.append("\n## Bridge-sign paired analysis (primary)\n\n")
    by_cond_seed = defaultdict(dict)
    for r in primary:
        key = (r["condition"], r["seed"])
        by_cond_seed[key][r["bridge_sign"]] = r

    for (cond, seed), bsmap in sorted(by_cond_seed.items()):
        if 1 not in bsmap or -1 not in bsmap:
            continue
        p = bsmap[1].get("central_eval", {})
        m = bsmap[-1].get("central_eval", {})
        lines.append(f"### {cond}|seed{seed}\n\n")
        lines.append(f"- bs+1 graph_same={p.get('graph_same', float('nan')):.3f}, "
                     f"bs-1 graph_same={m.get('graph_same', float('nan')):.3f}\n")
        lines.append(f"- bs+1 mixed_acc={p.get('mixed_acc', float('nan')):.3f}, "
                     f"bs-1 mixed_acc={m.get('mixed_acc', float('nan')):.3f}\n")
        lines.append(f"- bs+1 hh_closure={p.get('hh_closure', float('nan')):.3f}, "
                     f"bs-1 hh_closure={m.get('hh_closure', float('nan')):.3f}\n")
        lines.append(f"- bs+1 unchanged={p.get('unchanged', float('nan')):.3f}, "
                     f"bs-1 unchanged={m.get('unchanged', float('nan')):.3f}\n")
        lines.append(f"- bs+1 direct_same={p.get('direct_same', float('nan')):.3f}, "
                     f"bs-1 direct_same={m.get('direct_same', float('nan')):.3f}\n")
        lines.append(f"- bs+1 pair_both_graph_same={p.get('pair_both_graph_same', float('nan')):.3f}, "
                     f"bs-1 pair_both_graph_same={m.get('pair_both_graph_same', float('nan')):.3f}\n")
        lines.append(f"- bs+1 mixed_margin={p.get('mixed_margin', float('nan')):.2f}, "
                     f"bs-1 mixed_margin={m.get('mixed_margin', float('nan')):.2f}\n\n")

    # 6. Comparison with research <cand>/<other> baseline
    research = load_results(DIR)
    if research:
        lines.append("\n## research <cand>/<other> baseline comparison\n\n")
        lines.append("| experiment | condition | bs | graph_same | hh_closure | mixed_acc | mixed_margin |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        for r in research:
            ce = r.get("central_eval", {})
            lines.append(f"| cand_other | {r['condition']} | {r['bridge_sign']:+d} | "
                         f"{ce.get('graph_same', float('nan')):.3f} | "
                         f"{ce.get('hh_closure', float('nan')):.3f} | "
                         f"{ce.get('mixed_acc', float('nan')):.3f} | "
                         f"{ce.get('mixed_margin', float('nan')):.1f} |\n")
        for r in primary:
            ce = r.get("central_eval", {})
            lines.append(f"| raw_name | {r['condition']} | {r['bridge_sign']:+d} | "
                         f"{ce.get('graph_same', float('nan')):.3f} | "
                         f"{ce.get('hh_closure', float('nan')):.3f} | "
                         f"{ce.get('mixed_acc', float('nan')):.3f} | "
                         f"{ce.get('mixed_margin', float('nan')):.1f} |\n")

    # Scientific interpretation
    lines.append("\n## Scientific interpretation\n\n")
    # Check if binding succeeds
    primary_fits = [(r["condition"], r.get("central_eval", {}).get("train_state_acc", 0),
                     r.get("central_eval", {}).get("train_cmp_acc", 0))
                    for r in primary]
    all_fit = all(s >= 0.99 and c >= 0.99 for _, s, c in primary_fits if not math.isnan(s))
    lines.append(f"Training convergence: {'all runs reached 1.0' if all_fit else 'SOME RUNS DID NOT CONVERGE'}\n\n")

    if all_fit:
        # Check if transport works for shared_trunk
        shared_trunk_primary = [r for r in primary if r["condition"] == "shared_trunk"]
        if shared_trunk_primary:
            bs_map = {r["bridge_sign"]: r.get("central_eval", {}) for r in shared_trunk_primary}
            if 1 in bs_map and -1 in bs_map:
                gs_p = bs_map[1].get("graph_same", 0)
                gs_m = bs_map[-1].get("graph_same", 0)
                if gs_p > 0.9 and gs_m < 0.1:
                    lines.append("**Gauge transport survives raw-name binding.** shared_trunk graph_same reverses "
                                 "under bridge_sign, matching research result.\n")
                elif gs_p > 0.9:
                    lines.append(f"**Partial result.** bs+1 graph_same={gs_p:.3f} but bs-1 graph_same={gs_m:.3f}; "
                                 "transport is present but possibly weaker with raw names.\n")
                else:
                    lines.append(f"**Transport fails.** graph_same={gs_p:.3f}/{gs_m:.3f}; "
                                 "shared representation insufficient when binding must be learned.\n")
    else:
        lines.append("Binding appears to fail — the model cannot reliably learn character-pattern matching "
                     "with this architecture. Consider adding attention or a matching module.\n")

    md_path = out / "raw_name_analysis.md"
    md_path.write_text("".join(lines), encoding="utf-8")

    all_data = {"primary": primary, "comparison_only": cmponly, "baseline": research}
    json_path = out / "raw_name_analysis.json"
    json_path.write_text(json.dumps(all_data, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "json": str(json_path.relative_to(Path.cwd())),
        "n_primary": len(primary),
        "n_cmponly": len(cmponly),
        "n_step290": len(research),
        "status": "RAW_NAME_ANALYSIS_COMPLETE",
        "summary": str(md_path.relative_to(Path.cwd())),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
