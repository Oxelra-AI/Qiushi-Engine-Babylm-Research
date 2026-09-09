#!/usr/bin/env python3
"""Analyze research dual-position learned-equality gauge outputs."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "experiments/archive/representation_and_objectives/scripts")
import posalign_analysis as pair  # noqa: E402


def load_summary(d: Path):
    p = d / "dualpos_gauge_summary.json"
    if not p.exists():
        p = d / "posalign_summary.json"
    obj = json.loads(p.read_text())
    return obj.get("results", obj.get("all_results", []))


def main():
    bp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/archive/representation_and_objectives/data/dualpos_fullalpha_shared_bsplus_seed29930")
    bm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments/archive/representation_and_objectives/data/dualpos_fullalpha_shared_bsminus_seed29930")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("experiments/archive/representation_and_objectives/data/dualpos_fullalpha_pair_analysis")
    out.mkdir(parents=True, exist_ok=True)
    rowpair = pair.analyze(bp, bm, out)
    plus = load_summary(bp)
    minus = load_summary(bm)
    def slim(rs):
        if not rs: return {}
        r = rs[0]
        ti = r.get("train_info", {}).get("history", [{}])[-1] if r.get("train_info", {}).get("history") else {}
        ce = r.get("central_eval", {})
        return {
            "condition": r.get("condition"), "seed": r.get("seed"), "bridge_sign": r.get("bridge_sign"),
            "init_hash_prefix": r.get("init_hash_prefix"),
            "char_pair_mode": r.get("char_pair_mode", r.get("mode")),
            "char_pair_epochs": r.get("char_pair_epochs"),
            "matching_before": r.get("matching_before"),
            "matching_after_pretrain": r.get("matching_after_pretrain"),
            "matching_final": r.get("matching_final"),
            "train_state": ti.get("train_state"), "train_changed": ti.get("train_changed"),
            "train_cmp": ti.get("train_cmp"),
            "graph_same": ce.get("graph_same"), "unchanged": ce.get("unchanged"),
            "hh_closure": ce.get("hh_closure"), "mixed_acc": ce.get("mixed_acc"),
            "direct_same": ce.get("direct_same"), "pair_both_graph_same": ce.get("pair_both_graph_same"),
            "graph_same_margin": ce.get("graph_same_margin"), "mixed_acc_margin": ce.get("mixed_acc_margin"),
        }
    report = {"plus": slim(plus), "minus": slim(minus), "row_paired": rowpair}
    report["same_init_hash"] = report["plus"].get("init_hash_prefix") == report["minus"].get("init_hash_prefix")
    report["both_match_after_1"] = (report["plus"].get("matching_after_pretrain", {}).get("eval") == 1.0 and
                                     report["minus"].get("matching_after_pretrain", {}).get("eval") == 1.0)
    report["both_train_fit_1"] = (report["plus"].get("train_state") == 1.0 and report["plus"].get("train_cmp") == 1.0 and
                                  report["minus"].get("train_state") == 1.0 and report["minus"].get("train_cmp") == 1.0)
    (out / "dualpos_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# research dual-position learned-equality pair analysis", "",
             "## Per-sign summaries", "",
             "| sign | init | match before eval | match after eval | train state | train cmp | direct | graph | pair_both | unchanged | hh | mixed |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["plus", "minus"]:
        r = report[name]
        lines.append(f"| {r.get('bridge_sign')} | {r.get('init_hash_prefix')} | {r.get('matching_before',{}).get('eval',0):.3f} | {r.get('matching_after_pretrain',{}).get('eval',0):.3f} | {r.get('train_state',0):.3f} | {r.get('train_cmp',0):.3f} | {r.get('direct_same',0):.3f} | {r.get('graph_same',0):.3f} | {r.get('pair_both_graph_same',0):.3f} | {r.get('unchanged',0):.3f} | {r.get('hh_closure',0):.3f} | {r.get('mixed_acc',0):.3f} |")
    lines += ["", "## Row-paired sign analysis", ""]
    for fam, vals in rowpair.items():
        lines.append(f"- {fam}: {vals}")
    (out / "dualpos_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "DUALPOS_ANALYSIS_COMPLETE", "md": str(out / "dualpos_analysis.md"), "json": str(out / "dualpos_analysis.json"), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
