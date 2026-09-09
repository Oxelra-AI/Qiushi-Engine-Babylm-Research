#!/usr/bin/env python3
"""research: analyze clean-init supplied harness and raw-name attention outputs.

This script is deliberately post-hoc and file-based. It does not run training.
It distinguishes:
  1. central supplied-harness clean-init replication;
  2. raw-name query-attention binding fit;
  3. raw-name transport signals that should only be interpreted after (1).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def sign(x: float, eps: float = 1e-6) -> int:
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def run_dirs(base: Path) -> List[Path]:
    if not base.exists(): return []
    return sorted([d for d in base.iterdir() if d.is_dir() and (d / "result.json").exists()])


def normalize_central(rec: Dict[str, Any], source: str) -> Dict[str, Any]:
    c = rec.get("central_eval", {})
    ft = rec.get("final_train_metrics", {})
    # Attention rec stores train metrics in central_eval from sampled history.
    return {
        "source": source,
        "condition": rec.get("condition"),
        "seed": rec.get("seed"),
        "bridge_sign": rec.get("bridge_sign"),
        "cell_tag": rec.get("cell_tag"),
        "train_state_acc": ft.get("train_state_acc", c.get("train_state_acc")),
        "train_cmp_acc": ft.get("train_cmp_acc", c.get("train_cmp_acc")),
        "direct_same": c.get("direct_same"),
        "graph_same": c.get("graph_same"),
        "pair_both_graph_same": c.get("pair_both_graph_same"),
        "unchanged": c.get("unchanged"),
        "hh_closure": c.get("heldheld_unseen_edge_closure_acc", c.get("hh_closure")),
        "mixed_acc": c.get("mixed_held_seen_orientation_acc", c.get("mixed_acc")),
        "mixed_margin": c.get("mixed_held_seen_orientation_signed_margin", c.get("mixed_margin")),
        "direct_mean_de": c.get("direct_mean_de"),
        "graph_mean_de": c.get("graph_mean_de"),
        "direct_same_margin": c.get("direct_same_margin"),
        "graph_same_margin": c.get("graph_same_margin"),
        "elapsed_seconds": rec.get("train_info", {}).get("elapsed_seconds", rec.get("elapsed_seconds")),
        "init_hash": rec.get("init_hash"),
    }


def load_rec_table(base: Path, source: str) -> List[Dict[str, Any]]:
    rows = []
    # Prefer summary JSON when present, but per-run result files are safer and include init hashes.
    for d in run_dirs(base):
        rows.append(normalize_central(load_json(d / "result.json"), source))
    if rows:
        return rows
    # Fallback for attention summary JSON.
    for fname in ["raw_name_attention_summary.json", "raw_name_binding_summary.json"]:
        fp = base / fname
        if fp.exists():
            return [normalize_central(r, source) for r in load_json(fp)]
    return rows


def pair_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(dict)
    for r in rows:
        by[(r["condition"], r["seed"])][r["bridge_sign"]] = r
    out = {}
    for (cond, seed), d in sorted(by.items()):
        if 1 not in d or -1 not in d:
            continue
        p, m = d[1], d[-1]
        def flip_metric(name: str):
            if finite(p.get(name)) and finite(m.get(name)):
                return sign(float(p[name])) == -sign(float(m[name])) and sign(float(p[name])) != 0
            return None
        out[f"{cond}|seed{seed}"] = {
            "train_fit_plus": {"state": p.get("train_state_acc"), "cmp": p.get("train_cmp_acc")},
            "train_fit_minus": {"state": m.get("train_state_acc"), "cmp": m.get("train_cmp_acc")},
            "graph_same_plus": p.get("graph_same"),
            "graph_same_minus": m.get("graph_same"),
            "direct_same_plus": p.get("direct_same"),
            "direct_same_minus": m.get("direct_same"),
            "pair_both_plus": p.get("pair_both_graph_same"),
            "pair_both_minus": m.get("pair_both_graph_same"),
            "unchanged_plus": p.get("unchanged"),
            "unchanged_minus": m.get("unchanged"),
            "hh_closure_plus": p.get("hh_closure"),
            "hh_closure_minus": m.get("hh_closure"),
            "mixed_acc_plus": p.get("mixed_acc"),
            "mixed_acc_minus": m.get("mixed_acc"),
            "mixed_margin_plus": p.get("mixed_margin"),
            "mixed_margin_minus": m.get("mixed_margin"),
            "graph_same_margin_plus": p.get("graph_same_margin"),
            "graph_same_margin_minus": m.get("graph_same_margin"),
            "graph_mean_de_plus": p.get("graph_mean_de"),
            "graph_mean_de_minus": m.get("graph_mean_de"),
            "graph_margin_sign_flip": flip_metric("graph_same_margin"),
            "graph_de_sign_flip": flip_metric("graph_mean_de"),
            "mixed_margin_sign_flip": flip_metric("mixed_margin"),
        }
    return out


def find_run_dir(base: Path, condition: str, seed: int, bridge_sign: int) -> Optional[Path]:
    for d in run_dirs(base):
        try:
            r = load_json(d / "result.json")
        except Exception:
            continue
        if r.get("condition") == condition and r.get("seed") == seed and r.get("bridge_sign") == bridge_sign:
            return d
    return None


def load_state_preds(run_dir: Path) -> List[Dict[str, Any]]:
    for name in ["eval_state_predictions.jsonl", "state_predictions.jsonl"]:
        rows = load_jsonl(run_dir / name)
        if rows: return rows
    return []


def load_comp_preds(run_dir: Path) -> List[Dict[str, Any]]:
    for name in ["eval_comparison_predictions.jsonl", "comp_predictions.jsonl"]:
        rows = load_jsonl(run_dir / name)
        if rows: return rows
    return []


def state_de_by_key(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        if r.get("candidate_index") == 0 or "candidate_index" not in r:
            by[r.get("query_key")].append(r)
    demap, meta = {}, {}
    # prediction rows contain d_e repeated per candidate; candidate0 is enough.
    for r in rows:
        k = r.get("query_key")
        if k is None or k in demap:
            continue
        if "d_e" in r:
            demap[k] = float(r["d_e"])
            meta[k] = r
    return demap, meta


def row_paired_sign(base: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for key in pair_summary(rows):
        cond, seed_s = key.split("|seed")
        seed = int(seed_s)
        dp_dir = find_run_dir(base, cond, seed, 1)
        dm_dir = find_run_dir(base, cond, seed, -1)
        if not dp_dir or not dm_dir:
            continue
        sp, sm = load_state_preds(dp_dir), load_state_preds(dm_dir)
        de_p, meta_p = state_de_by_key(sp)
        de_m, meta_m = state_de_by_key(sm)
        common = sorted(set(de_p) & set(de_m))
        cats = {
            "graph_transfer_changed_same": lambda r: r.get("relation_family") == "graph_transfer" and bool(r.get("is_changed")) and r.get("initial_pattern") == "same",
            "graph_transfer_changed_all": lambda r: r.get("relation_family") == "graph_transfer" and bool(r.get("is_changed")),
            "direct_anchor_changed_same": lambda r: r.get("relation_family") == "direct_anchor" and bool(r.get("is_changed")) and r.get("initial_pattern") == "same",
            "unchanged_all": lambda r: not bool(r.get("is_changed")),
        }
        kres = {}
        for cname, filt in cats.items():
            ks = [k for k in common if k in meta_p and filt(meta_p[k])]
            same = opp = zero = 0
            vals_p, vals_m = [], []
            for k in ks:
                a, b = de_p[k], de_m[k]
                vals_p.append(a); vals_m.append(b)
                sa, sb = sign(a), sign(b)
                if sa == 0 or sb == 0: zero += 1
                elif sa == sb: same += 1
                else: opp += 1
            kres[cname] = {
                "n": len(ks), "same_sign": same, "opposite_sign": opp, "near_zero": zero,
                "opposite_frac": opp/len(ks) if ks else None,
                "mean_plus": mean(vals_p), "mean_minus": mean(vals_m),
            }
        out[key] = kres
    return out


def determine_clean_status(pairs: Dict[str, Any], init_audit: Dict[str, Any]) -> Dict[str, Any]:
    init_ok = all(a.get("untouched_after_all_runs") for a in init_audit.get("init_audits", [])) if init_audit else False
    st = pairs.get("shared_trunk|seed29000", {})
    un = pairs.get("untied|seed29000", {})
    shared_fit = (st.get("train_fit_plus",{}).get("state") == 1.0 and st.get("train_fit_plus",{}).get("cmp") == 1.0 and
                  st.get("train_fit_minus",{}).get("state") == 1.0 and st.get("train_fit_minus",{}).get("cmp") == 1.0)
    shared_transport = (st.get("graph_same_plus") == 1.0 and st.get("graph_same_minus") == 0.0 and
                        st.get("pair_both_plus") == 1.0 and st.get("pair_both_minus") == 0.0 and
                        st.get("hh_closure_plus") == 1.0 and st.get("hh_closure_minus") == 1.0 and
                        st.get("graph_de_sign_flip") is True)
    untied_fit = (un.get("train_fit_plus",{}).get("state") == 1.0 and un.get("train_fit_plus",{}).get("cmp") == 1.0 and
                  un.get("train_fit_minus",{}).get("state") == 1.0 and un.get("train_fit_minus",{}).get("cmp") == 1.0)
    untied_nontransport = not (un.get("graph_same_plus") == 1.0 and un.get("graph_same_minus") == 0.0 and un.get("graph_de_sign_flip") is True)
    return {
        "init_untouched": init_ok,
        "shared_trunk_full_fit": shared_fit,
        "shared_trunk_clean_transport": shared_transport,
        "untied_full_fit": untied_fit,
        "untied_lacks_shared_transport": untied_nontransport,
        "clean_mechanism_reestablished": bool(init_ok and shared_fit and shared_transport and untied_fit and untied_nontransport),
    }


def attention_status(pairs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, p in pairs.items():
        cond = key.split("|seed")[0]
        fit = (p.get("train_fit_plus",{}).get("state") is not None and p.get("train_fit_plus",{}).get("cmp") is not None and
               p.get("train_fit_minus",{}).get("state") is not None and p.get("train_fit_minus",{}).get("cmp") is not None and
               min(float(p["train_fit_plus"]["state"]), float(p["train_fit_plus"]["cmp"]),
                   float(p["train_fit_minus"]["state"]), float(p["train_fit_minus"]["cmp"])) >= 0.99)
        transport_like = (p.get("graph_same_plus") is not None and p.get("graph_same_minus") is not None and
                          float(p["graph_same_plus"]) >= 0.95 and float(p["graph_same_minus"]) <= 0.05 and
                          p.get("graph_de_sign_flip") is True)
        out[key] = {"condition": cond, "binding_fit": fit, "transport_like": transport_like, **p}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, default=Path("experiments/archive/representation_and_objectives/data/clean_gauge_from_init"))
    ap.add_argument("--attention-dir", type=Path, default=Path("experiments/archive/representation_and_objectives/data/raw_name_attention_primary"))
    ap.add_argument("--out", type=Path, default=Path("experiments/archive/representation_and_objectives/data/clean_attention_analysis"))
    args = ap.parse_args()

    clean_rows = load_rec_table(args.clean_dir, "clean_supplied_harness")
    attention_rows = load_rec_table(args.attention_dir, "raw_name_query_attention")
    clean_pairs = pair_summary(clean_rows)
    attention_pairs = pair_summary(attention_rows)
    init_audit = load_json(args.clean_dir / "clean_init_audit.json") if (args.clean_dir / "clean_init_audit.json").exists() else {}
    clean_status = determine_clean_status(clean_pairs, init_audit) if clean_rows else {"clean_mechanism_reestablished": False, "reason": "missing clean rows"}
    attention_stat = attention_status(attention_pairs) if attention_rows else {}
    rowpaired = {
        "clean_supplied_harness": row_paired_sign(args.clean_dir, clean_rows) if clean_rows else {},
        "raw_name_query_attention": row_paired_sign(args.attention_dir, attention_rows) if attention_rows else {},
    }

    result = {
        "clean_rows": clean_rows,
        "attention_rows": attention_rows,
        "clean_pairs": clean_pairs,
        "attention_pairs": attention_pairs,
        "clean_status": clean_status,
        "attention_status": attention_stat,
        "row_paired_state_de": rowpaired,
        "interpretation_rule": "Raw-name transport-like signals are not interpreted as mechanism evidence unless clean_status.clean_mechanism_reestablished is true; attention pilot first establishes whether learned candidate matching reaches train comparison fit.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "clean_attention_analysis.json", result)

    lines = ["# research clean-init and raw-name attention analysis\n\n",
             "## Clean supplied-harness status\n\n"]
    for k, v in clean_status.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n### Clean supplied-harness pairs\n\n")
    for key, val in clean_pairs.items():
        lines.append(f"#### {key}\n")
        for kk, vv in val.items():
            lines.append(f"- {kk}: {vv}\n")
        lines.append("\n")
    lines.append("\n## Raw-name query-attention pairs\n\n")
    if not attention_stat:
        lines.append("No completed attention rows found.\n")
    for key, val in attention_stat.items():
        lines.append(f"#### {key}\n")
        keep = ["binding_fit", "transport_like", "train_fit_plus", "train_fit_minus", "graph_same_plus", "graph_same_minus", "pair_both_plus", "pair_both_minus", "unchanged_plus", "unchanged_minus", "hh_closure_plus", "hh_closure_minus", "mixed_acc_plus", "mixed_acc_minus", "mixed_margin_plus", "mixed_margin_minus", "graph_de_sign_flip"]
        for kk in keep:
            lines.append(f"- {kk}: {val.get(kk)}\n")
        lines.append("\n")
    lines.append("\n## Row-paired state d_e sign reversal\n\n")
    for source, dct in rowpaired.items():
        lines.append(f"### {source}\n")
        if not dct:
            lines.append("No row-paired predictions found.\n")
        for key, cats in dct.items():
            lines.append(f"#### {key}\n")
            for cname, vals in cats.items():
                lines.append(f"- {cname}: n={vals.get('n')}, opposite_frac={vals.get('opposite_frac')}, mean_plus={vals.get('mean_plus')}, mean_minus={vals.get('mean_minus')}\n")
            lines.append("\n")
    lines.append("\n## Scientific reading\n\n")
    lines.append("The clean supplied-harness recheck is the load-bearing result: it must show untouched initializations, full local fit, shared_trunk sign-controlled h1/h3 reversal, and untied non-transport. The attention result first tests whether candidate matching is learnable in raw names; any transport-like pattern is provisional unless the clean recheck has reestablished the original mechanism.\n")
    (args.out / "clean_attention_analysis.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": "CLEAN_ATTENTION_ANALYSIS_COMPLETE", "json": str(args.out / "clean_attention_analysis.json"), "summary": str(args.out / "clean_attention_analysis.md"), "n_clean_rows": len(clean_rows), "n_attention_rows": len(attention_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
