#!/usr/bin/env python3
"""Collate research midlayer conditional-interaction probes.

Reads the three isolated research probe outputs (matched base, scale1.75 live,
scale1.75 disabled), checks final-layer reproduction against prior endpoint
readouts, computes paired layer deltas, and writes a compact scientific synthesis.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
ROOTS = {
    "matched_base_80M": A01_WS / "data/midlayer_base80",
    "scale1p75_live_80M": A01_WS / "data/midlayer_scale80",
    "scale1p75_disabled_80M": A01_WS / "data/midlayer_disabled80",
}
SUMMARY_INNER = "midlayer_conditional_interaction_summary.json"
OUT = A01_WS / "data/midlayer_synthesis"
NOTE = A01_WS / "notes/midlayer_conditional_interaction_synthesis.md"

PRIOR_EWOK = A01_WS / "data/scale1p75_ewok_inference_ablation/scale1p75_ewok_inference_ablation_summary.json"
PRIOR_GP80 = A01_WS / "data/scale1p75_inference_ablation/scale1p75_inference_ablation_summary.json"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def b(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def qstats(vals) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs)-1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.1), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def load_target(target: str) -> dict[str, Any]:
    root = ROOTS[target]
    summary = read_json(root / SUMMARY_INNER)["targets"][target]
    ewok_rows = read_csv(root / target / "ewok_midlayer_rows.csv")
    gp_rows = read_csv(root / target / "globalpiqa_midlayer_rows.csv")
    return {"root": root, "summary": summary, "ewok_rows": ewok_rows, "gp_rows": gp_rows}


def row_key_ewok(r: dict[str, str]) -> tuple[int, int]:
    return (int(r["global_index"]), int(r["layer_index"]))


def row_key_gp(r: dict[str, str]) -> tuple[str, int]:
    return (str(r["example_id"]), int(r["layer_index"]))


def endpoint_reproduction(targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prior_ewok = read_json(PRIOR_EWOK)
    prior_gp = read_json(PRIOR_GP80)
    mapping_ewok = {
        "matched_base_80M": "matched_legal16k_80M",
        "scale1p75_live_80M": "scale1p75_live",
        "scale1p75_disabled_80M": "scale1p75_disabled",
    }
    mapping_gp = {
        "scale1p75_live_80M": "scale1p75_live",
        "scale1p75_disabled_80M": "scale1p75_disabled",
    }
    # There is no direct prior GP summary for matched base 80M in research; the closest
    # matched legal16k item-flip value is stored in history, not in that JSON.
    out = {}
    for t, obj in targets.items():
        ew = obj["summary"].get("ewok", {})
        final_li = str(obj["summary"].get("n_layers", 9) - 1)
        final_ew = ew.get("layer_summary_all_selected", {}).get(final_li, {})
        prev = prior_ewok.get("variants", {}).get(mapping_ewok[t], {}).get("summary", {})
        gp = obj["summary"].get("globalpiqa", {})
        final_gp = gp.get("layer_summary", {}).get(final_li, {})
        prev_gp = prior_gp.get("variants", {}).get(mapping_gp.get(t, ""), {}).get("modes", {}).get("parallel", {})
        out[t] = {
            "ewok_selected_final_t1_accuracy": final_ew.get("t1_accuracy"),
            "ewok_full_prior_accuracy": prev.get("accuracy"),
            "ewok_selected_final_stable_frac": final_ew.get("stable_failure_frac"),
            "ewok_full_prior_stable_frac": prev.get("stable_failure_frac_all"),
            "globalpiqa_final_all_accuracy": final_gp.get("all", {}).get("accuracy"),
            "globalpiqa_prior_parallel_accuracy": prev_gp.get("accuracy"),
            "globalpiqa_final_hard52_accuracy": final_gp.get("hard52", {}).get("accuracy"),
            "globalpiqa_prior_hard52_accuracy": prev_gp.get("always_wrong_subset", {}).get("accuracy") if prev_gp else None,
        }
    return out


def paired_ewok_delta(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]], label: str) -> dict[str, Any]:
    a = {row_key_ewok(r): r for r in a_rows}
    bmap = {row_key_ewok(r): r for r in b_rows}
    common = sorted(set(a) & set(bmap))
    by_layer: dict[int, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for k in common:
        by_layer[k[1]].append((a[k], bmap[k]))
    out = {"label": label, "n_common_layer_rows": len(common), "by_layer": {}}
    for li, pairs in sorted(by_layer.items()):
        n = len(pairs)
        delta_inter = [f(rb["interaction_sum"]) - f(ra["interaction_sum"]) for ra, rb in pairs]
        delta_t1 = [f(rb["official_margin_t1_sum"]) - f(ra["official_margin_t1_sum"]) for ra, rb in pairs]
        delta_both = [int(b(rb["both_official_positive"])) - int(b(ra["both_official_positive"])) for ra, rb in pairs]
        delta_stable = [int(b(rb["stable_conditional_failure"])) - int(b(ra["stable_conditional_failure"])) for ra, rb in pairs]
        out["by_layer"][str(li)] = {
            "n": n,
            "mean_delta_interaction_sum_b_minus_a": statistics.fmean(delta_inter),
            "median_delta_interaction_sum_b_minus_a": statistics.median(delta_inter),
            "mean_delta_t1_margin_b_minus_a": statistics.fmean(delta_t1),
            "net_both_official_count_b_minus_a": sum(delta_both),
            "net_stable_failure_count_b_minus_a": sum(delta_stable),
            "b_better_interaction_n": sum(d > 0 for d in delta_inter),
            "a_better_interaction_n": sum(d < 0 for d in delta_inter),
        }
    return out


def paired_gp_delta(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]], label: str) -> dict[str, Any]:
    a = {row_key_gp(r): r for r in a_rows}
    bmap = {row_key_gp(r): r for r in b_rows}
    common = sorted(set(a) & set(bmap))
    by_layer: dict[int, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for k in common:
        by_layer[k[1]].append((a[k], bmap[k]))
    out = {"label": label, "n_common_layer_rows": len(common), "by_layer": {}, "hard52_by_layer": {}}
    for li, pairs in sorted(by_layer.items()):
        for hard_only, dest in [(False, out["by_layer"]), (True, out["hard52_by_layer"] )]:
            ps = [(ra, rb) for ra, rb in pairs if (not hard_only or b(ra["is_hard52"]))]
            if not ps:
                continue
            delta_margin = [f(rb["top_minus_correct"]) - f(ra["top_minus_correct"]) for ra, rb in ps]
            delta_correct = [int(b(rb["correct"])) - int(b(ra["correct"])) for ra, rb in ps]
            delta_rank = [int(rb["correct_rank"]) - int(ra["correct_rank"]) for ra, rb in ps]
            dest[str(li)] = {
                "n": len(ps),
                "mean_delta_top_minus_correct_b_minus_a": statistics.fmean(delta_margin),
                "median_delta_top_minus_correct_b_minus_a": statistics.median(delta_margin),
                "net_correct_count_b_minus_a": sum(delta_correct),
                "mean_delta_rank_b_minus_a": statistics.fmean(delta_rank),
                "b_lower_margin_n": sum(d < 0 for d in delta_margin),
                "a_lower_margin_n": sum(d > 0 for d in delta_margin),
            }
    return out


def adjacent_persistence_ewok(rows: list[dict[str, str]], final_layer: int) -> dict[str, Any]:
    by_rec: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_rec[int(r["global_index"])][int(r["layer_index"])] = r
    out = {}
    for signal in ["both_official_positive", "interaction_positive", "t1_correct"]:
        final_wrong = []
        any_single = 0; any_adjacent = 0; final_lost_after_adjacent = 0
        for idx, by_li in by_rec.items():
            fin = by_li[final_layer]
            # condition target set: final does not have signal, so a middle recovery could be erased.
            if b(fin.get(signal)):
                continue
            final_wrong.append(idx)
            mids = [li for li in range(final_layer) if li in by_li and b(by_li[li].get(signal))]
            if mids:
                any_single += 1
            if any((li+1 in mids) for li in mids):
                any_adjacent += 1
                final_lost_after_adjacent += 1
        out[signal] = {
            "n_final_without_signal": len(final_wrong),
            "any_single_mid_signal": any_single,
            "any_single_mid_signal_frac": any_single / len(final_wrong) if final_wrong else None,
            "any_adjacent_mid_signal": any_adjacent,
            "any_adjacent_mid_signal_frac": any_adjacent / len(final_wrong) if final_wrong else None,
        }
    return out


def adjacent_persistence_gp(rows: list[dict[str, str]], final_layer: int) -> dict[str, Any]:
    by_rec: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if b(r["is_hard52"]):
            by_rec[str(r["example_id"])][int(r["layer_index"])] = r
    out = {}
    for signal_name, pred in [
        ("correct", lambda r: b(r["correct"])),
        ("rank_le_2", lambda r: int(r["correct_rank"]) <= 2),
    ]:
        final_without = []
        any_single = 0; any_adj = 0
        for uid, by_li in by_rec.items():
            fin = by_li[final_layer]
            if pred(fin):
                continue
            final_without.append(uid)
            mids = [li for li in range(final_layer) if li in by_li and pred(by_li[li])]
            if mids:
                any_single += 1
            if any((li+1 in mids) for li in mids):
                any_adj += 1
        out[signal_name] = {"n_final_without_signal": len(final_without), "any_single_mid_signal": any_single, "any_single_mid_signal_frac": any_single/len(final_without) if final_without else None, "any_adjacent_mid_signal": any_adj, "any_adjacent_mid_signal_frac": any_adj/len(final_without) if final_without else None}
    return out


def best_layers(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for t, obj in summary.items():
        ew = obj["summary"]["ewok"]["layer_summary_all_selected"]
        gp = obj["summary"]["globalpiqa"]["layer_summary"]
        out[t] = {
            "ewok_best_both_official_layer": max(ew, key=lambda k: ew[k].get("both_official_frac") or -1),
            "ewok_best_interaction_positive_layer": max(ew, key=lambda k: ew[k].get("interaction_positive_frac") or -1),
            "ewok_lowest_stable_failure_layer": min(ew, key=lambda k: ew[k].get("stable_failure_frac") if ew[k].get("stable_failure_frac") is not None else 999),
            "gp_best_all_accuracy_layer": max(gp, key=lambda k: gp[k]["all"].get("accuracy") if gp[k]["all"].get("accuracy") is not None else -1),
            "gp_best_hard52_accuracy_layer": max(gp, key=lambda k: gp[k]["hard52"].get("accuracy") if gp[k]["hard52"].get("accuracy") is not None else -1),
        }
    return out


def make_note(out: dict[str, Any]) -> None:
    lines = []
    lines.append("# research midlayer conditional-interaction synthesis\n\n")
    lines.append("This synthesis compares matched legal16k base, scale1.75 live, and scale1.75-disabled layerwise residual-state readouts. The readout applies the final MLM head to each layer, so positive midlayer recovery is meaningful, while negative decodability is not proof of absence.\n\n")
    lines.append(f"Summary JSON: `{out['summary_json']}`\n\n")
    lines.append("## Final-layer reproduction\n\n")
    for t, rec in out["endpoint_reproduction"].items():
        lines.append(f"- `{t}`: EWoK selected-final t1={rec['ewok_selected_final_t1_accuracy']} vs prior full EWoK={rec['ewok_full_prior_accuracy']}; selected-final stable={rec['ewok_selected_final_stable_frac']} vs prior full stable={rec['ewok_full_prior_stable_frac']}; GP final all={rec['globalpiqa_final_all_accuracy']} vs prior parallel={rec['globalpiqa_prior_parallel_accuracy']}; GP final hard52={rec['globalpiqa_final_hard52_accuracy']} vs prior hard52={rec['globalpiqa_prior_hard52_accuracy']}.\n")
    lines.append("\n## Best-layer pattern\n\n")
    for t, rec in out["best_layers"].items():
        lines.append(f"- `{t}`: EWoK best both layer {rec['ewok_best_both_official_layer']}, best interaction layer {rec['ewok_best_interaction_positive_layer']}, lowest stable layer {rec['ewok_lowest_stable_failure_layer']}; GlobalPIQA best all layer {rec['gp_best_all_accuracy_layer']}, best hard52 layer {rec['gp_best_hard52_accuracy_layer']}.\n")
    lines.append("\n## Paired base-scale deltas\n\n")
    for name, rec in out["paired_deltas"].items():
        lines.append(f"### {name}\n")
        if "ewok" in rec:
            for li in sorted(rec["ewok"]["by_layer"], key=lambda x:int(x)):
                r = rec["ewok"]["by_layer"][li]
                lines.append(f"- EWoK L{li}: Δinteraction={r['mean_delta_interaction_sum_b_minus_a']:.4f}, Δt1={r['mean_delta_t1_margin_b_minus_a']:.4f}, net both={r['net_both_official_count_b_minus_a']}, net stable={r['net_stable_failure_count_b_minus_a']}.\n")
        if "gp" in rec:
            for li in sorted(rec["gp"]["hard52_by_layer"], key=lambda x:int(x)):
                r = rec["gp"]["hard52_by_layer"][li]
                lines.append(f"- GP hard52 L{li}: Δtop_minus_correct={r['mean_delta_top_minus_correct_b_minus_a']:.4f} (negative is better for b), net correct={r['net_correct_count_b_minus_a']}, Δrank={r['mean_delta_rank_b_minus_a']:.4f}.\n")
        lines.append("\n")
    lines.append("## Interpretation\n\n")
    lines.append(out["interpretation"] + "\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    targets = {t: load_target(t) for t in ROOTS}
    summaries = {t: obj for t, obj in targets.items()}
    final_layer = 8
    paired = {
        "scale_live_minus_base": {
            "ewok": paired_ewok_delta(targets["matched_base_80M"]["ewok_rows"], targets["scale1p75_live_80M"]["ewok_rows"], "scale_live_minus_base"),
            "gp": paired_gp_delta(targets["matched_base_80M"]["gp_rows"], targets["scale1p75_live_80M"]["gp_rows"], "scale_live_minus_base"),
        },
        "scale_disabled_minus_base": {
            "ewok": paired_ewok_delta(targets["matched_base_80M"]["ewok_rows"], targets["scale1p75_disabled_80M"]["ewok_rows"], "scale_disabled_minus_base"),
            "gp": paired_gp_delta(targets["matched_base_80M"]["gp_rows"], targets["scale1p75_disabled_80M"]["gp_rows"], "scale_disabled_minus_base"),
        },
        "scale_live_minus_disabled": {
            "ewok": paired_ewok_delta(targets["scale1p75_disabled_80M"]["ewok_rows"], targets["scale1p75_live_80M"]["ewok_rows"], "scale_live_minus_disabled"),
            "gp": paired_gp_delta(targets["scale1p75_disabled_80M"]["gp_rows"], targets["scale1p75_live_80M"]["gp_rows"], "scale_live_minus_disabled"),
        },
    }
    persistence = {}
    for t, obj in targets.items():
        persistence[t] = {
            "ewok": adjacent_persistence_ewok(obj["ewok_rows"], final_layer),
            "globalpiqa_hard52": adjacent_persistence_gp(obj["gp_rows"], final_layer),
        }
    interpretation = (
        "The direct frozen-head readout finds frequent intermediate decodability on final-wrong rows, especially GlobalPIQA hard52 rank/correctness and EWoK interaction positivity, but this is not yet a clean late-erasure mechanism: the same phenomenon appears in both matched base and scale1.75, and paired EWoK deltas show scale1.75 is already worse than the matched base across most middle/final layers rather than losing a uniquely correct late signal. The live adapter contributes some final GlobalPIQA improvement over disabled inference, but its EWoK effect is small and mixed; disabling does not restore the matched base trajectory. This supports representation/trajectory formation as the main deficit unless a later decoder-robust alignment check shows persistent, adjacent-layer correct signals erased at a specific late boundary."
    )
    out = {
        "status": "MIDLAYER_SYNTHESIS_DONE",
        "inputs": {t: {"root": rel(obj["root"]), "summary": rel(obj["root"] / SUMMARY_INNER)} for t, obj in targets.items()},
        "endpoint_reproduction": endpoint_reproduction(targets),
        "best_layers": best_layers(summaries),
        "paired_deltas": paired,
        "adjacent_persistence": persistence,
        "interpretation": interpretation,
    }
    summary_path = OUT / "midlayer_synthesis.json"
    out["summary_json"] = rel(summary_path)
    out["note"] = rel(NOTE)
    write_json(summary_path, out)
    make_note(out)
    print(json.dumps({"status": out["status"], "summary": rel(summary_path), "note": rel(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
