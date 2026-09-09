#!/usr/bin/env python3
"""research collation: bridge-fixed decoder/alignment band, official comparative readout.

This collator intentionally fixes the decoder mode and adjacent-depth equal blend
using the non-official research naturalistic bridge. EWoK and GlobalPIQA are then
used only as readouts under that fixed rule. It also reports target-swap/rotated
label nulls and adjacent persistence to avoid treating a single best official
layer as an architecture selector.
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
    "matched_base_80M": A01_WS / "data/decoder_alignment_base80",
    "scale1p75_live_80M": A01_WS / "data/decoder_alignment_scale80",
    "scale1p75_disabled_80M": A01_WS / "data/decoder_alignment_disabled80",
}
OUT = A01_WS / "data/decoder_alignment_synthesis"
NOTE = A01_WS / "notes/decoder_robust_alignment_synthesis.md"


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


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
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def load_targets() -> dict[str, dict[str, Any]]:
    out = {}
    for target, root in ROOTS.items():
        tdir = root / target
        summ = tdir / "decoder_alignment_target_summary.json"
        if not summ.exists():
            continue
        obj = {"root": root, "tdir": tdir, "summary": read_json(summ)}
        for name in ["bridge", "ewok", "globalpiqa"]:
            path = tdir / f"{name}_layer_rows.csv"
            obj[f"{name}_rows"] = read_csv(path) if path.exists() else []
        out[target] = obj
    return out


# ---------------------- bridge-fixed selector ----------------------


def bridge_band_row(r0: dict[str, str], r1: dict[str, str]) -> dict[str, Any]:
    # Equal-score blend across adjacent depths; this is fixed before official readout.
    vals = {}
    for key in ["s_AB_alt0", "s_AB_alt1", "s_BA_alt0", "s_BA_alt1", "s_erased_alt0", "s_erased_alt1"]:
        vals[key] = 0.5 * (f(r0[key]) + f(r1[key]))
    if r0["correct_AB"] == r0["alt_0"]:
        margin_AB = vals["s_AB_alt0"] - vals["s_AB_alt1"]
        margin_BA = vals["s_BA_alt1"] - vals["s_BA_alt0"]
        erased_bias = vals["s_erased_alt0"] - vals["s_erased_alt1"]
    else:
        margin_AB = vals["s_AB_alt1"] - vals["s_AB_alt0"]
        margin_BA = vals["s_BA_alt0"] - vals["s_BA_alt1"]
        erased_bias = vals["s_erased_alt1"] - vals["s_erased_alt0"]
    return {
        "target": r0["target"], "decoder_mode": r0["decoder_mode"], "band": f"L{r0['layer_index']}-L{r1['layer_index']}", "band_start": int(r0["layer_index"]), "band_end": int(r1["layer_index"]),
        "pair_id": r0["pair_id"], "family": r0.get("family"), "template_id": r0.get("template_id"),
        "margin_AB": margin_AB, "margin_BA": margin_BA, "four_cell_M": margin_AB + margin_BA, "erased_bias": erased_bias,
        "both_correct": margin_AB > 0 and margin_BA > 0,
        "swap_both_correct": margin_AB < 0 and margin_BA < 0,
        "persistent_both_correct": b(r0.get("both_correct")) and b(r1.get("both_correct")),
        "persistent_swap_correct": b(r0.get("swap_both_correct")) and b(r1.get("swap_both_correct")),
        "min_correct_margin": min(margin_AB, margin_BA),
    }


def make_adjacent_bridge_band_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for r in rows:
        by[(r["target"], r["decoder_mode"], r["pair_id"])][int(r["layer_index"])] = r
    out = []
    for key, byli in by.items():
        for li in sorted(byli):
            if li + 1 in byli:
                out.append(bridge_band_row(byli[li], byli[li+1]))
    return out


def selector_table(targets: dict[str, dict[str, Any]], selector_targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    band_rows = []
    for t in selector_targets:
        if t in targets:
            band_rows.extend(make_adjacent_bridge_band_rows(targets[t]["bridge_rows"]))
    groups = defaultdict(list)
    for r in band_rows:
        groups[(r["decoder_mode"], r["band_start"], r["band_end"])].append(r)
    table = []
    for (mode, lo, hi), rs in sorted(groups.items()):
        n = len(rs)
        actual = sum(r["both_correct"] for r in rs)
        swap = sum(r["swap_both_correct"] for r in rs)
        pers = sum(r["persistent_both_correct"] for r in rs)
        pswap = sum(r["persistent_swap_correct"] for r in rs)
        row = {
            "decoder_mode": mode, "band_start": lo, "band_end": hi, "band": f"L{lo}-L{hi}", "n": n,
            "band_both_correct": actual, "band_swap_both_correct": swap, "band_both_correct_frac": actual / n if n else None, "band_swap_frac": swap / n if n else None,
            "persistent_both_correct": pers, "persistent_swap_correct": pswap, "persistent_both_correct_frac": pers / n if n else None, "persistent_swap_frac": pswap / n if n else None,
            "actual_minus_swap_persistent": pers - pswap, "actual_minus_swap_band": actual - swap,
            "M_mean": statistics.fmean([r["four_cell_M"] for r in rs]) if rs else None,
            "min_correct_margin_actual": qstats(r["min_correct_margin"] for r in rs if r["both_correct"]),
        }
        # Non-official selector score: persistent separation is primary; equal-band separation secondary.
        row["selector_score"] = (pers - pswap) + 0.25 * (actual - swap)
        table.append(row)
    table.sort(key=lambda r: (r["selector_score"], r["actual_minus_swap_persistent"], r["band_both_correct_frac"] or -1, -r["band_start"]), reverse=True)
    selected = table[0] if table else {}
    return table, selected


# ---------------------- EWoK fixed readout ----------------------


def ewok_band_rows(rows: list[dict[str, str]], mode: str, lo: int, hi: int) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for r in rows:
        if r["decoder_mode"] == mode:
            by[r["global_index"]][int(r["layer_index"])] = r
    out = []
    for idx, byli in by.items():
        if lo not in byli or hi not in byli:
            continue
        r0, r1 = byli[lo], byli[hi]
        row = {k: r0.get(k) for k in ["target", "global_index", "domain", "local_index", "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB", "buckets"]}
        row.update({"decoder_mode": mode, "band": f"L{lo}-L{hi}", "band_start": lo, "band_end": hi})
        vals = {}
        for k in ["s11_sum", "s21_sum", "s12_sum", "s22_sum"]:
            vals[k] = 0.5 * (f(r0[k]) + f(r1[k]))
            row[k] = vals[k]
        s11, s21, s12, s22 = vals["s11_sum"], vals["s21_sum"], vals["s12_sum"], vals["s22_sum"]
        m1 = s11 - s21; m2 = s22 - s12; wc1 = s11 - s12; wc2 = s22 - s21; inter = (s11 + s22) - (s12 + s21)
        row.update({"official_margin_t1_sum": m1, "official_margin_t2_sum": m2, "within_context_margin_c1_sum": wc1, "within_context_margin_c2_sum": wc2, "interaction_sum": inter,
                    "t1_correct": m1 > 0, "t2_correct": m2 > 0, "both_official_positive": m1 > 0 and m2 > 0, "both_swapped_positive": m1 < 0 and m2 < 0, "interaction_positive": inter > 0, "both_within_context_positive": wc1 > 0 and wc2 > 0})
        row["stable_conditional_failure"] = bool((not row["t1_correct"]) and (not row["interaction_positive"]) and (not row["both_within_context_positive"]))
        row["persistent_both_official"] = b(r0.get("both_official_positive")) and b(r1.get("both_official_positive"))
        row["persistent_swapped"] = b(r0.get("both_swapped_positive")) and b(r1.get("both_swapped_positive"))
        out.append(row)
    return out


def final_ewok_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    max_li = max(int(r["layer_index"]) for r in rows if r["decoder_mode"] == "raw")
    return {r["global_index"]: r for r in rows if r["decoder_mode"] == "raw" and int(r["layer_index"]) == max_li}


def summarize_ewok_fixed(band: list[dict[str, Any]], final: dict[str, dict[str, str]]) -> dict[str, Any]:
    n = len(band)
    final_wrong_band = [r for r in band if r["global_index"] in final and not b(final[r["global_index"]].get("both_official_positive"))]
    final_stable_band = [r for r in band if r["global_index"] in final and b(final[r["global_index"]].get("stable_conditional_failure"))]
    return {
        "n": n,
        "band_both_official_frac": summarize_bool(band, "both_official_positive"),
        "band_swapped_frac": summarize_bool(band, "both_swapped_positive"),
        "persistent_both_official_frac": summarize_bool(band, "persistent_both_official"),
        "persistent_swapped_frac": summarize_bool(band, "persistent_swapped"),
        "t1_accuracy": summarize_bool(band, "t1_correct"),
        "stable_failure_frac": summarize_bool(band, "stable_conditional_failure"),
        "interaction": qstats(r["interaction_sum"] for r in band),
        "official_margin_t1": qstats(r["official_margin_t1_sum"] for r in band),
        "final_not_both_band_both": sum(r["both_official_positive"] for r in final_wrong_band),
        "final_not_both_band_persistent_both": sum(r["persistent_both_official"] for r in final_wrong_band),
        "final_stable_band_both": sum(r["both_official_positive"] for r in final_stable_band),
        "final_stable_band_persistent_both": sum(r["persistent_both_official"] for r in final_stable_band),
        "target_swap_persistent_count": sum(r["persistent_swapped"] for r in band),
        "target_swap_band_count": sum(r["both_swapped_positive"] for r in band),
    }


def summarize_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if b(r.get(key))) / len(rows)


# ---------------------- GlobalPIQA fixed readout ----------------------


def gp_band_rows(rows: list[dict[str, str]], mode: str, lo: int, hi: int) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for r in rows:
        if r["decoder_mode"] == mode:
            by[r["example_id"]][int(r["layer_index"])] = r
    out = []
    for uid, byli in by.items():
        if lo not in byli or hi not in byli:
            continue
        r0, r1 = byli[lo], byli[hi]
        scores0 = [float(x) for x in json.loads(r0["scores_json"])]
        scores1 = [float(x) for x in json.loads(r1["scores_json"])]
        scores = [0.5 * (a + b) for a, b in zip(scores0, scores1)]
        lab = int(r0["label"]); rot = int(r0.get("rotated_label", (lab + 1) % len(scores)))
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        choice = order[0]
        row = {"target": r0["target"], "decoder_mode": mode, "band": f"L{lo}-L{hi}", "band_start": lo, "band_end": hi, "example_id": uid, "is_hard52": b(r0.get("is_hard52")), "label": lab, "rotated_label": rot, "choice": choice,
               "true_correct": choice == lab, "rotated_correct": choice == rot, "true_rank": order.index(lab) + 1, "rotated_rank": order.index(rot) + 1,
               "top_minus_true": scores[choice] - scores[lab], "top_minus_rotated": scores[choice] - scores[rot],
               "true_minus_best_incorrect": scores[lab] - max(scores[j] for j in range(len(scores)) if j != lab),
               "rotated_minus_best_other": scores[rot] - max(scores[j] for j in range(len(scores)) if j != rot),
               "scores_json": json.dumps(scores, ensure_ascii=False), "prompt": r0.get("prompt"), "completions_json": r0.get("completions_json"),
               "persistent_true_correct": b(r0.get("true_correct")) and b(r1.get("true_correct")),
               "persistent_rotated_correct": b(r0.get("rotated_correct")) and b(r1.get("rotated_correct"))}
        out.append(row)
    return out


def final_gp_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    max_li = max(int(r["layer_index"]) for r in rows if r["decoder_mode"] == "raw")
    return {r["example_id"]: r for r in rows if r["decoder_mode"] == "raw" and int(r["layer_index"]) == max_li}


def summarize_gp_fixed(band: list[dict[str, Any]], final: dict[str, dict[str, str]]) -> dict[str, Any]:
    hard = [r for r in band if b(r.get("is_hard52"))]
    final_wrong = [r for r in band if r["example_id"] in final and not b(final[r["example_id"]].get("true_correct"))]
    hard_final_wrong = [r for r in final_wrong if b(r.get("is_hard52"))]
    def sub(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {"n": len(rs), "true_accuracy": summarize_bool(rs, "true_correct"), "rotated_accuracy": summarize_bool(rs, "rotated_correct"), "persistent_true_frac": summarize_bool(rs, "persistent_true_correct"), "persistent_rotated_frac": summarize_bool(rs, "persistent_rotated_correct"), "top_minus_true": qstats(r["top_minus_true"] for r in rs), "true_minus_best_incorrect": qstats(r["true_minus_best_incorrect"] for r in rs), "true_rank_counts": dict(Counter(str(r["true_rank"]) for r in rs))}
    return {"all": sub(band), "hard52": sub(hard), "final_wrong_band_true": sum(r["true_correct"] for r in final_wrong), "final_wrong_persistent_true": sum(r["persistent_true_correct"] for r in final_wrong), "hard52_final_wrong_band_true": sum(r["true_correct"] for r in hard_final_wrong), "hard52_final_wrong_persistent_true": sum(r["persistent_true_correct"] for r in hard_final_wrong), "hard52_persistent_rotated_count": sum(r["persistent_rotated_correct"] for r in hard)}


# ---------------------- paired official deltas ----------------------


def paired_ewok_delta(a: list[dict[str, Any]], b_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    amap = {r["global_index"]: r for r in a}; bmap = {r["global_index"]: r for r in b_rows}
    common = sorted(set(amap) & set(bmap))
    return {"label": label, "n": len(common),
            "delta_interaction_b_minus_a": qstats(f(bmap[i]["interaction_sum"]) - f(amap[i]["interaction_sum"]) for i in common),
            "net_both_official_b_minus_a": sum(int(b(bmap[i]["both_official_positive"])) - int(b(amap[i]["both_official_positive"])) for i in common),
            "net_persistent_both_b_minus_a": sum(int(b(bmap[i]["persistent_both_official"])) - int(b(amap[i]["persistent_both_official"])) for i in common),
            "net_stable_failure_b_minus_a": sum(int(b(bmap[i]["stable_conditional_failure"])) - int(b(amap[i]["stable_conditional_failure"])) for i in common),
            "b_better_interaction_n": sum(f(bmap[i]["interaction_sum"]) > f(amap[i]["interaction_sum"]) for i in common),
            "a_better_interaction_n": sum(f(bmap[i]["interaction_sum"]) < f(amap[i]["interaction_sum"]) for i in common)}


def paired_gp_delta(a: list[dict[str, Any]], b_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    amap = {r["example_id"]: r for r in a}; bmap = {r["example_id"]: r for r in b_rows}
    common_all = sorted(set(amap) & set(bmap))
    common_hard = [i for i in common_all if b(amap[i].get("is_hard52"))]
    def sub(ids: list[str]) -> dict[str, Any]:
        return {"n": len(ids),
                "delta_top_minus_true_b_minus_a": qstats(f(bmap[i]["top_minus_true"]) - f(amap[i]["top_minus_true"]) for i in ids),
                "net_true_correct_b_minus_a": sum(int(b(bmap[i]["true_correct"])) - int(b(amap[i]["true_correct"])) for i in ids),
                "net_persistent_true_b_minus_a": sum(int(b(bmap[i]["persistent_true_correct"])) - int(b(amap[i]["persistent_true_correct"])) for i in ids),
                "net_rotated_correct_b_minus_a": sum(int(b(bmap[i]["rotated_correct"])) - int(b(amap[i]["rotated_correct"])) for i in ids),
                "b_lower_margin_n": sum(f(bmap[i]["top_minus_true"]) < f(amap[i]["top_minus_true"]) for i in ids),
                "a_lower_margin_n": sum(f(bmap[i]["top_minus_true"]) > f(amap[i]["top_minus_true"]) for i in ids)}
    return {"label": label, "all": sub(common_all), "hard52": sub(common_hard)}


def make_note(out: dict[str, Any]) -> None:
    lines = []
    lines.append("# research decoder-robust alignment synthesis\n\n")
    lines.append("No model weights were updated. Each target fitted layerwise statistic/ridge maps on legal-corpus masked states only; the adjacent-depth/equal-blend readout was selected on the non-official research naturalistic bridge, then applied unchanged to EWoK and GlobalPIQA.\n\n")
    sel = out.get("selected_bridge_rule", {})
    lines.append(f"Selected bridge-fixed rule: decoder_mode={sel.get('decoder_mode')} band={sel.get('band')} selector_score={sel.get('selector_score')} persistent actual-swap={sel.get('actual_minus_swap_persistent')} band actual-swap={sel.get('actual_minus_swap_band')}.\n\n")
    lines.append("## Final-head and alignment checks\n\n")
    for t, chk in out.get("target_checks", {}).items():
        lines.append(f"- `{t}`: final-head max |logit diff|={chk.get('final_head_max_abs_logit')}, target logprob diff={chk.get('final_head_max_abs_target_logprob')}; adapter placement={chk.get('adapter_state_placement')}.\n")
    lines.append("\n## Fixed official readout\n\n")
    for t, s in out.get("fixed_readouts", {}).items():
        ew = s.get("ewok_summary", {})
        gp = s.get("globalpiqa_summary", {})
        lines.append(f"- `{t}` EWoK: both={ew.get('band_both_official_frac')}, persistent both={ew.get('persistent_both_official_frac')}, swapped null={ew.get('persistent_swapped_frac')}, stable={ew.get('stable_failure_frac')}, final-not-both persistent recovered={ew.get('final_not_both_band_persistent_both')}.\n")
        if gp:
            lines.append(f"  GlobalPIQA hard52: true acc={gp.get('hard52',{}).get('true_accuracy')}, persistent true={gp.get('hard52',{}).get('persistent_true_frac')}, rotated null={gp.get('hard52',{}).get('persistent_rotated_frac')}, final-wrong persistent recovered={gp.get('hard52_final_wrong_persistent_true')}.\n")
    lines.append("\n## Paired deltas\n\n")
    for k, v in out.get("paired_deltas", {}).items():
        lines.append(f"- `{k}` EWoK net persistent both={v.get('ewok',{}).get('net_persistent_both_b_minus_a')}, net stable={v.get('ewok',{}).get('net_stable_failure_b_minus_a')}; GP hard52 net persistent true={v.get('globalpiqa',{}).get('hard52',{}).get('net_persistent_true_b_minus_a')}, delta top-minus mean={v.get('globalpiqa',{}).get('hard52',{}).get('delta_top_minus_true_b_minus_a',{}).get('mean')}.\n")
    lines.append("\nInterpretation should be based on whether the bridge-fixed aligned band shows persistent official-row recovery beyond target-swap/rotated-label nulls and whether scale1.75 specifically loses such rows late relative to the matched base.\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    targets = load_targets()
    if "matched_base_80M" not in targets or "scale1p75_live_80M" not in targets:
        missing = [t for t in ["matched_base_80M", "scale1p75_live_80M"] if t not in targets]
        raise FileNotFoundError(f"Missing required research target outputs: {missing}")

    selector_targets = ["matched_base_80M", "scale1p75_live_80M"]
    table, selected = selector_table(targets, selector_targets)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "bridge_selector_table.csv", table)
    if not selected:
        raise RuntimeError("No bridge selector candidates")
    mode = selected["decoder_mode"]; lo = int(selected["band_start"]); hi = int(selected["band_end"])

    fixed = {}
    fixed_rows_paths = {}
    for t, obj in targets.items():
        ew_band = ewok_band_rows(obj["ewok_rows"], mode, lo, hi) if obj.get("ewok_rows") else []
        gp_band = gp_band_rows(obj["globalpiqa_rows"], mode, lo, hi) if obj.get("globalpiqa_rows") else []
        ew_final = final_ewok_rows(obj["ewok_rows"]) if obj.get("ewok_rows") else {}
        gp_final = final_gp_rows(obj["globalpiqa_rows"]) if obj.get("globalpiqa_rows") else {}
        write_csv(OUT / f"{t}_fixed_ewok_band_rows.csv", ew_band)
        write_csv(OUT / f"{t}_fixed_globalpiqa_band_rows.csv", gp_band)
        fixed[t] = {"ewok_summary": summarize_ewok_fixed(ew_band, ew_final) if ew_band else {}, "globalpiqa_summary": summarize_gp_fixed(gp_band, gp_final) if gp_band else {}, "ewok_rows_csv": rel(OUT / f"{t}_fixed_ewok_band_rows.csv"), "globalpiqa_rows_csv": rel(OUT / f"{t}_fixed_globalpiqa_band_rows.csv")}
        fixed_rows_paths[t] = {"ewok": OUT / f"{t}_fixed_ewok_band_rows.csv", "gp": OUT / f"{t}_fixed_globalpiqa_band_rows.csv"}

    paired = {}
    base_ew = read_csv(fixed_rows_paths["matched_base_80M"]["ewok"])
    base_gp = read_csv(fixed_rows_paths["matched_base_80M"]["gp"])
    for other in ["scale1p75_live_80M", "scale1p75_disabled_80M"]:
        if other not in fixed_rows_paths:
            continue
        oth_ew = read_csv(fixed_rows_paths[other]["ewok"])
        oth_gp = read_csv(fixed_rows_paths[other]["gp"])
        paired[f"{other}_minus_matched_base"] = {"ewok": paired_ewok_delta(base_ew, oth_ew, f"{other}_minus_matched_base"), "globalpiqa": paired_gp_delta(base_gp, oth_gp, f"{other}_minus_matched_base")}
    if "scale1p75_disabled_80M" in fixed_rows_paths:
        live_ew = read_csv(fixed_rows_paths["scale1p75_live_80M"]["ewok"])
        live_gp = read_csv(fixed_rows_paths["scale1p75_live_80M"]["gp"])
        dis_ew = read_csv(fixed_rows_paths["scale1p75_disabled_80M"]["ewok"])
        dis_gp = read_csv(fixed_rows_paths["scale1p75_disabled_80M"]["gp"])
        paired["scale1p75_live_minus_disabled"] = {"ewok": paired_ewok_delta(dis_ew, live_ew, "scale1p75_live_minus_disabled"), "globalpiqa": paired_gp_delta(dis_gp, live_gp, "scale1p75_live_minus_disabled")}

    checks = {}
    for t, obj in targets.items():
        summ = obj["summary"]
        checks[t] = {"final_head_max_abs_logit": summ.get("final_head_equivalence", {}).get("max_abs_logit"), "final_head_max_abs_target_logprob": summ.get("final_head_equivalence", {}).get("max_abs_target_logprob"), "adapter_state_placement": summ.get("adapter_state_placement"), "summary": rel(obj["tdir"] / "decoder_alignment_target_summary.json")}

    out = {"status": "DECODER_ALIGNMENT_SYNTHESIS_DONE", "selector_scope": "decoder mode and adjacent equal-blend band selected only from research naturalistic bridge across matched_base_80M and scale1p75_live_80M; official EWoK/GlobalPIQA not used for selection", "selector_table_csv": rel(OUT / "bridge_selector_table.csv"), "selected_bridge_rule": selected, "target_checks": checks, "fixed_readouts": fixed, "paired_deltas": paired, "output_root": rel(OUT), "note": rel(NOTE)}
    write_json(OUT / "decoder_alignment_synthesis.json", out)
    make_note(out)
    print(json.dumps({"status": out["status"], "summary": rel(OUT / "decoder_alignment_synthesis.json"), "note": rel(NOTE), "selected_bridge_rule": selected}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
