#!/usr/bin/env python3
"""Integrate research compact ordered-vs-scrambled official and channel readouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
STABLE5 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
RELSTATE = ["EWoK", "Entity"]
CHECKPOINTS = ["chck_20M", "chck_40M"]
ARMS = ["ordered", "scrambled"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(scores: dict[str, float], cols: list[str]) -> float | None:
    vals = [scores.get(c) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def r6(x: float | None) -> float | None:
    return round(float(x), 6) if x is not None else None


def find_summary(eval_dir: Path, arm: str, ck: str) -> Path:
    target = f"compact_order_{arm}_{ck}"
    p = eval_dir / arm / ck / f"{target}_summary.json"
    if p.exists():
        return p
    hits = sorted((eval_dir / arm / ck).glob("*_summary.json")) if (eval_dir / arm / ck).exists() else []
    if not hits:
        raise FileNotFoundError(p)
    return hits[-1]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# research compact ordered-vs-scrambled integrated readout", ""]
    lines.append(payload["meaning"])
    lines.append("")
    lines.append("## Official-compatible selected cheap scores")
    for ck in CHECKPOINTS:
        if ck not in payload["official"]:
            continue
        d = payload["official"][ck]["ordered_minus_scrambled"]
        lines.append(f"### {ck}")
        lines.append("| metric | ordered-scrambled |")
        lines.append("|---|---:|")
        for k in ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_Entity", *CHEAP]:
            lines.append(f"| {k} | {d.get(k)} |")
        lines.append("")
    lines.append("## Source-absent channel probe")
    lines.append("Negative ordered_minus_scrambled means ordered training has lower NLL on the fixed ordered compact event. Because the fixed probe text is ordered, the source_absent_content row is mechanism-specific only when it is more ordered-favorable than retained_content and function_other.")
    for ck in CHECKPOINTS:
        cks = payload.get("channel", {}).get("summary", {}).get(ck, {})
        if not cks:
            continue
        lines.append(f"### {ck}")
        lines.append("| category | piece Δ | 95% CI | P(ordered better) |")
        lines.append("|---|---:|---:|---:|")
        for cat, recwrap in cks.items():
            rec = recwrap.get("ordered_minus_scrambled", {})
            ci = rec.get("piece_cluster_bootstrap", {})
            p_ordered_better = 1.0 - float(rec.get("p_piece_delta_gt0", 0.0))
            lines.append(f"| {cat} | {rec.get('piece_weighted_delta')} | [{ci.get('p025')}, {ci.get('p975')}] | {p_ordered_better:.4f} |")
        inter = payload.get("channel", {}).get("category_interactions", {}).get(ck, {}).get("piece_weighted_contrasts", {})
        if inter:
            lines.append("")
            lines.append("Category interaction: negative means source_absent_content has a larger ordered advantage than the comparison category.")
            lines.append("| contrast | Δ difference | 95% interval | P(diff<0) |")
            lines.append("|---|---:|---:|---:|")
            for name, rec in inter.items():
                ci = rec.get("bootstrap_interval", {})
                lines.append(f"| {name} | {rec.get('contrast_delta')} | [{ci.get('p025')}, {ci.get('p975')}] | {rec.get('p_contrast_lt0')} |")
        lines.append("")
    lines.append("## Readout")
    for k, v in payload["decision_readout"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if payload.get("loader_confound_summary"):
        lc = payload["loader_confound_summary"]
        lines.append("## Realized BPE/WWM differences to keep in interpretation")
        lines.append(f"Changed-block ordered-minus-scrambled: Δ active tokens {lc.get('delta_active_tokens_sum')}, Δ candidate groups {lc.get('delta_candidate_groups_sum')}, Δ view-active tokens {lc.get('delta_view_active_tokens_sum')}, CPU-proxy Δ masked tokens {lc.get('delta_masked_tokens_cpu_proxy_sum')}, Δ masked view/source {lc.get('delta_masked_view_tokens_cpu_proxy_sum')}/{lc.get('delta_masked_source_tokens_cpu_proxy_sum')}.")
        lines.append("")
    lines.append(f"JSON: `{payload['json_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--channel-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "compact_order_integrated_readout.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    official: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        official[ck] = {}
        for arm in ARMS:
            payload = read_json(find_summary(args.eval_dir, arm, ck))
            rec = payload.get("record", {})
            sc = rec.get("scores", {})
            official[ck][arm] = {
                "summary_path": str(find_summary(args.eval_dir, arm, ck)),
                "returncode": rec.get("returncode"),
                "cheap7": rec.get("cheap7"),
                "scores": sc,
                "cheap6_no_GlobalPIQA": avg(sc, STABLE6),
                "cheap5_no_GlobalPIQA_Reading": avg(sc, STABLE5),
                "EWoK_Entity": avg(sc, RELSTATE),
                "per_target": rec.get("per_target"),
            }
        os = official[ck]["ordered"]
        ss = official[ck]["scrambled"]
        deltas = {
            "cheap7": r6(float(os["cheap7"]) - float(ss["cheap7"])),
            "cheap6_no_GlobalPIQA": r6(os["cheap6_no_GlobalPIQA"] - ss["cheap6_no_GlobalPIQA"]),
            "cheap5_no_GlobalPIQA_Reading": r6(os["cheap5_no_GlobalPIQA_Reading"] - ss["cheap5_no_GlobalPIQA_Reading"]),
            "EWoK_Entity": r6(os["EWoK_Entity"] - ss["EWoK_Entity"]),
        }
        for col in CHEAP:
            deltas[col] = r6(float(os["scores"][col]) - float(ss["scores"][col]))
        official[ck]["ordered_minus_scrambled"] = deltas
    channel_path = args.channel_dir / "compact_order_channel_probe.json"
    if not channel_path.exists():
        raise FileNotFoundError(channel_path)
    channel = read_json(channel_path)
    decision: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        d = official[ck]["ordered_minus_scrambled"]
        src_rec = channel.get("summary", {}).get(ck, {}).get("source_absent_content", {}).get("ordered_minus_scrambled", {})
        src_delta = src_rec.get("piece_weighted_delta")
        src_ci = src_rec.get("piece_cluster_bootstrap", {})
        inter = channel.get("category_interactions", {}).get(ck, {}).get("piece_weighted_contrasts", {})
        controls_inter = inter.get("source_absent_minus_controls_mean", {})
        src_minus_controls = controls_inter.get("contrast_delta")
        p_src_lt_controls = controls_inter.get("p_contrast_lt0")
        src_selective = (src_minus_controls is not None and p_src_lt_controls is not None and float(src_minus_controls) <= -0.05 and float(p_src_lt_controls) >= 0.95)
        stable_official = (d.get("cheap6_no_GlobalPIQA") is not None and d.get("EWoK_Entity") is not None and float(d["cheap6_no_GlobalPIQA"]) >= 0.1 and float(d["EWoK_Entity"]) >= 0.0)
        decision[ck] = {
            "official_delta_cheap7": d.get("cheap7"),
            "official_delta_cheap6_no_GlobalPIQA": d.get("cheap6_no_GlobalPIQA"),
            "official_delta_cheap5_no_GlobalPIQA_Reading": d.get("cheap5_no_GlobalPIQA_Reading"),
            "official_delta_EWoK_Entity": d.get("EWoK_Entity"),
            "source_absent_ordered_minus_scrambled_nll": src_delta,
            "source_absent_95pct_interval": [src_ci.get("p025"), src_ci.get("p975")],
            "source_absent_ordered_better": (src_delta is not None and float(src_delta) < 0),
            "source_absent_minus_controls_mean_nll": src_minus_controls,
            "source_absent_minus_controls_mean_interval": [controls_inter.get("bootstrap_interval", {}).get("p025"), controls_inter.get("bootstrap_interval", {}).get("p975")],
            "p_source_absent_more_ordered_favorable_than_controls": p_src_lt_controls,
            "source_absent_category_selective": src_selective,
            "channel_interpretation": "source_absent_selective" if src_selective else "native_order_or_general_order_effect_not_source_absent_specific",
            "reading_for_maturation": "ordered-and-source-absent-selective" if (stable_official and src_selective) else "not-yet-justified",
        }
    lc = channel.get("loader_confound_summary")
    payload = {
        "status": "COMPACT_ORDER_INTEGRATED_READOUT",
        "meaning": "Combined readout for the matched compact ordered-vs-scrambled 40M experiment. Official-family deltas alone are not sufficient. Because the channel probe evaluates ordered compact text, the source_absent_content fixed-event row must show an ordered advantage materially larger than retained_content and function_other before it identifies the source-absent compact channel rather than native-order familiarity.",
        "official": official,
        "channel": channel,
        "loader_confound_summary": lc,
        "decision_readout": decision,
        "json_path": str(out_json),
    }
    write_json(out_json, payload)
    (args.out_dir / "compact_order_integrated_readout.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "decision_readout": decision}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
