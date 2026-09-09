#!/usr/bin/env python3
"""research: per-subtask decomposition of the legal-tokenizer deficit.

Scientific purpose
------------------
The best fully-legal endpoint (research 16k tokenizer, Overall 41.2578) sits below the
old non-submittable inherited-tokenizer reference (Overall 42.0331). The deficit is NOT a simple spatial-relation loss: at the column level,
spatial-relations actually improved while material/physical dynamics and QA-congruence
weakened. To let the mature clean-vs-reinvest trajectory pick the right
intervention family, we need a benchmark-independent, UID-level map of exactly WHERE the
legal representation loses (and gains) strength, and whether that map is consistent
across two INDEPENDENTLY built legal 16k tokenizers.

This script parses the `best_temperature_report.txt` UID accuracy tables for four
endpoints, all on the identical 100M reinvest stream (SHA 3dd19f...), identical
architecture/seeds/recipe, differing only in the tokenizer:

  old_ref_42033   : inherited GPT2-Strict tokenizer (NON-SUBMITTABLE reference)
                    compact_view_reinvest full eval  (Overall 42.0331)
  legal_step35    : same-pool 16k tokenizer (best legal, Overall 41.2578)
  legal_bytealpha : byte-alphabet 16k tokenizer (Overall 40.7040)
  legal_a01_ss16k : strictsmalltok independent 16k (partial: no EWoK/SGLUE/AoA)

It computes, per UID subtask in BLiMP / Supplement / EWoK:
  - accuracy at each endpoint
  - delta = legal_endpoint - old_ref
and summarizes: which subtasks lose most, whether the two independent legal 16k
tokenizers (legal_step35 vs legal_a01_ss16k) agree on the loss pattern, and whether the
loss is concentrated in relation content (which relation-weighted masking would target)
or elsewhere (which it would not).

CPU-only. Reads existing eval artifacts. Does not train, evaluate, or change any corpus.
"""
import json
import re
import pathlib
import statistics

ROOT = pathlib.Path("Sessions")

# ---- endpoint report roots -------------------------------------------------
# Each entry: label -> dict of column -> best_temperature_report.txt path
A01_OLD = ROOT / "representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest"
A02_S35 = ROOT / "frontier_consolidation/data/compliant_full_eval/official_outputs/complianttok_reinvest_seed43022"
A02_BYTE = ROOT / "frontier_consolidation/data/bytealphatok_full_eval/official_outputs/bytealphatok_reinvest_seed43022"
A01_SS16K = ROOT / "representation_and_objectives/data/strictsmalltok_seed43022_full_eval/official_outputs/strictsmalltok_reinvest_seed43022"


def report_path(base, column):
    """Build the best_temperature_report.txt path under an endpoint output root."""
    mapping = {
        "BLiMP": ("BLiMP", "zero_shot/mlm/blimp/blimp_filtered"),
        "Supplement": ("Supplement", "zero_shot/mlm/blimp/supplement_filtered"),
        "EWoK": ("EWoK", "zero_shot/mlm/ewok/ewok_filtered"),
    }
    col_dir, tail = mapping[column]
    # discover the intermediate revision dir (contains "_%s" % column)
    top = base / col_dir / "chck_100M"
    if not top.exists():
        return None
    for child in top.iterdir():
        cand = child / tail / "best_temperature_report.txt"
        if cand.exists():
            return cand
    return None


def parse_uid_accuracy(path):
    """Return {uid: accuracy_float} from the '### UID ACCURACY' block."""
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    lines = text.splitlines()
    out = {}
    in_uid = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("###"):
            in_uid = s.upper().startswith("### UID ACCURACY")
            continue
        if in_uid and s:
            m = re.match(r"^(.+?):\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
    return out


def parse_average(path):
    if path is None or not path.exists():
        return None
    text = path.read_text()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("### AVERAGE ACCURACY"):
            for j in range(i + 1, min(i + 4, len(lines))):
                s = lines[j].strip()
                if re.match(r"^[-+]?\d+(?:\.\d+)?$", s):
                    return float(s)
    return None


ENDPOINTS = {
    "old_ref_42033": A01_OLD,
    "legal_step35": A02_S35,
    "legal_bytealpha": A02_BYTE,
    "legal_a01_ss16k": A01_SS16K,
}

# EWoK relation-like domains (what relation-weighted masking would amplify)
EWOK_RELATION_DOMAINS = {
    "spatial-relations", "physical-relations", "social-relations",
    "physical-dynamics", "material-dynamics", "physical-interactions",
    "social-interactions",
}
EWOK_PROPERTY_DOMAINS = {
    "agent-properties", "material-properties", "social-properties",
    "quantitative-properties",
}


def main():
    columns = ["BLiMP", "Supplement", "EWoK"]
    # Collect per-column, per-uid accuracies for each endpoint
    data = {}  # column -> {uid -> {endpoint -> acc}}
    averages = {}  # column -> {endpoint -> avg}
    for col in columns:
        data[col] = {}
        averages[col] = {}
        for label, base in ENDPOINTS.items():
            p = report_path(base, col)
            uid = parse_uid_accuracy(p)
            avg = parse_average(p)
            averages[col][label] = avg
            for u, a in uid.items():
                data[col].setdefault(u, {})[label] = a

    # Compute deltas vs old_ref for each legal endpoint
    result = {
        "status": "LEGAL_DEFICIT_SUBTASK_DECOMPOSITION",
        "reference": "old_ref_42033",
        "note": "All endpoints: identical 100M reinvest stream (SHA 3dd19f...), identical arch/seeds/recipe, tokenizer-only differences. old_ref is NON-SUBMITTABLE.",
        "column_averages": averages,
        "columns": {},
    }

    legal_labels = ["legal_step35", "legal_bytealpha", "legal_a01_ss16k"]

    for col in columns:
        col_rows = []
        for uid, accs in sorted(data[col].items()):
            ref = accs.get("old_ref_42033")
            row = {"uid": uid, "old_ref_42033": ref}
            for lab in legal_labels:
                v = accs.get(lab)
                row[lab] = v
                row[f"delta_{lab}"] = (round(v - ref, 4) if (v is not None and ref is not None) else None)
            col_rows.append(row)
        result["columns"][col] = col_rows

    # ---- Summaries ----------------------------------------------------------
    summary = {}

    # 1. Biggest legal_step35 losses/gains per column (uid-level)
    for col in columns:
        rows = [r for r in result["columns"][col] if r.get("delta_legal_step35") is not None]
        losses = sorted(rows, key=lambda r: r["delta_legal_step35"])[:8]
        gains = sorted(rows, key=lambda r: r["delta_legal_step35"], reverse=True)[:8]
        summary.setdefault("uid_extremes", {})[col] = {
            "biggest_losses": [(r["uid"], r["delta_legal_step35"]) for r in losses],
            "biggest_gains": [(r["uid"], r["delta_legal_step35"]) for r in gains],
        }

    # 2. Cross-tokenizer agreement: legal_step35 vs legal_a01_ss16k UID deltas
    #    (both independent legal 16k). Only BLiMP+Supplement (ss16k lacks EWoK).
    agree = {}
    for col in ["BLiMP", "Supplement"]:
        pairs = []
        for r in result["columns"][col]:
            d1 = r.get("delta_legal_step35")
            d2 = r.get("delta_legal_a01_ss16k")
            if d1 is not None and d2 is not None:
                pairs.append((r["uid"], d1, d2))
        if len(pairs) >= 3:
            xs = [p[1] for p in pairs]
            ys = [p[2] for p in pairs]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            pearson = cov / ((vx * vy) ** 0.5) if vx > 0 and vy > 0 else None
            # sign agreement fraction
            sign_agree = sum(1 for _, a, b in pairs if (a < 0) == (b < 0)) / len(pairs)
            agree[col] = {
                "n_uid": len(pairs),
                "mean_delta_step35": round(mx, 4),
                "mean_delta_ss16k": round(my, 4),
                "pearson_uid_deltas": (round(pearson, 4) if pearson is not None else None),
                "sign_agreement_frac": round(sign_agree, 4),
                "both_lose_uids": [u for u, a, b in pairs if a < 0 and b < 0],
            }
    summary["cross_tokenizer_agreement_step35_vs_ss16k"] = agree

    # 3. EWoK relation vs property mean delta (legal_step35)
    ewok_rows = result["columns"]["EWoK"]
    rel_deltas = [r["delta_legal_step35"] for r in ewok_rows
                  if r["uid"] in EWOK_RELATION_DOMAINS and r.get("delta_legal_step35") is not None]
    prop_deltas = [r["delta_legal_step35"] for r in ewok_rows
                   if r["uid"] in EWOK_PROPERTY_DOMAINS and r.get("delta_legal_step35") is not None]
    summary["ewok_relation_vs_property_step35"] = {
        "relation_domains": sorted(EWOK_RELATION_DOMAINS),
        "property_domains": sorted(EWOK_PROPERTY_DOMAINS),
        "mean_relation_delta": (round(statistics.mean(rel_deltas), 4) if rel_deltas else None),
        "mean_property_delta": (round(statistics.mean(prop_deltas), 4) if prop_deltas else None),
        "relation_domain_deltas": {r["uid"]: r["delta_legal_step35"] for r in ewok_rows
                                   if r["uid"] in EWOK_RELATION_DOMAINS},
        "property_domain_deltas": {r["uid"]: r["delta_legal_step35"] for r in ewok_rows
                                   if r["uid"] in EWOK_PROPERTY_DOMAINS},
    }

    result["summary"] = summary

    out_dir = ROOT / "frontier_consolidation/data/legal_deficit_subtask_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "legal_deficit_subtask_decomposition.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- Markdown -----------------------------------------------------------
    md = []
    md.append("# research legal-tokenizer deficit: UID-level decomposition\n")
    md.append("All endpoints share the identical 100M reinvest stream (SHA `3dd19f...`), "
              "architecture, seeds, and recipe. Only the tokenizer differs. `old_ref_42033` "
              "is the NON-SUBMITTABLE inherited-tokenizer reference (Overall 42.0331).\n")
    md.append("## Column averages\n")
    md.append("| column | old_ref_42033 | legal_step35 | legal_bytealpha | legal_a01_ss16k |")
    md.append("|---|---:|---:|---:|---:|")
    for col in columns:
        a = averages[col]
        def f(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "—"
        md.append(f"| {col} | {f(a.get('old_ref_42033'))} | {f(a.get('legal_step35'))} | "
                  f"{f(a.get('legal_bytealpha'))} | {f(a.get('legal_a01_ss16k'))} |")
    md.append("")

    md.append("## EWoK relation vs property (legal_step35 minus old_ref)\n")
    e = summary["ewok_relation_vs_property_step35"]
    md.append(f"- mean relation-domain delta: **{e['mean_relation_delta']}**")
    md.append(f"- mean property-domain delta: **{e['mean_property_delta']}**")
    md.append("\nRelation-domain deltas:")
    for k, v in sorted(e["relation_domain_deltas"].items(), key=lambda kv: (kv[1] if kv[1] is not None else 0)):
        md.append(f"  - {k}: {v}")
    md.append("\nProperty-domain deltas:")
    for k, v in sorted(e["property_domain_deltas"].items(), key=lambda kv: (kv[1] if kv[1] is not None else 0)):
        md.append(f"  - {k}: {v}")
    md.append("")

    md.append("## Cross-tokenizer agreement (two independent legal 16k tokenizers)\n")
    md.append("If the two independently built legal 16k tokenizers (A02 research vs A01 ss16k) "
              "lose on the SAME UID subtasks, the deficit is a representation property of the "
              "legal budget, not tokenizer noise.\n")
    for col, ag in agree.items():
        md.append(f"### {col}")
        md.append(f"- n_uid compared: {ag['n_uid']}")
        md.append(f"- mean delta research: {ag['mean_delta_step35']}, ss16k: {ag['mean_delta_ss16k']}")
        md.append(f"- Pearson(UID deltas): **{ag['pearson_uid_deltas']}**, "
                  f"sign agreement: **{ag['sign_agreement_frac']}**")
        md.append("")

    md.append("## Biggest legal_step35 UID losses/gains vs old_ref\n")
    for col in columns:
        ex = summary["uid_extremes"][col]
        md.append(f"### {col}")
        md.append("Losses: " + ", ".join(f"{u} ({d})" for u, d in ex["biggest_losses"]))
        md.append("Gains: " + ", ".join(f"{u} ({d})" for u, d in ex["biggest_gains"]))
        md.append("")

    md.append(f"\nFull JSON: `{out_json}`")
    out_md = out_dir / "legal_deficit_subtask_decomposition.md"
    out_md.write_text("\n".join(md))

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "column_averages": averages,
        "ewok_rel_vs_prop": {
            "mean_relation_delta": e["mean_relation_delta"],
            "mean_property_delta": e["mean_property_delta"],
        },
        "cross_tok_agreement": {c: {"pearson": ag["pearson_uid_deltas"],
                                    "sign_agree": ag["sign_agreement_frac"]}
                                for c, ag in agree.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
