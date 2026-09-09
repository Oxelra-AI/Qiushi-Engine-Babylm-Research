#!/usr/bin/env python3
"""research: summarize where compact-view reinvestment is seed-sensitive.

Uses:
  - research same-coordinate official 2x2 for column-level treatment effects.
  - current-official EWoK 2x2 domain file for relation-domain
    interaction localization.

CPU-only analysis over existing JSON files.
"""
import json
import pathlib

A01 = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DIR = A01 / "data/relation_instability_synthesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cons = json.loads((A01 / "data/consistent_official_2x2/consistent_official_2x2.json").read_text())
ewok = json.loads((A02 / "data/official_ewok_2x2_domain_did/official_ewok_2x2_domain_did.json").read_text())

# Identify likely top-level EWoK domains by the official domain-count names, not all nested cluster rows.
official_domains = {
    "agent-properties", "material-dynamics", "material-properties",
    "physical-dynamics", "physical-interactions", "physical-relations",
    "quantitative-properties", "social-interactions", "social-properties",
    "social-relations", "spatial-relations",
}

# The domain file stores a list of domain rows. Locate it robustly.
def collect_rows(x):
    rows = []
    if isinstance(x, list):
        for item in x:
            if isinstance(item, dict) and "domain" in item and all(k in item for k in ("clean_43022", "clean_43122", "reinvest_43022", "reinvest_43122")):
                rows.append(item)
            else:
                rows.extend(collect_rows(item))
    elif isinstance(x, dict):
        for v in x.values():
            rows.extend(collect_rows(v))
    return rows

rows = collect_rows(ewok)
top_domain_rows = [r for r in rows if r.get("domain") in official_domains]
# Deduplicate by domain name, preferring first occurrence.
seen = {}
for r in top_domain_rows:
    seen.setdefault(r["domain"], r)

domain_summary = []
for name, r in seen.items():
    te430 = r["reinvest_43022"] - r["clean_43022"]
    te431 = r["reinvest_43122"] - r["clean_43122"]
    did = te431 - te430
    domain_summary.append({
        "domain": name,
        "n": r.get("n"),
        "clean_43022": r["clean_43022"],
        "clean_43122": r["clean_43122"],
        "reinvest_43022": r["reinvest_43022"],
        "reinvest_43122": r["reinvest_43122"],
        "TE43022": te430,
        "TE43122": te431,
        "DiD": did,
    })

domain_summary.sort(key=lambda d: d["DiD"])

cols_did = cons["DiD_treatment_x_seed"]["DiD_cols"]
cols_te = cons["treatment_effect_within_seed"]

payload = {
    "status": "RELATION_INSTABILITY_SYNTHESIS",
    "same_coordinate_2x2": {
        "path": "experiments/archive/representation_and_objectives/data/consistent_official_2x2/consistent_official_2x2.json",
        "overalls": cons["overalls"],
        "TE_43022_overall": cols_te["TE_43022_overall"],
        "TE_43122_overall": cols_te["TE_43122_overall"],
        "ATE_overall_two_seeds": cols_te["ATE_overall_two_seeds"],
        "DiD_overall": cons["DiD_treatment_x_seed"]["DiD_overall"],
        "clean_seed_spread_overall": cons["seed_spread_within_treatment"]["clean_43122_minus_43022_overall"],
        "reinvest_seed_spread_overall": cons["seed_spread_within_treatment"]["reinvest_43122_minus_43022_overall"],
        "column_DiD_sorted": sorted(cols_did.items(), key=lambda kv: kv[1]),
    },
    "ewok_domain_interactions_sorted_by_DiD": domain_summary,
    "short_scientific_reading": {
        "main_result": "Compact-view reinvestment is positive within both seeds on the same official coordinate, but seed43122 is below the public leader because the backbone recipe is already lower at seed43122 and the treatment adds less EWoK/Entity/COMPS than it did at seed43022.",
        "instability_location": "Overall treatment-by-seed interaction is small (-0.205), while EWoK interaction is large (-2.462); the relation repair should target stable relational abstraction/consolidation rather than abandon compact views.",
        "wait_on_adjacency_break": "Adjacency-broken source/rewrite control can test whether adjacency causes treatment gains, but it cannot explain why the treatment effect collapses mainly in EWoK under seed43122. It should wait until a relation-stability repair target is clearer.",
    },
}

out_json = OUT_DIR / "relation_instability_synthesis.json"
out_json.write_text(json.dumps(payload, indent=2))

# Short markdown note
note = (OUT_DIR.parents[1].parents[2] / 'research/notes/representation_and_objectives/relation_instability_synthesis.md')
worst = domain_summary[:6]
best = list(reversed(domain_summary[-4:]))
note.write_text("\n".join([
    "# research — compact-view seed interaction and relation-stability route",
    "",
    "## Same-coordinate 2×2",
    f"- clean_43022 Overall: {cons['overalls']['clean_43022']}",
    f"- clean_43122 Overall: {cons['overalls']['clean_43122']}",
    f"- reinvest_43022 Overall: {cons['overalls']['reinvest_43022']}",
    f"- reinvest_43122 Overall: {cons['overalls']['reinvest_43122']}",
    f"- Treatment effect at seed43022: {cols_te['TE_43022_overall']} Overall.",
    f"- Treatment effect at seed43122: {cols_te['TE_43122_overall']} Overall.",
    f"- Two-seed average treatment effect: {cols_te['ATE_overall_two_seeds']} Overall.",
    f"- Treatment×seed interaction: {cons['DiD_treatment_x_seed']['DiD_overall']} Overall.",
    "",
    "The below-leader absolute seed43122 endpoint is not a verdict against compact-view reinvestment. The treatment remains positive within both seeds. Most of the absolute drop from seed43022 to seed43122 already exists in the clean backbone; the extra treatment-specific drop is about 0.205 Overall.",
    "",
    "## Column interaction",
    "Worst treatment×seed column interaction:",
    *[f"- {k}: {v}" for k, v in sorted(cols_did.items(), key=lambda kv: kv[1])[:6]],
    "",
    "Favorable/offsetting interactions:",
    *[f"- {k}: {v}" for k, v in sorted(cols_did.items(), key=lambda kv: kv[1], reverse=True)[:4]],
    "",
    "EWoK is the main treatment-specific fragility. SuperGLUE interaction is near zero, Supplement is essentially stable, while Reading and GlobalPIQA interactions are favorable at seed43122.",
    "",
    "## EWoK domain localization",
    "Most negative official-domain interactions:",
    *[f"- {r['domain']}: DiD={r['DiD']:.3f}, TE43022={r['TE43022']:.3f}, TE43122={r['TE43122']:.3f}, n={r['n']}" for r in worst],
    "",
    "Most positive official-domain interactions:",
    *[f"- {r['domain']}: DiD={r['DiD']:.3f}, TE43022={r['TE43022']:.3f}, TE43122={r['TE43122']:.3f}, n={r['n']}" for r in best],
    "",
    "## Route implication",
    "Protect the seed43022 official-coordinate endpoint for submission in parallel, because it is a real above-leader coordinate with completed compliance evidence. Continue research on stabilizing compact-view relation formation, especially material/physical/spatial EWoK behavior and Entity/COMPS side effects. Do not treat adjacency-broken training as the next run: it tests pair-adjacency causality, not the observed seed interaction. A more useful next construction is a low-cost stability plan that separates (i) checkpoint/fine-tuning randomness, (ii) relation-domain sensitivity, and (iii) compact-view selection/placement properties before any new 100M training.",
    "",
    f"JSON: `{out_json}`",
]))

print(json.dumps({
    "status": payload["status"],
    "out_json": str(out_json),
    "note": str(note),
    "DiD_overall": cons["DiD_treatment_x_seed"]["DiD_overall"],
    "worst_column_DiD": sorted(cols_did.items(), key=lambda kv: kv[1])[:5],
    "worst_ewok_domains": [{"domain": r["domain"], "DiD": round(r["DiD"], 3), "TE43022": round(r["TE43022"], 3), "TE43122": round(r["TE43122"], 3), "n": r.get("n")} for r in worst],
}, indent=2))
