#!/usr/bin/env python3
"""Summarize research private-pathway localization pockets into a compact note."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, time

ROOT = _public_path('.')
IN_JSON = _public_path('experiments/archive/representation_and_objectives/data/fastpath_private_localization/fastpath_private_localization.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/fastpath_private_pocket_summary.md')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/fastpath_private_localization/fastpath_private_pocket_summary.json')
DISCRETE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
FOCUS = {"Entity", "EWoK", "GlobalPIQA", "Supplement"}

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def score_group(g):
    return (
        int(g.get("private_gain_unique_vs_ordinary_and_shuffled", 0)),
        int(g.get("retained_anchor_correct_ordinary_wrong", 0)),
        int(g.get("coherent_minus_anchor_item_net", 0)),
        int(g.get("coherent_minus_ordinary_item_net", 0)),
    )

def main():
    j = json.loads(IN_JSON.read_text(encoding="utf-8"))
    loc = j["private_pathway_localization"]
    groups = []
    for col in DISCRETE:
        by = loc["by_column"][col]
        for tag in ["groups_best_by_unique_private_gain", "groups_best_by_retention_vs_ordinary", "groups_worst_by_coherent_vs_anchor_net", "groups_best_by_coherent_vs_anchor_net"]:
            for g in by.get(tag, []):
                g2 = dict(g); g2["column"] = col; g2["source_list"] = tag; groups.append(g2)
    # de-duplicate column/group records by keeping first with full metrics
    keyed = {}
    for g in groups:
        keyed.setdefault((g["column"], g["group"]), g)
    all_groups = list(keyed.values())
    top_unique = sorted(all_groups, key=lambda g: (int(g.get("private_gain_unique_vs_ordinary_and_shuffled",0)), int(g.get("private_gain",0)), int(g.get("coherent_minus_anchor_item_net",0))), reverse=True)[:20]
    top_retention = sorted(all_groups, key=lambda g: (int(g.get("retained_anchor_correct_ordinary_wrong",0)), int(g.get("retained_anchor_correct_shuffled_wrong",0)), int(g.get("coherent_minus_ordinary_item_net",0))), reverse=True)[:20]
    top_anchor_net = sorted(all_groups, key=lambda g: (int(g.get("coherent_minus_anchor_item_net",0)), int(g.get("private_gain",0))), reverse=True)[:20]
    worst_loss = sorted(all_groups, key=lambda g: (int(g.get("private_loss",0)), -int(g.get("private_gain",0))), reverse=True)[:20]
    focused = [g for g in all_groups if g["column"] in FOCUS]
    focused_unique = sorted(focused, key=lambda g: (int(g.get("private_gain_unique_vs_ordinary_and_shuffled",0)), int(g.get("coherent_minus_anchor_item_net",0))), reverse=True)[:20]
    focused_retention = sorted(focused, key=lambda g: (int(g.get("retained_anchor_correct_ordinary_wrong",0)), int(g.get("coherent_minus_ordinary_item_net",0))), reverse=True)[:20]
    out = {
        "status": "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": rel(IN_JSON),
        "aggregate": loc["aggregate"],
        "column_partitions": {col: loc["by_column"][col]["partition"] for col in DISCRETE},
        "top_unique_private_gains": top_unique,
        "top_anchor_retention_vs_ordinary": top_retention,
        "top_anchor_net_groups": top_anchor_net,
        "top_private_loss_groups": worst_loss,
        "focused_unique_private_gains": focused_unique,
        "focused_retention_vs_ordinary": focused_retention,
        "reading": "Private path has real but small coherent-specific pockets. Most count-level gains come from BLiMP/COMPS due to their size; official-score gains are macro-weighted and strongest in Supplement and GlobalPIQA. Focused Entity/EWoK pockets are mixed and not yet a broad relation-tracking repair.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    def table(rows, cols):
        lines = ["| column | group | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g in rows:
            lines.append(f"| {g.get('column')} | {g.get('group')} | {g.get('n_common')} | {g.get('coherent_minus_anchor_item_net')} | {g.get('coherent_minus_ordinary_item_net')} | {g.get('coherent_minus_shuffled_item_net')} | {g.get('private_gain')} | {g.get('private_loss')} | {g.get('private_gain_unique_vs_ordinary_and_shuffled')} | {g.get('retained_anchor_correct_ordinary_wrong')} |")
        return lines
    lines = [
        "# research private-pathway pocket summary",
        "",
        "Status: **COMPLETE**",
        "",
        "## Aggregate and column partitions",
        "",
        f"Aggregate: `{loc['aggregate']}`",
        "",
        "| column | n | coherent-anchor net | coherent-ordinary net | coherent-shuffled net | private gains | private losses | unique private gains | retained anchor correct while ordinary wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for col in DISCRETE:
        p = loc["by_column"][col]["partition"]
        lines.append(f"| {col} | {p.get('n_common')} | {p.get('coherent_minus_anchor_item_net')} | {p.get('coherent_minus_ordinary_item_net')} | {p.get('coherent_minus_shuffled_item_net')} | {p.get('private_gain')} | {p.get('private_loss')} | {p.get('private_gain_unique_vs_ordinary_and_shuffled')} | {p.get('retained_anchor_correct_ordinary_wrong')} |")
    lines += ["", "## Top coherent-specific private gains", ""] + table(top_unique[:12], [])
    lines += ["", "## Top retained anchor decisions where ordinary86 is wrong", ""] + table(top_retention[:12], [])
    lines += ["", "## Focused Entity/EWoK/Supplement/GlobalPIQA coherent-specific gains", ""] + table(focused_unique[:12], [])
    lines += ["", "## Largest private-loss families", ""] + table(worst_loss[:12], [])
    lines += [
        "",
        "## Interpretation",
        "",
        out["reading"],
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    OUT_MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","out_json":rel(OUT_JSON),"out_md":rel(OUT_MD)}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
