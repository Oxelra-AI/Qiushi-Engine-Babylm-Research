#!/usr/bin/env python3
"""Build official EWoK conditional-reversal bridge panel for raw-token memory work.

The active mechanism needs to move the same natural conditional-reversal items, not
just aggregate EWoK. This script reads the pristine official ewok_filtered jsonl
files and records, for every item, the two contexts/targets plus simple raw-token
features relevant to a metadata-free entity/event interface: repeated tokens,
context-target overlap, concept domain, and membership in the research relational
families used in compact-route analyses.
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from collections import Counter, defaultdict

EWOK = Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered")
OUT = Path("experiments/archive/representation_and_objectives/data/ewok_natural_bridge_panel")
OUT.mkdir(parents=True, exist_ok=True)
REL_DOMAINS = {"social-properties", "physical-dynamics", "spatial-relations", "physical-relations"}
STOP = {"the","a","an","is","are","was","were","to","of","in","on","at","and","or","but","not","with","than","as","it","this","that","then","after","before","both","now","nearby","from"}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def toks(s):
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?", s)]

def content(s):
    return [t for t in toks(s) if t not in STOP and len(t)>1]

def features(ctx,target):
    ct=content(ctx); tt=content(target); cset=set(ct); tset=set(tt)
    rep=[t for t,n in Counter(ct+tt).items() if n>=2]
    return {
        "context_content_n":len(ct),
        "target_content_n":len(tt),
        "context_target_overlap_n":len(cset & tset),
        "context_target_overlap":sorted(cset & tset),
        "repeated_content_tokens":sorted(rep),
        "repeated_content_n":len(rep),
    }

rows=[]; domain_counts={}
for p in sorted(EWOK.glob("*.jsonl")):
    domain=p.stem; n=0
    with p.open("r",encoding="utf-8") as f:
        for i,ln in enumerate(f):
            if not ln.strip(): continue
            r=json.loads(ln); n+=1
            item={
                "id":f"{domain}_{i}", "domain":domain,
                "is_step211_relational_domain": domain in REL_DOMAINS,
                "ConceptA":r.get("ConceptA"), "ConceptB":r.get("ConceptB"),
                "ContextType":r.get("ContextType"), "ContextDiff":r.get("ContextDiff"), "TargetDiff":r.get("TargetDiff"),
                "Context1":r.get("Context1"), "Context2":r.get("Context2"), "Target1":r.get("Target1"), "Target2":r.get("Target2"),
            }
            # Two conditional choices the official item requires.
            item["conditionals"]=[
                {"context":"Context1", "correct_target":"Target1", "foil_target":"Target2", "correct_full":(r.get("Context1","")+" "+r.get("Target1","")).strip(), "foil_full":(r.get("Context1","")+" "+r.get("Target2","")).strip(), "features":features(r.get("Context1",""),r.get("Target1",""))},
                {"context":"Context2", "correct_target":"Target2", "foil_target":"Target1", "correct_full":(r.get("Context2","")+" "+r.get("Target2","")).strip(), "foil_full":(r.get("Context2","")+" "+r.get("Target1","")).strip(), "features":features(r.get("Context2",""),r.get("Target2",""))},
            ]
            # Item-level feature union.
            all_text=" ".join([str(r.get(k,"")) for k in ["Context1","Context2","Target1","Target2"]])
            item["content_tokens_all"] = sorted(set(content(all_text)))
            item["repeated_tokens_all"] = sorted(t for t,c in Counter(content(all_text)).items() if c>=2)
            item["has_repeated_content"] = bool(item["repeated_tokens_all"])
            rows.append(item)
    domain_counts[domain]=n
panel=OUT/"ewok_bridge_panel.jsonl"
with panel.open("w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
# summary counts
summary={
    "status":"EWOK_NATURAL_BRIDGE_PANEL",
    "source_dir":str(EWOK),
    "panel":str(panel),
    "source_domain_counts":domain_counts,
    "total_items":len(rows),
    "relational_domain_items":sum(r["is_step211_relational_domain"] for r in rows),
    "has_repeated_content_items":sum(r["has_repeated_content"] for r in rows),
    "domain_repeated_counts":{d:{"items":sum(1 for r in rows if r['domain']==d),"repeated":sum(1 for r in rows if r['domain']==d and r['has_repeated_content'])} for d in domain_counts},
    "context_type_counts":dict(Counter(r.get("ContextType") for r in rows)),
    "context_diff_counts":dict(Counter(r.get("ContextDiff") for r in rows)),
    "target_diff_counts":dict(Counter(r.get("TargetDiff") for r in rows)),
    "sha256":sha(panel),
    "sample_relational": [r for r in rows if r['is_step211_relational_domain']][:5],
}
(OUT/"ewok_bridge_panel_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
md=(OUT.parents[4] / 'research/documents/representation_and_objectives/data/ewok_natural_bridge_panel/ewok_bridge_panel_summary.md')
md.write_text("\n".join([
    "# research EWoK natural bridge panel", "",
    f"Total items: {len(rows)}", f"Relational-domain items: {summary['relational_domain_items']}",
    f"Items with repeated raw content tokens: {summary['has_repeated_content_items']}", "",
    "This file is a bridge target for raw-token entity/event memory: a valid natural-transfer result must move item-level conditional choices here, especially research relational domains, rather than only aggregate EWoK.", "",
    "Panel: `"+str(panel)+"`",
    "Summary JSON: `"+str(OUT/'ewok_bridge_panel_summary.json')+"`",
])+"\n",encoding="utf-8")
print(json.dumps({k:summary[k] for k in ['status','total_items','relational_domain_items','has_repeated_content_items','sha256']},indent=2))
