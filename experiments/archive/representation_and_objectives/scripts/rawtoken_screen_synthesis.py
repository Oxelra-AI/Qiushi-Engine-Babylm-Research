#!/usr/bin/env python3
"""Synthesize research raw-token entity-memory screens."""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("experiments/archive/representation_and_objectives")
RUNS = {
    "vanilla_base20": ROOT/"data/rawtoken_vanilla_full20/result.json",
    "rawmem_base20": ROOT/"data/rawtoken_rawmem_full20/result.json",
    "rawmem_noevent_base20": ROOT/"data/rawtoken_noevent_full20/result.json",
    "lexmem_base20": ROOT/"data/lexmem_full20/result.json",
    "lexmem_noevent_base20": ROOT/"data/lexmem_noevent_full20/result.json",
    "vanilla_aug20": ROOT/"data/lexrec_vanilla_aug20/result.json",
    "lexrec_aug20": ROOT/"data/lexrec_aug20/result.json",
}
OUT = ROOT/"data/rawtoken_bridge_screen"
OUT.mkdir(parents=True, exist_ok=True)

def read_json(p):
    return json.loads(Path(p).read_text())

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def acc(res, key, sub=None):
    b=res.get("baseline",{})
    if sub:
        return b.get(key,{}).get(sub)
    return b.get(key,{}).get("accuracy")

def perm_acc(res, key):
    return res.get("write_permutation",{}).get(key,{}).get("accuracy")

def metric_table(results):
    rows=[]
    for name,r in results.items():
        row={
            "run": name,
            "status": r.get("status"),
            "arm": r.get("arm"),
            "train_records": r.get("train_records"),
            "eval_records": r.get("eval_records"),
            "epochs": r.get("epochs"),
            "runtime_sec": r.get("runtime_sec"),
            "loss_final": r.get("losses",[None])[-1],
            "held_recomb": acc(r,"eval_held_recomb"),
            "held_recomb_affected": acc(r,"selective_eval_held_recomb","affected_accuracy"),
            "held_recomb_unaffected": acc(r,"selective_eval_held_recomb","unaffected_accuracy"),
            "held_recomb_composite": acc(r,"selective_eval_held_recomb","composite"),
            "held_recomb_quartet_all": acc(r,"selective_eval_held_recomb","quartet_all_rate"),
            "paraphrase": acc(r,"eval_held_paraphrase"),
            "paraphrase_affected": acc(r,"selective_eval_held_paraphrase","affected_accuracy"),
            "paraphrase_unaffected": acc(r,"selective_eval_held_paraphrase","unaffected_accuracy"),
            "paraphrase_composite": acc(r,"selective_eval_held_paraphrase","composite"),
            "paraphrase_quartet_all": acc(r,"selective_eval_held_paraphrase","quartet_all_rate"),
            "multi_event": acc(r,"multi_event"),
            "write_perm_held_recomb": perm_acc(r,"eval_held_recomb"),
            "write_perm_paraphrase": perm_acc(r,"eval_held_paraphrase"),
            "write_perm_multi_event": perm_acc(r,"multi_event"),
        }
        if row["write_perm_held_recomb"] is not None and row["held_recomb"] is not None:
            row["write_perm_delta_held_recomb"] = row["write_perm_held_recomb"] - row["held_recomb"]
        if row["write_perm_multi_event"] is not None and row["multi_event"] is not None:
            row["write_perm_delta_multi_event"] = row["write_perm_multi_event"] - row["multi_event"]
        rows.append(row)
    return rows

def audit_paraphrase():
    p = ROOT/"data/rawtoken_paraphrase_eval/eval.jsonl"
    rows=read_jsonl(p)
    para=[r for r in rows if r.get("split")=="eval_held_paraphrase"]
    # Proper grouping strips actor/query suffix exactly as evaluator does.
    groups=defaultdict(list)
    for r in para:
        base = r["quartet_id"].rsplit("_",2)[0]
        groups[base].append(r)
    return {
        "eval_file": str(p),
        "eval_sha256": sha(p),
        "paraphrase_rows": len(para),
        "proper_quartet_groups": len(groups),
        "proper_group_size_counts": dict(Counter(len(v) for v in groups.values())),
        "proper_answer_set_size_counts": dict(Counter(len(set(r["answer"] for r in v)) for v in groups.values())),
        "proper_all_groups_have_mixed_answers": all(len(set(r["answer"] for r in v))>1 for v in groups.values()),
    }

def main():
    results={k:read_json(p) for k,p in RUNS.items() if p.exists()}
    rows=metric_table(results)
    by={r["run"]:r for r in rows}
    deltas={}
    def delta(a,b,key):
        if a in by and b in by and by[a].get(key) is not None and by[b].get(key) is not None:
            return by[a][key]-by[b][key]
        return None
    for arm in ["rawmem_base20","rawmem_noevent_base20","lexmem_base20","lexmem_noevent_base20"]:
        deltas[arm+"_minus_vanilla_base20"]={k:delta(arm,"vanilla_base20",k) for k in ["held_recomb","paraphrase","multi_event","held_recomb_affected","held_recomb_unaffected"]}
    deltas["lexrec_aug20_minus_vanilla_aug20"]={k:delta("lexrec_aug20","vanilla_aug20",k) for k in ["held_recomb","paraphrase","multi_event","held_recomb_affected","held_recomb_unaffected"]}
    payload={
        "status":"RAWTOKEN_BRIDGE_SCREEN_SYNTHESIS",
        "runs": {k:str(v) for k,v in RUNS.items()},
        "table": rows,
        "deltas": deltas,
        "paraphrase_audit": audit_paraphrase(),
        "scientific_interpretation": [
            "research's metadata-assisted learned memory is only an integration ceiling: it was supplied hard slot/state/event/query coordinates.",
            "Naive soft-slot raw memory overfits training templates but does not outperform vanilla on held/paraphrase and write-permutation barely changes results; event write is not being used as selective entity update.",
            "Lexical identity grouping and lexical recurrent memory also fail to produce write-sensitive held recombination; lexrec reaches multi-event only because non-held multi-event training makes overwrite learnable by ordinary vanilla too.",
            "No tested metadata-free raw-token interface satisfies the bridge precondition (held recombination >0.70, paraphrase robustness, multi-event overwrite, write-sensitive identity update). Frozen DeBERTa+memory natural transfer is therefore not scientifically interpretable yet."
        ]
    }
    outj=OUT/"rawtoken_bridge_screen_summary.json"; outj.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    md=(OUT.parents[4] / 'research/documents/representation_and_objectives/data/rawtoken_bridge_screen/rawtoken_bridge_screen_summary.md')
    lines=["# research raw-token bridge screen", "", "## Metric table", "", "| run | held | aff | unaff | paraphrase | multi | write_perm_delta_held | final_loss |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r.get('held_recomb')} | {r.get('held_recomb_affected')} | {r.get('held_recomb_unaffected')} | {r.get('paraphrase')} | {r.get('multi_event')} | {r.get('write_perm_delta_held_recomb')} | {r.get('loss_final')} |")
    lines += ["", "## Deltas", "", "```json", json.dumps(deltas,indent=2), "```", "", "## Interpretation"]
    for x in payload["scientific_interpretation"]: lines.append("- "+x)
    lines += ["", "## Paraphrase audit", "", "```json", json.dumps(payload['paraphrase_audit'],indent=2), "```"]
    md.write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":payload["status"],"out_json":str(outj),"out_md":str(md),"table":rows,"deltas":deltas,"paraphrase_audit":payload['paraphrase_audit']},indent=2))
if __name__=="__main__": main()
