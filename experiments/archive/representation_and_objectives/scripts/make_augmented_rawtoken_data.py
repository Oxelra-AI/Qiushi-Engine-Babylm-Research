#!/usr/bin/env python3
"""Augment research raw-token data with non-held multi-event training examples.

This construction requires a raw-token interface that survives temporal overwrite.
research's multi-event split was eval-only, so chance performance there does not
separate architecture failure from absent training signal. This data file keeps
the held entity×transition pairs untouched for eval and adds multi-event examples
only for non-held entity×transition pairs to train.jsonl.
"""
from __future__ import annotations
import json, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path

IN_DIR = Path("experiments/archive/representation_and_objectives/data/rawtoken_paraphrase_eval")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/augmented_rawtoken_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES = ["cup", "bowl", "box", "jar", "door", "window",
            "cloth", "rope", "lamp", "balloon", "drawer", "gate"]
TRANSITIONS = [
    ("empty_full",       "empty",    "full",     "filled",       "emptied"),
    ("open_closed",      "closed",   "open",     "opened",       "closed"),
    ("clean_dirty",      "dirty",    "clean",    "cleaned",      "dirtied"),
    ("dry_wet",          "dry",      "wet",      "wetted",       "dried"),
    ("cold_hot",         "cold",     "hot",      "heated",       "cooled"),
    ("dark_lit",         "dark",     "lit",      "lit",          "extinguished"),
    ("flat_inflated",    "flat",     "inflated", "inflated",     "deflated"),
    ("unlocked_locked",  "unlocked", "locked",   "locked",       "unlocked"),
]
MULTI_INTERLEAVE_TPL = (
    "The {a} was {s}. The {b} was {s}. "
    "Mira {fwd} the {a}. Next Mira {fwd} the {b}. "
    "Finally Mira {rev} the {a}. The {query} is now <mask>."
)
MULTI_INTERLEAVE_COUNTER_TPL = (
    "The {a} was {s}. The {b} was {s}. "
    "Mira {fwd} the {b}. Next Mira {fwd} the {a}. "
    "Finally Mira {rev} the {b}. The {query} is now <mask>."
)

def held_pair(entity_i:int, trans_i:int)->bool:
    return (entity_i + 3 * trans_i) % 4 == 0

def wc(text): return len(re.findall(r"\S+", text))

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]

def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def rec(text, answer, foil, entities, query_entity, fam, template_id, qid, reversed_entity, events):
    return {
        "kind":"multi_event", "split":"train_multi_event", "text":text,
        "answer":answer, "foil":foil, "entities":entities,
        "actor_entity":"both", "query_entity":query_entity,
        "is_affected_query": query_entity == reversed_entity,
        "family":fam, "template_id":template_id, "quartet_id":qid,
        "event_order":"interleaved_train", "word_count":wc(text), "events":events,
    }

def make_multi(a,b,fam,initial,result,fwd,rev,tpl,tid,qbase,reversed_entity):
    rows=[]
    # event list is metadata only; raw-token trainer never consumes it.
    if tid == "multi_train_std":
        events=[{"verb":fwd,"entity":a},{"verb":fwd,"entity":b},{"verb":rev,"entity":a}]
    else:
        events=[{"verb":fwd,"entity":b},{"verb":fwd,"entity":a},{"verb":rev,"entity":b}]
    for q in [a,b]:
        ans = initial if q == reversed_entity else result
        foil = result if q == reversed_entity else initial
        text = tpl.format(a=a,b=b,s=initial,fwd=fwd,rev=rev,query=q)
        rows.append(rec(text,ans,foil,[a,b],q,fam,tid,f"{qbase}_{q}",reversed_entity,events))
    return rows

base_train = read_jsonl(IN_DIR/"train.jsonl")
eval_rows = read_jsonl(IN_DIR/"eval.jsonl")
aug=[]
for ei,a in enumerate(ENTITIES):
    for ti,(fam,sx,sy,fwd,rev) in enumerate(TRANSITIONS):
        if held_pair(ei,ti):
            continue
        b=ENTITIES[(ei+3)%len(ENTITIES)]
        if b==a: b=ENTITIES[(ei+4)%len(ENTITIES)]
        for initial,result,fwd_v,rev_v in [(sx,sy,fwd,rev),(sy,sx,rev,fwd)]:
            qbase=f"TRAINMULTI_{a}_{b}_{fam}_{initial}"
            aug.extend(make_multi(a,b,fam,initial,result,fwd_v,rev_v,MULTI_INTERLEAVE_TPL,"multi_train_std",qbase+"_std",a))
            aug.extend(make_multi(a,b,fam,initial,result,fwd_v,rev_v,MULTI_INTERLEAVE_COUNTER_TPL,"multi_train_ctr",qbase+"_ctr",b))
train = base_train + aug
for i,r in enumerate(train): r["id"] = f"S245_train_{i:06d}"
# eval ids stay unchanged
write_jsonl(OUT_DIR/"train.jsonl", train)
write_jsonl(OUT_DIR/"eval.jsonl", eval_rows)
manifest={
    "status":"AUGMENTED_RAWTOKEN_DATA",
    "source":str(IN_DIR),
    "base_train_rows":len(base_train),
    "added_train_multi_event_rows":len(aug),
    "train_rows":len(train),
    "eval_rows":len(eval_rows),
    "train_split_counts":dict(Counter(r["split"] for r in train)),
    "eval_split_counts":dict(Counter(r["split"] for r in eval_rows)),
    "held_pair_rule":"(entity_index + 3*transition_index) % 4 == 0; train multi-event excludes held pairs",
    "sha256":{"train":sha(OUT_DIR/"train.jsonl"),"eval":sha(OUT_DIR/"eval.jsonl")},
    "sample_added":aug[:4],
}
(OUT_DIR/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(manifest,indent=2,ensure_ascii=False))
