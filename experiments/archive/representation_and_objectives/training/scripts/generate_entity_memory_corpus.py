#!/usr/bin/env python3
"""Freeze a recombinational entity/event/state corpus for the research mini-screen.

The decisive split never withholds a lexical component wholesale. Every entity,
action, and state occurs in acquisition examples and in non-held-out bindings.
Only entity x transition recombinations, templates, and event orders are held out.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

ENTITIES = ["cup", "bowl", "box", "jar", "door", "window", "cloth", "rope", "lamp", "balloon", "drawer", "gate"]
# (family, prior, result, action result->, reverse action)
TRANSITIONS = [
    ("empty_full", "empty", "full", "filled", "emptied"),
    ("open_closed", "closed", "open", "opened", "closed"),
    ("clean_dirty", "dirty", "clean", "cleaned", "dirtied"),
    ("dry_wet", "dry", "wet", "wetted", "dried"),
    ("cold_hot", "cold", "hot", "heated", "cooled"),
    ("dark_lit", "dark", "lit", "lit", "extinguished"),
    ("flat_inflated", "flat", "inflated", "inflated", "deflated"),
    ("unlocked_locked", "unlocked", "locked", "locked", "unlocked"),
]
TRAIN_TEMPLATES = [
    "The {a} was {sa}. The {b} was {sb}. Mira {verb} the {actor}. The {query} is now [MASK].",
    "At first, the {a} was {sa}, and the {b} was {sb}. Then Mira {verb} the {actor}. Now the {query} is [MASK].",
    "The {a} started {sa}. The {b} started {sb}. Mira then {verb} the {actor}. Afterwards, the {query} is [MASK].",
]
HELD_TEMPLATES = [
    "Initially {sa} was the {a}, while {sb} was the {b}. After Mira had {verb} the {actor}, the {query} became [MASK].",
    "The {a}, once {sa}, and the {b}, once {sb}, were nearby. Mira {verb} the {actor}; afterward the {query} was [MASK].",
]
SINGLE_TRANSITION = [
    "The {a} was {sa}. Mira {verb} the {a}. The {a} is now [MASK].",
    "Initially the {a} was {sa}. After Mira {verb} it, the {a} became [MASK].",
]
STATE_ACQ = [
    "The {a} is {sa}. The state of the {a} is [MASK].",
    "A {sa} {a} is here. This {a} is [MASK].",
]
IDENTITY = [
    "The {a} was {sa}. The {b} was {sb}. Without any change, the {query} remained [MASK].",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(re.findall(r"\S+", text))


def held_pair(entity_i: int, trans_i: int) -> bool:
    # Each entity holds two of eight transitions; every transition is held for 3 entities.
    return (entity_i + 3 * trans_i) % 4 == 0


def rec(kind, split, text, answer, foil, entities, events, query, family, template_id, pair_held=False, order="single"):
    return {
        "kind": kind, "split": split, "text": text, "answer": answer, "foil": foil,
        "entities": entities, "events": events, "query_entity": query, "family": family,
        "template_id": template_id, "pair_held": pair_held, "event_order": order,
        "word_count": wc(text),
    }


def render_binding(rng, ei, ti, template, template_id, affected_query, reverse=False, held=False, split="train_binding"):
    a = ENTITIES[ei]
    b = rng.choice([x for x in ENTITIES if x != a])
    fam, prior, result, fwd, rev = TRANSITIONS[ti]
    # Reverse direction swaps before/after and verb.
    if reverse:
        before, after, verb = result, prior, rev
    else:
        before, after, verb = prior, result, fwd
    # Other entity starts in the opposite answer state, creating a real interference alternative.
    actor = a
    query = a if affected_query else b
    text = template.format(a=a, b=b, sa=before, sb=after, verb=verb, actor=actor, query=query)
    ans = after
    foil = before
    if not affected_query:
        ans, foil = after, before  # b began in after
    return rec("binding", split, text, ans, foil, [a,b], [{"entity":a,"verb":verb,"before":before,"after":after}], query, fam, template_id, held, "single")


def render_multievent(rng, ei, ti, template_id, held=True):
    a = ENTITIES[ei]
    b = rng.choice([x for x in ENTITIES if x != a])
    fam, prior, result, fwd, rev = TRANSITIONS[ti]
    # Interleaving: update A, update B, reverse A. Final A=prior, B=result.
    if template_id == "held_multi_interleave":
        text = (f"The {a} was {prior}. The {b} was {prior}. Mira {fwd} the {a}. "
                f"Next Mira {fwd} the {b}. Finally Mira {rev} the {a}. The {a} is now [MASK].")
        events = [{"entity":a,"verb":fwd,"before":prior,"after":result},
                  {"entity":b,"verb":fwd,"before":prior,"after":result},
                  {"entity":a,"verb":rev,"before":result,"after":prior}]
        answer, foil, query = prior, result, a
    else:
        text = (f"The {a} was {prior}. The {b} was {result}. Mira {fwd} the {a}. "
                f"Later Mira {rev} the {a}. After those events, the {b} is [MASK].")
        events = [{"entity":a,"verb":fwd,"before":prior,"after":result},
                  {"entity":a,"verb":rev,"before":result,"after":prior}]
        answer, foil, query = result, prior, b
    return rec("multi_event", "eval_held_order", text, answer, foil, [a,b], events, query, fam, template_id, held, "interleaved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=19843)
    ap.add_argument("--train-records", type=int, default=12000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train=[]; eval_rows=[]

    # Explicit component acquisition: all states/entities and both action directions.
    for ei,a in enumerate(ENTITIES):
        for ti,(fam,prior,result,fwd,rev) in enumerate(TRANSITIONS):
            for state,foil in [(prior,result),(result,prior)]:
                for k,t in enumerate(STATE_ACQ):
                    text=t.format(a=a,sa=state)
                    train.append(rec("state_acquisition","train",text,state,foil,[a],[],a,fam,f"state{k}"))
            for reverse,before,after,verb in [(False,prior,result,fwd),(True,result,prior,rev)]:
                for k,t in enumerate(SINGLE_TRANSITION):
                    text=t.format(a=a,sa=before,verb=verb)
                    train.append(rec("transition_acquisition","train",text,after,before,[a],[{"entity":a,"verb":verb,"before":before,"after":after}],a,fam,f"trans{k}"))
    # Identity selection exposes queried-entity lookup without transitions.
    for ei,a in enumerate(ENTITIES):
        for ti,(fam,prior,result,_,_) in enumerate(TRANSITIONS):
            b=ENTITIES[(ei+1+ti)%len(ENTITIES)]
            for q,ans,foil in [(a,prior,result),(b,result,prior)]:
                text=IDENTITY[0].format(a=a,b=b,sa=prior,sb=result,query=q)
                train.append(rec("identity_acquisition","train",text,ans,foil,[a,b],[],q,fam,"identity0"))

    # Observed binding combinations only; every component also appeared above.
    binding_pool=[]
    held_pool=[]
    for ei in range(len(ENTITIES)):
        for ti in range(len(TRANSITIONS)):
            for reverse in (False,True):
                for aq in (False,True):
                    if held_pair(ei,ti):
                        for k,t in enumerate(HELD_TEMPLATES):
                            held_pool.append(render_binding(rng,ei,ti,t,f"held{k}",aq,reverse,True,"eval_held_recomb_template"))
                    else:
                        for k,t in enumerate(TRAIN_TEMPLATES):
                            binding_pool.append(render_binding(rng,ei,ti,t,f"train{k}",aq,reverse,False,"train_binding"))
    while len(train) < args.train_records:
        train.append(dict(rng.choice(binding_pool)))
    rng.shuffle(train)
    eval_rows.extend(held_pool)
    # In-distribution binding comparator.
    eval_rows.extend(rng.sample(binding_pool, min(512,len(binding_pool))))
    # Event-order recombination on held entity-transition pairs.
    held_pairs=[(ei,ti) for ei in range(len(ENTITIES)) for ti in range(len(TRANSITIONS)) if held_pair(ei,ti)]
    for ei,ti in held_pairs:
        eval_rows.append(render_multievent(rng,ei,ti,"held_multi_interleave"))
        eval_rows.append(render_multievent(rng,ei,ti,"held_multi_unaffected"))

    files={}
    for name,rows in [("train",train),("eval",eval_rows)]:
        p=out/f"{name}.jsonl"
        with p.open("w",encoding="utf-8") as f:
            for i,r in enumerate(rows):
                rr=dict(r); rr["id"]=f"S198_{name}_{i:06d}"; f.write(json.dumps(rr,ensure_ascii=False)+"\n")
        files[name]={"path":str(p),"sha256":sha(p),"records":len(rows),"words":sum(r["word_count"] for r in rows)}
    # Verify lexical coverage and recombination exclusion.
    train_bind={(r["entities"][0],r["family"]) for r in train if r["kind"]=="binding"}
    eval_held={(r["entities"][0],r["family"]) for r in eval_rows if r["pair_held"]}
    lexical=defaultdict(set)
    for r in train:
        lexical["entity"].update(r["entities"])
        lexical["state"].update([r["answer"],r["foil"]])
        lexical["verb"].update(e["verb"] for e in r["events"])
    overlap=sorted(train_bind & eval_held)
    manifest={
        "status":"ENTITY_MEMORY_CORPUS_FROZEN", "seed":args.seed,
        "files":files, "held_pair_rule":"(entity_index + 3*transition_index) % 4 == 0",
        "train_binding_pairs":len(train_bind), "eval_held_pairs":len(eval_held),
        "held_pair_overlap_with_train_binding":overlap,
        "lexical_coverage":{"entities":sorted(lexical["entity"]),"states":sorted(lexical["state"]),"verbs":sorted(lexical["verb"])},
        "design":"All lexical components occur in acquisition/disjoint combinations; decisive evaluation withholds affected-entity x transition recombination plus templates/order.",
    }
    mp=out/"manifest.json"; mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(mp),"train_records":len(train),"train_words":files["train"]["words"],"eval_records":len(eval_rows),"held_overlap":len(overlap)},indent=2))

if __name__ == "__main__": main()
