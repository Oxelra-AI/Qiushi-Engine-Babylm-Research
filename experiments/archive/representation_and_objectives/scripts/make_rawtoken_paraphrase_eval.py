#!/usr/bin/env python3
"""Build paraphrased held-recombination eval for research raw-token bridge.

The raw-token memory interface must not solve only the exact template surfaces.
This script copies the research annotated eval rows and adds paraphrased versions
of held_recomb binding rows with different sentence order and connective forms.
The metadata are kept only for evaluation grouping; the research raw-token trainer
never feeds entity/event/query/state positions to the model.
"""
from __future__ import annotations
import json, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path

IN_DIR = Path("experiments/archive/representation_and_objectives/data/annotated_corpus")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/rawtoken_paraphrase_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]

def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def verb_base(v):
    # simple surface repair for these controlled verb forms
    m = {
        "filled":"filled", "emptied":"emptied", "opened":"opened", "closed":"closed",
        "cleaned":"cleaned", "dirtied":"dirtied", "dried":"dried", "wet":"wet",
        "cooled":"cooled", "heated":"heated", "darkened":"darkened", "lit":"lit",
        "flattened":"flattened", "inflated":"inflated", "unlocked":"unlocked", "locked":"locked",
    }
    return m.get(v, v)

def make_para(row, k):
    e0,e1 = row["entities"][:2]
    init = row["foil"] if row["is_affected_query"] else row["answer"]
    # Because both entities start in same initial state in the held_recomb design:
    # affected query: answer=result, foil=initial; unaffected query: answer=initial, foil=result.
    ev = row["events"][0]
    verb = verb_base(ev["verb"])
    actor = ev["entity"]
    query = row["query_entity"]
    if k % 4 == 0:
        text = f"Initially, the {e0} was {init}, and the {e1} was {init}. Mira then {verb} the {actor}. In the end, the {query} was <mask>."
    elif k % 4 == 1:
        text = f"At first the {e0} and the {e1} were both {init}. Then Mira {verb} the {actor}. Later, the {query} was <mask>."
    elif k % 4 == 2:
        text = f"The {e0} began {init}. The {e1} began {init}. After Mira {verb} the {actor}, what was the {query}? It was <mask>."
    else:
        text = f"Before anything happened, the {e0} was {init} and the {e1} was {init}; Mira {verb} the {actor}. So the {query} ended up <mask>."
    nr = dict(row)
    nr["id"] = row["id"] + f"_para{k%4}"
    nr["text"] = text
    nr["split"] = "eval_held_paraphrase"
    nr["template_id"] = "para" + str(k % 4)
    nr["word_count"] = len(re.findall(r"\b\S+\b", text.replace("<mask>", "mask")))
    return nr

train = read_jsonl(IN_DIR/"train.jsonl")
eval_rows = read_jsonl(IN_DIR/"eval.jsonl")
para = []
for i,r in enumerate(eval_rows):
    if r.get("split") == "eval_held_recomb" and r.get("kind") == "binding":
        para.append(make_para(r, i))

combined = list(eval_rows) + para
write_jsonl(OUT_DIR/"train.jsonl", train)
write_jsonl(OUT_DIR/"eval.jsonl", combined)
write_jsonl(OUT_DIR/"paraphrase_only.jsonl", para)

# Verify quartet answer mixing in paraphrase split
quartets = defaultdict(set)
for r in para:
    quartets[r["quartet_id"]].add(r["answer"])
counts = Counter(len(v) for v in quartets.values())
manifest = {
    "status":"RAWTOKEN_PARAPHRASE_EVAL",
    "source_train": str(IN_DIR/"train.jsonl"),
    "source_eval": str(IN_DIR/"eval.jsonl"),
    "train_rows": len(train),
    "source_eval_rows": len(eval_rows),
    "paraphrase_rows": len(para),
    "combined_eval_rows": len(combined),
    "split_counts": dict(Counter(r["split"] for r in combined)),
    "kind_counts": dict(Counter(r["kind"] for r in combined)),
    "paraphrase_quartet_answer_set_size_counts": dict(counts),
    "sample_paraphrases": para[:4],
}
manifest_path = OUT_DIR/"manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
print(json.dumps({**manifest, "sha256": {"train": sha(OUT_DIR/"train.jsonl"), "eval": sha(OUT_DIR/"eval.jsonl")}}, indent=2, ensure_ascii=False))
