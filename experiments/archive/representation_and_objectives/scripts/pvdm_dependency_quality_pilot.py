#!/usr/bin/env python3
"""Conservative dependency-quality pilot for Pivot-Visible Dependent Masking.

Processes a deterministic 10k-sentence sample from the legal compact 10M corpus.
It does not prepare training data. It estimates yield and manually inspectable quality
for pivot→dependent events under conservative dependency and lexical rules.
"""
from __future__ import annotations
import json, re, time
from collections import Counter
from pathlib import Path
import spacy

CORPUS = Path("experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl")
OUT = Path("experiments/archive/representation_and_objectives/data/pvdm_dependency_quality")
NOTE = Path("research/notes/representation_and_objectives/pvdm_dependency_quality_pilot.md")
OUT.mkdir(parents=True, exist_ok=True)

CHANGE = {"cause","lead","result","make","force","prevent","allow","enable","melt","freeze","heat","cool","break","open","close","increase","decrease","rise","fall","grow","shrink","expand","contract","push","pull","lift","drop","move","enter","leave","give","take","add","remove","build","destroy","create"}
SPATIAL = {"above","below","over","under","behind","beside","between","inside","outside","near","within","across","through","toward","towards","into","onto"}
TEMPORAL = {"before","after","during","until","while","when","then","later","earlier","subsequently"}
COMPARATIVE = {"more","less","greater","smaller","larger","higher","lower","faster","slower","longer","shorter","better","worse","than"}
POLARITY_MODAL = {"not","never","cannot","can","could","may","might","must","should","would","will"}
TARGET_DEPS = {"nsubj","nsubjpass","dobj","obj","pobj","attr","acomp","xcomp","ccomp","oprd","obl","dative"}
CONTENT_POS = {"NOUN","PROPN","VERB","ADJ","ADV","NUM"}


def category(tok):
    l = tok.lemma_.lower(); w = tok.lower_
    if l in CHANGE: return "change_causal_verb"
    if w in SPATIAL: return "spatial"
    if w in TEMPORAL: return "temporal"
    if w in COMPARATIVE or tok.tag_ in {"JJR","JJS","RBR","RBS"}: return "comparative"
    if w in POLARITY_MODAL: return "polarity_modal"
    return None


def linked_targets(pivot):
    candidates = []
    # Direct dependents and preposition objects.
    for child in pivot.children:
        if child.dep_ in TARGET_DEPS and child.pos_ in CONTENT_POS:
            candidates.append(child)
        if child.dep_ == "prep":
            candidates += [g for g in child.children if g.dep_ in {"pobj","obj"} and g.pos_ in CONTENT_POS]
    # For ADP/SCONJ/ADV pivots, use the linked clause head and its arguments.
    head = pivot.head
    if head is not pivot:
        if head.pos_ in CONTENT_POS and abs(head.i-pivot.i) <= 6:
            candidates.append(head)
        for child in head.children:
            if child.dep_ in TARGET_DEPS and child.pos_ in CONTENT_POS and abs(child.i-pivot.i) <= 10:
                candidates.append(child)
    # Deduplicate; never mask pivot itself.
    out=[]; seen=set()
    for t in candidates:
        if t.i != pivot.i and t.i not in seen and not t.is_stop and len(t.text)>1:
            seen.add(t.i); out.append(t)
    return out


def main():
    start=time.time()
    # Deterministic sentence sample: keep every 69th sentence until 10k.
    sample=[]; sent_idx=0
    with CORPUS.open() as f:
        for line in f:
            if not line.strip(): continue
            text=json.loads(line)["text"]
            for sent in re.split(r'(?<=[.!?])\s+', text):
                if len(sent.split()) >= 5:
                    if sent_idx % 69 == 0: sample.append(sent)
                    sent_idx += 1
                    if len(sample) >= 10000: break
            if len(sample) >= 10000: break
    nlp=spacy.load("en_core_web_sm", disable=["ner"])
    cat=Counter(); dep=Counter(); events=[]; eligible=0
    for doc in nlp.pipe(sample, batch_size=128, n_process=1):
        local=[]
        for tok in doc:
            c=category(tok)
            if not c: continue
            tgts=linked_targets(tok)
            if not tgts: continue
            cat[c]+=1
            for t in tgts:
                dep[t.dep_]+=1
                local.append({"pivot":tok.text,"pivot_lemma":tok.lemma_,"pivot_cat":c,"pivot_pos":tok.pos_,"target":t.text,"target_lemma":t.lemma_,"target_pos":t.pos_,"target_dep":t.dep_,"distance":abs(t.i-tok.i)})
        if local:
            eligible+=1
            if len(events)<400:
                events.append({"text":doc.text,"events":local[:8]})
    summary={"status":"PVDM_DEPENDENCY_QUALITY_PILOT","sample_sentences":len(sample),"eligible_sentences":eligible,"eligible_fraction":eligible/len(sample),"events_by_category":dict(cat),"target_dependencies":dict(dep),"saved_examples":len(events),"elapsed_sec":time.time()-start,"boundary":"quality/yield pilot only; not training data"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    with (OUT/"events_sample.jsonl").open("w") as f:
        for e in events: f.write(json.dumps(e,ensure_ascii=False)+"\n")
    NOTE.write_text("# research — PVDM dependency-quality pilot\n\n"+"\n".join([f"- {k}: {v}" for k,v in summary.items()])+f"\n\nSamples: `{OUT/'events_sample.jsonl'}`\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
