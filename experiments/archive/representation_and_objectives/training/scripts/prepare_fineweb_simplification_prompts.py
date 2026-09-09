#!/usr/bin/env python3
"""Prepare Qwen simplification/paraphrase prompts for cached FineWeb seqsafe chunks.

This is a future fallback asset, not a generation launch. It targets the part of the
public leader's data principle we can legally reconstruct: original FineWeb-derived
English text paired with Qwen-generated meaning-preserving simpler restatements.
The exact gated leader train file is not used.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
from typing import Any

SRC_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl")
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_qwen_prompts/fineweb_simplification_paraphrase_prompts.jsonl")
META_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_qwen_prompts/prompt_metadata.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_qwen_prompt_plan.md")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:%|[a-zA-Z]+)?\b")

SYSTEM = "You rewrite English training text for a small masked language model. Preserve all named entities, numbers, dates, quantities, and factual relations. Do not add facts. Output only the rewritten text."


def wc(s: str) -> int:
    return len(str(s).split())


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def entities(text: str) -> list[str]:
    bad = {"The","This","That","These","Those","There","When","Where","What","How","Why","Because","For","And","But","New","All","Most","Some","Many","First","After","Before","During","Then","Each","Every"}
    out=[]
    for m in ENTITY_RE.finditer(text):
        e=" ".join(m.group(0).split())
        if e.split()[0] not in bad and len(e)>=4:
            out.append(e)
    return sorted(set(out))


def numbers(text: str) -> list[str]:
    return sorted(set(NUMBER_RE.findall(text)))


def content_score(text: str) -> float:
    low=text.lower()
    # Prefer factual/relational chunks over pure dialogue or symbol-heavy snippets.
    cues = [" is ", " are ", " was ", " were ", " has ", " have ", " located", "called", "known", "became", "born", "died", "founded", "published", "developed", "used", "includes", "contains", "because", "therefore"]
    cue_score=sum(1 for c in cues if c in low)
    ent=len(entities(text)); num=len(numbers(text))
    alpha=sum(ch.isalpha() for ch in text)/max(1,len(text))
    return cue_score + 0.35*ent + 0.15*num + alpha


def load_candidates(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            o=json.loads(line)
            if o.get('source') != 'fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies':
                continue
            text=norm_text(o['text']); words=int(o.get('words',wc(text)))
            if words != wc(text): raise RuntimeError(f'word mismatch {i}')
            rows.append({'row_index':i,'text':text,'words':words,'doc_id':str(o.get('doc_id','')),'source_row':o.get('source_row'),'chunk_start':o.get('chunk_start'), 'entities':entities(text), 'numbers':numbers(text), 'content_score':content_score(text)})
    return rows


def make_prompt(rec: dict[str, Any], kind: str, pid: str) -> dict[str, Any]:
    text=rec['text']
    if kind == 'simplification':
        instr = (
            "Rewrite the passage below into simpler, clearer English while preserving exactly the same facts. "
            "Keep all named entities, numbers, dates, quantities, and causal/temporal relations. Do not add examples or background. "
            "Keep the rewrite about the same length or shorter. Output only the rewrite.\n\nPASSAGE:\n" + text
        )
    elif kind == 'paraphrase':
        instr = (
            "Paraphrase the passage below in fluent English for language-model pretraining. Preserve every fact, entity, number, date, and relation. "
            "Do not add facts, explanations, headings, bullet points, or commentary. Output only the paraphrase.\n\nPASSAGE:\n" + text
        )
    else:
        raise ValueError(kind)
    return {
        'prompt_id': pid,
        'typ': kind,
        'source': 'cached_fineweb_seqsafe96_qwen_rewrite',
        'source_row_index': rec['row_index'],
        'source_doc_id': rec['doc_id'],
        'source_row': rec.get('source_row'),
        'chunk_start': rec.get('chunk_start'),
        'source_text': text,
        'source_words': rec['words'],
        'source_entities': rec['entities'],
        'source_numbers': rec['numbers'],
        'content_score': rec['content_score'],
        'system': SYSTEM,
        'prompt': instr,
    }


def stats(vals):
    xs=sorted(vals)
    if not xs: return {'n':0}
    def q(p): return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {'n':len(xs),'min':xs[0],'p05':q(0.05),'mean':statistics.mean(xs),'median':statistics.median(xs),'p95':q(0.95),'max':xs[-1]}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--source',default=str(SRC_DEFAULT)); ap.add_argument('--out',default=str(OUT_DEFAULT)); ap.add_argument('--metadata',default=str(META_DEFAULT)); ap.add_argument('--note',default=str(NOTE_DEFAULT)); ap.add_argument('--max-sources',type=int,default=4096); ap.add_argument('--seed',type=int,default=90917); args=ap.parse_args()
    rows=load_candidates(pathlib.Path(args.source))
    # Stratify by content score deciles while preserving many docs; sample at most one source per doc per pass when possible.
    rows_sorted=sorted(rows, key=lambda r:(-r['content_score'], r['doc_id'], r['chunk_start'] or 0))
    rng=random.Random(args.seed)
    by_doc=collections.defaultdict(list)
    for r in rows_sorted: by_doc[r['doc_id']].append(r)
    selected=[]
    docs=list(by_doc); rng.shuffle(docs)
    # First take best chunk per doc for breadth.
    for d in docs:
        selected.append(sorted(by_doc[d], key=lambda r:-r['content_score'])[0])
        if len(selected)>=args.max_sources: break
    # If asked for more than docs, fill remaining by score.
    if len(selected)<args.max_sources:
        seen={id(r) for r in selected}
        for r in rows_sorted:
            if id(r) not in seen:
                selected.append(r); seen.add(id(r))
                if len(selected)>=args.max_sources: break
    # Stable randomize prompt order but keep paired prompt ids tied to source ids.
    rng.shuffle(selected)
    prompts=[]
    for j,r in enumerate(selected):
        sid=f"fw_{j:05d}"
        prompts.append(make_prompt(r,'simplification',f'simp_{sid}'))
        prompts.append(make_prompt(r,'paraphrase',f'para_{sid}'))
    out=pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for p in prompts: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    meta={
        'status':'FINEWEB_QWEN_REWRITE_PROMPTS_PREPARED',
        'source_10M':str(args.source),'out':str(out),'max_sources':args.max_sources,'selected_sources':len(selected),'prompts':len(prompts),'seed':args.seed,
        'selected_source_words':sum(r['words'] for r in selected),'unique_docs':len(set(r['doc_id'] for r in selected)),'source_rows_available':len(rows),'source_words_available':sum(r['words'] for r in rows),
        'content_score_stats_selected':stats([r['content_score'] for r in selected]),'words_stats_selected':stats([r['words'] for r in selected]),'entity_count_stats_selected':stats([len(r['entities']) for r in selected]),'number_count_stats_selected':stats([len(r['numbers']) for r in selected]),
        'prompt_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'intended_generation':{'model':'qwen3.5-9b or available local Qwen','max_new_tokens':96,'temperature':0.2,'batch_size_hint':64,'expected_outputs':'simplification+paraphrase for each selected source, then source-faithfulness screening before any training materialization'},
        'constraint':'Exact gated go76dof/Fineweb_simplification_pairs train file is not used; every generated output must be counted as training words if later included.'
    }
    mpath=pathlib.Path(args.metadata); mpath.parent.mkdir(parents=True,exist_ok=True); mpath.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    note=pathlib.Path(args.note); note.parent.mkdir(parents=True,exist_ok=True)
    note.write_text(
        '# research cached FineWeb Qwen rewrite prompt plan\n\n'
        'Prepared prompts only; no Qwen generation was launched because both H100s are reserved for the semantic-view treatment/control contrast.\n\n'
        f"- Source candidate: `{args.source}`\n"
        f"- Selected sources: {len(selected):,} across {meta['unique_docs']:,} docs, {meta['selected_source_words']:,} words.\n"
        f"- Prompts: {len(prompts):,} = simplification + paraphrase for each source.\n"
        '- Scientific purpose: if raw cached FineWeb source replacement is promising or semantic-view is weak, this prepares a legal reconstruction closer to the leader phenotype: FineWeb-derived text plus Qwen-generated simpler/restated views, with source consistency screening before training.\n'
        '- It must remain distinct from the exact gated leader dataset; generated text counts toward BabyLM word exposure if used.\n\n'
        f"Metadata: `{mpath}`\n\nPrompts: `{out}`\n",
        encoding='utf-8')
    print(json.dumps({'status':meta['status'],'out':str(out),'metadata':str(mpath),'note':str(note),'selected_sources':len(selected),'prompts':len(prompts),'selected_source_words':meta['selected_source_words'],'unique_docs':meta['unique_docs']},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
