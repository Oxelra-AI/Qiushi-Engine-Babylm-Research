#!/usr/bin/env python3
"""Extract complete sentence-like sources from cached FineWeb-Edu rows.

The public leader's dataset is sentence-level original FineWeb text followed by a
Qwen simplification. research's seqsafe96 chunk prompts are good for MLM visibility
but can begin/end mid-sentence, which is bad substrate for generation. This script
extracts high-precision complete English sentence spans from the cached single-doc
FineWeb rows for future Qwen rewrite generation.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import re
import statistics
import unicodedata
from typing import Any, Iterable

DEFAULT_FINEWEB = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl")
SUMMARY_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/sentence_source_summary.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_sentence_sources.md")

MOJIBAKE_PAT = re.compile(r"(?:�|Ã.|Â.|1⁄[24]|o¥|Ç|Ð|Þ|þ)")
URL_PAT = re.compile(r"https?://|www\.|@\w+\.\w+|\w+@\w+", re.I)
HTML_PAT = re.compile(r"<[^>]{1,40}>|&[a-z]{2,8};", re.I)
BAD_SUBSTR = ["lorem ipsum", "click here", "cookie policy", "privacy policy", "terms of use", "all rights reserved", "subscribe now", "sign up", "download pdf", "buy now", "advertisement"]
SPLIT_PAT = re.compile(r"(?<=[.!?])\s+(?=(?:[\"“‘']?[A-Z0-9]))")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:%|[a-zA-Z]+)?\b")


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def wc(text: str) -> int:
    return len(text.split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs: return {"n":0}
    ys = sorted(xs)
    def q(p): return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {"n":len(xs),"min":ys[0],"p05":q(0.05),"mean":statistics.mean(xs),"median":statistics.median(xs),"p95":q(0.95),"p99":q(0.99),"max":ys[-1],"sum":sum(xs)}


def nonlatin_letter_frac(text: str) -> float:
    letters=0; nonlatin=0
    for ch in text:
        if unicodedata.category(ch).startswith('L'):
            letters+=1
            if 'LATIN' not in unicodedata.name(ch,''):
                nonlatin+=1
    return nonlatin/letters if letters else 0.0


def alpha_frac(text: str) -> float:
    chars=[ch for ch in text if not ch.isspace()]
    if not chars: return 0.0
    return sum(unicodedata.category(ch).startswith('L') for ch in chars)/len(chars)


def quality_flags(text: str) -> list[str]:
    flags=[]; low=text.lower(); words=text.split()
    if MOJIBAKE_PAT.search(text): flags.append('mojibake')
    if HTML_PAT.search(text): flags.append('html')
    if URL_PAT.search(text): flags.append('url_or_email')
    if any(s in low for s in BAD_SUBSTR): flags.append('bad_substring')
    if nonlatin_letter_frac(text)>0.03: flags.append('nonlatin')
    if alpha_frac(text)<0.55: flags.append('low_alpha')
    if text.count('|') or text.count('\t'): flags.append('tableish')
    if words:
        short=sum(len(re.sub(r'[^A-Za-z]', '', w))<=1 for w in words)/len(words)
        digit_tok=sum(any(c.isdigit() for c in w) for w in words)/len(words)
        comma_density=text.count(',')/len(words)
        if short>0.30: flags.append('symbol_or_index_like')
        if digit_tok>0.28: flags.append('many_digit_tokens')
        if comma_density>0.20 and wc(text)<35: flags.append('catalog_like')
    return flags


def is_complete_sentence(s: str, min_words: int, max_words: int) -> tuple[bool, list[str]]:
    s=norm_text(s).strip()
    reasons=[]
    if not s: return False, ['empty']
    if wc(s)<min_words: reasons.append('too_short')
    if wc(s)>max_words: reasons.append('too_long')
    if not re.match(r'^["“‘\']?[A-Z0-9]', s): reasons.append('bad_start')
    if not re.search(r'[.!?]["”’\']?$', s): reasons.append('bad_end')
    flags=quality_flags(s)
    reasons.extend(flags)
    if len(re.findall(r'\b\d{4}\b', s))>=4 and wc(s)<40: reasons.append('index_like_years')
    if s.count(':')>=2 and wc(s)<50: reasons.append('colon_heavy')
    return len(reasons)==0, reasons


def entities(text: str) -> list[str]:
    bad={"The","This","That","These","Those","There","When","Where","What","How","Why","Because","For","And","But","New","All","Most","Some","Many","First","After","Before","During","Then","Each","Every"}
    out=[]
    for m in ENTITY_RE.finditer(text):
        e=' '.join(m.group(0).split())
        if e.split()[0] not in bad and len(e)>=4:
            out.append(e)
    return sorted(set(out))


def numbers(text: str) -> list[str]:
    return sorted(set(NUMBER_RE.findall(text)))


def split_sentences(text: str) -> list[str]:
    text=norm_text(text)
    parts=SPLIT_PAT.split(text)
    # Keep as simple high-precision splits; no attempt to join abbreviations except by filters.
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--fineweb',default=str(DEFAULT_FINEWEB)); ap.add_argument('--out',default=str(OUT_DEFAULT)); ap.add_argument('--summary',default=str(SUMMARY_DEFAULT)); ap.add_argument('--note',default=str(NOTE_DEFAULT)); ap.add_argument('--min-words',type=int,default=8); ap.add_argument('--max-words',type=int,default=55); args=ap.parse_args()
    rows=[]; total=0; single=0; single_words=0; split_candidates=0; reject=collections.Counter(); row_flag=collections.Counter(); doc_counter=collections.Counter(); examples=[]; rejects=[]
    with pathlib.Path(args.fineweb).open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            total+=1; o=json.loads(line); ids=o.get('doc_ids') or []; text=norm_text(o.get('text',''))
            if len(ids)!=1: continue
            single+=1; single_words+=int(o.get('words',wc(text))); doc=str(ids[0])
            rf=quality_flags(text)
            for fl in rf: row_flag[fl]+=1
            # Do not discard a row for URL/bad flags before sentence extraction; a clean sentence can occur nearby.
            sents=split_sentences(text)
            split_candidates+=len(sents)
            for j,s in enumerate(sents):
                ok, reasons=is_complete_sentence(s,args.min_words,args.max_words)
                if not ok:
                    for r in reasons: reject[r]+=1
                    if len(rejects)<8 and wc(s)>=5:
                        rejects.append({'source_row':i,'doc_id':doc,'sent_index':j,'words':wc(s),'reasons':reasons,'text':s[:500]})
                    continue
                rec={'sentence_id':len(rows),'source':'fineweb_edu_sentence_cached_initial_model_studies','source_row':i,'doc_id':doc,'sent_index_in_row':j,'text':s,'words':wc(s),'entities':entities(s),'numbers':numbers(s),'source_row_quality_flags':rf}
                rows.append(rec); doc_counter[doc]+=1
                if len(examples)<10: examples.append(rec)
    out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'status':'FINEWEB_SENTENCE_SOURCES_EXTRACTED','input':str(args.fineweb),'out':str(out),'total_cached_rows':total,'single_doc_rows':single,'single_doc_words':single_words,'split_sentence_candidates':split_candidates,'accepted_sentences':len(rows),'accepted_words':sum(r['words'] for r in rows),'unique_docs':len(doc_counter),'accepted_per_doc_top10':doc_counter.most_common(10),'row_quality_flag_counts':dict(row_flag),'rejection_reason_counts':dict(reject),'sentence_word_stats':stats([r['words'] for r in rows]),'entity_count_stats':stats([len(r['entities']) for r in rows]),'number_count_stats':stats([len(r['numbers']) for r in rows]),'examples':examples,'reject_examples':rejects,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'interpretation':'High-precision complete sentence sources for legal Qwen rewrite reconstruction; avoids fragment prompts from seqsafe chunks and is closer to the public leader data format, but total mass is limited by cached FineWeb availability.'}
    sp=pathlib.Path(args.summary); sp.parent.mkdir(parents=True,exist_ok=True); sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    note=pathlib.Path(args.note); note.parent.mkdir(parents=True,exist_ok=True)
    lines=['# research cached FineWeb complete sentence sources\n\n',f"Input: `{args.fineweb}`\n\n",f"Accepted {len(rows):,} sentence sources, {summary['accepted_words']:,} words, across {len(doc_counter):,} docs from {single:,} single-doc cached rows ({single_words:,} words).\n\n",'| metric | value |\n|---|---:|\n',f"| split candidates | {split_candidates:,} |\n",f"| accepted sentences | {len(rows):,} |\n",f"| accepted words | {summary['accepted_words']:,} |\n",f"| unique docs | {len(doc_counter):,} |\n",f"| mean sentence words | {summary['sentence_word_stats'].get('mean',0):.2f} |\n",'\nTop rejection reasons: '+json.dumps(reject.most_common(10),ensure_ascii=False)+'\n\n','Interpretation: this is the preferred source for any future Qwen FineWeb simplification/paraphrase generation because it supplies complete sentence-like spans, not arbitrary row fragments. It is not a trained result.\n\n',f"JSON: `{sp}`\n\nJSONL: `{out}`\n"]
    note.write_text(''.join(lines),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'out':str(out),'summary':str(sp),'note':str(note),'accepted_sentences':len(rows),'accepted_words':summary['accepted_words'],'unique_docs':len(doc_counter),'sentence_word_stats':summary['sentence_word_stats']},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
