#!/usr/bin/env python3
"""research corrected counterfactual history + content-prediction probe.

Fixes research before any MCEC training decision:
1. Stores separate target/content offsets for original, same-entity-history swap,
   and unrelated-history swap. This removes the target-token misalignment caused
   by replacement words of different character length.
2. Adds an equal-distance, token-length-preserving unrelated-history replacement
   as a control for generic upstream text perturbation.
3. Tests not only representation change at the later mention, but whether the
   earlier same-entity mention gives extra causal support to masked content tokens
   immediately following the later mention.

Interpretation needed before training:
- Same-entity swap should affect later content prediction more than unrelated
  history swap. Otherwise the representation effect is not yet evidence that a
  mechanism like MCEC will improve Entity/EWoK/GlobalPIQA.
"""
from __future__ import annotations
import argparse, collections, json, math, os, pathlib, random, re, time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DEFAULT = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset'
OUT_DEFAULT = ROOT/'data/corrected_history_content_probe.json'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/corrected_history_content_probe.md')
BASE_TOK_PATH = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M'
STOP = set('''the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another about after before because through between where what which who whom whose than only also over again against every each such make made making'''.split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def norm_word(w: str) -> str:
    return w.strip("'\".,!?;:()[]{}“”‘’").lower()


def candidate_word(w: str) -> bool:
    n = norm_word(w)
    return len(n) >= 4 and n not in STOP and re.match(r"^[a-z][a-z'\-]+$", n) is not None


def iter_docs(raw_dir: pathlib.Path, max_docs: int):
    files = ['gutenberg.train.txt','simple_wiki.train.txt','childes.train.txt','open_subtitles.train.txt','bnc_spoken.train.txt','switchboard.train.txt']
    n = 0
    for fn in files:
        p = raw_dir/fn
        if not p.exists():
            continue
        buf = []
        words = 0
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s = line.strip()
                if s:
                    buf.append(s); words += len(s.split())
                    if words >= 180:
                        yield fn, ' '.join(buf); n += 1; buf = []; words = 0
                        if n >= max_docs: return
                elif buf:
                    yield fn, ' '.join(buf); n += 1; buf = []; words = 0
                    if n >= max_docs: return
            if buf:
                yield fn, ' '.join(buf); n += 1
                if n >= max_docs: return


@dataclass
class ConditionText:
    text: str
    target_start: int
    target_end: int
    content_spans: list[tuple[int,int]]


@dataclass
class ProbeCase:
    target_word: str
    entity_swap_word: str
    unrelated_word: str
    unrelated_swap_word: str
    source: str
    entity_gap_words: int
    unrelated_gap_words: int
    orig: ConditionText
    entity_cf: ConditionText
    unrelated_cf: ConditionText


def tok_len(tok, surface: str) -> int:
    return len(tok(surface, add_special_tokens=False)['input_ids'])


def same_case_surface(repl: str, template: str) -> str:
    return repl.capitalize() if template[:1].isupper() else repl.lower()


def make_condition(doc_text: str, p_start: int, p_end: int, content_spans_abs: list[tuple[int,int]], left_chars: int, right_chars: int) -> ConditionText:
    left = max(0, p_start - left_chars)
    right = min(len(doc_text), max([p_end] + [e for _, e in content_spans_abs]) + right_chars)
    return ConditionText(
        text=doc_text[left:right],
        target_start=p_start-left,
        target_end=p_end-left,
        content_spans=[(s-left, e-left) for s,e in content_spans_abs if s >= left and e <= right],
    )


def build_cases(raw_dir: pathlib.Path, max_docs: int, max_cases: int, seed: int, base_tokenizer_path: pathlib.Path) -> list[ProbeCase]:
    rng = random.Random(seed)
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer_path, use_fast=True, trust_remote_code=True)
    docs = list(iter_docs(raw_dir, max_docs))
    freq = collections.Counter()
    occurrences_by_word = collections.defaultdict(list)
    for di, (src, doc) in enumerate(docs):
        for m in WORD_RE.finditer(doc):
            surf = m.group(0); nw = norm_word(surf)
            if candidate_word(surf):
                freq[nw] += 1
                occurrences_by_word[nw].append((di, m.start(), m.end(), surf))
    by_toklen = collections.defaultdict(list)
    for w in freq:
        try:
            by_toklen[tok_len(base_tok, w)].append(w)
        except Exception:
            pass
    for k in by_toklen:
        rng.shuffle(by_toklen[k])

    def replacement_for(surface: str, forbidden: set[str]) -> Optional[str]:
        n = norm_word(surface); tl = tok_len(base_tok, surface)
        cand = [w for w in by_toklen.get(tl, []) if w not in forbidden and w != n]
        if not cand:
            return None
        repl = rng.choice(cand)
        return same_case_surface(repl, surface)

    cases: list[ProbeCase] = []
    for source, doc in docs:
        matches = [(i, m.group(0), m.start(), m.end(), norm_word(m.group(0))) for i,m in enumerate(WORD_RE.finditer(doc)) if candidate_word(m.group(0))]
        if len(matches) < 25:
            continue
        positions = collections.defaultdict(list)
        for j, tup in enumerate(matches):
            positions[tup[4]].append(j)
        for nw, idxs in list(positions.items()):
            if len(idxs) < 2:
                continue
            q_j, p_j = idxs[0], idxs[1]
            gap = p_j - q_j
            if gap < 10:
                continue
            _, q_surf, q_start, q_end, _ = matches[q_j]
            _, p_surf, p_start, p_end, _ = matches[p_j]
            # Need 1-3 content words immediately after the later mention.
            content = []
            k = p_j + 1
            while k < len(matches) and len(content) < 3:
                _, surf, s, e, cn = matches[k]
                if candidate_word(surf):
                    content.append((s,e))
                k += 1
            if not content:
                continue
            entity_repl = replacement_for(q_surf, {nw})
            if entity_repl is None:
                continue
            # Choose unrelated occurrence before p at similar distance and same token length replacement.
            unrelated_options = []
            for r_j, (_, r_surf, r_start, r_end, rn) in enumerate(matches[:p_j-3]):
                if r_j == q_j or rn == nw:
                    continue
                rgap = p_j - r_j
                if abs(rgap - gap) > max(5, 0.25*gap):
                    continue
                if abs(r_start - p_start) < 120:
                    continue
                r_repl = replacement_for(r_surf, {rn, nw, norm_word(entity_repl)})
                if r_repl is not None:
                    unrelated_options.append((abs(rgap-gap), r_j, r_surf, r_start, r_end, rn, rgap, r_repl))
            if not unrelated_options:
                continue
            unrelated_options.sort(key=lambda x: x[0])
            _, r_j, r_surf, r_start, r_end, rn, rgap, r_repl = unrelated_options[0]
            # Mutate documents and shift target/content positions separately.
            ent_doc = doc[:q_start] + entity_repl + doc[q_end:]
            ent_shift = len(entity_repl) - len(q_surf) if q_start < p_start else 0
            ent_p_start, ent_p_end = p_start + ent_shift, p_end + ent_shift
            ent_content = [(s+ent_shift, e+ent_shift) for s,e in content]
            irr_doc = doc[:r_start] + r_repl + doc[r_end:]
            irr_shift = len(r_repl) - len(r_surf) if r_start < p_start else 0
            irr_p_start, irr_p_end = p_start + irr_shift, p_end + irr_shift
            irr_content = [(s+irr_shift, e+irr_shift) for s,e in content]
            orig = make_condition(doc, p_start, p_end, content, 420, 220)
            ent = make_condition(ent_doc, ent_p_start, ent_p_end, ent_content, 420, 220)
            irr = make_condition(irr_doc, irr_p_start, irr_p_end, irr_content, 420, 220)
            # Verify target and content substrings are identical across conditions.
            def span_text(cond, span): return cond.text[span[0]:span[1]].lower()
            target0 = orig.text[orig.target_start:orig.target_end].lower()
            if ent.text[ent.target_start:ent.target_end].lower() != target0: continue
            if irr.text[irr.target_start:irr.target_end].lower() != target0: continue
            if len(orig.content_spans) != len(ent.content_spans) or len(orig.content_spans) != len(irr.content_spans): continue
            if [span_text(orig,s) for s in orig.content_spans] != [span_text(ent,s) for s in ent.content_spans]: continue
            if [span_text(orig,s) for s in orig.content_spans] != [span_text(irr,s) for s in irr.content_spans]: continue
            # Verify local target/content context unchanged (±60 chars around target through content).
            lo0=max(0,orig.target_start-60); hi0=min(len(orig.text), orig.content_spans[-1][1]+60)
            loe=max(0,ent.target_start-60); hie=min(len(ent.text), ent.content_spans[-1][1]+60)
            loi=max(0,irr.target_start-60); hii=min(len(irr.text), irr.content_spans[-1][1]+60)
            if orig.text[lo0:hi0] != ent.text[loe:hie]: continue
            if orig.text[lo0:hi0] != irr.text[loi:hii]: continue
            cases.append(ProbeCase(nw, norm_word(entity_repl), rn, norm_word(r_repl), source, gap, rgap, orig, ent, irr))
            if len(cases) >= max_cases:
                return cases
    return cases


def token_positions(tokenizer, text: str, start: int, end: int, max_len: int):
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    offs = enc.pop('offset_mapping')[0].tolist()
    pos = [i for i,(s,e) in enumerate(offs) if e > start and s < end]
    if not pos and offs:
        mid=(start+end)/2
        pos=[min(range(len(offs)), key=lambda i: abs((offs[i][0]+offs[i][1])/2-mid))]
    return enc, pos, offs


def content_loss(model, tokenizer, cond: ConditionText, max_len: int, device: str) -> Optional[float]:
    enc, _, offs = token_positions(tokenizer, cond.text, cond.target_start, cond.target_end, max_len)
    if not offs:
        return None
    positions=[]
    for s,e in cond.content_spans:
        positions += [i for i,(a,b) in enumerate(offs) if b > s and a < e]
    positions=sorted(set(positions))
    if not positions:
        return None
    input_ids=enc['input_ids'].to(device)
    attn=enc['attention_mask'].to(device)
    labels=torch.full_like(input_ids, -100)
    for p in positions:
        labels[0,p]=input_ids[0,p]
        input_ids[0,p]=tokenizer.mask_token_id
    with torch.no_grad():
        out=model(input_ids=input_ids, attention_mask=attn)
        logits=out.logits[0,positions,:]
        target=labels[0,positions]
        loss=F.cross_entropy(logits, target, reduction='mean').item()
    return float(loss)


def mention_repr(model, tokenizer, cond: ConditionText, layers: list[int], max_len: int, device: str):
    enc,pos,_=token_positions(tokenizer, cond.text, cond.target_start, cond.target_end, max_len)
    if not pos:
        return None
    enc={k:v.to(device) for k,v in enc.items()}
    with torch.no_grad():
        out=model(**enc, output_hidden_states=True)
    return {l: out.hidden_states[l][0,pos,:].mean(dim=0).detach().cpu() for l in layers}


def analyze_model(name: str, path: pathlib.Path, cases: list[ProbeCase], layers: list[int], max_len: int, device: str) -> dict:
    tok=AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    model=AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True).to(device).eval()
    cos_ent={l:[] for l in layers}; cos_irr={l:[] for l in layers}
    loss_orig=[]; loss_ent=[]; loss_irr=[]
    used=0
    for c in cases:
        ro=mention_repr(model,tok,c.orig,layers,max_len,device)
        re=mention_repr(model,tok,c.entity_cf,layers,max_len,device)
        ri=mention_repr(model,tok,c.unrelated_cf,layers,max_len,device)
        lo=content_loss(model,tok,c.orig,max_len,device)
        le=content_loss(model,tok,c.entity_cf,max_len,device)
        li=content_loss(model,tok,c.unrelated_cf,max_len,device)
        if ro is None or re is None or ri is None or lo is None or le is None or li is None:
            continue
        used += 1
        loss_orig.append(lo); loss_ent.append(le); loss_irr.append(li)
        for l in layers:
            cos_ent[l].append(F.cosine_similarity(ro[l].unsqueeze(0), re[l].unsqueeze(0)).item())
            cos_irr[l].append(F.cosine_similarity(ro[l].unsqueeze(0), ri[l].unsqueeze(0)).item())
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    loss_orig=np.asarray(loss_orig); loss_ent=np.asarray(loss_ent); loss_irr=np.asarray(loss_irr)
    ent_delta=loss_ent-loss_orig; irr_delta=loss_irr-loss_orig
    return {
        'num_used': used,
        'repr_entity_swap_cosine_mean': {str(l): float(np.mean(cos_ent[l])) for l in layers},
        'repr_unrelated_swap_cosine_mean': {str(l): float(np.mean(cos_irr[l])) for l in layers},
        'repr_entity_extra_divergence': {str(l): float(np.mean(cos_irr[l]) - np.mean(cos_ent[l])) for l in layers},
        'content_loss_orig': float(np.mean(loss_orig)),
        'content_loss_entity_cf': float(np.mean(loss_ent)),
        'content_loss_unrelated_cf': float(np.mean(loss_irr)),
        'content_entity_delta': float(np.mean(ent_delta)),
        'content_unrelated_delta': float(np.mean(irr_delta)),
        'content_extra_entity_effect': float(np.mean(ent_delta - irr_delta)),
        'content_extra_entity_effect_std': float(np.std(ent_delta - irr_delta)),
        'content_extra_entity_effect_sem': float(np.std(ent_delta - irr_delta) / math.sqrt(max(1, len(ent_delta)))),
        'fraction_entity_delta_gt_unrelated': float(np.mean(ent_delta > irr_delta)) if len(ent_delta) else None,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DEFAULT))
    ap.add_argument('--max_docs', type=int, default=3000)
    ap.add_argument('--max_cases', type=int, default=300)
    ap.add_argument('--max_len', type=int, default=160)
    ap.add_argument('--layers', nargs='+', type=int, default=[2,4,6,8])
    ap.add_argument('--seed', type=int, default=273)
    ap.add_argument('--out_json', default=str(OUT_DEFAULT))
    ap.add_argument('--out_note', default=str(NOTE_DEFAULT))
    ap.add_argument('--models', nargs='*', default=[
        'wwm43_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
        'wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
        'amlm42_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed42_100M_b256/hf_model/chck_40M',
        'amlm42_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed42_100M_b256/hf_model/chck_100M',
    ])
    args=ap.parse_args(); setup_env(); t0=time.time()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cases=build_cases(pathlib.Path(args.raw_dir), args.max_docs, args.max_cases, args.seed, BASE_TOK_PATH)
    if len(cases) < 30:
        raise RuntimeError(f'too few valid corrected cases: {len(cases)}')
    payload={'status':'CORRECTED_HISTORY_CONTENT_PROBE','num_cases':len(cases),'layers':args.layers,'max_len':args.max_len,
             'case_stats':{'mean_entity_gap':float(np.mean([c.entity_gap_words for c in cases])),'mean_unrelated_gap':float(np.mean([c.unrelated_gap_words for c in cases])),'unique_targets':len(set(c.target_word for c in cases))},
             'models':{},'elapsed_sec':None,
             'validity_note':'Separate offsets per condition; target/content substrings and local context are verified identical across original/entity_cf/unrelated_cf.'}
    for spec in args.models:
        name,path=spec.split('=',1); p=pathlib.Path(path)
        if not p.exists():
            continue
        payload['models'][name]=analyze_model(name,p,cases,args.layers,args.max_len,device)
        pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out_json).write_text(json.dumps(payload, indent=2)+'\n')
    payload['elapsed_sec']=time.time()-t0
    pathlib.Path(args.out_json).write_text(json.dumps(payload, indent=2)+'\n')
    lines=['# research corrected history/content probe','',f'Evidence JSON: `{args.out_json}`','',f"Cases: {len(cases)}; mean entity gap {payload['case_stats']['mean_entity_gap']:.1f}; mean unrelated gap {payload['case_stats']['mean_unrelated_gap']:.1f}; unique targets {payload['case_stats']['unique_targets']}",'', '## Content prediction effect','', '| model | n | loss orig | entity-cf delta | unrelated-cf delta | extra entity effect | SEM | frac(entity>unrel) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in payload['models'].items():
        lines.append(f"| {name} | {d['num_used']} | {d['content_loss_orig']:.4f} | {d['content_entity_delta']:+.4f} | {d['content_unrelated_delta']:+.4f} | {d['content_extra_entity_effect']:+.4f} | {d['content_extra_entity_effect_sem']:.4f} | {d['fraction_entity_delta_gt_unrelated']:.3f} |")
    lines += ['', '## Representation extra divergence (unrelated cosine - entity cosine; positive means same-entity history matters more than unrelated perturbation)', '', '| model | L2 | L4 | L6 | L8 |', '|---|---:|---:|---:|---:|']
    for name,d in payload['models'].items():
        x=d['repr_entity_extra_divergence']; lines.append(f"| {name} | {x['2']:+.5f} | {x['4']:+.5f} | {x['6']:+.5f} | {x['8']:+.5f} |")
    lines += ['', 'Interpretation: MCEC-style training is supported only if the same-entity history replacement has a larger effect than the matched unrelated-history replacement on masked content prediction, not merely on hidden-state geometry.']
    pathlib.Path(args.out_note).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_note).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out_json':args.out_json,'out_note':args.out_note,'num_cases':len(cases),'elapsed_sec':payload['elapsed_sec']}, indent=2))

if __name__=='__main__':
    main()
