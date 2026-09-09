#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, statistics
from transformers import AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
MANIFEST = ROOT / 'data/structure_density_revision_157/manifest.json'
OUT = ROOT / 'data/structure_arms_tokenization_check.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/structure_arms_tokenization_check.md')
TOKENIZER_REPO = 'BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict'
ARMS = ['high_entity_state','matched_low','uniform']

def is_word_start(tok: str) -> bool:
    return tok.startswith('Ġ') or tok.startswith('▁')

def count_word_groups(tokenizer, ids):
    groups=0; special=set(tokenizer.all_special_ids)
    for j, tid in enumerate(ids):
        if int(tid) in special: continue
        s=str(tokenizer.convert_ids_to_tokens(int(tid)))
        if groups == 0 or is_word_start(s) or j == 0:
            groups += 1
    return groups

def summarize(path: pathlib.Path, tokenizer, max_seq=256):
    rows=0; words=0; toks=0; kept=0; groups=0; over=0; max_len=0; lengths=[]; source_words={}
    first_over=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            text=r['text']; w=int(r['words']); rows += 1; words += w
            ids=tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
            L=len(ids); lengths.append(L); toks += L; max_len=max(max_len,L)
            k=min(L,max_seq); kept += k
            if L > max_seq:
                over += 1
                if len(first_over) < 5:
                    first_over.append({'row':rows-1,'tokens':L,'words':w,'source_file':r.get('source_file'),'start_line':r.get('start_line'),'text_preview':text[:180]})
            groups += count_word_groups(tokenizer, ids[:max_seq])
            source_words[r.get('source_file','?')] = source_words.get(r.get('source_file','?'),0)+w
    return {
        'path': str(path), 'rows': rows, 'words': words, 'tokens': toks, 'kept_tokens_at_256': kept,
        'word_groups_kept_at_256': groups, 'tokens_per_word': toks/max(1,words),
        'kept_tokens_per_word': kept/max(1,words), 'word_groups_kept_per_word': groups/max(1,words),
        'expected_wwm_selected_word_groups_at_0p15': 0.15*groups,
        'expected_wwm_predicted_tokens_at_0p15': 0.15*kept,
        'over_256_examples': over, 'over_256_fraction': over/max(1,rows), 'max_tokens': max_len,
        'token_len_mean': statistics.mean(lengths) if lengths else 0,
        'token_len_median': statistics.median(lengths) if lengths else 0,
        'source_words': dict(sorted(source_words.items())), 'first_over_256': first_over,
    }

def main():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    tok=AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision='main', use_fast=True, cache_dir=os.environ['HF_HUB_CACHE'])
    manifest=json.loads(MANIFEST.read_text())
    payload={'status':'STRUCTURE_ARM_TOKENIZATION_CHECK','manifest':str(MANIFEST),'max_seq_length':256,'arms':{}}
    for arm in ARMS:
        path=pathlib.Path(manifest['arm_paths'][arm]['path'])
        payload['arms'][arm]=summarize(path,tok,256)
    # Pairwise compact differences against matched_low.
    base=payload['arms']['matched_low']
    payload['vs_matched_low']={}
    for arm in ARMS:
        if arm=='matched_low': continue
        a=payload['arms'][arm]
        payload['vs_matched_low'][arm]={k:a[k]-base[k] for k in ['words','rows','tokens_per_word','kept_tokens_per_word','word_groups_kept_per_word','over_256_fraction','expected_wwm_predicted_tokens_at_0p15']}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    lines=['# research — structure arms tokenization check','',f'Evidence JSON: `{OUT}`','', '| arm | rows | words | tokens/word | kept tokens/word | groups/word | over-256 frac | max tokens |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s=payload['arms'][arm]
        lines.append(f"| {arm} | {s['rows']} | {s['words']} | {s['tokens_per_word']:.4f} | {s['kept_tokens_per_word']:.4f} | {s['word_groups_kept_per_word']:.4f} | {s['over_256_fraction']:.4%} | {s['max_tokens']} |")
    lines += ['', 'Interpretation: these checks should be read before interpreting training differences; large hidden differences in kept tokens, truncation, or WWM groups would weaken a structure-density result.']
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'vs_matched_low':payload['vs_matched_low']}, indent=2))
if __name__=='__main__': main()
