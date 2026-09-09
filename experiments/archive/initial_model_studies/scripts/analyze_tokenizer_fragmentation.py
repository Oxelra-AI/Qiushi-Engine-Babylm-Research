#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from collections import Counter
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUN_BASE = ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/hf_model"
TOK40 = ROOT / "training/tokenizers/official40k"
STRICT_CORPUS = ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/raw_dataset"
OUT = ROOT / "data/tokenizer_fragmentation_analysis.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/tokenizer_fragmentation_analysis.md')

# Probe words are drawn from GlobalPIQA options/prompts, Entity-style words, and relation/state terms.
PROBE_WORDS = sorted(set('''
window glass metal paper ball cup water towel bucket spinach coin flour shelf eraser behind below above side east west left right before after Wednesday January morning afternoon knife razor blade oven mitts cardboard windshield sweater pills library classroom chair friend sibling shoes drawer kitchen rope rubber wooden marbles rice beans tomatoes apples jar box car tunnel lake visible invisible crushed dented break stick touch overflow freeze faster slower inside outside front back top bottom same different closer farther more fewer empty full animal apple arm door key room bag air ground fabric lightbulb object person man woman book paper sun shadow echo sound heat cold wet dry hard soft sharp dull heavy light'''.split()))


def load_lines(limit=20000):
    files = sorted(STRICT_CORPUS.glob("*.txt"))
    texts=[]; words=[]
    for p in files:
        for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip(): continue
            texts.append(line.strip())
            words.extend(line.split())
            if len(texts) >= limit:
                return texts, words
    return texts, words


def token_word_count(tok, word):
    # Prefix with 'The ' to make leading-space subword behavior comparable.
    base = tok('The', add_special_tokens=False)['input_ids']
    ids = tok('The ' + word, add_special_tokens=False)['input_ids']
    return max(0, len(ids)-len(base))


def sentence_stats(tok, texts):
    lens=[]; word_counts=[]; truncated=0; covered_words_est=[]
    for t in texts:
        ids = tok(t, add_special_tokens=True)['input_ids']
        wc=len(t.split()); word_counts.append(wc); lens.append(len(ids))
        if len(ids)>256: truncated += 1
        # crude estimate: assume uniform token/word in the sentence.
        covered_words_est.append(min(wc, int(wc * 256 / max(len(ids),1))))
    return {"n_texts":len(texts),"mean_words":sum(word_counts)/len(word_counts),"mean_tokens":sum(lens)/len(lens),"tokens_per_word":sum(lens)/max(sum(word_counts),1),"frac_over_256":truncated/len(texts),"mean_est_words_covered_at256":sum(covered_words_est)/len(covered_words_est)}


def summarize(tok, texts):
    probe = {w: token_word_count(tok,w) for w in PROBE_WORDS}
    hist = Counter(probe.values())
    content_groups = {
        'spatial_relation': ['left','right','above','below','behind','front','back','inside','outside','before','after','closer','farther'],
        'object_material_affordance': ['glass','metal','paper','wooden','rubber','knife','razor','towel','bucket','jar','box','rope','marbles','rice'],
        'entity_state': ['person','friend','sibling','man','woman','room','key','door','bag','cup','ball','water','coin'],
        'physical_change': ['crushed','dented','break','stick','touch','overflow','freeze','visible','invisible','wet','dry','hard','soft','sharp','dull']
    }
    group_stats={}
    for g, ws in content_groups.items():
        vals=[probe[w] for w in ws if w in probe]
        group_stats[g]={"mean_subtokens":sum(vals)/len(vals),"max_subtokens":max(vals),"hist":dict(Counter(vals))}
    return {"probe_subtokens":probe,"probe_hist":dict(hist),"probe_mean_subtokens":sum(probe.values())/len(probe),"group_stats":group_stats,"sentence_stats_20k_lines":sentence_stats(tok,texts)}


def main():
    texts, words = load_lines(limit=20000)
    tok16 = AutoTokenizer.from_pretrained(RUN_BASE)
    tok40 = AutoTokenizer.from_pretrained(TOK40)
    s16=summarize(tok16,texts); s40=summarize(tok40,texts)
    payload={"tokenizers":{"baseline16k":str(RUN_BASE),"official40k":str(TOK40)},"num_probe_words":len(PROBE_WORDS),"baseline16k":s16,"official40k":s40,"delta_40k_minus_16k":{"probe_mean_subtokens":s40['probe_mean_subtokens']-s16['probe_mean_subtokens'],"tokens_per_word_20k_lines":s40['sentence_stats_20k_lines']['tokens_per_word']-s16['sentence_stats_20k_lines']['tokens_per_word'],"frac_over_256":s40['sentence_stats_20k_lines']['frac_over_256']-s16['sentence_stats_20k_lines']['frac_over_256'],"mean_est_words_covered_at256":s40['sentence_stats_20k_lines']['mean_est_words_covered_at256']-s16['sentence_stats_20k_lines']['mean_est_words_covered_at256']}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=["# research — Tokenizer fragmentation analysis", "", f"Evidence JSON: `{OUT}`", "", "| tokenizer | probe mean subtokens | tokens/word (20k lines) | frac lines >256 | est words covered @256 |", "|---|---:|---:|---:|---:|", f"| baseline16k | {s16['probe_mean_subtokens']:.3f} | {s16['sentence_stats_20k_lines']['tokens_per_word']:.3f} | {s16['sentence_stats_20k_lines']['frac_over_256']:.3f} | {s16['sentence_stats_20k_lines']['mean_est_words_covered_at256']:.1f} |", f"| official40k | {s40['probe_mean_subtokens']:.3f} | {s40['sentence_stats_20k_lines']['tokens_per_word']:.3f} | {s40['sentence_stats_20k_lines']['frac_over_256']:.3f} | {s40['sentence_stats_20k_lines']['mean_est_words_covered_at256']:.1f} |", "", "Group stats and per-word fragmentation are in the JSON."]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"status":"TOKENIZER_FRAGMENTATION_DONE","out":str(OUT),"delta":payload['delta_40k_minus_16k']},indent=2))

if __name__=='__main__': main()
