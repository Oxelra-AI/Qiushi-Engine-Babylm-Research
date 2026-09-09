#!/usr/bin/env python3
"""research repaired context-intervention probe for RTD+MLM vs matched WWM.

No retraining and no re-evaluation. This repairs research's failed probe-case
construction by reconstructing the exact research consumed official 160-word
examples (same raw_dataset, iter_examples, seed, 1M selected words), then
splitting examples by word position rather than relying on punctuation-heavy
multi-sentence raw lines.

Matched-intervention constraints:
- Apply the SAME earlier-context interventions to matched WWM and RTD+MLM.
- Keep later suffix and target fixed.
- Compare fixed later-token MLM log-prob deltas in RTD vs WWM.
- Use unrelated-prefix perturbation to test specificity, not just sensitivity.

Outputs:
- data/rtd_wwm_context_probe_from_examples.json
- data/context_probe_from_examples_rows.csv
- notes/rtd_wwm_context_probe_from_examples.md
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, re, statistics, sys, time
from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
WWM_RUN = ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched'
RTD_RUN = ROOT/'training/runs/hybrid_rtd_mlm_debertav2_8x480_official_1M_b128_seed42'
WWM_CKPT = WWM_RUN/'hf_model/chck_1M'
RTD_CKPT = RTD_RUN/'hf_model/chck_1M'
OUT_JSON = ROOT/'data/rtd_wwm_context_probe_from_examples.json'
OUT_CSV = ROOT/'data/context_probe_from_examples_rows.csv'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/rtd_wwm_context_probe_from_examples.md')

sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples  # noqa: E402

STOP = set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")

@dataclass
class ProbeCase:
    case_id: int
    source: str
    prefix_words: list[str]
    later_words: list[str]
    target_word: str
    target_later_index: int
    unrelated_prefix_words: list[str]


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def reconstruct_examples(selected_words: int = 1_000_000, pool_words: int = 1_000_000, words_per_example: int = 160, seed: int = 42):
    raw_dir = WWM_RUN/'raw_dataset'
    files = [raw_dir / n for n in TRAIN_FILES]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f'Missing raw dataset files: {missing[:3]}')
    pool = list(iter_examples(files, pool_words, words_per_example))
    for i, ex in enumerate(pool):
        ex.example_id = i
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    actual = 0
    for ex in pool:
        if actual >= selected_words:
            break
        if actual + ex.words <= selected_words:
            out.append(ex)
            actual += ex.words
        else:
            take = selected_words - actual
            out.append(type(ex)(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source))
            actual += take
    if actual != selected_words:
        raise RuntimeError(f'word reconstruction mismatch {actual} vs {selected_words}')
    return out


def clean_word(w: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)


def token_ids_for_word(tokenizer, w: str):
    return tokenizer(' ' + w, add_special_tokens=False)['input_ids']


def choose_target(later_words: list[str], tokenizer):
    # Prefer a later content word with one-token encoding. Avoid first few suffix words to keep local context.
    start = min(8, max(0, len(later_words)//5))
    for i in range(start, len(later_words)):
        w = clean_word(later_words[i])
        if not WORD_RE.match(w) or w.lower() in STOP:
            continue
        ids = token_ids_for_word(tokenizer, w)
        if len(ids) == 1:
            return w, i
    return None


def make_cases(examples, tokenizer, n_cases=240, seed=306):
    rng = random.Random(seed)
    candidates = []
    for ex in examples:
        words = ex.text.split()
        if len(words) < 110:
            continue
        split = min(80, max(50, len(words)//2))
        prefix = words[:split]
        later = words[split:]
        if len(prefix) < 45 or len(later) < 35:
            continue
        cand = choose_target(later, tokenizer)
        if cand is None:
            continue
        candidates.append((ex, prefix, later, cand))
    if len(candidates) < 30:
        raise RuntimeError(f'too few candidates {len(candidates)}')
    rng.shuffle(candidates)
    # unrelated prefix pool matched roughly by length; avoid same example/source if possible
    cases=[]
    for ex, prefix, later, (target, tidx) in candidates:
        # choose an unrelated prefix of similar length from a different source or far example id
        options = [(ex2, p2) for ex2,p2,_,_ in candidates if ex2.example_id != ex.example_id and abs(len(p2)-len(prefix)) <= 15]
        if not options:
            options = [(ex2,p2) for ex2,p2,_,_ in candidates if ex2.example_id != ex.example_id]
        exu, uprefix = rng.choice(options)
        cases.append(ProbeCase(len(cases), ex.source, prefix, later, target, tidx, uprefix[:len(prefix)]))
        if len(cases) >= n_cases:
            break
    return cases


def shuffle_prefix(prefix: list[str], rng: random.Random) -> list[str]:
    # Shuffle contiguous 8-word blocks rather than individual words, preserving local word sequences
    blocks = [prefix[i:i+8] for i in range(0, len(prefix), 8)]
    rng.shuffle(blocks)
    return [w for b in blocks for w in b]


def masked_text(prefix_words: list[str], later_words: list[str], target_idx: int, tokenizer) -> str:
    later = list(later_words)
    later[target_idx] = tokenizer.mask_token
    if prefix_words:
        return ' '.join(prefix_words + later)
    return ' '.join(later)


def logprob_target(model, tokenizer, text: str, target_word: str, device) -> float:
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=256, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if mask_pos.numel() == 0:
        return float('nan')
    pos = int(mask_pos[-1, 1].item())
    tid = token_ids_for_word(tokenizer, target_word)[0]
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits[0, pos]
        return float(torch.log_softmax(logits, dim=-1)[tid].item())


def safe_mean(xs): return float(sum(xs)/len(xs)) if xs else float('nan')
def safe_median(xs): return float(statistics.median(xs)) if xs else float('nan')
def pos_frac(xs): return float(sum(1 for x in xs if x > 0)/len(xs)) if xs else float('nan')

def aggregate(rows):
    keys = [
        'wwm_delta_full_minus_deleted','rtd_mlm_delta_full_minus_deleted','rtd_minus_wwm_delta_deleted',
        'wwm_delta_full_minus_shuffled','rtd_mlm_delta_full_minus_shuffled','rtd_minus_wwm_delta_shuffled',
        'wwm_delta_full_minus_unrelated','rtd_mlm_delta_full_minus_unrelated','rtd_minus_wwm_delta_unrelated',
        'wwm_specificity_deleted_vs_unrelated','rtd_mlm_specificity_deleted_vs_unrelated','rtd_minus_wwm_specificity_deleted_vs_unrelated',
        'wwm_specificity_shuffled_vs_unrelated','rtd_mlm_specificity_shuffled_vs_unrelated','rtd_minus_wwm_specificity_shuffled_vs_unrelated',
    ]
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and not math.isnan(float(r[k]))]
        out[k] = {'mean': safe_mean(vals), 'median': safe_median(vals), 'pos_fraction': pos_frac(vals), 'n': len(vals)}
    return out


def main():
    t0=time.time(); setup_env()
    tokenizer = AutoTokenizer.from_pretrained(str(WWM_CKPT.resolve()), use_fast=True)
    examples = reconstruct_examples()
    cases = make_cases(examples, tokenizer, n_cases=240)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        'wwm': AutoModelForMaskedLM.from_pretrained(str(WWM_CKPT.resolve()), trust_remote_code=True).to(device).eval(),
        'rtd_mlm': AutoModelForMaskedLM.from_pretrained(str(RTD_CKPT.resolve()), trust_remote_code=True).to(device).eval(),
    }
    rng = random.Random(3061)
    rows=[]
    for c in cases:
        shuf_prefix = shuffle_prefix(list(c.prefix_words), rng)
        variants = {
            'full': masked_text(c.prefix_words, c.later_words, c.target_later_index, tokenizer),
            'deleted': masked_text([], c.later_words, c.target_later_index, tokenizer),
            'shuffled': masked_text(shuf_prefix, c.later_words, c.target_later_index, tokenizer),
            'unrelated': masked_text(c.unrelated_prefix_words, c.later_words, c.target_later_index, tokenizer),
        }
        rec = {
            'case_id': c.case_id,
            'source': c.source,
            'target_word': c.target_word,
            'prefix_len': len(c.prefix_words),
            'later_len': len(c.later_words),
            'target_later_index': c.target_later_index,
        }
        for mname, model in models.items():
            vals = {v: logprob_target(model, tokenizer, txt, c.target_word, device) for v,txt in variants.items()}
            for v,val in vals.items():
                rec[f'{mname}_lp_{v}'] = val
            rec[f'{mname}_delta_full_minus_deleted'] = vals['full'] - vals['deleted']
            rec[f'{mname}_delta_full_minus_shuffled'] = vals['full'] - vals['shuffled']
            rec[f'{mname}_delta_full_minus_unrelated'] = vals['full'] - vals['unrelated']
            rec[f'{mname}_specificity_deleted_vs_unrelated'] = (vals['full'] - vals['deleted']) - (vals['full'] - vals['unrelated'])
            rec[f'{mname}_specificity_shuffled_vs_unrelated'] = (vals['full'] - vals['shuffled']) - (vals['full'] - vals['unrelated'])
        rec['rtd_minus_wwm_delta_deleted'] = rec['rtd_mlm_delta_full_minus_deleted'] - rec['wwm_delta_full_minus_deleted']
        rec['rtd_minus_wwm_delta_shuffled'] = rec['rtd_mlm_delta_full_minus_shuffled'] - rec['wwm_delta_full_minus_shuffled']
        rec['rtd_minus_wwm_delta_unrelated'] = rec['rtd_mlm_delta_full_minus_unrelated'] - rec['wwm_delta_full_minus_unrelated']
        rec['rtd_minus_wwm_specificity_deleted_vs_unrelated'] = rec['rtd_mlm_specificity_deleted_vs_unrelated'] - rec['wwm_specificity_deleted_vs_unrelated']
        rec['rtd_minus_wwm_specificity_shuffled_vs_unrelated'] = rec['rtd_mlm_specificity_shuffled_vs_unrelated'] - rec['wwm_specificity_shuffled_vs_unrelated']
        rows.append(rec)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    agg = aggregate(rows)
    # Fast-profile numbers from completed research stdout/report files (direct checkpoint paths).
    fast_profile = {
        'wwm': {'blimp_fast':55.75,'supplement_fast':49.60,'ewok_fast':49.00,'entity_tracking_fast':17.03,'comps':50.40,'reading_eye_tracking':9.09,'reading_self_paced':2.98,'Reading_mean':6.035},
        'rtd_mlm': {'blimp_fast':55.92,'supplement_fast':46.80,'ewok_fast':51.18,'entity_tracking_fast':15.64,'comps':50.53,'reading_eye_tracking':9.14,'reading_self_paced':3.02,'Reading_mean':6.08},
    }
    cols = list(fast_profile['wwm'].keys())
    fast_delta = {k: round(fast_profile['rtd_mlm'][k] - fast_profile['wwm'][k], 4) for k in cols}
    rtd_metrics = json.loads((RTD_RUN/'scientific_metrics.json').read_text(encoding='utf-8'))
    wwm_metrics = json.loads((WWM_RUN/'scientific_metrics.json').read_text(encoding='utf-8'))
    payload = {
        'status':'REPAIRED_RTD_WWM_CONTEXT_PROBE_FROM_EXAMPLES',
        'design':'Reconstruct exact research consumed official 160-word examples; split by word position into earlier prefix and fixed later suffix; apply identical full/deleted/block-shuffled/unrelated-prefix variants to matched WWM and RTD+MLM; compare fixed later MLM target log-prob deltas and RTD-minus-WWM specificity relative to unrelated prefix.',
        'n_cases': len(rows),
        'rows_csv': str(OUT_CSV),
        'aggregate': agg,
        'fast_profile_from_step305_completed_eval': fast_profile,
        'rtd_minus_wwm_fast_profile': fast_delta,
        'rtd_training_metrics': {k:rtd_metrics.get(k) for k in ['rtd_replaced_recall_last','rtd_above_majority_last','rtd_pred_original_rate_last','rtd_label_original_rate_last','gen_loss_last','disc_mlm_loss_last']},
        'wwm_training_metrics': {k:wwm_metrics.get(k) for k in ['loss_last','masked_tokens_total','word_exposure']},
        'example_rows': rows[:5],
        'elapsed_sec': time.time()-t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research repaired RTD+MLM vs WWM context probe from exact examples','',f'Evidence JSON: `{OUT_JSON}`',f'Rows CSV: `{OUT_CSV}`','',
             '## Fast profile from completed research direct-checkpoint evaluation','',
             '| ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading |','|---:|---:|---:|---:|---:|---:|',
             f"| {fast_delta['blimp_fast']:+.2f} | {fast_delta['supplement_fast']:+.2f} | {fast_delta['ewok_fast']:+.2f} | {fast_delta['entity_tracking_fast']:+.2f} | {fast_delta['comps']:+.2f} | {fast_delta['Reading_mean']:+.3f} |",'',
             '## RTD shortcut-exclusion metrics at 1M','',
             f"Replaced recall {payload['rtd_training_metrics']['rtd_replaced_recall_last']:.4f}; above-majority {payload['rtd_training_metrics']['rtd_above_majority_last']:.4f}; pred-original {payload['rtd_training_metrics']['rtd_pred_original_rate_last']:.4f}; label-original {payload['rtd_training_metrics']['rtd_label_original_rate_last']:.4f}.",'',
             '## Matched context intervention aggregate','',
             '| metric | mean | median | positive fraction | n |','|---|---:|---:|---:|---:|']
    for k,v in agg.items():
        lines.append(f"| {k} | {v['mean']:+.5f} | {v['median']:+.5f} | {v['pos_fraction']:.3f} | {v['n']} |")
    lines += ['','## Decision-relevant reading','',
              'The route needs RTD-minus-WWM deleted/shuffled deltas and specificity-vs-unrelated to be positive. Single-model sensitivity is not enough because WWM already contains recoverable history signal.']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'n_cases':len(rows),'fast_delta':fast_delta,'aggregate':agg,'elapsed_sec':payload['elapsed_sec']}, indent=2))

if __name__ == '__main__':
    main()
