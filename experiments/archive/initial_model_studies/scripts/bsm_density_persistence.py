#!/usr/bin/env python3
"""research exposure-matched BSM density and persistence probe.

This repairs the research over-interpretation.  research used answer-masked,
templated binding examples and saturated quickly; research asks whether that signal
survives realistic dilution with official BabyLM text, whether it can form from an
early checkpoint as well as a mature checkpoint, and whether it persists after a
matched official-only continuation.

It is still a small mechanism probe, not an official training route.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time
from dataclasses import asdict
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SCRIPT_DIR = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR.resolve()))
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))

from binding_switch_margin_pretest import (  # noqa: E402
    FROZEN_CKPT, TRAIN_TEMPLATES, HELDOUT_TEMPLATES, BindingPair,
    build_inventories, generate_pairs, measure_binding_pairs, tokenize_binding_query,
)
from babylm_masked_train import TRAIN_FILES, Example, download_dataset, iter_examples  # noqa: E402

MODEL_ROOT = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model'
OUT_DIR = ROOT / 'training/runs/bsm_density_persistence'
OUT_JSON = ROOT / 'data/bsm_density_persistence.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/bsm_density_persistence.md')

NATURAL_TEMPLATES = [
    ("At school, {E1} kept the {V1} in a bag, while {E2} kept the {V2} on the desk. Later, {Eq} asked for the [MASK].",
     "At school, {E1} kept the {V1} in a bag, while {E2} kept the {V2} on the desk. Later, {Eq} asked for the [MASK]."),
    ("During the game, the thing near {E1} was the {V1}, but the thing near {E2} was the {V2}. When the teacher called {Eq}, everyone pointed to the [MASK].",
     "During the game, the thing near {E1} was the {V1}, but the thing near {E2} was the {V2}. When the teacher called {Eq}, everyone pointed to the [MASK]."),
    ("{E1} talked about a {V1} before lunch. After lunch, {E2} talked about a {V2}. In the story, {Eq} remembered the [MASK].",
     "{E1} talked about a {V1} before lunch. After lunch, {E2} talked about a {V2}. In the story, {Eq} remembered the [MASK]."),
    ("The report said that {E1} was linked with {V1}. A different line said that {E2} was linked with {V2}. The question about {Eq} should end with [MASK].",
     "The report said that {E1} was linked with {V1}. A different line said that {E2} was linked with {V2}. The question about {Eq} should end with [MASK]."),
]


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def text_words(s: str) -> int:
    return len(s.replace('[MASK]', 'TOKEN').split())


def pair_words(p: BindingPair) -> int:
    return text_words(p.text_q1) + text_words(p.text_q2)


def official_examples(args, outdir: pathlib.Path, max_words: int, words_per_example: int = 120) -> List[Example]:
    dargs = SimpleNamespace(dataset_id=args.dataset_id, dataset_revision=args.dataset_revision)
    raw_dir, _ = download_dataset(dargs, outdir)
    files = [raw_dir / n for n in TRAIN_FILES]
    return list(iter_examples(files, max_words, words_per_example))


def make_training_items(binding_pairs: List[BindingPair], official_pool: List[Example], ratio: float, total_words: int, rng: random.Random):
    """Return shuffled items with approximately ratio of binding words by exposure."""
    target_bind = int(round(total_words * ratio))
    target_off = total_words - target_bind
    items = []
    bw = 0
    ow = 0
    # repeat binding pairs as needed
    bind_cycle = binding_pairs[:]
    rng.shuffle(bind_cycle)
    bi = 0
    while bw < target_bind and bind_cycle:
        p = bind_cycle[bi % len(bind_cycle)]
        items.append(('binding', p, pair_words(p)))
        bw += pair_words(p)
        bi += 1
        if bi % len(bind_cycle) == 0:
            rng.shuffle(bind_cycle)
    oi = 0
    pool = official_pool[:]
    rng.shuffle(pool)
    while ow < target_off and pool:
        ex = pool[oi % len(pool)]
        items.append(('official', ex, ex.words))
        ow += ex.words
        oi += 1
        if oi % len(pool) == 0:
            rng.shuffle(pool)
    rng.shuffle(items)
    return items, {'target_total_words': total_words, 'target_binding_ratio': ratio, 'actual_binding_words': bw, 'actual_official_words': ow, 'actual_total_words': bw + ow, 'actual_binding_ratio': bw / max(1, bw+ow), 'n_binding_items': sum(1 for x in items if x[0]=='binding'), 'n_official_items': sum(1 for x in items if x[0]=='official')}


def official_loss(model, tokenizer, ex: Example, device, rng: random.Random, mask_prob: float = 0.15, max_len: int = 192):
    ids = tokenizer(ex.text, add_special_tokens=False, truncation=True, max_length=max_len)['input_ids']
    if len(ids) < 2:
        return None
    labels = [-100] * len(ids)
    masked = list(ids)
    cand = list(range(len(ids)))
    rng.shuffle(cand)
    n_mask = max(1, int(round(len(ids) * mask_prob)))
    for pos in cand[:n_mask]:
        labels[pos] = ids[pos]
        masked[pos] = tokenizer.mask_token_id
    inp = torch.tensor([masked], dtype=torch.long, device=device)
    lab = torch.tensor([labels], dtype=torch.long, device=device)
    return model(input_ids=inp, labels=lab).loss


def binding_loss(model, tokenizer, p: BindingPair, device):
    mask_id = tokenizer.mask_token_id
    ids1, mp1 = tokenize_binding_query(tokenizer, p.text_q1, mask_id)
    ids2, mp2 = tokenize_binding_query(tokenizer, p.text_q2, mask_id)
    if not mp1 or not mp2:
        return None
    lab1 = [-100] * len(ids1); lab1[mp1[0]] = p.v1_tid
    lab2 = [-100] * len(ids2); lab2[mp2[0]] = p.v2_tid
    inp1 = torch.tensor([ids1], dtype=torch.long, device=device); y1 = torch.tensor([lab1], dtype=torch.long, device=device)
    inp2 = torch.tensor([ids2], dtype=torch.long, device=device); y2 = torch.tensor([lab2], dtype=torch.long, device=device)
    return 0.5 * (model(input_ids=inp1, labels=y1).loss + model(input_ids=inp2, labels=y2).loss)


def train_items(model, tokenizer, items, device, lr: float, batch_items: int, rng: random.Random, phase: str):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    logs = []
    accum = []
    t0 = time.time()
    for idx, (kind, obj, words) in enumerate(items):
        if kind == 'binding':
            loss = binding_loss(model, tokenizer, obj, device)
        else:
            loss = official_loss(model, tokenizer, obj, device, rng)
        if loss is None:
            continue
        accum.append(loss)
        if len(accum) >= batch_items or idx == len(items)-1:
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.stack(accum).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if len(logs) < 5 or len(logs) % 10 == 0:
                logs.append({'phase': phase, 'update': len(logs), 'loss': float(batch_loss.detach().cpu()), 'elapsed': time.time()-t0})
            accum = []
    return logs


def evaluate_sets(model, tokenizer, eval_sets, device):
    out = {}
    for name, pairs in eval_sets.items():
        agg, rows = measure_binding_pairs(model, tokenizer, pairs, device)
        out[name] = agg
    return out


def run_one(start_ckpt: str, ratio: float, args, tokenizer, train_pairs, eval_sets, official_pool, device, seed_offset: int = 0):
    rng = random.Random(args.seed + seed_offset + int(ratio * 10000))
    ckpt_path = MODEL_ROOT / start_ckpt
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path.resolve()), trust_remote_code=True).to(device)
    pre = evaluate_sets(model, tokenizer, eval_sets, device)
    items, exposure = make_training_items(train_pairs, official_pool, ratio, args.total_words, rng)
    train_logs = train_items(model, tokenizer, items, device, args.lr, args.batch_items, rng, phase='mixed_density')
    post_mixed = evaluate_sets(model, tokenizer, eval_sets, device)
    # persistence under official-only continuation with matched additional word exposure
    persist_items, persist_exposure = make_training_items(train_pairs, official_pool, 0.0, args.persistence_words, rng)
    persist_logs = train_items(model, tokenizer, persist_items, device, args.lr, args.batch_items, rng, phase='official_persistence')
    post_persist = evaluate_sets(model, tokenizer, eval_sets, device)
    del model
    torch.cuda.empty_cache()
    return {'start_ckpt': start_ckpt, 'ratio': ratio, 'pre': pre, 'mixed_exposure': exposure, 'train_logs_tail': train_logs[-5:], 'post_mixed': post_mixed, 'persistence_exposure': persist_exposure, 'persist_logs_tail': persist_logs[-5:], 'post_persist': post_persist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', default='1.0,0.2,0.05,0.01,0.0')
    ap.add_argument('--start_ckpts', default='chck_1M,chck_100M')
    ap.add_argument('--total_words', type=int, default=20000)
    ap.add_argument('--persistence_words', type=int, default=20000)
    ap.add_argument('--n_train_pairs', type=int, default=200)
    ap.add_argument('--n_eval_pairs', type=int, default=80)
    ap.add_argument('--batch_items', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()
    env_setup(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str((MODEL_ROOT/'chck_100M').resolve()), use_fast=True)
    inv = build_inventories(tokenizer)
    rng = random.Random(args.seed)
    train_pairs = generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'train', rng, n_pairs=args.n_train_pairs)
    eval_sets = {
        'train': train_pairs[:args.n_eval_pairs],
        'heldout_entities': generate_pairs({'entities': inv['heldout_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'heldout_entities', rng, n_pairs=args.n_eval_pairs),
        'heldout_values': generate_pairs({'entities': inv['train_entities'], 'values': inv['heldout_values']}, TRAIN_TEMPLATES, 'heldout_values', rng, n_pairs=args.n_eval_pairs),
        'heldout_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, HELDOUT_TEMPLATES, 'heldout_templates', rng, n_pairs=args.n_eval_pairs),
        'heldout_order_flip': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'heldout_order_flip', rng, n_pairs=args.n_eval_pairs, flip_order=True),
        'natural_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, NATURAL_TEMPLATES, 'natural_templates', rng, n_pairs=args.n_eval_pairs),
    }
    official_pool = official_examples(args, OUT_DIR, max_words=max(200000, args.total_words*20 + args.persistence_words*20))
    ratios = [float(x) for x in args.ratios.split(',') if x.strip()]
    start_ckpts = [x.strip() for x in args.start_ckpts.split(',') if x.strip()]
    results = []
    for si, ckpt in enumerate(start_ckpts):
        for ri, ratio in enumerate(ratios):
            print(json.dumps({'event':'start_arm','ckpt':ckpt,'ratio':ratio}), flush=True)
            res = run_one(ckpt, ratio, args, tokenizer, train_pairs, eval_sets, official_pool, device, seed_offset=1000*si+ri)
            results.append(res)
            key = res['post_mixed']['heldout_templates']['both_correct_frac']
            ret = res['post_persist']['heldout_templates']['both_correct_frac']
            print(json.dumps({'event':'done_arm','ckpt':ckpt,'ratio':ratio,'heldout_templates_post_mixed':key,'heldout_templates_post_persist':ret,'exposure':res['mixed_exposure']}), flush=True)
    payload = {'status':'BSM_DENSITY_PERSISTENCE', 'params': vars(args), 'inventories': {k:list(v.keys()) for k,v in inv.items()}, 'eval_counts': {k:len(v) for k,v in eval_sets.items()}, 'results': results, 'elapsed_sec': time.time()-t0}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    # Markdown summary
    lines = ['# research — BSM exposure-matched density and persistence probe','',f'Evidence JSON: `{OUT_JSON}`','',f'Total mixed exposure target per arm: {args.total_words} words; official-only persistence exposure: {args.persistence_words} words.','', '| start | ratio | actual ratio | train post | tmpl post | natural post | tmpl after official | natural after official |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        ex = r['mixed_exposure']
        pm = r['post_mixed']; pp = r['post_persist']
        lines.append(f"| {r['start_ckpt']} | {r['ratio']:.3f} | {ex['actual_binding_ratio']:.3f} | {pm['train']['both_correct_frac']:.3f} | {pm['heldout_templates']['both_correct_frac']:.3f} | {pm['natural_templates']['both_correct_frac']:.3f} | {pp['heldout_templates']['both_correct_frac']:.3f} | {pp['natural_templates']['both_correct_frac']:.3f} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'event':'done','out':str(OUT_JSON),'note':str(OUT_NOTE),'elapsed_sec':payload['elapsed_sec']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
