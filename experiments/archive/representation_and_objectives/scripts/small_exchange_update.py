#!/usr/bin/env python3
"""research: minimal representation-forming update for exchange transfer bridge.

This is deliberately not an endpoint-scale run and not a submission candidate.  It
trains only on the research synthetic role-switch training grammar (train families,
train templates) and evaluates on:
  * research naturalistic exchange bridge (never used for training),
  * research synthetic packet suite (to verify acquisition),
  * a fixed ordinary MLM probe on clean compact-view corpus rows (to check whether
    the update destroys ordinary MLM behavior).

The scientific question is: can centered
exchange credit improve naturalistic unseen constructions beyond ordinary packet CE
controls, while preserving ordinary MLM behavior enough to justify a future careful
mechanism construction?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
WS = ROOT / 'experiments/archive/representation_and_objectives'
BRIDGE_READER = WS / 'scripts/bridge_readout.py'
PAIRS = WS / 'data/role_switch_expanded_packets/scoring_pairs.jsonl'
COMMON_CORPUS = WS / 'data/shared_intersection_tokenizer/common_intersection_9977920w.jsonl'
OUT_ROOT = WS / 'data/small_exchange_update'

TARGETS: dict[str, dict[str, Any]] = {
    'anchor_80M': {
        'model_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'tokenizer_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'description': 'legal40k fixed-256 compact-view anchor, 80M',
    },
    'reheat_100M': {
        'model_root': WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'reheat-cosine 100M broad-learning reference',
    },
}
TRAIN_FAMILIES = {'spatial', 'transfer', 'comparative', 'state_change'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_bridge_mod():
    spec = importlib.util.spec_from_file_location('bridge_readout_mod', BRIDGE_READER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def filter_train_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in pairs if p.get('family') in TRAIN_FAMILIES and p.get('style') == 'train']


def build_case_maps(bridge_mod, pairs: list[dict[str, Any]], tokenizer, max_len: int):
    cases, meta = bridge_mod.make_cases(pairs, tokenizer, max_len)
    by_pair: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in cases:
        if c['ctx_key'] in ('AB', 'BA'):
            by_pair[int(c['local_pair_index'])].append(c)
    complete_indices = [i for i in range(len(pairs)) if len(by_pair.get(i, [])) == 4]
    return cases, by_pair, complete_indices, meta


def pad_batch(cases: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c['input_ids']) for c in cases)
    ids, attn = [], []
    for c in cases:
        pad_n = max_len - len(c['input_ids'])
        ids.append(c['input_ids'] + [pad_id] * pad_n)
        attn.append(c['attention_mask'] + [0] * pad_n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def score_loss_for_pairs(model, pairs: list[dict[str, Any]], cases: list[dict[str, Any]], device: str,
                         margin_target: float, objective: str) -> tuple[torch.Tensor, dict[str, float]]:
    pad_id = 0
    # pad id is already in input ids only for short dynamic pads; any actual pad is masked out by attention.
    input_cpu, attn_cpu = pad_batch(cases, pad_id)
    out = model(input_ids=input_cpu.to(device), attention_mask=attn_cpu.to(device), return_dict=True)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)
    score: dict[tuple[int, str, str], torch.Tensor] = {}
    for bi, c in enumerate(cases):
        s = None
        for pos, tid in zip(c['target_positions'], c['target_ids']):
            val = log_probs[bi, int(pos), int(tid)]
            s = val if s is None else s + val
        assert s is not None
        score[(int(c['local_pair_index']), c['ctx_key'], c['alt_key'])] = s

    Ms, boths, cterms, fterms = [], [], [], []
    mabs, mbas = [], []
    for i, p in enumerate(pairs):
        s_AB_0 = score[(i, 'AB', 'alt0')]
        s_AB_1 = score[(i, 'AB', 'alt1')]
        s_BA_0 = score[(i, 'BA', 'alt0')]
        s_BA_1 = score[(i, 'BA', 'alt1')]
        if p['correct_AB'] == p['alt_0']:
            margin_AB = s_AB_0 - s_AB_1
            margin_BA = s_BA_1 - s_BA_0
            ce_ab = -s_AB_0
            ce_ba = -s_BA_1
        else:
            margin_AB = s_AB_1 - s_AB_0
            margin_BA = s_BA_0 - s_BA_1
            ce_ab = -s_AB_1
            ce_ba = -s_BA_0
        fixed_alt = 'alt0' if int(p['pair_id']) % 2 == 0 else 'alt1'
        fterms.append(-(score[(i, 'AB', fixed_alt)] + score[(i, 'BA', fixed_alt)]) / 2.0)
        cterms.append((ce_ab + ce_ba) / 2.0)
        M = margin_AB + margin_BA
        Ms.append(M)
        mabs.append(margin_AB)
        mbas.append(margin_BA)
        boths.append(((margin_AB.detach() > 0) & (margin_BA.detach() > 0)).float())
    M_t = torch.stack(Ms)
    correct_ce = torch.stack(cterms).mean()
    fixed_ce = torch.stack(fterms).mean()
    exchange = F.softplus(torch.tensor(float(margin_target), device=device) - M_t).mean()
    if objective == 'exchange':
        loss = exchange
    elif objective == 'correct_ce':
        loss = correct_ce
    elif objective == 'fixed_ce':
        loss = fixed_ce
    else:
        raise ValueError(objective)
    stats = {
        'loss': float(loss.detach().cpu()),
        'exchange_loss': float(exchange.detach().cpu()),
        'correct_ce': float(correct_ce.detach().cpu()),
        'fixed_ce': float(fixed_ce.detach().cpu()),
        'M_mean': float(M_t.detach().mean().cpu()),
        'both_correct': float(torch.stack(boths).mean().cpu()),
        'margin_AB_mean': float(torch.stack(mabs).detach().mean().cpu()),
        'margin_BA_mean': float(torch.stack(mbas).detach().mean().cpu()),
    }
    return loss, stats


def select_batch_indices(complete_indices: list[int], batch_pairs: int, rng: random.Random) -> list[int]:
    if batch_pairs >= len(complete_indices):
        return list(complete_indices)
    return rng.sample(complete_indices, batch_pairs)


def make_selected_cases(selected_indices: list[int], by_pair: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    cases: list[dict[str, Any]] = []
    for old in selected_indices:
        for c in by_pair[old]:
            cc = dict(c)
            cc['local_pair_index'] = old_to_new[old]
            cc['case_id'] = len(cases)
            cases.append(cc)
    return cases, old_to_new


def sample_clean_texts(n: int, seed: int, max_scan: int = 20000) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with COMMON_CORPUS.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            text = json.loads(line)['text']
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = text
            if seen >= max_scan:
                break
    return reservoir


def build_clean_mlm_cases(tokenizer, n_texts: int, seed: int, max_len: int, mask_frac: float) -> dict[str, Any]:
    rng = random.Random(seed)
    texts = sample_clean_texts(n_texts, seed + 17)
    cases = []
    special = set(x for x in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id,
                              tokenizer.bos_token_id, tokenizer.eos_token_id] if x is not None)
    for idx, text in enumerate(texts):
        enc = tokenizer(text, return_attention_mask=True, truncation=True, max_length=max_len, add_special_tokens=True)
        ids = list(enc['input_ids'])
        candidates = [i for i, tid in enumerate(ids) if tid not in special]
        if not candidates:
            continue
        k = max(1, int(round(mask_frac * len(candidates))))
        positions = sorted(rng.sample(candidates, min(k, len(candidates))))
        target_ids = [int(ids[i]) for i in positions]
        masked = list(ids)
        for pos in positions:
            masked[pos] = int(tokenizer.mask_token_id)
        cases.append({'case_id': idx, 'input_ids': masked, 'attention_mask': list(enc['attention_mask']),
                      'positions': positions, 'target_ids': target_ids, 'n_targets': len(target_ids)})
    return {'cases': cases, 'texts_sampled': len(texts), 'seed': seed, 'mask_frac': mask_frac, 'max_len': max_len}


def clean_mlm_loss(model, tokenizer, clean_bundle: dict[str, Any], device: str, batch_size: int) -> dict[str, Any]:
    cases = clean_bundle['cases']
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    total_loss = 0.0
    total_tok = 0
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            max_len = max(len(c['input_ids']) for c in batch)
            ids, attn = [], []
            for c in batch:
                pad_n = max_len - len(c['input_ids'])
                ids.append(c['input_ids'] + [int(pad_id)] * pad_n)
                attn.append(c['attention_mask'] + [0] * pad_n)
            out = model(input_ids=torch.tensor(ids, dtype=torch.long, device=device),
                        attention_mask=torch.tensor(attn, dtype=torch.long, device=device), return_dict=True)
            log_probs = F.log_softmax(out.logits.float(), dim=-1)
            for bi, c in enumerate(batch):
                for pos, tid in zip(c['positions'], c['target_ids']):
                    total_loss += float(-log_probs[bi, int(pos), int(tid)].item())
                    total_tok += 1
    return {'loss_per_masked_token': total_loss / total_tok if total_tok else float('nan'),
            'n_cases': len(cases), 'n_masked_tokens': total_tok,
            'texts_sampled': clean_bundle.get('texts_sampled'), 'seed': clean_bundle.get('seed')}


def eval_pairs(model, tokenizer, bridge_mod, pairs: list[dict[str, Any]], device: str, max_len: int, batch_size: int, label: str) -> dict[str, Any]:
    cases, case_meta = bridge_mod.make_cases(pairs, tokenizer, max_len)
    scores = bridge_mod.score_cases(model, cases, tokenizer, device, batch_size)
    rows = bridge_mod.materialize_rows(pairs, scores)
    summary = bridge_mod.summarize_rows(rows)
    summary['label'] = label
    summary['cases'] = case_meta
    return {'rows': rows, 'summary': summary}


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def readiness(base: str) -> dict[str, Any]:
    meta = TARGETS[base]
    mr = pathlib.Path(meta['model_root'])
    tr = pathlib.Path(meta['tokenizer_root'])
    return {
        'base': base,
        'model_root': str(mr),
        'tokenizer_root': str(tr),
        'model_ready': bool((mr / 'model.safetensors').exists()),
        'tokenizer_ready': bool((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists()),
        'pairs_ready': PAIRS.exists(),
        'bridge_pairs_ready': (WS / 'data/naturalistic_exchange_bridge/naturalistic_exchange_bridge_pairs.jsonl').exists(),
        'common_corpus_ready': COMMON_CORPUS.exists(),
        'bridge_reader_ready': BRIDGE_READER.exists(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', choices=sorted(TARGETS), default='anchor_80M')
    ap.add_argument('--objective', choices=['exchange', 'correct_ce', 'fixed_ce'], default='exchange')
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--updates', type=int, default=80)
    ap.add_argument('--batch-pairs', type=int, default=32)
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=151701)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--eval-batch-size', type=int, default=256)
    ap.add_argument('--clean-probe-texts', type=int, default=384)
    ap.add_argument('--clean-mask-frac', type=float, default=0.15)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--stage', choices=['ready', 'run'], default='run')
    args = ap.parse_args()

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        out_dir = OUT_ROOT / f"{args.base}_{args.objective}_lr{args.lr:g}_u{args.updates}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ready = readiness(args.base)
    (out_dir / 'readiness.json').write_text(json.dumps(ready, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'readiness', 'ready': ready, 'out_dir': str(out_dir)}, indent=2), flush=True)
    if args.stage == 'ready':
        return
    if not all(ready[k] for k in ['model_ready','tokenizer_ready','pairs_ready','bridge_pairs_ready','common_corpus_ready','bridge_reader_ready']):
        raise RuntimeError({'not_ready': ready})

    t0 = time.time()
    bridge_mod = load_bridge_mod()
    all_synth_pairs = read_jsonl(PAIRS)
    train_pairs = filter_train_pairs(all_synth_pairs)
    bridge_pairs = bridge_mod.read_pairs(WS / 'data/naturalistic_exchange_bridge/naturalistic_exchange_bridge_pairs.jsonl')

    meta = TARGETS[args.base]
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), torch_dtype=torch.float32, trust_remote_code=True).to(device)
    model.train()

    train_cases_all, train_by_pair, complete_indices, train_case_meta = build_case_maps(bridge_mod, train_pairs, tokenizer, args.max_len)
    if len(complete_indices) != len(train_pairs):
        print(json.dumps({'event': 'train_pair_incomplete_warning', 'complete': len(complete_indices), 'total': len(train_pairs)}), flush=True)
    clean_bundle = build_clean_mlm_cases(tokenizer, args.clean_probe_texts, seed=args.seed + 1009, max_len=args.max_len, mask_frac=args.clean_mask_frac)

    # Baseline readouts for this exact tokenizer/model before update.
    model.eval()
    before_clean = clean_mlm_loss(model, tokenizer, clean_bundle, device, args.eval_batch_size)
    before_bridge = eval_pairs(model, tokenizer, bridge_mod, bridge_pairs, device, args.max_len, args.eval_batch_size, 'bridge_before')
    before_synth_train = eval_pairs(model, tokenizer, bridge_mod, train_pairs, device, args.max_len, args.eval_batch_size, 'synth_train_before')
    model.train()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train_log = []
    for step in range(1, args.updates + 1):
        sel_old = select_batch_indices(complete_indices, args.batch_pairs, rng)
        cases, old_to_new = make_selected_cases(sel_old, train_by_pair)
        batch_pairs = [train_pairs[old] for old in sel_old]
        # Need local pair ids compacted to match reindexed cases.
        batch_pairs = [dict(p) for p in batch_pairs]
        opt.zero_grad(set_to_none=True)
        loss, stats = score_loss_for_pairs(model, batch_pairs, cases, device, args.margin, args.objective)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        opt.step()
        if step <= 5 or step % max(1, args.updates // 10) == 0 or step == args.updates:
            rec = {'step': step, **stats, 'grad_norm_before_clip': grad_norm}
            train_log.append(rec)
            print(json.dumps({'event': 'train_step', **rec}), flush=True)

    model.eval()
    after_clean = clean_mlm_loss(model, tokenizer, clean_bundle, device, args.eval_batch_size)
    after_bridge = eval_pairs(model, tokenizer, bridge_mod, bridge_pairs, device, args.max_len, args.eval_batch_size, 'bridge_after')
    after_synth_train = eval_pairs(model, tokenizer, bridge_mod, train_pairs, device, args.max_len, args.eval_batch_size, 'synth_train_after')
    after_synth_all = eval_pairs(model, tokenizer, bridge_mod, all_synth_pairs, device, args.max_len, args.eval_batch_size, 'synth_all_after')

    # Save model for possible downstream readout only. This is a disposable research artifact.
    ckpt_dir = out_dir / 'hf_model' / 'chck_step151_final'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt_dir))

    rows_dir = out_dir / 'rows'
    write_rows(rows_dir / 'bridge_before_rows.jsonl', before_bridge['rows'])
    write_rows(rows_dir / 'bridge_after_rows.jsonl', after_bridge['rows'])
    write_rows(rows_dir / 'synth_train_before_rows.jsonl', before_synth_train['rows'])
    write_rows(rows_dir / 'synth_train_after_rows.jsonl', after_synth_train['rows'])
    write_rows(rows_dir / 'synth_all_after_rows.jsonl', after_synth_all['rows'])

    def delta_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        keys = ['M_mean','M_pos_frac','CM_mean','bias_abs_mean','acc_AB','acc_BA','both_correct']
        d = {'overall': {k: after['overall'][k] - before['overall'][k] for k in keys}, 'by_family_both': {}, 'by_family_M': {}}
        for fam in sorted(set(before.get('by_family', {})) & set(after.get('by_family', {}))):
            d['by_family_both'][fam] = after['by_family'][fam]['both_correct'] - before['by_family'][fam]['both_correct']
            d['by_family_M'][fam] = after['by_family'][fam]['M_mean'] - before['by_family'][fam]['M_mean']
        return d

    summary = {
        'status': 'SMALL_EXCHANGE_UPDATE',
        'created_utc': now_utc(),
        'scientific_question': 'Can a small representation-forming update trained only on research packet grammar improve the evaluation-independent naturalistic bridge while preserving ordinary MLM behavior?',
        'args': vars(args),
        'out_dir': str(out_dir),
        'base': args.base,
        'base_description': meta['description'],
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'device': device,
        'train_pairs': {'n': len(train_pairs), 'families': sorted(TRAIN_FAMILIES), 'case_meta': train_case_meta},
        'bridge_pairs': {'n': len(bridge_pairs), 'path': str(WS / 'data/naturalistic_exchange_bridge/naturalistic_exchange_bridge_pairs.jsonl')},
        'clean_probe': {'before': before_clean, 'after': after_clean, 'delta_loss_per_masked_token': after_clean['loss_per_masked_token'] - before_clean['loss_per_masked_token']},
        'before': {
            'bridge': before_bridge['summary'],
            'synth_train': before_synth_train['summary'],
        },
        'after': {
            'bridge': after_bridge['summary'],
            'synth_train': after_synth_train['summary'],
            'synth_all': after_synth_all['summary'],
        },
        'deltas': {
            'bridge': delta_summary(before_bridge['summary'], after_bridge['summary']),
            'synth_train': delta_summary(before_synth_train['summary'], after_synth_train['summary']),
        },
        'train_log': train_log,
        'checkpoint_dir': str(ckpt_dir),
        'rows_dir': str(rows_dir),
        'elapsed_sec': time.time() - t0,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/small_exchange_update.py')),
        'tokenizer_json_sha256': sha256_file(tok_root / 'tokenizer.json'),
    }
    spath = out_dir / 'small_update_summary.json'
    spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'SMALL_UPDATE_DONE', 'summary': str(spath),
                      'objective': args.objective, 'base': args.base,
                      'bridge_delta': summary['deltas']['bridge']['overall'],
                      'clean_loss_delta': summary['clean_probe']['delta_loss_per_masked_token'],
                      'checkpoint_dir': str(ckpt_dir)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
