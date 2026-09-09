#!/usr/bin/env python3
"""research: gradient anatomy of a centered packet exchange loss.

This trains nothing.  It asks whether the principled four-cell exchange objective
recommended in research would apply a genuinely contextual gradient, or whether it
mostly degenerates into ordinary target-token/output calibration already covered by
failed packet and adapter routes.

For a stratified subset of research packet pairs, compare gradients from:
  * exchange:  mean softplus(margin - M), where M is the four-cell interaction;
  * correct_ce: ordinary CE on the correct AB and BA consequences;
  * fixed_ce: role-fixed CE using one pair-fixed alternative in both contexts.

If exchange is a viable next mechanism, its gradient should be less aligned with
ordinary CE/fixed CE and should place appreciable mass in contextual encoder layers;
its output bias gradient should cancel by construction.  The result is only a
mechanistic readout, not a downstream score.
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
import re
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
LAYER_SCRIPT = A01_WS / 'scripts/layerwise_packet_M_decomposition.py'
OUT_ROOT = A01_WS / 'data/exchange_gradient_anatomy'

TARGETS = {
    'anchor_80M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'description': 'legal40k fixed-256 compact-view anchor, 80M',
    },
    'anchor_100M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'description': 'legal40k fixed-256 compact-view anchor, 100M',
    },
    'reheat_100M': {
        'model_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'reheat-cosine 100M broad-learning reference',
    },
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_layer_mod():
    spec = importlib.util.spec_from_file_location('layer_mod', LAYER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pad_batch(cases: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c['input_ids']) for c in cases)
    ids, attn = [], []
    for c in cases:
        pad_n = max_len - len(c['input_ids'])
        ids.append(c['input_ids'] + [pad_id] * pad_n)
        attn.append(c['attention_mask'] + [0] * pad_n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def build_subset_and_cases(layer_mod, tokenizer, *, max_per_group: int, seed: int, include_temporal: bool, max_len: int):
    pairs_all = layer_mod.read_pairs(layer_mod.PAIRS_PATH)
    pairs, selection_meta = layer_mod.select_stratified_pairs(
        pairs_all, tokenizer, max_per_group=max_per_group, seed=seed,
        one_token_only=False, include_temporal=include_temporal)
    cases_all, case_meta = layer_mod.make_cases(pairs, tokenizer, max_len)
    cases = [c for c in cases_all if c['ctx_key'] in ('AB', 'BA')]
    # Keep only complete pairs for AB/BA alternatives.
    cc = collections.Counter((int(c['local_pair_index']), c['ctx_key'], c['alt_key']) for c in cases)
    keep_pair_indices = []
    for i in range(len(pairs)):
        if all(cc[(i, ctx, alt)] == 1 for ctx in ('AB', 'BA') for alt in ('alt0', 'alt1')):
            keep_pair_indices.append(i)
    keep = set(keep_pair_indices)
    cases = [c for c in cases if int(c['local_pair_index']) in keep]
    pairs = [p for i, p in enumerate(pairs) if i in keep]
    # Reindex local_pair_index to compact order.
    old_to_new = {old: new for new, old in enumerate(keep_pair_indices)}
    for c in cases:
        c['local_pair_index'] = old_to_new[int(c['local_pair_index'])]
    meta = {
        'selection': selection_meta,
        'cases_original': case_meta,
        'n_pairs_for_gradient': len(pairs),
        'n_cases_for_gradient': len(cases),
        'family_counts': dict(sorted(collections.Counter(p['family'] for p in pairs).items())),
        'style_counts': dict(sorted(collections.Counter(p['style'] for p in pairs).items())),
    }
    return pairs, cases, meta


def compute_scores_and_losses(model, tokenizer, pairs, cases, device: str, margin: float) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_cpu, attn_cpu = pad_batch(cases, int(pad_id))
    input_ids = input_cpu.to(device)
    attention_mask = attn_cpu.to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = out.logits
    log_probs = F.log_softmax(logits.float(), dim=-1)

    score: dict[tuple[int, str, str], torch.Tensor] = {}
    token_len: dict[tuple[int, str, str], int] = {}
    for bi, c in enumerate(cases):
        s = None
        for pos, tid in zip(c['target_positions'], c['target_ids']):
            val = log_probs[bi, int(pos), int(tid)]
            s = val if s is None else s + val
        assert s is not None
        key = (int(c['local_pair_index']), c['ctx_key'], c['alt_key'])
        score[key] = s
        token_len[key] = len(c['target_ids'])

    Ms, boths, mABs, mBAs = [], [], [], []
    correct_terms, fixed_terms = [], []
    row_metrics = []
    for i, p in enumerate(pairs):
        s_AB_0 = score[(i, 'AB', 'alt0')]
        s_AB_1 = score[(i, 'AB', 'alt1')]
        s_BA_0 = score[(i, 'BA', 'alt0')]
        s_BA_1 = score[(i, 'BA', 'alt1')]
        alt0, alt1 = p['alt_0'], p['alt_1']
        if p['correct_AB'] == alt0:
            margin_AB = s_AB_0 - s_AB_1
            margin_BA = s_BA_1 - s_BA_0
            ce_ab = -s_AB_0
            ce_ba = -s_BA_1
        else:
            margin_AB = s_AB_1 - s_AB_0
            margin_BA = s_BA_0 - s_BA_1
            ce_ab = -s_AB_1
            ce_ba = -s_BA_0
        fixed_alt = 'alt0' if (int(p['pair_id']) % 2 == 0) else 'alt1'
        fixed_terms.append(-(score[(i, 'AB', fixed_alt)] + score[(i, 'BA', fixed_alt)]) / 2.0)
        correct_terms.append((ce_ab + ce_ba) / 2.0)
        M = margin_AB + margin_BA
        Ms.append(M)
        mABs.append(margin_AB)
        mBAs.append(margin_BA)
        both = (margin_AB.detach() > 0 and margin_BA.detach() > 0)
        boths.append(1.0 if both else 0.0)
        row_metrics.append({
            'pair_id': p.get('pair_id'), 'family': p.get('family'), 'style': p.get('style'),
            'M': float(M.detach().cpu()), 'margin_AB': float(margin_AB.detach().cpu()),
            'margin_BA': float(margin_BA.detach().cpu()), 'both_correct': 1.0 if both else 0.0,
        })
    M_t = torch.stack(Ms)
    exchange_loss = F.softplus(torch.tensor(float(margin), device=M_t.device) - M_t).mean()
    correct_ce = torch.stack(correct_terms).mean()
    fixed_ce = torch.stack(fixed_terms).mean()
    score_summary = {
        'n_pairs': len(pairs),
        'M_mean': float(M_t.detach().mean().cpu()),
        'M_pos_frac': float((M_t.detach() > 0).float().mean().cpu()),
        'margin_AB_mean': float(torch.stack(mABs).detach().mean().cpu()),
        'margin_BA_mean': float(torch.stack(mBAs).detach().mean().cpu()),
        'both_correct': sum(boths) / len(boths) if boths else float('nan'),
        'exchange_loss_value': float(exchange_loss.detach().cpu()),
        'correct_ce_value': float(correct_ce.detach().cpu()),
        'fixed_ce_value': float(fixed_ce.detach().cpu()),
    }
    return {
        'losses': {'exchange': exchange_loss, 'correct_ce': correct_ce, 'fixed_ce': fixed_ce},
        'score_summary': score_summary,
        'row_metrics': row_metrics,
    }


def group_name(name: str) -> str:
    m = re.search(r'deberta\.encoder\.layer\.(\d+)\.', name)
    if m:
        return f'encoder_layer_{int(m.group(1)):02d}'
    if name.startswith('deberta.embeddings'):
        return 'embeddings'
    if name.startswith('cls.'):
        if name.endswith('bias') or '.bias' in name:
            return 'mlm_head_biases'
        return 'mlm_head_weights'
    return 'other'


def collect_grad_vector_and_groups(model) -> tuple[torch.Tensor, dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    chunks = []
    group_norm2: dict[str, float] = collections.defaultdict(float)
    group_numel: dict[str, int] = collections.defaultdict(int)
    group_chunks: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    zero_param_names = []
    for name, p in model.named_parameters():
        if p.grad is None:
            g = torch.zeros_like(p, dtype=torch.float32, device='cpu').reshape(-1)
            zero_param_names.append(name)
        else:
            g = p.grad.detach().float().cpu().reshape(-1)
        chunks.append(g)
        gn = group_name(name)
        group_chunks[gn].append(g)
        group_norm2[gn] += float(torch.dot(g, g).item())
        group_numel[gn] += g.numel()
    vec = torch.cat(chunks) if chunks else torch.empty(0)
    groups = {gn: {'norm': math.sqrt(group_norm2[gn]), 'numel': group_numel[gn]} for gn in sorted(group_norm2)}
    group_vecs = {gn: torch.cat(chs) for gn, chs in group_chunks.items()}
    total_norm = float(torch.linalg.vector_norm(vec).item()) if vec.numel() else 0.0
    for v in groups.values():
        v['fraction_total_norm2'] = (v['norm'] ** 2) / (total_norm ** 2) if total_norm > 0 else float('nan')
    groups['_total'] = {'norm': total_norm, 'numel': int(vec.numel()), 'zero_grad_param_count': len(zero_param_names)}
    return vec, groups, group_vecs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if float(na) == 0.0 or float(nb) == 0.0:
        return float('nan')
    return float(torch.dot(a, b).item() / (float(na.item()) * float(nb.item())))


def run_target(label: str, meta: dict[str, Any], args: argparse.Namespace, layer_mod) -> dict[str, Any]:
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    pairs, cases, subset_meta = build_subset_and_cases(layer_mod, tokenizer, max_per_group=args.max_per_group,
                                                       seed=args.seed, include_temporal=args.include_temporal,
                                                       max_len=args.max_len)
    if args.stage == 'inspect':
        return {'target': label, 'model_root': str(model_root), 'tokenizer_root': str(tok_root),
                'description': meta['description'], 'subset': subset_meta}
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), torch_dtype=torch.float32, trust_remote_code=True).to(device)
    model.eval()
    print(json.dumps({'event': 'target_start', 'target': label, 'device': device, 'pairs': len(pairs), 'cases': len(cases), 'model_root': str(model_root)}), flush=True)

    grad_vectors: dict[str, torch.Tensor] = {}
    grad_groups: dict[str, Any] = {}
    group_vecs_by_obj: dict[str, dict[str, torch.Tensor]] = {}
    score_summary: dict[str, Any] | None = None
    row_metrics: list[dict[str, Any]] | None = None
    loss_values: dict[str, float] = {}
    for obj_name in ['exchange', 'correct_ce', 'fixed_ce']:
        model.zero_grad(set_to_none=True)
        bundle = compute_scores_and_losses(model, tokenizer, pairs, cases, device, args.margin)
        if score_summary is None:
            score_summary = bundle['score_summary']
            row_metrics = bundle['row_metrics']
        loss = bundle['losses'][obj_name]
        loss.backward()
        vec, groups, group_vecs = collect_grad_vector_and_groups(model)
        grad_vectors[obj_name] = vec
        grad_groups[obj_name] = groups
        group_vecs_by_obj[obj_name] = group_vecs
        loss_values[obj_name] = float(loss.detach().cpu())
        print(json.dumps({'event': 'objective_done', 'target': label, 'objective': obj_name,
                          'loss': loss_values[obj_name], 'total_grad_norm': groups['_total']['norm']}), flush=True)

    pairwise_cos = {}
    objs = ['exchange', 'correct_ce', 'fixed_ce']
    for i, a in enumerate(objs):
        for b in objs[i+1:]:
            pairwise_cos[f'{a}__{b}'] = cosine(grad_vectors[a], grad_vectors[b])
    exchange_group_cos = {}
    for b in ['correct_ce', 'fixed_ce']:
        gcos = {}
        for gn, gv in group_vecs_by_obj['exchange'].items():
            if gn in group_vecs_by_obj[b]:
                gcos[gn] = cosine(gv, group_vecs_by_obj[b][gn])
        exchange_group_cos[f'exchange__{b}'] = gcos

    target_dir = pathlib.Path(args.out_root) / 'targets'
    target_dir.mkdir(parents=True, exist_ok=True)
    rows_path = target_dir / f'{label}_row_metrics.jsonl'
    with rows_path.open('w', encoding='utf-8') as f:
        for r in (row_metrics or []):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {
        'target': label,
        'description': meta['description'],
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'subset': subset_meta,
        'score_summary': score_summary,
        'loss_values': loss_values,
        'gradient_groups': grad_groups,
        'pairwise_gradient_cosine': pairwise_cos,
        'exchange_group_cosine_with_controls': exchange_group_cos,
        'row_metrics_path': str(rows_path),
        'tokenizer_json_sha256': sha256_file(tok_root / 'tokenizer.json') if (tok_root / 'tokenizer.json').exists() else None,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({'event': 'target_done', 'target': label, 'score_summary': score_summary, 'pairwise_cos': pairwise_cos}, indent=2), flush=True)
    return summary


def readiness(selected: list[str]) -> dict[str, Any]:
    ready = {'targets': {}, 'layer_script': str(LAYER_SCRIPT), 'layer_script_ready': LAYER_SCRIPT.exists()}
    for label in selected:
        meta = TARGETS[label]
        mr = pathlib.Path(meta['model_root'])
        tr = pathlib.Path(meta['tokenizer_root'])
        ready['targets'][label] = {
            'model_root': str(mr), 'tokenizer_root': str(tr),
            'model_ready': bool((mr / 'model.safetensors').exists()),
            'tokenizer_ready': bool((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists()),
            'description': meta['description'],
        }
    ready['selected_ready'] = ready['layer_script_ready'] and all(v['model_ready'] and v['tokenizer_ready'] for v in ready['targets'].values())
    return ready


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'inspect', 'score'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['anchor_80M', 'reheat_100M'])
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--max-per-group', type=int, default=8)
    ap.add_argument('--seed', type=int, default=150031)
    ap.add_argument('--include-temporal', action='store_true')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ready = readiness(args.targets)
    (out_root / 'readiness.json').write_text(json.dumps(ready, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'readiness', 'ready': ready}, indent=2), flush=True)
    if args.stage == 'ready':
        return
    if not ready.get('selected_ready'):
        raise RuntimeError({'not_ready': ready})
    layer_mod = load_layer_mod()
    summaries = {}
    t0 = time.time()
    for label in args.targets:
        summaries[label] = run_target(label, TARGETS[label], args, layer_mod)
    combined = {
        'status': 'EXCHANGE_GRADIENT_ANATOMY_' + args.stage.upper(),
        'created_utc': now_utc(),
        'scientific_question': 'Would centered exchange supervision produce contextual gradient distinct from ordinary CE/fixed-packet CE?',
        'stage': args.stage,
        'args': vars(args),
        'readiness': ready,
        'target_summaries': summaries,
        'elapsed_sec': time.time() - t0,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/exchange_gradient_anatomy.py')),
    }
    out_path = out_root / f'exchange_gradient_anatomy_summary_{args.stage}.json'
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'EXCHANGE_GRADIENT_ANATOMY_DONE', 'summary': str(out_path), 'stage': args.stage}, indent=2), flush=True)


if __name__ == '__main__':
    main()
