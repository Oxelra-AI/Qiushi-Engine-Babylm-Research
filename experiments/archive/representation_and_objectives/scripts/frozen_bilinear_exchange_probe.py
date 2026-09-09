#!/usr/bin/env python3
"""research: frozen diagonal-bilinear exchange probe.

Question: do existing checkpoints contain context-conditioned role-exchange
information in frozen hidden states that is not used by the current MLM head?

For each packet pair and context direction, form a candidate-difference feature

    phi(C; x,y) = (E[x] - E[y]) * h_C

where h_C is the frozen hidden state at the masked target slot and E is the output
embedding/decoder table.  A small diagonal bilinear readout learns one vector w so
that phi @ w is positive when alt_0 is correct and negative when alt_1 is correct.
Training uses only research train-style pairs from train families; evaluation reports
held-out templates and the withheld container/temporal families.  The base model is
frozen and no endpoint is trained.
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
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
LAYER_SCRIPT = A01_WS / 'scripts/layerwise_packet_M_decomposition.py'
OUT_ROOT = A01_WS / 'data/frozen_bilinear_exchange_probe'
TRAIN_FAMILIES = {'spatial', 'transfer', 'comparative', 'state_change'}

TARGETS = {
    'anchor_20M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_20M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_20M',
        'description': 'legal40k fixed-256 compact-view anchor, 20M',
    },
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
    spec = importlib.util.spec_from_file_location('layer_mod_for_probe', LAYER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pad_batch(cases: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c['input_ids']) for c in cases)
    ids, attn = [], []
    for c in cases:
        n = max_len - len(c['input_ids'])
        ids.append(c['input_ids'] + [pad_id] * n)
        attn.append(c['attention_mask'] + [0] * n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def prepare_pairs_cases(layer_mod, tokenizer, *, include_temporal: bool, max_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs_all = layer_mod.read_pairs(layer_mod.PAIRS_PATH)
    # No subsampling: this is still small and we need robust family-level transfer.
    pairs, selection = layer_mod.select_stratified_pairs(
        pairs_all, tokenizer, max_per_group=0, seed=150047,
        one_token_only=False, include_temporal=include_temporal)
    cases_all, case_meta = layer_mod.make_cases(pairs, tokenizer, max_len)
    case_map = {(int(c['local_pair_index']), c['ctx_key'], c['alt_key']): c for c in cases_all if c['ctx_key'] in ('AB', 'BA')}
    usable_pairs, usable_cases = [], []
    reject = collections.Counter()
    for i, p in enumerate(pairs):
        keys = [(i, ctx, alt) for ctx in ('AB', 'BA') for alt in ('alt0', 'alt1')]
        if any(k not in case_map for k in keys):
            reject['missing_case'] += 1
            continue
        # This probe uses a single masked slot and a single decoder embedding per candidate.
        if any(len(case_map[k]['target_ids']) != 1 or len(case_map[k]['target_positions']) != 1 for k in keys):
            reject['multi_token_candidate'] += 1
            continue
        # The AB input should be candidate-neutral after masking, and likewise BA.
        if case_map[(i, 'AB', 'alt0')]['input_ids'] != case_map[(i, 'AB', 'alt1')]['input_ids']:
            reject['AB_masked_inputs_differ'] += 1
            continue
        if case_map[(i, 'BA', 'alt0')]['input_ids'] != case_map[(i, 'BA', 'alt1')]['input_ids']:
            reject['BA_masked_inputs_differ'] += 1
            continue
        new_i = len(usable_pairs)
        usable_pairs.append(p)
        for ctx in ('AB', 'BA'):
            # Keep one context case; record both target ids in the example metadata.
            c0 = dict(case_map[(i, ctx, 'alt0')])
            c1 = case_map[(i, ctx, 'alt1')]
            c0['local_pair_index'] = new_i
            c0['ctx_key'] = ctx
            c0['alt0_id'] = int(c0['target_ids'][0])
            c0['alt1_id'] = int(c1['target_ids'][0])
            c0['label_alt0_correct'] = 1.0 if p[f'correct_{ctx}'] == p['alt_0'] else -1.0
            c0['family'] = p['family']
            c0['style'] = p['style']
            c0['pair_id'] = p['pair_id']
            usable_cases.append(c0)
    meta = {
        'selection': selection,
        'case_meta_before_probe_filter': case_meta,
        'usable_pairs': len(usable_pairs),
        'usable_context_examples': len(usable_cases),
        'reject': dict(reject),
        'usable_family_counts': dict(sorted(collections.Counter(p['family'] for p in usable_pairs).items())),
        'usable_style_counts': dict(sorted(collections.Counter(p['style'] for p in usable_pairs).items())),
    }
    return usable_pairs, usable_cases, meta


def extract_features(model, tokenizer, pairs, cases, *, device: str, batch_size: int) -> tuple[dict[int, dict[str, torch.Tensor]], list[int]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    emb = model.cls.predictions.decoder.weight.detach().float().to(device)
    by_layer: dict[int, dict[str, list[Any]]] = {}
    layer_indices: list[int] | None = None
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            ids_cpu, attn_cpu = pad_batch(batch, int(pad_id))
            ids = ids_cpu.to(device)
            attn = attn_cpu.to(device)
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            hs = list(out.hidden_states)
            if layer_indices is None:
                layer_indices = list(range(len(hs)))
                for li in layer_indices:
                    by_layer[li] = {'X': [], 'y': [], 'pair_index': [], 'ctx_key': [], 'family': [], 'style': [], 'pair_id': []}
            for li, h in enumerate(hs):
                for bi, c in enumerate(batch):
                    q = emb[int(c['alt0_id'])] - emb[int(c['alt1_id'])]
                    pos = int(c['target_positions'][0])
                    feat = (q * h[bi, pos].float()).detach().cpu()
                    by_layer[li]['X'].append(feat)
                    by_layer[li]['y'].append(float(c['label_alt0_correct']))
                    by_layer[li]['pair_index'].append(int(c['local_pair_index']))
                    by_layer[li]['ctx_key'].append(c['ctx_key'])
                    by_layer[li]['family'].append(c['family'])
                    by_layer[li]['style'].append(c['style'])
                    by_layer[li]['pair_id'].append(int(c['pair_id']))
            del out, hs, ids, attn
    assert layer_indices is not None
    packed: dict[int, dict[str, torch.Tensor | list[Any]]] = {}
    for li in layer_indices:
        packed[li] = {
            'X': torch.stack(by_layer[li]['X']).float(),
            'y': torch.tensor(by_layer[li]['y'], dtype=torch.float32),
            'pair_index': by_layer[li]['pair_index'],
            'ctx_key': by_layer[li]['ctx_key'],
            'family': by_layer[li]['family'],
            'style': by_layer[li]['style'],
            'pair_id': by_layer[li]['pair_id'],
        }
    return packed, layer_indices


def train_diag_probe(X: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, *, l2: float, steps: int, lr: float, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    Xtr = X[train_mask]
    ytr = y[train_mask]
    mu = Xtr.mean(dim=0)
    sd = Xtr.std(dim=0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True)
        logits = Xs[train_mask].matmul(w)
        loss = F.softplus(-ytr * logits).mean() + l2 * (w.square().mean())
        loss.backward()
        return loss
    final_loss = float(opt.step(closure).detach().cpu())
    with torch.no_grad():
        logits_all = Xs.matmul(w).detach()
        train_margin = ytr * Xs[train_mask].matmul(w).detach()
    return {
        'w': w.detach(),
        'mu': mu, 'sd': sd,
        'logits': logits_all,
        'train_loss': final_loss,
        'train_margin_mean': float(train_margin.mean().item()),
        'train_context_acc': float((train_margin > 0).float().mean().item()),
        'weight_norm': float(torch.linalg.vector_norm(w.detach()).item()),
    }


def summarize_context(indices: list[int], y: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    if not indices:
        return {'n_context': 0}
    idx = torch.tensor(indices, dtype=torch.long)
    margins = y[idx] * logits[idx]
    return {
        'n_context': len(indices),
        'context_acc': float((margins > 0).float().mean().item()),
        'margin_mean': float(margins.mean().item()),
        'margin_median': float(margins.median().item()),
    }


def summarize_pairs(indices: list[int], payload: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    # Convert context logits into pair-level AB/BA correct margins.
    by_pair = collections.defaultdict(dict)
    y = payload['y']
    for i in indices:
        pair_index = int(payload['pair_index'][i])
        ctx = payload['ctx_key'][i]
        by_pair[pair_index][ctx] = float((y[i] * logits[i]).item())
    Ms, boths = [], []
    for _pi, d in by_pair.items():
        if 'AB' in d and 'BA' in d:
            M = d['AB'] + d['BA']
            Ms.append(M)
            boths.append(1.0 if (d['AB'] > 0 and d['BA'] > 0) else 0.0)
    if not Ms:
        return {'n_pairs': 0}
    return {
        'n_pairs': len(Ms),
        'pair_M_mean': sum(Ms) / len(Ms),
        'pair_both_correct': sum(boths) / len(boths),
        'pair_M_pos_frac': sum(1.0 for m in Ms if m > 0) / len(Ms),
    }


def evaluate_layer(payload: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    y: torch.Tensor = payload['y']
    logits: torch.Tensor = probe['logits']
    train_idx, held_trainfam_idx, container_idx, temporal_idx, all_test_idx = [], [], [], [], []
    for i, (fam, sty) in enumerate(zip(payload['family'], payload['style'])):
        is_train = (sty == 'train' and fam in TRAIN_FAMILIES)
        if is_train:
            train_idx.append(i)
        else:
            all_test_idx.append(i)
            if fam in TRAIN_FAMILIES and sty == 'held_out':
                held_trainfam_idx.append(i)
            if fam == 'container':
                container_idx.append(i)
            if fam == 'temporal':
                temporal_idx.append(i)
    groups = {
        'train_families_train_templates': train_idx,
        'train_families_heldout_templates': held_trainfam_idx,
        'container_heldout_family': container_idx,
        'temporal_excluded_family': temporal_idx,
        'all_nontrain_eval': all_test_idx,
    }
    out = {}
    for name, idx in groups.items():
        out[name] = {**summarize_context(idx, y, logits), **summarize_pairs(idx, payload, logits)}
    fams = sorted(set(payload['family']))
    by_family = {}
    for fam in fams:
        idx = [i for i, f in enumerate(payload['family']) if f == fam]
        by_family[fam] = {**summarize_context(idx, y, logits), **summarize_pairs(idx, payload, logits)}
    out['by_family_all_styles'] = by_family
    out['probe_train_loss'] = probe['train_loss']
    out['probe_train_context_acc'] = probe['train_context_acc']
    out['probe_train_margin_mean'] = probe['train_margin_mean']
    out['probe_weight_norm'] = probe['weight_norm']
    return out


def run_target(label: str, meta: dict[str, Any], args: argparse.Namespace, layer_mod) -> dict[str, Any]:
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    pairs, cases, prep_meta = prepare_pairs_cases(layer_mod, tokenizer, include_temporal=True, max_len=args.max_len)
    if args.stage == 'inspect':
        return {'target': label, 'model_root': str(model_root), 'tokenizer_root': str(tok_root), 'prep': prep_meta}
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32, trust_remote_code=True).to(device).eval()
    print(json.dumps({'event': 'target_start', 'target': label, 'device': device, 'pairs': len(pairs), 'cases': len(cases), 'model_root': str(model_root)}), flush=True)
    feats_by_layer, layer_indices = extract_features(model, tokenizer, pairs, cases, device=device, batch_size=args.batch_size)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_summaries = {}
    for li in layer_indices:
        payload = feats_by_layer[li]
        train_mask = torch.tensor([(sty == 'train' and fam in TRAIN_FAMILIES) for fam, sty in zip(payload['family'], payload['style'])], dtype=torch.bool)
        probe = train_diag_probe(payload['X'], payload['y'], train_mask, l2=args.l2, steps=args.lbfgs_steps, lr=args.lbfgs_lr, seed=args.seed + li)
        layer_summaries[str(li)] = evaluate_layer(payload, probe)
        print(json.dumps({'event': 'layer_done', 'target': label, 'layer': li,
                          'train_acc': layer_summaries[str(li)]['train_families_train_templates']['context_acc'],
                          'held_acc': layer_summaries[str(li)]['train_families_heldout_templates']['context_acc'],
                          'container_acc': layer_summaries[str(li)]['container_heldout_family'].get('context_acc'),
                          'held_pair_both': layer_summaries[str(li)]['train_families_heldout_templates'].get('pair_both_correct'),
                          'container_pair_both': layer_summaries[str(li)]['container_heldout_family'].get('pair_both_correct')}), flush=True)
    # Pick best layer by held-out train-family pair-level both-correct, then context accuracy.
    def key(li: int):
        s = layer_summaries[str(li)]['train_families_heldout_templates']
        return (s.get('pair_both_correct', float('nan')), s.get('context_acc', float('nan')))
    best = max(layer_indices, key=key)
    return {
        'target': label,
        'description': meta['description'],
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'prep': prep_meta,
        'layer_indices': layer_indices,
        'layer_summaries': layer_summaries,
        'best_layer_by_heldout_trainfamily_pair_both': best,
        'best_layer_summary': layer_summaries[str(best)],
        'tokenizer_json_sha256': sha256_file(tok_root / 'tokenizer.json') if (tok_root / 'tokenizer.json').exists() else None,
    }


def readiness(selected: list[str]) -> dict[str, Any]:
    out = {'targets': {}, 'layer_script': str(LAYER_SCRIPT), 'layer_script_ready': LAYER_SCRIPT.exists()}
    for label in selected:
        meta = TARGETS[label]
        mr = pathlib.Path(meta['model_root']); tr = pathlib.Path(meta['tokenizer_root'])
        out['targets'][label] = {
            'model_root': str(mr), 'tokenizer_root': str(tr),
            'model_ready': bool((mr / 'model.safetensors').exists()),
            'tokenizer_ready': bool((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists()),
            'description': meta['description'],
        }
    out['selected_ready'] = out['layer_script_ready'] and all(v['model_ready'] and v['tokenizer_ready'] for v in out['targets'].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'inspect', 'score'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['anchor_80M', 'anchor_100M', 'reheat_100M'])
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--lbfgs-steps', type=int, default=100)
    ap.add_argument('--lbfgs-lr', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=150071)
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
        'status': 'FROZEN_BILINEAR_EXCHANGE_PROBE_' + args.stage.upper(),
        'created_utc': now_utc(),
        'scientific_question': 'Can a small antisymmetric candidate-conditioned readout extract exchange structure from frozen states?',
        'stage': args.stage,
        'args': vars(args),
        'readiness': ready,
        'target_summaries': summaries,
        'elapsed_sec': time.time() - t0,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/frozen_bilinear_exchange_probe.py')),
    }
    out_path = out_root / f'frozen_bilinear_exchange_probe_summary_{args.stage}.json'
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'FROZEN_BILINEAR_EXCHANGE_PROBE_DONE', 'summary': str(out_path), 'stage': args.stage}, indent=2), flush=True)


if __name__ == '__main__':
    main()
