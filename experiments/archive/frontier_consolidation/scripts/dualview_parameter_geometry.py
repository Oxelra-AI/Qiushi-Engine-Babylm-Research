#!/usr/bin/env python3
"""research dual-view 20M parameter-geometry analyzer.

This CPU-only script compares the completed corrected-trainer arms:
  aligned, shuffled, and exact mlm_only.
It does not train or evaluate.  It quantifies whether the dual-view auxiliary
correspondence mostly changes the private adapter pathway or already induces a large
shared-stock/backbone displacement at the 20M decision point.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any

import torch
from safetensors.torch import load_file

USER_ROOT = pathlib.Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
OUT_ROOT = WORKSPACE / 'data/dualview_parameter_geometry'

RUNS = {
    'aligned': WORKSPACE / 'training/runs/dualview_aligned_20M_seed43022/hf_model/final',
    'shuffled': WORKSPACE / 'training/runs/dualview_shuffled_20M_seed43022/hf_model/final',
    'mlm_only': WORKSPACE / 'training/runs/dualview_mlm_only_20M_seed43022/hf_model/final',
}
METRICS = {
    'aligned': WORKSPACE / 'training/runs/dualview_aligned_20M_seed43022/scientific_metrics.json',
    'shuffled': WORKSPACE / 'training/runs/dualview_shuffled_20M_seed43022/scientific_metrics.json',
    'mlm_only': WORKSPACE / 'training/runs/dualview_mlm_only_20M_seed43022/scientific_metrics.json',
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def ckpt_file(root: pathlib.Path) -> pathlib.Path:
    st = root / 'model.safetensors'
    if st.exists():
        return st
    pt = root / 'pytorch_model.bin'
    if pt.exists():
        return pt
    raise FileNotFoundError(root)


def load_state(label: str) -> dict[str, torch.Tensor]:
    p = ckpt_file(RUNS[label])
    if p.suffix == '.safetensors':
        return load_file(str(p), device='cpu')
    return torch.load(str(p), map_location='cpu')


def group_for_key(k: str) -> str:
    if '.adapter.' in k:
        if '.adapter.up.' in k:
            return 'adapter_up'
        if '.adapter.down.' in k:
            return 'adapter_down'
        if '.adapter.norm.' in k or '.adapter.adapter_norm.' in k or '.adapter.LayerNorm.' in k:
            return 'adapter_norm'
        return 'adapter_other'
    if k.startswith('deberta.embeddings'):
        return 'stock_embeddings'
    if '.attention.' in k:
        return 'stock_attention'
    if '.intermediate.' in k or '.output.' in k:
        return 'stock_ffn_output'
    if k.startswith('cls.') or 'lm_predictions' in k:
        return 'stock_lm_head'
    if k.startswith('deberta.encoder'):
        return 'stock_encoder_other'
    return 'stock_other'


def supergroup(g: str) -> str:
    if g.startswith('adapter'):
        return 'adapter_all'
    return 'stock_all'


def zeros_acc() -> dict[str, float]:
    return {'n_tensors': 0.0, 'n_params': 0.0, 'dot': 0.0, 'norm_a2': 0.0, 'norm_b2': 0.0, 'diff2': 0.0, 'abs_sum': 0.0}


def add_stats(acc: dict[str, float], a: torch.Tensor, b: torch.Tensor) -> None:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    d = af - bf
    acc['n_tensors'] += 1
    acc['n_params'] += af.numel()
    acc['dot'] += float(torch.dot(af, bf).item())
    acc['norm_a2'] += float(torch.dot(af, af).item())
    acc['norm_b2'] += float(torch.dot(bf, bf).item())
    acc['diff2'] += float(torch.dot(d, d).item())
    acc['abs_sum'] += float(d.abs().sum().item())


def finalize(acc: dict[str, float]) -> dict[str, Any]:
    na = math.sqrt(max(acc['norm_a2'], 0.0))
    nb = math.sqrt(max(acc['norm_b2'], 0.0))
    nd = math.sqrt(max(acc['diff2'], 0.0))
    cos = None if na == 0 or nb == 0 else acc['dot'] / (na * nb)
    return {
        'n_tensors': int(acc['n_tensors']),
        'n_params': int(acc['n_params']),
        'norm_a': na,
        'norm_b': nb,
        'diff_norm': nd,
        'rel_l2_to_b': None if nb == 0 else nd / nb,
        'cosine': cos,
        'mean_abs_delta': None if acc['n_params'] == 0 else acc['abs_sum'] / acc['n_params'],
    }


def pairwise_metrics(a_label: str, b_label: str, a_state: dict[str, torch.Tensor], b_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    if set(a_state) != set(b_state):
        missing = sorted(set(a_state).symmetric_difference(set(b_state)))[:20]
        raise RuntimeError(f'key mismatch {a_label} vs {b_label}: {missing}')
    by_group: dict[str, dict[str, float]] = defaultdict(zeros_acc)
    top = []
    for k in sorted(a_state):
        a = a_state[k]
        b = b_state[k]
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f'shape mismatch {k}: {tuple(a.shape)} vs {tuple(b.shape)}')
        g = group_for_key(k)
        add_stats(by_group[g], a, b)
        add_stats(by_group[supergroup(g)], a, b)
        add_stats(by_group['all_params'], a, b)
        af = a.detach().float().reshape(-1)
        bf = b.detach().float().reshape(-1)
        d = af - bf
        nd = float(torch.linalg.vector_norm(d).item())
        nb = float(torch.linalg.vector_norm(bf).item())
        rel_l2 = None if nb == 0 else nd / nb
        # keep both absolute and relative leaders; avoid massive output by trimming later.
        top.append({
            'name': k,
            'group': g,
            'n_params': int(af.numel()),
            'diff_norm': nd,
            'rel_l2_to_b': rel_l2,
            'norm_b': nb,
            'mean_abs_delta': float(d.abs().mean().item()) if af.numel() else 0.0,
        })
    group_final = {g: finalize(acc) for g, acc in sorted(by_group.items())}
    top_abs = sorted(top, key=lambda x: x['diff_norm'], reverse=True)[:30]
    top_rel = sorted(top, key=lambda x: (-1.0 if x['rel_l2_to_b'] is None else x['rel_l2_to_b']), reverse=True)[:30]
    return {
        'a': a_label,
        'b': b_label,
        'groups': group_final,
        'top_abs_diff_tensors': top_abs,
        'top_rel_diff_tensors': top_rel,
    }


def state_norms(label: str, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    by_group: dict[str, dict[str, float]] = defaultdict(lambda: {'n_tensors': 0.0, 'n_params': 0.0, 'norm2': 0.0, 'abs_sum': 0.0})
    layer_adapter = defaultdict(lambda: defaultdict(float))
    for k, v in state.items():
        vf = v.detach().float().reshape(-1)
        g = group_for_key(k)
        for gg in [g, supergroup(g), 'all_params']:
            by_group[gg]['n_tensors'] += 1
            by_group[gg]['n_params'] += vf.numel()
            by_group[gg]['norm2'] += float(torch.dot(vf, vf).item())
            by_group[gg]['abs_sum'] += float(vf.abs().sum().item())
        if '.adapter.' in k:
            # typical prefix: deberta.encoder.layer.{i}.adapter....
            parts = k.split('.')
            layer = None
            for i, p in enumerate(parts[:-1]):
                if p == 'layer' and i + 1 < len(parts):
                    try:
                        layer = int(parts[i+1])
                    except Exception:
                        layer = None
                    break
            if layer is not None:
                kind = 'up' if '.adapter.up.' in k else 'down' if '.adapter.down.' in k else 'norm_or_other'
                layer_adapter[layer][kind + '_norm2'] += float(torch.dot(vf, vf).item())
                layer_adapter[layer][kind + '_params'] += vf.numel()
    groups = {}
    for g, acc in sorted(by_group.items()):
        n = int(acc['n_params'])
        norm = math.sqrt(max(acc['norm2'], 0.0))
        groups[g] = {
            'n_tensors': int(acc['n_tensors']),
            'n_params': n,
            'norm': norm,
            'rms': None if n == 0 else norm / math.sqrt(n),
            'mean_abs': None if n == 0 else acc['abs_sum'] / n,
        }
    layer_summary = {}
    for layer, d in sorted(layer_adapter.items()):
        row = {}
        for kind in ['up', 'down', 'norm_or_other']:
            p = int(d.get(kind + '_params', 0))
            norm = math.sqrt(max(d.get(kind + '_norm2', 0.0), 0.0))
            row[kind + '_params'] = p
            row[kind + '_norm'] = norm
            row[kind + '_rms'] = None if p == 0 else norm / math.sqrt(p)
        layer_summary[str(layer)] = row
    return {'label': label, 'groups': groups, 'adapter_by_layer': layer_summary}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    states = {}
    for label in RUNS:
        p = ckpt_file(RUNS[label])
        print(json.dumps({'event': 'loading', 'label': label, 'path': rel(p)}), flush=True)
        states[label] = load_state(label)
    metrics = {label: read_json(METRICS[label]) for label in METRICS}
    pair_names = [('aligned', 'shuffled'), ('aligned', 'mlm_only'), ('shuffled', 'mlm_only')]
    pairwise = {f'{a}_vs_{b}': pairwise_metrics(a, b, states[a], states[b]) for a, b in pair_names}
    norms = {label: state_norms(label, states[label]) for label in states}
    output = {
        'status': 'DUALVIEW_PARAMETER_GEOMETRY',
        'created_utc': now(),
        'runs': {label: rel(path) for label, path in RUNS.items()},
        'training_metrics': metrics,
        'pairwise': pairwise,
        'state_norms': norms,
        'interpretation_hint': 'Aligned-vs-shuffled isolates source correspondence at fixed aux budget; aligned/shuffled-vs-mlm_only show the effect of spending ~0.967M words on the dual-view private pathway instead of later main-stream rows. Stock_all displacement indicates how much the private-pathway route has already coupled back into the shared backbone by 20M.',
    }
    out_json = OUT_ROOT / 'dualview_parameter_geometry.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/dualview_parameter_geometry/dualview_parameter_geometry.md')
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = ['# research dual-view 20M parameter geometry', '', '## Pairwise global geometry', '', '| comparison | group | params | rel-L2 | cosine | diff-norm | mean |', '|---|---|---:|---:|---:|---:|---:|']
    for comp, rec in pairwise.items():
        for g in ['all_params', 'stock_all', 'adapter_all', 'adapter_up', 'adapter_down', 'stock_embeddings', 'stock_attention', 'stock_ffn_output', 'stock_lm_head']:
            if g in rec['groups']:
                r = rec['groups'][g]
                lines.append(f"| {comp} | {g} | {r['n_params']} | {r['rel_l2_to_b']:.6g} | {r['cosine']:.9f} | {r['diff_norm']:.6g} | {r['mean_abs_delta']:.6g} |")
    lines += ['', '## Adapter RMS by arm', '', '| arm | adapter_all rms | adapter_up rms | adapter_down rms | stock_all rms |', '|---|---:|---:|---:|---:|']
    for label, rec in norms.items():
        gr = rec['groups']
        lines.append(f"| {label} | {gr['adapter_all']['rms']:.6g} | {gr['adapter_up']['rms']:.6g} | {gr['adapter_down']['rms']:.6g} | {gr['stock_all']['rms']:.6g} |")
    lines += ['', 'Output JSON: `' + rel(out_json) + '`']
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'out_json': rel(out_json), 'out_md': rel(out_md),
                      'aligned_vs_shuffled_stock_rel_l2': pairwise['aligned_vs_shuffled']['groups']['stock_all']['rel_l2_to_b'],
                      'aligned_vs_shuffled_adapter_rel_l2': pairwise['aligned_vs_shuffled']['groups']['adapter_all']['rel_l2_to_b'],
                      'aligned_vs_mlm_stock_rel_l2': pairwise['aligned_vs_mlm_only']['groups']['stock_all']['rel_l2_to_b'],
                      'aligned_vs_mlm_adapter_rel_l2': pairwise['aligned_vs_mlm_only']['groups']['adapter_all']['rel_l2_to_b']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
