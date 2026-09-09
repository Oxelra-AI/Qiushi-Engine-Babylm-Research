#!/usr/bin/env python3
"""research: synthetic packet readout for the shared-tokenizer legal Route-B screen.

The research/148 readout machinery measures natural BabyLM surfaces.  This script
adds a separate interpretive readout on the research expanded packet scoring pairs.
Synthetic packet transfer is not a success condition for the SOTA route, but it is
scientifically useful for separating three possible outcomes after the legal 80M
screen:

1. packets not learned under sparse from-scratch insertion;
2. packets learned synthetically but not transferred to natural official surfaces;
3. packets learned and natural surfaces also improve relative to role-fixed.

The script trains nothing.  It scores checkpoint-local tokenizers/models on the
research role-switch scoring pairs and reports role_switch-minus-role_fixed deltas.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
SCORER_PATH = A01_WS / 'scripts/frozen_exchange_scoring.py'
PAIRS_PATH = A01_WS / 'data/role_switch_expanded_packets/scoring_pairs.jsonl'
OUT_ROOT = A01_WS / 'data/packet_synthetic_reader'

TARGETS: dict[str, dict[str, Any]] = {
    'role_switch_80M': {
        'model_root': A01_WS / 'training/runs/role_switch_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'tokenizer_root': A01_WS / 'training/runs/role_switch_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'description': 'legal role-switch replacement corpus, shared tokenizer, 80M from-scratch screen',
    },
    'role_fixed_80M': {
        'model_root': A01_WS / 'training/runs/role_fixed_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'tokenizer_root': A01_WS / 'training/runs/role_fixed_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'description': 'legal role-fixed matched control corpus, shared tokenizer, 80M from-scratch screen',
    },
    'anchor_fixed256_80M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'description': 'matched legal40k fixed-256 compact-view anchor at 80M; different but nearly overlapping legal40k tokenizer',
    },
    'anchor_fixed256_100M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'description': 'matched legal40k fixed-256 compact-view anchor at 100M; context only',
    },
    'reheat_100M': {
        'model_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'reheat-cosine 100M broad-learning reference; context only',
    },
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_scoremod():
    spec = importlib.util.spec_from_file_location('scoremod_for_step149', SCORER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Keep the expanded-packet readout compact enough for this synthetic grammar.
    mod.MAX_LEN = 128
    return mod


def read_pairs() -> list[dict[str, Any]]:
    rows = []
    with PAIRS_PATH.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def readiness(selected: list[str]) -> dict[str, Any]:
    out = {}
    for label in selected:
        meta = TARGETS[label]
        mr = pathlib.Path(meta['model_root'])
        tr = pathlib.Path(meta['tokenizer_root'])
        out[label] = {
            'model_root': str(mr),
            'tokenizer_root': str(tr),
            'model_ready': bool(mr.exists() and (mr / 'model.safetensors').exists()),
            'tokenizer_ready': bool(tr.exists() and ((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists())),
            'description': meta['description'],
        }
    out['pairs_path'] = str(PAIRS_PATH)
    out['pairs_ready'] = PAIRS_PATH.exists()
    out['selected_ready'] = bool(PAIRS_PATH.exists()) and all(
        isinstance(v, dict) and v.get('model_ready') and v.get('tokenizer_ready')
        for v in out.values() if isinstance(v, dict)
    )
    return out


def score_target(label: str, meta: dict[str, Any], pairs: list[dict[str, Any]], scoremod, device: str, out_dir: pathlib.Path) -> dict[str, Any]:
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), trust_remote_code=True).to(device).eval()
    print(json.dumps({
        'event': 'synthetic_score_start',
        'label': label,
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'device': device,
        'n_pairs': len(pairs),
        'mask_token_id': tokenizer.mask_token_id,
    }), flush=True)
    results = scoremod.score_all_pairs(model, tokenizer, pairs, device)
    rpath = out_dir / f'{label}_pair_results.jsonl'
    with rpath.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = scoremod.summarize(results, label)
    summary['pair_results'] = str(rpath)
    summary['model_root'] = str(model_root)
    summary['tokenizer_root'] = str(tok_root)
    if (model_root / 'tokenizer.json').exists():
        summary['tokenizer_json_sha256'] = sha256_file(model_root / 'tokenizer.json')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({'event': 'synthetic_score_done', 'label': label, 'overall': summary.get('overall')}, indent=2), flush=True)
    return summary


def delta_summary(base_s: dict[str, Any], arm_s: dict[str, Any]) -> dict[str, Any]:
    def get(path: list[str], obj: dict[str, Any]):
        cur: Any = obj
        for p in path:
            cur = cur[p]
        return cur
    d: dict[str, Any] = {'overall': {}, 'by_family_both_correct': {}, 'by_family_M_mean': {}, 'by_style_both_correct': {}}
    for k in ['M_mean', 'M_pos_frac', 'CM_mean', 'bias_abs_mean', 'acc_AB', 'acc_BA', 'both_correct']:
        d['overall'][k] = get(['overall', k], arm_s) - get(['overall', k], base_s)
    for fam in sorted(set(base_s.get('by_family', {})) & set(arm_s.get('by_family', {}))):
        d['by_family_both_correct'][fam] = arm_s['by_family'][fam]['both_correct'] - base_s['by_family'][fam]['both_correct']
        d['by_family_M_mean'][fam] = arm_s['by_family'][fam]['M_mean'] - base_s['by_family'][fam]['M_mean']
    for sty in sorted(set(base_s.get('by_style', {})) & set(arm_s.get('by_style', {}))):
        d['by_style_both_correct'][sty] = arm_s['by_style'][sty]['both_correct'] - base_s['by_style'][sty]['both_correct']
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'score'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['role_switch_80M', 'role_fixed_80M', 'anchor_fixed256_80M'])
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
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

    scoremod = load_scoremod()
    pairs = read_pairs()
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    summaries = {}
    score_dir = out_root / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    for label in args.targets:
        summaries[label] = score_target(label, TARGETS[label], pairs, scoremod, device, score_dir)

    deltas = {}
    if 'role_switch_80M' in summaries and 'role_fixed_80M' in summaries:
        deltas['role_switch_minus_role_fixed'] = delta_summary(summaries['role_fixed_80M'], summaries['role_switch_80M'])
    if 'anchor_fixed256_80M' in summaries:
        base = summaries['anchor_fixed256_80M']
        for label, summary in summaries.items():
            if label != 'anchor_fixed256_80M':
                deltas[f'{label}_minus_anchor_fixed256_80M'] = delta_summary(base, summary)

    combined = {
        'status': 'PACKET_SYNTHETIC_READER',
        'created_utc': now_utc(),
        'scientific_use': 'Interpret whether legal low-dose insertion learned the constructed context-conditioned packet grammar; natural BabyLM hard-surface movement remains the route-relevant result.',
        'pairs_path': str(PAIRS_PATH),
        'n_pairs': len(pairs),
        'targets': args.targets,
        'readiness': ready,
        'summaries': summaries,
        'deltas': deltas,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/packet_synthetic_reader.py')),
    }
    out_path = out_root / 'packet_synthetic_readout_summary.json'
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'PACKET_SYNTHETIC_READER_DONE', 'summary': str(out_path), 'deltas': deltas}, indent=2), flush=True)


if __name__ == '__main__':
    main()
