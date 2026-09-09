#!/usr/bin/env python3
"""research: official-hard-surface readouts for research small update checkpoints.

Scientific purpose
------------------
The research naturalistic exchange bridge moved after small packet updates, but the
active research question is whether that bridge movement predicts the real natural
surfaces: GlobalPIQA_parallel hard ranks/margins and EWoK stable conditional
reversals. This script reuses the already validated research GlobalPIQA all-option
reader and research EWoK four-cell interaction reader, but with explicit dynamic
model paths for the research small-update checkpoints.

Boundary
--------
Readout only. No training. No official-example shaping. These checkpoints are
small mechanism probes from an existing anchor and are not submission candidates.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch

ROOT = pathlib.Path.cwd()
WS = ROOT / 'experiments/archive/representation_and_objectives'
SCRIPT_DIR = WS / 'scripts'
OUT_ROOT = WS / 'data/small_update_hard_surface_readout'

GP_SCRIPT = SCRIPT_DIR / 'globalpiqa_margin_reader.py'
EW_SCRIPT = SCRIPT_DIR / 'fw_ewok_interaction_reader.py'

SMALL_TARGETS: dict[str, dict[str, Any]] = {
    'anchor80_exchange_lr5e-5_u80': {
        'objective': 'exchange', 'lr': 5e-5, 'updates': 80,
        'model_root': WS / 'data/small_exchange_update/anchor80_exchange_lr5e-5_u80/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/anchor80_exchange_lr5e-5_u80/small_update_summary.json',
    },
    'anchor80_correct_ce_lr5e-5_u80': {
        'objective': 'correct_ce', 'lr': 5e-5, 'updates': 80,
        'model_root': WS / 'data/small_exchange_update/anchor80_correct_ce_lr5e-5_u80/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/anchor80_correct_ce_lr5e-5_u80/small_update_summary.json',
    },
    'anchor80_fixed_ce_lr5e-5_u80': {
        'objective': 'fixed_ce', 'lr': 5e-5, 'updates': 80,
        'model_root': WS / 'data/small_exchange_update/anchor80_fixed_ce_lr5e-5_u80/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/anchor80_fixed_ce_lr5e-5_u80/small_update_summary.json',
    },
    'pareto_exchange_lr2e-5_u20': {
        'objective': 'exchange', 'lr': 2e-5, 'updates': 20,
        'model_root': WS / 'data/small_exchange_update/pareto_anchor80_exchange_lr2e-5_u20/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/pareto_anchor80_exchange_lr2e-5_u20/small_update_summary.json',
    },
    'pareto_correct_ce_lr2e-5_u20': {
        'objective': 'correct_ce', 'lr': 2e-5, 'updates': 20,
        'model_root': WS / 'data/small_exchange_update/pareto_anchor80_correct_ce_lr2e-5_u20/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/pareto_anchor80_correct_ce_lr2e-5_u20/small_update_summary.json',
    },
    'pareto_exchange_lr1e-5_u40': {
        'objective': 'exchange', 'lr': 1e-5, 'updates': 40,
        'model_root': WS / 'data/small_exchange_update/pareto_anchor80_exchange_lr1e-5_u40/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/pareto_anchor80_exchange_lr1e-5_u40/small_update_summary.json',
    },
    'pareto_correct_ce_lr1e-5_u40': {
        'objective': 'correct_ce', 'lr': 1e-5, 'updates': 40,
        'model_root': WS / 'data/small_exchange_update/pareto_anchor80_correct_ce_lr1e-5_u40/hf_model/chck_step151_final',
        'summary': WS / 'data/small_exchange_update/pareto_anchor80_correct_ce_lr1e-5_u40/small_update_summary.json',
    },
}

ANCHOR = {
    'label': 'anchor_fixed256_80M',
    'model_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_rows_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ('scores', 'completions', 'metadata')}
        if isinstance(r.get('scores'), list):
            rr.update({f'score_{i}': s for i, s in enumerate(r['scores'])})
        if isinstance(r.get('completions'), list):
            rr.update({f'completion_{i}': c for i, c in enumerate(r['completions'])})
        flat.append(rr)
    keys = sorted({k for rr in flat for k in rr.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(flat)


def read_small_summary(target: str) -> dict[str, Any]:
    sp = pathlib.Path(SMALL_TARGETS[target]['summary'])
    if not sp.exists():
        return {'exists': False, 'path': str(sp)}
    d = json.loads(sp.read_text())
    def get(path: list[str], default=None):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur
    before_bridge = get(['before', 'bridge', 'overall'], {}) or {}
    after_bridge = get(['after', 'bridge', 'overall'], {}) or {}
    return {
        'exists': True,
        'path': str(sp),
        'objective': get(['args', 'objective'], SMALL_TARGETS[target].get('objective')),
        'lr': get(['args', 'lr'], SMALL_TARGETS[target].get('lr')),
        'updates': get(['args', 'updates'], SMALL_TARGETS[target].get('updates')),
        'bridge_before_M': before_bridge.get('M_mean'),
        'bridge_after_M': after_bridge.get('M_mean'),
        'bridge_delta_M': (after_bridge.get('M_mean') - before_bridge.get('M_mean')) if isinstance(after_bridge.get('M_mean'), (int, float)) and isinstance(before_bridge.get('M_mean'), (int, float)) else None,
        'bridge_before_both': before_bridge.get('both_correct'),
        'bridge_after_both': after_bridge.get('both_correct'),
        'bridge_delta_both': (after_bridge.get('both_correct') - before_bridge.get('both_correct')) if isinstance(after_bridge.get('both_correct'), (int, float)) and isinstance(before_bridge.get('both_correct'), (int, float)) else None,
        'clean_loss_delta': get(['clean_probe', 'delta_loss_per_masked_token']),
    }


def readiness(targets: list[str]) -> dict[str, Any]:
    out = {
        'created_utc': now_utc(),
        'gp_script': str(GP_SCRIPT),
        'ew_script': str(EW_SCRIPT),
        'targets': {},
        'anchor': {'model_root': str(ANCHOR['model_root']), 'model_ready': (ANCHOR['model_root'] / 'model.safetensors').exists()},
    }
    for t in targets:
        meta = SMALL_TARGETS[t]
        mr = pathlib.Path(meta['model_root'])
        out['targets'][t] = {
            'model_root': str(mr),
            'model_ready': (mr / 'model.safetensors').exists(),
            'tokenizer_ready': (mr / 'tokenizer.json').exists(),
            'summary': read_small_summary(t),
        }
    out['selected_ready'] = all(v['model_ready'] and v['tokenizer_ready'] for v in out['targets'].values())
    return out


def score_globalpiqa(target: str, out_root: pathlib.Path, modes: list[str], batch_size: int, non_causal_batch_size: int, max_items: int | None, threads: int) -> dict[str, Any]:
    gp = load_module(GP_SCRIPT, 'gp_reader_module')
    meta = SMALL_TARGETS[target]
    gp.TARGETS = {
        target: {
            'label': target,
            'model_root': pathlib.Path(meta['model_root']),
            'revision': None,
        }
    }
    gp.OUT_ROOT = out_root
    out_root.mkdir(parents=True, exist_ok=True)
    res = gp.run_target(target, modes, batch_size, non_causal_batch_size, max_items, threads)
    # Separate summary from rows to keep summary light while preserving rows in CSV.
    light_modes: dict[str, Any] = {}
    for mode, md in res.get('modes', {}).items():
        rows = md.get('rows', [])
        write_rows_csv(out_root / f'{target}_{mode}_rows.csv', rows)
        light_modes[mode] = {'summary': md.get('summary', {})}
    result = {
        'status': 'GLOBALPIQA_SMALL_UPDATE_READOUT_DONE',
        'created_utc': now_utc(),
        'boundary': 'post-hoc readout only; no official-example tuning',
        'target': target,
        'target_meta': {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in meta.items() if k != 'summary'},
        'small_update_summary': read_small_summary(target),
        'reader_summary': {k: v for k, v in res.items() if k != 'modes'},
        'modes': light_modes,
    }
    (out_root / f'{target}_globalpiqa_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def score_ewok(target: str, out_root: pathlib.Path, device_s: str, threads: int, row_limit: int, row_offset: int, row_batch_size: int, masked_batch_size: int) -> dict[str, Any]:
    ew = load_module(EW_SCRIPT, 'ewok_reader_module')
    meta = SMALL_TARGETS[target]
    model_path = pathlib.Path(meta['model_root'])
    device = torch.device('cuda' if device_s == 'cuda' and torch.cuda.is_available() else 'cpu')
    out_root.mkdir(parents=True, exist_ok=True)
    res = ew.run_target(target, model_path, device, threads, row_limit, row_offset, row_batch_size, masked_batch_size)
    records = res.pop('records')
    ew.write_csv(out_root / f'{target}_ewok_records.csv', records)
    ew.write_csv(out_root / f'{target}_ewok_by_domain.csv', res.get('by_domain', []))
    ew.write_csv(out_root / f'{target}_ewok_by_context_diff.csv', res.get('by_context_diff', []))
    result = {
        'status': 'EWOK_SMALL_UPDATE_READOUT_DONE',
        'created_utc': now_utc(),
        'boundary': 'post-hoc readout only; no official-example tuning',
        'target': target,
        'target_meta': {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in meta.items() if k != 'summary'},
        'small_update_summary': read_small_summary(target),
        'result': res,
    }
    (out_root / f'{target}_ewok_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'globalpiqa', 'ewok'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(SMALL_TARGETS), default=list(SMALL_TARGETS))
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--modes', nargs='+', choices=['parallel', 'nonparallel'], default=['parallel'])
    ap.add_argument('--gp-batch-size', type=int, default=8)
    ap.add_argument('--gp-non-causal-batch-size', type=int, default=32)
    ap.add_argument('--max-items', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--row-limit', type=int, default=0)
    ap.add_argument('--row-offset', type=int, default=0)
    ap.add_argument('--row-batch-size', type=int, default=64)
    ap.add_argument('--masked-batch-size', type=int, default=256)
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ready = readiness(args.targets)
    (out_root / 'readiness.json').write_text(json.dumps(ready, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'readiness', 'ready': ready}, indent=2), flush=True)
    if args.stage == 'ready':
        return
    if not ready['selected_ready']:
        raise RuntimeError({'ready': ready})
    max_items = None if args.max_items == 0 else int(args.max_items)
    summaries = {}
    for target in args.targets:
        if args.stage == 'globalpiqa':
            summaries[target] = score_globalpiqa(target, out_root / target / 'globalpiqa', args.modes, args.gp_batch_size, args.gp_non_causal_batch_size, max_items, args.threads)
        elif args.stage == 'ewok':
            summaries[target] = score_ewok(target, out_root / target / 'ewok', args.device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size)
        print(json.dumps({'event': 'target_done', 'stage': args.stage, 'target': target, 'summary_path': str(out_root / target / args.stage)}, indent=2), flush=True)
    combined = {
        'status': f'SMALL_UPDATE_{args.stage.upper()}_READOUT_BATCH_DONE',
        'created_utc': now_utc(),
        'stage': args.stage,
        'targets': args.targets,
        'summaries': summaries,
    }
    (out_root / f'{args.stage}_batch_summary.json').write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': combined['status'], 'out_root': str(out_root)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
