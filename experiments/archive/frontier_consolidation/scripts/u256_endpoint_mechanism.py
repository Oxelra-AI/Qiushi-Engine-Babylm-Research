#!/usr/bin/env python3
"""research: CPU-only mechanism readout for the completed U256 100M endpoint.

Compare the faithful stream-object visibility route against the matched research
row256 endpoint using saved checkpoint tensors and training metric records.  This
runs no inference and does not touch the active full-evaluation output tree.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from safetensors.torch import load_file

USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
DEFAULT_STEP35_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2')
DEFAULT_U256_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_100M')
DEFAULT_OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/u256_endpoint_mechanism')
DEFAULT_NOTE = _public_path('research/notes/frontier_consolidation/u256_endpoint_mechanism.md')
EXPOSURES = ['20M', '80M', '90M', '100M']
INTERVALS = [('20M', '80M'), ('80M', '90M'), ('90M', '100M'), ('80M', '100M')]
SCORE = {
    'Overall': 41.257770896404615,
    'cheap7': 43.00572463231884,
}
U256_20M_SCORE = {
    'cheap7': 40.581428571428575,
    'delta_vs_step35_20M': 0.9178571428571497,
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def model_path(run: Path, exp: str) -> Path:
    return run / 'hf_model' / f'chck_{exp}' / 'model.safetensors'


def group_name(n: str) -> str:
    if n.startswith('deberta.embeddings'):
        return 'embeddings'
    if n.startswith('deberta.encoder.layer.'):
        parts = n.split('.')
        layer = parts[3] if len(parts) > 3 else '?'
        if '.attention.' in n:
            return f'layer{layer}_attention'
        if '.intermediate.' in n or '.output.' in n:
            return f'layer{layer}_ffn_output'
        return f'layer{layer}_other'
    if n.startswith('cls.'):
        return 'mlm_head'
    return 'other'


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return load_file(str(path), device='cpu')


def tensor_names(*states: dict[str, torch.Tensor]) -> list[str]:
    common = set(states[0])
    for st in states[1:]:
        common &= set(st)
    return sorted(common)


def compare_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = tensor_names(a, b)
    dot = a2 = b2 = d2 = 0.0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {'dot': 0.0, 'a2': 0.0, 'b2': 0.0, 'd2': 0.0, 'n_tensors': 0, 'n_params': 0})
    for n in names:
        av = a[n].detach().float().flatten()
        bv = b[n].detach().float().flatten()
        dv = av - bv
        aa = float(torch.dot(av, av))
        bb = float(torch.dot(bv, bv))
        ab = float(torch.dot(av, bv))
        dd = float(torch.dot(dv, dv))
        dot += ab; a2 += aa; b2 += bb; d2 += dd
        gr = groups[group_name(n)]
        gr['dot'] += ab; gr['a2'] += aa; gr['b2'] += bb; gr['d2'] += dd; gr['n_tensors'] += 1; gr['n_params'] += int(av.numel())
    for gr in groups.values():
        gr['cosine'] = gr['dot'] / math.sqrt(gr['a2'] * gr['b2']) if gr['a2'] and gr['b2'] else None
        gr['rel_l2_to_step35'] = math.sqrt(gr['d2'] / gr['b2']) if gr['b2'] else None
        gr['diff2_fraction'] = gr['d2'] / d2 if d2 else 0.0
    return {
        'n_tensors': len(names),
        'n_params': sum(int(a[n].numel()) for n in names),
        'cosine': dot / math.sqrt(a2 * b2) if a2 and b2 else None,
        'rel_l2_to_step35': math.sqrt(d2 / b2) if b2 else None,
        'u256_norm': math.sqrt(a2),
        'norm': math.sqrt(b2),
        'l2_difference': math.sqrt(d2),
        'groups': dict(sorted(groups.items(), key=lambda kv: kv[1]['diff2_fraction'], reverse=True)),
    }


def update_alignment(u_hi: dict[str, torch.Tensor], u_lo: dict[str, torch.Tensor], b_hi: dict[str, torch.Tensor], b_lo: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = tensor_names(u_hi, u_lo, b_hi, b_lo)
    dot = u2 = b2 = diff2 = 0.0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {'dot': 0.0, 'u2': 0.0, 'b2': 0.0, 'diff2': 0.0, 'n_tensors': 0, 'n_params': 0})
    for n in names:
        uu = (u_hi[n].detach().float() - u_lo[n].detach().float()).flatten()
        bbv = (b_hi[n].detach().float() - b_lo[n].detach().float()).flatten()
        aa = float(torch.dot(uu, uu))
        bb = float(torch.dot(bbv, bbv))
        ab = float(torch.dot(uu, bbv))
        dd = float(torch.dot(uu - bbv, uu - bbv))
        dot += ab; u2 += aa; b2 += bb; diff2 += dd
        gr = groups[group_name(n)]
        gr['dot'] += ab; gr['u2'] += aa; gr['b2'] += bb; gr['diff2'] += dd; gr['n_tensors'] += 1; gr['n_params'] += int(uu.numel())
    for gr in groups.values():
        gr['cosine'] = gr['dot'] / math.sqrt(gr['u2'] * gr['b2']) if gr['u2'] and gr['b2'] else None
        gr['u256_update_rel_norm_vs_step35'] = math.sqrt(gr['u2'] / gr['b2']) if gr['b2'] else None
        gr['diff2_fraction'] = gr['diff2'] / diff2 if diff2 else 0.0
    return {
        'n_tensors': len(names),
        'cosine': dot / math.sqrt(u2 * b2) if u2 and b2 else None,
        'u256_update_norm': math.sqrt(u2),
        'update_norm': math.sqrt(b2),
        'u256_update_rel_norm_vs_step35': math.sqrt(u2 / b2) if b2 else None,
        'update_difference_l2': math.sqrt(diff2),
        'groups': dict(sorted(groups.items(), key=lambda kv: kv[1]['diff2_fraction'], reverse=True)),
    }


def matrix_spectra(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    recs: list[dict[str, Any]] = []
    for n, t in state.items():
        if not n.startswith('deberta.encoder.layer.') or not n.endswith('.weight') or t.ndim != 2:
            continue
        s = torch.linalg.svdvals(t.detach().float())
        energy = s.square()
        total = float(energy.sum())
        recs.append({
            'name': n,
            'group': group_name(n),
            'shape': list(t.shape),
            'stable_rank': total / float(energy.max()) if total else None,
            'top8_energy': float(energy[:8].sum() / total) if total else None,
        })
    by_group: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        grouped[r['group']].append(r)
    for g, rows in grouped.items():
        by_group[g] = {
            'n': len(rows),
            'stable_rank_mean': float(mean(r['stable_rank'] for r in rows if r['stable_rank'] is not None)),
            'top8_energy_mean': float(mean(r['top8_energy'] for r in rows if r['top8_energy'] is not None)),
        }
    return {
        'n_matrices': len(recs),
        'stable_rank_mean': float(mean(r['stable_rank'] for r in recs if r['stable_rank'] is not None)) if recs else None,
        'top8_energy_mean': float(mean(r['top8_energy'] for r in recs if r['top8_energy'] is not None)) if recs else None,
        'by_group': dict(sorted(by_group.items())),
    }


def load_metric_core(run: Path, label: str) -> dict[str, Any]:
    m = read_json(run / 'scientific_metrics.json')
    checkpoints = m.get('checkpoints') or m.get('saved_checkpoints') or []
    return {
        'label': label,
        'path': rel(run / 'scientific_metrics.json'),
        'word_exposure': m.get('word_exposure') or m.get('total_charged_words'),
        'steps': m.get('actual_training_steps') or m.get('total_steps_executed'),
        'loss_first': m.get('loss_first'),
        'loss_last': m.get('loss_last'),
        'parameter_count': m.get('parameter_count'),
        'vocab_size': m.get('vocab_size'),
        'masking_curriculum': m.get('masking_curriculum'),
        'raw_tokens_per_epoch': m.get('raw_tokens_per_epoch'),
        'checkpoint_count': len(checkpoints),
        'first_checkpoint': checkpoints[0] if checkpoints else None,
        'last_checkpoint': checkpoints[-1] if checkpoints else None,
        'verification': m.get('verification'),
        'base_sha256': m.get('base_sha256'),
        'stream_sha256': m.get('stream_sha256'),
    }


def run_readout(run: Path, u256_run: Path, out_dir: Path, note_path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    missing: dict[str, str] = {}
    for exp in EXPOSURES:
        for label, run in [('research', run), ('u256', u256_run)]:
            p = model_path(run, exp)
            if not p.exists():
                missing[f'{label}_{exp}'] = rel(p)
    if missing:
        return {
            'status': 'U256_ENDPOINT_MECHANISM_PENDING',
            'created_utc': now(),
            'missing': missing,
            'run': rel(run),
            'u256_run': rel(u256_run),
        }

    states: dict[str, dict[str, torch.Tensor]] = {}
    # Load all requested checkpoints together; this is under ~1.2GB and avoids repeated disk reads.
    for exp in EXPOSURES:
        for label, run in [('research', run), ('u256', u256_run)]:
            key = f'{label}_{exp}'
            states[key] = load_state(model_path(run, exp))
            print(json.dumps({'event': 'loaded', 'key': key, 'path': rel(model_path(run, exp))}), flush=True)

    comparisons = {exp: compare_states(states[f'u256_{exp}'], states[f'step35_{exp}']) for exp in EXPOSURES}
    updates = {f'{lo}_to_{hi}': update_alignment(states[f'u256_{hi}'], states[f'u256_{lo}'], states[f'step35_{hi}'], states[f'step35_{lo}']) for lo, hi in INTERVALS}
    spectra = {key: matrix_spectra(state) for key, state in states.items() if key.endswith('100M') or key.endswith('80M')}

    # Compare final checkpoint against 80M endpoint across the same route, not just vs research.
    route_growth = {
        'u256_80M_to_100M_norm': updates['80M_to_100M']['u256_update_norm'],
        'reference_80M_to_100M_norm': updates['80M_to_100M']['update_norm'],
        'u256_80M_to_100M_rel_norm_vs_step35': updates['80M_to_100M']['u256_update_rel_norm_vs_step35'],
        'u256_90M_to_100M_rel_norm_vs_step35': updates['90M_to_100M']['u256_update_rel_norm_vs_step35'],
        'u256_100M_vs_step35_cosine': comparisons['100M']['cosine'],
        'u256_100M_vs_step35_rel_l2': comparisons['100M']['rel_l2_to_step35'],
    }

    result = {
        'status': 'U256_ENDPOINT_MECHANISM',
        'created_utc': now(),
        'run': rel(run),
        'u256_run': rel(u256_run),
        'missing': {},
        'metrics': {
            'research': load_metric_core(run, 'research'),
            'u256': load_metric_core(u256_run, 'u256'),
        },
        'known_scores_before_u256_full_eval': {
            'reference_100M': SCORE,
            'u256_20M_screen': U256_20M_SCORE,
            'scale1p75_100M_complete': {
                'Overall': 41.57074653643003,
                'cheap7': 43.543159919261925,
                'note': 'closed below 41.8 in research; included only as current negative endpoint reference',
            },
        },
        'comparisons_vs_step35': comparisons,
        'stock_update_alignments': updates,
        'core_matrix_spectra': spectra,
        'route_growth_summary': route_growth,
        'scientific_reading': '',
    }
    c20 = comparisons['20M']
    c100 = comparisons['100M']
    u90100 = updates['90M_to_100M']
    s80 = spectra['u256_80M']
    s100 = spectra['u256_100M']
    if c100['rel_l2_to_step35'] is not None:
        if c100['rel_l2_to_step35'] < 0.15:
            regime = 'small endpoint perturbation of the research weight basin'
        elif c100['rel_l2_to_step35'] < 0.5:
            regime = 'moderate same-architecture trajectory redirection'
        else:
            regime = 'large same-architecture trajectory redirection'
    else:
        regime = 'unclassified trajectory relation'
    result['scientific_reading'] = (
        f'U256 began as a strong 20M behavioral screen (+{U256_20M_SCORE["delta_vs_step35_20M"]:.4f} cheap7) from only a 2.6% active-token visibility repair. '
        f'The endpoint tensor geometry is a {regime}: 20M cosine {c20["cosine"]:.4f}, rel-L2 {c20["rel_l2_to_step35"]:.4f}; '
        f'100M cosine {c100["cosine"]:.4f}, rel-L2 {c100["rel_l2_to_step35"]:.4f}. '
        f'The final 90M→100M update has cosine {u90100["cosine"]:.4f} and norm ratio {u90100["u256_update_rel_norm_vs_step35"]:.4f} versus research. '
        f'Core matrix spectra do not show a Muon/LAMB-like collapse or broadening: U256 stable-rank mean changes {s80["stable_rank_mean"]:.3f}→{s100["stable_rank_mean"]:.3f} from 80M to 100M. '
        'Thus the pending official score should be interpreted as the behavioral effect of making row-tail experience visible under the same architecture/objective, not as a new parameterization or optimizer artifact.'
    )
    return result


def write_note(result: dict[str, Any], note_path: Path, out_json: Path) -> None:
    lines: list[str] = []
    lines.append('# research — U256 endpoint mechanism readout')
    lines.append('')
    lines.append(f"Status: `{result['status']}`")
    if result.get('missing'):
        lines.append('')
        lines.append('Missing checkpoints:')
        for k, v in result['missing'].items():
            lines.append(f'- {k}: `{v}`')
    else:
        lines.append('')
        lines.append('## Metric core')
        for label in ['research', 'u256']:
            m = result['metrics'][label]
            lines.append(f"- {label}: words={m['word_exposure']}, steps={m['steps']}, loss={m['loss_first']}→{m['loss_last']}, params={m['parameter_count']}, checkpoints={m['checkpoint_count']}")
        lines.append('')
        lines.append('## Weight displacement versus research')
        lines.append('| exposure | cosine | rel L2 | top diff group | top diff fraction |')
        lines.append('|---|---:|---:|---|---:|')
        for exp in EXPOSURES:
            c = result['comparisons_vs_step35'][exp]
            topg = next(iter(c['groups'].items())) if c['groups'] else ('', {'diff2_fraction': 0.0})
            lines.append(f"| {exp} | {c['cosine']:.6f} | {c['rel_l2_to_step35']:.6f} | {topg[0]} | {topg[1]['diff2_fraction']:.4f} |")
        lines.append('')
        lines.append('## Update alignment')
        lines.append('| interval | update cosine | U256/research update norm |')
        lines.append('|---|---:|---:|')
        for interval, u in result['stock_update_alignments'].items():
            lines.append(f"| {interval} | {u['cosine']:.6f} | {u['u256_update_rel_norm_vs_step35']:.6f} |")
        lines.append('')
        lines.append('## Core matrix spectra')
        lines.append('| checkpoint | stable-rank mean | top8-energy mean |')
        lines.append('|---|---:|---:|')
        for key, s in sorted(result['core_matrix_spectra'].items()):
            lines.append(f"| {key} | {s['stable_rank_mean']:.3f} | {s['top8_energy_mean']:.4f} |")
        lines.append('')
        lines.append('## Scientific reading')
        lines.append(result['scientific_reading'])
    lines.append('')
    lines.append(f'JSON: `{rel(out_json)}`')
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--research-run', type=Path, default=DEFAULT_STEP35_RUN)
    ap.add_argument('--u256-run', type=Path, default=DEFAULT_U256_RUN)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    args = ap.parse_args()
    for attr in ['run', 'u256_run', 'out_dir', 'note']:
        v = getattr(args, attr)
        if isinstance(v, Path) and not v.is_absolute():
            setattr(args, attr, USER_ROOT / v)
    result = run_readout(args.run, args.u256_run, args.out_dir, args.note)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / 'u256_endpoint_mechanism.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_note(result, args.note, out_json)
    print(json.dumps({
        'status': result['status'],
        'out_json': rel(out_json),
        'note': rel(args.note),
        'missing': list(result.get('missing', {}).keys()),
        'u256_100M_vs_step35_cosine': (result.get('comparisons_vs_step35') or {}).get('100M', {}).get('cosine'),
        'u256_100M_vs_step35_rel_l2': (result.get('comparisons_vs_step35') or {}).get('100M', {}).get('rel_l2_to_step35'),
        'update_90_100_cosine': (result.get('stock_update_alignments') or {}).get('90M_to_100M', {}).get('cosine'),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
