#!/usr/bin/env python3
"""research aligned-vs-shuffled dual-view training trajectory comparison.

Compares the completed aligned and shuffled 20M charged-prefix logs under exactly
matched accounting. This is mechanism evidence only: it shows whether true source
correspondence changes the optimized losses under the same masks/targets/exposure;
official-compatible cheap7 evaluation remains the route-changing evidence.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

USER_ROOT = pathlib.Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
ALIGNED = WORKSPACE / 'training/runs/dualview_aligned_20M_seed43022'
SHUFFLED = WORKSPACE / 'training/runs/dualview_shuffled_20M_seed43022'
OUT_DIR = WORKSPACE / 'data/dualview_training_trajectory'


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_log(run: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with (run / 'training_log.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def mean(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def stdev(xs: list[float]) -> float | None:
    return float(statistics.stdev(xs)) if len(xs) > 1 else None


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x-ma)**2 for x in a)
    vb = sum((y-mb)**2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return float(sum((x-ma)*(y-mb) for x,y in zip(a,b)) / math.sqrt(va*vb))


def window(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [r for r in rows if start <= int(r['update']) <= end]


def summarize_diffs(alog: list[dict[str, Any]], slog: list[dict[str, Any]], key: str) -> dict[str, Any]:
    diffs = [float(a[key]) - float(s[key]) for a, s in zip(alog, slog)]
    return {
        'key': key,
        'mean_aligned_minus_shuffled': mean(diffs),
        'stdev': stdev(diffs),
        'min': min(diffs) if diffs else None,
        'max': max(diffs) if diffs else None,
        'final_aligned_minus_shuffled': diffs[-1] if diffs else None,
        'aligned_lower_count': sum(1 for d in diffs if d < 0),
        'aligned_higher_count': sum(1 for d in diffs if d > 0),
        'zero_count': sum(1 for d in diffs if d == 0),
        'windows': {
            f'{s}-{e}': {
                'mean_aligned_minus_shuffled': mean([float(a[key]) - float(b[key]) for a,b in zip(window(alog,s,e), window(slog,s,e))])
            }
            for s,e in [(1,50),(51,100),(101,150),(151,250),(251,350),(351,481)]
        },
    }


def compare_accounting(alog: list[dict[str, Any]], slog: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ['update','loader_step','batch_words','aux_words','cumulative_main_words','cumulative_aux_words',
            'cumulative_charged_words','masked_tokens','aux_targets','aux_units','aux_conditioned_views','aux_free_views','lr']
    diffs = {k: 0 for k in keys}
    first = {}
    for i,(a,s) in enumerate(zip(alog,slog)):
        for k in keys:
            av, sv = a.get(k), s.get(k)
            ok = (math.isclose(float(av), float(sv), rel_tol=0, abs_tol=1e-12) if isinstance(av, float) or isinstance(sv, float) else av == sv)
            if not ok:
                diffs[k] += 1
                first.setdefault(k, {'idx0': i, 'aligned': av, 'shuffled': sv})
    return {'n_aligned': len(alog), 'n_shuffled': len(slog), 'same_length': len(alog)==len(slog), 'diff_counts': diffs, 'first_diff': first, 'all_equal': len(alog)==len(slog) and all(v==0 for v in diffs.values())}


def main() -> None:
    alog, slog = read_log(ALIGNED), read_log(SHUFFLED)
    am, sm = read_json(ALIGNED / 'scientific_metrics.json'), read_json(SHUFFLED / 'scientific_metrics.json')
    main_losses_a = [float(r['loss']) for r in alog]
    main_losses_s = [float(r['loss']) for r in slog]
    aux_losses_a = [float(r['aux_loss']) for r in alog]
    aux_losses_s = [float(r['aux_loss']) for r in slog]
    out = {
        'status': 'DUALVIEW_TRAINING_TRAJECTORY_COMPARE',
        'aligned_run': rel(ALIGNED),
        'shuffled_run': rel(SHUFFLED),
        'aligned_metrics': am,
        'shuffled_metrics': sm,
        'accounting': compare_accounting(alog, slog),
        'loss_comparison': summarize_diffs(alog, slog, 'loss'),
        'aux_loss_comparison': summarize_diffs(alog, slog, 'aux_loss'),
        'loss_correlation': {
            'main_loss_aligned_vs_shuffled': pearson(main_losses_a, main_losses_s),
            'aux_loss_aligned_vs_shuffled': pearson(aux_losses_a, aux_losses_s),
        },
        'interpretation': 'Aligned and shuffled have identical masks/targets/exposure. Lower aligned aux_loss means true source correspondence is useful to the private dual-view objective; official cheap7 and fragile-family results still decide whether this transfers to broad BabyLM competence.',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / 'aligned_vs_shuffled_training_trajectory.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/dualview_training_trajectory/aligned_vs_shuffled_training_trajectory.md')
    out['out_json'] = rel(out_json)
    out['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research aligned-vs-shuffled dual-view training trajectory', '',
        f"Accounting equal: `{out['accounting']['all_equal']}` over `{len(alog)}` updates.",
        f"Main final loss aligned/shuffled: `{am['final_loss']}` / `{sm['final_loss']}`; aligned minus shuffled `{am['final_loss']-sm['final_loss']:+.6f}`.",
        f"Mean main loss aligned/shuffled: `{am['mean_loss']}` / `{sm['mean_loss']}`; delta `{am['mean_loss']-sm['mean_loss']:+.6f}`.",
        f"Mean aux loss aligned/shuffled: `{am['mean_aux_loss']}` / `{sm['mean_aux_loss']}`; delta `{am['mean_aux_loss']-sm['mean_aux_loss']:+.6f}`.",
        '',
        'This is mechanism evidence only; cheap7 and fragile-family saved predictions decide whether the effect helps the target route.',
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'accounting_equal': out['accounting']['all_equal'],
        'main_final_loss_delta_aligned_minus_shuffled': am['final_loss'] - sm['final_loss'],
        'mean_main_loss_delta_aligned_minus_shuffled': am['mean_loss'] - sm['mean_loss'],
        'mean_aux_loss_delta_aligned_minus_shuffled': am['mean_aux_loss'] - sm['mean_aux_loss'],
        'out_json': rel(out_json),
        'out_md': rel(out_md),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
