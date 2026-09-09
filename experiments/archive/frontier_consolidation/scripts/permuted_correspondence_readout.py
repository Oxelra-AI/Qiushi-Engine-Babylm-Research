#!/usr/bin/env python3
"""research: file-only readout for the MAX aligned-vs-permuted compact control.

After a future conditional training/evaluation of the research permuted companion
arm, this script compares existing first-basin MAX view/repeat EWoK+Entity scores
with future permuted scores:

  V-P: value of source-to-own-view correspondence, holding source and rewrite
       multisets nearly exactly fixed;
  P-R: value of compact rewrite text as non-corresponding companion text versus
       exact source repetition;
  V-R: inherited aligned semantic leg.

It performs no model inference and is safe to run before permuted scores exist;
missing rows are written explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics as stats
import time
from typing import Any

ROOT = pathlib.Path.cwd()
WS = ROOT / 'experiments/archive/frontier_consolidation'
OUT_DEFAULT = WS / 'data/permuted_correspondence_readout'
MAX_PER_TARGET = WS / 'data/dose_ladder_stable_eval/eval/per_target'
PERM_EVAL_ROOT = WS / 'data/permuted_entity_ewok_eval/eval'
META_PATH = WS / 'data/dose_2p64x_permuted_companion_rowholdout_pools/permuted_companion_rowholdout_metadata.json'
TOKEN_GEOMETRY = WS / 'data/dose_2p64x_permuted_companion_rowholdout_pools/permuted_companion_token_geometry.json'
ENTITY_RECON = WS / 'data/entity_leg_reconstruction/entity_leg_reconstruction_summary.json'
CHECKPOINTS = [f'chck_{i}M' for i in range(10, 101, 10)]
FAMILIES = ['EWoK', 'Entity']


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ['empty']
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def score_from_payload(path: pathlib.Path, family: str) -> float | None:
    if not path.exists():
        return None
    obj = read_json(path)
    stable = obj.get('stable_family_scores') or {}
    if finite(stable.get(family)):
        return float(stable[family])
    rec = (obj.get('tasks') or {}).get(family) or {}
    val = rec.get('score')
    if finite(val):
        return float(val)
    return None


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None, 'stdev': None, 'positive': 0, 'negative': 0}
    return {
        'n': len(vals),
        'mean': sum(vals) / len(vals),
        'median': stats.median(vals),
        'min': min(vals),
        'max': max(vals),
        'stdev': stats.pstdev(vals) if len(vals) > 1 else 0.0,
        'positive': sum(1 for v in vals if v > 0),
        'negative': sum(1 for v in vals if v < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    ap.add_argument('--permuted-root', default=str(PERM_EVAL_ROOT))
    ap.add_argument('--families', nargs='*', default=FAMILIES)
    ap.add_argument('--require-permuted', action='store_true')
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    perm_root = pathlib.Path(args.permuted_root)
    if not out.is_absolute():
        out = ROOT / out
    if not perm_root.is_absolute():
        perm_root = ROOT / perm_root
    out.mkdir(parents=True, exist_ok=True)

    meta = read_json(META_PATH) if META_PATH.exists() else {}
    geom = read_json(TOKEN_GEOMETRY) if TOKEN_GEOMETRY.exists() else {}
    recon = read_json(ENTITY_RECON) if ENTITY_RECON.exists() else {}

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        words = int(ck.split('_')[1].replace('M', ''))
        vp = MAX_PER_TARGET / f'dose_max_view_{ck}.json'
        rp = MAX_PER_TARGET / f'dose_max_repeat_{ck}.json'
        pp = perm_root / 'per_target' / f'max_permuted_seed43022_{ck}.json'
        for fam in args.families:
            v = score_from_payload(vp, fam)
            r = score_from_payload(rp, fam)
            p = score_from_payload(pp, fam)
            if v is None or r is None or p is None:
                missing.append({
                    'checkpoint': ck,
                    'family': fam,
                    'view_present': v is not None,
                    'repeat_present': r is not None,
                    'permuted_present': p is not None,
                    'view_per_target': rel(vp),
                    'repeat_per_target': rel(rp),
                    'permuted_per_target': rel(pp),
                })
                continue
            rows.append({
                'checkpoint': ck,
                'words_m': words,
                'family': fam,
                'view_score': v,
                'permuted_score': p,
                'repeat_score': r,
                'view_minus_permuted': v - p,
                'permuted_minus_repeat': p - r,
                'view_minus_repeat': v - r,
                'decomposition_residual': (v - p) + (p - r) - (v - r),
                'view_per_target': rel(vp),
                'repeat_per_target': rel(rp),
                'permuted_per_target': rel(pp),
            })
    write_csv(out / 'permuted_correspondence_by_checkpoint.csv', rows)
    write_csv(out / 'permuted_correspondence_missing.csv', missing)

    summary_rows: list[dict[str, Any]] = []
    windows = [
        ('common_10M_80M', lambda m: m <= 80),
        ('full_10M_100M', lambda m: True),
        ('late_90M_100M', lambda m: m >= 90),
    ]
    for fam in args.families:
        for window, keep in windows:
            subset = [r for r in rows if r['family'] == fam and keep(int(r['words_m']))]
            for contrast in ['view_minus_permuted', 'permuted_minus_repeat', 'view_minus_repeat']:
                vals = [float(r[contrast]) for r in subset]
                summary_rows.append({'family': fam, 'window': window, 'contrast': contrast, **summarize(vals)})
    write_csv(out / 'permuted_correspondence_summary.csv', summary_rows)

    first_max_entity = 2.0625
    ent_vp = next((r for r in summary_rows if r['family'] == 'Entity' and r['window'] == 'common_10M_80M' and r['contrast'] == 'view_minus_permuted'), None)
    ent_pr = next((r for r in summary_rows if r['family'] == 'Entity' and r['window'] == 'common_10M_80M' and r['contrast'] == 'permuted_minus_repeat'), None)
    status = 'PERMUTED_CORRESPONDENCE_READOUT_COMPLETE' if len(missing) == 0 else 'PERMUTED_CORRESPONDENCE_READOUT_WAITING_FOR_PERMUTED'
    if args.require_permuted and missing:
        status = 'PERMUTED_CORRESPONDENCE_READOUT_MISSING_REQUIRED_PERMUTED'
    interpretation = 'Permuted scores are not complete; no aligned-minus-permuted correspondence conclusion is drawn.'
    if ent_vp and ent_vp.get('mean') is not None and ent_pr and ent_pr.get('mean') is not None:
        interpretation = (
            f"Entity common-window V-P={float(ent_vp['mean']):+.3f}, P-R={float(ent_pr['mean']):+.3f}, "
            f"against inherited V-R={first_max_entity:+.4f}. Large V-P with small P-R supports source-view correspondence/record addressability; "
            "small V-P with large P-R supports compact-style companion text without correspondence."
        )

    summary = {
        'status': status,
        'created_utc': now(),
        'max_view_repeat_source_root': rel(MAX_PER_TARGET),
        'permuted_eval_root': rel(perm_root),
        'permuted_metadata': rel(META_PATH),
        'permuted_token_geometry': rel(TOKEN_GEOMETRY),
        'materialization_audit_subset': {
            k: (meta.get('audit') or {}).get(k)
            for k in [
                'assignment_is_permutation',
                'companion_text_multiset_identical_to_max_view',
                'all_sources_receive_different_pair_rewrite',
                'same_doc_assignment_count',
                'same_row_assignment_count',
                'donor_doc_in_target_row_docset_count',
                'row_length_sequence_matches_max_view',
                'word_total_exact_10M',
                'per_slot_rewrite_length_changed_count',
            ]
        },
        'token_shift_vs_max_view': ((geom.get('relative_shift_vs_max_view') or {}).get('permuted')),
        'deberta_reference': {
            'MAX_view_minus_repeat_entity_common10_80': first_max_entity,
            'entity_slope_r2': (((recon.get('entity_curve_fit') or {}).get('ordinary_least_squares') or {}).get('r2')),
        },
        'row_count': len(rows),
        'missing_count': len(missing),
        'summary_rows': summary_rows,
        'missing': missing,
        'interpretation': interpretation,
        'files': {
            'by_checkpoint_csv': rel(out / 'permuted_correspondence_by_checkpoint.csv'),
            'summary_csv': rel(out / 'permuted_correspondence_summary.csv'),
            'missing_csv': rel(out / 'permuted_correspondence_missing.csv'),
            'summary_json': rel(out / 'permuted_correspondence_readout_summary.json'),
            'summary_md': rel(out / 'permuted_correspondence_readout_summary.md'),
        },
        'no_model_inference_no_globalpiqa_superglue_aoa_upload_or_leaderboard': True,
    }
    write_json(out / 'permuted_correspondence_readout_summary.json', summary)
    lines = ['# research permuted correspondence readout', '', f'Status: `{status}`', '', interpretation, '', '## Materialization audit subset', '']
    for k, v in summary['materialization_audit_subset'].items():
        lines.append(f'- {k}: {v}')
    shift = summary.get('token_shift_vs_max_view') or {}
    lines += ['', '## Token/WWM shift versus aligned MAX view', '']
    if shift:
        lines += [
            f"- legal16k token delta: {shift.get('tokens_legal16k_minus_max_view')} ({shift.get('tokens_legal16k_pct_vs_max_view'):+.6f}%)",
            f"- visible seq256 token delta: {shift.get('tokens_visible_seq256_minus_max_view')} ({shift.get('tokens_visible_seq256_pct_vs_max_view'):+.6f}%)",
            f"- WWM group delta: {shift.get('wwm_groups_visible_minus_max_view')} ({shift.get('wwm_groups_visible_pct_vs_max_view'):+.6f}%)",
        ]
    lines += ['', '## Missing rows', '']
    for m in missing[:30]:
        lines.append(f"- {m['checkpoint']} {m['family']}: view={m['view_present']} repeat={m['repeat_present']} permuted={m['permuted_present']}")
    lines += ['', '## Files', '']
    for k, v in summary['files'].items():
        lines.append(f'- {k}: `{v}`')
    (out / 'permuted_correspondence_readout_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': status, 'row_count': len(rows), 'missing_count': len(missing), 'summary_json': summary['files']['summary_json'], 'interpretation': interpretation}, indent=2, ensure_ascii=False), flush=True)
    if args.require_permuted and missing:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
