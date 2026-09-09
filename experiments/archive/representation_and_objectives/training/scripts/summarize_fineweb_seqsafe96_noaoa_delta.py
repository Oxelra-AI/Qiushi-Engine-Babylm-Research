#!/usr/bin/env python3
"""Summarize no-AoA official-compatible cached FineWeb seqsafe96 contrast deltas."""
from __future__ import annotations
import argparse, json, pathlib
from typing import Any

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "equal7_full_eval"]


def load_summary(out_root: pathlib.Path, target: str) -> dict[str, Any]:
    p = out_root / f"{target}_trajectory_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding='utf-8'))


def as_float(x: Any) -> float | None:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None


def ck_key(s: str) -> int:
    try:
        return int(s.split('_')[1].rstrip('M'))
    except Exception:
        return 10**9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--treatment-target', default='fineweb_seqsafe96_treatment')
    ap.add_argument('--control-target', default='fineweb_seqsafe96_control')
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)
    treatment = load_summary(out_root, args.treatment_target)
    control = load_summary(out_root, args.control_target)
    ttab, ctab = treatment.get('table', {}), control.get('table', {})
    checkpoints = sorted(set(ttab) & set(ctab), key=ck_key)
    rows: dict[str, dict[str, Any]] = {}
    for ck in checkpoints:
        tr, cr = ttab.get(ck, {}), ctab.get(ck, {})
        row = {'treatment': {}, 'control': {}, 'delta_treatment_minus_control': {}}
        for col in COLUMNS:
            tv, cv = as_float(tr.get(col)), as_float(cr.get(col))
            row['treatment'][col] = tv; row['control'][col] = cv
            row['delta_treatment_minus_control'][col] = None if tv is None or cv is None else round(tv - cv, 6)
        rows[ck] = row
    valid = {ck:r for ck,r in rows.items() if r['delta_treatment_minus_control'].get('equal7_full_eval') is not None}
    best_delta = max(valid.items(), key=lambda kv: kv[1]['delta_treatment_minus_control']['equal7_full_eval']) if valid else None
    best_treatment = max(valid.items(), key=lambda kv: kv[1]['treatment']['equal7_full_eval']) if valid else None
    best_control = max(valid.items(), key=lambda kv: kv[1]['control']['equal7_full_eval']) if valid else None
    payload = {
        'status': 'FINEWEB_SEQSAFE96_NOAOA_DELTA_SUMMARY',
        'out_root': str(out_root), 'treatment_target': args.treatment_target, 'control_target': args.control_target,
        'columns': COLUMNS, 'checkpoints': checkpoints, 'matched_rows': rows,
        'best_by_delta_equal7': {'checkpoint': best_delta[0], 'row': best_delta[1]} if best_delta else None,
        'best_treatment_equal7': {'checkpoint': best_treatment[0], 'row': best_treatment[1]} if best_treatment else None,
        'best_control_equal7': {'checkpoint': best_control[0], 'row': best_control[1]} if best_control else None,
        'interpretation': {
            'positive_source_signal': 'A robust treatment-control equal7 gain, especially on EWoK/Entity/COMPS/GlobalPIQA without losing Supplement/Reading, supports broader factual-source replacement from cached FineWeb-Edu under seq-safe matched construction.',
            'weak_or_negative_signal': 'Near-zero or negative matched delta weakens cached FineWeb random-quality replacement as a route; the source is not leader-quality simplification-pair data and should not be scaled blindly.'
        }
    }
    out_json = out_root / 'fineweb_seqsafe96_delta_summary.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research cached FineWeb seqsafe96 no-AoA delta summary\n\n', f"Treatment: `{args.treatment_target}`\n\n", f"Control: `{args.control_target}`\n\n", '| checkpoint | treatment equal7 | control equal7 | delta | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |\n', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n']
    def fmt(x): return '' if x is None else f'{x:.4f}'
    for ck in checkpoints:
        r=rows[ck]; d=r['delta_treatment_minus_control']
        lines.append(f"| {ck} | {fmt(r['treatment']['equal7_full_eval'])} | {fmt(r['control']['equal7_full_eval'])} | {fmt(d['equal7_full_eval'])} | {fmt(d['BLiMP'])} | {fmt(d['Supplement'])} | {fmt(d['EWoK'])} | {fmt(d['Entity'])} | {fmt(d['COMPS'])} | {fmt(d['GlobalPIQA'])} | {fmt(d['Reading'])} |\n")
    if best_delta:
        lines.append(f"\nBest delta checkpoint: `{best_delta[0]}` with equal7 delta {best_delta[1]['delta_treatment_minus_control']['equal7_full_eval']:.4f}.\n")
    out_md = out_root / 'fineweb_seqsafe96_delta_summary.md'
    out_md.write_text(''.join(lines)+f"\nJSON: `{out_json}`\n", encoding='utf-8')
    print(json.dumps({'status':payload['status'], 'checkpoints':checkpoints, 'best_by_delta_equal7':payload['best_by_delta_equal7'], 'best_treatment_equal7':payload['best_treatment_equal7'], 'best_control_equal7':payload['best_control_equal7'], 'out_json':str(out_json), 'out_md':str(out_md)}, indent=2, ensure_ascii=False))

if __name__ == '__main__': main()
