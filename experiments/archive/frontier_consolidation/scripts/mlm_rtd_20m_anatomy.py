#!/usr/bin/env python3
"""research: anatomy of MLM+RTD-GDES 20M screen.

Parses existing best_temperature_report.txt files and training logs.  No model
inference or GPU work.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path('experiments/archive/frontier_consolidation')
BASE_PER = ROOT/'data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json'
RTD_PER = ROOT/'data/mlm_rtd_20M_eval/per_target/mlm_rtd_lambda1_seed43022_20M.json'
BASE_RUN = ROOT/'training/runs/complianttok_reinvest_seed43022_r2'
RTD_RUN = ROOT/'training/runs/mlm_rtd_lambda1_seed43022_20M'
OUT_DIR = ROOT/'data/mlm_rtd_20M_anatomy'


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def parse_report(path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    cur = None
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('### '):
            name = line[4:].strip()
            cur = name
            sections.setdefault(cur, {})
            continue
        if cur and ':' in line:
            k, v = line.rsplit(':', 1)
            try:
                sections[cur][k.strip()] = float(v.strip())
            except ValueError:
                pass
    return sections


def score_from_payload(payload: dict, col: str) -> float:
    tasks = payload['tasks']
    if col in ['BLiMP','Supplement','EWoK','Entity','COMPS']:
        return float(tasks[col]['score'])
    if col == 'GlobalPIQA_parallel':
        return float(tasks[col]['score'])
    if col == 'GlobalPIQA_nonparallel':
        return float(tasks[col]['score'])
    if col == 'GlobalPIQA':
        return mean([float(tasks['GlobalPIQA_parallel']['score']), float(tasks['GlobalPIQA_nonparallel']['score'])])
    if col == 'Reading':
        return float(tasks[col]['scores']['Reading'])
    raise KeyError(col)


def report_path(payload: dict, col: str) -> Path | None:
    rec = payload['tasks'].get(col)
    if not rec:
        return None
    p = rec.get('report')
    return Path(p) if p else None


def compare_section(base_sections, rtd_sections, section):
    b = base_sections.get(section, {})
    r = rtd_sections.get(section, {})
    keys = sorted(set(b) | set(r))
    rows = []
    for k in keys:
        if k in b and k in r:
            rows.append({'key': k, 'baseline': b[k], 'mlm_rtd': r[k], 'delta': r[k]-b[k]})
    rows.sort(key=lambda x: x['delta'])
    return rows


def load_log(p: Path):
    rows = []
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_json(BASE_PER)
    rtd = read_json(RTD_PER)
    cols = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA_parallel','GlobalPIQA_nonparallel','GlobalPIQA','Reading']
    scores = []
    for c in cols:
        b = score_from_payload(base, c)
        r = score_from_payload(rtd, c)
        scores.append({'column': c, 'baseline': b, 'mlm_rtd': r, 'delta': r-b})
    cheap_cols = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']
    cheap7_base = mean([score_from_payload(base, c) for c in cheap_cols])
    cheap7_rtd = mean([score_from_payload(rtd, c) for c in cheap_cols])

    sections = {}
    for c in ['BLiMP','Supplement','EWoK','Entity','COMPS']:
        bp = report_path(base, c)
        rp = report_path(rtd, c)
        if bp and rp and bp.exists() and rp.exists():
            bs = parse_report(bp)
            rs = parse_report(rp)
            sections[c] = {}
            for sec in sorted(set(bs) | set(rs)):
                rows = compare_section(bs, rs, sec)
                if rows:
                    sections[c][sec] = rows

    # Training loss comparison over matched first 506 steps.
    b_log = load_log(BASE_RUN/'training_log.jsonl')
    r_log = load_log(RTD_RUN/'training_log.jsonl')
    loss_rows = []
    for b, r in zip(b_log[:len(r_log)], r_log):
        bl = float(b.get('loss', b.get('mlm_loss')))
        rl = float(r['mlm_loss'])
        loss_rows.append({'step': int(r['step']), 'baseline_loss': bl, 'mlm_rtd_mlm_loss': rl, 'delta': rl-bl})
    loss_summary = {
        'n_matched_steps': len(loss_rows),
        'mean_delta_all': mean([x['delta'] for x in loss_rows]),
        'mean_delta_first50': mean([x['delta'] for x in loss_rows[:50]]),
        'mean_delta_last50': mean([x['delta'] for x in loss_rows[-50:]]),
        'selected_steps': [x for x in loss_rows if x['step'] in {1,50,100,150,200,250,300,350,400,450,500,506}],
    }

    # Note likely score anatomy in moved columns.
    result = {
        'status': 'MLM_RTD_20M_ANATOMY',
        'scores': scores,
        'cheap7_baseline': cheap7_base,
        'cheap7_mlm_rtd': cheap7_rtd,
        'cheap7_delta': cheap7_rtd - cheap7_base,
        'finegrained_sections': sections,
        'loss_summary': loss_summary,
        'paths': {
            'baseline_per_target': str(BASE_PER),
            'mlm_rtd_per_target': str(RTD_PER),
            'baseline_run': str(BASE_RUN),
            'mlm_rtd_run': str(RTD_RUN),
        }
    }
    (OUT_DIR/'mlm_rtd_20M_anatomy.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    lines = ['# research MLM+RTD-GDES 20M score anatomy', '']
    lines.append('## Score movement')
    lines.append('| column | baseline20M | MLM+RTD20M | delta |')
    lines.append('|---|---:|---:|---:|')
    for row in scores:
        lines.append(f"| {row['column']} | {row['baseline']:.4f} | {row['mlm_rtd']:.4f} | {row['delta']:+.4f} |")
    lines.append(f"| cheap7 | {cheap7_base:.4f} | {cheap7_rtd:.4f} | {cheap7_rtd-cheap7_base:+.4f} |")
    lines.append('')
    lines.append('## MLM loss comparison against research legal baseline')
    lines.append(f"- Matched steps: {loss_summary['n_matched_steps']}")
    lines.append(f"- Mean MLM-loss delta all steps: {loss_summary['mean_delta_all']:+.6f}")
    lines.append(f"- Mean delta first 50 steps: {loss_summary['mean_delta_first50']:+.6f}")
    lines.append(f"- Mean delta last 50 steps: {loss_summary['mean_delta_last50']:+.6f}")
    lines.append('- Selected steps:')
    for x in loss_summary['selected_steps']:
        lines.append(f"  - step {x['step']}: baseline {x['baseline_loss']:.5f}, MLM+RTD MLM {x['mlm_rtd_mlm_loss']:.5f}, delta {x['delta']:+.5f}")
    lines.append('')
    lines.append('## Largest fine-grained movements')
    for c in ['Supplement','EWoK','Entity','BLiMP']:
        lines.append(f"### {c}")
        for sec in ['UID ACCURACY','FIELD ACCURACY','CONTEXT_CONTRAST ACCURACY','CONTEXT_TYPE ACCURACY','TARGET_CONTRAST ACCURACY','LINGUISTICS_TERM ACCURACY']:
            rows = sections.get(c, {}).get(sec, [])
            if not rows:
                continue
            lines.append(f"#### {sec}")
            lines.append('| key | baseline | MLM+RTD | delta |')
            lines.append('|---|---:|---:|---:|')
            # show all if <=12 else five worst and five best
            show = rows if len(rows) <= 12 else rows[:5] + rows[-5:]
            for row in show:
                lines.append(f"| {row['key']} | {row['baseline']:.2f} | {row['mlm_rtd']:.2f} | {row['delta']:+.2f} |")
            lines.append('')
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/mlm_rtd_20M_anatomy/mlm_rtd_20M_anatomy.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'cheap7_delta': result['cheap7_delta'], 'out_dir': str(OUT_DIR)}, indent=2))


if __name__ == '__main__':
    main()
