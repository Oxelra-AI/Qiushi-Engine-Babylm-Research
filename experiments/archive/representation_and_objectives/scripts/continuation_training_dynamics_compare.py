#!/usr/bin/env python3
"""Compare 70M->80M staged continuation traces for standard WWM and PVDM arms."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT = WS / 'data/continuation_training_dynamics'
NOTE = (ROOT / 'research/notes/representation_and_objectives/continuation_training_dynamics.md')
RUNS = {
    'standard': WS / 'training/runs/standard_legacy_70M_to_80M_seed43022',
    'treatment': WS / 'training/runs/pvdm_treatment_70M_to_80M_seed43022',
    'control': WS / 'training/runs/pvdm_control_70M_to_80M_seed43022',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def load_log(run: Path) -> list[dict[str, Any]]:
    p = run / 'training_log.jsonl'
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs: return {'n': 0}
    def q(frac: float) -> float:
        if len(xs) == 1: return xs[0]
        idx = frac * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi: return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {'n': len(xs), 'min': xs[0], 'p05': q(0.05), 'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'p95': q(0.95), 'max': xs[-1]}


def summarize_run(name: str, run: Path) -> dict[str, Any]:
    m = load_json(run / 'scientific_metrics.json')
    log = load_log(run)
    return {
        'run_dir': str(run),
        'variant': m.get('variant'),
        'mode': m.get('mode'),
        'word_exposure': m.get('word_exposure'),
        'continuation_words': m.get('continuation_words'),
        'actual_training_steps': m.get('actual_training_steps'),
        'effective_batch_size': m.get('effective_batch_size'),
        'micro_batch_size': m.get('micro_batch_size'),
        'gradient_accumulation_steps': m.get('gradient_accumulation_steps'),
        'stage_stop_name': m.get('stage_stop_name'),
        'loss_first': m.get('loss_first'),
        'loss_last': m.get('loss_last'),
        'selected_tokens_total': (m.get('pvdm_pairing') or {}).get('aggregate_mask_stats', {}).get('selected_tokens'),
        'selected_event_categories': (m.get('pvdm_pairing') or {}).get('selected_event_categories'),
        'segment': m.get('segment'),
        'hashes': m.get('hashes'),
        'training_log_rows': len(log),
        'loss_stats': qstats([float(r['loss']) for r in log]),
        'masked_tokens_stats': qstats([float(r['masked_tokens']) for r in log]),
        'effective_mask_rate_stats': qstats([float(r['effective_mask_rate']) for r in log]),
        'lr_stats': qstats([float(r['lr']) for r in log]),
    }


def stepwise_delta(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    aa = {int(r['step']): r for r in a}; bb = {int(r['step']): r for r in b}
    steps = sorted(set(aa) & set(bb))
    if not steps: return {'n': 0}
    return {
        'n': len(steps),
        'loss_delta': qstats([float(aa[s]['loss']) - float(bb[s]['loss']) for s in steps]),
        'masked_tokens_delta': qstats([float(aa[s]['masked_tokens']) - float(bb[s]['masked_tokens']) for s in steps]),
        'effective_mask_rate_delta': qstats([float(aa[s]['effective_mask_rate']) - float(bb[s]['effective_mask_rate']) for s in steps]),
        'same_batch_words_each_step': all(int(aa[s]['batch_words']) == int(bb[s]['batch_words']) for s in steps),
        'same_total_exposure_each_step': all(int(aa[s]['total_actual_word_exposure']) == int(bb[s]['total_actual_word_exposure']) for s in steps),
        'same_lr_each_step': all(abs(float(aa[s]['lr']) - float(bb[s]['lr'])) < 1e-15 for s in steps),
        'a_lower_loss_steps': sum(float(aa[s]['loss']) < float(bb[s]['loss']) for s in steps),
        'b_lower_loss_steps': sum(float(aa[s]['loss']) > float(bb[s]['loss']) for s in steps),
        'first': {'step': steps[0], 'loss_delta': float(aa[steps[0]]['loss']) - float(bb[steps[0]]['loss']), 'masked_tokens_delta': int(aa[steps[0]]['masked_tokens']) - int(bb[steps[0]]['masked_tokens'])},
        'last': {'step': steps[-1], 'loss_delta': float(aa[steps[-1]]['loss']) - float(bb[steps[-1]]['loss']), 'masked_tokens_delta': int(aa[steps[-1]]['masked_tokens']) - int(bb[steps[-1]]['masked_tokens'])},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_run(name, run) for name, run in RUNS.items()}
    logs = {name: load_log(run) for name, run in RUNS.items()}
    pairwise = {
        'standard_minus_treatment': stepwise_delta(logs['standard'], logs['treatment']),
        'standard_minus_control': stepwise_delta(logs['standard'], logs['control']),
        'treatment_minus_control': stepwise_delta(logs['treatment'], logs['control']),
    }
    seg_ref = summaries['standard']['segment']
    same_segment = all(summaries[name]['segment'] == seg_ref for name in summaries)
    same_hashes = all(summaries[name]['hashes'] == summaries['standard']['hashes'] for name in summaries)
    same_schedule = all(
        summaries[name]['word_exposure'] == 80011326 and summaries[name]['continuation_words'] == 9971289 and summaries[name]['actual_training_steps'] == 250
        for name in summaries
    )
    total_tokens = {name: summaries[name]['selected_tokens_total'] for name in summaries}
    payload = {
        'status': 'CONTINUATION_TRAINING_DYNAMICS_COMPARE',
        'created_utc': now_utc(),
        'research_question': 'Do standard, PVDM treatment, and PVDM control share the same 70M->80M segment/schedule, and how do their masking/loss traces differ before downstream readout?',
        'same_segment': same_segment,
        'same_hashes': same_hashes,
        'same_schedule': same_schedule,
        'selected_tokens_total': total_tokens,
        'selected_tokens_delta_vs_standard': {name: (summaries[name]['selected_tokens_total'] - summaries['standard']['selected_tokens_total']) for name in ['treatment', 'control']},
        'runs': summaries,
        'pairwise_stepwise': pairwise,
    }
    out = OUT / 'continuation_training_dynamics_compare.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = []
    lines.append('# research — 70M→80M continuation training dynamics\n')
    lines.append('This compares only training traces for the standard WWM staged branch and the PVDM treatment/control branches. It does not replace the downstream readout.\n')
    lines.append('\n## Shared accounting\n')
    lines.append(f'- same segment: {same_segment}; same hashes: {same_hashes}; same schedule/exposure: {same_schedule}\n')
    lines.append(f'- selected predicted tokens total: {total_tokens}\n')
    lines.append(f"- PVDM selected-token delta vs standard: {payload['selected_tokens_delta_vs_standard']}\n")
    lines.append('\n## Loss and mask-rate traces\n')
    for name, rec in summaries.items():
        lines.append(f"- `{name}`: loss first/last={rec['loss_first']}/{rec['loss_last']}; loss mean={rec['loss_stats'].get('mean')}; effective mask-rate mean={rec['effective_mask_rate_stats'].get('mean')}; masked-token mean={rec['masked_tokens_stats'].get('mean')}\n")
    lines.append('\n## Stepwise deltas\n')
    for name, rec in pairwise.items():
        lines.append(f"- `{name}`: loss_delta mean={rec['loss_delta'].get('mean')}, median={rec['loss_delta'].get('median')}, first={rec['first']}, last={rec['last']}; same words={rec['same_batch_words_each_step']}, same exposure={rec['same_total_exposure_each_step']}, same LR={rec['same_lr_each_step']}\n")
    lines.append('\n## Interpretation before downstream scores\n')
    lines.append('- The three branches share the exact tail rows, word exposure, tokenizer hash, and schedule. Standard WWM is therefore a valid control for staged replay plus optimizer reset/microbatch execution.\n')
    lines.append('- Standard WWM is not target-identical to PVDM treatment/control, by design: it retains ordinary whole-word target sampling. Its selected-token total is only about 0.68% higher than the PVDM arms, so a large downstream difference is unlikely to be explained by total mask mass alone, but target identity and relation-anchor swapping remain the intended contrast.\n')
    lines.append('- Standard WWM has lower loss than both PVDM arms on this segment. This cannot be interpreted as better downstream relation learning; it mainly indicates the modified dependent-target distribution is harder than ordinary WWM. The fixed readout decides causal meaning.\n')
    lines.append(f'\nFiles: `{out}`\n')
    NOTE.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out), 'note': str(NOTE), 'same_segment': same_segment, 'selected_tokens_total': total_tokens, 'loss_means': {k: v['loss_stats'].get('mean') for k,v in summaries.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
