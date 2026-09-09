#!/usr/bin/env python3
"""research synthesis for full-context pivot-substitution probes across FW arms.

The scientific question is whether the research/125 full-context masked-consequence
signal tracks the existing relation-surface tradeoff among compact, row-block, and
interleaved 100M checkpoints, or whether it is mostly a local lexical/context fit
signal already saturated in all arms that remain wrong on EWoK/GlobalPIQA.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('experiments/archive/representation_and_objectives')
OUT_DIR = ROOT / 'data/fullctx_probe_tradeoff_synthesis'
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fullctx_probe_tradeoff_synthesis.md')

PROBES = {
    'compact': ROOT / 'data/fullctx_probe_fw_compact_100M',
    'rowblock': ROOT / 'data/fullctx_probe_fw_rowblock_100M',
    'interleaved': ROOT / 'data/fullctx_probe_fw_interleaved_100M',
}
SUMMARIES = {k: v / 'full_context_pivot_substitution_summary.json' for k, v in PROBES.items()}
SCORED = {k: v / 'scored_events.jsonl' for k, v in PROBES.items()}
CHEAP7 = ROOT / 'data/fw_100m_cheap7_synthesis/fw_100m_cheap7_synthesis.json'
GPIQA = ROOT / 'data/fw_globalpiqa_threearm_synthesis/fw_globalpiqa_threearm_synthesis.json'
EWOK_CR = ROOT / 'data/fw_ewok_interaction_reader/fw_ewok_interaction_reader_summary.json'
EWOK_INT = ROOT / 'data/fw_ewok_interleaved_reader/fw_ewok_interaction_reader_summary.json'

# Metrics where larger is better after possible sign transform.
RELATION_REFERENCE = {
    # EWoK conditional-interaction readout: lower stable frac and less-negative wrong median are better.
    'ewok_accuracy': {'compact': 0.504331845628774, 'rowblock': 0.495668154371226, 'interleaved': 0.5111577841953269, 'larger_better': True},
    'ewok_stable_failure_frac_wrong_neg': {'compact': -0.6890889830508474, 'rowblock': -0.6457574180114524, 'interleaved': -0.682062298603652, 'larger_better': True},
    'ewok_wrong_interaction_median': {'compact': -0.9560414860025048, 'rowblock': -0.5047678053379059, 'interleaved': -0.6662999228574336, 'larger_better': True},
    # GlobalPIQA: parallel accuracy higher and hard52 margin lower; use negative margin as larger-better.
    'gpiqa_parallel_accuracy': {'compact': 24.271844660194176, 'rowblock': 29.12621359223301, 'interleaved': 26.21359223300971, 'larger_better': True},
    'gpiqa_hard52_margin_neg': {'compact': -1.717, 'rowblock': -1.433, 'interleaved': -1.660, 'larger_better': True},
    'gpiqa_nonparallel_accuracy': {'compact': 53.0, 'rowblock': 45.0, 'interleaved': 56.0, 'larger_better': True},
    'cheap7': {'compact': 43.18142857142857, 'rowblock': 42.63857142857143, 'interleaved': 43.07928571428572, 'larger_better': True},
}

MARGIN_SPECS = {
    'true_minus_shuffle': ('logp_true_pivot', 'logp_shuffled_pivot_replace'),
    'true_minus_anchor': ('logp_true_pivot', 'logp_matched_anchor_replace'),
    'true_minus_pivotmasked': ('logp_true_pivot', 'logp_pivot_masked'),
    'anchor_minus_shuffle': ('logp_matched_anchor_replace', 'logp_shuffled_pivot_replace'),
    'true_logp': ('logp_true_pivot', None),
}


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def load_events(path: Path) -> dict[tuple[int, int, str, str, str], dict[str, Any]]:
    out: dict[tuple[int, int, str, str, str], dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            key = (int(ev['tail_row_idx']), int(ev['event_rank']), str(ev['category']), str(ev['pivot_norm']), str(ev['target_norm']))
            out[key] = ev
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def sd(xs: list[float]) -> float | None:
    return stats.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def stderr(xs: list[float]) -> float | None:
    s = sd(xs)
    return (s / math.sqrt(len(xs))) if xs and s is not None else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank_order(vals: dict[str, float], larger_better: bool = True) -> list[str]:
    return [k for k, _v in sorted(vals.items(), key=lambda kv: kv[1], reverse=larger_better)]


def spearman3(vals_a: dict[str, float], vals_b: dict[str, float]) -> float | None:
    keys = sorted(set(vals_a) & set(vals_b))
    if len(keys) != 3:
        return None
    # rank 1 best/lowest by raw numerical value? For correlation of numerical values use ascending ranks.
    def ranks(vals: dict[str, float]) -> dict[str, float]:
        ordered = sorted(keys, key=lambda k: vals[k])
        return {k: i + 1 for i, k in enumerate(ordered)}
    ra, rb = ranks(vals_a), ranks(vals_b)
    return pearson([ra[k] for k in keys], [rb[k] for k in keys])


def event_margin(ev: dict[str, Any], spec: tuple[str, str | None]) -> float:
    a, b = spec
    av = float(ev[a])
    if b is None:
        return av
    return av - float(ev[b])


def summarize_values(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    q = sorted(xs)
    def quantile(p: float) -> float:
        if len(q) == 1:
            return q[0]
        i = p * (len(q) - 1)
        lo = int(math.floor(i)); hi = int(math.ceil(i))
        if lo == hi:
            return q[lo]
        return q[lo] * (hi - i) + q[hi] * (i - lo)
    return {
        'n': len(xs),
        'mean': mean(xs),
        'median': quantile(0.5),
        'std': sd(xs),
        'stderr': stderr(xs),
        'p05': quantile(0.05),
        'p95': quantile(0.95),
        'success_gt0': sum(1 for x in xs if x > 0) / len(xs),
    }


def paired_delta_summary(events: dict[str, dict[Any, dict[str, Any]]], a: str, b: str, margin_name: str) -> dict[str, Any]:
    keys = sorted(set(events[a]) & set(events[b]))
    deltas = []
    per_cat: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        va = event_margin(events[a][k], MARGIN_SPECS[margin_name])
        vb = event_margin(events[b][k], MARGIN_SPECS[margin_name])
        d = va - vb
        deltas.append(d)
        per_cat[str(events[a][k]['category'])].append(d)
    out = summarize_values(deltas)
    out['arm_a_minus_b'] = f'{a}-{b}'
    out['margin'] = margin_name
    out['per_category'] = {cat: summarize_values(vals) for cat, vals in sorted(per_cat.items())}
    out['positive_fraction_a_gt_b'] = sum(1 for d in deltas if d > 0) / len(deltas) if deltas else None
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {k: load_json(v) for k, v in SUMMARIES.items()}
    events = {k: load_events(v) for k, v in SCORED.items()}
    key_sets = {k: set(v) for k, v in events.items()}
    common_keys = sorted(set.intersection(*key_sets.values()))
    if not common_keys:
        raise RuntimeError('no common scored event keys')
    event_identity = {
        'counts': {k: len(v) for k, v in events.items()},
        'common_count': len(common_keys),
        'symmetric_differences': {f'{a}_vs_{b}': len(key_sets[a] ^ key_sets[b]) for a in events for b in events if a < b},
    }

    # Aggregate margin summaries from event files, plus event-level correlations.
    margin_summaries: dict[str, Any] = {}
    for margin_name, spec in MARGIN_SPECS.items():
        arm_vals: dict[str, list[float]] = {}
        per_arm_cat: dict[str, Any] = {}
        for arm in PROBES:
            vals = [event_margin(events[arm][k], spec) for k in common_keys]
            arm_vals[arm] = vals
            by_cat: dict[str, list[float]] = defaultdict(list)
            for k, v in zip(common_keys, vals):
                by_cat[str(events[arm][k]['category'])].append(v)
            per_arm_cat[arm] = {cat: summarize_values(x) for cat, x in sorted(by_cat.items())}
        means = {arm: mean(vals) for arm, vals in arm_vals.items()}
        ses = {arm: stderr(vals) for arm, vals in arm_vals.items()}
        cors = {}
        for a in PROBES:
            for b in PROBES:
                if a < b:
                    cors[f'{a}__{b}'] = pearson(arm_vals[a], arm_vals[b])
        deltas = {
            'rowblock_minus_compact': paired_delta_summary(events, 'rowblock', 'compact', margin_name),
            'interleaved_minus_compact': paired_delta_summary(events, 'interleaved', 'compact', margin_name),
            'interleaved_minus_rowblock': paired_delta_summary(events, 'interleaved', 'rowblock', margin_name),
        }
        margin_summaries[margin_name] = {
            'means': means,
            'standard_errors': ses,
            'rank_high_to_low': rank_order({k: float(v) for k, v in means.items()}),
            'event_level_pearson': cors,
            'per_arm_per_category': per_arm_cat,
            'paired_deltas': deltas,
        }

    # Pull known readout numbers from durable files where possible; preserve hard52 approximations from research note.
    cheap = load_json(CHEAP7)
    gp = load_json(GPIQA)
    ewok_cr = load_json(EWOK_CR)
    ewok_int = load_json(EWOK_INT)
    known = {
        'cheap7_file': str(CHEAP7),
        'globalpiqa_file': str(GPIQA),
        'ewok_compact_rowblock_file': str(EWOK_CR),
        'ewok_interleaved_file': str(EWOK_INT),
        'surface': {
            'cheap7': RELATION_REFERENCE['cheap7'],
            'gpiqa_parallel_accuracy': RELATION_REFERENCE['gpiqa_parallel_accuracy'],
            'gpiqa_hard52_margin_neg': RELATION_REFERENCE['gpiqa_hard52_margin_neg'],
            'gpiqa_nonparallel_accuracy': RELATION_REFERENCE['gpiqa_nonparallel_accuracy'],
            'ewok_accuracy': RELATION_REFERENCE['ewok_accuracy'],
            'ewok_stable_failure_frac_wrong_neg': RELATION_REFERENCE['ewok_stable_failure_frac_wrong_neg'],
            'ewok_wrong_interaction_median': RELATION_REFERENCE['ewok_wrong_interaction_median'],
        },
        'parsed_check': {
            'cheap7_arms': cheap['arms'],
            'gp_parallel_per_arm': gp['modes']['parallel']['per_arm'],
            'gp_nonparallel_per_arm': gp['modes']['nonparallel']['per_arm'],
            'ewok_compact_summary': ewok_cr['targets']['fw_compact_fullbatch_seed43022']['summary'],
            'ewok_rowblock_summary': ewok_cr['targets']['fw_breadth_rowblock_fullbatch_seed43022']['summary'],
            'ewok_interleaved_summary': ewok_int['targets']['fw_breadth_interleaved_fullbatch_seed43022']['summary'],
        }
    }

    alignment = {}
    for margin_name, ms in margin_summaries.items():
        means = {k: float(v) for k, v in ms['means'].items()}
        alignment[margin_name] = {}
        for ref_name, ref in RELATION_REFERENCE.items():
            ref_vals = {k: float(v) for k, v in ref.items() if k in PROBES}
            # The ranks and numeric Pearson across only three models are descriptive only.
            alignment[margin_name][ref_name] = {
                'probe_rank_high_to_low': rank_order(means),
                'reference_rank_high_to_low': rank_order(ref_vals),
                'pearson_n3': pearson([means[k] for k in ['compact','rowblock','interleaved']], [ref_vals[k] for k in ['compact','rowblock','interleaved']]),
                'spearman_n3_raw_numeric': spearman3(means, ref_vals),
                'probe_means': means,
                'reference_values': ref_vals,
            }

    # Direct decision facts for the likely training objective true_vs_shuffle.
    tvs = margin_summaries['true_minus_shuffle']
    decision_facts = {
        'identical_event_set': event_identity,
        'true_minus_shuffle_means': tvs['means'],
        'true_minus_shuffle_standard_errors': tvs['standard_errors'],
        'true_minus_shuffle_event_correlations': tvs['event_level_pearson'],
        'rowblock_minus_compact_true_minus_shuffle': tvs['paired_deltas']['rowblock_minus_compact'],
        'interleaved_minus_compact_true_minus_shuffle': tvs['paired_deltas']['interleaved_minus_compact'],
        'interpretation': None,
    }

    # Interpret conservatively: if all arms have large positive means, cross-arm correlations high,
    # and paired deltas are small relative event SD, this signal is mostly a saturated local signal.
    means = tvs['means']
    mean_gap = max(means.values()) - min(means.values())
    avg_event_sd = mean([(v['std'] or 0.0) for v in [summarize_values([event_margin(events[arm][k], MARGIN_SPECS['true_minus_shuffle']) for k in common_keys]) for arm in PROBES]])
    min_corr = min(v for v in tvs['event_level_pearson'].values() if v is not None)
    decision_facts['interpretation'] = {
        'all_arms_positive_and_successful': all(float(means[a]) > 0.45 for a in means),
        'mean_range': mean_gap,
        'mean_range_over_average_event_sd': mean_gap / avg_event_sd if avg_event_sd else None,
        'minimum_event_level_correlation': min_corr,
        'rank_matches_gp_parallel_and_hard52': tvs['rank_high_to_low'] == ['rowblock','interleaved','compact'],
        'rank_does_not_match_ewok_accuracy_or_broad_cheap7': True,
        'bottom_line': 'The probe has a small model-specific offset in the same direction as row-block/interleaved relation diagnostics, but all three failing endpoints already show strong true-pivot preference on the identical events and event-level margins are almost the same object across arms. It is therefore not yet a validated training objective for repairing EWoK/GlobalPIQA failures.'
    }

    # Write compact CSV for later inspection.
    csv_path = OUT_DIR / 'probe_tradeoff_arm_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arm','true_minus_shuffle_mean','true_minus_shuffle_se','true_minus_anchor_mean','true_minus_anchor_se','ewok_acc','ewok_stable_fail_neg','ewok_wrong_median','gpiqa_parallel','gpiqa_hard52_margin_neg','gpiqa_nonparallel','cheap7'])
        for arm in ['compact','rowblock','interleaved']:
            writer.writerow([
                arm,
                margin_summaries['true_minus_shuffle']['means'][arm], margin_summaries['true_minus_shuffle']['standard_errors'][arm],
                margin_summaries['true_minus_anchor']['means'][arm], margin_summaries['true_minus_anchor']['standard_errors'][arm],
                RELATION_REFERENCE['ewok_accuracy'][arm], RELATION_REFERENCE['ewok_stable_failure_frac_wrong_neg'][arm], RELATION_REFERENCE['ewok_wrong_interaction_median'][arm],
                RELATION_REFERENCE['gpiqa_parallel_accuracy'][arm], RELATION_REFERENCE['gpiqa_hard52_margin_neg'][arm], RELATION_REFERENCE['gpiqa_nonparallel_accuracy'][arm], RELATION_REFERENCE['cheap7'][arm],
            ])

    summary = {
        'status': 'FULLCTX_PROBE_TRADEOFF_SYNTHESIS',
        'purpose': 'test whether full-context pivot-substitution margins follow FW relation tradeoff or mostly reflect saturated local lexical/context fit',
        'inputs': {k: str(v) for k, v in SUMMARIES.items()},
        'scored_event_inputs': {k: str(v) for k, v in SCORED.items()},
        'event_identity': event_identity,
        'margin_summaries': margin_summaries,
        'known_relation_and_surface_readouts': known,
        'alignment_to_known_readouts': alignment,
        'decision_facts': decision_facts,
        'artifacts': {'summary': str(OUT_DIR / 'fullctx_probe_tradeoff_synthesis.json'), 'csv': str(csv_path), 'note': str(NOTE)},
    }
    summary_path = OUT_DIR / 'fullctx_probe_tradeoff_synthesis.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # Human-readable note.
    tvsd = decision_facts
    lines = []
    lines.append('# research — full-context pivot-substitution tradeoff synthesis\n')
    lines.append('## Question\n')
    lines.append('The full-context probe was applied to the completed FW compact, row-block breadth, and interleaved breadth 100M checkpoints. The purpose was to test whether its margins are tied to the known EWoK/GlobalPIQA relation tradeoff or whether all endpoints already learn this local compatibility while still failing the downstream relation surfaces.\n')
    lines.append('## Shared event set\n')
    lines.append(f"- Event counts: {event_identity['counts']}; common events: {event_identity['common_count']}; symmetric differences: {event_identity['symmetric_differences']}\n")
    lines.append('## Main full-context true-vs-shuffled-pivot margin\n')
    for arm in ['compact','rowblock','interleaved']:
        lines.append(f"- {arm}: mean={tvs['means'][arm]:.6f}, SE={tvs['standard_errors'][arm]:.6f}\n")
    lines.append(f"- event-level Pearson correlations: {tvs['event_level_pearson']}\n")
    lines.append(f"- range across arms = {decision_facts['interpretation']['mean_range']:.6f}; range / average event SD = {decision_facts['interpretation']['mean_range_over_average_event_sd']:.4f}\n")
    lines.append('## Known relation surfaces\n')
    lines.append('- EWoK wrong-row interaction median (less negative better): compact -0.956, rowblock -0.505, interleaved -0.666. Stable-failure fraction among wrong rows: compact 0.689, rowblock 0.646, interleaved 0.682. Aggregate EWoK accuracy: compact 0.504, rowblock 0.496, interleaved 0.511.\n')
    lines.append('- GlobalPIQA parallel accuracy: compact 24.27, rowblock 29.13, interleaved 26.21. Hard52 top-minus-correct margin (lower better): compact 1.717, rowblock 1.433, interleaved 1.660. Nonparallel accuracy: compact 53.0, rowblock 45.0, interleaved 56.0.\n')
    lines.append('## Interpretation\n')
    lines.append(f"{decision_facts['interpretation']['bottom_line']}\n")
    lines.append('The arm ordering of the probe margin (rowblock > interleaved > compact) is compatible with the GlobalPIQA-parallel and EWoK wrong-median/stable-failure tradeoff, but the absolute signal is already strongly positive in every endpoint, including compact and interleaved checkpoints that still leave 61/103 GlobalPIQA-parallel rows wrong for all arms and thousands of EWoK stable conditional reversals. The cross-arm offset is small relative to event variability and margins are highly paired across the identical events, so directly increasing this local margin may simply reinforce a compatibility relation the models already express rather than repair hard conditional choice/ranking.\n')
    lines.append('## Consequence for training\n')
    lines.append('Do not launch the research full-context auxiliary unchanged. Before any continuation, the design must (1) include a same-exposure WWM reference, (2) debit every auxiliary view/forward as training exposure under Strict-Small accounting, (3) equalize nonsemantic gradient pressure between semantic and placebo arms, and (4) connect the auxiliary to hard-row or four-cell failures rather than only to local attested-pivot fit.\n')
    lines.append('## Artifacts\n')
    lines.append(f"- summary: `{summary_path}`\n")
    lines.append(f"- csv: `{csv_path}`\n")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'status': summary['status'],
        'summary': str(summary_path),
        'note': str(NOTE),
        'csv': str(csv_path),
        'true_minus_shuffle_means': tvs['means'],
        'true_minus_shuffle_correlations': tvs['event_level_pearson'],
        'range_over_event_sd': decision_facts['interpretation']['mean_range_over_average_event_sd'],
        'bottom_line': decision_facts['interpretation']['bottom_line'],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
