#!/usr/bin/env python3
"""research route arithmetic for compacting inherited Qwen second views.

This is a CPU-only design audit. It does not generate text, train, or evaluate.
It estimates how much strict-small word budget could be recovered if the
inherited near-length Qwen rewrites were replaced by shorter faithful views,
and how that compares with the official slice displaced in the FineWeb compact
overlay.
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List

ROOT = pathlib.Path('.')
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/qwen_pair_density_redesign_budget'
NOTE = ROOT / 'research/notes/frontier_consolidation/qwen_pair_density_redesign_budget.md'
PAIR_PATH = ROOT / 'experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl'
CLEAN_META = ROOT / 'experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json'
DENSITY_META = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json'
LOSS_NOTE_JSON = ROOT / 'experiments/archive/frontier_consolidation/data/compact_core_full_loss_anatomy/compact_core_full_loss_anatomy.json'

RATIOS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MIN_REWRITE_WORDS = 6


def read_json(path: pathlib.Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stats(xs: List[float]) -> Dict[str, float]:
    ys = sorted(xs)
    if not ys:
        return {'n': 0}
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)
    return {
        'n': len(ys),
        'min': ys[0],
        'p05': q(0.05),
        'mean': mean(ys),
        'median': median(ys),
        'p95': q(0.95),
        'max': ys[-1],
        'sum': sum(ys),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(read_jsonl(PAIR_PATH))
    clean_meta = read_json(CLEAN_META)
    density_meta = read_json(DENSITY_META)
    loss = read_json(LOSS_NOTE_JSON) if LOSS_NOTE_JSON.exists() else None

    orig_words = [int(p['original_words']) for p in pairs]
    rew_words = [int(p['rewrite_words']) for p in pairs]
    pair_words = [int(p['pair_words']) for p in pairs]
    by_source = defaultdict(lambda: {'pairs': 0, 'orig_words': 0, 'rewrite_words': 0, 'pair_words': 0})
    by_cohort = defaultdict(lambda: {'pairs': 0, 'orig_words': 0, 'rewrite_words': 0, 'pair_words': 0})
    duplicate_by_example = Counter()
    for p in pairs:
        s = by_source[p['source']]
        c = by_cohort[p.get('cohort', '(none)')]
        for rec in (s, c):
            rec['pairs'] += 1
            rec['orig_words'] += int(p['original_words'])
            rec['rewrite_words'] += int(p['rewrite_words'])
            rec['pair_words'] += int(p['pair_words'])
        duplicate_by_example[str(p['example_id'])] += 1

    ratio_scenarios = []
    for r in RATIOS:
        target_rew = [max(MIN_REWRITE_WORDS, math.ceil(int(p['original_words']) * r)) for p in pairs]
        target_sum = sum(target_rew)
        saved = sum(rew_words) - target_sum
        total_pair_after = sum(orig_words) + target_sum
        ratio_scenarios.append({
            'target_ratio': r,
            'min_rewrite_words': MIN_REWRITE_WORDS,
            'projected_rewrite_words': target_sum,
            'projected_total_pair_words': total_pair_after,
            'saved_words_vs_current_rewrites': saved,
            'saved_fraction_of_10M': saved / 10_000_000,
            'current_pair_words': sum(pair_words),
            'current_rewrite_words': sum(rew_words),
            'current_original_words': sum(orig_words),
            'saved_vs_fineweb_changed_block_fraction': saved / density_meta['changed_block_budget_words'],
            'additional_originals_at_existing_mean_source_words': saved / (sum(orig_words) / len(orig_words)) if orig_words else None,
            'additional_pairs_at_projected_pair_word_cost': saved / ((sum(orig_words) / len(orig_words)) + max(MIN_REWRITE_WORDS, math.ceil((sum(orig_words) / len(orig_words)) * r))),
        })

    # What if only high-overlap / near-length pairs are compacted? These are likely
    # the redundant cases and need less semantic surgery than already diverse pairs.
    targeted = []
    bins = {
        'len_ratio_ge_0p90': lambda p: float(p['len_ratio']) >= 0.90,
        'len_ratio_0p75_0p90': lambda p: 0.75 <= float(p['len_ratio']) < 0.90,
        'content_overlap_ge_0p80': lambda p: float(p['content_overlap']) >= 0.80,
        'content_overlap_ge_0p80_and_len_ge_0p90': lambda p: float(p['content_overlap']) >= 0.80 and float(p['len_ratio']) >= 0.90,
    }
    for name, pred in bins.items():
        subset = [p for p in pairs if pred(p)]
        cur = sum(int(p['rewrite_words']) for p in subset)
        for r in [0.55, 0.60, 0.65, 0.70]:
            tgt = sum(max(MIN_REWRITE_WORDS, math.ceil(int(p['original_words']) * r)) for p in subset)
            saved = cur - tgt
            targeted.append({
                'subset': name,
                'pairs': len(subset),
                'current_rewrite_words': cur,
                'target_ratio': r,
                'projected_rewrite_words': tgt,
                'saved_words': saved,
                'saved_fraction_of_10M': saved / 10_000_000,
                'additional_originals_at_existing_mean_source_words': saved / (sum(orig_words) / len(orig_words)) if orig_words else None,
            })

    restored_source_budget = density_meta['base_split']['row_holdout']['heldout_words_by_source']
    compact_core_pair_words = density_meta['families']['compact_core_neutral']['pair_words']
    compact_core_topup = density_meta['families']['compact_core_neutral']['neutral_cleanqwen_topup_words_inside_changed_block']
    result = {
        'status': 'QWEN_PAIR_DENSITY_REDESIGN_BUDGET',
        'method': 'CPU-only word-budget projection over inherited COMPACT_EXPERIENCE selected Qwen pairs. It is not evidence that compacted Qwen views will preserve meaning or scores.',
        'inputs': {
            'selected_pairs': str(PAIR_PATH),
            'clean_materialization_metadata': str(CLEAN_META),
            'density_overlay_metadata': str(DENSITY_META),
            'loss_anatomy': str(LOSS_NOTE_JSON),
        },
        'current_qwen_pairs': {
            'pairs': len(pairs),
            'unique_example_ids': len(duplicate_by_example),
            'example_reuse_histogram': dict(Counter(duplicate_by_example.values())),
            'original_words': sum(orig_words),
            'rewrite_words': sum(rew_words),
            'pair_words': sum(pair_words),
            'rewrite_over_source_weighted': sum(rew_words) / sum(orig_words),
            'pair_word_fraction_10M': sum(pair_words) / 10_000_000,
            'rewrite_word_fraction_10M': sum(rew_words) / 10_000_000,
            'by_source': dict(by_source),
            'by_cohort': dict(by_cohort),
            'original_word_stats': stats(orig_words),
            'rewrite_word_stats': stats(rew_words),
            'len_ratio_stats': stats([float(p['len_ratio']) for p in pairs]),
            'content_overlap_stats': stats([float(p['content_overlap']) for p in pairs]),
        },
        'ratio_scenarios': ratio_scenarios,
        'targeted_redundant_subset_scenarios': targeted,
        'fineweb_overlay_displaced_official_slice': {
            'heldout_words_total': density_meta['base_split']['heldout_words'],
            'heldout_words_by_source': restored_source_budget,
            'compact_core_source_plus_view_pair_words': compact_core_pair_words,
            'compact_core_neutral_topup_words': compact_core_topup,
            'changed_block_words': density_meta['changed_block_budget_words'],
            'interpretation': 'FineWeb compact_core both inserted compact generated views and removed 423,520 official clean-Qwen words. Qwen-internal compacting would instead preserve those official words and release budget from the inherited near-length generated side.',
        },
        'loss_anatomy_summary': {
            'compact_minus_clean_component_deltas': loss['route_arithmetic']['component_deltas_compact_minus_clean'] if loss else None,
            'structured_loss_note': 'research loss anatomy localizes compact-core losses to QA-congruence, weak practical GlobalPIQA, and RTE/MRPC rather than broad collapse.' if loss else None,
        },
    }
    json_path = OUT_DIR / 'qwen_pair_density_redesign_budget.json'
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = []
    lines.append('# research Qwen-pair density redesign budget')
    lines.append('')
    lines.append('CPU-only route arithmetic over the inherited COMPACT_EXPERIENCE clean-Qwen selected pairs. This estimates a safer density redesign: release word budget inside the already-beneficial same-window Qwen-pair block instead of replacing an additional broad official slice with FineWeb packets. It does not generate compacted text or prove fidelity.')
    lines.append('')
    cur = result['current_qwen_pairs']
    lines.append('## Current inherited pair block')
    lines.append(f"- {cur['pairs']} selected Qwen pairs, {cur['unique_example_ids']} unique official example IDs, {cur['pair_words']:,} pair words = {100*cur['pair_word_fraction_10M']:.3f}% of the strict-small pool.")
    lines.append(f"- Original side {cur['original_words']:,} words; generated side {cur['rewrite_words']:,} words; weighted rewrite/source ratio {cur['rewrite_over_source_weighted']:.3f}.")
    lines.append('- Pair words by source: ' + ', '.join(f"{k} {v['pair_words']:,}" for k, v in sorted(cur['by_source'].items(), key=lambda kv: -kv[1]['pair_words'])))
    lines.append('')
    lines.append('## If Qwen rewrites were compacted while originals and official filler stayed protected')
    for sc in ratio_scenarios:
        lines.append(f"- Target rewrite/source {sc['target_ratio']:.2f}: projected rewrite words {sc['projected_rewrite_words']:,}, pair block {sc['projected_total_pair_words']:,}, saved {sc['saved_words_vs_current_rewrites']:,} words ({100*sc['saved_fraction_of_10M']:.2f}% of corpus), enough for about {sc['additional_originals_at_existing_mean_source_words']:.0f} additional mean-length originals; saved mass is {100*sc['saved_vs_fineweb_changed_block_fraction']:.1f}% of the research FineWeb changed block.")
    lines.append('')
    lines.append('## Conservative subset compaction')
    for sc in targeted:
        if sc['target_ratio'] in (0.60, 0.65):
            lines.append(f"- {sc['subset']} at ratio {sc['target_ratio']:.2f}: {sc['pairs']} pairs, saved {sc['saved_words']:,} words (~{sc['additional_originals_at_existing_mean_source_words']:.0f} mean-length originals).")
    lines.append('')
    h = result['fineweb_overlay_displaced_official_slice']
    lines.append('## Why this differs from the failed FineWeb overlay')
    lines.append(f"- research compact_core displaced {h['heldout_words_total']:,} official clean-Qwen words: " + ', '.join(f"{k} {v:,}" for k, v in h['heldout_words_by_source'].items()) + '.')
    lines.append(f"- Its compact_core inserted {h['compact_core_source_plus_view_pair_words']:,} FineWeb source+view words plus {h['compact_core_neutral_topup_words']:,} neutral clean-Qwen top-up words. The research loss anatomy suggests that replacing the official slice may be entangled with the compact-view mechanism.")
    lines.append('- Qwen-internal density would protect the full official source distribution and ask a cleaner question: can shorter faithful second views inside an already successful paired block buy additional reusable official/practical/event evidence without losing QA-congruence and RTE/MRPC calibration?')
    lines.append('')
    lines.append('## Research implication')
    lines.append('- If pending reinvest/full-AoA evidence shows compact views remain scientifically useful but FineWeb replacement hurts complete surface, the next experiment should be a small pilot for compacting inherited Qwen pairs, not another FineWeb overlay. The minimum-cost path is to generate/accept only a few thousand high-redundancy Qwen-pair compact views, materialize a 10M probe that preserves official filler and row geometry, and screen it before any 100M run.')
    lines.append('- If pending evidence instead shows compression itself broadly causes AoA/SuperGLUE harm independent of source displacement, this Qwen-internal route should not be trained; the saved arithmetic is a design option, not authorization.')
    lines.append('')
    lines.append(f"Machine-readable JSON: `{json_path}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'json': str(json_path), 'note': str(NOTE)}, indent=2))


if __name__ == '__main__':
    main()
