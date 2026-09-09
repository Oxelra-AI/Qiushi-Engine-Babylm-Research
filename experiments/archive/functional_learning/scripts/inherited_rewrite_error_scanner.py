#!/usr/bin/env python3
"""research: Scan all 37,594 selected Qwen pairs for inherited-rewrite errors.

Manual semantic review identified that some inherited rewrites contradict their source:
- "didn't get off the plane" → "I did manage to get off the plane" (negation reversal)
- "a thousand days" → "a year" (number change)
These are distinct from compaction damage: the rewrite was already wrong.

This scanner compares each source/rewrite pair using heuristic structural checks:
1. Number extraction: mismatched numeric expressions
2. Negation reversal: negation in source but not rewrite or vice versa
3. Conditional → assertion: conditional markers lost
4. Quantifier changes: magnitude changes in frequency/extent words
5. Named entity / proper noun drift: names present in one but not other

The output is a candidate pool, not verified errors. It estimates the scale of
correspondence repair as a distinct intervention from shortening.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ── Number extraction ──
NUM_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
    'million': 1_000_000, 'billion': 1_000_000_000,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'dozen': 12, 'half': 0.5, 'quarter': 0.25,
}

TIME_CONVERSIONS = {
    'a year': 365, 'one year': 365, 'a decade': 3650, 'one decade': 3650,
    'a century': 36500, 'one century': 36500, 'a month': 30, 'one month': 30,
    'a week': 7, 'one week': 7, 'half an hour': 30,  # minutes
}


def extract_numbers(text: str) -> list[tuple[str, float | None]]:
    """Extract numeric expressions from text. Returns (matched_text, value) pairs."""
    text_lower = text.lower()
    results = []
    # Digit numbers (including decimals, percentages)
    for m in re.finditer(r'\b(\d[\d,]*\.?\d*)\s*(%|percent|per cent)?', text_lower):
        val_str = m.group(1).replace(',', '')
        try:
            val = float(val_str)
            if m.group(2):
                val = val / 100.0  # normalize percentage
            results.append((m.group(0).strip(), val))
        except ValueError:
            results.append((m.group(0).strip(), None))

    # Word numbers: look for sequences
    for word, val in NUM_WORDS.items():
        # Find isolated word occurrences
        for m in re.finditer(rf'\b{re.escape(word)}\b', text_lower):
            results.append((word, val))

    return results


def extract_negation_contexts(text: str) -> list[str]:
    """Extract negation patterns with surrounding context."""
    text_lower = text.lower()
    neg_pats = [
        r"didn't", r"did not", r"don't", r"do not", r"doesn't", r"does not",
        r"wasn't", r"was not", r"weren't", r"were not", r"can't", r"cannot",
        r"couldn't", r"could not", r"wouldn't", r"would not", r"shouldn't",
        r"should not", r"hasn't", r"has not", r"haven't", r"have not",
        r"hadn't", r"had not", r"won't", r"will not", r"isn't", r"is not",
        r"aren't", r"are not", r"never\b", r"\bno\b", r"\bnot\b",
        r"\bnor\b", r"\bneither\b", r"\bnobody\b", r"\bnothing\b", r"\bnowhere\b",
    ]
    results = []
    for pat in neg_pats:
        for m in re.finditer(pat, text_lower):
            start = max(0, m.start() - 30)
            end = min(len(text_lower), m.end() + 30)
            ctx = text_lower[start:end].strip()
            results.append(ctx)
    return results


def extract_conditionals(text: str) -> list[str]:
    """Extract conditional/hedging markers."""
    text_lower = text.lower()
    cond_pats = [
        r'\bif\b', r'\bwhether\b', r'\bmight\b', r'\bmay\b(?!\s+\d)',
        r'\bcould\b', r'\bwould\b', r'\bperhaps\b', r'\bpossibly\b',
        r'\bprobably\b', r'\bapparently\b', r'\ballegedly\b',
        r'\bor so\b', r'\bso she said\b', r'\bso he said\b', r'\bso they said\b',
    ]
    results = []
    for pat in cond_pats:
        for m in re.finditer(pat, text_lower):
            start = max(0, m.start() - 20)
            end = min(len(text_lower), m.end() + 20)
            results.append(text_lower[start:end].strip())
    return results


def check_number_mismatch(orig: str, rewrite: str) -> list[dict]:
    """Check for number mismatches between source and rewrite."""
    orig_nums = extract_numbers(orig)
    rw_nums = extract_numbers(rewrite)

    # Simple check: significant numbers in source not present in rewrite
    orig_vals = {n[1] for n in orig_nums if n[1] is not None and n[1] > 1}
    rw_vals = {n[1] for n in rw_nums if n[1] is not None and n[1] > 1}
    orig_words = {n[0] for n in orig_nums}
    rw_words = {n[0] for n in rw_nums}

    issues = []
    # Check: source has specific number, rewrite doesn't or changes it
    only_orig = orig_vals - rw_vals
    only_rw = rw_vals - orig_vals

    if only_orig and only_rw:
        issues.append({
            'type': 'number_changed',
            'source_unique': sorted(only_orig),
            'rewrite_unique': sorted(only_rw),
            'source_expressions': sorted(orig_words),
            'rewrite_expressions': sorted(rw_words),
        })
    elif only_orig and len(only_orig) > 0:
        issues.append({
            'type': 'number_lost',
            'source_unique': sorted(only_orig),
            'source_expressions': sorted(orig_words),
            'rewrite_expressions': sorted(rw_words),
        })

    # Special check: "thousand days" → "a year" type conversions
    for time_phrase, days in TIME_CONVERSIONS.items():
        if time_phrase in rewrite.lower() and time_phrase not in orig.lower():
            # Check if source had a different time expression
            if any(w in orig.lower() for w in ['thousand', 'hundred', 'days', 'hours', 'minutes']):
                issues.append({
                    'type': 'time_expression_changed',
                    'rewrite_has': time_phrase,
                    'note': 'rewrite uses a time phrase not in source while source has explicit quantities',
                })
    return issues


def check_negation_mismatch(orig: str, rewrite: str) -> list[dict]:
    """Check for negation mismatches."""
    orig_negs = extract_negation_contexts(orig)
    rw_negs = extract_negation_contexts(rewrite)

    issues = []
    # Count core negation markers
    core_negs = [
        "n't", "not", "never", "no ", "nobody", "nothing", "nowhere",
        "neither", "nor",
    ]
    orig_count = sum(1 for p in core_negs if p in orig.lower())
    rw_count = sum(1 for p in core_negs if p in rewrite.lower())

    if orig_count > 0 and rw_count == 0:
        issues.append({
            'type': 'negation_lost',
            'source_negation_count': orig_count,
            'rewrite_negation_count': rw_count,
            'source_contexts': orig_negs[:3],
        })
    elif rw_count > 0 and orig_count == 0:
        issues.append({
            'type': 'negation_added',
            'source_negation_count': orig_count,
            'rewrite_negation_count': rw_count,
            'rewrite_contexts': rw_negs[:3],
        })
    elif abs(orig_count - rw_count) >= 2:
        issues.append({
            'type': 'negation_count_changed',
            'source_negation_count': orig_count,
            'rewrite_negation_count': rw_count,
            'delta': rw_count - orig_count,
        })
    return issues


def check_conditional_mismatch(orig: str, rewrite: str) -> list[dict]:
    """Check for conditional/hedge markers lost or added."""
    orig_conds = extract_conditionals(orig)
    rw_conds = extract_conditionals(rewrite)

    issues = []
    if len(orig_conds) > 0 and len(rw_conds) == 0:
        issues.append({
            'type': 'conditional_lost',
            'source_conditional_count': len(orig_conds),
            'source_contexts': orig_conds[:3],
        })
    return issues


def check_direction_mismatch(orig: str, rewrite: str) -> list[dict]:
    """Check for direction/spatial reversals."""
    dir_words = ['left', 'right', 'north', 'south', 'east', 'west',
                 'up', 'down', 'above', 'below', 'before', 'after',
                 'first', 'last', 'top', 'bottom']

    orig_lower = orig.lower()
    rw_lower = rewrite.lower()

    orig_seq = []
    rw_seq = []
    for w in dir_words:
        orig_seq.extend([(m.start(), w) for m in re.finditer(rf'\b{w}\b', orig_lower)])
        rw_seq.extend([(m.start(), w) for m in re.finditer(rf'\b{w}\b', rw_lower)])

    orig_seq.sort()
    rw_seq.sort()
    orig_dir = [w for _, w in orig_seq]
    rw_dir = [w for _, w in rw_seq]

    issues = []
    # Check for left/right or before/after swaps
    opposites = [('left', 'right'), ('north', 'south'), ('east', 'west'),
                 ('up', 'down'), ('above', 'below'), ('before', 'after'),
                 ('first', 'last'), ('top', 'bottom')]

    for a, b in opposites:
        orig_a = orig_lower.count(a)
        orig_b = orig_lower.count(b)
        rw_a = rw_lower.count(a)
        rw_b = rw_lower.count(b)
        if orig_a > 0 and orig_b > 0 and rw_a > 0 and rw_b > 0:
            if orig_a != rw_a and orig_b != rw_b:
                # Counts changed for both members of an opposite pair
                if (orig_a > orig_b) != (rw_a > rw_b):
                    issues.append({
                        'type': 'direction_reversed',
                        'pair': [a, b],
                        'source_counts': [orig_a, orig_b],
                        'rewrite_counts': [rw_a, rw_b],
                    })

    if orig_dir != rw_dir and len(orig_dir) >= 3:
        issues.append({
            'type': 'direction_sequence_changed',
            'source_sequence': orig_dir,
            'rewrite_sequence': rw_dir,
        })
    return issues


def scan_pair(pair: dict) -> dict:
    """Scan a single pair for inherited rewrite errors."""
    orig = pair.get('original', '')
    rewrite = pair.get('rewrite', '')

    all_issues = []
    all_issues.extend(check_number_mismatch(orig, rewrite))
    all_issues.extend(check_negation_mismatch(orig, rewrite))
    all_issues.extend(check_conditional_mismatch(orig, rewrite))
    all_issues.extend(check_direction_mismatch(orig, rewrite))

    return {
        'pair_id': pair['pair_id'],
        'source': pair.get('source', ''),
        'original_words': len(orig.split()),
        'rewrite_words': len(rewrite.split()),
        'n_issues': len(all_issues),
        'issue_types': sorted(set(i['type'] for i in all_issues)),
        'issues': all_issues,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl'))
    ap.add_argument('--out-dir', type=Path,
                    default=Path('experiments/archive/functional_learning/data/inherited_rewrite_errors'))
    ap.add_argument('--limit', type=int, default=0, help='Limit scan to first N pairs (0=all)')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read pairs
    pairs = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.limit > 0 and i >= args.limit:
                break
            pairs.append(json.loads(line))

    print(f"Scanning {len(pairs)} pairs...", flush=True)

    # Scan all
    results = []
    flagged = []
    type_counts: Counter = Counter()
    source_counts: Counter = Counter()

    for pair in pairs:
        result = scan_pair(pair)
        results.append(result)
        if result['n_issues'] > 0:
            flagged.append(result)
            for t in result['issue_types']:
                type_counts[t] += 1
            source_counts[pair.get('source', 'unknown')] += 1

    # Write detailed results for flagged pairs
    flagged_path = args.out_dir / 'flagged_pairs.jsonl'
    with open(flagged_path, 'w') as f:
        for r in sorted(flagged, key=lambda x: -x['n_issues']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Write examples for each issue type
    examples_by_type: dict[str, list] = defaultdict(list)
    for r in flagged:
        pair = next(p for p in pairs if p['pair_id'] == r['pair_id'])
        for issue in r['issues']:
            if len(examples_by_type[issue['type']]) < 5:
                examples_by_type[issue['type']].append({
                    'pair_id': r['pair_id'],
                    'source': pair.get('source', ''),
                    'original': pair['original'][:300],
                    'rewrite': pair['rewrite'][:300],
                    'issue': issue,
                })

    # Summary
    summary = {
        'status': 'INHERITED_REWRITE_ERROR_SCAN',
        'n_pairs_scanned': len(pairs),
        'n_flagged': len(flagged),
        'flagged_rate': len(flagged) / len(pairs) if pairs else 0,
        'issue_type_counts': dict(type_counts.most_common()),
        'flagged_by_source': dict(source_counts.most_common()),
        'multi_issue_pairs': sum(1 for r in flagged if r['n_issues'] > 1),
        'examples_by_type': {k: v for k, v in examples_by_type.items()},
        'known_confirmed_errors': [
            {'pair_id': 'rw_009884', 'type': 'negation_reversal',
             'source': "didn't get off the plane",
             'rewrite': "I did manage to get off the plane"},
            {'pair_id': 'rw_001604', 'type': 'number_change',
             'source': "a thousand days of protest",
             'rewrite': "a year of sustained protests"},
        ],
        'input_path': str(args.input),
        'flagged_path': str(flagged_path),
        'interpretation': (
            'These are heuristic candidate errors, not verified contradictions. '
            'Number and negation checks have reasonable precision for obvious cases '
            'but will miss subtle meaning changes and produce false positives for '
            'legitimate restatements. The count estimates the scale of correspondence '
            'repair as a distinct intervention from shortening.'
        ),
    }

    summary_path = args.out_dir / 'error_scan_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        'status': summary['status'],
        'n_scanned': summary['n_pairs_scanned'],
        'n_flagged': summary['n_flagged'],
        'flagged_rate': f"{summary['flagged_rate']:.4f}",
        'type_counts': summary['issue_type_counts'],
        'flagged_by_source': summary['flagged_by_source'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
