#!/usr/bin/env python3
"""research: regression-test research SuperGLUE primary-metric patch.

Uses the known old-tokenizer compact_view_reinvest SuperGLUE results and
verifies that the research harness patch reproduces the current official-coordinate
SuperGLUE scalar (F1 for MRPC/QQP, accuracy otherwise) rather than the inherited
accuracy-only mean.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path('.').resolve()
SCRIPTS = USER_ROOT / 'experiments/archive/frontier_consolidation/scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_compliant_endpoint as harness  # noqa: E402

OUT = USER_ROOT / 'experiments/archive/frontier_consolidation/data/superglue_metric_patch_regression'
A01_FULL = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval'
A01_SUMMARY = USER_ROOT / 'experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json'
TARGET = 'compact_view_reinvest'


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_results(path: pathlib.Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        k, sep, v = line.partition(':')
        if sep:
            out[k.strip()] = float(v.strip()) * 100.0
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(A01_SUMMARY.read_text(encoding='utf-8'))
    expected = float(summary['score_summary']['official_overall']['scores']['SuperGLUE'])

    direct_details: list[dict[str, Any]] = []
    primary_vals = []
    accuracy_vals = []
    for task, metric in harness.SUPERGLUE_PRIMARY_METRIC.items():
        root = A01_FULL / 'superglue_results' / TARGET / task
        res = latest_file(root, 'results.txt')
        if res is None:
            raise FileNotFoundError(root)
        vals = parse_results(res)
        primary_vals.append(vals[metric])
        accuracy_vals.append(vals.get('accuracy'))
        direct_details.append({'task': task, 'metric': metric, 'primary_score': vals[metric], 'accuracy_score': vals.get('accuracy'), 'results_txt': str(res)})
    direct_primary_mean = sum(primary_vals) / len(primary_vals)
    direct_accuracy_mean = sum(float(x) for x in accuracy_vals if x is not None) / len(accuracy_vals)

    # Exercise the actual harness patch function in an isolated output directory.
    harness.base.OUT_ROOT = A01_FULL
    harness.base.PER_TARGET_DIR = OUT / 'per_target_payloads'
    payload: dict[str, Any] = {'target': TARGET, 'tasks': {'SuperGLUE': {'column': 'SuperGLUE', 'tasks': [], 'superglue_mean': direct_accuracy_mean}}}
    harness.patch_superglue_primary_metric(TARGET, payload)
    patched = payload['tasks']['SuperGLUE']['superglue_mean']

    result = {
        'status': 'SUPERGLUE_METRIC_PATCH_REGRESSION',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'purpose': 'Verify research compliant evaluation harness uses current official SuperGLUE primary metrics before any new expensive SuperGLUE run.',
        'target_reference': TARGET,
        'a01_summary': str(A01_SUMMARY),
        'expected_from_a01_pristine_collate': expected,
        'direct_primary_mean': direct_primary_mean,
        'direct_accuracy_only_mean': direct_accuracy_mean,
        'patched_harness_mean': patched,
        'abs_diff_patched_vs_expected': abs(patched - expected),
        'abs_diff_direct_vs_expected': abs(direct_primary_mean - expected),
        'legacy_accuracy_minus_primary': direct_accuracy_mean - direct_primary_mean,
        'details': direct_details,
        'patched_payload': str(harness.base.PER_TARGET_DIR / f'{TARGET}.json'),
        'pass': abs(patched - expected) < 1e-9 and abs(direct_primary_mean - expected) < 1e-9,
    }
    out_json = OUT / 'superglue_metric_patch_regression.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/superglue_metric_patch_regression/superglue_metric_patch_regression.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = []
    md.append('# research — SuperGLUE primary-metric patch regression\n')
    md.append(f"Reference target: `{TARGET}` using A01 old-tokenizer seed43022 results.\n\n")
    md.append(f"- Expected current-coordinate SuperGLUE from pristine collate: {expected:.12f}.\n")
    md.append(f"- Direct primary-metric mean: {direct_primary_mean:.12f}.\n")
    md.append(f"- Harness patched mean: {patched:.12f}.\n")
    md.append(f"- Legacy accuracy-only mean: {direct_accuracy_mean:.12f}.\n")
    md.append(f"- Accuracy-only minus primary: {direct_accuracy_mean - direct_primary_mean:.12f}.\n")
    md.append(f"- Regression pass: {result['pass']}.\n")
    md.append('\nThis validates that the compliant-endpoint evaluation harness no longer summarizes SuperGLUE in the inherited accuracy-only coordinate. Final pristine collation still remains authoritative.\n')
    out_md.write_text(''.join(md), encoding='utf-8')
    print(json.dumps({
        'status': result['status'],
        'out_json': str(out_json),
        'out_md': str(out_md),
        'expected': expected,
        'patched': patched,
        'legacy_accuracy_only': direct_accuracy_mean,
        'pass': result['pass'],
    }, indent=2), flush=True)
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
