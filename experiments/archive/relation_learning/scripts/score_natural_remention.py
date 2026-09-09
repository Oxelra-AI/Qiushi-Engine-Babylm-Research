#!/usr/bin/env python3
"""research: Score natural re-mention probe on VIEW/REPEAT/CLEAN checkpoints.

Each scoring record has two conditions:
  - antecedent present  (full_text_original)
  - antecedent replaced (corruption_plan.full_text_antecedent_replaced)

For each, we mask the re-mention span, score masked tokens, and compute
token-mean NLL. antecedent_gain = NLL(replaced) - NLL(present).

Pre-stated prediction:
  REPEAT > CLEAN on verbatim_remention gain
  VIEW > CLEAN on nonidentical_remention gain

Confirmatory interaction:
  (REPEAT-CLEAN)_verbatim > (REPEAT-CLEAN)_nonidentical
  (VIEW-CLEAN)_nonidentical > (VIEW-CLEAN)_verbatim
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, csv, json, math, os, pathlib, re, statistics, sys, time
from collections import defaultdict
from typing import Any

import torch
import numpy as np

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_natural_remention.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
PROBE_FILE = _public_path('experiments/archive/relation_learning/analysis/natural_remention_probe/natural_remention_probe.jsonl')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/natural_remention_scores')

# Arms to score
ARM_CONFIGS = {
    # DeBERTa dose-2.64x seed43022
    'D_V_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    'D_R_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    'D_C_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    # DeBERTa dose-2.64x seed43122
    'D_V_43122': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    'D_R_43122': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    'D_C_43122': _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    # RoBERTa seed43022
    'R_V_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/roberta_view_dose2p64x_matched_rowholdout_100M_seed43022'),
    'R_R_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022'),
    'R_C_43022': _public_path('experiments/archive/frontier_consolidation/training/runs/roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022'),
}

LATE_CKS = ['chck_80M', 'chck_90M', 'chck_100M']

def rel(p):
    try: return str(p.relative_to(ROOT))
    except: return str(p)

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def mean(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float('nan')


def load_probe():
    records = []
    with open(PROBE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model_and_tokenizer(model_path, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path))
    model.eval().to(device)
    return model, tokenizer


def char_to_token_span(tokenizer, text, char_start, char_end, max_seq_len=256):
    """Find token indices for a character span in text."""
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_seq_len, return_tensors='pt')
    offsets = enc['offset_mapping'][0].tolist()  # list of (start, end)
    token_ids = []
    for tid, (os, oe) in enumerate(offsets):
        if os == 0 and oe == 0:
            continue  # special token
        # Token overlaps with [char_start, char_end)?
        if oe > char_start and os < char_end:
            token_ids.append(tid)
    return enc, token_ids


@torch.no_grad()
def score_one_condition(model, tokenizer, text, char_start, char_end, device, max_seq_len=256):
    """Mask re-mention tokens, compute token-mean NLL."""
    enc, target_tids = char_to_token_span(tokenizer, text, char_start, char_end, max_seq_len)
    if not target_tids:
        return {'nll': float('nan'), 'n_tokens': 0, 'truncated': True}

    input_ids = enc['input_ids'].clone().to(device)
    attention_mask = enc['attention_mask'].to(device)
    labels = torch.full_like(input_ids, -100)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')

    for tid in target_tids:
        labels[0, tid] = input_ids[0, tid]
        input_ids[0, tid] = mask_token_id

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss.item()  # mean over masked tokens

    # Check if target was truncated
    max_tid = max(target_tids)
    truncated = max_tid >= enc['input_ids'].shape[1] - 1

    return {'nll': loss, 'n_tokens': len(target_tids), 'truncated': truncated}


def score_arm(arm_name, run_dir, records, checkpoints, device):
    """Score all records for one arm across checkpoints."""
    results = []
    for ck in checkpoints:
        model_path = run_dir / 'hf_model' / ck
        if not model_path.exists():
            print(f"[SKIP] {arm_name} {ck}: model not found at {model_path}", flush=True)
            continue

        print(f"[LOAD] {arm_name} {ck}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_path, device)

        for rec in records:
            pid = rec['probe_id']
            cls = rec['class_label']
            subtype = rec.get('subtype', '')
            pair_id = rec.get('class_matching', {}).get('pair_id', '')
            dist_bin = rec.get('mention_distance', {}).get('distance_bin', '')
            interv = rec.get('mention_distance', {}).get('intervening_words', -1)

            # Condition 1: antecedent present
            text_present = rec['full_text_original']
            cs_p = rec['remention']['char_start']
            ce_p = rec['remention']['char_end']
            res_present = score_one_condition(model, tokenizer, text_present, cs_p, ce_p, device)

            # Condition 2: antecedent replaced
            cp = rec['corruption_plan']
            text_replaced = cp['full_text_antecedent_replaced']
            cs_r = cp.get('remention_char_start_replaced', cs_p)
            ce_r = cp.get('remention_char_end_replaced', ce_p)
            res_replaced = score_one_condition(model, tokenizer, text_replaced, cs_r, ce_r, device)

            gain = res_replaced['nll'] - res_present['nll']

            results.append({
                'arm': arm_name, 'checkpoint': ck,
                'probe_id': pid, 'class_label': cls, 'subtype': subtype,
                'pair_id': pair_id, 'distance_bin': dist_bin,
                'intervening_words': interv,
                'nll_present': res_present['nll'],
                'nll_replaced': res_replaced['nll'],
                'antecedent_gain': gain,
                'n_tokens_present': res_present['n_tokens'],
                'n_tokens_replaced': res_replaced['n_tokens'],
                'truncated_present': res_present['truncated'],
                'truncated_replaced': res_replaced['truncated'],
            })

        del model
        torch.cuda.empty_cache()
        print(f"[DONE] {arm_name} {ck}: scored {len(records)} records", flush=True)

    return results


def summarize(all_results):
    """Aggregate by arm, checkpoint, class."""
    d = defaultdict(list)
    for r in all_results:
        for g in [r['class_label'], 'ALL']:
            d[(r['arm'], r['checkpoint'], g)].append(r)
    out = []
    for (arm, ck, g), vals in sorted(d.items()):
        gains = [v['antecedent_gain'] for v in vals if math.isfinite(v['antecedent_gain'])]
        out.append({
            'arm': arm, 'checkpoint': ck, 'group': g,
            'n': len(gains),
            'mean_gain': mean(gains),
            'sd_gain': statistics.stdev(gains) if len(gains) > 1 else float('nan'),
            'se_gain': (statistics.stdev(gains) / math.sqrt(len(gains))) if len(gains) > 1 else float('nan'),
            'mean_nll_present': mean([v['nll_present'] for v in vals]),
            'mean_nll_replaced': mean([v['nll_replaced'] for v in vals]),
        })
    return out


def compute_late_contrasts(summary):
    """Compute late-mean contrasts across arms per group."""
    # First compute late means
    late = defaultdict(list)
    for r in summary:
        if r['checkpoint'] in LATE_CKS:
            late[(r['arm'], r['group'])].append(r['mean_gain'])

    late_mean = {k: mean(v) for k, v in late.items()}

    # Then compute contrasts
    arch_groups = defaultdict(set)
    for (arm, g) in late_mean:
        parts = arm.split('_')
        arch = parts[0]  # D or R
        seed = parts[2]
        arch_groups[(arch, seed)].add(g)

    contrasts = []
    for (arch, seed), groups in sorted(arch_groups.items()):
        for g in sorted(groups):
            v_key = (f'{arch}_V_{seed}', g)
            r_key = (f'{arch}_R_{seed}', g)
            c_key = (f'{arch}_C_{seed}', g)
            v_gain = late_mean.get(v_key, float('nan'))
            r_gain = late_mean.get(r_key, float('nan'))
            c_gain = late_mean.get(c_key, float('nan'))
            contrasts.append({
                'arch': arch, 'seed': seed, 'group': g,
                'V_gain': round(v_gain, 4) if math.isfinite(v_gain) else None,
                'R_gain': round(r_gain, 4) if math.isfinite(r_gain) else None,
                'C_gain': round(c_gain, 4) if math.isfinite(c_gain) else None,
                'VminusC': round(v_gain - c_gain, 4) if all(math.isfinite(x) for x in [v_gain, c_gain]) else None,
                'RminusC': round(r_gain - c_gain, 4) if all(math.isfinite(x) for x in [r_gain, c_gain]) else None,
                'VminusR': round(v_gain - r_gain, 4) if all(math.isfinite(x) for x in [v_gain, r_gain]) else None,
            })
    return contrasts


def write_csv_rows(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('\n', encoding='utf-8'); return
    fields = list(rows[0].keys())
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_note(contrasts, out_dir):
    note_path = out_dir / 'natural_remention_probe_result.md'
    lines = ['# research natural re-mention probe result', '']
    lines.append('Scored 258 held-out natural re-mention records (129 verbatim + 129 nonidentical).')
    lines.append('antecedent_gain = NLL(replaced) - NLL(present). Positive = antecedent helps.')
    lines.append('')
    lines.append('## Late-mean contrasts')
    lines.append('')
    lines.append('| arch | seed | group | V gain | R gain | C gain | V−C | R−C | V−R |')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for c in contrasts:
        def f(x): return f'{x:+.4f}' if x is not None else 'NA'
        lines.append(f"| {c['arch']} | {c['seed']} | {c['group']} | {f(c['V_gain'])} | {f(c['R_gain'])} | {f(c['C_gain'])} | {f(c['VminusC'])} | {f(c['RminusC'])} | {f(c['VminusR'])} |")
    lines.append('')
    lines.append('## Prediction check')
    lines.append('')
    lines.append('Pre-stated: REPEAT > CLEAN on verbatim gain; VIEW > CLEAN on nonidentical gain.')
    lines.append('Confirmatory: (R-C)_verbatim > (R-C)_nonidentical AND (V-C)_nonidentical > (V-C)_verbatim.')
    lines.append('')

    # Check predictions
    for c in contrasts:
        if c['group'] in ['verbatim_remention', 'nonidentical_remention']:
            lines.append(f"  {c['arch']}_{c['seed']} {c['group']}: V−C={c['VminusC']}, R−C={c['RminusC']}")
    lines.append('')

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return note_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=list(ARM_CONFIGS.keys()))
    ap.add_argument('--checkpoints', nargs='+', default=LATE_CKS)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading probe from {rel(PROBE_FILE)}", flush=True)
    records = load_probe()
    print(f"Loaded {len(records)} probe records", flush=True)

    all_results = []
    for arm_name in args.arms:
        run_dir = ARM_CONFIGS.get(arm_name)
        if run_dir is None:
            print(f"[SKIP] Unknown arm: {arm_name}", flush=True)
            continue
        if not run_dir.exists():
            print(f"[SKIP] {arm_name}: run dir not found at {run_dir}", flush=True)
            continue
        results = score_arm(arm_name, run_dir, records, args.checkpoints, device)
        all_results.extend(results)

    if not all_results:
        print("ERROR: No results scored", flush=True)
        return

    write_csv_rows(out_dir / 'natural_remention_raw_scores.csv', all_results)
    summary = summarize(all_results)
    write_csv_rows(out_dir / 'natural_remention_summary.csv', summary)
    contrasts = compute_late_contrasts(summary)
    write_csv_rows(out_dir / 'natural_remention_late_contrasts.csv', contrasts)
    note_path = write_note(contrasts, out_dir)

    result = {
        'status': 'NATURAL_REMENTION_SCORE_DONE',
        'finished_utc': now(),
        'n_records': len(records),
        'n_arms_scored': len(set(r['arm'] for r in all_results)),
        'n_results': len(all_results),
        'contrasts': contrasts,
        'note': rel(note_path),
    }
    (out_dir / 'natural_remention_result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
