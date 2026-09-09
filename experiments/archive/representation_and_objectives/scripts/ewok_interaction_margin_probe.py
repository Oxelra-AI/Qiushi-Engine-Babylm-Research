#!/usr/bin/env python3
"""research EWoK context-target interaction margin probe.

This is a CPU/GPU-light diagnostic over existing checkpoints. It does not train,
evaluate the full benchmark. Its purpose is to decide
whether the research SCRA idea has a real target: weak/wrong-signed context-target
interaction on EWoK, rather than rare-token support or an official scorer artifact.

For each raw EWoK row, compute MLM pseudo-log-likelihood scores
    s_ij = S(Context_i, Target_j), i,j in {1,2}
using the same completion-token masking convention as official sentence_zero_shot
MLM scoring. Then report official row margins, within-context target margins,
interaction I=(s11+s22)-(s12+s21), and PMI-corrected analogues.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path('experiments/archive/representation_and_objectives')
WS = ROOT
EVAL_EWOK = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'
ITEM_CSV = WS / 'data/ewok_contrast_preservation_anatomy/ewok_item_correctness_and_coverage.csv'
OUT = WS / 'data/ewok_interaction_margin_probe'
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/ewok_interaction_margin_probe.md')

DEFAULT_MODELS = {
    'legal40_depth_12x384_43022': WS / 'training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M',
    'legal40_8x480_43022': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
}


def now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(EVAL_EWOK.glob('*.jsonl')):
        domain = path.stem
        with path.open('r', encoding='utf-8') as f:
            for local_index, line in enumerate(f):
                raw = json.loads(line)
                raw['_domain'] = domain
                raw['_local_index'] = local_index
                raw['_global_index'] = len(rows)
                rows.append(raw)
    return rows


def load_item_flags() -> dict[int, dict[str, Any]]:
    flags = {}
    with ITEM_CSV.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            idx = int(row['global_index'])
            flags[idx] = row
    return flags


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {'1','true','yes'}


def choose_subset(rows: list[dict[str, Any]], flags: dict[int, dict[str, Any]], per_domain: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row['_domain']].append(row['_global_index'])
    selected: list[int] = []
    selected_set: set[int] = set()

    def add_from(domain: str, candidates: list[int], n: int) -> None:
        rng.shuffle(candidates)
        candidates.sort(key=lambda idx: (0 if idx not in selected_set else 1, idx))
        added = 0
        for idx in candidates:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            added += 1
            if added >= n:
                break

    # Per domain: persistent legal40+depth errors, stable legal successes, depth losses/gains.
    for domain in sorted(by_domain):
        idxs = by_domain[domain]
        persistent = [i for i in idxs if truthy(flags[i].get('depth_wrong_legal40_wrong'))]
        stable_correct = [i for i in idxs if truthy(flags[i].get('legal40_8x480_43022')) and truthy(flags[i].get('legal40_12x384_depth_43022'))]
        depth_lost = [i for i in idxs if truthy(flags[i].get('legal40_8x480_43022')) and not truthy(flags[i].get('legal40_12x384_depth_43022'))]
        depth_gained = [i for i in idxs if (not truthy(flags[i].get('legal40_8x480_43022'))) and truthy(flags[i].get('legal40_12x384_depth_43022'))]
        n_persist = max(2, per_domain // 2)
        n_correct = max(1, per_domain // 6)
        n_loss = max(1, per_domain // 6)
        n_gain = max(1, per_domain - n_persist - n_correct - n_loss)
        add_from(domain, persistent, n_persist)
        add_from(domain, stable_correct, n_correct)
        add_from(domain, depth_lost, n_loss)
        add_from(domain, depth_gained, n_gain)
    return selected


def tokenizer_encode(tokenizer, text: str):
    return tokenizer(text, return_offsets_mapping=True, return_tensors=None)


def score_completion(model, tokenizer, context: str, target: str, device: torch.device, non_causal_batch_size: int) -> dict[str, Any]:
    # Official EWoK MLM scoring: sentence="Context Target", completion=" Target",
    # mask each token overlapping the completion span, sum target log-probs.
    sentence = (context.rstrip() + ' ' + target.strip()).strip()
    completion = ' ' + target.strip()
    start_char_idx = len(sentence) - len(completion)
    enc = tokenizer(sentence, return_offsets_mapping=True, return_tensors=None)
    tokens = list(enc['input_ids'])
    attn = list(enc['attention_mask'])
    offsets = list(enc['offset_mapping'])
    phrase_indices = []
    target_tokens = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_indices.append(i)
            target_tokens.append(tokens[i])
    if not phrase_indices:
        return {'sum': float('nan'), 'mean': float('nan'), 'n_tokens': 0, 'phrase_indices': []}
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError('Tokenizer has no mask_token_id')
    sums = []
    for batch_start in range(0, len(phrase_indices), non_causal_batch_size):
        batch_indices = phrase_indices[batch_start:batch_start + non_causal_batch_size]
        batch_targets = target_tokens[batch_start:batch_start + non_causal_batch_size]
        batch_tokens = []
        batch_attn = []
        for idx in batch_indices:
            cur = list(tokens)
            cur[idx] = mask_id
            batch_tokens.append(cur)
            batch_attn.append(attn)
        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch_attn, dtype=torch.long, device=device)
        target_t = torch.tensor(batch_targets, dtype=torch.long, device=device)
        idx_t = torch.tensor(batch_indices, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            minibatch = torch.arange(logits.shape[0], device=device)
            masked = logits[minibatch, idx_t]
            lp = F.log_softmax(masked, dim=-1)
            vals = torch.gather(lp, -1, target_t.unsqueeze(-1)).squeeze(-1)
            sums.extend(float(x) for x in vals.detach().cpu().tolist())
    s = float(sum(sums))
    return {'sum': s, 'mean': s / len(sums), 'n_tokens': len(sums), 'phrase_indices': phrase_indices}


def score_row(model, tokenizer, row: dict[str, Any], device: torch.device, non_causal_batch_size: int) -> dict[str, Any]:
    c1, c2, t1, t2 = row['Context1'], row['Context2'], row['Target1'], row['Target2']
    s11 = score_completion(model, tokenizer, c1, t1, device, non_causal_batch_size)
    s21 = score_completion(model, tokenizer, c2, t1, device, non_causal_batch_size)
    s12 = score_completion(model, tokenizer, c1, t2, device, non_causal_batch_size)
    s22 = score_completion(model, tokenizer, c2, t2, device, non_causal_batch_size)
    # Target-prior baseline under empty context; this is not official, only diagnostic.
    p1 = score_completion(model, tokenizer, '', t1, device, non_causal_batch_size)
    p2 = score_completion(model, tokenizer, '', t2, device, non_causal_batch_size)

    def val(x, key='sum') -> float:
        return float(x[key])
    out = {
        's11_sum': val(s11), 's21_sum': val(s21), 's12_sum': val(s12), 's22_sum': val(s22),
        's11_mean': val(s11, 'mean'), 's21_mean': val(s21, 'mean'), 's12_mean': val(s12, 'mean'), 's22_mean': val(s22, 'mean'),
        'p1_sum': val(p1), 'p2_sum': val(p2), 'p1_mean': val(p1, 'mean'), 'p2_mean': val(p2, 'mean'),
        't1_tokens': s11['n_tokens'], 't2_tokens': s12['n_tokens'],
    }
    # Official-like margins: Target1 should be better in Context1 than Context2;
    # Target2 should be better in Context2 than Context1.
    out['official_margin_t1_sum'] = out['s11_sum'] - out['s21_sum']
    out['official_margin_t2_sum'] = out['s22_sum'] - out['s12_sum']
    out['within_context_margin_c1_sum'] = out['s11_sum'] - out['s12_sum']
    out['within_context_margin_c2_sum'] = out['s22_sum'] - out['s21_sum']
    out['interaction_sum'] = (out['s11_sum'] + out['s22_sum']) - (out['s12_sum'] + out['s21_sum'])
    out['official_margin_t1_mean'] = out['s11_mean'] - out['s21_mean']
    out['official_margin_t2_mean'] = out['s22_mean'] - out['s12_mean']
    out['within_context_margin_c1_mean'] = out['s11_mean'] - out['s12_mean']
    out['within_context_margin_c2_mean'] = out['s22_mean'] - out['s21_mean']
    out['interaction_mean'] = (out['s11_mean'] + out['s22_mean']) - (out['s12_mean'] + out['s21_mean'])
    # PMI-corrected target scores.
    pmi11 = out['s11_sum'] - out['p1_sum']; pmi21 = out['s21_sum'] - out['p1_sum']
    pmi12 = out['s12_sum'] - out['p2_sum']; pmi22 = out['s22_sum'] - out['p2_sum']
    out['pmi_official_margin_t1_sum'] = pmi11 - pmi21
    out['pmi_official_margin_t2_sum'] = pmi22 - pmi12
    out['pmi_within_context_margin_c1_sum'] = pmi11 - pmi12
    out['pmi_within_context_margin_c2_sum'] = pmi22 - pmi21
    out['pmi_interaction_sum'] = (pmi11 + pmi22) - (pmi12 + pmi21)
    return out


def summarize(records: list[dict[str, Any]], label: str, mask_fn=lambda r: True) -> dict[str, Any]:
    rs = [r for r in records if mask_fn(r)]
    out = {'label': label, 'n': len(rs)}
    if not rs:
        return out
    for field in [
        'official_margin_t1_sum','official_margin_t2_sum','within_context_margin_c1_sum','within_context_margin_c2_sum','interaction_sum',
        'official_margin_t1_mean','official_margin_t2_mean','interaction_mean','pmi_interaction_sum'
    ]:
        vals = [float(r[field]) for r in rs if math.isfinite(float(r[field]))]
        out[field + '_mean'] = statistics.fmean(vals) if vals else None
        out[field + '_median'] = statistics.median(vals) if vals else None
        out[field + '_pos_frac'] = sum(v > 0 for v in vals) / len(vals) if vals else None
        out[field + '_near0_frac_abs_lt_0p25'] = sum(abs(v) < 0.25 for v in vals) / len(vals) if vals else None
    out['both_official_margins_positive_frac'] = sum((r['official_margin_t1_sum'] > 0 and r['official_margin_t2_sum'] > 0) for r in rs) / len(rs)
    out['both_within_context_margins_positive_frac'] = sum((r['within_context_margin_c1_sum'] > 0 and r['within_context_margin_c2_sum'] > 0) for r in rs) / len(rs)
    out['interaction_positive_but_official_t1_wrong_frac'] = sum((r['interaction_sum'] > 0 and r['official_margin_t1_sum'] <= 0) for r in rs) / len(rs)
    out['interaction_negative_frac'] = sum(r['interaction_sum'] <= 0 for r in rs) / len(rs)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_domain', type=int, default=12, help='stratified rows per EWoK domain')
    ap.add_argument('--seed', type=int, default=90090)
    ap.add_argument('--models', nargs='*', default=['legal40_depth_12x384_43022'], choices=list(DEFAULT_MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    ap.add_argument('--non_causal_batch_size', type=int, default=16)
    ap.add_argument('--max_rows', type=int, default=0, help='optional cap after stratified selection')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    flags = load_item_flags()
    selected = choose_subset(rows, flags, args.per_domain, args.seed)
    if args.max_rows and len(selected) > args.max_rows:
        selected = selected[:args.max_rows]
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    all_records: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    for model_label in args.models:
        model_path = DEFAULT_MODELS[model_label]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
        model.eval().to(device)
        records = []
        for k, idx in enumerate(selected):
            row = rows[idx]
            f = flags[idx]
            scores = score_row(model, tokenizer, row, device, args.non_causal_batch_size)
            rec = {
                'model': model_label,
                'global_index': idx,
                'domain': row['_domain'],
                'local_index': row['_local_index'],
                'ContextType': row.get('ContextType'),
                'ContextDiff': row.get('ContextDiff'),
                'TargetDiff': row.get('TargetDiff'),
                'ConceptA': row.get('ConceptA'),
                'ConceptB': row.get('ConceptB'),
                'legal40_depth_both_wrong_flag': truthy(f.get('depth_wrong_legal40_wrong')),
                'all_legal_wrong_flag': truthy(f.get('all_legal_wrong')),
                'legal40_correct_flag': truthy(f.get('legal40_8x480_43022')),
                'depth_correct_flag': truthy(f.get('legal40_12x384_depth_43022')),
                **scores,
            }
            # Sanity: official saved prediction for this row in this model should match t1 margin sign for matching models.
            if model_label == 'legal40_depth_12x384_43022':
                rec['saved_depth_correct_flag'] = truthy(f.get('legal40_12x384_depth_43022'))
                rec['official_t1_sign_matches_saved'] = (rec['official_margin_t1_sum'] > 0) == rec['saved_depth_correct_flag']
            if model_label == 'legal40_8x480_43022':
                rec['saved_legal40_correct_flag'] = truthy(f.get('legal40_8x480_43022'))
                rec['official_t1_sign_matches_saved'] = (rec['official_margin_t1_sum'] > 0) == rec['saved_legal40_correct_flag']
            records.append(rec)
            all_records.append(rec)
        model_summaries[model_label] = {
            'model_path': str(model_path),
            'n_rows': len(records),
            'device': str(device),
            'sanity_official_sign_match_frac': sum(r.get('official_t1_sign_matches_saved', False) for r in records) / len(records),
            'all_selected': summarize(records, 'all_selected'),
            'legal40_depth_both_wrong_subset': summarize(records, 'legal40_depth_both_wrong_subset', lambda r: r['legal40_depth_both_wrong_flag']),
            'all_legal_wrong_subset': summarize(records, 'all_legal_wrong_subset', lambda r: r['all_legal_wrong_flag']),
            'depth_saved_correct_subset': summarize(records, 'depth_saved_correct_subset', lambda r: r['depth_correct_flag']),
            'depth_saved_wrong_subset': summarize(records, 'depth_saved_wrong_subset', lambda r: not r['depth_correct_flag']),
            'by_domain': {dom: summarize(records, dom, lambda r, dom=dom: r['domain'] == dom) for dom in sorted({r['domain'] for r in records})},
            'by_context_type': {ct: summarize(records, ct, lambda r, ct=ct: r['ContextType'] == ct) for ct in sorted({str(r['ContextType']) for r in records})},
        }
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    write_csv(OUT / 'ewok_interaction_margin_records.csv', all_records)
    payload = {
        'status': 'EWOK_INTERACTION_MARGIN_PROBE',
        'created_utc': now_utc(),
        'purpose': 'Low-cost context-target interaction probe for EWoK/SCRA before any SCRA training commitment.',
        'script_never_trains_or_evaluates_full_benchmark': True,
        'device_requested': args.device,
        'device_used': str(device),
        'models': args.models,
        'selected_rows': selected,
        'per_domain': args.per_domain,
        'max_rows': args.max_rows,
        'model_summaries': model_summaries,
        'csv_records': str(OUT / 'ewok_interaction_margin_records.csv'),
        'json_output': str(OUT / 'ewok_interaction_margin_probe.json'),
        'note': str(NOTE),
    }
    (OUT / 'ewok_interaction_margin_probe.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = []
    lines.append('# research — EWoK interaction-margin probe\n\n')
    lines.append('This diagnostic recomputes EWoK four-score pseudo-log-likelihood matrices on a stratified subset of existing checkpoint(s). It does not train or run full official evaluation.\n\n')
    for model_label, summ in model_summaries.items():
        all_s = summ['all_selected']
        pers = summ['legal40_depth_both_wrong_subset']
        lines.append(f"## {model_label}\n\n")
        lines.append(f"Rows `{summ['n_rows']}`, device `{summ['device']}`, saved-official sign agreement `{summ['sanity_official_sign_match_frac']:.3f}`.\n\n")
        lines.append('| subset | n | official_t1 positive | official_t2 positive | interaction positive | interaction median | both within-context positive | near-zero interaction |\n')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for sub in [all_s, pers, summ['all_legal_wrong_subset'], summ['depth_saved_correct_subset'], summ['depth_saved_wrong_subset']]:
            lines.append(f"| {sub['label']} | {sub['n']} | {sub.get('official_margin_t1_sum_pos_frac')} | {sub.get('official_margin_t2_sum_pos_frac')} | {sub.get('interaction_sum_pos_frac')} | {sub.get('interaction_sum_median')} | {sub.get('both_within_context_margins_positive_frac')} | {sub.get('interaction_sum_near0_frac_abs_lt_0p25')} |\n")
        lines.append('\n')
    lines.append('Interpretation rule: persistent official errors with negative or near-zero interaction support a relational compatibility learning deficit; positive interaction but official failure points toward scorer/calibration/context-prior artifacts; large sensitivity to mean or PMI margins weakens the case for a training objective.\n\n')
    lines.append(f"JSON: `{payload['json_output']}`\n\nCSV: `{payload['csv_records']}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')

    compact = {
        'status': payload['status'],
        'device_used': str(device),
        'models': args.models,
        'n_rows': len(selected),
        'out_json': payload['json_output'],
        'note': payload['note'],
        'summary': {m: {
            'sign_match': s['sanity_official_sign_match_frac'],
            'all_interaction_pos_frac': s['all_selected'].get('interaction_sum_pos_frac'),
            'all_interaction_median': s['all_selected'].get('interaction_sum_median'),
            'persistent_interaction_pos_frac': s['legal40_depth_both_wrong_subset'].get('interaction_sum_pos_frac'),
            'persistent_interaction_median': s['legal40_depth_both_wrong_subset'].get('interaction_sum_median'),
            'persistent_both_within_context_pos_frac': s['legal40_depth_both_wrong_subset'].get('both_within_context_margins_positive_frac'),
        } for m, s in model_summaries.items()},
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
