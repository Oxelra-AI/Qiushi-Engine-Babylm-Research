#!/usr/bin/env python3
"""research: layerwise four-cell packet interaction readout on frozen checkpoints.

Scientific purpose
------------------
The active BabyLM route has converged on a context-conditioned alternative-binding
failure.  Before spending another endpoint, this script asks a cheaper question on
existing checkpoints: is the packet four-cell interaction

    M = [s(C_AB, correct_AB)-s(C_AB, wrong_AB)]
      + [s(C_BA, correct_BA)-s(C_BA, wrong_BA)]

already present in intermediate DeBERTa layers when decoded through the trained MLM
head, but lost or unread at the final logits?  Or is it weak throughout the frozen
network?  This is not an official-evaluation score and trains no model; it is a
mechanistic readout to separate "encoded-but-unread" from "not formed by this
trajectory" before any new H100 training route.

The scoring intentionally mirrors frozen_exchange_scoring.py: for each
pair, context (AB/BA/erased), and alternative, the candidate word is filled into
the consequence, its last occurrence is masked, and log P(candidate tokens) is
summed.  The only difference is that logits are decoded from every hidden-state
layer via the same trained MLM head.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
PAIRS_PATH = A01_WS / 'data/role_switch_expanded_packets/scoring_pairs.jsonl'
OUT_ROOT = A01_WS / 'data/layerwise_packet_M_decomposition'

TARGETS: dict[str, dict[str, Any]] = {
    'anchor_20M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_20M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_20M',
        'description': 'legal40k fixed-256 compact-view anchor, 20M',
    },
    'anchor_50M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_50M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_50M',
        'description': 'legal40k fixed-256 compact-view anchor, 50M',
    },
    'anchor_80M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'description': 'legal40k fixed-256 compact-view anchor, 80M',
    },
    'anchor_100M': {
        'model_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'description': 'legal40k fixed-256 compact-view anchor, 100M',
    },
    'reheat_100M': {
        'model_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'reheat-cosine 100M broad-learning reference',
    },
    'residual_tail_100M': {
        'model_root': A01_WS / 'training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': A01_WS / 'training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'residual-low-LR fresh-moment tail 100M reference',
    },
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(path: pathlib.Path = PAIRS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_stratified_pairs(pairs: list[dict[str, Any]], tokenizer: Any, *, max_per_group: int, seed: int,
                            one_token_only: bool, include_temporal: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    rejected = collections.Counter()
    token_len_hist = collections.Counter()
    for p in pairs:
        if (not include_temporal) and p.get('family') == 'temporal':
            rejected['temporal_excluded'] += 1
            continue
        ids0 = tokenizer.encode(' ' + p['alt_0'], add_special_tokens=False)
        ids1 = tokenizer.encode(' ' + p['alt_1'], add_special_tokens=False)
        token_len_hist[(len(ids0), len(ids1))] += 1
        if one_token_only and not (len(ids0) == 1 and len(ids1) == 1):
            rejected['not_one_token_both_alts'] += 1
            continue
        groups[(p.get('family', ''), p.get('style', ''))].append(p)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    group_sizes: dict[str, dict[str, int]] = {}
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: int(x.get('pair_id', 0)))
        rng.shuffle(rows)
        take = rows[:max_per_group] if max_per_group > 0 else rows
        take = sorted(take, key=lambda x: int(x.get('pair_id', 0)))
        selected.extend(take)
        group_sizes[f'{key[0]}::{key[1]}'] = {'available': len(rows), 'selected': len(take)}
    selected = sorted(selected, key=lambda x: int(x.get('pair_id', 0)))
    meta = {
        'n_input_pairs': len(pairs),
        'n_selected_pairs': len(selected),
        'max_per_group': max_per_group,
        'seed': seed,
        'one_token_only': one_token_only,
        'include_temporal': include_temporal,
        'group_sizes': group_sizes,
        'rejected': dict(rejected),
        'alt_leading_space_token_len_hist': {f'{k[0]},{k[1]}': v for k, v in sorted(token_len_hist.items())},
    }
    return selected, meta


def find_last_subseq(full_ids: list[int], sub_ids: list[int]) -> int | None:
    n = len(sub_ids)
    for i in range(len(full_ids) - n, -1, -1):
        if full_ids[i:i+n] == sub_ids:
            return i
    return None


def render_text(pair: dict[str, Any], ctx_key: str, alt_word: str) -> str:
    tmpl = pair['consequence_masked']
    # Most research scoring consequences only contain __TARGET__.  If __OTHER__ is
    # present in a future template, fill it with the other alternative so the text
    # remains well-formed; this preserves the research training semantics.
    other = pair['alt_1'] if alt_word == pair['alt_0'] else pair['alt_0']
    con = tmpl.replace('__TARGET__', alt_word).replace('__OTHER__', other)
    if ctx_key == 'erased':
        return con
    return pair[f'context_{ctx_key}'] + ' ' + con


def make_cases(pairs: list[dict[str, Any]], tokenizer: Any, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for local_idx, p in enumerate(pairs):
        for ctx_key in ['AB', 'BA', 'erased']:
            for alt_key, alt_word in [('alt0', p['alt_0']), ('alt1', p['alt_1'])]:
                text = render_text(p, ctx_key, alt_word)
                enc = tokenizer(text, return_attention_mask=True, truncation=True, max_length=max_len, add_special_tokens=True)
                input_ids = list(enc['input_ids'])
                target_ids = tokenizer.encode(' ' + alt_word, add_special_tokens=False)
                start = find_last_subseq(input_ids, target_ids)
                token_form = 'leading_space'
                if start is None:
                    target_ids = tokenizer.encode(alt_word, add_special_tokens=False)
                    start = find_last_subseq(input_ids, target_ids)
                    token_form = 'bare'
                if start is None or not target_ids:
                    failures.append({'pair_id': p.get('pair_id'), 'ctx_key': ctx_key, 'alt_key': alt_key,
                                     'alt_word': alt_word, 'reason': 'target_subseq_not_found'})
                    continue
                masked = list(input_ids)
                mask_id = tokenizer.mask_token_id
                if mask_id is None:
                    raise RuntimeError('tokenizer has no mask_token_id')
                for j in range(len(target_ids)):
                    masked[start + j] = int(mask_id)
                cases.append({
                    'case_id': len(cases),
                    'local_pair_index': local_idx,
                    'pair_id': p.get('pair_id'),
                    'family': p.get('family'),
                    'style': p.get('style'),
                    'entity_split': p.get('entity_split'),
                    'alt_type': p.get('alt_type'),
                    'ctx_key': ctx_key,
                    'alt_key': alt_key,
                    'alt_word': alt_word,
                    'target_ids': [int(x) for x in target_ids],
                    'target_positions': list(range(start, start + len(target_ids))),
                    'input_ids': [int(x) for x in masked],
                    'attention_mask': [int(x) for x in enc['attention_mask']],
                    'seq_len': len(masked),
                    'target_token_len': len(target_ids),
                    'token_form': token_form,
                })
    # Complete-pair check: all 6 cases must exist.
    complete = collections.Counter(c['local_pair_index'] for c in cases)
    complete_pair_indices = {k for k, v in complete.items() if v == 6}
    meta = {
        'n_cases': len(cases),
        'n_failures': len(failures),
        'failures_head': failures[:20],
        'n_complete_pairs': len(complete_pair_indices),
        'n_incomplete_pairs': len(pairs) - len(complete_pair_indices),
        'case_seq_len': describe([c['seq_len'] for c in cases]),
        'target_token_len_hist': dict(sorted(collections.Counter(c['target_token_len'] for c in cases).items())),
    }
    return cases, meta


def describe(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {'n': 0, 'min': None, 'mean': None, 'median': None, 'max': None}
    xs = sorted(float(v) for v in vals)
    n = len(xs)
    return {'n': n, 'min': xs[0], 'mean': sum(xs) / n, 'median': xs[n // 2], 'max': xs[-1]}


def batch_iter(xs: list[Any], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield i, xs[i:i + batch_size]


def pad_batch(cases: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c['input_ids']) for c in cases)
    ids = []
    attn = []
    for c in cases:
        pad_n = max_len - len(c['input_ids'])
        ids.append(c['input_ids'] + [pad_id] * pad_n)
        attn.append(c['attention_mask'] + [0] * pad_n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def score_cases_layerwise(model: Any, tokenizer: Any, cases: list[dict[str, Any]], *, device: str,
                          batch_size: int, max_layers: int | None) -> tuple[list[int], dict[int, dict[int, float]]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    scores_by_case: dict[int, dict[int, float]] = {int(c['case_id']): {} for c in cases}
    layer_indices: list[int] | None = None
    with torch.no_grad():
        for start, batch in batch_iter(cases, batch_size):
            input_ids_cpu, attention_cpu = pad_batch(batch, int(pad_id))
            input_ids = input_ids_cpu.to(device)
            attention_mask = attention_cpu.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            hidden_states = list(out.hidden_states)
            if max_layers is not None:
                hidden_states = hidden_states[:max_layers]
            if layer_indices is None:
                layer_indices = list(range(len(hidden_states)))
            for li, hidden in enumerate(hidden_states):
                # Decode every layer through the trained MLM head.  Cast to fp32
                # before log-softmax for stable layer comparisons under bf16/fp16.
                logits = model.cls(hidden)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                for bi, c in enumerate(batch):
                    s = 0.0
                    for pos, tid in zip(c['target_positions'], c['target_ids']):
                        s += float(log_probs[bi, int(pos), int(tid)].item())
                    scores_by_case[int(c['case_id'])][li] = s
                del logits, log_probs
            del out, hidden_states, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({'event': 'batch_done', 'start': start, 'n': len(batch)}), flush=True)
    assert layer_indices is not None
    return layer_indices, scores_by_case


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float('nan')


def summarize_layer_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'n': len(rows),
        'M_mean': safe_mean([r['four_cell_M'] for r in rows]),
        'M_pos_frac': safe_mean([1.0 if r['four_cell_M'] > 0 else 0.0 for r in rows]),
        'CM_mean': safe_mean([r['context_conditioned_M'] for r in rows]),
        'bias_mean': safe_mean([r['erased_bias'] for r in rows]),
        'bias_abs_mean': safe_mean([abs(r['erased_bias']) for r in rows]),
        'margin_AB_mean': safe_mean([r['margin_AB'] for r in rows]),
        'margin_BA_mean': safe_mean([r['margin_BA'] for r in rows]),
        'acc_AB': safe_mean([r['acc_AB'] for r in rows]),
        'acc_BA': safe_mean([r['acc_BA'] for r in rows]),
        'both_correct': safe_mean([r['both_correct'] for r in rows]),
    }


def materialize_layer_rows(pairs: list[dict[str, Any]], cases: list[dict[str, Any]], layer_indices: list[int],
                           scores_by_case: dict[int, dict[int, float]]) -> list[dict[str, Any]]:
    case_map: dict[tuple[int, str, str], dict[str, Any]] = {}
    for c in cases:
        case_map[(int(c['local_pair_index']), c['ctx_key'], c['alt_key'])] = c
    rows: list[dict[str, Any]] = []
    for local_idx, p in enumerate(pairs):
        needed = [(local_idx, ctx, alt) for ctx in ['AB', 'BA', 'erased'] for alt in ['alt0', 'alt1']]
        if any(k not in case_map for k in needed):
            continue
        for li in layer_indices:
            try:
                s_AB_0 = scores_by_case[case_map[(local_idx, 'AB', 'alt0')]['case_id']][li]
                s_AB_1 = scores_by_case[case_map[(local_idx, 'AB', 'alt1')]['case_id']][li]
                s_BA_0 = scores_by_case[case_map[(local_idx, 'BA', 'alt0')]['case_id']][li]
                s_BA_1 = scores_by_case[case_map[(local_idx, 'BA', 'alt1')]['case_id']][li]
                s_er_0 = scores_by_case[case_map[(local_idx, 'erased', 'alt0')]['case_id']][li]
                s_er_1 = scores_by_case[case_map[(local_idx, 'erased', 'alt1')]['case_id']][li]
            except KeyError:
                continue
            alt0, alt1 = p['alt_0'], p['alt_1']
            cor_AB = p['correct_AB']
            if cor_AB == alt0:
                margin_AB = s_AB_0 - s_AB_1
                margin_BA = s_BA_1 - s_BA_0
                erased_bias = s_er_0 - s_er_1
            else:
                margin_AB = s_AB_1 - s_AB_0
                margin_BA = s_BA_0 - s_BA_1
                erased_bias = s_er_1 - s_er_0
            M = margin_AB + margin_BA
            CM = M - 2.0 * erased_bias
            rows.append({
                'layer': li,
                'pair_id': p.get('pair_id'),
                'family': p.get('family'),
                'template_id': p.get('template_id'),
                'style': p.get('style'),
                'entity_split': p.get('entity_split'),
                'alt_type': p.get('alt_type'),
                'alt_0': alt0,
                'alt_1': alt1,
                'correct_AB': p.get('correct_AB'),
                'correct_BA': p.get('correct_BA'),
                'margin_AB': margin_AB,
                'margin_BA': margin_BA,
                'four_cell_M': M,
                'erased_bias': erased_bias,
                'context_conditioned_M': CM,
                'acc_AB': 1.0 if margin_AB > 0 else 0.0,
                'acc_BA': 1.0 if margin_BA > 0 else 0.0,
                'both_correct': 1.0 if (margin_AB > 0 and margin_BA > 0) else 0.0,
            })
    return rows


def summarize_rows(rows: list[dict[str, Any]], target_label: str, layer_indices: list[int]) -> dict[str, Any]:
    by_layer = {str(li): summarize_layer_rows([r for r in rows if r['layer'] == li]) for li in layer_indices}
    # Final layer is the last hidden-state layer.  Best internal layer ignores the embedding layer 0.
    valid_layers = [li for li in layer_indices if str(li) in by_layer and by_layer[str(li)]['n'] > 0]
    final_layer = max(valid_layers) if valid_layers else None
    non_embedding = [li for li in valid_layers if li > 0]
    best_by_both = max(non_embedding, key=lambda li: by_layer[str(li)]['both_correct']) if non_embedding else None
    best_by_M = max(non_embedding, key=lambda li: by_layer[str(li)]['M_mean']) if non_embedding else None
    final = by_layer[str(final_layer)] if final_layer is not None else {}

    fam_final: dict[str, Any] = {}
    style_final: dict[str, Any] = {}
    if final_layer is not None:
        final_rows = [r for r in rows if r['layer'] == final_layer]
        for fam in sorted({r['family'] for r in final_rows}):
            fam_final[str(fam)] = summarize_layer_rows([r for r in final_rows if r['family'] == fam])
        for sty in sorted({r['style'] for r in final_rows}):
            style_final[str(sty)] = summarize_layer_rows([r for r in final_rows if r['style'] == sty])

    return {
        'target': target_label,
        'n_layer_rows': len(rows),
        'layer_indices': layer_indices,
        'final_layer': final_layer,
        'by_layer': by_layer,
        'best_internal_layer_by_both_correct': best_by_both,
        'best_internal_layer_by_M_mean': best_by_M,
        'final_summary': final,
        'best_internal_by_both_summary': by_layer[str(best_by_both)] if best_by_both is not None else None,
        'best_internal_by_M_summary': by_layer[str(best_by_M)] if best_by_M is not None else None,
        'final_by_family': fam_final,
        'final_by_style': style_final,
    }


def write_rows_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def run_target(label: str, meta: dict[str, Any], args: argparse.Namespace, pairs_all: list[dict[str, Any]], out_root: pathlib.Path) -> dict[str, Any]:
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    selected_pairs, selection_meta = select_stratified_pairs(
        pairs_all, tokenizer, max_per_group=args.max_per_group, seed=args.seed,
        one_token_only=args.one_token_only, include_temporal=args.include_temporal)
    cases, case_meta = make_cases(selected_pairs, tokenizer, args.max_len)
    if args.stage == 'inspect':
        return {'target': label, 'model_root': str(model_root), 'tokenizer_root': str(tok_root),
                'description': meta.get('description'), 'selection': selection_meta, 'cases': case_meta}

    torch_dtype = torch.bfloat16 if (args.dtype == 'bf16' and args.device == 'cuda') else torch.float16 if (args.dtype == 'fp16' and args.device == 'cuda') else torch.float32
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), torch_dtype=torch_dtype, trust_remote_code=True).to(device).eval()
    print(json.dumps({'event': 'target_start', 'target': label, 'device': device, 'dtype': str(torch_dtype),
                      'n_pairs_selected': len(selected_pairs), 'n_cases': len(cases),
                      'model_root': str(model_root)}), flush=True)
    t0 = time.time()
    layer_indices, scores_by_case = score_cases_layerwise(model, tokenizer, cases, device=device,
                                                          batch_size=args.batch_size,
                                                          max_layers=args.max_layers)
    rows = materialize_layer_rows(selected_pairs, cases, layer_indices, scores_by_case)
    target_dir = out_root / 'targets'
    target_dir.mkdir(parents=True, exist_ok=True)
    rows_path = target_dir / f'{label}_layer_rows.jsonl'
    write_rows_jsonl(rows_path, rows)
    summary = summarize_rows(rows, label, layer_indices)
    summary.update({
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'description': meta.get('description'),
        'selection': selection_meta,
        'cases': case_meta,
        'rows_path': str(rows_path),
        'elapsed_sec': time.time() - t0,
        'tokenizer_json_sha256': sha256_file(tok_root / 'tokenizer.json') if (tok_root / 'tokenizer.json').exists() else None,
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({'event': 'target_done', 'target': label,
                      'final_summary': summary.get('final_summary'),
                      'best_internal_by_both': summary.get('best_internal_layer_by_both_correct'),
                      'elapsed_sec': summary['elapsed_sec']}), flush=True)
    return summary


def readiness(selected: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {'pairs_path': str(PAIRS_PATH), 'pairs_ready': PAIRS_PATH.exists(), 'targets': {}}
    for label in selected:
        meta = TARGETS[label]
        mr = pathlib.Path(meta['model_root'])
        tr = pathlib.Path(meta['tokenizer_root'])
        out['targets'][label] = {
            'model_root': str(mr),
            'tokenizer_root': str(tr),
            'model_ready': bool(mr.exists() and (mr / 'model.safetensors').exists()),
            'tokenizer_ready': bool(tr.exists() and ((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists())),
            'description': meta.get('description'),
        }
    out['selected_ready'] = bool(out['pairs_ready']) and all(v['model_ready'] and v['tokenizer_ready'] for v in out['targets'].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'inspect', 'score'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['anchor_20M', 'anchor_50M', 'anchor_80M', 'anchor_100M', 'reheat_100M'])
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--max-per-group', type=int, default=24, help='stratified pairs per (family, style); 0 means all')
    ap.add_argument('--seed', type=int, default=150013)
    ap.add_argument('--one-token-only', action='store_true', help='restrict to pairs where both alternatives are single leading-space tokens')
    ap.add_argument('--include-temporal', action='store_true', help='include temporal family despite research negative/special behavior')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    ap.add_argument('--max-layers', type=int, default=None, help='optional prefix of hidden-state layers for fast smoke')
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ready = readiness(args.targets)
    (out_root / 'readiness.json').write_text(json.dumps(ready, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'readiness', 'ready': ready}, indent=2), flush=True)
    if args.stage == 'ready':
        return
    if not ready.get('selected_ready'):
        raise RuntimeError({'not_ready': ready})

    pairs_all = read_pairs(PAIRS_PATH)
    target_summaries: dict[str, Any] = {}
    for label in args.targets:
        target_summaries[label] = run_target(label, TARGETS[label], args, pairs_all, out_root)

    combined = {
        'status': 'LAYERWISE_PACKET_M_DECOMPOSITION_' + args.stage.upper(),
        'created_utc': now_utc(),
        'scientific_question': 'Does context-conditioned four-cell packet interaction exist inside existing checkpoints before new endpoint work?',
        'stage': args.stage,
        'targets': args.targets,
        'args': vars(args),
        'readiness': ready,
        'target_summaries': target_summaries,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/layerwise_packet_M_decomposition.py')),
    }
    out_path = out_root / ('layerwise_packet_M_summary_' + args.stage + '.json')
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'LAYERWISE_PACKET_M_DECOMPOSITION_DONE',
                      'summary': str(out_path),
                      'stage': args.stage,
                      'targets': args.targets}, indent=2), flush=True)


if __name__ == '__main__':
    main()
