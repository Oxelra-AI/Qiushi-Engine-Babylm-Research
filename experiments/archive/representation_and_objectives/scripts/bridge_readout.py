#!/usr/bin/env python3
"""research: vectorized readout for naturalistic exchange bridge.

Scores one or more frozen checkpoints on the research evaluation-independent bridge.
This trains nothing.  It uses the same four-cell PLL-style target scoring as the
packet readouts, but on naturalistic held-out constructions that are not official
evaluation rows and not research training grammar.
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
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('.')
WS = ROOT / 'experiments/archive/representation_and_objectives'
PAIR_PATH = WS / 'data/naturalistic_exchange_bridge/naturalistic_exchange_bridge_pairs.jsonl'
OUT_ROOT = WS / 'data/naturalistic_exchange_bridge_readout'

TARGETS: dict[str, dict[str, Any]] = {
    'anchor_80M': {
        'model_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'tokenizer_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M',
        'description': 'legal40k fixed-256 compact-view anchor, 80M',
    },
    'anchor_100M': {
        'model_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'tokenizer_root': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
        'description': 'legal40k fixed-256 compact-view anchor, 100M',
    },
    'reheat_100M': {
        'model_root': WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': WS / 'training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'reheat-cosine 100M broad-learning reference',
    },
    'residual_tail_100M': {
        'model_root': WS / 'training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M',
        'tokenizer_root': WS / 'training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M',
        'description': 'residual-low-LR tail 100M reference',
    },
    'role_switch_80M': {
        'model_root': WS / 'training/runs/role_switch_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'tokenizer_root': WS / 'training/runs/role_switch_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'description': 'research role-switch shared-tokenizer legal 80M screen',
    },
    'role_fixed_80M': {
        'model_root': WS / 'training/runs/role_fixed_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'tokenizer_root': WS / 'training/runs/role_fixed_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M',
        'description': 'research role-fixed shared-tokenizer legal 80M screen control',
    },
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_last_subseq(full_ids: list[int], sub_ids: list[int]) -> int | None:
    n = len(sub_ids)
    if n == 0:
        return None
    for i in range(len(full_ids) - n, -1, -1):
        if full_ids[i:i+n] == sub_ids:
            return i
    return None


def render_text(pair: dict[str, Any], ctx_key: str, alt_word: str) -> str:
    consequence = pair['consequence_masked'].replace('__TARGET__', alt_word)
    if ctx_key == 'erased':
        return consequence
    return pair[f'context_{ctx_key}'] + ' ' + consequence


def make_cases(pairs: list[dict[str, Any]], tokenizer, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    failures = []
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
                for j in range(len(target_ids)):
                    masked[start + j] = int(tokenizer.mask_token_id)
                cases.append({
                    'case_id': len(cases),
                    'local_pair_index': local_idx,
                    'pair_id': p.get('pair_id'),
                    'family': p.get('family'),
                    'template_id': p.get('template_id'),
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
    cnt = collections.Counter(c['local_pair_index'] for c in cases)
    complete = {i for i, n in cnt.items() if n == 6}
    hist = collections.Counter(c['target_token_len'] for c in cases)
    meta = {
        'n_input_pairs': len(pairs),
        'n_cases': len(cases),
        'n_failures': len(failures),
        'failures_head': failures[:20],
        'n_complete_pairs': len(complete),
        'n_incomplete_pairs': len(pairs) - len(complete),
        'target_token_len_hist': dict(sorted((str(k), v) for k, v in hist.items())),
        'seq_len_min': min((c['seq_len'] for c in cases), default=None),
        'seq_len_mean': sum(c['seq_len'] for c in cases) / len(cases) if cases else None,
        'seq_len_max': max((c['seq_len'] for c in cases), default=None),
    }
    return cases, meta


def pad_batch(cases: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c['input_ids']) for c in cases)
    ids, attn = [], []
    for c in cases:
        pad_n = max_len - len(c['input_ids'])
        ids.append(c['input_ids'] + [pad_id] * pad_n)
        attn.append(c['attention_mask'] + [0] * pad_n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def score_cases(model, cases: list[dict[str, Any]], tokenizer, device: str, batch_size: int) -> dict[tuple[int, str, str], float]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scores: dict[tuple[int, str, str], float] = {}
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            input_cpu, attn_cpu = pad_batch(batch, int(pad_id))
            out = model(input_ids=input_cpu.to(device), attention_mask=attn_cpu.to(device), return_dict=True)
            log_probs = F.log_softmax(out.logits.float(), dim=-1)
            for bi, c in enumerate(batch):
                s = 0.0
                for pos, tid in zip(c['target_positions'], c['target_ids']):
                    s += float(log_probs[bi, int(pos), int(tid)].item())
                scores[(int(c['local_pair_index']), c['ctx_key'], c['alt_key'])] = s
            del out, log_probs
    return scores


def safe_mean(vals: list[float]) -> float:
    xs = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(xs) / len(xs) if xs else float('nan')


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(grp: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'n': len(grp),
            'M_mean': safe_mean([r['four_cell_M'] for r in grp]),
            'M_pos_frac': safe_mean([1.0 if r['four_cell_M'] > 0 else 0.0 for r in grp]),
            'CM_mean': safe_mean([r['context_conditioned_M'] for r in grp]),
            'bias_mean': safe_mean([r['erased_bias'] for r in grp]),
            'bias_abs_mean': safe_mean([abs(r['erased_bias']) for r in grp]),
            'margin_AB_mean': safe_mean([r['margin_AB'] for r in grp]),
            'margin_BA_mean': safe_mean([r['margin_BA'] for r in grp]),
            'acc_AB': safe_mean([r['acc_AB'] for r in grp]),
            'acc_BA': safe_mean([r['acc_BA'] for r in grp]),
            'both_correct': safe_mean([r['both_correct'] for r in grp]),
        }
    fams: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    tmpls: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        fams[str(r['family'])].append(r)
        tmpls[str(r['template_id'])].append(r)
    return {
        'overall': stats(rows),
        'by_family': {k: stats(v) for k, v in sorted(fams.items())},
        'by_template': {k: stats(v) for k, v in sorted(tmpls.items())},
    }


def materialize_rows(pairs: list[dict[str, Any]], scores: dict[tuple[int, str, str], float]) -> list[dict[str, Any]]:
    rows = []
    for i, p in enumerate(pairs):
        needed = [(i, ctx, alt) for ctx in ['AB', 'BA', 'erased'] for alt in ['alt0', 'alt1']]
        if any(k not in scores for k in needed):
            continue
        s_AB_0 = scores[(i, 'AB', 'alt0')]
        s_AB_1 = scores[(i, 'AB', 'alt1')]
        s_BA_0 = scores[(i, 'BA', 'alt0')]
        s_BA_1 = scores[(i, 'BA', 'alt1')]
        s_er_0 = scores[(i, 'erased', 'alt0')]
        s_er_1 = scores[(i, 'erased', 'alt1')]
        if p['correct_AB'] == p['alt_0']:
            margin_AB = s_AB_0 - s_AB_1
            margin_BA = s_BA_1 - s_BA_0
            erased_bias = s_er_0 - s_er_1
        else:
            margin_AB = s_AB_1 - s_AB_0
            margin_BA = s_BA_0 - s_BA_1
            erased_bias = s_er_1 - s_er_0
        M = margin_AB + margin_BA
        row = {
            'pair_id': p.get('pair_id'),
            'family': p.get('family'),
            'template_id': p.get('template_id'),
            'style': p.get('style'),
            'entity_split': p.get('entity_split'),
            'alt_type': p.get('alt_type'),
            'alt_0': p.get('alt_0'),
            'alt_1': p.get('alt_1'),
            'correct_AB': p.get('correct_AB'),
            'correct_BA': p.get('correct_BA'),
            's_AB_alt0': s_AB_0,
            's_AB_alt1': s_AB_1,
            's_BA_alt0': s_BA_0,
            's_BA_alt1': s_BA_1,
            's_erased_alt0': s_er_0,
            's_erased_alt1': s_er_1,
            'margin_AB': margin_AB,
            'margin_BA': margin_BA,
            'four_cell_M': M,
            'erased_bias': erased_bias,
            'context_conditioned_M': M - 2.0 * erased_bias,
            'acc_AB': 1.0 if margin_AB > 0 else 0.0,
            'acc_BA': 1.0 if margin_BA > 0 else 0.0,
            'both_correct': 1.0 if (margin_AB > 0 and margin_BA > 0) else 0.0,
        }
        rows.append(row)
    return rows


def readiness(selected: list[str]) -> dict[str, Any]:
    out = {'pair_path': str(PAIR_PATH), 'pairs_ready': PAIR_PATH.exists(), 'targets': {}}
    for label in selected:
        meta = TARGETS[label]
        mr = pathlib.Path(meta['model_root'])
        tr = pathlib.Path(meta['tokenizer_root'])
        out['targets'][label] = {
            'model_root': str(mr),
            'tokenizer_root': str(tr),
            'model_ready': bool((mr / 'model.safetensors').exists()),
            'tokenizer_ready': bool((tr / 'tokenizer.json').exists() or (tr / 'tokenizer_config.json').exists()),
            'description': meta['description'],
        }
    out['selected_ready'] = bool(out['pairs_ready']) and all(v['model_ready'] and v['tokenizer_ready'] for v in out['targets'].values())
    return out


def run_target(label: str, args: argparse.Namespace, pairs: list[dict[str, Any]], out_root: pathlib.Path) -> dict[str, Any]:
    meta = TARGETS[label]
    model_root = pathlib.Path(meta['model_root'])
    tok_root = pathlib.Path(meta['tokenizer_root'])
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    cases, case_meta = make_cases(pairs, tokenizer, args.max_len)
    if args.stage == 'inspect':
        return {'target': label, 'description': meta['description'], 'model_root': str(model_root), 'tokenizer_root': str(tok_root), 'cases': case_meta}
    device = f'cuda:{args.gpu}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bf16' and device.startswith('cuda') else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(str(model_root), torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    print(json.dumps({'event': 'bridge_target_start', 'target': label, 'device': device, 'dtype': str(dtype), 'n_pairs': len(pairs), 'n_cases': len(cases)}), flush=True)
    t0 = time.time()
    scores = score_cases(model, cases, tokenizer, device, args.batch_size)
    rows = materialize_rows(pairs, scores)
    target_dir = out_root / 'targets'
    target_dir.mkdir(parents=True, exist_ok=True)
    rows_path = target_dir / f'{label}_bridge_rows.jsonl'
    with rows_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = summarize_rows(rows)
    summary.update({
        'target': label,
        'description': meta['description'],
        'model_root': str(model_root),
        'tokenizer_root': str(tok_root),
        'cases': case_meta,
        'rows_path': str(rows_path),
        'elapsed_sec': time.time() - t0,
        'tokenizer_json_sha256': sha256_file(tok_root / 'tokenizer.json'),
    })
    print(json.dumps({'event': 'bridge_target_done', 'target': label, 'overall': summary['overall'], 'elapsed_sec': summary['elapsed_sec']}, indent=2), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ready', 'inspect', 'score'], default='ready')
    ap.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['anchor_80M', 'anchor_100M', 'reheat_100M'])
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-len', type=int, default=128)
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
    pairs = read_pairs(PAIR_PATH)
    summaries = {}
    t0 = time.time()
    for label in args.targets:
        summaries[label] = run_target(label, args, pairs, out_root)
    combined = {
        'status': 'NATURALISTIC_BRIDGE_READOUT_' + args.stage.upper(),
        'created_utc': now_utc(),
        'scientific_question': 'Are existing checkpoints saturated on evaluation-independent naturalistic exchange contexts?',
        'pair_path': str(PAIR_PATH),
        'stage': args.stage,
        'args': vars(args),
        'readiness': ready,
        'target_summaries': summaries,
        'elapsed_sec': time.time() - t0,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/bridge_readout.py')),
    }
    out_path = out_root / f'naturalistic_bridge_readout_summary_{args.stage}.json'
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'NATURALISTIC_BRIDGE_READOUT_DONE', 'summary': str(out_path), 'stage': args.stage}, indent=2), flush=True)


if __name__ == '__main__':
    main()
