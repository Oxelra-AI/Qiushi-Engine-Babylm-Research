#!/usr/bin/env python3
"""research full-context pivot-substitution masked likelihood probe.

This is a second symmetric auxiliary-view check after the cue-only probe.  The
base WWM stream is untouched.  For probe views only, the dependent target is
masked and all other context is visible; the true-pivot view is compared with
views that replace the pivot group at the same positions with a matched nonrelation
anchor or with a same-category shuffled pivot.  Attention masks, target positions,
and visible-token counts are identical across the three main views.

The scientific question is whether the true pivot makes the model assign higher
held-out masked-token likelihood to the consequence than matched anchors and
shuffled pivots under identical information flow, including physical_change and
comparative families.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts'
for p in [str(COMPACT_EXPERIENCE_SCRIPTS), str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402
import symmetric_cue_view_likelihood_probe as cueprobe  # noqa: E402

DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/full_context_pivot_substitution_probe.md'
DEFAULT_CKPT = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M'
DEFAULT_HELDOUT_START_TAIL_ROW = 64255
DEFAULT_HELDOUT_EXPECTED_START_WORDS = 10011326
DEFAULT_HELDOUT_MAX_ROWS = 64000


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Reservoir:
    def __init__(self, quota: int):
        self.quota = int(quota)
        self.heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
        self.counter = 0
    def add(self, ev: dict[str, Any], score: float) -> None:
        self.counter += 1
        cat = str(ev['category'])
        item = (-float(score), self.counter, ev)
        h = self.heaps[cat]
        if len(h) < self.quota:
            heapq.heappush(h, item)
        elif item[0] > h[0][0]:
            heapq.heapreplace(h, item)
    def events(self) -> list[dict[str, Any]]:
        out = []
        for cat in sorted(self.heaps):
            for neg_score, _i, ev in sorted(self.heaps[cat], reverse=True):
                e2 = dict(ev); e2['final_sample_hash'] = -float(neg_score); out.append(e2)
        return out
    def counts(self) -> dict[str, int]:
        return {cat: len(v) for cat, v in self.heaps.items()}


def downsample_by_category(events: list[dict[str, Any]], quota: int, seed: int) -> list[dict[str, Any]]:
    r = Reservoir(quota)
    for ev in events:
        h = pvdm.hash_uniform(seed, 'fullctx_final', ev['tail_row_idx'], ev['event_rank'], ev['category'], ev.get('target_norm'), ev.get('pivot_norm'))
        r.add(ev, h)
    return r.events()


def filter_same_len(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    rej = Counter()
    for ev in events:
        cat = str(ev.get('category'))
        if int(ev.get('pivot_len', -1)) != int(ev.get('control_len', -2)):
            rej[f'pivot_control_len_mismatch::{cat}'] += 1
            continue
        if int(ev.get('pivot_len', 0)) <= 0:
            rej[f'bad_pivot_len::{cat}'] += 1
            continue
        out.append(ev)
    return out, {'input_events': len(events), 'kept_same_len': len(out), 'reject': dict(rej), 'kept_categories': dict(Counter([e['category'] for e in out]))}


def assign_shuffle(events: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment: dict[int, int] = {}
    level_counts = Counter(); remaining = set(range(len(events)))
    levels = [
        lambda e: (e.get('category'), e.get('target_class'), e.get('distance_bin'), e.get('pivot_len')),
        lambda e: (e.get('category'), e.get('target_class'), e.get('pivot_len')),
        lambda e: (e.get('category'), e.get('pivot_len')),
        lambda e: (e.get('category'),),
    ]
    for level, keyfn in enumerate(levels):
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for i in sorted(remaining):
            groups[keyfn(events[i])].append(i)
        newly = set()
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            ordered = sorted(idxs, key=lambda i: pvdm.hash_uniform(seed, 'fullctx_shuffle_order', level, events[i]['tail_row_idx'], events[i]['event_rank'], events[i].get('pivot_norm')))
            n = len(ordered)
            for pos, rec_idx in enumerate(ordered):
                donor = None
                for shift in range(1, n):
                    cand = ordered[(pos + shift) % n]
                    if events[cand]['tail_row_idx'] == events[rec_idx]['tail_row_idx']:
                        continue
                    if str(events[cand].get('pivot_norm')) == str(events[rec_idx].get('pivot_norm')):
                        continue
                    if int(events[cand].get('pivot_len', -1)) != int(events[rec_idx].get('pivot_len', -2)):
                        continue
                    donor = cand; break
                if donor is not None:
                    assignment[rec_idx] = donor; newly.add(rec_idx)
            level_counts[str(level)] += len(newly)
        remaining -= newly
    out = []
    for i, ev in enumerate(events):
        if i not in assignment:
            continue
        donor = events[assignment[i]]
        e2 = dict(ev)
        e2['shuffle_pivot_ids'] = [int(donor['input_ids'][p]) for p in donor['pivot_positions']]
        e2['shuffle_pivot_norm'] = str(donor.get('pivot_norm'))
        e2['shuffle_tail_row_idx'] = int(donor['tail_row_idx'])
        e2['shuffle_event_rank'] = int(donor['event_rank'])
        e2['shuffle_match_category'] = str(donor.get('category'))
        e2['shuffle_match_target_class'] = str(donor.get('target_class'))
        e2['shuffle_match_distance_bin'] = str(donor.get('distance_bin'))
        out.append(e2)
    stats = {'input_events': len(events), 'events_with_different_pivot_shuffle': len(out), 'dropped_no_shuffle': len(events) - len(out), 'level_counts': dict(level_counts), 'categories': dict(Counter(e['category'] for e in out))}
    return out, stats


def make_view(ev: dict[str, Any], kind: str, *, seq_length: int, mask_id: int) -> tuple[list[int], list[int], list[int]]:
    ids = [int(x) for x in ev['input_ids'][:seq_length]]
    attn = [int(x) for x in ev['attention_mask'][:seq_length]]
    labels = [-100] * seq_length
    tpos = [int(p) for p in ev['target_positions'] if 0 <= int(p) < seq_length]
    tids = [int(x) for x in ev['target_ids']]
    for p, tid in zip(tpos, tids):
        ids[p] = int(mask_id)
        labels[p] = int(tid)
    ppos = [int(p) for p in ev['pivot_positions'] if 0 <= int(p) < seq_length]
    cpos = [int(p) for p in ev['control_positions'] if 0 <= int(p) < seq_length]
    if kind == 'true_pivot':
        pass
    elif kind == 'matched_anchor_replace':
        if len(ppos) != len(cpos):
            raise ValueError('pivot/control length mismatch')
        for p, c in zip(ppos, cpos):
            ids[p] = int(ev['input_ids'][c])
    elif kind == 'shuffled_pivot_replace':
        shuf = [int(x) for x in ev['shuffle_pivot_ids']]
        if len(ppos) != len(shuf):
            raise ValueError('pivot/shuffle length mismatch')
        for p, tid in zip(ppos, shuf):
            ids[p] = tid
    elif kind == 'pivot_masked':
        for p in ppos:
            ids[p] = int(mask_id)
    elif kind == 'target_only_no_context':
        pad_id = 0
        ids = [pad_id] * seq_length; attn = [0] * seq_length; labels = [-100] * seq_length
        for p, tid in zip(tpos, tids):
            ids[p] = int(mask_id); attn[p] = 1; labels[p] = int(tid)
    else:
        raise ValueError(kind)
    return ids, attn, labels


def score_events(events: list[dict[str, Any]], model: torch.nn.Module, tokenizer: Any, *, device: torch.device, seq_length: int, view_batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kinds = ['true_pivot', 'matched_anchor_replace', 'shuffled_pivot_replace', 'pivot_masked', 'target_only_no_context']
    mask_id = int(tokenizer.mask_token_id)
    items = []
    for i, ev in enumerate(events):
        for kind in kinds:
            ids, attn, labels = make_view(ev, kind, seq_length=seq_length, mask_id=mask_id)
            items.append((i, kind, ids, attn, labels))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    scored = [dict(e) for e in events]
    with torch.no_grad():
        for start in range(0, len(items), view_batch_size):
            chunk = items[start:start + view_batch_size]
            ids = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            attn = torch.tensor([x[3] for x in chunk], dtype=torch.long, device=device)
            labels = torch.tensor([x[4] for x in chunk], dtype=torch.long, device=device)
            out = model(input_ids=ids, attention_mask=attn)
            logp = F.log_softmax(out.logits.float(), dim=-1)
            for row, (ei, kind, _ids, _attn, _labels) in enumerate(chunk):
                pos = (labels[row] != -100).nonzero(as_tuple=False).view(-1)
                target = labels[row].index_select(0, pos)
                vals = logp[row].index_select(0, pos).gather(1, target.view(-1, 1)).view(-1)
                scored[ei][f'logp_{kind}'] = float(vals.mean().detach().cpu().item())
                scored[ei][f'ntok_{kind}'] = int(pos.numel())
            del ids, attn, labels, out, logp
            if start == 0 or (start // view_batch_size) % 20 == 0:
                print(json.dumps({'event': 'score_progress', 'views_done': min(start + len(chunk), len(items)), 'views_total': len(items)}), flush=True)
    return scored, {'events_scored': len(scored), 'views_scored': len(items), 'view_kinds': kinds, 'cuda_peak_memory_gb': (torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == 'cuda' else None)}


def num_add(d: dict[str, Any], pfx: str, val: float) -> None:
    cueprobe.numeric_add(d, pfx, val)


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = defaultdict(dict)
    ctr: dict[str, Counter] = defaultdict(Counter)
    pairs = [
        ('true_minus_anchor_replace', 'true_pivot', 'matched_anchor_replace'),
        ('true_minus_shuffle_replace', 'true_pivot', 'shuffled_pivot_replace'),
        ('true_minus_pivot_masked', 'true_pivot', 'pivot_masked'),
        ('true_minus_no_context', 'true_pivot', 'target_only_no_context'),
        ('anchor_minus_shuffle', 'matched_anchor_replace', 'shuffled_pivot_replace'),
    ]
    score_names = ['true_pivot','matched_anchor_replace','shuffled_pivot_replace','pivot_masked','target_only_no_context']
    for ev in scored:
        for cat in ['ALL', str(ev['category'])]:
            d = by[cat]; d['n_events'] = int(d.get('n_events', 0)) + 1
            for name in score_names:
                num_add(d, f'logp_{name}', float(ev[f'logp_{name}']))
            for dname, a, b in pairs:
                delta = float(ev[f'logp_{a}']) - float(ev[f'logp_{b}'])
                num_add(d, dname, delta)
                d[f'{dname}_succ'] = int(d.get(f'{dname}_succ', 0)) + int(delta > 0)
                d[f'{dname}_den'] = int(d.get(f'{dname}_den', 0)) + 1
            ctr[cat]['pivot_norm::' + str(ev.get('pivot_norm'))] += 1
            ctr[cat]['control_norm::' + str(ev.get('control_norm'))] += 1
            ctr[cat]['shuffle_pivot_norm::' + str(ev.get('shuffle_pivot_norm'))] += 1
            ctr[cat]['target_norm::' + str(ev.get('target_norm'))] += 1
            ctr[cat]['target_class::' + str(ev.get('target_class'))] += 1
            ctr[cat]['source::' + str(ev.get('source'))] += 1
    out = {}
    for cat, d in by.items():
        item = cueprobe.finalize_numeric(d)
        item['n_events'] = int(d.get('n_events', 0))
        for dname, _, _ in pairs:
            den = int(d.get(f'{dname}_den', 0))
            item[f'{dname}_success_rate'] = int(d.get(f'{dname}_succ', 0)) / den if den else None
        for field in ['pivot_norm','control_norm','shuffle_pivot_norm','target_norm','target_class','source']:
            item[f'top_{field}'] = dict(Counter({k.split(field+'::',1)[1]: v for k, v in ctr[cat].items() if k.startswith(field+'::')}).most_common(15))
        out[cat] = item
    readiness = {}
    for cat in ['physical_change', 'comparative']:
        item = out.get(cat, {})
        n = int(item.get('n_events', 0) or 0)
        ma = (item.get('true_minus_anchor_replace') or {}).get('mean')
        ms = (item.get('true_minus_shuffle_replace') or {}).get('mean')
        sa = item.get('true_minus_anchor_replace_success_rate')
        ss = item.get('true_minus_shuffle_replace_success_rate')
        readiness[cat] = {'n_events': n, 'mean_true_minus_anchor_replace': ma, 'mean_true_minus_shuffle_replace': ms, 'success_true_gt_anchor_replace': sa, 'success_true_gt_shuffle_replace': ss, 'supports_separation': bool(n >= 50 and ma is not None and ms is not None and ma > 0.02 and ms > 0.02 and sa is not None and ss is not None and sa > 0.55 and ss > 0.55)}
    out['readiness_by_missing_family'] = readiness
    return out


def write_note(path: Path, summary: dict[str, Any], scored_path: Path, sample_path: Path) -> None:
    lines = []
    lines.append('# research — full-context pivot-substitution likelihood probe\n')
    lines.append(f"Created: {summary['created_utc']} Runtime: {summary['runtime_sec']} s Device: {summary['device']}\n")
    lines.append('## View\n')
    lines.append('Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.\n')
    lines.append('## Coverage\n')
    lines.append(f"Segment: {summary['segment']}\n")
    lines.append(f"Initial scan candidates: {summary['scan_stats'].get('candidate_categories')}\n")
    lines.append(f"Same-length filter: {summary['same_len_stats']}\n")
    lines.append(f"Final sampled categories: {summary['sample_counts']}\n")
    lines.append(f"Shuffle stats: {summary['shuffle_stats']}\n")
    lines.append('## Likelihood separation\n')
    for cat in ['ALL','physical_change','comparative','causal_connector','temporal','spatial','negation']:
        if cat in summary['score_summary']:
            item = summary['score_summary'][cat]
            lines.append(f"- {cat}: n={item.get('n_events')}; true-anchor mean={(item.get('true_minus_anchor_replace') or {}).get('mean')} success={item.get('true_minus_anchor_replace_success_rate')}; true-shuffle mean={(item.get('true_minus_shuffle_replace') or {}).get('mean')} success={item.get('true_minus_shuffle_replace_success_rate')}; true-pivotmasked mean={(item.get('true_minus_pivot_masked') or {}).get('mean')} success={item.get('true_minus_pivot_masked_success_rate')}\n")
    lines.append('## Missing-family readiness\n')
    lines.append(json.dumps(summary['score_summary'].get('readiness_by_missing_family', {}), indent=2, ensure_ascii=False) + '\n')
    lines.append('## Files\n')
    lines.append(f"- summary: `{summary['artifacts']['summary']}`\n")
    lines.append(f"- scored events: `{scored_path}`\n")
    lines.append(f"- samples: `{sample_path}`\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--init_checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--checkpoint_label', default='standard_legacy_chck80M')
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--start_tail_row', type=int, default=DEFAULT_HELDOUT_START_TAIL_ROW)
    ap.add_argument('--expected_start_tail_words', type=int, default=DEFAULT_HELDOUT_EXPECTED_START_WORDS)
    ap.add_argument('--max_rows', type=int, default=DEFAULT_HELDOUT_MAX_ROWS)
    ap.add_argument('--max_word_exposure', type=int, default=9971289)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--scan_quota_per_category', type=int, default=1600)
    ap.add_argument('--score_quota_per_category', type=int, default=384)
    ap.add_argument('--view_batch_size', type=int, default=96)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=8)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=2)
    ap.add_argument('--max_target_token_len', type=int, default=6)
    ap.add_argument('--categories', nargs='*', default=sorted(relaux.ALL_CATEGORIES))
    ap.add_argument('--seed', type=int, default=43025)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time(); outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail_path), 'labels_sha256': base.sha256_file(labels_path), 'tokenizer_sha256': base.sha256_file(tok_path / 'tokenizer.json')}
    for k, expected in cont.EXPECTED.items():
        if hashes[k] != expected:
            raise RuntimeError(f'{k} mismatch {hashes[k]} != {expected}')
    if not init_ckpt.exists():
        raise RuntimeError(f'checkpoint not found: {init_ckpt}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, labels, segment = cont.load_segment(tail_path, labels_path, start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    print(json.dumps({'event': 'loaded_segment', 'checkpoint_label': args.checkpoint_label, 'examples': len(examples), 'segment': segment}), flush=True)
    selected, scan_stats = cueprobe.select_candidate_events(
        examples, labels, tokenizer,
        batch_size=args.batch_size, seq_length=args.seq_length,
        allowed_categories=set(args.categories) & relaux.ALL_CATEGORIES,
        quota_per_category=args.scan_quota_per_category,
        seed=args.seed,
        max_control_distance_abs_diff=args.max_control_distance_abs_diff,
        max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff,
        max_target_token_len=args.max_target_token_len,
        num_workers=args.num_workers,
    )
    same_len, same_len_stats = filter_same_len(selected)
    sampled = downsample_by_category(same_len, args.score_quota_per_category, args.seed)
    shuffled, shuffle_stats = assign_shuffle(sampled, args.seed)
    sample_counts = dict(Counter(e['category'] for e in shuffled))
    print(json.dumps({'event': 'events_prepared', 'scan_reservoir': dict(Counter(e['category'] for e in selected)), 'same_len': same_len_stats, 'sample_counts': sample_counts, 'shuffle': shuffle_stats}), flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all(args.seed + 99)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(f'vocab mismatch {model.config.vocab_size} != {len(tokenizer)}')
    model.to(device)
    scored, scoring_stats = score_events(shuffled, model, tokenizer, device=device, seq_length=args.seq_length, view_batch_size=args.view_batch_size)
    score_summary = summarize(scored)
    scan_json = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in scan_stats.items()}
    summary = {
        'status': 'FULL_CONTEXT_PIVOT_SUBSTITUTION_PROBE',
        'created_utc': now_utc(), 'runtime_sec': round(time.time() - t0, 2), 'device': str(device),
        'purpose': 'measure true pivot masked-target likelihood contribution under identical full-context information flow before any auxiliary training',
        'checkpoint_label': args.checkpoint_label,
        'init_checkpoint': str(init_ckpt),
        'hashes': hashes,
        'segment': segment,
        'view_definition': {
            'base_wwm_pass_modified': False,
            'target_group': 'masked and scored in every view',
            'true_pivot': 'original context with target masked',
            'matched_anchor_replace': 'same context and attention but pivot group replaced at same positions by same-row matched nonrelation anchor tokens',
            'shuffled_pivot_replace': 'same context and attention but pivot group replaced at same positions by a different same-category pivot group',
            'pivot_masked': 'same context and target masked, pivot group also masked',
            'target_only_no_context': 'only target masks visible, a lower-context reference',
        },
        'args': vars(args),
        'scan_stats': scan_json,
        'same_len_stats': same_len_stats,
        'sample_counts': sample_counts,
        'shuffle_stats': shuffle_stats,
        'scoring_stats': scoring_stats,
        'score_summary': score_summary,
        'artifacts': {
            'summary': str(outdir / 'full_context_pivot_substitution_summary.json'),
            'scored_events': str(outdir / 'scored_events.jsonl'),
            'sample_events': str(outdir / 'sample_events.jsonl'),
            'note': str(Path(args.note_path)),
        },
    }
    summary_path = outdir / 'full_context_pivot_substitution_summary.json'
    scored_path = outdir / 'scored_events.jsonl'
    sample_path = outdir / 'sample_events.jsonl'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with scored_path.open('w', encoding='utf-8') as f:
        for ev in scored:
            slim = {k: v for k, v in ev.items() if k not in {'input_ids','attention_mask'}}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    with sample_path.open('w', encoding='utf-8') as f:
        for ev in scored[:120]:
            slim = {k: v for k, v in ev.items() if k not in {'input_ids','attention_mask'}}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    write_note(Path(args.note_path), summary, scored_path, sample_path)
    print(json.dumps({'status': summary['status'], 'summary': str(summary_path), 'note': str(Path(args.note_path)), 'events_scored': len(scored), 'sample_counts': sample_counts, 'readiness': score_summary.get('readiness_by_missing_family'), 'runtime_sec': summary['runtime_sec']}), flush=True)


if __name__ == '__main__':
    main()
