#!/usr/bin/env python3
"""research: intra-row two-argument relation funnel.

Builds a legal active-token reservoir of single-sentence relation expressions with
a pivot and two content arguments on opposite sides of the pivot. This is a repair
after research/early research cross-row pair pools mostly measured arbitrary target
replacement. Here the two targets come from the same local relation context, so a
future no-update or training objective can test slot/role assignment:

  score(original left/right assignment) vs score(swapped left/right assignment)

Every saved target position and token id comes from the actual MaskedChunkDataset
256-token view. No model is loaded and no official evaluation text is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import pvdm_strict_label_builder as lab  # noqa: E402
import relation_active_pair_funnel as active_funnel  # noqa: E402

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/intrarow_twoarg_relation_funnel.md')
WORD_RE = re.compile(r"\S+")
PUNCT_STRIP = "\"'“”‘’.,!?;:()[]{}<>«»‹›*_-=+/\\|`~"
PIVOT_CATS = {'temporal','spatial','causal_connector','comparative'}
# A tighter subset where opposite/ordered arguments are semantically meaningful.
DIRECTIONAL_PIVOTS = {
    'before','after','during','until','while','when','since','once',
    'above','below','under','beneath','behind','beside','between','inside','outside','within','across','through','toward','towards','into','onto','against','beyond',
    'because','although','though','whereas','if','unless','than',
    'more','less','greater','smaller','larger','higher','lower','faster','slower','longer','shorter','better','worse','least','most',
}
EXTRA_GENERIC = {'thing','things','something','anything','everything','someone','people','person','way','time','day','year','part','kind','sort','place','work','use','used','using','good','bad','new','old','first','last','many','much','same','different','right','left','actually','really','well','course','maybe','probably'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stable_u(seed: int, *items: Any) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for it in items:
        h.update(b'\0'); h.update(str(it).encode('utf-8', errors='ignore'))
    return int.from_bytes(h.digest(), 'big') / float(2**64)


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float64)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p*(len(xs)-1); lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n': len(xs), 'mean': float(arr.mean()), 'std': float(arr.std()), 'median': q(0.5), 'p05': q(0.05), 'p25': q(0.25), 'p75': q(0.75), 'p95': q(0.95), 'min': xs[0], 'max': xs[-1]}


def norm(w: str) -> str:
    return lab.norm(w)


def token_pattern(tokenizer, ids: list[int]) -> str:
    return active_funnel.token_pattern(tokenizer, ids)


def target_ok(words: list[str], idx: int, pivot_idx: int) -> bool:
    if idx < 0 or idx >= len(words) or idx == pivot_idx:
        return False
    n = norm(words[idx])
    if not lab.is_content_target(words[idx]):
        return False
    if n in EXTRA_GENERIC or n in DIRECTIONAL_PIVOTS:
        return False
    return True


def nearest_targets(words: list[str], pivot_idx: int, window: int, cap_each_side: int) -> tuple[list[int], list[int]]:
    left=[]; right=[]
    # Stop at strong punctuation if a closer target already exists; this keeps the
    # target pair local to the pivot expression without requiring a parser.
    for j in range(pivot_idx-1, max(-1, pivot_idx-1-window), -1):
        if target_ok(words, j, pivot_idx):
            left.append(j)
            if len(left) >= cap_each_side:
                break
        if left and str(words[j]).endswith(('.', '?', '!', ';', ':')):
            break
    for j in range(pivot_idx+1, min(len(words), pivot_idx+1+window)):
        if target_ok(words, j, pivot_idx):
            right.append(j)
            if len(right) >= cap_each_side:
                break
        if right and str(words[j]).endswith(('.', '?', '!', ';', ':')):
            break
    return left, right


def local_window(words: list[str], pivot_i: int, target_i: int, radius: int = 5) -> str:
    lo, hi = max(0, min(pivot_i, target_i)-radius), min(len(words), max(pivot_i, target_i)+radius+1)
    out=[]
    for k in range(lo, hi):
        if k == pivot_i:
            out.append('<P>')
        elif k == target_i:
            out.append('<T>')
        else:
            nw = norm(words[k])
            if nw:
                out.append(nw)
    return ' '.join(out)


def span_between(words: list[str], a: int, b: int) -> list[str]:
    lo, hi = sorted([a, b])
    toks=[]
    for k in range(lo+1, hi):
        nw = norm(words[k])
        if nw and nw not in lab.FUNCTION_ANCHORS and nw not in lab.STOP_TARGETS:
            toks.append(nw)
    return toks


def pair_record(lr: dict[str, Any], input_ids_row: torch.Tensor, group_pos: dict[int, list[int]], tokenizer, freqs: Counter[str], pivot_i: int, left_i: int, right_i: int, cat: str, args: argparse.Namespace, rank: int) -> tuple[dict[str, Any] | None, str]:
    words = str(lr.get('text','')).split()
    pnorm = norm(words[pivot_i]); lnorm = norm(words[left_i]); rnorm = norm(words[right_i])
    if pnorm not in DIRECTIONAL_PIVOTS:
        return None, 'pivot_not_directional'
    if lnorm == rnorm:
        return None, 'same_target_norm'
    left_pos = [int(x) for x in group_pos.get(left_i, [])]
    right_pos = [int(x) for x in group_pos.get(right_i, [])]
    pivot_pos = [int(x) for x in group_pos.get(pivot_i, [])]
    if not left_pos or not right_pos or not pivot_pos:
        return None, 'missing_active_token_group'
    if len(left_pos) > args.max_target_token_len or len(right_pos) > args.max_target_token_len:
        return None, 'target_token_len_gt_max'
    left_ids = [int(input_ids_row[p].item()) for p in left_pos]
    right_ids = [int(input_ids_row[p].item()) for p in right_pos]
    if args.require_same_token_shape:
        if len(left_ids) != len(right_ids) or token_pattern(tokenizer, left_ids) != token_pattern(tokenizer, right_ids):
            return None, 'token_shape_mismatch'
    if lab.target_class(words[left_i]) != lab.target_class(words[right_i]):
        return None, 'target_class_mismatch'
    fdl = lab.freq_bin_id(freqs.get(lnorm, 0)); fdr = lab.freq_bin_id(freqs.get(rnorm, 0))
    if abs(fdl-fdr) > args.max_target_freq_delta:
        return None, 'target_freq_delta'
    ctx_l = active_funnel.context_without_target(str(lr.get('text','')), left_i)
    ctx_r = active_funnel.context_without_target(str(lr.get('text','')), right_i)
    if args.reject_duplicate_target_leak:
        # Reject only duplicate mentions beyond the counterpart target itself.
        if active_funnel.contains_norm_as_word(ctx_l.replace(rnorm, '', 1), lnorm):
            return None, 'left_target_duplicate_leak'
        if active_funnel.contains_norm_as_word(ctx_r.replace(lnorm, '', 1), rnorm):
            return None, 'right_target_duplicate_leak'
    dleft = abs(pivot_i-left_i); dright = abs(right_i-pivot_i)
    between = span_between(words, left_i, right_i)
    rec = {
        'pair_id': f'p{rank:07d}', 'match_level': 0, 'pair_cost': float(abs(dleft-dright) + 2*abs(fdl-fdr) + 0.01*abs(left_i+right_i-2*pivot_i)),
        'category': cat, 'pivot_norm_a': pnorm, 'pivot_norm_b': pnorm, 'pivot_gid_a': int(pivot_i), 'pivot_gid_b': int(pivot_i),
        'row_a': int(lr.get('tail_row_idx', -1)), 'row_b': int(lr.get('tail_row_idx', -1)), 'source_a': lr.get('source'), 'source_b': lr.get('source'),
        'uid_a': f"{lr.get('tail_row_idx')}:{pivot_i}:{left_i}:L", 'uid_b': f"{lr.get('tail_row_idx')}:{pivot_i}:{right_i}:R",
        'target_gid_a': int(left_i), 'target_gid_b': int(right_i), 'target_norm_a': lnorm, 'target_norm_b': rnorm,
        'target_ids_a': left_ids, 'target_ids_b': right_ids, 'target_positions_a': left_pos, 'target_positions_b': right_pos,
        'target_class': lab.target_class(words[left_i]), 'target_token_len': len(left_ids), 'target_token_pattern': token_pattern(tokenizer, left_ids),
        'target_freq_bin_a': lab.freq_bin(freqs.get(lnorm,0)), 'target_freq_bin_b': lab.freq_bin(freqs.get(rnorm,0)), 'target_freq_bin_delta': abs(fdl-fdr),
        'distance_bin_a': lab.dist_bin(dleft), 'distance_bin_b': lab.dist_bin(dright), 'distance_bin_delta': abs(active_funnel.relaux.DIST_BIN_ID.get(lab.dist_bin(dleft), -1) - active_funnel.relaux.DIST_BIN_ID.get(lab.dist_bin(dright), -1)) if hasattr(active_funnel, 'relaux') else 0,
        'target_side_a': 'left', 'target_side_b': 'right', 'pivot_between_targets': left_i < pivot_i < right_i,
        'left_distance': dleft, 'right_distance': dright, 'between_content': between[:20], 'between_content_count': len(between),
        'local_window_a': local_window(words, pivot_i, left_i), 'local_window_b': local_window(words, pivot_i, right_i),
        'text_a': str(lr.get('text','')), 'text_b': str(lr.get('text','')),
    }
    return rec, 'ok'


def load_pairs(args: argparse.Namespace, tokenizer, freqs: Counter[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples, labels, segment = cont.load_segment(Path(args.tail_jsonl), Path(args.labels_jsonl), start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    for ex, lr in zip(examples, labels):
        lr['text'] = ex.text; lr['words'] = ex.words; lr['source'] = getattr(ex, 'source', ''); lr['example_id'] = getattr(ex, 'example_id', lr.get('example_id'))
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers)
    pairs=[]; counts=Counter(); cats=Counter(); pivots=Counter(); rejects=Counter(); target_use=Counter(); examples_out=[]
    t0=time.time()
    for step, batch in enumerate(loader, 1):
        lo=(research)*args.batch_size; hi=lo+int(batch['input_ids'].shape[0])
        input_ids=batch['input_ids'][:,:args.seq_length].contiguous(); attn=batch['attention_mask'][:,:args.seq_length].contiguous(); wg=batch['word_group'][:,:args.seq_length].contiguous()
        for b, lr in enumerate(labels[lo:hi]):
            counts['rows_seen'] += 1
            words = str(lr.get('text','')).split()
            group_pos = pvdm.group_positions_for_row(wg[b], attn[b])
            row_added = 0
            for pivot_i, raw in enumerate(words):
                pnorm = norm(raw)
                cat = lab.pivot_category(words, pivot_i)
                if cat not in PIVOT_CATS or pnorm not in DIRECTIONAL_PIVOTS:
                    continue
                lefts, rights = nearest_targets(words, pivot_i, args.argument_window, args.candidates_each_side)
                if not lefts or not rights:
                    rejects['missing_left_or_right_arg'] += 1; rejects[f'missing_left_or_right_arg::{cat}'] += 1; continue
                local_candidates=[]
                for li in lefts:
                    for ri in rights:
                        if row_added >= args.max_pairs_per_row:
                            break
                        if target_use[norm(words[li])] >= args.max_pairs_per_target or target_use[norm(words[ri])] >= args.max_pairs_per_target:
                            rejects['target_cap'] += 1; continue
                        rec, status = pair_record(lr, input_ids[b], group_pos, tokenizer, freqs, pivot_i, li, ri, cat, args, len(pairs)+len(local_candidates))
                        if rec is None:
                            rejects[status] += 1; rejects[f'{status}::{cat}'] += 1; continue
                        local_candidates.append(rec)
                    if row_added >= args.max_pairs_per_row:
                        break
                local_candidates.sort(key=lambda r: (r['pair_cost'], stable_u(args.seed, 'rowcand', r['uid_a'], r['uid_b'])))
                for rec in local_candidates[:max(0, args.max_pairs_per_pivot)]:
                    rec['pair_id'] = f'p{len(pairs):07d}'
                    pairs.append(rec); row_added += 1; cats[cat] += 1; pivots[pnorm] += 1; target_use[rec['target_norm_a']] += 1; target_use[rec['target_norm_b']] += 1
                    if len(examples_out) < args.sample_pairs:
                        examples_out.append(rec)
                    if args.max_pairs and len(pairs) >= args.max_pairs:
                        break
                if args.max_pairs and len(pairs) >= args.max_pairs:
                    break
            if row_added:
                counts['rows_with_pairs'] += 1
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break
        if step == 1 or step % 50 == 0:
            print(json.dumps({'event':'twoarg_progress','step':step,'rows_seen':counts['rows_seen'],'pairs':len(pairs),'elapsed_sec':round(time.time()-t0,1)}), flush=True)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break
    dist_delta=[float(p['distance_bin_delta']) for p in pairs]
    fdelta=[float(p['target_freq_bin_delta']) for p in pairs]
    between=[float(p['between_content_count']) for p in pairs]
    summary={'segment': segment, 'counts': dict(counts), 'pairs': len(pairs), 'pairs_by_category': dict(cats), 'top_pivots': pivots.most_common(40), 'target_use_top20': target_use.most_common(20), 'rejects': dict(rejects), 'distance_bin_delta': qstats(dist_delta), 'target_freq_delta': qstats(fdelta), 'between_content_count': qstats(between), 'sample_pairs': examples_out}
    return pairs, summary


def write_outputs(pairs: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace, runtime: float) -> None:
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pair_path=out/'intrarow_twoarg_pair_pool.jsonl'
    with pair_path.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False)+'\n')
    csv_path=out/'intrarow_twoarg_pair_pool_summary.csv'
    fields=['pair_id','category','pivot_norm_a','target_norm_a','target_norm_b','target_class','target_token_len','target_freq_bin_a','target_freq_bin_b','left_distance','right_distance','between_content_count','row_a','source_a']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p in pairs:
            w.writerow({k:p.get(k) for k in fields})
    full={'status':'INTRAROW_TWOARG_RELATION_FUNNEL','created_utc':now_utc(),'purpose':'Single-context two-argument relation-slot pool with active token positions; no model scoring or training.','inputs':{'labels_jsonl':str(args.labels_jsonl),'tail_jsonl':str(args.tail_jsonl),'tokenizer_path':str(args.tokenizer_path),'freq_source':str(lab.POOL_10M)},'parameters':vars(args),'summary':summary,'runtime_sec':runtime,'outputs':{'pairs_jsonl':str(pair_path),'pairs_csv':str(csv_path),'summary_json':str(out/'intrarow_twoarg_funnel_summary.json'),'note':str(args.note_path)}}
    (out/'intrarow_twoarg_funnel_summary.json').write_text(json.dumps(full, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — intra-row two-argument relation funnel','']
    lines.append('This pool searches the legal active-token tail for one relation pivot with two active content arguments on opposite sides, so a later four-cell scorer can compare original left/right target assignment to swapped assignment inside the same relation context.')
    seg=summary['segment']; lines.append(f"Segment: {seg['selected_rows']} rows / {seg['selected_words']} words starting tail row {seg['start_tail_row']}.")
    lines.append(f"Pairs: **{summary['pairs']:,}**; rows with pairs: {summary['counts'].get('rows_with_pairs',0):,}/{summary['counts'].get('rows_seen',0):,}.")
    lines.append('')
    lines.append('## Pairs by relation family')
    lines.append('| family | pairs |'); lines.append('|---|---:|')
    for k,v in sorted(summary['pairs_by_category'].items(), key=lambda kv:(-kv[1], kv[0])):
        lines.append(f'| {k} | {v} |')
    lines.append('')
    lines.append(f"Target freq delta: `{summary['target_freq_delta']}`")
    lines.append(f"Distance-bin delta: `{summary['distance_bin_delta']}`")
    lines.append(f"Between-content count: `{summary['between_content_count']}`")
    lines.append('')
    lines.append('Top pivots:')
    for p,c in summary['top_pivots'][:18]:
        lines.append(f'- {p}: {c}')
    lines.append('')
    lines.append('Files:')
    lines.append(f"- summary: `{out/'intrarow_twoarg_funnel_summary.json'}`")
    lines.append(f"- pairs: `{pair_path}`")
    lines.append(f"- csv: `{csv_path}`")
    Path(args.note_path).parent.mkdir(parents=True, exist_ok=True); Path(args.note_path).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':full['status'],'pairs':len(pairs),'summary':str(out/'intrarow_twoarg_funnel_summary.json'),'note':str(args.note_path)}, indent=2), flush=True)


def build_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS)); ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL)); ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT)); ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--start_tail_row', type=int, default=cont.DEFAULT_START_TAIL_ROW); ap.add_argument('--expected_start_tail_words', type=int, default=cont.DEFAULT_START_TAIL_WORDS); ap.add_argument('--max_word_exposure', type=int, default=cont.DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument('--max_rows', type=int, default=0); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--seq_length', type=int, default=256); ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--argument_window', type=int, default=10); ap.add_argument('--candidates_each_side', type=int, default=2); ap.add_argument('--max_pairs_per_pivot', type=int, default=1); ap.add_argument('--max_pairs_per_row', type=int, default=2); ap.add_argument('--max_pairs_per_target', type=int, default=24); ap.add_argument('--max_pairs', type=int, default=0); ap.add_argument('--sample_pairs', type=int, default=80)
    ap.add_argument('--max_target_token_len', type=int, default=4); ap.add_argument('--require_same_token_shape', action='store_true', default=True); ap.add_argument('--no_require_same_token_shape', dest='require_same_token_shape', action='store_false'); ap.add_argument('--max_target_freq_delta', type=int, default=2); ap.add_argument('--reject_duplicate_target_leak', action='store_true', default=False); ap.add_argument('--seed', type=int, default=43030)
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    t0=time.time(); args=build_args()
    print(json.dumps({'event':'build_freq_start','source':str(lab.POOL_10M)}), flush=True)
    freqs=lab.build_freqs(lab.POOL_10M)
    print(json.dumps({'event':'build_freq_done','types':len(freqs),'elapsed_sec':round(time.time()-t0,1)}), flush=True)
    tok=AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    pairs, summary=load_pairs(args, tok, freqs)
    write_outputs(pairs, summary, args, round(time.time()-t0, 2))


if __name__=='__main__':
    main()
