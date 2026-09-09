#!/usr/bin/env python3
"""research natural two-context/two-target feasibility probe.

This is a CPU-only route-construction probe for the narrowed EWoK/SCRA object.
It mines *natural* (context, target) sentence transitions from the actual
compact-view-reinvest 10M corpus. A candidate pair consists of two observed
transitions (C1->T1, C2->T2) whose targets share content words but come from
different corpus rows. The diagnostic then scores the four-cell matrix
S(C_i,T_j). Because T1 and T2 are both natural targets and each appears once as
positive and once as cross-negative, lexical target priors are much closer to
cancelling than in sentence-level word replacement.

It does not train, does not use EWoK wording or EWoK relation inventories, and
does not run official evaluation. The purpose is only to decide whether a
corpus-derived interaction-ranking objective is even worth precise construction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path('experiments/archive/representation_and_objectives')
WS = ROOT
CORPUS = Path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
OUT = WS / 'data/natural_two_context_target_feasibility'
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/natural_two_context_target_feasibility.md')

DEFAULT_MODELS = {
    'legal40_depth_12x384_43022': WS / 'training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M',
    'legal40_8x480_43022': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
}

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")
STOP = {
    'a','an','the','and','or','but','if','then','than','that','this','these','those','there','here','with','without','into','onto','from','for','of','to','in','on','at','by','as','is','are','was','were','be','been','being','it','its','they','them','their','he','she','his','her','we','you','i','me','my','our','your','who','what','when','where','why','how','which','all','some','any','each','many','much','more','most','less','least','not','no','yes','can','could','may','might','must','shall','should','will','would','do','does','did','done','have','has','had','having','also','only','just','very','one','two','three','first','second','new','old','such','same','other','another','about','after','before','over','under','between','during','while','through','within','because','so','however','therefore','example','including'
}
PUNCT_STRIP = string.punctuation + '“”‘’«»‹›'


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def norm_token(tok: str) -> str:
    return tok.lower().strip(PUNCT_STRIP)


def word_count(text: str) -> int:
    return len(text.split())


def content_tokens(text: str) -> list[str]:
    toks = []
    for m in WORD_RE.finditer(text):
        t = norm_token(m.group(0))
        if len(t) >= 4 and t not in STOP:
            toks.append(t)
    return toks


def sent_split(text: str) -> list[str]:
    rough = SENT_SPLIT_RE.split(text.strip())
    out = []
    for s in rough:
        s = re.sub(r'\s+', ' ', s.strip())
        if s:
            out.append(s)
    return out


@dataclass
class Transition:
    tid: int
    row_i: int
    sent_i: int
    source: str
    example_id: Any
    context: str
    target: str
    context_words: int
    target_words: int
    target_terms: tuple[str, ...]
    context_terms: tuple[str, ...]


def load_transitions(max_rows_scan: int, max_transitions: int, seed: int) -> tuple[list[Transition], dict[str, Any]]:
    rng = random.Random(seed)
    transitions: list[Transition] = []
    source_counts = Counter()
    rows_seen = 0
    sentence_fragments = 0
    with CORPUS.open('r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            if max_rows_scan and row_i >= max_rows_scan:
                break
            rows_seen += 1
            obj = json.loads(line)
            text = obj.get('text', '')
            source = str(obj.get('source', ''))
            sents = sent_split(text)
            sentence_fragments += len(sents)
            for j in range(len(sents) - 1):
                c = sents[j]
                t = sents[j + 1]
                cw = word_count(c); tw = word_count(t)
                if not (5 <= cw <= 35 and 5 <= tw <= 28):
                    continue
                t_terms = tuple(sorted(set(content_tokens(t))))
                c_terms = tuple(sorted(set(content_tokens(c))))
                if len(t_terms) < 3 or len(c_terms) < 2:
                    continue
                transitions.append(Transition(
                    tid=len(transitions), row_i=row_i, sent_i=j, source=source,
                    example_id=obj.get('example_id'), context=c, target=t,
                    context_words=cw, target_words=tw, target_terms=t_terms, context_terms=c_terms,
                ))
                source_counts[source] += 1
                if max_transitions and len(transitions) >= max_transitions:
                    break
            if max_transitions and len(transitions) >= max_transitions:
                break
    meta = {
        'corpus': str(CORPUS),
        'rows_seen': rows_seen,
        'sentence_fragments': sentence_fragments,
        'transitions': len(transitions),
        'source_counts': dict(source_counts.most_common()),
        'seed': seed,
    }
    return transitions, meta


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_pairs(transitions: list[Transition], pair_limit: int, seed: int, per_source_pair_cap: int = 80) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    inv: dict[str, list[int]] = defaultdict(list)
    for tr in transitions:
        for tok in tr.target_terms:
            if len(inv[tok]) < 2000:
                inv[tok].append(tr.tid)
    candidates: list[dict[str, Any]] = []
    seen = set()
    source_pair_counts = Counter()
    order = list(range(len(transitions)))
    rng.shuffle(order)
    for tid in order:
        a = transitions[tid]
        a_terms = set(a.target_terms)
        counts = Counter()
        for tok in a_terms:
            for oid in inv.get(tok, []):
                if oid != tid:
                    counts[oid] += 1
        # Favor hard natural negatives: overlap enough to make target priors and lexical
        # plausibility comparable, but not identical target strings or same source row.
        for oid, shared in counts.most_common(80):
            b = transitions[oid]
            if a.row_i == b.row_i:
                continue
            key = tuple(sorted((a.tid, b.tid)))
            if key in seen:
                continue
            if a.target.strip().lower() == b.target.strip().lower():
                continue
            len_ratio = max(a.target_words, b.target_words) / max(1, min(a.target_words, b.target_words))
            if len_ratio > 1.8:
                continue
            b_terms = set(b.target_terms)
            target_j = jaccard(a_terms, b_terms)
            if shared < 2 or target_j < 0.14:
                continue
            context_j = jaccard(set(a.context_terms), set(b.context_terms))
            # Avoid almost duplicate contexts/targets. We want two natural contexts, not copies.
            if context_j > 0.8 or target_j > 0.85:
                continue
            spair = tuple(sorted((a.source, b.source)))
            if source_pair_counts[spair] >= per_source_pair_cap:
                continue
            seen.add(key)
            source_pair_counts[spair] += 1
            candidates.append({
                'pair_id': len(candidates),
                't1_id': a.tid,
                't2_id': b.tid,
                'source1': a.source,
                'source2': b.source,
                'row1': a.row_i,
                'row2': b.row_i,
                'sent1': a.sent_i,
                'sent2': b.sent_i,
                'context1_words': a.context_words,
                'context2_words': b.context_words,
                'target1_words': a.target_words,
                'target2_words': b.target_words,
                'shared_target_terms': ' '.join(sorted(a_terms & b_terms)),
                'shared_target_terms_n': shared,
                'target_jaccard': target_j,
                'context_jaccard': context_j,
                'context1': a.context,
                'target1': a.target,
                'context2': b.context,
                'target2': b.target,
            })
            if len(candidates) >= pair_limit:
                return candidates
    return candidates


@dataclass
class ScoreTask:
    rec_i: int
    key: str
    sentence: str
    start_boundary: int


class CompletionScorer:
    def __init__(self, model, tokenizer, device: torch.device, masked_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.masked_batch_size = masked_batch_size
        self.mask_id = tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError('Tokenizer has no mask_token_id')

    def score(self, tasks: list[ScoreTask]) -> dict[tuple[int, str], dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        acc: dict[tuple[int, str], dict[str, Any]] = {}
        for task in tasks:
            sid = (task.rec_i, task.key)
            acc[sid] = {'sum': 0.0, 'n_tokens': 0, 'status': 'ok'}
            enc = self.tokenizer(task.sentence, return_offsets_mapping=True, return_tensors=None)
            ids = list(enc['input_ids']); attn = list(enc['attention_mask']); offs = list(enc['offset_mapping'])
            selected = [i for i, (a, b) in enumerate(offs) if b > task.start_boundary]
            if not selected:
                acc[sid]['status'] = 'no_tokens'
                continue
            for pos in selected:
                cur = list(ids); cur[pos] = self.mask_id
                examples.append({'sid': sid, 'input_ids': cur, 'attn': attn, 'pos': pos, 'target': ids[pos], 'length': len(cur)})
        examples.sort(key=lambda x: x['length'])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        with torch.no_grad():
            for start in range(0, len(examples), self.masked_batch_size):
                batch = examples[start:start + self.masked_batch_size]
                max_len = max(ex['length'] for ex in batch)
                inp=[]; att=[]; pos=[]; tar=[]; sids=[]
                for ex in batch:
                    pad_n = max_len - ex['length']
                    inp.append(ex['input_ids'] + [pad_id]*pad_n)
                    att.append(ex['attn'] + [0]*pad_n)
                    pos.append(ex['pos']); tar.append(ex['target']); sids.append(ex['sid'])
                input_ids = torch.tensor(inp, dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(att, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos, dtype=torch.long, device=self.device)
                tar_t = torch.tensor(tar, dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                vals = torch.gather(F.log_softmax(logits[mb, pos_t], dim=-1), -1, tar_t.unsqueeze(-1)).squeeze(-1)
                for sid, val in zip(sids, vals.detach().cpu().tolist()):
                    acc[sid]['sum'] += float(val); acc[sid]['n_tokens'] += 1
        for d in acc.values():
            d['mean'] = d['sum'] / d['n_tokens'] if d['n_tokens'] else float('nan')
        return acc


def completion_task(i: int, key: str, context: str, target: str) -> ScoreTask:
    context = context.rstrip(); target = target.strip()
    sentence = (context + ' ' + target).strip()
    completion = ' ' + target
    return ScoreTask(i, key, sentence, len(sentence) - len(completion))


def safe_float(x: Any) -> float:
    try:
        y = float(x); return y if math.isfinite(y) else float('nan')
    except Exception:
        return float('nan')


def add_scores_to_pair(rec: dict[str, Any], scores: dict[tuple[int, str], dict[str, Any]], rec_i: int) -> None:
    for key in ['s11','s12','s21','s22','p1','p2']:
        d = scores.get((rec_i, key), {'sum': float('nan'), 'mean': float('nan'), 'n_tokens': 0})
        rec[f'{key}_sum'] = d.get('sum', float('nan'))
        rec[f'{key}_mean'] = d.get('mean', float('nan'))
        rec[f'{key}_n_tokens'] = d.get('n_tokens', 0)
    for form in ['sum', 'mean']:
        s11=safe_float(rec[f's11_{form}']); s12=safe_float(rec[f's12_{form}']); s21=safe_float(rec[f's21_{form}']); s22=safe_float(rec[f's22_{form}'])
        p1=safe_float(rec[f'p1_{form}']); p2=safe_float(rec[f'p2_{form}'])
        rec[f'c1_target_margin_{form}'] = s11 - s12
        rec[f'c2_target_margin_{form}'] = s22 - s21
        rec[f'interaction_{form}'] = (s11 + s22) - (s12 + s21)
        rec[f'pmi_c1_target_margin_{form}'] = (s11 - p1) - (s12 - p2)
        rec[f'pmi_c2_target_margin_{form}'] = (s22 - p2) - (s21 - p1)
    rec['both_contexts_choose_observed_sum'] = rec['c1_target_margin_sum'] > 0 and rec['c2_target_margin_sum'] > 0
    rec['both_contexts_choose_observed_mean'] = rec['c1_target_margin_mean'] > 0 and rec['c2_target_margin_mean'] > 0
    rec['interaction_sum_positive'] = rec['interaction_sum'] > 0
    rec['interaction_mean_positive'] = rec['interaction_mean'] > 0
    rec['natural_interaction_failure_stable'] = (not rec['both_contexts_choose_observed_sum']) and rec['interaction_sum'] <= 0 and rec['interaction_mean'] <= 0


def frac(rows: list[dict[str, Any]], pred) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if pred(r)) / len(rows)


def vals(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [safe_float(r.get(field)) for r in rows if math.isfinite(safe_float(r.get(field)))]


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    out = {'label': label, 'n': len(rows)}
    if not rows:
        return out
    for f in ['target_jaccard','context_jaccard','c1_target_margin_sum','c2_target_margin_sum','interaction_sum','interaction_mean','pmi_c1_target_margin_sum','pmi_c2_target_margin_sum']:
        vs = vals(rows, f)
        out[f + '_mean'] = statistics.fmean(vs) if vs else None
        out[f + '_median'] = statistics.median(vs) if vs else None
        out[f + '_pos_frac'] = sum(v > 0 for v in vs) / len(vs) if vs else None
    out['both_contexts_choose_observed_sum_frac'] = frac(rows, lambda r: r['both_contexts_choose_observed_sum'])
    out['both_contexts_choose_observed_mean_frac'] = frac(rows, lambda r: r['both_contexts_choose_observed_mean'])
    out['stable_natural_interaction_failure_frac'] = frac(rows, lambda r: r['natural_interaction_failure_stable'])
    out['stable_natural_interaction_failure_count'] = sum(1 for r in rows if r['natural_interaction_failure_stable'])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_rows_scan', type=int, default=12000)
    ap.add_argument('--max_transitions', type=int, default=30000)
    ap.add_argument('--pair_limit', type=int, default=300)
    ap.add_argument('--score_pair_limit', type=int, default=0, help='0 skips model scoring')
    ap.add_argument('--seed', type=int, default=92092)
    ap.add_argument('--models', nargs='*', default=['legal40_depth_12x384_43022','legal40_8x480_43022'], choices=list(DEFAULT_MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--masked_batch_size', type=int, default=96)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    transitions, scan_meta = load_transitions(args.max_rows_scan, args.max_transitions, args.seed)
    pairs = make_pairs(transitions, args.pair_limit, args.seed)
    candidates_path = OUT / 'natural_two_context_target_candidates.csv'
    scored_path = OUT / 'natural_two_context_target_scored_pairs.csv'
    summary_path = OUT / 'natural_two_context_target_feasibility.json'
    write_csv(candidates_path, pairs)

    all_scored: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    if args.score_pair_limit:
        score_pairs = pairs[:args.score_pair_limit]
        device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        for model_label in args.models:
            model_path = DEFAULT_MODELS[model_label]
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
            model.eval().to(device)
            scorer = CompletionScorer(model, tokenizer, device, args.masked_batch_size)
            tasks=[]
            for i, pair in enumerate(score_pairs):
                tasks.extend([
                    completion_task(i, 's11', pair['context1'], pair['target1']),
                    completion_task(i, 's12', pair['context1'], pair['target2']),
                    completion_task(i, 's21', pair['context2'], pair['target1']),
                    completion_task(i, 's22', pair['context2'], pair['target2']),
                    completion_task(i, 'p1', '', pair['target1']),
                    completion_task(i, 'p2', '', pair['target2']),
                ])
            scores = scorer.score(tasks)
            rows=[]
            for i, pair in enumerate(score_pairs):
                rec = dict(pair)
                rec['model'] = model_label
                add_scores_to_pair(rec, scores, i)
                rows.append(rec)
            all_scored.extend(rows)
            model_summaries[model_label] = {
                'model_path': str(model_path),
                'n_pairs_scored': len(rows),
                'device': str(device),
                'summary': summarize(rows, 'all_scored_pairs'),
                'by_source_pair': {sp: summarize(v, sp) for sp, v in sorted(group_rows(rows, 'source_pair_label').items())} if False else {},
            }
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        write_csv(scored_path, all_scored)
    else:
        device = 'not_used'
        write_csv(scored_path, [])

    pairwise = {}
    if all_scored and len(args.models) == 2:
        a,b = args.models
        ra={int(r['pair_id']): r for r in all_scored if r['model']==a}
        rb={int(r['pair_id']): r for r in all_scored if r['model']==b}
        common=sorted(set(ra)&set(rb))
        both_fail=[i for i in common if ra[i]['natural_interaction_failure_stable'] and rb[i]['natural_interaction_failure_stable']]
        both_choose=[i for i in common if ra[i]['both_contexts_choose_observed_sum'] and rb[i]['both_contexts_choose_observed_sum']]
        pairwise={
            'models':[a,b],
            'common_pairs':len(common),
            'both_stable_natural_interaction_failure_count':len(both_fail),
            'both_stable_natural_interaction_failure_frac':len(both_fail)/len(common) if common else None,
            'both_models_choose_observed_count':len(both_choose),
            'both_models_choose_observed_frac':len(both_choose)/len(common) if common else None,
            'first_40_both_failure_pair_ids':both_fail[:40],
        }

    payload = {
        'status': 'NATURAL_TWO_CONTEXT_TARGET_FEASIBILITY',
        'created_utc': now_utc(),
        'purpose': 'Low-cost corpus-derived two-context/two-target feasibility probe for an interaction-ranking objective independent of EWoK wording and relation inventories.',
        'script_never_trains': True,
        'script_never_runs_official_eval': True,
        'corpus_path': str(CORPUS),
        'scan_meta': scan_meta,
        'candidate_pairs': len(pairs),
        'score_pair_limit': args.score_pair_limit,
        'device_used': str(device),
        'models': args.models if args.score_pair_limit else [],
        'model_summaries': model_summaries,
        'pairwise_two_model_summary': pairwise,
        'candidates_csv': str(candidates_path),
        'scored_pairs_csv': str(scored_path),
        'json_output': str(summary_path),
        'note': str(NOTE),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    lines=[]
    lines.append('# research — natural two-context/two-target feasibility\n\n')
    lines.append('This CPU-only probe mines natural adjacent-sentence transitions from the actual compact-view-reinvest 10M corpus and pairs targets with overlapping content. It avoids EWoK wording and relation inventories and does not train.\n\n')
    lines.append(f"Scanned `{scan_meta['rows_seen']}` rows and `{scan_meta['sentence_fragments']}` sentence fragments; kept `{scan_meta['transitions']}` transitions and `{len(pairs)}` candidate natural two-context/two-target pairs.\n\n")
    if model_summaries:
        lines.append('| model | pairs scored | both contexts choose observed | stable natural interaction failure | interaction median |\n')
        lines.append('|---|---:|---:|---:|---:|\n')
        for model_label, summ in model_summaries.items():
            s=summ['summary']
            lines.append(f"| {model_label} | {s['n']} | {s.get('both_contexts_choose_observed_sum_frac')} | {s.get('stable_natural_interaction_failure_frac')} | {s.get('interaction_sum_median')} |\n")
        lines.append('\n')
    if pairwise:
        lines.append(f"Both-model stable natural interaction failures: `{pairwise['both_stable_natural_interaction_failure_count']}` / `{pairwise['common_pairs']}` ({pairwise['both_stable_natural_interaction_failure_frac']}).\n\n")
    lines.append('Interpretation: if the candidate natural pairs are mostly already solved, then a broad natural sentence-transition objective is redundant. If a sizable stable failure population exists after quality inspection, a subsequent comparison could use a stricter interaction-ranking objective and accounting.\n\n')
    lines.append(f"JSON: `{summary_path}`\n\nCandidates: `{candidates_path}`\n\nScored pairs: `{scored_path}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')

    compact = {
        'status': payload['status'],
        'scan_rows': scan_meta['rows_seen'],
        'transitions': scan_meta['transitions'],
        'candidate_pairs': len(pairs),
        'score_pair_limit': args.score_pair_limit,
        'pairwise_two_model_summary': pairwise,
        'out_json': str(summary_path),
        'note': str(NOTE),
        'summary': {m: {
            'pairs_scored': s['summary'].get('n'),
            'both_choose_observed': s['summary'].get('both_contexts_choose_observed_sum_frac'),
            'stable_failure_frac': s['summary'].get('stable_natural_interaction_failure_frac'),
            'interaction_median': s['summary'].get('interaction_sum_median'),
        } for m,s in model_summaries.items()},
    }
    print(json.dumps(compact, indent=2), flush=True)


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out=defaultdict(list)
    for r in rows:
        out[str(r.get(key,''))].append(r)
    return out


if __name__ == '__main__':
    main()
