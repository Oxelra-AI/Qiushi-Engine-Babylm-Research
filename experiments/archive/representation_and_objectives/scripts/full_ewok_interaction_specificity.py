#!/usr/bin/env python3
"""research full EWoK interaction-specificity measurement.

This script uses existing checkpoints only. It scores all (or a bounded prefix of)
BabyLM EWoK rows with four context-target combinations and simple removal of the
context-difference span. It is designed to answer whether the research/91 surviving
object is real: conditional target reversal failure, where changing context does
not reorder the two target alternatives even after target priors cancel.

No training is performed. No official benchmark run is performed.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path('experiments/archive/representation_and_objectives')
WS = ROOT
EVAL_EWOK = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'
ITEM_CSV = WS / 'data/ewok_contrast_preservation_anatomy/ewok_item_correctness_and_coverage.csv'
OUT = WS / 'data/full_ewok_interaction_specificity'
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/full_ewok_interaction_specificity.md')

DEFAULT_MODELS = {
    'legal40_depth_12x384_43022': WS / 'training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M',
    'legal40_8x480_43022': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M',
}
MODEL_CORRECT_COLUMNS = {
    'legal40_depth_12x384_43022': 'legal40_12x384_depth_43022',
    'legal40_8x480_43022': 'legal40_8x480_43022',
}

WORD_RE = re.compile(r"\S+")
PUNCT_STRIP = string.punctuation + "“”‘’«»‹›"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {'1', 'true', 'yes'}


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_float(x: Any, default: float = float('nan')) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def sign3(x: float, eps: float = 0.0) -> int:
    if not math.isfinite(x):
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    flags: dict[int, dict[str, Any]] = {}
    with ITEM_CSV.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            flags[int(row['global_index'])] = row
    return flags


@dataclass(frozen=True)
class WordTok:
    text: str
    norm: str
    start: int
    end: int


def word_toks(text: str) -> list[WordTok]:
    toks = []
    for m in WORD_RE.finditer(text):
        raw = m.group(0)
        norm = raw.lower().strip(PUNCT_STRIP)
        toks.append(WordTok(raw, norm, m.start(), m.end()))
    return toks


def merged_span(toks: list[WordTok], i1: int, i2: int) -> tuple[int, int] | None:
    if i1 >= i2:
        return None
    return toks[i1].start, toks[i2 - 1].end


def diff_spans(c1: str, c2: str) -> dict[str, Any]:
    """Return approximate non-equal whitespace-token spans in both contexts."""
    t1 = word_toks(c1)
    t2 = word_toks(c2)
    sm = difflib.SequenceMatcher(a=[t.norm for t in t1], b=[t.norm for t in t2], autojunk=False)
    c1_spans: list[tuple[int, int]] = []
    c2_spans: list[tuple[int, int]] = []
    c1_texts: list[str] = []
    c2_texts: list[str] = []
    opcodes: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        s1 = merged_span(t1, i1, i2)
        s2 = merged_span(t2, j1, j2)
        txt1 = c1[s1[0]:s1[1]] if s1 else ''
        txt2 = c2[s2[0]:s2[1]] if s2 else ''
        if s1:
            c1_spans.append(s1); c1_texts.append(txt1)
        if s2:
            c2_spans.append(s2); c2_texts.append(txt2)
        opcodes.append({'tag': tag, 'c1_word_span': [i1, i2], 'c2_word_span': [j1, j2], 'c1_text': txt1, 'c2_text': txt2})
    return {
        'c1_spans': c1_spans,
        'c2_spans': c2_spans,
        'c1_texts': c1_texts,
        'c2_texts': c2_texts,
        'n_hunks': len(opcodes),
        'opcodes': opcodes,
    }


def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    parts = []
    last = 0
    for s, e in spans:
        if s > last:
            parts.append(text[last:s])
        last = max(last, e)
    parts.append(text[last:])
    return normalize_spaces(''.join(parts))


def replace_spans(text: str, spans: list[tuple[int, int]], repl_texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Replace spans simultaneously; return new text and spans of inserted material."""
    if not spans or len(spans) != len(repl_texts):
        return text, []
    pairs = sorted(zip(spans, repl_texts), key=lambda x: x[0][0])
    out_parts = []
    inserted: list[tuple[int, int]] = []
    last = 0
    cur = 0
    for (s, e), repl in pairs:
        prefix = text[last:s]
        out_parts.append(prefix)
        cur += len(prefix)
        ins_start = cur
        out_parts.append(repl)
        cur += len(repl)
        ins_end = cur
        if repl:
            inserted.append((ins_start, ins_end))
        last = e
    suffix = text[last:]
    out_parts.append(suffix)
    new_text = ''.join(out_parts)
    return new_text, inserted


@dataclass
class ScoreTask:
    rec_i: int
    key: str
    sentence: str
    spans: list[tuple[int, int]]
    mode: str = 'span'  # span or completion


class BatchedPseudoScorer:
    def __init__(self, model, tokenizer, device: torch.device, masked_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.masked_batch_size = masked_batch_size
        self.mask_id = tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError('Tokenizer has no mask_token_id')

    def _build_examples(self, tasks: list[ScoreTask]) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
        examples: list[dict[str, Any]] = []
        acc: dict[tuple[int, str], dict[str, Any]] = {}
        for task in tasks:
            score_id = (task.rec_i, task.key)
            if score_id not in acc:
                acc[score_id] = {'sum': 0.0, 'n_tokens': 0, 'status': 'ok'}
            if not task.sentence or not task.spans:
                acc[score_id]['status'] = 'empty'
                continue
            enc = self.tokenizer(task.sentence, return_offsets_mapping=True, return_tensors=None)
            token_ids = list(enc['input_ids'])
            attention = list(enc['attention_mask'])
            offsets = list(enc['offset_mapping'])
            selected: list[int] = []
            if task.mode == 'completion':
                # The span start is the official completion boundary. The official MLM code
                # selects tokens whose offset end is beyond that boundary.
                start_boundary = min(s for s, _ in task.spans)
                for pos, (a, b) in enumerate(offsets):
                    if b > start_boundary:
                        selected.append(pos)
            else:
                for pos, (a, b) in enumerate(offsets):
                    if any((b > s and a < e) for s, e in task.spans):
                        selected.append(pos)
            if not selected:
                acc[score_id]['status'] = 'no_tokens'
                continue
            for pos in selected:
                cur = list(token_ids)
                cur[pos] = self.mask_id
                examples.append({
                    'score_id': score_id,
                    'input_ids': cur,
                    'attention_mask': attention,
                    'pos': pos,
                    'target_id': token_ids[pos],
                    'length': len(cur),
                })
        return examples, acc

    def score(self, tasks: list[ScoreTask]) -> dict[tuple[int, str], dict[str, Any]]:
        examples, acc = self._build_examples(tasks)
        examples.sort(key=lambda x: x['length'])
        with torch.no_grad():
            for start in range(0, len(examples), self.masked_batch_size):
                batch = examples[start:start + self.masked_batch_size]
                max_len = max(ex['length'] for ex in batch)
                input_rows = []
                attn_rows = []
                positions = []
                targets = []
                score_ids = []
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 0
                for ex in batch:
                    pad_n = max_len - ex['length']
                    input_rows.append(ex['input_ids'] + [pad_id] * pad_n)
                    attn_rows.append(ex['attention_mask'] + [0] * pad_n)
                    positions.append(ex['pos'])
                    targets.append(ex['target_id'])
                    score_ids.append(ex['score_id'])
                input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(attn_rows, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
                target_t = torch.tensor(targets, dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                masked = logits[mb, pos_t]
                vals = torch.gather(F.log_softmax(masked, dim=-1), -1, target_t.unsqueeze(-1)).squeeze(-1)
                for score_id, val in zip(score_ids, vals.detach().cpu().tolist()):
                    acc[score_id]['sum'] += float(val)
                    acc[score_id]['n_tokens'] += 1
        for score_id, d in acc.items():
            n = d['n_tokens']
            d['mean'] = d['sum'] / n if n else float('nan')
        return acc


def completion_task(rec_i: int, key: str, context: str, target: str) -> ScoreTask:
    context = context.rstrip()
    target = target.strip()
    sentence = (context + ' ' + target).strip() if context else target
    completion = ' ' + target
    start_char_idx = len(sentence) - len(completion)
    # Match the official EWoK MLM completion-token selection. For empty context,
    # this boundary is -1 and thus selects the whole target sentence.
    return ScoreTask(rec_i=rec_i, key=key, sentence=sentence, spans=[(start_char_idx, len(sentence))], mode='completion')


def span_task(rec_i: int, key: str, sentence: str, spans: list[tuple[int, int]]) -> ScoreTask:
    return ScoreTask(rec_i=rec_i, key=key, sentence=sentence, spans=spans, mode='span')


def make_base_record(row: dict[str, Any], flags: dict[str, Any]) -> dict[str, Any]:
    return {
        'global_index': row['_global_index'],
        'domain': row['_domain'],
        'local_index': row['_local_index'],
        'ContextType': row.get('ContextType'),
        'ContextDiff': row.get('ContextDiff'),
        'TargetDiff': row.get('TargetDiff'),
        'ConceptA': row.get('ConceptA'),
        'ConceptB': row.get('ConceptB'),
        'context_diff_words': flags.get('context_diff_words', ''),
        'target_diff_words': flags.get('target_diff_words', ''),
        'all_legal_wrong_flag': truthy(flags.get('all_legal_wrong')),
        'all_legal_right_flag': truthy(flags.get('all_legal_right')),
        'depth_wrong_legal40_wrong_flag': truthy(flags.get('depth_wrong_legal40_wrong')),
        'inherited43022_right_but_legal40_depth_wrong_flag': truthy(flags.get('inherited43022_right_but_legal40_depth_wrong')),
        'legal40_8x480_correct_flag': truthy(flags.get('legal40_8x480_43022')),
        'depth_correct_flag': truthy(flags.get('legal40_12x384_depth_43022')),
        'pattern': flags.get('pattern', ''),
    }


def add_derived_scores(rec: dict[str, Any], correct_col: str) -> None:
    def g(key: str, form: str = 'sum') -> float:
        return safe_float(rec.get(f'{key}_{form}'))

    for form in ['sum', 'mean']:
        s11 = g('s11', form); s21 = g('s21', form); s12 = g('s12', form); s22 = g('s22', form)
        d11 = g('del_s11', form); d21 = g('del_s21', form); d12 = g('del_s12', form); d22 = g('del_s22', form)
        p1 = g('prior_t1', form); p2 = g('prior_t2', form)
        rec[f'official_margin_t1_{form}'] = s11 - s21
        rec[f'official_margin_t2_{form}'] = s22 - s12
        rec[f'within_context_margin_c1_{form}'] = s11 - s12
        rec[f'within_context_margin_c2_{form}'] = s22 - s21
        rec[f'interaction_{form}'] = (s11 + s22) - (s12 + s21)
        rec[f'deletion_official_margin_t1_{form}'] = d11 - d21
        rec[f'deletion_official_margin_t2_{form}'] = d22 - d12
        rec[f'deletion_within_context_margin_c1_{form}'] = d11 - d12
        rec[f'deletion_within_context_margin_c2_{form}'] = d22 - d21
        rec[f'deletion_interaction_{form}'] = (d11 + d22) - (d12 + d21)
        rec[f'interaction_minus_deletion_{form}'] = rec[f'interaction_{form}'] - rec[f'deletion_interaction_{form}']
        # Target priors cancel for the interaction, but they change within-context margins.
        rec[f'pmi_within_context_margin_c1_{form}'] = (s11 - p1) - (s12 - p2)
        rec[f'pmi_within_context_margin_c2_{form}'] = (s22 - p2) - (s21 - p1)
    rec['saved_model_correct_flag'] = truthy(rec.get(correct_col)) if correct_col in rec else (rec['depth_correct_flag'] if correct_col == 'legal40_12x384_depth_43022' else rec['legal40_8x480_correct_flag'])
    rec['saved_model_wrong_flag'] = not rec['saved_model_correct_flag']
    rec['official_t1_sum_sign_matches_saved'] = (rec['official_margin_t1_sum'] > 0) == rec['saved_model_correct_flag']
    rec['both_official_sum_positive'] = (rec['official_margin_t1_sum'] > 0 and rec['official_margin_t2_sum'] > 0)
    rec['both_within_context_sum_positive'] = (rec['within_context_margin_c1_sum'] > 0 and rec['within_context_margin_c2_sum'] > 0)
    rec['both_within_context_mean_positive'] = (rec['within_context_margin_c1_mean'] > 0 and rec['within_context_margin_c2_mean'] > 0)
    rec['interaction_sum_positive'] = rec['interaction_sum'] > 0
    rec['interaction_mean_positive'] = rec['interaction_mean'] > 0
    rec['interaction_sign_sum_mean_agree'] = sign3(rec['interaction_sum']) == sign3(rec['interaction_mean'])
    rec['stable_nonpositive_interaction'] = rec['interaction_sum'] <= 0 and rec['interaction_mean'] <= 0
    rec['weak_abs_interaction_sum_le_0p25'] = abs(rec['interaction_sum']) <= 0.25
    rec['weak_abs_interaction_mean_le_0p05'] = abs(rec['interaction_mean']) <= 0.05
    rec['conditional_reversal_failure_sum'] = bool(rec['saved_model_wrong_flag'] and rec['interaction_sum'] <= 0 and not rec['both_within_context_sum_positive'])
    rec['conditional_reversal_failure_stable'] = bool(rec['saved_model_wrong_flag'] and rec['stable_nonpositive_interaction'] and not rec['both_within_context_sum_positive'] and not rec['both_within_context_mean_positive'])
    if finite(rec.get('local_c1_actual_over_swapped_sum')) and finite(rec.get('local_c2_actual_over_swapped_sum')):
        rec['local_both_actual_over_swapped_positive'] = rec['local_c1_actual_over_swapped_sum'] > 0 and rec['local_c2_actual_over_swapped_sum'] > 0
        rec['local_min_actual_over_swapped_sum'] = min(rec['local_c1_actual_over_swapped_sum'], rec['local_c2_actual_over_swapped_sum'])
        rec['local_abs_preference_sum'] = abs(rec['local_c1_actual_over_swapped_sum'])
    else:
        rec['local_both_actual_over_swapped_positive'] = False
        rec['local_min_actual_over_swapped_sum'] = float('nan')
        rec['local_abs_preference_sum'] = float('nan')


def vals(records: list[dict[str, Any]], field: str) -> list[float]:
    return [safe_float(r.get(field)) for r in records if finite(r.get(field))]


def frac(records: list[dict[str, Any]], pred) -> float | None:
    if not records:
        return None
    return sum(1 for r in records if pred(r)) / len(records)


def summarize(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    out: dict[str, Any] = {'label': label, 'n': len(records)}
    if not records:
        return out
    numeric_fields = [
        'official_margin_t1_sum', 'official_margin_t2_sum', 'within_context_margin_c1_sum', 'within_context_margin_c2_sum', 'interaction_sum',
        'official_margin_t1_mean', 'official_margin_t2_mean', 'within_context_margin_c1_mean', 'within_context_margin_c2_mean', 'interaction_mean',
        'deletion_interaction_sum', 'deletion_interaction_mean', 'interaction_minus_deletion_sum', 'interaction_minus_deletion_mean',
        'pmi_within_context_margin_c1_sum', 'pmi_within_context_margin_c2_sum', 'local_c1_actual_over_swapped_sum', 'local_c2_actual_over_swapped_sum',
        'local_min_actual_over_swapped_sum', 'local_abs_preference_sum',
    ]
    for field in numeric_fields:
        vs = vals(records, field)
        if vs:
            out[field + '_mean'] = statistics.fmean(vs)
            out[field + '_median'] = statistics.median(vs)
            out[field + '_pos_frac'] = sum(v > 0 for v in vs) / len(vs)
            out[field + '_near0_abs_le_0p25_frac'] = sum(abs(v) <= 0.25 for v in vs) / len(vs)
        else:
            out[field + '_mean'] = None
            out[field + '_median'] = None
            out[field + '_pos_frac'] = None
            out[field + '_near0_abs_le_0p25_frac'] = None
    out.update({
        'saved_model_accuracy': frac(records, lambda r: r['saved_model_correct_flag']),
        'both_official_sum_positive_frac': frac(records, lambda r: r['both_official_sum_positive']),
        'both_within_context_sum_positive_frac': frac(records, lambda r: r['both_within_context_sum_positive']),
        'both_within_context_mean_positive_frac': frac(records, lambda r: r['both_within_context_mean_positive']),
        'interaction_sum_positive_frac': frac(records, lambda r: r['interaction_sum_positive']),
        'interaction_mean_positive_frac': frac(records, lambda r: r['interaction_mean_positive']),
        'interaction_sum_mean_sign_agree_frac': frac(records, lambda r: r['interaction_sign_sum_mean_agree']),
        'stable_nonpositive_interaction_frac': frac(records, lambda r: r['stable_nonpositive_interaction']),
        'conditional_reversal_failure_sum_frac': frac(records, lambda r: r['conditional_reversal_failure_sum']),
        'conditional_reversal_failure_stable_frac': frac(records, lambda r: r['conditional_reversal_failure_stable']),
        'local_both_actual_over_swapped_positive_frac': frac(records, lambda r: r['local_both_actual_over_swapped_positive']),
        'official_t1_sign_match_frac': frac(records, lambda r: r['official_t1_sum_sign_matches_saved']),
    })
    out['conditional_reversal_failure_stable_count'] = sum(1 for r in records if r['conditional_reversal_failure_stable'])
    out['saved_wrong_count'] = sum(1 for r in records if r['saved_model_wrong_flag'])
    out['stable_failure_fraction_among_saved_wrong'] = (out['conditional_reversal_failure_stable_count'] / out['saved_wrong_count']) if out['saved_wrong_count'] else None
    return out


def summarize_by(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get(field, ''))].append(r)
    return {k: summarize(v, k) for k, v in sorted(groups.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_group_csv(path: Path, model_summaries: dict[str, Any], group_key: str) -> None:
    rows: list[dict[str, Any]] = []
    for model_label, summ in model_summaries.items():
        for group, d in summ[group_key].items():
            row = {'model': model_label, group_key: group}
            for k in [
                'n', 'saved_model_accuracy', 'interaction_sum_median', 'interaction_sum_positive_frac', 'interaction_mean_median', 'interaction_mean_positive_frac',
                'both_within_context_sum_positive_frac', 'stable_nonpositive_interaction_frac', 'conditional_reversal_failure_stable_count',
                'saved_wrong_count', 'stable_failure_fraction_among_saved_wrong', 'deletion_interaction_sum_median', 'interaction_minus_deletion_sum_median',
                'local_min_actual_over_swapped_sum_median', 'local_both_actual_over_swapped_positive_frac', 'official_t1_sign_match_frac'
            ]:
                row[k] = d.get(k)
            rows.append(row)
    write_csv(path, rows)


def build_batch_records(rows: list[dict[str, Any]], flags_map: dict[int, dict[str, Any]], indices: list[int]) -> tuple[list[dict[str, Any]], list[ScoreTask]]:
    recs: list[dict[str, Any]] = []
    tasks: list[ScoreTask] = []
    for rec_i, idx in enumerate(indices):
        row = rows[idx]
        flags = flags_map[idx]
        rec = make_base_record(row, flags)
        c1, c2 = row['Context1'], row['Context2']
        t1, t2 = row['Target1'], row['Target2']
        ds = diff_spans(c1, c2)
        rec['context_diff_n_hunks'] = ds['n_hunks']
        rec['context_diff_c1_texts_joined'] = ' ||| '.join(ds['c1_texts'])
        rec['context_diff_c2_texts_joined'] = ' ||| '.join(ds['c2_texts'])
        c1_del = delete_spans(c1, ds['c1_spans'])
        c2_del = delete_spans(c2, ds['c2_spans'])
        rec['context1_deleted_diff'] = c1_del
        rec['context2_deleted_diff'] = c2_del
        rec['deleted_contexts_identical'] = normalize_spaces(c1_del).lower() == normalize_spaces(c2_del).lower()
        recs.append(rec)
        # Full four-cell target scores plus empty-context target scores.
        tasks.extend([
            completion_task(rec_i, 's11', c1, t1), completion_task(rec_i, 's21', c2, t1),
            completion_task(rec_i, 's12', c1, t2), completion_task(rec_i, 's22', c2, t2),
            completion_task(rec_i, 'prior_t1', '', t1), completion_task(rec_i, 'prior_t2', '', t2),
            completion_task(rec_i, 'del_s11', c1_del, t1), completion_task(rec_i, 'del_s21', c2_del, t1),
            completion_task(rec_i, 'del_s12', c1_del, t2), completion_task(rec_i, 'del_s22', c2_del, t2),
        ])
        # Approximate local context-difference span scoring. This is not an EWoK target
        # score; it asks whether the model strongly prefers the actual diff span over
        # the counterpart span in the corresponding context frame.
        if ds['c1_spans'] and ds['c2_spans'] and len(ds['c1_spans']) == len(ds['c2_texts']) and len(ds['c2_spans']) == len(ds['c1_texts']):
            c1_swapped, c1_inserted = replace_spans(c1, ds['c1_spans'], ds['c2_texts'])
            c2_swapped, c2_inserted = replace_spans(c2, ds['c2_spans'], ds['c1_texts'])
            rec['local_context_status'] = 'scored'
            rec['context1_swapped_diff'] = c1_swapped
            rec['context2_swapped_diff'] = c2_swapped
            tasks.extend([
                span_task(rec_i, 'local_c1_actual', c1, ds['c1_spans']),
                span_task(rec_i, 'local_c1_swapped', c1_swapped, c1_inserted),
                span_task(rec_i, 'local_c2_actual', c2, ds['c2_spans']),
                span_task(rec_i, 'local_c2_swapped', c2_swapped, c2_inserted),
            ])
        else:
            rec['local_context_status'] = 'not_scored'
    return recs, tasks


def attach_scores(recs: list[dict[str, Any]], scores: dict[tuple[int, str], dict[str, Any]], correct_col: str) -> None:
    score_keys = ['s11', 's21', 's12', 's22', 'prior_t1', 'prior_t2', 'del_s11', 'del_s21', 'del_s12', 'del_s22', 'local_c1_actual', 'local_c1_swapped', 'local_c2_actual', 'local_c2_swapped']
    for rec_i, rec in enumerate(recs):
        for key in score_keys:
            d = scores.get((rec_i, key), {'sum': float('nan'), 'mean': float('nan'), 'n_tokens': 0, 'status': 'missing'})
            rec[f'{key}_sum'] = d.get('sum', float('nan'))
            rec[f'{key}_mean'] = d.get('mean', float('nan'))
            rec[f'{key}_n_tokens'] = d.get('n_tokens', 0)
            rec[f'{key}_status'] = d.get('status', 'missing')
        rec['local_c1_actual_over_swapped_sum'] = safe_float(rec.get('local_c1_actual_sum')) - safe_float(rec.get('local_c1_swapped_sum'))
        rec['local_c2_actual_over_swapped_sum'] = safe_float(rec.get('local_c2_actual_sum')) - safe_float(rec.get('local_c2_swapped_sum'))
        rec['local_c1_actual_over_swapped_mean'] = safe_float(rec.get('local_c1_actual_mean')) - safe_float(rec.get('local_c1_swapped_mean'))
        rec['local_c2_actual_over_swapped_mean'] = safe_float(rec.get('local_c2_actual_mean')) - safe_float(rec.get('local_c2_swapped_mean'))
        add_derived_scores(rec, correct_col)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=['legal40_depth_12x384_43022', 'legal40_8x480_43022'], choices=list(DEFAULT_MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--row_limit', type=int, default=0, help='0 means all EWoK rows')
    ap.add_argument('--row_offset', type=int, default=0)
    ap.add_argument('--row_batch_size', type=int, default=64)
    ap.add_argument('--masked_batch_size', type=int, default=128)
    ap.add_argument('--output_suffix', default='')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rows = load_rows()
    flags = load_item_flags()
    selected = list(range(len(rows)))
    if args.row_offset:
        selected = selected[args.row_offset:]
    if args.row_limit:
        selected = selected[:args.row_limit]
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    suffix = ('_' + args.output_suffix.strip('_')) if args.output_suffix else ''

    all_records: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    runtime_models: dict[str, Any] = {}
    for model_label in args.models:
        model_path = DEFAULT_MODELS[model_label]
        correct_col = MODEL_CORRECT_COLUMNS[model_label]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
        model.eval().to(device)
        scorer = BatchedPseudoScorer(model, tokenizer, device, args.masked_batch_size)
        model_records: list[dict[str, Any]] = []
        for b0 in range(0, len(selected), args.row_batch_size):
            idx_batch = selected[b0:b0 + args.row_batch_size]
            recs, tasks = build_batch_records(rows, flags, idx_batch)
            scores = scorer.score(tasks)
            attach_scores(recs, scores, correct_col)
            for rec in recs:
                rec['model'] = model_label
            model_records.extend(recs)
            if (b0 // args.row_batch_size) % 10 == 0:
                print(json.dumps({'model': model_label, 'rows_done': len(model_records), 'rows_total': len(selected), 'utc': now_utc()}), flush=True)
        all_records.extend(model_records)
        model_summaries[model_label] = {
            'model_path': str(model_path),
            'n_rows': len(model_records),
            'correct_column': correct_col,
            'device': str(device),
            'threads': args.threads,
            'masked_batch_size': args.masked_batch_size,
            'row_batch_size': args.row_batch_size,
            'all_rows': summarize(model_records, 'all_rows'),
            'saved_wrong_rows': summarize([r for r in model_records if r['saved_model_wrong_flag']], 'saved_wrong_rows'),
            'saved_correct_rows': summarize([r for r in model_records if r['saved_model_correct_flag']], 'saved_correct_rows'),
            'persistent_legal40_depth_wrong_rows': summarize([r for r in model_records if r['depth_wrong_legal40_wrong_flag']], 'persistent_legal40_depth_wrong_rows'),
            'all_legal_wrong_rows': summarize([r for r in model_records if r['all_legal_wrong_flag']], 'all_legal_wrong_rows'),
            'inherited43022_right_but_legal40_depth_wrong_rows': summarize([r for r in model_records if r['inherited43022_right_but_legal40_depth_wrong_flag']], 'inherited43022_right_but_legal40_depth_wrong_rows'),
            'by_domain': summarize_by(model_records, 'domain'),
            'by_context_type': summarize_by(model_records, 'ContextType'),
            'by_context_diff': summarize_by(model_records, 'ContextDiff'),
            'by_target_diff': summarize_by(model_records, 'TargetDiff'),
        }
        runtime_models[model_label] = {'loaded_path': str(model_path)}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    records_path = OUT / f'ewok_full_interaction_specificity_records{suffix}.csv'
    summary_path = OUT / f'ewok_full_interaction_specificity{suffix}.json'
    by_domain_path = OUT / f'ewok_full_interaction_specificity_by_domain{suffix}.csv'
    by_context_type_path = OUT / f'ewok_full_interaction_specificity_by_context_type{suffix}.csv'
    write_csv(records_path, all_records)
    write_group_csv(by_domain_path, model_summaries, 'by_domain')
    write_group_csv(by_context_type_path, model_summaries, 'by_context_type')

    pairwise: dict[str, Any] = {}
    if len(args.models) == 2:
        a, b = args.models
        ra = {int(r['global_index']): r for r in all_records if r['model'] == a}
        rb = {int(r['global_index']): r for r in all_records if r['model'] == b}
        common = sorted(set(ra) & set(rb))
        both_stable_fail = [i for i in common if ra[i]['conditional_reversal_failure_stable'] and rb[i]['conditional_reversal_failure_stable']]
        either_stable_fail = [i for i in common if ra[i]['conditional_reversal_failure_stable'] or rb[i]['conditional_reversal_failure_stable']]
        pairwise = {
            'models': [a, b],
            'common_rows': len(common),
            'both_conditional_reversal_failure_stable_count': len(both_stable_fail),
            'either_conditional_reversal_failure_stable_count': len(either_stable_fail),
            'both_conditional_reversal_failure_stable_frac': len(both_stable_fail) / len(common) if common else None,
            'either_conditional_reversal_failure_stable_frac': len(either_stable_fail) / len(common) if common else None,
            'first_50_both_stable_failure_indices': both_stable_fail[:50],
        }

    payload = {
        'status': 'FULL_EWOK_INTERACTION_SPECIFICITY',
        'created_utc': now_utc(),
        'purpose': 'Full-EWoK low-cost measurement of conditional target reversal and context-difference specificity before any new objective or training route.',
        'script_never_trains': True,
        'script_never_runs_official_eval': True,
        'rows_total_available': len(rows),
        'rows_scored_per_model': len(selected),
        'row_offset': args.row_offset,
        'row_limit': args.row_limit,
        'models': args.models,
        'device_requested': args.device,
        'device_used': str(device),
        'threads': args.threads,
        'masked_batch_size': args.masked_batch_size,
        'row_batch_size': args.row_batch_size,
        'model_summaries': model_summaries,
        'pairwise_two_model_summary': pairwise,
        'records_csv': str(records_path),
        'by_domain_csv': str(by_domain_path),
        'by_context_type_csv': str(by_context_type_path),
        'json_output': str(summary_path),
        'note': str(NOTE),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    # Human-readable note. Keep it short enough to read in later steps; details are in JSON/CSV.
    lines: list[str] = []
    lines.append('# research — full EWoK interaction-specificity measurement\n\n')
    lines.append('Existing checkpoints only; no training and no official benchmark run. The measurement asks whether EWoK errors are conditional target-reversal failures: changing context fails to reorder the two targets after target priors cancel.\n\n')
    lines.append(f"Rows scored per model: `{len(selected)}` of `{len(rows)}`. Device `{device}`, threads `{args.threads}`.\n\n")
    lines.append('| model | saved accuracy | saved wrong | stable conditional-reversal failures | frac among saved wrong | interaction median all | interaction median saved-wrong | both target margins positive saved-wrong | sign match official |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for model_label in args.models:
        summ = model_summaries[model_label]
        all_s = summ['all_rows']
        wrong_s = summ['saved_wrong_rows']
        lines.append(
            f"| {model_label} | {all_s.get('saved_model_accuracy')} | {wrong_s.get('n')} | {wrong_s.get('conditional_reversal_failure_stable_count')} | "
            f"{wrong_s.get('stable_failure_fraction_among_saved_wrong')} | {all_s.get('interaction_sum_median')} | {wrong_s.get('interaction_sum_median')} | "
            f"{wrong_s.get('both_within_context_sum_positive_frac')} | {all_s.get('official_t1_sign_match_frac')} |\n"
        )
    lines.append('\n')
    if pairwise:
        lines.append('Two-model overlap: stable conditional-reversal failure in both models on '
                     f"`{pairwise['both_conditional_reversal_failure_stable_count']}` / `{pairwise['common_rows']}` rows "
                     f"({pairwise['both_conditional_reversal_failure_stable_frac']}).\n\n")
    lines.append('Interpretation: a surviving future route would need corpus-derived two-context/two-target structures in which target priors and lexical plausibility cancel. Broad sentence-level replacement is not supported by this measurement alone.\n\n')
    lines.append(f"JSON: `{summary_path}`\n\nRecords CSV: `{records_path}`\n\nBy-domain CSV: `{by_domain_path}`\n\nBy-context-type CSV: `{by_context_type_path}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')

    compact = {
        'status': payload['status'],
        'rows_scored_per_model': len(selected),
        'device_used': str(device),
        'models': args.models,
        'pairwise_two_model_summary': pairwise,
        'out_json': str(summary_path),
        'records_csv': str(records_path),
        'note': str(NOTE),
        'summary': {
            m: {
                'accuracy': s['all_rows'].get('saved_model_accuracy'),
                'saved_wrong': s['saved_wrong_rows'].get('n'),
                'saved_wrong_interaction_median': s['saved_wrong_rows'].get('interaction_sum_median'),
                'saved_wrong_stable_failure_count': s['saved_wrong_rows'].get('conditional_reversal_failure_stable_count'),
                'saved_wrong_stable_failure_frac': s['saved_wrong_rows'].get('stable_failure_fraction_among_saved_wrong'),
                'saved_wrong_both_within_context_positive': s['saved_wrong_rows'].get('both_within_context_sum_positive_frac'),
                'official_sign_match': s['all_rows'].get('official_t1_sign_match_frac'),
            }
            for m, s in model_summaries.items()
        },
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
