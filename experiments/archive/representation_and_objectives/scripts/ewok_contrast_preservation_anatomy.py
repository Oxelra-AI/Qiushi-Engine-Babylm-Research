#!/usr/bin/env python3
"""research: EWoK / broad-preservation route anatomy while SGCR runs.

Scientific purpose
------------------
SGCR is designed to repair support-sensitive legal40k weaknesses, especially COMPS
and GlobalPIQA. research showed EWoK has weak exact-prefix SGCR burden alignment.
This CPU-only script asks a different question: are the current EWoK deficits mainly
rare-token support effects, or do they look like missing *signed relational-context*
learning that should be attacked by a data/objective mechanism distinct from SGCR
and from the innovation-masking route?

The script uses only existing official-compatible evaluation outputs and the exact
compact-view-reinvest 10M training corpus. It does not launch training, evaluation,
or query managed tasks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tokenizers import Tokenizer
except Exception as exc:  # pragma: no cover
    Tokenizer = None
    TOKENIZER_IMPORT_ERROR = repr(exc)
else:
    TOKENIZER_IMPORT_ERROR = None

ROOT = Path('experiments/archive/representation_and_objectives')
WS = ROOT
OUT = WS / 'data/ewok_contrast_preservation_anatomy'
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/ewok_contrast_preservation_anatomy.md')

EVAL_EWOK = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'
TRAIN_10M = Path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
CHANGED_META = Path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
LEGAL16_TOK = WS / 'data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer/tokenizer.json'
LEGAL40_TOK = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k/tokenizer.json'

PREDICTION_FILES = {
    # Non-submission inherited tokenizer coordinate: mechanism evidence only.
    'inherited16_non_submission_43022': WS / 'data/pristine_collate_seed43022/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    'inherited16_non_submission_43122': WS / 'data/pristine_collate_seed43122/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    # Legal tokenizer endpoints.
    'legal16_8x480_43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    'legal16_8x480_43122': WS / 'data/strictsmalltok_seed43122_pristine_collate/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    'legal40_8x480_43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    'legal40_8x480_43122': WS / 'data/legal40k_accum_seed43122_pristine_collate/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
    'legal40_12x384_depth_43022': WS / 'data/legal40k_12x384_depth_seed43022_pristine_collate/results/hf_model/all_full_preds_and_fast_scores_mlm.json',
}

SUMMARY_FILES = {
    'inherited16_non_submission_43022': WS / 'data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json',
    'inherited16_non_submission_43122': WS / 'data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json',
    'legal16_8x480_43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json',
    'legal16_8x480_43122': WS / 'data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json',
    'legal40_8x480_43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    'legal40_8x480_43122': WS / 'data/legal40k_accum_seed43122_pristine_collate/pristine_collate_legal40k_seed43122_summary.json',
    'legal40_12x384_depth_43022': WS / 'data/legal40k_12x384_depth_seed43022_pristine_collate/pristine_collate_legal40k_12x384_depth_seed43022_summary.json',
}

# Visible public leader vector from verified leaderboard/model-card records.
LEADER = {
    'BLiMP': 67.20,
    'Supplement': 56.01,
    'EWoK': 56.07,
    'Entity': 28.45,
    'COMPS': 53.57,
    'SuperGLUE': 69.79,
    'GlobalPIQA': 39.67,
    'Reading': 5.42,
    'AoA': 0.0,
    'Overall': 41.80,
}

# General relation/counterfactual lexicon for measurement only. A future training
# lexicon must be derived from non-evaluation resources or a precommitted general
# list; this script never writes training text from EWoK rows.
CONTRAST_PAIRS = [
    ('inside', 'outside'), ('in', 'out'), ('on', 'off'), ('over', 'under'), ('above', 'below'),
    ('left', 'right'), ('front', 'back'), ('near', 'far'), ('before', 'after'), ('first', 'last'),
    ('earlier', 'later'), ('up', 'down'), ('increase', 'decrease'), ('increases', 'decreases'),
    ('increased', 'decreased'), ('more', 'less'), ('many', 'few'), ('most', 'least'), ('all', 'none'),
    ('full', 'empty'), ('open', 'closed'), ('hot', 'cold'), ('warm', 'cool'), ('wet', 'dry'),
    ('liquid', 'solid'), ('hard', 'soft'), ('heavy', 'light'), ('large', 'small'), ('bigger', 'smaller'),
    ('taller', 'shorter'), ('longer', 'shorter'), ('same', 'different'), ('similar', 'different'),
    ('break', 'fix'), ('breaks', 'fixes'), ('broken', 'fixed'), ('push', 'pull'), ('pushed', 'pulled'),
    ('help', 'hurt'), ('helps', 'hurts'), ('friend', 'enemy'), ('friends', 'enemies'), ('like', 'dislike'),
    ('likes', 'dislikes'), ('believe', 'doubt'), ('believes', 'doubts'), ('know', 'ignore'),
    ('true', 'false'), ('safe', 'dangerous'), ('clean', 'dirty'), ('alive', 'dead'), ('win', 'lose'),
    ('wins', 'loses'), ('give', 'take'), ('gives', 'takes'), ('buy', 'sell'), ('enter', 'exit'),
    ('enters', 'exits'), ('arrive', 'leave'), ('arrives', 'leaves'), ('include', 'exclude'),
    ('includes', 'excludes'), ('cause', 'prevent'), ('causes', 'prevents'), ('allow', 'forbid'),
    ('allows', 'forbids'), ('can', 'cannot'), ('possible', 'impossible'), ('yes', 'no'),
]
CONTRAST_TERMS = {w for pair in CONTRAST_PAIRS for w in pair}
CONTRAST_BY_TERM: dict[str, set[str]] = defaultdict(set)
for a, b in CONTRAST_PAIRS:
    CONTRAST_BY_TERM[a].add(b)
    CONTRAST_BY_TERM[b].add(a)

STOPWORDS = {
    'a','an','the','and','or','but','if','then','that','this','these','those','is','are','was','were','be','been',
    'being','it','its','they','them','their','there','here','to','of','for','from','with','without','as','at','by',
    'in','on','into','onto','inside','outside','up','down','over','under','before','after','than','so','because',
    'he','she','his','her','him','hers','we','you','i','me','my','your','our','ours','who','what','where','when',
    'why','how','will','would','can','could','should','must','may','might','do','does','did','not','no','yes',
    'ali','chao','benny','alex','sam','max','anna','emma','oliver','liam','noah','mia','ava','lucas',
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def sent(raw: dict[str, Any], context_key: str = 'Context1', target_key: str = 'Target1') -> str:
    return ' '.join([str(raw[context_key]), str(raw[target_key])])


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(x: float, denom: float) -> float | None:
    return 100.0 * x / denom if denom else None


def load_ewok_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(EVAL_EWOK.glob('*.jsonl')):
        domain = path.stem
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                raw = json.loads(line)
                row = dict(raw)
                row['_domain'] = domain
                row['_local_index'] = i
                row['_global_index'] = len(rows)
                row['_correct_sentence'] = sent(raw)
                cw1 = set(content_words(raw.get('Context1', '')))
                cw2 = set(content_words(raw.get('Context2', '')))
                tw1 = set(content_words(raw.get('Target1', '')))
                tw2 = set(content_words(raw.get('Target2', '')))
                row['_context_diff_words'] = sorted((cw1 ^ cw2) - STOPWORDS)
                row['_target_diff_words'] = sorted((tw1 ^ tw2) - STOPWORDS)
                row['_all_eval_content_words'] = sorted(set(content_words(' '.join([
                    raw.get('Context1',''), raw.get('Context2',''), raw.get('Target1',''), raw.get('Target2','')
                ]))))
                rows.append(row)
    return rows


def load_endpoint_correctness(rows: list[dict[str, Any]]) -> dict[str, dict[int, bool]]:
    by_endpoint: dict[str, dict[int, bool]] = {}
    domain_offsets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        domain_offsets[row['_domain']].append(row['_global_index'])
    for name, path in PREDICTION_FILES.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        ewok = payload['ewok']
        endpoint: dict[int, bool] = {}
        for domain, indices in domain_offsets.items():
            preds = ewok[domain]['predictions']
            if len(preds) != len(indices):
                raise RuntimeError(f'{name} {domain}: predictions {len(preds)} != rows {len(indices)}')
            for pred, idx in zip(preds, indices):
                endpoint[idx] = str(pred['pred']).strip() == str(rows[idx]['_correct_sentence']).strip()
        by_endpoint[name] = endpoint
    return by_endpoint


def endpoint_summary_scores() -> dict[str, dict[str, float]]:
    out = {}
    for name, path in SUMMARY_FILES.items():
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        ss = d.get('score_summary') or {}
        scores = ss.get('scores')
        if not isinstance(scores, dict):
            # research stores official_overall and details; recover enough for EWoK if possible.
            details = ss.get('details') or {}
            scores = {'Overall': ss.get('official_overall')}
            if 'EWoK' in details and isinstance(details['EWoK'], (int, float)):
                scores['EWoK'] = details['EWoK']
        out[name] = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
    return out


def group_accuracy(rows: list[dict[str, Any]], corr: dict[int, bool], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(corr[row['_global_index']])
    return {
        k: {'n': len(v), 'acc': 100.0 * sum(v) / len(v), 'correct': int(sum(v)), 'wrong': int(len(v) - sum(v))}
        for k, v in sorted(buckets.items())
    }


def score_macro(accs: dict[str, dict[str, Any]]) -> float:
    return sum(v['acc'] for v in accs.values()) / len(accs)


def corpus_word_and_reservoir_counts() -> tuple[Counter, dict[str, Any]]:
    wc: Counter[str] = Counter()
    reservoir = {
        'train_10m_path': str(TRAIN_10M),
        'train_10m_sha256': sha256_file(TRAIN_10M),
        'rows': 0,
        'words': 0,
        'source_words': Counter(),
        'source_rows': Counter(),
        'contrast_rows_any_term': 0,
        'contrast_words_any_term': 0,
        'contrast_rows_exact_pair_both_sides': 0,
        'contrast_words_exact_pair_both_sides': 0,
        'contrast_rows_single_side_swappable': 0,
        'contrast_words_single_side_swappable': 0,
        'pair_row_counts': Counter(),
        'pair_word_counts': Counter(),
        'pair_both_side_row_counts': Counter(),
        'pair_single_side_row_counts': Counter(),
        'source_contrast_single_side_words': Counter(),
        'source_contrast_both_side_words': Counter(),
    }
    with TRAIN_10M.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get('text', '')
            row_words = int(obj.get('words') or len(text.split()))
            source = str(obj.get('source', 'unknown'))
            toks = words(text)
            toks_set = set(toks)
            wc.update(toks)
            reservoir['rows'] += 1
            reservoir['words'] += row_words
            reservoir['source_rows'][source] += 1
            reservoir['source_words'][source] += row_words
            if toks_set & CONTRAST_TERMS:
                reservoir['contrast_rows_any_term'] += 1
                reservoir['contrast_words_any_term'] += row_words
            any_both = False
            any_single = False
            for a, b in CONTRAST_PAIRS:
                has_a = a in toks_set
                has_b = b in toks_set
                if has_a or has_b:
                    reservoir['pair_row_counts'][f'{a}/{b}'] += 1
                    reservoir['pair_word_counts'][f'{a}/{b}'] += row_words
                if has_a and has_b:
                    any_both = True
                    reservoir['pair_both_side_row_counts'][f'{a}/{b}'] += 1
                elif has_a ^ has_b:
                    any_single = True
                    reservoir['pair_single_side_row_counts'][f'{a}/{b}'] += 1
            if any_both:
                reservoir['contrast_rows_exact_pair_both_sides'] += 1
                reservoir['contrast_words_exact_pair_both_sides'] += row_words
                reservoir['source_contrast_both_side_words'][source] += row_words
            if any_single:
                reservoir['contrast_rows_single_side_swappable'] += 1
                reservoir['contrast_words_single_side_swappable'] += row_words
                reservoir['source_contrast_single_side_words'][source] += row_words
    # Convert Counters to plain sorted dicts.
    for k in list(reservoir.keys()):
        if isinstance(reservoir[k], Counter):
            reservoir[k] = dict(reservoir[k].most_common())
    reservoir['contrast_any_term_word_frac'] = reservoir['contrast_words_any_term'] / reservoir['words']
    reservoir['contrast_single_side_swappable_word_frac'] = reservoir['contrast_words_single_side_swappable'] / reservoir['words']
    reservoir['contrast_exact_pair_both_sides_word_frac'] = reservoir['contrast_words_exact_pair_both_sides'] / reservoir['words']
    return wc, reservoir


def token_support_counts(tokenizer_path: Path) -> tuple[Any, Counter[int], str]:
    if Tokenizer is None:
        raise RuntimeError(f'tokenizers import failed: {TOKENIZER_IMPORT_ERROR}')
    tok = Tokenizer.from_file(str(tokenizer_path))
    counts: Counter[int] = Counter()
    with TRAIN_10M.open('r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line).get('text', '')
            counts.update(tok.encode(text).ids)
    return tok, counts, sha256_file(tokenizer_path)


def add_row_coverage(rows: list[dict[str, Any]], wc: Counter, tok40=None, counts40=None, tok16=None, counts16=None) -> None:
    for row in rows:
        concept_words = content_words(' '.join([str(row.get('ConceptA','')), str(row.get('ConceptB',''))]))
        diff_words = list(row['_context_diff_words'])
        target_words = list(row['_target_diff_words'])
        all_words = list(row['_all_eval_content_words'])
        def count_stats(ws: list[str]) -> dict[str, Any]:
            cs = [int(wc.get(w, 0)) for w in ws]
            return {
                'n_words': len(ws),
                'min': min(cs) if cs else None,
                'mean': mean([float(c) for c in cs]) if cs else None,
                'zero_frac': sum(1 for c in cs if c == 0) / len(cs) if cs else None,
                'lt10_frac': sum(1 for c in cs if c < 10) / len(cs) if cs else None,
                'lt100_frac': sum(1 for c in cs if c < 100) / len(cs) if cs else None,
            }
        row['_corpus_counts'] = {
            'concept_words': count_stats(concept_words),
            'context_diff_words': count_stats(diff_words),
            'target_diff_words': count_stats(target_words),
            'all_eval_content_words': count_stats(all_words),
        }
        row['_contrast_terms_in_context_diff'] = sorted(set(diff_words) & CONTRAST_TERMS)
        row['_contrast_terms_in_all_eval'] = sorted(set(all_words) & CONTRAST_TERMS)
        if tok40 is not None and counts40 is not None:
            ids = tok40.encode(' '.join([str(row.get('Context1','')), str(row.get('Context2',''))])).ids
            low = [counts40.get(i, 0) for i in ids]
            row['_legal40_context_pair_token_support'] = {
                'n_tokens': len(ids), 'frac_lt20': sum(c < 20 for c in low) / len(low) if low else None,
                'frac_lt50': sum(c < 50 for c in low) / len(low) if low else None,
                'min': min(low) if low else None,
                'mean': mean([float(c) for c in low]) if low else None,
            }
        if tok16 is not None and counts16 is not None:
            ids = tok16.encode(' '.join([str(row.get('Context1','')), str(row.get('Context2',''))])).ids
            low = [counts16.get(i, 0) for i in ids]
            row['_legal16_context_pair_token_support'] = {
                'n_tokens': len(ids), 'frac_lt20': sum(c < 20 for c in low) / len(low) if low else None,
                'frac_lt50': sum(c < 50 for c in low) / len(low) if low else None,
                'min': min(low) if low else None,
                'mean': mean([float(c) for c in low]) if low else None,
            }


def cov_aggregate(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...]) -> list[float]:
        vals = []
        for r in rows_subset:
            x: Any = r
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
                if x is None:
                    break
            if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
                vals.append(float(x))
        return vals
    out = {'n': len(rows_subset)}
    for label, path in {
        'concept_min_count': ('_corpus_counts','concept_words','min'),
        'context_diff_min_count': ('_corpus_counts','context_diff_words','min'),
        'target_diff_min_count': ('_corpus_counts','target_diff_words','min'),
        'all_content_mean_count': ('_corpus_counts','all_eval_content_words','mean'),
        'legal40_context_frac_lt50': ('_legal40_context_pair_token_support','frac_lt50'),
        'legal16_context_frac_lt50': ('_legal16_context_pair_token_support','frac_lt50'),
    }.items():
        vals = collect(path)
        out[label + '_mean'] = mean(vals)
        out[label + '_median'] = sorted(vals)[len(vals)//2] if vals else None
    out['contrast_term_in_context_diff_frac'] = sum(1 for r in rows_subset if r.get('_contrast_terms_in_context_diff')) / len(rows_subset) if rows_subset else None
    out['contrast_term_anywhere_frac'] = sum(1 for r in rows_subset if r.get('_contrast_terms_in_all_eval')) / len(rows_subset) if rows_subset else None
    return out


def build_tables(rows: list[dict[str, Any]], correctness: dict[str, dict[int, bool]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_domain = {name: group_accuracy(rows, corr, lambda r: r['_domain']) for name, corr in correctness.items()}
    endpoint_context_diff = {name: group_accuracy(rows, corr, lambda r: r.get('ContextDiff','')) for name, corr in correctness.items()}
    endpoint_target_diff = {name: group_accuracy(rows, corr, lambda r: r.get('TargetDiff','')) for name, corr in correctness.items()}
    endpoint_context_type = {name: group_accuracy(rows, corr, lambda r: r.get('ContextType','')) for name, corr in correctness.items()}

    domain_rows = []
    domains = sorted({r['_domain'] for r in rows})
    for domain in domains:
        rec: dict[str, Any] = {'domain': domain, 'n': sum(1 for r in rows if r['_domain'] == domain)}
        for name in correctness:
            rec[name + '_acc'] = endpoint_domain[name][domain]['acc']
        if 'legal40_12x384_depth_43022' in correctness and 'legal40_8x480_43022' in correctness:
            rec['depth_minus_legal40_43022'] = rec['legal40_12x384_depth_43022_acc'] - rec['legal40_8x480_43022_acc']
        if 'legal40_8x480_43022' in correctness and 'legal16_8x480_43022' in correctness:
            rec['legal40_minus_legal16_43022'] = rec['legal40_8x480_43022_acc'] - rec['legal16_8x480_43022_acc']
        if 'inherited16_non_submission_43022' in correctness and 'legal40_8x480_43022' in correctness:
            rec['legal40_minus_inherited_43022'] = rec['legal40_8x480_43022_acc'] - rec['inherited16_non_submission_43022_acc']
        domain_rows.append(rec)

    # Per-item patterns.
    pattern_buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    item_records: list[dict[str, Any]] = []
    names = list(correctness.keys())
    for row in rows:
        idx = row['_global_index']
        bits = {name: correctness[name][idx] for name in names}
        legal_names = [n for n in names if n.startswith('legal')]
        seed43022_names = [n for n in names if n.endswith('43022')]
        all_legal_wrong = all(not bits[n] for n in legal_names)
        all_legal_right = all(bits[n] for n in legal_names)
        depth_wrong_legal40_wrong = ('legal40_12x384_depth_43022' in bits and 'legal40_8x480_43022' in bits and (not bits['legal40_12x384_depth_43022']) and (not bits['legal40_8x480_43022']))
        inherited_right_legal_wrong = ('inherited16_non_submission_43022' in bits and bits['inherited16_non_submission_43022'] and depth_wrong_legal40_wrong)
        pattern = tuple(1 if bits[n] else 0 for n in names)
        rec = {
            'global_index': idx,
            'domain': row['_domain'],
            'local_index': row['_local_index'],
            'ContextType': row.get('ContextType'),
            'ContextDiff': row.get('ContextDiff'),
            'TargetDiff': row.get('TargetDiff'),
            'ConceptA': row.get('ConceptA'),
            'ConceptB': row.get('ConceptB'),
            'context_diff_words': ' '.join(row['_context_diff_words']),
            'target_diff_words': ' '.join(row['_target_diff_words']),
            'contrast_terms_in_context_diff': ' '.join(row['_contrast_terms_in_context_diff']),
            'all_legal_wrong': all_legal_wrong,
            'all_legal_right': all_legal_right,
            'depth_wrong_legal40_wrong': depth_wrong_legal40_wrong,
            'inherited43022_right_but_legal40_depth_wrong': inherited_right_legal_wrong,
            **{n: int(bits[n]) for n in names},
        }
        rec['pattern'] = ''.join(map(str, pattern))
        item_records.append(rec)
        pattern_buckets[pattern].append(row)

    pattern_rows = []
    for pattern, rs in sorted(pattern_buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        labels = {name: bool(bit) for name, bit in zip(names, pattern)}
        pattern_rows.append({
            'pattern': ''.join(map(str, pattern)),
            'n': len(rs),
            'frac': len(rs) / len(rows),
            **{name: int(val) for name, val in labels.items()},
            'domains_top': dict(Counter(r['_domain'] for r in rs).most_common(5)),
            'ContextDiff_top': dict(Counter(r.get('ContextDiff','') for r in rs).most_common(5)),
            'coverage': cov_aggregate(rs),
        })

    cross = {}
    for name, corr in correctness.items():
        correct_rows = [r for r in rows if corr[r['_global_index']]]
        wrong_rows = [r for r in rows if not corr[r['_global_index']]]
        cross[name] = {'correct': cov_aggregate(correct_rows), 'wrong': cov_aggregate(wrong_rows)}

    tables = {
        'endpoint_domain': endpoint_domain,
        'endpoint_context_diff': endpoint_context_diff,
        'endpoint_target_diff': endpoint_target_diff,
        'endpoint_context_type': endpoint_context_type,
        'coverage_by_endpoint_correctness': cross,
    }
    return tables, domain_rows, pattern_rows, item_records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            flat = {}
            for k, v in r.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                else:
                    flat[k] = v
            w.writerow(flat)


def compact_top(counter_dict: dict[str, int], n: int = 20) -> dict[str, int]:
    return dict(list(counter_dict.items())[:n])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_ewok_rows()
    correctness = load_endpoint_correctness(rows)
    score_summaries = endpoint_summary_scores()
    wc, reservoir = corpus_word_and_reservoir_counts()

    tokenizer_status: dict[str, Any] = {'tokenizers_import_error': TOKENIZER_IMPORT_ERROR}
    tok16 = tok40 = counts16 = counts40 = None
    if Tokenizer is not None:
        tok16, counts16, sha16 = token_support_counts(LEGAL16_TOK)
        tok40, counts40, sha40 = token_support_counts(LEGAL40_TOK)
        tokenizer_status.update({
            'legal16_tokenizer': str(LEGAL16_TOK),
            'legal16_tokenizer_sha256': sha16,
            'legal16_unique_token_ids_in_pool': len(counts16),
            'legal16_pool_token_total': int(sum(counts16.values())),
            'legal40_tokenizer': str(LEGAL40_TOK),
            'legal40_tokenizer_sha256': sha40,
            'legal40_unique_token_ids_in_pool': len(counts40),
            'legal40_pool_token_total': int(sum(counts40.values())),
        })
    add_row_coverage(rows, wc, tok40, counts40, tok16, counts16)
    tables, domain_rows, pattern_rows, item_records = build_tables(rows, correctness)

    # Recompute macro EWoK from item correctness for sanity against summaries.
    macro_checks = {}
    for name, domain_acc in tables['endpoint_domain'].items():
        macro = score_macro(domain_acc)
        summary_score = score_summaries.get(name, {}).get('EWoK')
        macro_checks[name] = {
            'macro_from_predictions': macro,
            'summary_EWoK': summary_score,
            'abs_diff': abs(macro - summary_score) if summary_score is not None else None,
        }

    # Domain-level route signals.
    legal40_depth_both_wrong_rows = [r for r in rows if (not correctness['legal40_8x480_43022'][r['_global_index']]) and (not correctness['legal40_12x384_depth_43022'][r['_global_index']])]
    all_legal_wrong_rows = [r for r in rows if all(not correctness[n][r['_global_index']] for n in correctness if n.startswith('legal'))]
    inherited_rescued_rows = [r for r in rows if correctness['inherited16_non_submission_43022'][r['_global_index']] and (not correctness['legal40_8x480_43022'][r['_global_index']]) and (not correctness['legal40_12x384_depth_43022'][r['_global_index']])]
    depth_lost_from_legal40_rows = [r for r in rows if correctness['legal40_8x480_43022'][r['_global_index']] and (not correctness['legal40_12x384_depth_43022'][r['_global_index']])]
    depth_gained_from_legal40_rows = [r for r in rows if (not correctness['legal40_8x480_43022'][r['_global_index']]) and correctness['legal40_12x384_depth_43022'][r['_global_index']]]

    route_signal = {
        'total_ewok_rows': len(rows),
        'domains': sorted({r['_domain'] for r in rows}),
        'legal40_depth_both_wrong': {
            'n': len(legal40_depth_both_wrong_rows),
            'frac': len(legal40_depth_both_wrong_rows) / len(rows),
            'domains_top': dict(Counter(r['_domain'] for r in legal40_depth_both_wrong_rows).most_common()),
            'ContextDiff_top': dict(Counter(r.get('ContextDiff','') for r in legal40_depth_both_wrong_rows).most_common(12)),
            'coverage': cov_aggregate(legal40_depth_both_wrong_rows),
        },
        'all_legal_wrong': {
            'n': len(all_legal_wrong_rows),
            'frac': len(all_legal_wrong_rows) / len(rows),
            'domains_top': dict(Counter(r['_domain'] for r in all_legal_wrong_rows).most_common()),
            'ContextDiff_top': dict(Counter(r.get('ContextDiff','') for r in all_legal_wrong_rows).most_common(12)),
            'coverage': cov_aggregate(all_legal_wrong_rows),
        },
        'inherited43022_right_but_legal40_depth_wrong': {
            'n': len(inherited_rescued_rows),
            'frac': len(inherited_rescued_rows) / len(rows),
            'domains_top': dict(Counter(r['_domain'] for r in inherited_rescued_rows).most_common()),
            'ContextDiff_top': dict(Counter(r.get('ContextDiff','') for r in inherited_rescued_rows).most_common(12)),
            'coverage': cov_aggregate(inherited_rescued_rows),
        },
        'depth_lost_from_legal40_43022': {
            'n': len(depth_lost_from_legal40_rows),
            'domains_top': dict(Counter(r['_domain'] for r in depth_lost_from_legal40_rows).most_common()),
            'coverage': cov_aggregate(depth_lost_from_legal40_rows),
        },
        'depth_gained_from_legal40_43022': {
            'n': len(depth_gained_from_legal40_rows),
            'domains_top': dict(Counter(r['_domain'] for r in depth_gained_from_legal40_rows).most_common()),
            'coverage': cov_aggregate(depth_gained_from_legal40_rows),
        },
    }

    # Top swappable contrast pairs by corpus reservoir; save separate CSV.
    pair_rows = []
    both_counts = reservoir['pair_both_side_row_counts']
    single_counts = reservoir['pair_single_side_row_counts']
    word_counts = reservoir['pair_word_counts']
    for pair in sorted(set(pair for pair in word_counts) | set(single_counts) | set(both_counts)):
        pair_rows.append({
            'pair': pair,
            'rows_any_side': reservoir['pair_row_counts'].get(pair, 0),
            'words_any_side': word_counts.get(pair, 0),
            'rows_single_side_swappable': single_counts.get(pair, 0),
            'rows_both_sides': both_counts.get(pair, 0),
        })
    pair_rows.sort(key=lambda r: (-r['rows_single_side_swappable'], r['pair']))

    # Per-domain compact route table: endpoint accuracies plus persistent-error support.
    domain_route_rows = []
    for rec in domain_rows:
        domain = rec['domain']
        rs = [r for r in rows if r['_domain'] == domain]
        both_wrong = [r for r in rs if r in legal40_depth_both_wrong_rows]
        all_wrong = [r for r in rs if r in all_legal_wrong_rows]
        inherited_right = [r for r in rs if r in inherited_rescued_rows]
        new = dict(rec)
        new['leader_EWoK_macro'] = LEADER['EWoK']
        new['legal40_depth_both_wrong_frac'] = len(both_wrong) / len(rs)
        new['all_legal_wrong_frac'] = len(all_wrong) / len(rs)
        new['inherited43022_right_but_legal40_depth_wrong_frac'] = len(inherited_right) / len(rs)
        new['contrast_term_in_context_diff_frac'] = cov_aggregate(rs)['contrast_term_in_context_diff_frac']
        new['both_wrong_context_diff_min_count_mean'] = cov_aggregate(both_wrong)['context_diff_min_count_mean']
        new['both_wrong_legal40_context_frac_lt50_mean'] = cov_aggregate(both_wrong)['legal40_context_frac_lt50_mean']
        domain_route_rows.append(new)
    domain_route_rows.sort(key=lambda r: (r.get('legal40_12x384_depth_43022_acc', 100), -r['legal40_depth_both_wrong_frac']))

    # Small illustrative samples for human reading; not for training use.
    sample_records = []
    for bucket_name, rs in [
        ('legal40_depth_both_wrong', legal40_depth_both_wrong_rows),
        ('all_legal_wrong', all_legal_wrong_rows),
        ('inherited43022_right_but_legal40_depth_wrong', inherited_rescued_rows),
        ('depth_lost_from_legal40', depth_lost_from_legal40_rows),
    ]:
        # sample high lexical support cases first; those argue against rare-token-only explanation.
        rs_sorted = sorted(rs, key=lambda r: (
            -float((r.get('_corpus_counts', {}).get('context_diff_words', {}).get('min') or 0)),
            r['_domain'], r['_local_index']
        ))[:12]
        for r in rs_sorted:
            sample_records.append({
                'bucket': bucket_name,
                'domain': r['_domain'],
                'local_index': r['_local_index'],
                'ContextDiff': r.get('ContextDiff'),
                'TargetDiff': r.get('TargetDiff'),
                'ConceptA': r.get('ConceptA'),
                'ConceptB': r.get('ConceptB'),
                'Context1': r.get('Context1'),
                'Context2': r.get('Context2'),
                'Target1': r.get('Target1'),
                'context_diff_words': ' '.join(r['_context_diff_words']),
                'context_diff_min_count': r['_corpus_counts']['context_diff_words']['min'],
                'legal40_context_frac_lt50': (r.get('_legal40_context_pair_token_support') or {}).get('frac_lt50'),
                **{name: int(correctness[name][r['_global_index']]) for name in correctness},
            })

    write_csv(OUT / 'ewok_domain_route_table.csv', domain_route_rows)
    write_csv(OUT / 'ewok_error_patterns.csv', pattern_rows)
    write_csv(OUT / 'contrast_lexicon_corpus_reservoir.csv', pair_rows)
    write_csv(OUT / 'ewok_high_support_error_samples.csv', sample_records)
    write_csv(OUT / 'ewok_item_correctness_and_coverage.csv', item_records)

    # Derive a concise route recommendation from the measurements.
    depth_ewok = score_summaries['legal40_12x384_depth_43022']['EWoK']
    legal40_ewok = score_summaries['legal40_8x480_43022']['EWoK']
    legal16_ewok = score_summaries['legal16_8x480_43022']['EWoK']
    inherited_ewok = score_summaries['inherited16_non_submission_43022'].get('EWoK')
    both_wrong_cov = route_signal['legal40_depth_both_wrong']['coverage']
    route_recommendation = {
        'mechanism_name': 'self-contrastive relational-context anchoring (SCRA)',
        'mechanism': (
            'Train the model not only to reconstruct tokens, but to prefer an observed sentence/context over a minimally '
            'relation-corrupted counterfactual generated from the same allowed corpus row. The corruption lexicon must be general '
            'and non-evaluation-derived; corrupted words must be charged inside the <=100M exposure budget or used in a matched '
            'replacement slice. This targets EWoK-style signed context compatibility while keeping compact-view data and SGCR distinct.'
        ),
        'why_distinct_from_sgcr': (
            'SGCR changes the estimator for rare legal40k rows and research maps strongest burden to COMPS/GlobalPIQA; SCRA changes the '
            'learning signal for high-support relation words and minimal context counterfactuals, the place where EWoK errors persist.'
        ),
        'why_distinct_from_a02_innovation_masking': (
            'A02 masks source-absent rewrite innovations inside compact pairs; SCRA would build source-grounded real-vs-corrupted '
            'relation preferences from ordinary allowed corpus rows, with no dependence on source/rewrite innovation labels.'
        ),
        'low_cost_evidence_used_here': {
            'depth_EWoK': depth_ewok,
            'legal40_EWoK': legal40_ewok,
            'legal16_EWoK': legal16_ewok,
            'inherited_non_submission_EWoK': inherited_ewok,
            'leader_EWoK': LEADER['EWoK'],
            'depth_minus_legal40_EWoK': depth_ewok - legal40_ewok,
            'legal40_minus_legal16_EWoK': legal40_ewok - legal16_ewok,
            'leader_minus_depth_EWoK': LEADER['EWoK'] - depth_ewok,
            'legal40_depth_both_wrong_count': route_signal['legal40_depth_both_wrong']['n'],
            'legal40_depth_both_wrong_frac': route_signal['legal40_depth_both_wrong']['frac'],
            'all_legal_wrong_count': route_signal['all_legal_wrong']['n'],
            'inherited_right_but_legal40_depth_wrong_count': route_signal['inherited43022_right_but_legal40_depth_wrong']['n'],
            'both_wrong_context_diff_min_count_mean': both_wrong_cov['context_diff_min_count_mean'],
            'both_wrong_legal40_context_frac_lt50_mean': both_wrong_cov['legal40_context_frac_lt50_mean'],
            'corpus_contrast_single_side_swappable_words': reservoir['contrast_words_single_side_swappable'],
            'corpus_contrast_single_side_swappable_frac': reservoir['contrast_single_side_swappable_word_frac'],
        },
        'first_decision_changing_experiment_when_gpu_available': (
            'Do not launch until SGCR endpoint is read. If EWoK remains a large gap and SGCR does not solve broad preservation, run a '
            'small matched late-exposure or full-run test in which a fixed budget slice is replaced by source-grounded SCRA packets. '
            'The stop/continue signal is EWoK improvement over matched legal40/depth without losing BLiMP/Supplement/SuperGLUE/Reading; '
            'if only EWoK rises while broad columns fall, the route is not a SOTA route.'
        ),
    }

    result = {
        'status': 'EWOK_CONTRAST_PRESERVATION_ANATOMY',
        'created_utc': now_utc(),
        'script_never_launches_training_or_evaluation': True,
        'official_ewok_path': str(EVAL_EWOK),
        'training_corpus_path': str(TRAIN_10M),
        'changed_block_meta_path': str(CHANGED_META),
        'visible_leader': LEADER,
        'prediction_files': {k: str(v) for k, v in PREDICTION_FILES.items()},
        'summary_files': {k: str(v) for k, v in SUMMARY_FILES.items()},
        'score_summaries': score_summaries,
        'macro_checks': macro_checks,
        'tokenizer_status': tokenizer_status,
        'corpus_reservoir': reservoir,
        'endpoint_domain_accuracy': tables['endpoint_domain'],
        'endpoint_context_type_accuracy': tables['endpoint_context_type'],
        'endpoint_context_diff_accuracy': tables['endpoint_context_diff'],
        'endpoint_target_diff_accuracy': tables['endpoint_target_diff'],
        'route_signal': route_signal,
        'route_recommendation': route_recommendation,
        'csv_outputs': {
            'ewok_domain_route_table': str(OUT / 'ewok_domain_route_table.csv'),
            'ewok_error_patterns': str(OUT / 'ewok_error_patterns.csv'),
            'contrast_lexicon_corpus_reservoir': str(OUT / 'contrast_lexicon_corpus_reservoir.csv'),
            'ewok_high_support_error_samples': str(OUT / 'ewok_high_support_error_samples.csv'),
            'ewok_item_correctness_and_coverage': str(OUT / 'ewok_item_correctness_and_coverage.csv'),
        },
        'json_output': str(OUT / 'ewok_contrast_preservation_anatomy.json'),
        'note': str(NOTE),
    }

    (OUT / 'ewok_contrast_preservation_anatomy.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # Human-readable note.
    worst_domains = domain_route_rows[:6]
    top_pairs = pair_rows[:12]
    lines = []
    lines.append('# research — EWoK contrast and broad-preservation anatomy\n\n')
    lines.append('This CPU-only analysis used existing official-compatible prediction files and the exact compact-view-reinvest 10M corpus. It did not launch training/evaluation.\n\n')
    lines.append('## Score-level anchor\n\n')
    def fmt(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else 'n/a'
    for name in ['legal16_8x480_43022','legal40_8x480_43022','legal40_12x384_depth_43022','inherited16_non_submission_43022']:
        ss = score_summaries.get(name, {})
        lines.append(f"- `{name}` EWoK `{fmt(ss.get('EWoK'))}` Overall `{fmt(ss.get('Overall'), 4)}`\n")
    lines.append(f"- visible leader EWoK `{LEADER['EWoK']:.4f}`; depth remains `{LEADER['EWoK'] - depth_ewok:.4f}` points behind on this column.\n\n")
    lines.append('## Persistent-error signal\n\n')
    lines.append(f"- Legal40 8x480 and legal40 12x384 depth are both wrong on `{len(legal40_depth_both_wrong_rows)}` / `{len(rows)}` EWoK rows (`{len(legal40_depth_both_wrong_rows)/len(rows):.3f}`).\n")
    lines.append(f"- All measured legal endpoints are wrong on `{len(all_legal_wrong_rows)}` rows (`{len(all_legal_wrong_rows)/len(rows):.3f}`).\n")
    lines.append(f"- The non-submission inherited-tokenizer seed43022 is right while legal40 and depth are both wrong on `{len(inherited_rescued_rows)}` rows; these are representation-sensitive but not solved by current legal/depth coordinates.\n")
    lines.append(f"- For the legal40+depth both-wrong rows, mean minimum corpus count of context-difference words is `{both_wrong_cov['context_diff_min_count_mean']:.2f}` and mean legal40 context-token fraction below 50 pool occurrences is `{both_wrong_cov['legal40_context_frac_lt50_mean']:.4f}`. This argues against a rare-token-only explanation for much of EWoK.\n\n")
    lines.append('## Weakest EWoK domains under depth\n\n')
    lines.append('| domain | n | depth acc | legal40 acc | legal16 acc | both-wrong frac | context contrast term frac |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|\n')
    for r in worst_domains:
        lines.append(f"| {r['domain']} | {r['n']} | {r.get('legal40_12x384_depth_43022_acc', float('nan')):.2f} | {r.get('legal40_8x480_43022_acc', float('nan')):.2f} | {r.get('legal16_8x480_43022_acc', float('nan')):.2f} | {r['legal40_depth_both_wrong_frac']:.3f} | {r['contrast_term_in_context_diff_frac']:.3f} |\n")
    lines.append('\n## Corpus reservoir for a distinct self-contrastive route\n\n')
    lines.append(f"- Exact 10M corpus SHA `{reservoir['train_10m_sha256']}`, words `{reservoir['words']}`, rows `{reservoir['rows']}`.\n")
    lines.append(f"- Rows containing at least one side of a general contrast pair: `{reservoir['contrast_rows_any_term']}` rows / `{reservoir['contrast_words_any_term']}` words (`{reservoir['contrast_any_term_word_frac']:.3f}` of corpus words).\n")
    lines.append(f"- Rows with exactly one side of at least one contrast pair, suitable for deterministic real-vs-corrupted sentence preference if charged against budget: `{reservoir['contrast_rows_single_side_swappable']}` rows / `{reservoir['contrast_words_single_side_swappable']}` words (`{reservoir['contrast_single_side_swappable_word_frac']:.3f}`).\n")
    lines.append(f"- Rows with both sides of a pair, useful for natural contrast mining rather than corruption: `{reservoir['contrast_rows_exact_pair_both_sides']}` rows / `{reservoir['contrast_words_exact_pair_both_sides']}` words (`{reservoir['contrast_exact_pair_both_sides_word_frac']:.3f}`).\n\n")
    lines.append('Top general contrast-pair reservoirs:\n\n')
    lines.append('| pair | rows single-side | rows both-sides | rows any-side |\n')
    lines.append('|---|---:|---:|---:|\n')
    for r in top_pairs:
        lines.append(f"| {r['pair']} | {r['rows_single_side_swappable']} | {r['rows_both_sides']} | {r['rows_any_side']} |\n")
    lines.append('\n## Mechanism opened\n\n')
    lines.append('The supported route is **self-contrastive relational-context anchoring (SCRA)**: preserve the compact-view data mechanism and any SGCR endpoint, but add a small, exactly accounted source-grounded preference signal where an observed corpus sentence/context must outrank a minimally relation-corrupted counterfactual. This targets EWoK because EWoK scores signed context compatibility, not just rare subword support. It is not an SGCR parameter variant and not A02-style innovation masking. The future training lexicon must be non-evaluation-derived, and every corrupted sequence/word exposure must be charged or placed in a replacement slice under the Strict-Small budget.\n\n')
    lines.append('## Files\n\n')
    for label, path in result['csv_outputs'].items():
        lines.append(f"- {label}: `{path}`\n")
    lines.append(f"- JSON: `{result['json_output']}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')

    print(json.dumps({
        'status': result['status'],
        'ewok_rows': len(rows),
        'depth_ewok': depth_ewok,
        'legal40_ewok': legal40_ewok,
        'leader_minus_depth_ewok': LEADER['EWoK'] - depth_ewok,
        'legal40_depth_both_wrong': route_signal['legal40_depth_both_wrong']['n'],
        'all_legal_wrong': route_signal['all_legal_wrong']['n'],
        'inherited_right_but_legal40_depth_wrong': route_signal['inherited43022_right_but_legal40_depth_wrong']['n'],
        'both_wrong_context_diff_min_count_mean': both_wrong_cov['context_diff_min_count_mean'],
        'both_wrong_legal40_context_frac_lt50_mean': both_wrong_cov['legal40_context_frac_lt50_mean'],
        'contrast_single_side_swappable_words': reservoir['contrast_words_single_side_swappable'],
        'out_json': result['json_output'],
        'note': result['note'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
