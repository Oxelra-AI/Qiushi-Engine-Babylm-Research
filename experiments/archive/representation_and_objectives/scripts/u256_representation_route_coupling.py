#!/usr/bin/env python3
"""research: couple U256 recovered-suffix content with the representation-corner evidence.

CPU-only. No training, no model evaluation, no managed-task status query.
The purpose is to prevent the prepared U256 launch from becoming automatic after
research showed that its newly visible mass is mostly CHILDES row tails rather
than compact FineWeb source/rewrite pairs.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
for p in (WS / 'scripts', ROOT / 'experiments/archive/compact_experience/scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import experience_utilization_trainer as chunkbase  # noqa: E402

BASE_10M = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
TOKENIZER = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
LEGAL40K_TWO_SEED = WS / 'data/legal40k_two_seed_comparison/legal40k_two_seed_comparison.json'
SUPPORT_CSV = WS / 'data/tokenizer_support_spectrum/eval_family_low_support.csv'
POOL_SUPPORT_CSV = WS / 'data/tokenizer_support_spectrum/pool_support_summary.csv'
OUT_DIR = WS / 'data/u256_representation_route_coupling'
OUT_JSON = OUT_DIR / 'u256_representation_route_coupling.json'
FEATURE_CSV = OUT_DIR / 'u256_feature_enrichment_by_source.csv'
SUPPORT_ROUTE_CSV = OUT_DIR / 'support_floor_eval_family_tradeoff.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/u256_representation_route_coupling.md')

EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_TOK_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'

TOK_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?\b", re.I)
CAP_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b")
SPEAKER_RE = re.compile(r"(?:\*[A-Z]{2,4}:|\[[^\]]{2,80}\]|^-\s)")
BAD_MARKERS = ['�', 'À', 'Á', 'Å', 'Ç', 'È', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'æ', 'Æ', '⁄', '½', '¼', '\ufffd']

ACTION = {'put','take','get','give','go','come','make','made','open','close','hold','push','pull','move','moved','turn','turned','drop','dropped','pick','picked','use','used','using','eat','drink','throw','bring','carry','keep','help','build','built','break','change','changed','find','found','play','walk','run','fall','fell','sit','stand','touch','wash','cut','fix'}
CAUSAL = {'because','cause','caused','causes','so','therefore','after','before','when','while','if','then','result','results','resulted','became','become','prevent','allow','allows','requires','required','during','until','since','leads','led','makes','made'}
PHYSICAL = {'water','fire','box','ball','door','cup','table','toy','paper','hand','body','food','stone','wood','glass','machine','tool','wheel','container','bag','room','floor','wall','book','plant','animal','air','light','heat','material','metal','house','car','window','cloth','chair','bed','bottle','milk','train'}
SPATIAL = {'in','on','under','over','inside','outside','behind','front','near','across','through','between','below','above','around','left','right','top','bottom','beside','into','out','off','down','up','back','there','here','where'}
SOCIAL = {'think','thought','know','knew','want','wanted','say','said','ask','asked','tell','told','see','saw','look','looked','feel','felt','believe','learn','teach','child','mother','father','person','people','friend','teacher','man','woman','boy','girl','family','name','mom','dad','baby','he','she','him','her'}
PRONOUN = {'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','their','our','mine','yours','hers','theirs','myself','yourself','himself','herself','itself','ourselves','themselves'}

LEADER = {
    'BLiMP': 67.20,
    'Supplement': 56.01,
    'EWoK': 56.07,
    'Entity': 28.45,
    'COMPS': 53.57,
    'GlobalPIQA': 39.67,
    'SuperGLUE': 69.79,
    'Reading': 5.42,
    'AoA': 0.0,
    'Overall': 41.80,
}

KEEP_TOKENIZERS = [
    'legal_a01_16k',
    'legal_byte_bpe_40k',
    'legal_byte_bpe_40k_minfreq25',
    'legal_byte_bpe_40k_minfreq50',
]
KEEP_FAMILIES = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'SuperGLUE', 'Reading']
LOW_SUPPORT_COLS = ['frac_lt50', 'frac_lt100', 'frac_lt200']


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def visible_word_count(text: str, tokenizer) -> tuple[int, int, int, int]:
    by_word, raw_tokens, unassigned = chunkbase.token_ids_by_whitespace_word(text, tokenizer)
    if unassigned:
        raise RuntimeError(f'unassigned offsets: {unassigned}')
    cum = 0
    vis_words = 0
    boundary_hidden = 0
    for ids in by_word:
        wc = len(ids)
        if wc == 0:
            if cum < 256:
                vis_words += 1
            continue
        if cum < 256:
            vis_words += 1
            if cum + wc > 256:
                boundary_hidden += cum + wc - 256
        cum += wc
    return vis_words, raw_tokens, max(0, raw_tokens - 256), boundary_hidden


def feature_counts(text: str) -> collections.Counter[str]:
    words_lower = [m.group(0).lower() for m in TOK_RE.finditer(text)]
    cnt = collections.Counter(words_lower)
    caps = CAP_RE.findall(text)
    nums = NUM_RE.findall(text)
    speaker = SPEAKER_RE.findall(text)
    bad_marker_hits = sum(text.count(m) for m in BAD_MARKERS)
    chars = len(text)
    non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
    out = collections.Counter()
    out['lexical_words'] = len(words_lower)
    out['surface_words'] = len(text.split())
    out['action'] = sum(cnt[w] for w in ACTION)
    out['causal_temporal'] = sum(cnt[w] for w in CAUSAL)
    out['physical_object'] = sum(cnt[w] for w in PHYSICAL)
    out['spatial_state'] = sum(cnt[w] for w in SPATIAL)
    out['mental_social'] = sum(cnt[w] for w in SOCIAL)
    out['pronoun'] = sum(cnt[w] for w in PRONOUN)
    out['numbers'] = len(nums)
    out['capitalized'] = len(caps)
    out['speaker_markers'] = len(speaker)
    out['bad_marker_hits'] = bad_marker_hits
    out['non_ascii_chars'] = non_ascii_chars
    out['chars'] = chars
    out['relation_action_social_union'] = 1 if (out['action'] + out['causal_temporal'] + out['physical_object'] + out['spatial_state'] + out['mental_social']) >= 3 else 0
    return out


def add_counter(dst: collections.Counter[str], src: collections.Counter[str]) -> None:
    for k, v in src.items():
        dst[k] += v


def enrich_row(source: str, visible: collections.Counter[str], suffix: collections.Counter[str]) -> dict[str, Any]:
    row: dict[str, Any] = {'source': source}
    row['visible_surface_words'] = visible['surface_words']
    row['suffix_surface_words'] = suffix['surface_words']
    row['u256_word_gain_frac'] = safe_div(suffix['surface_words'], visible['surface_words'])
    for feat in ['action','causal_temporal','physical_object','spatial_state','mental_social','pronoun','capitalized','numbers','speaker_markers','bad_marker_hits']:
        vr = 1000.0 * visible[feat] / max(1, visible['lexical_words'])
        sr = 1000.0 * suffix[feat] / max(1, suffix['lexical_words'])
        row[f'{feat}_visible_per_1k'] = vr
        row[f'{feat}_suffix_per_1k'] = sr
        row[f'{feat}_suffix_over_visible_rate'] = safe_div(sr, vr)
        row[f'{feat}_gain_frac_vs_visible'] = safe_div(suffix[feat], visible[feat])
        row[f'{feat}_suffix_share_of_total'] = safe_div(suffix[feat], suffix[feat] + visible[feat])
    return row


def measure_u256_content() -> dict[str, Any]:
    if sha(BASE_10M) != EXPECTED_BASE_SHA:
        raise RuntimeError('10M corpus SHA mismatch')
    if sha(TOKENIZER / 'tokenizer.json') != EXPECTED_TOK_SHA:
        raise RuntimeError('legal40k tokenizer SHA mismatch')
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER))
    visible_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    suffix_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    total_visible = collections.Counter()
    total_suffix = collections.Counter()
    hidden_row_counts = collections.Counter()
    hidden_token_total = 0
    boundary_hidden_total = 0
    rows = 0
    with BASE_10M.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            source = str(obj.get('source', 'unknown'))
            words = text.split()
            vis_words, raw_tokens, hidden_tokens, boundary_hidden = visible_word_count(text, tokenizer)
            visible_text = ' '.join(words[:vis_words])
            suffix_text = ' '.join(words[vis_words:])
            vc = feature_counts(visible_text)
            sc = feature_counts(suffix_text)
            add_counter(visible_by_source[source], vc)
            add_counter(suffix_by_source[source], sc)
            add_counter(total_visible, vc)
            add_counter(total_suffix, sc)
            if hidden_tokens > 0 or sc['surface_words'] > 0:
                hidden_row_counts[source] += 1
                hidden_token_total += hidden_tokens
                boundary_hidden_total += boundary_hidden
            rows += 1
            if (i + 1) % 20000 == 0:
                print(json.dumps({'event': 'u256_content_progress', 'rows': i + 1}), flush=True)
    feature_rows = []
    for source in sorted(set(visible_by_source) | set(suffix_by_source)):
        row = enrich_row(source, visible_by_source[source], suffix_by_source[source])
        row['hidden_rows'] = hidden_row_counts[source]
        row['suffix_word_share'] = safe_div(suffix_by_source[source]['surface_words'], total_suffix['surface_words'])
        feature_rows.append(row)
    all_row = enrich_row('__all__', total_visible, total_suffix)
    all_row['hidden_rows'] = sum(hidden_row_counts.values())
    all_row['suffix_word_share'] = 1.0
    feature_rows.insert(0, all_row)

    compatibility = {
        'Entity_state_tracking': {
            'relevant_suffix_words': int(total_suffix['capitalized'] + total_suffix['pronoun'] + total_suffix['spatial_state'] + total_suffix['action']),
            'suffix_gain_frac_vs_visible': safe_div(total_suffix['capitalized'] + total_suffix['pronoun'] + total_suffix['spatial_state'] + total_suffix['action'], total_visible['capitalized'] + total_visible['pronoun'] + total_visible['spatial_state'] + total_visible['action']),
            'reading': 'U256 adds a small but concentrated stream of dialogue/name/pronoun/spatial/action tails; this is relevant only if pending vectors show residual state/dialogue weakness rather than a pure representation-support deficit.'
        },
        'GlobalPIQA_affordance': {
            'relevant_suffix_words': int(total_suffix['action'] + total_suffix['causal_temporal'] + total_suffix['physical_object']),
            'suffix_gain_frac_vs_visible': safe_div(total_suffix['action'] + total_suffix['causal_temporal'] + total_suffix['physical_object'], total_visible['action'] + total_visible['causal_temporal'] + total_visible['physical_object']),
            'reading': 'U256 supplies little compact-FineWeb/world-knowledge material and only a modest action/causal/physical increment; it should not be the default GlobalPIQA repair unless scores specifically implicate dialogue/action tails.'
        },
        'EWoK_relation': {
            'relevant_suffix_words': int(total_suffix['action'] + total_suffix['causal_temporal'] + total_suffix['physical_object'] + total_suffix['spatial_state'] + total_suffix['mental_social']),
            'suffix_gain_frac_vs_visible': safe_div(total_suffix['action'] + total_suffix['causal_temporal'] + total_suffix['physical_object'] + total_suffix['spatial_state'] + total_suffix['mental_social'], total_visible['action'] + total_visible['causal_temporal'] + total_visible['physical_object'] + total_visible['spatial_state'] + total_visible['mental_social']),
            'reading': 'The recovered tails contain relation/social/spatial cues, but the mass is mostly CHILDES and not compact-view pairs; depth/minfreq columns must show this kind of missing experience before U256 is the strongest next run.'
        },
        'Compact_view_reinvestment': {
            'compact_fineweb_suffix_words': int(suffix_by_source['cleanqwen_fineweb_compact_view_reinvest']['surface_words']),
            'qwen_pair_suffix_words': int(suffix_by_source['qwen_pair_packed']['surface_words']),
            'reading': 'U256 is essentially not an amplifier of the compact FineWeb changed block: only 17 compact-FineWeb and 152 qwen_pair_packed suffix words per 10M pass.'
        }
    }

    return {
        'rows': rows,
        'hidden_rows_total': int(sum(hidden_row_counts.values())),
        'hidden_tokens_total': int(hidden_token_total),
        'boundary_hidden_tokens_total': int(boundary_hidden_total),
        'visible_totals': dict(total_visible),
        'suffix_totals': dict(total_suffix),
        'feature_rows': feature_rows,
        'compatibility': compatibility,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', errors='replace', newline='') as f:
        return list(csv.DictReader(f))


def load_support_tradeoff() -> dict[str, Any]:
    support_rows = [r for r in read_csv_rows(SUPPORT_CSV) if r.get('tokenizer') in KEEP_TOKENIZERS and r.get('family') in KEEP_FAMILIES]
    pool_rows = {r['tokenizer']: r for r in read_csv_rows(POOL_SUPPORT_CSV) if r.get('tokenizer') in KEEP_TOKENIZERS}
    by_tok_fam = {(r['tokenizer'], r['family']): r for r in support_rows}
    base40 = {fam: by_tok_fam[('legal_byte_bpe_40k', fam)] for fam in KEEP_FAMILIES if ('legal_byte_bpe_40k', fam) in by_tok_fam}
    out_rows = []
    for tok in KEEP_TOKENIZERS:
        for fam in KEEP_FAMILIES:
            r = by_tok_fam.get((tok, fam))
            if not r:
                continue
            br = base40.get(fam)
            row: dict[str, Any] = {
                'tokenizer': tok,
                'family': fam,
                'eval_tokens': int(float(r['eval_tokens'])),
                'p10_support': float(r['p10_support']),
                'p50_support': float(r['p50_support']),
                'frac_lt50': float(r['frac_lt50']),
                'frac_lt100': float(r['frac_lt100']),
                'frac_lt200': float(r['frac_lt200']),
                'pool_vocab_size': int(float(pool_rows[tok]['vocab_size'])) if tok in pool_rows else None,
                'pool_tokens_per_word': float(pool_rows[tok]['tokens_per_word']) if tok in pool_rows else None,
            }
            if br:
                base_tokens = int(float(br['eval_tokens']))
                row['eval_token_ratio_vs_40k'] = safe_div(row['eval_tokens'], base_tokens)
                for col in LOW_SUPPORT_COLS:
                    row[f'{col}_reduction_vs_40k'] = float(br[col]) - row[col]
            out_rows.append(row)

    # Direct route reading for minfreq candidates: how much low-support exposure is repaired and how much segmentation is surrendered.
    summary: dict[str, Any] = {}
    for tok in ['legal_byte_bpe_40k_minfreq25', 'legal_byte_bpe_40k_minfreq50', 'legal_a01_16k']:
        fams = [r for r in out_rows if r['tokenizer'] == tok]
        if not fams:
            continue
        summary[tok] = {
            'vocab_size': fams[0]['pool_vocab_size'],
            'pool_tokens_per_word': fams[0]['pool_tokens_per_word'],
            'families': {
                r['family']: {
                    'eval_token_ratio_vs_40k': r.get('eval_token_ratio_vs_40k'),
                    'frac_lt50': r['frac_lt50'],
                    'frac_lt100': r['frac_lt100'],
                    'frac_lt50_reduction_vs_40k': r.get('frac_lt50_reduction_vs_40k'),
                    'frac_lt100_reduction_vs_40k': r.get('frac_lt100_reduction_vs_40k'),
                    'p10_support': r['p10_support'],
                    'p50_support': r['p50_support'],
                }
                for r in fams
            }
        }
    return {'rows': out_rows, 'summary_vs_40k': summary}


def current_score_context() -> dict[str, Any]:
    payload = json.loads(LEGAL40K_TWO_SEED.read_text(encoding='utf-8'))
    seed430 = payload['by_seed']['43022']['legal40k']
    gaps = {k: LEADER[k] - seed430[k] for k in LEADER if k in seed430}
    return {
        'legal40k_seed43022': seed430,
        'visible_leader': LEADER,
        'gap_leader_minus_seed43022': gaps,
        'largest_remaining_score_gaps': sorted(
            [{'column': k, 'gap_needed_to_leader': v, 'base': seed430[k], 'leader': LEADER[k]} for k, v in gaps.items() if k != 'Overall'],
            key=lambda x: -x['gap_needed_to_leader']
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_float(x: Any, nd=4) -> str:
    if x is None:
        return 'NA'
    try:
        if not math.isfinite(float(x)):
            return 'NA'
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)


def write_note(payload: dict[str, Any]) -> None:
    u = payload['u256_content']
    s = payload['support_tradeoff']
    ctx = payload['score_context']
    all_row = next(r for r in u['feature_rows'] if r['source'] == '__all__')
    childes_row = next((r for r in u['feature_rows'] if r['source'] == 'childes'), None)
    compact_row = next((r for r in u['feature_rows'] if r['source'] == 'cleanqwen_fineweb_compact_view_reinvest'), None)
    lines: list[str] = []
    lines.append('# research — U256 / representation route coupling measurement')
    lines.append('')
    lines.append('CPU-only analysis without training or model evaluation. Existing U256 suffix measurements are compared with tokenizer-support evidence; a proposed H100 comparison depends on depth and minfreq50 score vectors, not implementation readiness.')
    lines.append('')
    lines.append('## U256 recovered mass as a capability intervention')
    lines.append('')
    lines.append(f"- Newly visible full suffix words per 10M pass: {int(u['suffix_totals']['surface_words']):,}; hidden-token total: {u['hidden_tokens_total']:,}; hidden rows: {u['hidden_rows_total']:,}.")
    lines.append(f"- Overall word gain versus row256-visible words: {fmt_float(all_row['u256_word_gain_frac'], 4)}; compact FineWeb suffix words: {int(compact_row['suffix_surface_words']) if compact_row else 0}; CHILDES suffix words: {int(childes_row['suffix_surface_words']) if childes_row else 0}.")
    lines.append(f"- Feature gain versus visible prefix mass: action {fmt_float(all_row['action_gain_frac_vs_visible'], 4)}, causal/temporal {fmt_float(all_row['causal_temporal_gain_frac_vs_visible'], 4)}, physical/object {fmt_float(all_row['physical_object_gain_frac_vs_visible'], 4)}, spatial/state {fmt_float(all_row['spatial_state_gain_frac_vs_visible'], 4)}, mental/social {fmt_float(all_row['mental_social_gain_frac_vs_visible'], 4)}, pronoun {fmt_float(all_row['pronoun_gain_frac_vs_visible'], 4)}, capitalized/name-like {fmt_float(all_row['capitalized_gain_frac_vs_visible'], 4)}.")
    lines.append('')
    lines.append('This supports the interpretation: U256 is mostly additional visibility for dialogue/transcript tails and name/pronoun/spatial/action material, not a direct compact-view reinvestment amplifier.')
    lines.append('')
    lines.append('## Representation support corner from existing CPU evidence')
    lines.append('')
    lines.append('| tokenizer | family | eval-token ratio vs 40k | frac<50 | frac<100 | p10 support | p50 support |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for tok in ['legal_byte_bpe_40k', 'legal_byte_bpe_40k_minfreq25', 'legal_byte_bpe_40k_minfreq50', 'legal_a01_16k']:
        for fam in ['EWoK', 'Entity', 'GlobalPIQA', 'Supplement', 'SuperGLUE']:
            row = next((r for r in s['rows'] if r['tokenizer'] == tok and r['family'] == fam), None)
            if row:
                lines.append(f"| {tok} | {fam} | {fmt_float(row.get('eval_token_ratio_vs_40k'), 4)} | {fmt_float(row['frac_lt50'], 4)} | {fmt_float(row['frac_lt100'], 4)} | {fmt_float(row['p10_support'], 1)} | {fmt_float(row['p50_support'], 1)} |")
    lines.append('')
    lines.append('Minfreq50 strongly repairs low-support evaluation exposure on EWoK and GlobalPIQA relative to plain 40k, but it also surrenders some segmentation efficiency. Therefore the pending A02 minfreq50 vector is essential: if its score vector preserves GlobalPIQA/Entity while depth preserves the 40k language columns, the stronger next problem is a legal representation corner rather than U256.')
    lines.append('')
    lines.append('## Current score context for reading the pending vectors')
    lines.append('')
    lines.append(f"- Legal40k 8x480 seed43022 Overall: {fmt_float(ctx['legal40k_seed43022']['Overall'], 6)}; visible leader: {fmt_float(ctx['visible_leader']['Overall'], 2)}; gap: {fmt_float(ctx['gap_leader_minus_seed43022']['Overall'], 6)}.")
    lines.append('- Largest positive gaps to the leader: ' + ', '.join(f"{x['column']} {fmt_float(x['gap_needed_to_leader'], 3)}" for x in ctx['largest_remaining_score_gaps'][:5]))
    lines.append('')
    lines.append('## Route consequence')
    lines.append('')
    lines.append('When the depth and A02 minfreq50 vectors arrive, read them together. Complementary recovery from depth and token support should move the next construction toward a support-aware legal representation coordinate. U256@0.15 remains launch-ready but should be used first only if the combined score pattern makes extra dialogue/social/state-tracking experience a credible frontier route; it is not the automatic successor to depth.')
    lines.append('')
    lines.append(f'JSON: `{OUT_JSON}`')
    lines.append(f'CSV: `{FEATURE_CSV}`, `{SUPPORT_ROUTE_CSV}`')
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    u256_content = measure_u256_content()
    support_tradeoff = load_support_tradeoff()
    score_context = current_score_context()
    payload = {
        'status': 'U256_REPRESENTATION_ROUTE_COUPLING',
        'created_utc': now(),
        'purpose': 'Use existing CPU evidence to couple U256 recovered content with representation-support evidence before the next expensive run is chosen.',
        'no_training_or_model_evaluation': True,
        'no_managed_task_state_query': True,
        'inputs': {
            'base_10M': str(BASE_10M),
            'base_10M_sha256': sha(BASE_10M),
            'legal40k_tokenizer': str(TOKENIZER),
            'legal40k_tokenizer_sha256': sha(TOKENIZER / 'tokenizer.json'),
            'support_csv': str(SUPPORT_CSV),
            'legal40k_two_seed': str(LEGAL40K_TWO_SEED),
        },
        'u256_content': u256_content,
        'support_tradeoff': support_tradeoff,
        'score_context': score_context,
        'route_reading': [
            'U256 mainly exposes CHILDES/dialogue tail material and almost no compact FineWeb changed-block material.',
            'Plain 40k leaves much higher low-support evaluation exposure on EWoK and GlobalPIQA than minfreq50, while minfreq50 costs segmentation length.',
            'The next expensive route should be selected from the actual depth vector and A02 minfreq50 vector; U256 is strongest only for a residual dialogue/social/state-tracking pattern.',
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(FEATURE_CSV, u256_content['feature_rows'])
    write_csv(SUPPORT_ROUTE_CSV, support_tradeoff['rows'])
    write_note(payload)
    print(json.dumps({
        'status': payload['status'],
        'u256_suffix_words': u256_content['suffix_totals']['surface_words'],
        'u256_compact_fineweb_suffix_words': u256_content['compatibility']['Compact_view_reinvestment']['compact_fineweb_suffix_words'],
        'u256_qwen_pair_suffix_words': u256_content['compatibility']['Compact_view_reinvestment']['qwen_pair_suffix_words'],
        'all_action_gain_frac': u256_content['feature_rows'][0]['action_gain_frac_vs_visible'],
        'all_spatial_gain_frac': u256_content['feature_rows'][0]['spatial_state_gain_frac_vs_visible'],
        'out_json': str(OUT_JSON),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
