#!/usr/bin/env python3
"""research: error-conditioned support-sharing probe for legal40k.

CPU-only. No training and no model evaluation. This script tests whether the
support-shared legal40k representation idea is connected to actual mistakes in the
existing legal40k 8x480 seed43022 endpoint.  It uses only existing official
prediction files and official evaluation data.

Question: are low-support legal40k tokens (especially those decomposable into
high-support legal16k components from the same allowed 10M corpus) enriched in the
answer-discriminating spans of wrong examples?  If yes, a support-sharing
architecture targets a genuinely unsaturated quantity. If no, the route is mostly a
plausible construction without error-conditioned evidence.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

ROOT = Path('.').resolve()
A01 = ROOT / 'experiments/archive/representation_and_objectives'
A02 = ROOT / 'experiments/archive/frontier_consolidation'
if str(A01 / 'scripts') not in sys.path:
    sys.path.insert(0, str(A01 / 'scripts'))
import tokenizer_support_spectrum as supportbase  # noqa: E402

LEGAL40K = A01 / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
LEGAL16K = A01 / 'data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'
PRISTINE = A01 / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval'
GLOBALPIQA = A01 / 'data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval'
PRED_ROOT = A01 / 'data/legal40k_accum_seed43022_pristine_collate/results/hf_model/main/zero_shot/mlm'
OUT_DIR = A02 / 'data/support_error_conditioned_probe'
OUT_JSON = OUT_DIR / 'support_error_conditioned_probe.json'
ITEM_CSV = OUT_DIR / 'support_error_conditioned_items.csv'
FAMILY_CSV = OUT_DIR / 'support_error_conditioned_by_family.csv'
NOTE = (ROOT / 'research/notes/frontier_consolidation/support_error_conditioned_probe.md')

EXPECTED_40K_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
EXPECTED_16K_SHA = '4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738'
THRESHOLDS = (20, 50, 100)


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_tok(path: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path / 'tokenizer.json'))
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def token_multiset(tok: Tokenizer, text: str, specials: set[int]) -> collections.Counter[int]:
    if not text:
        return collections.Counter()
    ids = [int(x) for x in tok.encode(text, add_special_tokens=False).ids]
    return collections.Counter(i for i in ids if i not in specials)


def diff_multisets(candidates: list[collections.Counter[int]]) -> collections.Counter[int]:
    """Token instances whose multiplicity differs across candidates."""
    if not candidates:
        return collections.Counter()
    all_ids: set[int] = set()
    for c in candidates:
        all_ids.update(c)
    out = collections.Counter()
    for tid in all_ids:
        vals = [c.get(tid, 0) for c in candidates]
        common = min(vals)
        extra = sum(v - common for v in vals)
        if extra:
            out[tid] += extra
    return out


def special_ids(rec: dict[str, Any]) -> set[int]:
    return {int(v) for v in rec.get('special_ids', {}).values() if v is not None}


def decode_piece(tok: Tokenizer, token_id: int, token_text: str) -> str:
    try:
        s = tok.decode([int(token_id)], skip_special_tokens=False)
        if s:
            return s
    except Exception:
        pass
    return token_text


def support_and_components() -> tuple[Tokenizer, dict[int, int], set[int], dict[int, dict[str, Any]]]:
    tok40 = load_tok(LEGAL40K)
    tok16 = load_tok(LEGAL16K)
    rec40 = supportbase.count_pool_support('legal40k_step072_error_recount', tok40)
    rec16 = supportbase.count_pool_support('legal16k_step072_error_recount', tok16)
    counts40 = {i: int(v) for i, v in enumerate(rec40['counts_by_id'])}
    counts16 = [int(v) for v in rec16['counts_by_id']]
    idtok40 = {int(k): v for k, v in rec40['id_to_token'].items()}
    idtok16 = {int(k): v for k, v in rec16['id_to_token'].items()}
    specials40 = special_ids(rec40)
    cmap: dict[int, dict[str, Any]] = {}
    for tid, token_text in idtok40.items():
        if tid in specials40:
            continue
        surface = decode_piece(tok40, tid, token_text)
        enc16 = tok16.encode(surface, add_special_tokens=False)
        comp_ids = [int(x) for x in enc16.ids]
        comp_counts = [counts16[x] if x < len(counts16) else 0 for x in comp_ids]
        cmin = min(comp_counts) if comp_counts else 0
        cgeo = math.exp(sum(math.log(max(1, c)) for c in comp_counts) / len(comp_counts)) if comp_counts else 0.0
        cmap[tid] = {
            'direct40': counts40.get(tid, 0),
            'component_n16': len(comp_ids),
            'component_min16': cmin,
            'component_geo16': cgeo,
            'component_ids16': comp_ids,
            'component_tokens16': [idtok16.get(x, '') for x in comp_ids],
        }
    return tok40, counts40, specials40, cmap


def item_features(disc: collections.Counter[int], cmap: dict[int, dict[str, Any]]) -> dict[str, Any]:
    total = sum(disc.values())
    out: dict[str, Any] = {'disc_tokens': int(total)}
    low_ids: dict[int, list[int]] = {th: [] for th in THRESHOLDS}
    rescue_ids: dict[int, list[int]] = {th: [] for th in THRESHOLDS}
    for tid, mult in disc.items():
        comp = cmap.get(int(tid))
        if comp is None:
            continue
        d = int(comp['direct40'])
        cmin = int(comp['component_min16'])
        for th in THRESHOLDS:
            if d < th:
                low_ids[th].extend([int(tid)] * int(mult))
                if cmin >= 50:
                    rescue_ids[th].extend([int(tid)] * int(mult))
    for th in THRESHOLDS:
        low = len(low_ids[th])
        res = len(rescue_ids[th])
        out[f'disc_low_lt{th}_count'] = int(low)
        out[f'disc_low_lt{th}_frac'] = (low / total) if total else 0.0
        out[f'disc_rescue_lt{th}_ge50_count'] = int(res)
        out[f'disc_rescue_lt{th}_ge50_frac_all'] = (res / total) if total else 0.0
        out[f'disc_rescue_lt{th}_ge50_frac_low'] = (res / low) if low else 0.0
    # A continuous opportunity signal: low-support mass weighted by the log support
    # of its best same-corpus legal16 decomposition.
    opp = 0.0
    min_directs: list[int] = []
    comp_mins: list[int] = []
    for tid, mult in disc.items():
        comp = cmap.get(int(tid))
        if not comp:
            continue
        d = int(comp['direct40'])
        cmin = int(comp['component_min16'])
        min_directs.extend([d] * int(mult))
        comp_mins.extend([cmin] * int(mult))
        if d < 100 and cmin >= 50:
            opp += mult * math.log1p(cmin) / math.log1p(max(1, d))
    out['support_share_opportunity'] = opp / max(1, total)
    out['disc_direct40_min'] = min(min_directs) if min_directs else None
    out['disc_component_min16_median'] = statistics.median(comp_mins) if comp_mins else None
    return out


def correct_str(a: Any, b: Any) -> bool:
    return str(a).strip() == str(b).strip()


def process_blimp_like(name: str, pred_path: Path, data_dir: Path, tok: Tokenizer, specials: set[int], cmap: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding='utf-8'))
    rows: list[dict[str, Any]] = []
    for subtask, block in preds.items():
        dpath = data_dir / f'{subtask}.jsonl'
        if not dpath.exists():
            continue
        plist = block['predictions']
        for idx, (pr, obj) in enumerate(zip(plist, iter_jsonl(dpath))):
            good = obj.get('sentence_good', '')
            bad = obj.get('sentence_bad', '')
            disc = diff_multisets([token_multiset(tok, good, specials), token_multiset(tok, bad, specials)])
            feat = item_features(disc, cmap)
            feat.update({
                'family': name,
                'subtask': subtask,
                'item_index': idx,
                'correct': int(correct_str(pr.get('pred', ''), good)),
            })
            rows.append(feat)
    return rows


def process_ewok(tok: Tokenizer, specials: set[int], cmap: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    pred_path = PRED_ROOT / 'ewok/ewok_filtered/predictions.json'
    preds = json.loads(pred_path.read_text(encoding='utf-8'))
    rows: list[dict[str, Any]] = []
    for subtask, block in preds.items():
        dpath = PRISTINE / 'ewok_filtered' / f'{subtask}.jsonl'
        plist = block['predictions']
        for idx, (pr, obj) in enumerate(zip(plist, iter_jsonl(dpath))):
            c1 = obj.get('Context1', '')
            c2 = obj.get('Context2', '')
            t1 = obj.get('Target1', '')
            cand1 = (c1 + ' ' + t1).strip()
            cand2 = (c2 + ' ' + t1).strip()
            disc = diff_multisets([token_multiset(tok, cand1, specials), token_multiset(tok, cand2, specials)])
            feat = item_features(disc, cmap)
            feat.update({
                'family': 'EWoK',
                'subtask': subtask,
                'item_index': idx,
                'correct': int(correct_str(pr.get('pred', ''), cand1)),
            })
            rows.append(feat)
    return rows


def process_comps(tok: Tokenizer, specials: set[int], cmap: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    pred_path = PRED_ROOT / 'comps/comps/predictions.json'
    preds = json.loads(pred_path.read_text(encoding='utf-8'))
    mapping = {
        'base': 'comps_base',
        'wugs_dist_before': 'comps_wugs_dist-before',
        'wugs_dist_in_between': 'comps_wugs_dist-in-between',
        'wugs': 'comps_wugs',
    }
    rows: list[dict[str, Any]] = []
    for subtask, block in preds.items():
        dpath = PRISTINE / 'comps' / f'{mapping[subtask]}.jsonl'
        plist = block['predictions']
        for idx, (pr, obj) in enumerate(zip(plist, iter_jsonl(dpath))):
            prop = obj.get('property_phrase', '') or ''
            good = (obj.get('prefix_acceptable', '') + ' ' + prop).strip()
            bad = (obj.get('prefix_unacceptable', '') + ' ' + prop).strip()
            disc = diff_multisets([token_multiset(tok, good, specials), token_multiset(tok, bad, specials)])
            feat = item_features(disc, cmap)
            feat.update({
                'family': 'COMPS',
                'subtask': subtask,
                'item_index': idx,
                'correct': int(correct_str(pr.get('pred', ''), good)),
            })
            rows.append(feat)
    return rows


def process_globalpiqa(split: str, tok: Tokenizer, specials: set[int], cmap: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    pred_path = PRED_ROOT / f'{split}/{split}/predictions.json'
    preds = json.loads(pred_path.read_text(encoding='utf-8'))
    dpath = GLOBALPIQA / split / 'eng_latn.jsonl'
    rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(iter_jsonl(dpath)):
        exid = obj['example_id']
        pred = preds[exid]['predictions'][0]['pred']
        sols = [obj.get(f'solution{i}') for i in range(4)]
        sols = [s for s in sols if isinstance(s, str) and s.strip()]
        disc = diff_multisets([token_multiset(tok, s, specials) for s in sols])
        gold = obj[f"solution{obj['label']}"]
        feat = item_features(disc, cmap)
        feat.update({
            'family': 'GlobalPIQA_parallel' if split.endswith('parallel') and not split.endswith('nonparallel') else 'GlobalPIQA_nonparallel',
            'subtask': split,
            'item_index': idx,
            'correct': int(correct_str(pred, gold)),
        })
        rows.append(feat)
    return rows


def rank_auc_wrong_high(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [(float(r.get(field, 0.0) or 0.0), int(r['correct'])) for r in rows]
    wrong = [v for v, c in vals if c == 0]
    corr = [v for v, c in vals if c == 1]
    if not wrong or not corr:
        return None
    # Mann-Whitney AUC: probability a wrong item has higher feature than correct item, ties half.
    sorted_vals = sorted([(v, 0) for v in wrong] + [(v, 1) for v in corr], key=lambda x: x[0])
    ranksum_wrong = 0.0
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j][0] == sorted_vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        wcount = sum(1 for _, group in sorted_vals[i:j] if group == 0)
        ranksum_wrong += wcount * avg_rank
        i = j
    n_w = len(wrong); n_c = len(corr)
    u = ranksum_wrong - n_w * (n_w + 1) / 2.0
    return u / (n_w * n_c)


def summarize_family(rows: list[dict[str, Any]], family: str, subtask: str | None = None) -> dict[str, Any]:
    subset = [r for r in rows if r['family'] == family and (subtask is None or r['subtask'] == subtask)]
    n = len(subset)
    correct = sum(int(r['correct']) for r in subset)
    wrong = n - correct
    def mean_of(field: str, condition: int | None = None) -> float | None:
        vals = [float(r.get(field, 0.0) or 0.0) for r in subset if condition is None or int(r['correct']) == condition]
        return sum(vals) / len(vals) if vals else None
    out = {
        'family': family,
        'subtask': subtask or '__ALL__',
        'items': n,
        'correct': correct,
        'wrong': wrong,
        'score_pct': (100.0 * correct / n) if n else None,
    }
    for field in ['disc_tokens', 'disc_low_lt50_frac', 'disc_low_lt100_frac', 'disc_rescue_lt50_ge50_frac_all', 'disc_rescue_lt100_ge50_frac_all', 'support_share_opportunity']:
        out[f'{field}_correct_mean'] = mean_of(field, 1)
        out[f'{field}_wrong_mean'] = mean_of(field, 0)
        c = out[f'{field}_correct_mean']; w = out[f'{field}_wrong_mean']
        out[f'{field}_wrong_minus_correct'] = (w - c) if w is not None and c is not None else None
        out[f'{field}_wrong_high_auc'] = rank_auc_wrong_high(subset, field)
    return out


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return 'NA'
    try:
        v = float(x)
        return f'{v:.{nd}f}' if math.isfinite(v) else 'NA'
    except Exception:
        return str(x)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def write_note(payload: dict[str, Any]) -> None:
    rows = payload['family_summaries']
    lines: list[str] = []
    lines.append('# research — error-conditioned support-sharing probe')
    lines.append('')
    lines.append('CPU-only. No model training and no new model evaluation. Existing A01 legal40k 8x480 seed43022 official-compatible predictions are joined to gold data to ask whether low-support legal40k tokens in answer-discriminating spans are enriched in wrong examples.')
    lines.append('')
    lines.append('## Family-level error conditioning')
    lines.append('')
    lines.append('| family | items | score | Δ wrong-correct frac<50 | AUC wrong-high frac<50 | Δ wrong-correct frac<100 | AUC wrong-high frac<100 | Δ wrong-correct rescue<100/ge50 | AUC wrong-high opportunity |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    preferred = ['BLiMP','Supplement','EWoK','COMPS','GlobalPIQA_parallel','GlobalPIQA_nonparallel']
    bykey = {(r['family'], r['subtask']): r for r in rows}
    for fam in preferred:
        r = bykey.get((fam, '__ALL__'))
        if not r:
            continue
        lines.append(
            f"| {fam} | {r['items']} | {fmt(r['score_pct'],2)} | "
            f"{fmt(r['disc_low_lt50_frac_wrong_minus_correct'])} | {fmt(r['disc_low_lt50_frac_wrong_high_auc'])} | "
            f"{fmt(r['disc_low_lt100_frac_wrong_minus_correct'])} | {fmt(r['disc_low_lt100_frac_wrong_high_auc'])} | "
            f"{fmt(r['disc_rescue_lt100_ge50_frac_all_wrong_minus_correct'])} | {fmt(r['support_share_opportunity_wrong_high_auc'])} |"
        )
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('')
    lines.extend(payload['reading'])
    lines.append('')
    lines.append(f"Full JSON: `{OUT_JSON}`")
    lines.append(f"CSVs: `{FAMILY_CSV}`, `{ITEM_CSV}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha40 = sha(LEGAL40K / 'tokenizer.json')
    sha16 = sha(LEGAL16K / 'tokenizer.json')
    if sha40 != EXPECTED_40K_SHA:
        raise RuntimeError(f'legal40k sha mismatch: {sha40}')
    if sha16 != EXPECTED_16K_SHA:
        raise RuntimeError(f'legal16k sha mismatch: {sha16}')
    tok40, counts40, specials40, cmap = support_and_components()
    item_rows: list[dict[str, Any]] = []
    item_rows += process_blimp_like('BLiMP', PRED_ROOT / 'blimp/blimp_filtered/predictions.json', PRISTINE / 'blimp_filtered', tok40, specials40, cmap)
    item_rows += process_blimp_like('Supplement', PRED_ROOT / 'blimp/supplement_filtered/predictions.json', PRISTINE / 'supplement_filtered', tok40, specials40, cmap)
    item_rows += process_ewok(tok40, specials40, cmap)
    item_rows += process_comps(tok40, specials40, cmap)
    item_rows += process_globalpiqa('global_piqa_parallel', tok40, specials40, cmap)
    item_rows += process_globalpiqa('global_piqa_nonparallel', tok40, specials40, cmap)

    fams = ['BLiMP', 'Supplement', 'EWoK', 'COMPS', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel']
    summaries = [summarize_family(item_rows, fam) for fam in fams]
    # Also preserve a few subtask-level summaries for EWoK/Supplement, where the route has been most brittle.
    for fam in ['Supplement', 'EWoK']:
        for st in sorted({r['subtask'] for r in item_rows if r['family'] == fam}):
            summaries.append(summarize_family(item_rows, fam, st))

    key = {r['family']: r for r in summaries if r['subtask'] == '__ALL__'}
    reading = []
    gpnp = key.get('GlobalPIQA_nonparallel')
    gpp = key.get('GlobalPIQA_parallel')
    ewok = key.get('EWoK')
    supp = key.get('Supplement')
    comps = key.get('COMPS')
    blimp = key.get('BLiMP')
    reading.append(
        '- This probe conditions A01 legal40k errors on the support-sharing quantity itself: low-support legal40k tokens in answer-discriminating spans whose legal16 components have much higher same-corpus support.'
    )
    if gpnp:
        reading.append(
            f"- GlobalPIQA_nonparallel, the column repeatedly moved by word-mean/minfreq50, shows score {fmt(gpnp['score_pct'],2)} with wrong-high AUC {fmt(gpnp['disc_low_lt100_frac_wrong_high_auc'])} for disc frac<100 and {fmt(gpnp['support_share_opportunity_wrong_high_auc'])} for the continuous support-sharing opportunity."
        )
    if gpp:
        reading.append(
            f"- GlobalPIQA_parallel shows score {fmt(gpp['score_pct'],2)} with wrong-high AUC {fmt(gpp['disc_low_lt100_frac_wrong_high_auc'])}; this separates whether the support signal is only the nonparallel idiosyncrasy."
        )
    if ewok:
        reading.append(
            f"- EWoK shows score {fmt(ewok['score_pct'],2)} with wrong-correct disc frac<100 delta {fmt(ewok['disc_low_lt100_frac_wrong_minus_correct'])} and wrong-high AUC {fmt(ewok['disc_low_lt100_frac_wrong_high_auc'])}; this is the load-bearing test for broad transfer."
        )
    if supp:
        reading.append(
            f"- Supplement shows score {fmt(supp['score_pct'],2)} with wrong-correct disc frac<100 delta {fmt(supp['disc_low_lt100_frac_wrong_minus_correct'])}; support-sharing must not damage the high Supplement behavior that minfreq50 lost."
        )
    if comps:
        reading.append(
            f"- COMPS shows score {fmt(comps['score_pct'],2)} with wrong-high AUC {fmt(comps['disc_low_lt100_frac_wrong_high_auc'])}; its huge item count makes small effects reliable but not necessarily route-deciding."
        )
    if blimp:
        reading.append(
            f"- BLiMP shows score {fmt(blimp['score_pct'],2)} with wrong-high AUC {fmt(blimp['disc_low_lt100_frac_wrong_high_auc'])}; support-sharing should preserve BLiMP rather than repeat minfreq50's language-column trade."
        )
    reading.append('- Treat positive AUC/delta as construction evidence, not as score proof. Training would still require a single-variable, official-compatible trainer and a preliminary implementation check.')

    payload = {
        'status': 'SUPPORT_ERROR_CONDITIONED_PROBE',
        'created_utc': now(),
        'no_training': True,
        'no_new_model_evaluation': True,
        'existing_endpoint': 'A01 legal40k 8x480 compact_view_reinvest seed43022, chck_100M, Overall 41.140577774478444',
        'inputs': {
            'legal40k_tokenizer': str(LEGAL40K),
            'legal40k_tokenizer_sha256': sha40,
            'legal16k_tokenizer': str(LEGAL16K),
            'legal16k_tokenizer_sha256': sha16,
            'pred_root': str(PRED_ROOT),
            'official_eval_root': str(PRISTINE),
            'globalpiqa_root': str(GLOBALPIQA),
        },
        'feature_definitions': {
            'disc_low_lt50_frac': 'fraction of answer-discriminating legal40k token instances with allowed-pool count <50',
            'disc_low_lt100_frac': 'fraction of answer-discriminating legal40k token instances with allowed-pool count <100',
            'disc_rescue_lt100_ge50_frac_all': 'fraction of answer-discriminating token instances with legal40k count <100 and all legal16 components count >=50',
            'support_share_opportunity': 'for each discriminating token with count40<100 and component_min16>=50, sum log1p(component_min16)/log1p(count40) divided by disc token count',
            'wrong_high_auc': 'Mann-Whitney AUC: probability a wrong item has a higher feature than a correct item, ties half',
        },
        'family_summaries': summaries,
        'reading': reading,
        'files': {'json': str(OUT_JSON), 'note': str(NOTE), 'family_csv': str(FAMILY_CSV), 'item_csv': str(ITEM_CSV)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(FAMILY_CSV, summaries)
    # Item CSV can be large but still manageable; write the key fields needed for later construction.
    write_csv(ITEM_CSV, item_rows)
    write_note(payload)
    print(json.dumps({
        'status': payload['status'],
        'items': len(item_rows),
        'family_rows': len(summaries),
        'summary': [{
            'family': r['family'],
            'score': r['score_pct'],
            'wrong_minus_correct_lt100': r['disc_low_lt100_frac_wrong_minus_correct'],
            'auc_wrong_high_lt100': r['disc_low_lt100_frac_wrong_high_auc'],
            'auc_wrong_high_opportunity': r['support_share_opportunity_wrong_high_auc'],
        } for r in summaries if r['subtask'] == '__ALL__'],
        'out_json': str(OUT_JSON),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
