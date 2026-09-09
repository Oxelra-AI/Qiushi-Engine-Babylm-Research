#!/usr/bin/env python3
"""Independent audit for the research permuted-companion correspondence control.

This script does not reuse the materializer's construction functions.  It reads
original MAX pairs, research row metadata/view pool, and the written research
assignment/pool/training stream, then verifies that only source-to-view
correspondence was broken while row geometry, suffix, compact-rewrite multiset,
and 100M order were preserved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import time
from typing import Any

ROOT = pathlib.Path('.').resolve()
WS = ROOT / 'experiments/archive/frontier_consolidation'
PAIRS = WS / 'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
VIEW_POOL = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'
VIEW_META = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
PERM_DIR = WS / 'data/dose_2p64x_permuted_companion_rowholdout_pools'
PERM_POOL = PERM_DIR / 'compact_permuted_view_dose2p64x_10M.jsonl'
PERM_TRAIN = PERM_DIR / 'compact_permuted_view_dose2p64x_100M.jsonl'
PERM_ASSIGN = PERM_DIR / 'compact_permuted_view_dose2p64x_companion_assignment.jsonl'
PERM_ROW_META = PERM_DIR / 'compact_permuted_view_dose2p64x_changed_block_rows_meta.jsonl'
PERM_META = PERM_DIR / 'permuted_companion_rowholdout_metadata.json'
TOKEN_GEOM = PERM_DIR / 'permuted_companion_token_geometry.json'
OUT_DIR = WS / 'data/permuted_companion_independent_audit'
TOTAL_WORDS = 10_000_000
PASSES = 10
STREAM_SEED = 82914124 + 7000


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def norm(text: Any) -> str:
    return ' '.join(str(text or '').split())


def wc(text: Any) -> int:
    return len(norm(text).split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: Any) -> str:
    return hashlib.sha256(norm(text).encode('utf-8')).hexdigest()


def multiset_digest(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(hashlib.sha256(norm(x).encode('utf-8')).hexdigest() for x in items):
        h.update(s.encode('ascii')); h.update(b'\n')
    return h.hexdigest()


def iter_jsonl(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_pairs() -> dict[str, dict[str, Any]]:
    pairs = {}
    for d in iter_jsonl(PAIRS):
        pid = str(d.get('pair_id') or '')
        src = norm(d.get('source_text'))
        rew = norm(d.get('rewrite_text'))
        sw = int(d.get('source_words') or wc(src))
        rw = int(d.get('rewrite_words') or wc(rew))
        if wc(src) != sw or wc(rew) != rw:
            raise RuntimeError(f'pair word mismatch {pid}')
        pairs[pid] = {**d, 'source_text': src, 'rewrite_text': rew, 'source_words': sw, 'rewrite_words': rw, 'doc_id': str(d.get('doc_id') or ''), 'sentence_id': str(d.get('sentence_id') or '')}
    return pairs


def load_view_meta(pairs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows=[]; pair_to_slot={}
    for m in iter_jsonl(VIEW_META):
        ids=[str(x) for x in (m.get('pair_ids') or [])]
        if not ids:
            continue
        source_words=sum(int(pairs[p]['source_words']) for p in ids)
        rewrite_words=sum(int(pairs[p]['rewrite_words']) for p in ids)
        if source_words + rewrite_words != int(m['words']):
            raise RuntimeError(f'view row word mismatch {m.get("row_index")}')
        row={'row_index':int(m['row_index']),'example_id':int(m['example_id']),'words':int(m['words']),'pair_ids':ids,'source_words':source_words,'rewrite_words':rewrite_words,'docset':{pairs[p]['doc_id'] for p in ids}}
        rows.append(row)
        for j,pid in enumerate(ids):
            pair_to_slot[pid]={'row_index':row['row_index'],'slot_index':j,'doc_id':pairs[pid]['doc_id'],'rewrite_words':pairs[pid]['rewrite_words']}
    if [r['row_index'] for r in rows] != list(range(len(rows))):
        raise RuntimeError('view meta pair rows not contiguous prefix')
    return rows,pair_to_slot


def score_counts(vals: list[int]) -> dict[str, Any]:
    if not vals:
        return {'n':0}
    xs=sorted(vals)
    return {'n':len(xs),'min':xs[0],'median':xs[len(xs)//2],'max':xs[-1],'mean':sum(xs)/len(xs),'sum':sum(xs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--check-training-order', action='store_true')
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    t0=time.time()

    pairs=load_pairs()
    view_rows,pair_to_slot=load_view_meta(pairs)
    assignments=read_jsonl(PERM_ASSIGN)
    perm_rows=read_jsonl(PERM_POOL)
    view_pool=read_jsonl(VIEW_POOL)
    perm_row_meta=read_jsonl(PERM_ROW_META)
    meta=json.loads(PERM_META.read_text(encoding='utf-8')) if PERM_META.exists() else {}
    geom=json.loads(TOKEN_GEOM.read_text(encoding='utf-8')) if TOKEN_GEOM.exists() else {}

    all_pair_ids=set(pairs)
    target_ids=[str(a['target_pair_id']) for a in assignments]
    donor_ids=[str(a['donor_pair_id']) for a in assignments]
    target_unique=set(target_ids); donor_unique=set(donor_ids)
    assignment_map={str(a['target_pair_id']):str(a['donor_pair_id']) for a in assignments}

    bad=[]
    same_doc=same_row=same_pair=donor_doc_in_target_row=0
    row_assignment_counts=collections.Counter()
    row_changed_length=0
    donor_dist=[]
    for a in assignments:
        t=str(a['target_pair_id']); d=str(a['donor_pair_id'])
        if t not in pairs or d not in pairs or t not in pair_to_slot or d not in pair_to_slot:
            bad.append({'kind':'missing_pair_or_slot','assignment':a}); continue
        ts=pair_to_slot[t]; ds=pair_to_slot[d]
        row_assignment_counts[int(ts['row_index'])]+=1
        if t==d: same_pair+=1
        if pairs[t]['doc_id']==pairs[d]['doc_id']: same_doc+=1
        if ts['row_index']==ds['row_index']: same_row+=1
        if pairs[d]['doc_id'] in view_rows[int(ts['row_index'])]['docset']: donor_doc_in_target_row+=1
        if int(pairs[t]['rewrite_words']) != int(pairs[d]['rewrite_words']): row_changed_length+=1
        donor_dist.append(abs(int(ds['row_index'])-int(ts['row_index'])))

    # Reconstruct permuted prefix exactly from original source and donor rewrite texts.
    reconstructed_prefix=[]
    row_word_residuals=[]
    for r in view_rows:
        segs=[]
        src_words=donor_words=0
        for pid in r['pair_ids']:
            donor=assignment_map[pid]
            segs.append(pairs[pid]['source_text'])
            segs.append(pairs[donor]['rewrite_text'])
            src_words+=int(pairs[pid]['source_words'])
            donor_words+=int(pairs[donor]['rewrite_words'])
        text=norm(' '.join(segs))
        words=wc(text)
        row_word_residuals.append(words-int(r['words']))
        reconstructed_prefix.append({'text':text,'words':words,'example_id':int(r['example_id']),'source':'compact_permuted_view_dose2p64x_matched_rowholdout'})
    prefix_match=True
    prefix_mismatches=[]
    for i,rec in enumerate(reconstructed_prefix):
        if i>=len(perm_rows) or rec != {k:perm_rows[i].get(k) for k in ['text','words','example_id','source']}:
            prefix_match=False
            if len(prefix_mismatches)<10:
                prefix_mismatches.append({'row_index':i,'expected_words':rec.get('words'),'observed':perm_rows[i] if i<len(perm_rows) else None})
    suffix_match = perm_rows[len(view_rows):] == view_pool[len(view_rows):]
    row_lengths_match = [int(r.get('words') or wc(r.get('text'))) for r in perm_rows] == [int(r.get('words') or wc(r.get('text'))) for r in view_pool]
    pool_words=sum(int(r.get('words') or wc(r.get('text'))) for r in perm_rows)

    source_digest_perm=multiset_digest([pairs[pid]['source_text'] for pid in target_ids if pid in pairs])
    source_digest_view=multiset_digest([pairs[pid]['source_text'] for pid in pairs])
    rewrite_digest_donor=multiset_digest([pairs[pid]['rewrite_text'] for pid in donor_ids if pid in pairs])
    rewrite_digest_view=multiset_digest([pairs[pid]['rewrite_text'] for pid in pairs])

    training = {'checked': False}
    if args.check_training_order:
        pool_keys=[(text_hash(r.get('text')), int(r.get('words') or wc(r.get('text'))), int(r.get('example_id'))) for r in perm_rows]
        rows_seen=words_seen=0; mismatches=[]
        with PERM_TRAIN.open('r',encoding='utf-8') as f:
            for pass_i in range(PASSES):
                order=list(range(len(pool_keys)))
                random.Random(STREAM_SEED+1000+pass_i).shuffle(order)
                for j,idx in enumerate(order):
                    line=f.readline()
                    if not line:
                        mismatches.append({'pass':pass_i,'position':j,'error':'unexpected_eof'}); break
                    obj=json.loads(line)
                    key=(text_hash(obj.get('text')), int(obj.get('words') or wc(obj.get('text'))), int(obj.get('example_id')))
                    rows_seen+=1; words_seen+=key[1]
                    if key!=pool_keys[idx] and len(mismatches)<20:
                        mismatches.append({'pass':pass_i,'position':j,'expected_index':idx,'expected':pool_keys[idx],'observed':key})
                if mismatches and mismatches[-1].get('error')=='unexpected_eof':
                    break
            extra=f.readline()
            if extra:
                mismatches.append({'error':'extra_rows_after_expected'})
        training={'checked':True,'rows_seen':rows_seen,'words_seen':words_seen,'exact_100M_words':words_seen==TOTAL_WORDS*PASSES,'order_matches_step256_formula':len(mismatches)==0,'mismatches':mismatches[:20],'training_sha256':sha256_file(PERM_TRAIN)}

    critical={
        'assignment_rows_equal_pair_count': len(assignments)==len(pairs)==33291,
        'target_set_is_all_pairs': target_unique==all_pair_ids,
        'donor_set_is_all_pairs': donor_unique==all_pair_ids,
        'same_pair_assignments': same_pair,
        'same_doc_assignments': same_doc,
        'same_row_assignments': same_row,
        'donor_doc_in_target_row_docset': donor_doc_in_target_row,
        'row_assignment_counts_match_view_slots': all(row_assignment_counts[r['row_index']]==len(r['pair_ids']) for r in view_rows),
        'pool_row_count_matches_view': len(perm_rows)==len(view_pool),
        'pool_exact_10M_words': pool_words==TOTAL_WORDS,
        'row_length_sequence_matches_view': row_lengths_match,
        'reconstructed_prefix_matches_permuted_pool': prefix_match,
        'suffix_rows_exactly_equal_view_pool': suffix_match,
        'all_row_word_residuals_zero': all(x==0 for x in row_word_residuals),
        'source_multiset_digest_matches_view': source_digest_perm==source_digest_view,
        'rewrite_multiset_digest_matches_view': rewrite_digest_donor==rewrite_digest_view,
        'metadata_status': meta.get('status'),
        'metadata_stream_sha256': (meta.get('sha256') or {}).get('compact_permuted_view_dose2p64x_100M.jsonl'),
        'actual_stream_sha256': sha256_file(PERM_TRAIN),
    }
    critical['metadata_stream_sha_matches_actual'] = bool(critical['metadata_stream_sha256']) and critical['metadata_stream_sha256']==critical['actual_stream_sha256']
    critical['training_order_matches_step256_formula'] = training.get('order_matches_step256_formula') if args.check_training_order else None

    passed = all([
        critical['assignment_rows_equal_pair_count'], critical['target_set_is_all_pairs'], critical['donor_set_is_all_pairs'],
        same_pair==0, same_doc==0, same_row==0, donor_doc_in_target_row==0,
        critical['row_assignment_counts_match_view_slots'], critical['pool_row_count_matches_view'], critical['pool_exact_10M_words'],
        critical['row_length_sequence_matches_view'], critical['reconstructed_prefix_matches_permuted_pool'], critical['suffix_rows_exactly_equal_view_pool'],
        critical['all_row_word_residuals_zero'], critical['source_multiset_digest_matches_view'], critical['rewrite_multiset_digest_matches_view'],
        critical['metadata_stream_sha_matches_actual'], (not args.check_training_order or bool(training.get('order_matches_step256_formula'))),
    ])

    summary={
        'status':'PERMUTED_COMPANION_INDEPENDENT_AUDIT_PASS' if passed else 'PERMUTED_COMPANION_INDEPENDENT_AUDIT_FAIL',
        'created_utc':now(),
        'inputs':{k:rel(v) for k,v in {'pairs':PAIRS,'view_pool':VIEW_POOL,'view_meta':VIEW_META,'permuted_pool':PERM_POOL,'permuted_train':PERM_TRAIN,'assignment':PERM_ASSIGN,'permuted_row_meta':PERM_ROW_META,'metadata':PERM_META,'token_geometry':TOKEN_GEOM}.items()},
        'critical':critical,
        'counts':{'pairs':len(pairs),'assignments':len(assignments),'view_pair_rows':len(view_rows),'permuted_rows':len(perm_rows),'pool_words':pool_words,'row_meta_rows':len(perm_row_meta)},
        'assignment_shape':{'rewrite_length_changed_slots':row_changed_length,'donor_row_distance_stats':score_counts(donor_dist),'row_word_residual_minmax':[min(row_word_residuals),max(row_word_residuals)]},
        'prefix_mismatches':prefix_mismatches,
        'token_shift_vs_max_view':((geom.get('relative_shift_vs_max_view') or {}).get('permuted')),
        'training_order':training,
        'interpretation':'The written research control preserves source/rewrite multisets and row/filler/order geometry while breaking same-row/same-document source-view correspondence.' if passed else 'The written research control failed an invariant; do not train until repaired.',
        'no_model_training_no_scoring_no_globalpiqa_superglue_aoa_upload_or_leaderboard':True,
        'elapsed_sec':round(time.time()-t0,2),
    }
    (out/'independent_audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research independent audit of permuted companion control','',f"Status: `{summary['status']}`",'',summary['interpretation'],'','## Critical invariants','']
    for k,v in critical.items(): lines.append(f'- {k}: {v}')
    lines += ['','## Token/WWM shift versus MAX view','']
    shift=summary['token_shift_vs_max_view'] or {}
    for k,v in shift.items(): lines.append(f'- {k}: {v}')
    (out/'independent_audit_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'passed':passed,'critical':critical,'out':rel(out),'elapsed_sec':summary['elapsed_sec']},indent=2,ensure_ascii=False),flush=True)
    if not passed:
        raise SystemExit(2)

if __name__=='__main__':
    main()
