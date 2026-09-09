#!/usr/bin/env python3
"""Materialize a cleaner cached-FineWeb single-document broad-source candidate.

Compared with `materialize_cached_fineweb_broad_source_candidate.py`, this
variant drops every cached FineWeb row whose 160-word chunk crosses document IDs.
It keeps all remaining single-doc rows, preserving their factual breadth while
reducing the packing/coherence confound seen in the raw INITIAL_MODEL_STUDIES cached corpus.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

TOTAL_WORDS = 10_000_000
PASSES = 10
DEFAULT_QWEN_POOL = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
DEFAULT_QWEN_META = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json")
DEFAULT_FINEWEB = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
DEFAULT_ROW_AUDIT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/fineweb_row_structure_audit.json")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/cached_fineweb_single_doc_candidate.md")


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    meta: dict[str, Any] = field(default_factory=dict)


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))]
    return {"n": len(xs), "min": min(xs), "p05": q(0.05), "mean": round(statistics.mean(xs), 4), "median": statistics.median(xs), "p95": q(0.95), "max": max(xs), "sum": sum(xs)}


def load_qwen_rows(path: pathlib.Path) -> list[Row]:
    rows=[]
    for i,line in enumerate(path.open(encoding='utf-8')):
        if not line.strip(): continue
        o=json.loads(line); text=norm_text(o['text']); words=int(o.get('words', len(text.split())))
        if words != len(text.split()): raise RuntimeError(f'qwen word mismatch row {i}')
        rows.append(Row(text=text, words=words, source=str(o.get('source','')), example_id=int(o.get('example_id', i))))
    return rows


def load_single_doc_fineweb(path: pathlib.Path) -> tuple[list[Row], dict[str, Any]]:
    rows=[]; total_rows=0; dropped=0; doc_counter=collections.Counter(); examples=[]
    for i,line in enumerate(path.open(encoding='utf-8')):
        if not line.strip(): continue
        total_rows += 1
        o=json.loads(line); ids=o.get('doc_ids') or []
        if len(ids) != 1:
            dropped += 1
            continue
        text=norm_text(o['text']); words=int(o.get('words', len(text.split())))
        if words != len(text.split()): raise RuntimeError(f'fineweb word mismatch row {i}')
        doc=str(ids[0]); doc_counter[doc]+=1
        r=Row(text=text, words=words, source='fineweb_edu_random_quality_single_doc_cached_initial_model_studies', example_id=8_100_000+len(rows), meta={'doc_ids': ids, 'source_row': i})
        rows.append(r)
        if len(examples)<8:
            examples.append({'source_row': i, 'doc_id': doc, 'words': words, 'text_excerpt': text[:500] + ('…' if len(text)>500 else '')})
    return rows, {'input_rows': total_rows, 'dropped_multi_doc_rows': dropped, 'kept_rows': len(rows), 'kept_words': sum(r.words for r in rows), 'unique_docs': len(doc_counter), 'rows_per_doc_top10': doc_counter.most_common(10), 'examples': examples}


@dataclass
class StreamState:
    rows: list[Row]
    row_i: int = 0
    word_i: int = 0
    def take_words(self, n: int):
        out=[]; comp=collections.Counter(); segs=[]
        while len(out)<n:
            if self.row_i>=len(self.rows): raise RuntimeError('official filler exhausted')
            r=self.rows[self.row_i]; ws=r.text.split(); rem=len(ws)-self.word_i; need=n-len(out); take=min(rem, need)
            if take<=0:
                self.row_i+=1; self.word_i=0; continue
            out.extend(ws[self.word_i:self.word_i+take]); comp[r.source]+=take
            segs.append({'source': r.source, 'example_id': r.example_id, 'start_word': self.word_i, 'words': take})
            self.word_i += take
            if self.word_i >= len(ws): self.row_i += 1; self.word_i = 0
        return out, comp, segs
    def consumed_words(self):
        return sum(r.words for r in self.rows[:self.row_i]) + self.word_i


def chunk_to_lengths(state: StreamState, lengths: list[int], source: str, start_id: int):
    rows=[]; comp_total=collections.Counter(); multi=0; sample=[]
    for j,n in enumerate(lengths):
        ws,comp,segs=state.take_words(n); comp_total.update(comp)
        if len(comp)>1: multi+=1
        meta={'component_sources': dict(comp)}
        if j<12:
            meta['segments']=segs; sample.append({'row_index': j, 'words': n, 'component_sources': dict(comp), 'segments': segs})
        rows.append(Row(text=' '.join(ws), words=n, source=source, example_id=start_id+j, meta=meta))
    return rows, {'rows': len(rows), 'words': sum(lengths), 'component_sources_total': dict(comp_total), 'multi_source_rows': multi, 'multi_source_row_fraction': multi/len(rows) if rows else 0, 'sample_meta': sample, 'stream_consumed_words_after': state.consumed_words(), 'stream_row_i_after': state.row_i, 'stream_word_i_after': state.word_i}


def tail_rows(state: StreamState, needed: int, source: str, start_id: int):
    rows=[]; total=0; comp_total=collections.Counter(); partial=0
    while total<needed:
        remain=needed-total
        if state.row_i<len(state.rows) and state.word_i==0 and state.rows[state.row_i].words<=remain:
            r=state.rows[state.row_i]
            rows.append(Row(text=r.text, words=r.words, source=source, example_id=start_id+len(rows), meta={'original_source': r.source, 'original_example_id': r.example_id}))
            comp_total[r.source]+=r.words; total+=r.words; state.row_i+=1
        else:
            ws,comp,segs=state.take_words(remain)
            rows.append(Row(text=' '.join(ws), words=len(ws), source=source, example_id=start_id+len(rows), meta={'component_sources': dict(comp), 'segments': segs, 'partial_tail_row': True}))
            comp_total.update(comp); partial+=1; total+=len(ws)
    return rows, {'needed_words': needed, 'rows': len(rows), 'partial_rows': partial, 'component_sources_total': dict(comp_total), 'stream_consumed_words_after': state.consumed_words(), 'stream_row_i_after': state.row_i, 'stream_word_i_after': state.word_i}


def write_rows(path: pathlib.Path, rows: list[Row]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            if r.words != len(r.text.split()): raise RuntimeError(f'word mismatch before write {path}')
            o={'text': r.text, 'words': r.words, 'example_id': r.example_id, 'source': r.source}; o.update(r.meta)
            f.write(json.dumps(o, ensure_ascii=False)+'\n')


def write_training(path: pathlib.Path, rows: list[Row], orders: list[list[int]]):
    total=0; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for order in orders:
            for idx in order:
                r=rows[idx]; o={'text': r.text, 'words': r.words, 'example_id': r.example_id, 'source': r.source}; o.update(r.meta)
                f.write(json.dumps(o, ensure_ascii=False)+'\n'); total += r.words
    expected=sum(r.words for r in rows)*len(orders)
    if total != expected: raise RuntimeError(f'training exposure mismatch {total}!={expected}')
    return total


def source_words(rows):
    c=collections.Counter()
    for r in rows: c[r.source]+=r.words
    return dict(c)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--qwen-pool', default=str(DEFAULT_QWEN_POOL))
    ap.add_argument('--qwen-meta', default=str(DEFAULT_QWEN_META))
    ap.add_argument('--fineweb', default=str(DEFAULT_FINEWEB))
    ap.add_argument('--row-audit', default=str(DEFAULT_ROW_AUDIT))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note', default=str(DEFAULT_NOTE))
    ap.add_argument('--total-words', type=int, default=TOTAL_WORDS)
    ap.add_argument('--passes', type=int, default=PASSES)
    ap.add_argument('--seed', type=int, default=80809)
    ap.add_argument('--write-training', action='store_true')
    args=ap.parse_args(); t0=time.time(); out=pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qwen_all=load_qwen_rows(pathlib.Path(args.qwen_pool))
    qwen_pair=[r for r in qwen_all if r.source=='qwen_pair_packed']; filler=[r for r in qwen_all if r.source!='qwen_pair_packed']
    qwen_words=sum(r.words for r in qwen_pair); filler_words=sum(r.words for r in filler)
    fineweb, fw_filter=load_single_doc_fineweb(pathlib.Path(args.fineweb)); fw_words=sum(r.words for r in fineweb)
    if qwen_words+fw_words >= args.total_words: raise RuntimeError('qwen+fineweb exceeds total')
    state=StreamState(filler)
    ctrl_block, ctrl_meta=chunk_to_lengths(state, [r.words for r in fineweb], 'official_lengthmatched_to_cached_fineweb_single_doc', 8_600_000)
    tail_needed=args.total_words-qwen_words-fw_words
    tail, tail_meta=tail_rows(state, tail_needed, 'official_identical_tail_after_single_doc_fineweb_block', 9_100_000)
    treat=qwen_pair+fineweb+tail; ctrl=qwen_pair+ctrl_block+tail
    required={
        'row_length_sequence_identical': [r.words for r in treat]==[r.words for r in ctrl],
        'qwen_pair_block_identical': all(treat[i].text==ctrl[i].text and treat[i].source==ctrl[i].source for i in range(len(qwen_pair))),
        'tail_filler_identical': all(treat[len(qwen_pair)+len(fineweb)+i].text==ctrl[len(qwen_pair)+len(fineweb)+i].text and treat[len(qwen_pair)+len(fineweb)+i].source==ctrl[len(qwen_pair)+len(fineweb)+i].source for i in range(len(tail))),
        'total_words_treatment': sum(r.words for r in treat)==args.total_words,
        'total_words_control': sum(r.words for r in ctrl)==args.total_words,
        'fineweb_single_doc_words_positive': fw_words>0,
        'official_filler_sufficient': state.consumed_words() <= filler_words,
    }
    if not all(required.values()): raise RuntimeError(required)
    rng=random.Random(args.seed); base=list(range(len(treat))); orders=[]
    for _ in range(args.passes):
        o=list(base); rng.shuffle(o); orders.append(o)
    paths={
        'treatment_10M': out/'cleanqwen_cached_fineweb_single_doc_10M.jsonl',
        'control_10M': out/'cleanqwen_official_lengthmatched_single_doc_control_10M.jsonl',
        'treatment_100M': out/'cleanqwen_cached_fineweb_single_doc_100M.jsonl',
        'control_100M': out/'cleanqwen_official_lengthmatched_single_doc_control_100M.jsonl',
        'verification': out/'materialization_verification.json',
        'metadata': out/'materialization_metadata.json',
        'samples': out/'sample_rows.json',
        'pass_manifest': out/'pass_order_manifest.json',
        'note': pathlib.Path(args.note),
    }
    write_rows(paths['treatment_10M'], treat); write_rows(paths['control_10M'], ctrl)
    train_exp=None
    if args.write_training:
        e1=write_training(paths['treatment_100M'], treat, orders); e2=write_training(paths['control_100M'], ctrl, orders)
        if e1!=e2: raise RuntimeError('training exposure mismatch')
        train_exp=e1
    paths['pass_manifest'].write_text(json.dumps({'seed': args.seed, 'passes': args.passes, 'rows_per_pool': len(treat), 'same_order_used_for_treatment_and_control': True, 'orders_sha256': hashlib.sha256(json.dumps(orders,separators=(',',':')).encode()).hexdigest()}, indent=2)+'\n', encoding='utf-8')
    row_audit=json.loads(pathlib.Path(args.row_audit).read_text()) if pathlib.Path(args.row_audit).exists() else None
    qwen_meta=json.loads(pathlib.Path(args.qwen_meta).read_text()) if pathlib.Path(args.qwen_meta).exists() else None
    verification={
        'status': 'CACHED_FINEWEB_SINGLE_DOC_CANDIDATE_VERIFIED',
        'total_words_per_pool': args.total_words,
        'passes': args.passes,
        'write_training': bool(args.write_training),
        'training_exposure_words': train_exp,
        'rows': {'treatment': len(treat), 'control': len(ctrl)},
        'source_word_counts_treatment': source_words(treat),
        'source_word_counts_control': source_words(ctrl),
        'qwen_pair_rows': len(qwen_pair), 'qwen_pair_words': qwen_words,
        'fineweb_single_doc_rows': len(fineweb), 'fineweb_single_doc_words': fw_words,
        'tail_filler_words': tail_needed,
        'word_fractions': {'clean_qwen_pairs': round(qwen_words/args.total_words,6), 'cached_fineweb_single_doc': round(fw_words/args.total_words,6), 'combined_non_tail_prefix': round((qwen_words+fw_words)/args.total_words,6), 'identical_official_tail': round(tail_needed/args.total_words,6)},
        'row_stats': {'qwen_pair': stats([r.words for r in qwen_pair]), 'fineweb_single_doc': stats([r.words for r in fineweb]), 'control_block': stats([r.words for r in ctrl_block]), 'tail': stats([r.words for r in tail]), 'pool': stats([r.words for r in treat])},
        'fineweb_filter': fw_filter,
        'source_row_audit': {'path': str(args.row_audit), 'multi_doc_fraction_in_raw_cached': row_audit.get('multi_doc_fraction') if isinstance(row_audit, dict) else None, 'single_doc_words_in_raw_cached': row_audit.get('single_doc_words') if isinstance(row_audit, dict) else None},
        'control_block_from_official_filler': ctrl_meta,
        'identical_tail_from_official_filler': tail_meta,
        'inherited_clean_qwen_metadata': {'path': str(args.qwen_meta), 'status': qwen_meta.get('status') if isinstance(qwen_meta, dict) else None, 'selected_pair_words': qwen_meta.get('selected_pair_words') if isinstance(qwen_meta, dict) else None, 'pair_boundary_preserved': qwen_meta.get('pair_boundary_preserved') if isinstance(qwen_meta, dict) else None},
        'required_checks': required,
        'all_required_checks_pass': True,
    }
    paths['verification'].write_text(json.dumps(verification, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    samples={'fineweb_single_doc_first_rows': fw_filter['examples'], 'control_block_sample': ctrl_meta['sample_meta'][:8], 'tail_first_source_counts': tail_meta['component_sources_total']}
    paths['samples'].write_text(json.dumps(samples, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    meta={'status': 'CACHED_FINEWEB_SINGLE_DOC_CANDIDATE_MATERIALIZED', 'created_unix': int(time.time()), 'purpose': 'cleaner cached FineWeb-Edu single-document factual-source candidate on top of clean-Qwen; not trained here', 'inputs': {'qwen_pool': str(args.qwen_pool), 'fineweb': str(args.fineweb), 'row_audit': str(args.row_audit)}, 'outputs': {k:str(v) for k,v in paths.items()}, 'verification': verification, 'sha256': {k:sha256_file(v) for k,v in paths.items() if v.exists() and v.is_file() and k!='note'}, 'elapsed_sec': round(time.time()-t0,3)}
    paths['metadata'].write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research cached FineWeb-Edu single-document candidate\n\n']
    lines.append('This is a cleaner broad-factual-source candidate, not trained evidence. It drops cached FineWeb rows whose 160-word chunk crosses document IDs, reducing the packing confound found in the raw cached 3M file.\n\n')
    lines.append(f"- Clean-Qwen pair block: {len(qwen_pair):,} rows, {qwen_words:,} words ({qwen_words/args.total_words:.2%}).\n")
    lines.append(f"- Cached FineWeb single-doc block: {len(fineweb):,} rows, {fw_words:,} words ({fw_words/args.total_words:.2%}); unique docs {fw_filter['unique_docs']:,}.\n")
    lines.append(f"- Official length-matched control block: {len(ctrl_block):,} rows, {sum(r.words for r in ctrl_block):,} words.\n")
    lines.append(f"- Shared official tail: {len(tail):,} rows, {tail_needed:,} words ({tail_needed/args.total_words:.2%}).\n")
    lines.append('- Treatment/control preserve identical clean-Qwen rows, identical tail, exact 10M words, and identical row-length sequence.\n\n')
    lines.append('Interpretation if later trained: this is a lower-confound public FineWeb-Edu source-distribution test. It is smaller than the raw cached 3M arm (1.765M words instead of 3M) and still comes from INITIAL_MODEL_STUDIES cached material, because live HF streaming is currently unavailable.\n\n')
    lines.append(f"Verification JSON: `{paths['verification']}`\n\nMetadata JSON: `{paths['metadata']}`\n\nSample rows: `{paths['samples']}`\n")
    paths['note'].parent.mkdir(parents=True, exist_ok=True); paths['note'].write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'status': meta['status'], 'out_dir': str(out), 'qwen_pair_words': qwen_words, 'fineweb_single_doc_words': fw_words, 'tail_words': tail_needed, 'rows': verification['rows'], 'word_fractions': verification['word_fractions'], 'all_required_checks_pass': True, 'verification': str(paths['verification']), 'metadata': str(paths['metadata']), 'note': str(paths['note'])}, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
