#!/usr/bin/env python3
"""Repair the prepared research live-FineWeb view pilot source slice.

This is a narrow prompt-readiness repair, not another broad FineWeb selector.  The
first research prompt sample exposed view-generation hazards that would waste Qwen
and contaminate acceptance measurements: mojibake in names, epistemic/future
sentences, report/title fragments, discourse-open starts, and high numeric load.
This script keeps the same research slot manifest contract, filters only for rows
that are safer to ask a generator to restate faithfully, and writes a replacement
prompt slice.  It does not generate text, train, or evaluate BabyLM.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
IN_DIR = ROOT / "training/data/live_fineweb_view_pilot"
MANIFEST = IN_DIR / "live_fineweb_slot_manifest_ratio0p88_samefocus_exact.jsonl"
OUT_DIR = ROOT / "training/data/live_fineweb_view_pilot_repaired"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_view_pilot_source_repair.md')

PROMPT_VARIANTS = {
    "faithful_simplify": (
        "Rewrite the sentence below as one clear sentence for a younger reader. "
        "Use only information in the sentence. Keep every name, date, number, place, object, and cause/effect relation. "
        "Do not add background facts, examples, explanations, guesses, or opinions. Use different wording when possible. "
        "Aim for about {target_words} words, but preserving the facts is more important than exact length. "
        "Output only the rewritten sentence.\n\nSentence: {text}"
    ),
    "relation_preserving_restate": (
        "Restate the sentence as a single factual sentence with simpler wording while preserving the exact relationship it expresses. "
        "Do not introduce any new entity, event, date, number, location, cause, result, comparison, or attribution. "
        "Keep all names and numbers from the source sentence. Avoid copying the whole sentence verbatim when a faithful rewording is possible. "
        "Aim for about {target_words} words. Output only one sentence.\n\nSentence: {text}"
    ),
}
QUOTAS = {
    "causal_temporal_process": 96,
    "physical_spatial_object": 64,
    "social_entity_state": 40,
    "other_expository_relation": 40,
    "definition_taxonomic_fact": 16,
}
WORD_RE = re.compile(r"\S+")
BAD_CORRUPT_RE = re.compile(r"�|Ã.|Â.|\b\w\?\w|\?s\b|Ch\?", re.I)
EPISTEMIC_FUTURE_RE = re.compile(r"\b(according to|said|says|reported|claimed|believed|believe|estimated|estimate|suggested|forecast|apparently|possibly|probably|likely|may|might|could|would|will|currently|recently|today|now|up to|soon)\b", re.I)
PROMPT_META_RE = re.compile(r"\b(report called|released in a \d{4} report|preferred citation|doi|isbn|chapter\s+\d+|table\s+\d+|figure\s+\d+|appendix|book of|letters of)\b", re.I)
DISCOURSE_START_RE = re.compile(r"^(overall|however|meanwhile|therefore|thus|then|later|previously|instead|in addition|for example|for instance|about\s+the|editor'?s note)\b", re.I)
TITLELIKE_COLON_RE = re.compile(r"^[A-Z][^.!?]{0,80}:\s+[A-Z]")
MEDIA_OR_PROCEDURE_RE = re.compile(r"\b(film|movie|novel|episode|season|television series|video game|fictional|character|how to|use acetic acid|click here|website|privacy policy|terms of use)\b", re.I)
UNRESOLVED_RE = re.compile(r"\b(this|that|these|those|such)\s+(report|survey|decision|position|country|region|area|group|team|system|process|method|project|program|issue|case|model|result|effect|study|relationship|phenomenon)\b", re.I)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')


def words(text: str) -> list[str]:
    return WORD_RE.findall(str(text or '').strip())


def issue_reasons(pair: dict[str, Any]) -> list[str]:
    text=str(pair.get('source_text') or '').strip()
    ws=words(text)
    reasons=[]
    if len(ws) < 10 or len(ws) > 38: reasons.append('length_outside_10_38_for_generation')
    if text and text[-1] not in '.!?': reasons.append('no_terminal_punctuation')
    if BAD_CORRUPT_RE.search(text): reasons.append('mojibake_or_corrupt_name')
    if EPISTEMIC_FUTURE_RE.search(text): reasons.append('epistemic_future_or_time_relative')
    if PROMPT_META_RE.search(text): reasons.append('report_title_citation_or_metadata')
    if DISCOURSE_START_RE.search(text): reasons.append('discourse_dependent_start')
    if TITLELIKE_COLON_RE.search(text): reasons.append('titlelike_colon_fragment')
    if MEDIA_OR_PROCEDURE_RE.search(text): reasons.append('media_procedure_or_web_residue')
    if UNRESOLVED_RE.search(text): reasons.append('unresolved_reference_risk')
    digit_frac=sum(any(ch.isdigit() for ch in w) for w in ws)/max(1,len(ws))
    if digit_frac > 0.20: reasons.append('too_digit_dense_for_faithful_rewrite')
    comma_count=text.count(',')
    if comma_count >= 5 and len(ws) <= 30: reasons.append('compressed_catalogue_or_clause_stack')
    if len(str(pair.get('repeat_slot_text_simulated') or '').split()) != int(pair.get('target_view_words_simulated') or -1):
        reasons.append('repeat_slot_length_mismatch')
    return reasons


def select_sources(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable=[]; rejected=[]
    for p in pairs:
        reasons=issue_reasons(p)
        q=dict(p); q['view_prompt_repair_reasons']=reasons; q['view_prompt_usable']=not reasons
        (usable if not reasons else rejected).append(q)
    by_focus=defaultdict(list)
    for p in usable:
        by_focus[p['source_focus']].append(p)
    for focus in by_focus:
        by_focus[focus].sort(key=lambda p: (-float(p.get('source_selection_score') or 0.0), p['pair_id']))
    selected=[]; selected_ids=set(); used_docs=set()
    for focus, quota in QUOTAS.items():
        bucket=by_focus.get(focus, [])
        chosen=[]
        for p in bucket:
            if len(chosen) >= quota: break
            if p['source_doc'] in used_docs: continue
            chosen.append(p); selected_ids.add(p['pair_id']); used_docs.add(p['source_doc'])
        if len(chosen) < quota:
            for p in bucket:
                if len(chosen) >= quota: break
                if p['pair_id'] in selected_ids: continue
                chosen.append(p); selected_ids.add(p['pair_id']); used_docs.add(p['source_doc'])
        selected.extend(chosen[:quota])
    # Fill any shortage with best usable rows from all focus buckets.
    if len(selected) < sum(QUOTAS.values()):
        rest=sorted([p for p in usable if p['pair_id'] not in selected_ids], key=lambda p: (-float(p.get('source_selection_score') or 0.0), p['pair_id']))
        for p in rest:
            if len(selected) >= sum(QUOTAS.values()): break
            selected.append(p); selected_ids.add(p['pair_id'])
    return selected, usable, rejected


def make_prompts(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts=[]
    for p in selected:
        for variant, template in PROMPT_VARIANTS.items():
            prompts.append({
                'id': f"{p['pair_id']}__{variant}",
                'pair_id': p['pair_id'],
                'variant': variant,
                'prompt': template.format(text=p['source_text'], target_words=p['target_view_words_simulated']),
                'source_text': p['source_text'],
                'source_words': p['source_words'],
                'target_view_words_simulated': p['target_view_words_simulated'],
                'source_focus': p['source_focus'],
                'source_class': p['source_class'],
                'source_pool': p['source_pool'],
                'source_doc': p['source_doc'],
                'source_types': p.get('source_types'),
                'source_relation_hits': p.get('source_relation_hits'),
            })
    return prompts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs=read_jsonl(MANIFEST)
    selected, usable, rejected=select_sources(pairs)
    prompts=make_prompts(selected)
    paths={
        'selected_sources': OUT_DIR/'live_fineweb_view_pilot_repaired_sources_256.jsonl',
        'prompts': OUT_DIR/'live_fineweb_view_pilot_repaired_prompts_512.jsonl',
        'usable_manifest': OUT_DIR/'live_fineweb_slot_manifest_view_prompt_usable.jsonl',
        'rejected_manifest': OUT_DIR/'live_fineweb_slot_manifest_view_prompt_rejected.jsonl',
        'samples': OUT_DIR/'live_fineweb_view_pilot_repaired_samples.json',
        'summary': OUT_DIR/'live_fineweb_view_pilot_source_repair_summary.json',
        'generation_config': OUT_DIR/'GENERATION_CONFIG.json',
    }
    write_jsonl(paths['selected_sources'], selected)
    write_jsonl(paths['prompts'], prompts)
    write_jsonl(paths['usable_manifest'], usable)
    write_jsonl(paths['rejected_manifest'], rejected)
    reason_counts=Counter(reason for p in rejected for reason in p.get('view_prompt_repair_reasons', []))
    focus_usable=Counter(p['source_focus'] for p in usable)
    focus_selected=Counter(p['source_focus'] for p in selected)
    class_selected=Counter(p['source_class'] for p in selected)
    summary={
        'status':'LIVE_FINEWEB_VIEW_PILOT_SOURCE_REPAIRED',
        'purpose':'Narrow prompt-readiness repair of the research source+view pilot; no generation, training, or BabyLM evaluation.',
        'input_manifest':str(MANIFEST),
        'input_pairs':len(pairs),
        'usable_pairs':len(usable),
        'rejected_pairs':len(rejected),
        'usable_packet_words_simulated':sum(int(p.get('packet_words_simulated') or 0) for p in usable),
        'selected_sources':len(selected),
        'selected_prompts':len(prompts),
        'selected_source_words':sum(int(p.get('source_words') or 0) for p in selected),
        'selected_target_view_words_simulated':sum(int(p.get('target_view_words_simulated') or 0) for p in selected),
        'selected_packet_words_simulated':sum(int(p.get('packet_words_simulated') or 0) for p in selected),
        'focus_usable_counts':dict(focus_usable.most_common()),
        'focus_selected_counts':dict(focus_selected.most_common()),
        'class_selected_counts':dict(class_selected.most_common()),
        'top_rejection_reasons':reason_counts.most_common(30),
        'paths':{k:str(v) for k,v in paths.items()},
        'generation_is_deferred':True,
        'future_generation': {
            'model_alias': 'qwen3.5-9b', 'prompts_jsonl': str(paths['prompts']),
            'output_jsonl': 'experiments/archive/representation_and_objectives/training/runs/live_fineweb_view_pilot_repaired_qwen/outputs.jsonl',
            'batch_size': 64, 'max_new_tokens': 80, 'temperature': 0.15,
            'device': 'cuda', 'status': 'proposed_not_executed',
        },
        'future_analysis_command_template':f"python -B experiments/archive/representation_and_objectives/training/scripts/analyze_live_fineweb_view_pilot_outputs.py --prompts {paths['prompts']} --outputs experiments/archive/representation_and_objectives/training/runs/live_fineweb_view_pilot_repaired_qwen/outputs.jsonl --out {OUT_DIR/'live_fineweb_view_pilot_repaired_analysis.json'} --note research/notes/representation_and_objectives/live_fineweb_view_pilot_repaired_analysis.md",
        'use_policy':'The repaired prompt slice removes the diagnosed source defects. Generated views still require faithfulness analysis before training materialization.',
    }
    paths['summary'].write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    paths['samples'].write_text(json.dumps({'selected_head':selected[:12], 'rejected_head':rejected[:12], 'prompt_head':prompts[:12]}, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    paths['generation_config'].write_text(json.dumps(summary['future_generation'], indent=2)+'\n', encoding='utf-8')
    lines=['# research live FineWeb view pilot source repair\n\n']
    lines.append('This is a narrow prompt-readiness repair of the prepared source+view pilot, not a new broad selector and not a generation/training run. It removes rows that are likely to make a generator hallucinate or copy because of corrupt names, epistemic/future wording, title/report metadata, discourse-open starts, unresolved references, or high numeric load.\n\n')
    lines.append(f"Input slot manifest: {len(pairs):,} pairs. Usable for view prompts: {len(usable):,} pairs ({100*len(usable)/max(1,len(pairs)):.2f}%).\n\n")
    lines.append(f"Repaired prompt slice: {len(selected):,} sources / {len(prompts):,} prompts; source words {summary['selected_source_words']:,}; simulated view-slot words {summary['selected_target_view_words_simulated']:,}; packet words {summary['selected_packet_words_simulated']:,}.\n\n")
    lines.append('Focus counts in repaired slice: '+json.dumps(dict(focus_selected), ensure_ascii=False)+'\n\n')
    lines.append('Top rejected reasons: '+json.dumps(reason_counts.most_common(12), ensure_ascii=False)+'\n\n')
    lines.append('Use this repaired slice, not the unrepaired prompt file, if a later step runs the compact live FineWeb view generation. Actual accepted view lengths and substantive-change rates still decide whether any training corpus is materialized.\n\n')
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nPrompts: `{paths['prompts']}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':str(paths['summary']),'note':str(NOTE),'usable_pairs':len(usable),'selected_sources':len(selected),'selected_prompts':len(prompts)}, indent=2, ensure_ascii=False))


if __name__=='__main__':
    main()
