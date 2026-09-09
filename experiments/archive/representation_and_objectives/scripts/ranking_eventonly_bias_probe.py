#!/usr/bin/env python3
"""research: event-only shortcut probe for the ranking-attested pilot.

research found independently recorded ranking states, but the retained families were
filtered so that in both worlds the match winner was also ranked higher. This
script tests whether the ranking NLI labels can be recovered from the event text
alone. If yes, the pilot is an attested source scaffold, not yet an independent
event-to-state learning benchmark.
"""
from __future__ import annotations

import collections, json, re, time
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/representation_and_objectives")
IN_DIR = STUDY / "data/ranking_attested_pilot"
OUT_DIR = STUDY / "data/ranking_eventonly_bias_probe"


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def parse_hyp_first(hyp: str, a: str, b: str) -> str | None:
    if hyp.startswith(a): return "A"
    if hyp.startswith(b): return "B"
    return None

def make_rows(fams: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows=[]
    for fam in fams:
        a,b=fam['participant_a'], fam['participant_b']
        for q in fam['nli_queries']:
            cx = 'context1' if q['context'] == 'c1' else 'context2'
            ctx = fam[cx]
            first = parse_hyp_first(q['hypothesis'], a, b)
            winner_lab = 'A' if ctx['winner'] == a else 'B'
            event_shortcut = 'ENTAILED' if first == winner_lab else 'NOT_ENTAILED'
            rows.append({
                'id': f"{fam['family_id']}_{q['context']}_{q['hyp_type']}_{q['orient']}",
                'family_id': fam['family_id'], 'split': split,
                'participant_a': a, 'participant_b': b,
                'context_key': cx,
                'event_text': ctx['event_text'],
                'ranking_state': ctx['ranking_state'],
                'full_text': ctx['full_text'],
                'hypothesis': q['hypothesis'],
                'hyp_type': q['hyp_type'], 'orient': q['orient'],
                'gold': q['gold'], 'y': int(q['gold'] == 'ENTAILED'),
                'winner_label': winner_lab,
                'hyp_first_entity': first,
                'event_only_winner_to_rank_prediction': event_shortcut,
                'event_only_winner_to_rank_correct': event_shortcut == q['gold'],
                'event_context_has_rank_number': bool(re.search(r"#|ranked|ranking|\bNo\.?\s*\d+", ctx['event_text'], re.I)),
            })
    return rows

def transparent_event_parser(row: dict[str, Any]) -> str:
    # Because event text in the pilot always names the match winner/loser and the
    # selected families are both ranking-consistent, this is the suspected shortcut.
    return 'ENTAILED' if row['hyp_first_entity'] == row['winner_label'] else 'NOT_ENTAILED'

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train=read_jsonl(IN_DIR/'ranking_families_train.jsonl')
    held=read_jsonl(IN_DIR/'ranking_families_held.jsonl')
    rows_train=make_rows(train,'train'); rows_held=make_rows(held,'held')
    all_rows=rows_train+rows_held
    for fn, rows in [('ranking_eventonly_train_rows.jsonl', rows_train), ('ranking_eventonly_held_rows.jsonl', rows_held)]:
        write_jsonl(OUT_DIR/fn, rows)
    def score(rows):
        return sum(transparent_event_parser(r) == r['gold'] for r in rows)/max(1,len(rows))
    by_type=collections.defaultdict(list)
    by_split=collections.defaultdict(list)
    for r in all_rows:
        ok = transparent_event_parser(r) == r['gold']
        by_type[r['hyp_type']].append(ok); by_split[r['split']].append(ok)
    rank_leak=sum(r['event_context_has_rank_number'] for r in all_rows)
    summary={
        'status':'RANKING_EVENTONLY_BIAS_PROBE',
        'created_utc':now(),
        'input_dir':str(IN_DIR),
        'row_counts':{'train':len(rows_train),'held':len(rows_held),'total':len(all_rows)},
        'event_context_rank_number_leaks':rank_leak,
        'transparent_event_winner_to_rank_accuracy':{'train':score(rows_train),'held':score(rows_held),'all':score(all_rows)},
        'by_hyp_type_accuracy':{k:sum(v)/len(v) for k,v in sorted(by_type.items())},
        'by_split_accuracy':{k:sum(v)/len(v) for k,v in sorted(by_split.items())},
        'gold_label_counts':dict(collections.Counter(r['gold'] for r in all_rows)),
        'winner_label_counts':dict(collections.Counter(r['winner_label'] for r in all_rows)),
        'interpretation':'The event_text contains no ranking numbers, but because retained families require winner_rank < loser_rank in both worlds, ranking-state labels are exactly recoverable from event winner identity. The pilot is independently attested, but not yet a clean event-to-state benchmark.',
    }
    write_json(OUT_DIR/'ranking_eventonly_bias_summary.json', summary)
    md=['# research ranking event-only bias probe','',f"Rows: train {len(rows_train)}, held {len(rows_held)}.",f"Event text rank-number leaks: {rank_leak}.",f"Transparent winner→higher-rank shortcut accuracy: train {score(rows_train):.3f}, held {score(rows_held):.3f}, all {score(all_rows):.3f}.",'','This means the ranking pilot has independent state attestation, but its current both-consistent filter turns event outcome into a perfect proxy for the ranking label. A valid event/state route must include inconsistent or temporally changing cases where event winner and independently recorded state can diverge, or use the ranking source as explicit state input rather than treating event-only inference as a principle.','',f"Summary JSON: `{OUT_DIR/'ranking_eventonly_bias_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/ranking_eventonly_bias_probe/ranking_eventonly_bias_summary.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT_DIR/'ranking_eventonly_bias_summary.json'),'event_only_shortcut_all':summary['transparent_event_winner_to_rank_accuracy']['all'],'rank_leaks':rank_leak}, indent=2), flush=True)

if __name__=='__main__': main()
