#!/usr/bin/env python3
"""research: ultra-clean transition 50k probe assets.

This is a stricter, smaller successor to the broad research/research transition
substrates.  It builds a 50k-word treatment slice with explicit relation markers
plus concrete/action anchors and a 50k-word no-explicit-relation control from the
same allowed reservoirs.  It is intended only as a future cheap probe asset if the
running FW compact/breadth endpoints fail to move relation-conditioned errors.

No official evaluation item text is read.  No model training is launched.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
CORE_SCRIPT = A01_WS / "scripts/core_transition_filter_and_controls.py"
CANDIDATES = A01_WS / "data/core_transition_filter/core_transition_candidates.jsonl"
OUT = A01_WS / "data/ultraclean_transition_probe"
NOTE = A01_WS / "notes/ultraclean_transition_probe_assets.md"
TARGET = 50_000

spec = importlib.util.spec_from_file_location("core", CORE_SCRIPT)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)  # type: ignore[union-attr]

NOISE_PATTERNS = {
    "childes_path": re.compile(r"\bchildes/|\.cha\b", re.I),
    "transcript_speaker": re.compile(r"\*[A-Z0-9]{2,6}:", re.I),
    "subtitle_pos": re.compile(r"\\pos\(|\{\\", re.I),
    "ellipsis_artifact": re.compile(r"\.\.\.|…"),
    "bracket_stage_dir": re.compile(r"\[[^\]]{0,90}\]"),
    "underscore_markup": re.compile(r"_{2,}|\b_\w+_\b"),
    "equals_markup": re.compile(r"(?:=\s*){3,}"),
    "url_html": re.compile(r"https?://|www\.|</?\w+", re.I),
    "starts_lowercase_fragment": re.compile(r"^[a-z]"),
}
ABSTRACT_EXTRA_RE = re.compile(
    r"\b(government|policy|policies|election|parliament|minister|senate|conservative|rwanda|islamists|market|company|business|rights|law|court|argument|opinion|belief|idea|story|movie|religion|campaign|internet|website|software|data|research|study|studies|competition|winning|love|happiness|music|dance|agency|report|political|army|party|finance|stock|school|student|teacher|university|windows|microsoft|file allocation table)\b",
    re.I,
)

# These words are retained in the underlying training reservoirs, so replacing a
# candidate block with them does not create a clean semantic contrast.  They are
# specific and frequent enough to keep controls/research slices readable without
# suppressing ordinary causal, quantity, or affordance wording.
LOW_SIGNAL_META_RE = re.compile(r"\b(er|erm|uh|hmm|yeah|mummy|mommy|papa|poppa|councillor|councillors|thesis|theses)\b", re.I)

# Controls must suppress explicit relation discourse and change terms.  They may
# contain concrete objects, spatial nouns, quantities, or actions.
CONTROL_RELATION_RE = re.compile(
    r"\b(if|when|whenever|unless|because|since|therefore|thus|so that|in order to|as a result|result(?:s|ed)? in|leads? to|led to|causes?|caused|prevent(?:s|ed)?|allows?|allowed|"
    r"become|became|becomes|turn(?:s|ed)? into|changed?|changes?|increase[sd]?|decrease[sd]?|rather than|instead|whereas|although|but|while|before|after|first|then|next|finally|from\b.{0,45}\bto)\b",
    re.I,
)
# Tighter form used for the final treatment filter; action verbs that can be
# accidental in non-transition prose are allowed to remain.
TREATMENT_RELATION_RE = re.compile(
    r"\b(if|when|whenever|unless|because|since|therefore|thus|so that|in order to|as a result|result(?:s|ed)? in|leads? to|led to|causes?|caused|prevent(?:s|ed)?|allows?|allowed|"
    r"become|became|becomes|turn(?:s|ed)? into|changed?|changes?|increase[sd]?|decrease[sd]?|rather than|instead|whereas|although|but|while|before|after|first|then|next|finally|from\b.{0,45}\bto)\b",
    re.I,
)
QUANTITY_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|twice|once|many|few|several|some|all|both|each|every|number|amount|part|piece|hour|minute|day|week|month|year|age|old|young|small|large|short|long|high|low|deep|wide|narrow|temperature|weight|size|inch|foot|feet|meter|metre|mile|pound|gram|degree)\b|\b\d+(?:\.\d+)?\b",
    re.I,
)
PRIMARY_BUCKETS = ["physical_material_transition", "spatial_transition", "temporal_quantity_transition"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")


def noise_flags(text: str) -> list[str]:
    return [k for k,rx in NOISE_PATTERNS.items() if rx.search(text)]


def clean_text(text: str) -> bool:
    if noise_flags(text): return False
    if ABSTRACT_EXTRA_RE.search(text): return False
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) >= max(7, core.wc(text)//6): return False
    return True


def treatment_clean_text(text: str) -> bool:
    """Clean for final treatment rows: also suppress low-signal conversational/meta terms."""
    return clean_text(text) and not LOW_SIGNAL_META_RE.search(text)


def relation_marker_sum(row: dict[str,Any]) -> int:
    m = row.get("marker") or {}
    return int(m.get("causal",0)) + int(m.get("change",0)) + int(m.get("contrast_time",0))


def treatment_keep(row: dict[str, Any]) -> bool:
    text = str(row.get("text", ""))
    b = str(row.get("selected_route_bucket") or "")
    if b not in PRIMARY_BUCKETS: return False
    if not treatment_clean_text(text): return False
    if not bool(row.get("core_strict")): return False
    if int(row.get("words", core.wc(text))) < 12 or int(row.get("words", core.wc(text))) > 70: return False
    if float(row.get("core_score", 0.0)) < 12.0: return False
    if int(row.get("abstract_noise_count", 0)) > 0: return False
    if relation_marker_sum(row) < 2: return False
    if int(row.get("non_social_hit", 0)) < 4: return False
    m = row.get("marker") or {}
    if int(row.get("concrete_count", 0)) < 1 and int(m.get("action", 0)) < 1: return False
    return True


def required_cap(row: dict[str,Any]) -> str:
    b = str(row.get("selected_route_bucket") or "")
    if b == "spatial_transition": return "spatial_anchor"
    if b == "temporal_quantity_transition": return "quantity_anchor"
    return "physical_anchor"


def anchor_counts(text: str) -> dict[str,int]:
    toks = core.words(text)
    return {
        "concrete": sum(1 for t in toks if t in core.CONCRETE_WORDS),
        "action": len(core.ACTION_RE.findall(text)),
        "spatial": len(core.SPATIAL_RE.findall(text)),
        "affordance": len(core.AFFORDANCE_RE.findall(text)),
        "quantity": len(QUANTITY_RE.findall(text)),
    }


def caps(text: str) -> list[str]:
    c=anchor_counts(text); out=[]
    if c["concrete"] >= 2 or (c["concrete"] >= 1 and c["action"] >= 1): out.append("physical_anchor")
    if c["spatial"] >= 1 and (c["concrete"] >= 1 or c["action"] >= 1): out.append("spatial_anchor")
    if c["quantity"] >= 1 and (c["concrete"] >= 1 or c["action"] >= 1): out.append("quantity_anchor")
    return out


def control_keep(text: str, n: int) -> bool:
    if n < 8 or n > 90: return False
    if not clean_text(text): return False
    if CONTROL_RELATION_RE.search(text): return False
    if not caps(text): return False
    return True


def subset_exact(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Select the highest-score reachable subset, maximizing filled words first."""
    if target <= 0 or not rows:
        return []
    best = [-10**12]*(target+1)
    prev: list[tuple[int,int] | None] = [None]*(target+1)
    best[0] = 0
    for i,r in enumerate(rows):
        w = int(r["words"])
        if w > target:
            continue
        val = int(round(float(r.get("core_score",0.0))*100)) + 10
        for s in range(target, w-1, -1):
            if best[s-w] <= -10**11:
                continue
            nv = best[s-w] + val
            if nv > best[s]:
                best[s] = nv; prev[s] = (s-w, i)
    reachable = [s for s, v in enumerate(best) if v > -10**11]
    if not reachable:
        return []
    # Fill words first; score only ranks within each reachable size.
    best_s = max(reachable)
    used=[]; s=best_s; seen=set()
    while s>0 and prev[s] is not None:
        ps,i=prev[s]
        if i in seen: break
        seen.add(i); used.append(rows[i]); s=ps
    used.reverse(); return used


def build_treatment(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by=defaultdict(list)
    for r in pool:
        by[str(r.get("selected_route_bucket"))].append(r)
    for arr in by.values():
        arr.sort(key=lambda r:(-float(r.get("core_score",0)), abs(int(r["words"])-36), r.get("text_sha256","")))
    temporal_total=sum(int(r["words"]) for r in by["temporal_quantity_transition"])
    temporal_target=min(3500, temporal_total)
    spatial_target=min(12000, sum(int(r["words"]) for r in by["spatial_transition"]))
    temporal=subset_exact(by["temporal_quantity_transition"], temporal_target)
    spatial=subset_exact(by["spatial_transition"], spatial_target)
    rem=TARGET - sum(int(r["words"]) for r in temporal+spatial)
    physical=subset_exact(by["physical_material_transition"], rem)
    selected=temporal+spatial+physical
    total=sum(int(r["words"]) for r in selected)
    if total != TARGET:
        # Fallback: exact subset over all pool if quotas failed.  This sacrifices some
        # balance but preserves exact word count.
        exact=subset_exact(pool, TARGET)
        if sum(int(r["words"]) for r in exact)==TARGET:
            selected=exact
    selected.sort(key=lambda r:(str(r.get("selected_route_bucket")), -float(r.get("core_score",0)), r.get("text_sha256","")))
    for r in selected:
        r["ultraclean_required_capability"] = required_cap(r)
    return selected


def collect_control_pool() -> tuple[list[dict[str,Any]], dict[str,Any]]:
    pool=[]; seen=set(); scan={}
    # scan full reservoirs; this ran fast enough in research and makes the v3 controls
    # independent of imperfect v2 pools.
    for label, spec in core.RESERVOIRS.items():
        path=Path(spec["path"]); accepted=0; got=0; sent_scanned=0; words_scanned=0; cap_words=Counter()
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for row_i,line in enumerate(f):
                if not line.strip(): continue
                try: obj=json.loads(line)
                except json.JSONDecodeError: continue
                for field in spec["fields"]:
                    raw=str(obj.get(field,"") or "")
                    if not raw: continue
                    for sent_i,sent in enumerate(core.split_sentences(raw)):
                        sent=core.norm(sent); n=core.wc(sent); sent_scanned+=1; words_scanned+=n
                        if not control_keep(sent,n): continue
                        key=sent.lower()
                        if key in seen: continue
                        seen.add(key); cs=caps(sent); ac=anchor_counts(sent)
                        rec={
                            "source_label": label,
                            "source_kind": spec["kind"],
                            "input_path": str(path),
                            "row_index": row_i,
                            "sentence_index": sent_i,
                            "field": field,
                            "text": sent,
                            "words": n,
                            "length_bin": core.len_bin(n),
                            "anchor_capabilities": cs,
                            "anchor_counts": ac,
                            "relation_marker_count": len(CONTROL_RELATION_RE.findall(sent)),
                            "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                            "doc_id": obj.get("doc_id"),
                            "norm_hash": obj.get("norm_hash"),
                        }
                        pool.append(rec); accepted+=1; got+=n
                        for c in cs: cap_words[c]+=n
        scan[label]={"path":str(path),"accepted_sentences":accepted,"accepted_words":got,"sentences_scanned":sent_scanned,"words_scanned":words_scanned,"capability_words":dict(cap_words)}
    return pool, scan


def pick_matched_controls(treatment: list[dict[str,Any]], pool: list[dict[str,Any]]) -> list[dict[str,Any]]:
    idx=defaultdict(list); idx_sc=defaultdict(list); idx_lc=defaultdict(list); idx_s=defaultdict(list); idx_c=defaultdict(list)
    for r in pool:
        src=str(r["source_label"]); lb=str(r["length_bin"])
        idx_s[src].append(r)
        for c in r.get("anchor_capabilities",[]):
            idx[(src,lb,c)].append(r); idx_sc[(src,c)].append(r); idx_lc[(lb,c)].append(r); idx_c[c].append(r)
    for rows in list(idx.values())+list(idx_sc.values())+list(idx_lc.values())+list(idx_s.values())+list(idx_c.values())+[pool]:
        rows.sort(key=lambda r:(int(r["words"]), -int(r.get("anchor_counts",{}).get("concrete",0)), r["text"]))
    used=set(); controls=[]; modes=Counter()
    def choose(rows, tw, req):
        best=None; bk=None
        for r in rows:
            if id(r) in used: continue
            key=(0 if req in r.get("anchor_capabilities",[]) else 1, abs(int(r["words"])-tw), -int(r.get("anchor_counts",{}).get("concrete",0)), r["text"])
            if bk is None or key<bk: best=r; bk=key
        if best is not None: used.add(id(best))
        return best
    for tr in sorted(treatment, key=lambda r:(-int(r["words"]), required_cap(r), str(r.get("text_sha256","")))):
        src=str(tr.get("source_label","")); lb=core.len_bin(int(tr["words"])); req=required_cap(tr); tw=int(tr["words"])
        ladders=[("same_source_lenbin_cap",idx[(src,lb,req)]),("same_source_cap",idx_sc[(src,req)]),("same_lenbin_cap",idx_lc[(lb,req)]),("any_source_cap",idx_c[req]),("same_source_any",idx_s[src]),("any",pool)]
        sel=None; mode="unmatched"
        for mi, rows in ladders:
            sel=choose(rows,tw,req)
            if sel is not None: mode=mi; break
        if sel is not None:
            rr=dict(sel); rr.update({"matched_treatment_words":tw,"matched_treatment_source_label":src,"matched_treatment_length_bin":lb,"required_capability":req,"match_mode":mode})
            controls.append(rr); modes[mode]+=1
        else:
            modes["unmatched"]+=1
    total=sum(int(r["words"]) for r in controls)
    # If row-matched controls are short, add no-relation anchored rows closest to the
    # current source/capability deficit until the word budget is as close as possible.
    if total < TARGET:
        rem=TARGET-total
        # Simple exact subset from unused pool with no score; prefer same source mix
        unused=[r for r in pool if id(r) not in used and int(r["words"])<=rem]
        # DP fill by word count first, slight preference for physical/concrete anchors.
        best=[-10**9]*(rem+1); prev=[None]*(rem+1); best[0]=0
        for i,r in enumerate(unused):
            w=int(r["words"]); val=10+int(r.get("anchor_counts",{}).get("concrete",0))*2+len(r.get("anchor_capabilities",[]))
            for s in range(rem,w-1,-1):
                if best[s-w] <= -10**8:
                    continue
                nv=best[s-w]+val
                if nv>best[s]: best[s]=nv; prev[s]=(s-w,i)
        reachable=[x for x,v in enumerate(best) if v>-10**8]
        s=max(reachable) if reachable else 0
        extra=[]; seen_i=set()
        while s>0 and prev[s] is not None:
            ps,i=prev[s]
            if i in seen_i: break
            seen_i.add(i); ex=dict(unused[i]); ex["match_mode"]="budget_topup_anchor_control"; extra.append(ex); s=ps
        controls.extend(extra); modes["budget_topup_anchor_control"]+=len(extra)
    for r in controls:
        r["control_modes_summary"] = dict(modes)
    return controls


def counter_words(rows, keyfunc):
    c=Counter()
    for r in rows: c[str(keyfunc(r))]+=int(r["words"])
    return c


def summarize(rows: list[dict[str,Any]], arm: str) -> dict[str,Any]:
    total=sum(int(r["words"]) for r in rows)
    out={"sentences":len(rows),"words":total,"mean_words":total/len(rows) if rows else 0,"median_words":statistics.median([int(r['words']) for r in rows]) if rows else 0,"source_words":dict(counter_words(rows,lambda r:r.get('source_label',''))),"length_bin_words":dict(counter_words(rows,lambda r:r.get('length_bin') or core.len_bin(int(r['words']))))}
    if arm=="treatment":
        out["bucket_words"]=dict(counter_words(rows,lambda r:r.get('selected_route_bucket','')))
        out["required_capability_words"]=dict(counter_words(rows,required_cap))
        out["mean_core_score"]=statistics.mean([float(r.get('core_score',0)) for r in rows]) if rows else 0
    else:
        out["required_capability_words"]=dict(counter_words(rows,lambda r:r.get('required_capability') or 'topup'))
        cap_words=Counter()
        for r in rows:
            for c in r.get('anchor_capabilities',[]): cap_words[c]+=int(r['words'])
        out["all_capability_words"]=dict(cap_words)
        out["match_modes"]=dict(counter_words(rows,lambda r:r.get('match_mode','')))
    return out


def l1(a: dict[str,int], b: dict[str,int]) -> float:
    keys=set(a)|set(b); ta=sum(a.values()) or 1; tb=sum(b.values()) or 1
    return sum(abs(a.get(k,0)/ta-b.get(k,0)/tb) for k in keys)


def main():
    t0=time.time(); OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    raw=load_jsonl(CANDIDATES)
    pool=[r for r in raw if treatment_keep(r)]
    pool.sort(key=lambda r:(str(r.get('selected_route_bucket')), -float(r.get('core_score',0)), abs(int(r['words'])-36), r.get('text_sha256','')))
    treatment=build_treatment(pool)
    control_pool, scan=collect_control_pool()
    controls=pick_matched_controls(treatment, control_pool)
    write_jsonl(OUT/'ultraclean_transition_candidate_pool.jsonl', pool)
    write_jsonl(OUT/'ultraclean_transition_50k.jsonl', treatment)
    write_jsonl(OUT/'ultraclean_anchor_control_50k.jsonl', controls)
    st=summarize(treatment,'treatment'); sc=summarize(controls,'control')
    metrics={
        'word_diff_control_minus_treatment': sc['words']-st['words'],
        'source_l1': l1(st['source_words'], sc['source_words']),
        'length_bin_l1': l1(st['length_bin_words'], sc['length_bin_words']),
        'required_capability_l1': l1(st['required_capability_words'], sc['required_capability_words']),
    }
    row_csv=OUT/'ultraclean_probe_summary.csv'
    with row_csv.open('w', newline='', encoding='utf-8') as f:
        fields=['arm','sentences','words','mean_words','median_words','source_words','length_bin_words','bucket_words','required_capability_words','all_capability_words','match_modes','mean_core_score']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm,sm in [('treatment',st),('anchor_control',sc)]:
            row={'arm':arm}
            for k in fields[1:]:
                v=sm.get(k,'')
                row[k]=json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
            w.writerow(row)
    summary={
        'status':'ULTRACLEAN_TRANSITION_PROBE_ASSETS_BUILT',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'input_core_candidates': str(CANDIDATES),
        'ultraclean_candidate_count': len(pool),
        'ultraclean_candidate_words': sum(int(r['words']) for r in pool),
        'treatment_summary': st,
        'control_summary': sc,
        'match_metrics': metrics,
        'control_pool_count': len(control_pool),
        'control_pool_words': sum(int(r['words']) for r in control_pool),
        'control_scan': scan,
        'files': {'candidate_pool':str(OUT/'ultraclean_transition_candidate_pool.jsonl'), 'treatment_50k':str(OUT/'ultraclean_transition_50k.jsonl'), 'anchor_control_50k':str(OUT/'ultraclean_anchor_control_50k.jsonl'), 'summary_csv':str(row_csv), 'note':str(NOTE)},
        'elapsed_sec': round(time.time()-t0,3),
    }
    (OUT/'ultraclean_transition_probe_assets.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    note=[
        '# research — ultra-clean transition probe assets', '',
        '## Purpose', '',
        'research/early research produced broad transition substrates, but sample audits exposed transcript, markup, abstract-political, and metaphor risk. This stricter CPU-only asset keeps only explicit relation markers plus concrete/action anchors and builds a no-explicit-relation anchored control. It is for a future cheap probe only if the running FW compact/breadth endpoints fail to move EWoK/GlobalPIQA relation failures.', '',
        '## Counts', '',
        f"- Ultra-clean candidate pool: {len(pool)} sentences / {summary['ultraclean_candidate_words']} words.",
        f"- Treatment slice: {st['sentences']} sentences / {st['words']} words; bucket words {st.get('bucket_words')}.",
        f"- Anchor control: {sc['sentences']} sentences / {sc['words']} words; word difference {metrics['word_diff_control_minus_treatment']}; source L1 {metrics['source_l1']:.4f}; length-bin L1 {metrics['length_bin_l1']:.4f}; required-capability L1 {metrics['required_capability_l1']:.4f}.", '',
        '## Scientific reading', '',
        'This 50k asset is cleaner and smaller than the 190k core slice, but it remains a probe substrate, not a new full route. The earlier INITIAL_MODEL_STUDIES evidence shows relation/binding data can move EWoK while hurting GlobalPIQA, so any use must compare treatment against this anchored control and read both EWoK interaction failures and GlobalPIQA hard-row margins. Do not launch it before the active FW endpoints are scored.', '',
        '## Files', '',
        f"- summary JSON: `{OUT/'ultraclean_transition_probe_assets.json'}`",
        f"- treatment: `{OUT/'ultraclean_transition_50k.jsonl'}`",
        f"- anchor control: `{OUT/'ultraclean_anchor_control_50k.jsonl'}`",
        f"- candidate pool: `{OUT/'ultraclean_transition_candidate_pool.jsonl'}`",
        f"- summary CSV: `{row_csv}`",
    ]
    NOTE.write_text('\n'.join(note)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'candidate_count':len(pool),'candidate_words':summary['ultraclean_candidate_words'],'treatment_words':st['words'],'control_words':sc['words'],'match_metrics':metrics,'json':str(OUT/'ultraclean_transition_probe_assets.json'),'note':str(NOTE),'elapsed_sec':summary['elapsed_sec']}, indent=2), flush=True)

if __name__=='__main__':
    main()
