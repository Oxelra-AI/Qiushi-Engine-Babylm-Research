#!/usr/bin/env python3
"""research strict deterministic PVDM labels for a compliant warm-start test.

This is the training-facing replacement for the too-permissive first deterministic
label pass. It intentionally avoids learned NLP tools and external linguistic
models. It uses fixed, inspectable lexical/window rules and frequency bins derived
only from the allowed compact 10M pool.

Scientific goal: make a legal PVDM/control pair where treatment and control have
identical dependent-target identities and masked mass, while differing in whether
the protected visible anchor is the relational pivot or a nonrelational surrogate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
STREAM_100M = WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
POOL_10M = WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
TOKENIZER_JSON = WS / "data/shared_tokenizer/shared_16k_tokenizer/tokenizer.json"
OUT = WS / "data/pvdm_strict_labels"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/pvdm_strict_label_quality.md')
EXPECTED_STREAM_SHA = "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68"
EXPECTED_POOL_SHA = "ee08a1f8d974248aa9bdda14dfe48724b93873e6961db137d9bbdd078e5ef914"
EXPECTED_TOKENIZER_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"
START_WORDS = 70_000_000
TAIL_WORDS = 30_000_000
LEAD_TRAIL = " \t\n\r\"'“”‘’()[]{}<>.,;:!?/\\|`~*_=+"
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
NUM_RE = re.compile(r"^[+-]?(?:\d+[\d,]*(?:\.\d+)?|\.\d+)(?:[%°])?$")

SPEAKER_OR_ARTIFACT = {
    "chi","mot","fat","mar","bro","sis","mom","dad","par","inv","int","exp","add","com","act","pho","sit","gra","gpx","x","xx","xxx","uh","uhhuh","huh","hm","yeah","okay","ok","ooh","ah","oh"
}
STOP_TARGETS = {
    "a","an","the","and","or","but","of","for","to","from","with","without","by","as","at","be","am","is","are","was","were","been","being",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","there","here",
    "do","does","did","have","has","had","what","which","who","whom","whose","where","why","how","yes","no","not","just","only","also","very","all","some","any","one","two",
    "get","go","come","say","said","tell","know","think","see","look","like","want","need","let","put","take","make","got","gonna","wanna","sposta","sorta","kinda"
} | SPEAKER_OR_ARTIFACT
FUNCTION_ANCHORS = {
    "the","a","an","and","or","but","of","for","by","as","at","this","that","these","those","there","here","also","very","just","only","some","any","all","another","other"
}

PHYSICAL_CHANGE = {
    "cause","causes","caused","causing","lead","leads","led","leading","result","results","resulted","resulting",
    "force","forces","forced","forcing","prevent","prevents","prevented","preventing","allow","allows","allowed","allowing","enable","enables","enabled","enabling",
    "melt","melts","melted","melting","freeze","freezes","froze","frozen","freezing","heat","heats","heated","heating","cool","cools","cooled","cooling",
    "break","breaks","broke","broken","breaking","fix","fixes","fixed","fixing","open","opens","opened","opening","close","closes","closed","closing",
    "increase","increases","increased","increasing","decrease","decreases","decreased","decreasing","rise","rises","rose","risen","rising","fall","falls","fell","fallen","falling",
    "grow","grows","grew","grown","growing","shrink","shrinks","shrank","shrunk","shrinking","expand","expands","expanded","expanding","contract","contracts","contracted","contracting",
    "push","pushes","pushed","pushing","pull","pulls","pulled","pulling","lift","lifts","lifted","lifting","drop","drops","dropped","dropping",
    "move","moves","moved","moving","enter","enters","entered","entering","leave","leaves","left","leaving","add","adds","added","adding","remove","removes","removed","removing",
    "build","builds","built","building","destroy","destroys","destroyed","destroying","create","creates","created","creating","turn","turns","turned","turning","change","changes","changed","changing",
    "carry","carries","carried","carrying","throw","throws","threw","thrown","throwing","pour","pours","poured","pouring","fill","fills","filled","filling","empty","empties","emptied","emptying"
}
STRONG_SPATIAL = {"above","below","under","beneath","behind","beside","between","inside","outside","within","across","through","toward","towards","into","onto","against","beyond"}
TEMPORAL = {"before","after","during","until","while","when","since","once"}
CAUSAL = {"because","therefore","thereby","thus","hence","unless","although","though","whereas","if"}
COMPARATIVE = {"more","less","greater","smaller","larger","higher","lower","faster","slower","longer","shorter","better","worse","least","most"}
NEGATION = {"not","never","cannot","can't","couldn't","won't","wouldn't","shouldn't","isn't","aren't","wasn't","weren't","don't","doesn't","didn't","without"}
PIVOTS = PHYSICAL_CHANGE | STRONG_SPATIAL | TEMPORAL | CAUSAL | COMPARATIVE | NEGATION | {"than"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(w: str) -> str:
    return w.strip(LEAD_TRAIL).lower()


def bad_surface(w: str) -> bool:
    raw = w.strip()
    n = norm(raw)
    if not n or PUNCT_RE.match(n): return True
    if raw.startswith(("*", "%", "[", "]", "=")): return True
    if "[" in raw or "]" in raw or "=" in raw: return True
    if n in SPEAKER_OR_ARTIFACT: return True
    return False


def is_number(n: str) -> bool:
    return bool(NUM_RE.match(n.replace(",", "")))


def target_class(w: str) -> str:
    n = norm(w)
    if is_number(n): return "number"
    if w[:1].isupper(): return "capitalized"
    if n.endswith(("ing","ed","en","ize","ise","ify")): return "verbish"
    if n.endswith(("tion","ment","ness","ity","ance","ence","ship","age","ism","ure")): return "nominal"
    if n.endswith(("ive","al","ous","ful","less","able","ible","ic","ary")): return "adjectival"
    if n.endswith("ly"): return "adverbish"
    return "content"


def anchor_class_for_pivot(cat: str, pivot_word: str) -> str:
    if cat in {"spatial", "temporal", "causal_connector", "negation"}:
        return "function"
    if cat == "comparative":
        return "adjectival"
    return "verbish"


def coarse_anchor_class(w: str) -> str:
    n = norm(w)
    if n in FUNCTION_ANCHORS: return "function"
    if n in NEGATION or n in TEMPORAL or n in CAUSAL or n in STRONG_SPATIAL: return "function"
    if n in COMPARATIVE or n.endswith(("er","est")): return "adjectival"
    if n in PHYSICAL_CHANGE or n.endswith(("ing","ed","en","ize","ise","ify")): return "verbish"
    if is_number(n): return "number"
    if w[:1].isupper(): return "capitalized"
    return "content"


def is_content_target(w: str) -> bool:
    n = norm(w)
    if bad_surface(w): return False
    if n in STOP_TARGETS or n in PIVOTS: return False
    if len(n) <= 2 and not is_number(n): return False
    return True


def freq_bin_id(f: int) -> int:
    for i, hi in enumerate([0,2,5,10,25,50,100,250,500,1000]):
        if f <= hi: return i
    return 10


def freq_bin(f: int) -> str:
    names=["0","1-2","3-5","6-10","11-25","26-50","51-100","101-250","251-500","501-1000","1001+"]
    return names[freq_bin_id(f)]


def dist_bin(d: int) -> str:
    if d <= 1: return "1"
    if d <= 2: return "2"
    if d <= 4: return "3-4"
    if d <= 8: return "5-8"
    if d <= 16: return "9-16"
    return "17+"


def pivot_category(words: list[str], i: int) -> str | None:
    n = norm(words[i])
    if bad_surface(words[i]): return None
    if n in PHYSICAL_CHANGE: return "physical_change"
    if n in STRONG_SPATIAL: return "spatial"
    if n in TEMPORAL: return "temporal"
    if n in CAUSAL: return "causal_connector"
    if n in NEGATION: return "negation"
    if n in COMPARATIVE:
        # Keep comparatives only when a nearby comparison cue exists; this removes generic "more" floods.
        lo, hi = max(0, i-4), min(len(words), i+6)
        if any(norm(w) == "than" for w in words[lo:hi]) or n not in {"more","most","less","least"}:
            return "comparative"
    if n == "than": return "comparative"
    return None


def first_targets_after(words: list[str], i: int, window: int, cap: int) -> list[int]:
    out=[]
    for j in range(i+1, min(len(words), i+1+window)):
        raw=words[j].strip()
        if raw.endswith((".","?","!",";")) and out:
            break
        if is_content_target(words[j]):
            out.append(j)
            if len(out) >= cap: break
    return out


def first_targets_before(words: list[str], i: int, window: int, cap: int) -> list[int]:
    out=[]
    for j in range(i-1, max(-1, i-1-window), -1):
        if is_content_target(words[j]):
            out.append(j)
            if len(out) >= cap: break
    return out


def collect_targets(words: list[str], i: int, cat: str) -> list[int]:
    if cat == "physical_change":
        return first_targets_after(words, i, 6, 1)
    if cat == "spatial":
        return first_targets_after(words, i, 5, 1)
    if cat in {"temporal", "causal_connector"}:
        out = first_targets_after(words, i, 8, 1)
        if not out:
            out = first_targets_before(words, i, 6, 1)
        return out
    if cat == "negation":
        return first_targets_after(words, i, 5, 1)
    if cat == "comparative":
        out=[]
        # If this is a than-comparison, include the compared noun/property on either side when present.
        if norm(words[i]) == "than":
            out += first_targets_before(words, i, 5, 1)
            out += first_targets_after(words, i, 5, 1)
        else:
            out += first_targets_after(words, i, 6, 1)
        seen=set(); final=[]
        for j in out:
            if j not in seen and j != i:
                seen.add(j); final.append(j)
        return final[:1]
    return []


def build_freqs(path: Path) -> Counter[str]:
    c=Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            for w in json.loads(line)["text"].split():
                n=norm(w)
                if n and not bad_surface(w): c[n]+=1
    return c


def choose_control(words: list[str], pivot_i: int, target_i: int, cat: str, freqs: Counter[str], used_controls: set[int]) -> tuple[int | None, dict[str, Any]]:
    p_norm = norm(words[pivot_i])
    p_class = anchor_class_for_pivot(cat, p_norm)
    p_freq_bin = freq_bin_id(freqs.get(p_norm, 0))
    desired_dist = abs(target_i - pivot_i)
    desired_side = 1 if pivot_i > target_i else -1
    best=None
    for j,w in enumerate(words):
        if j in {pivot_i, target_i} or j in used_controls: continue
        nw=norm(w)
        if bad_surface(w) or nw in PIVOTS: continue
        c_class=coarse_anchor_class(w)
        if c_class != p_class: continue
        d=abs(target_i-j)
        if d < 1 or d > 20: continue
        jbin=freq_bin_id(freqs.get(nw, 0))
        side=1 if j > target_i else -1
        score=2.8*abs(math.log1p(d)-math.log1p(desired_dist))+1.5*abs(jbin-p_freq_bin)+0.25*(side != desired_side)+0.01*abs(j-target_i)
        if best is None or score < best[0]:
            best=(score,j,{"control_norm":nw,"control_class":c_class,"pivot_anchor_class":p_class,"control_freq_bin":freq_bin(freqs.get(nw,0)),"pivot_freq_bin":freq_bin(freqs.get(p_norm,0)),"control_distance":d,"pivot_target_distance":desired_dist,"distance_abs_diff":abs(d-desired_dist),"freq_bin_abs_diff":abs(jbin-p_freq_bin),"score":round(score,4)})
    if best is None:
        return None, {"matched": False, "pivot_anchor_class": p_class}
    return best[1], {"matched": True, **best[2]}


def stream_tail(path: Path):
    cum=0; tail=0
    with path.open(encoding="utf-8") as f:
        for row_idx,line in enumerate(f):
            if not line.strip(): continue
            obj=json.loads(line)
            words=int(obj.get("words", len(str(obj["text"]).split())))
            actual=len(str(obj["text"]).split())
            if words != actual: raise RuntimeError(f"word mismatch row {row_idx}: {words}!={actual}")
            if cum + words <= START_WORDS:
                cum += words; continue
            if cum < START_WORDS: raise RuntimeError(f"tail starts inside row {row_idx}: cum={cum} words={words}")
            if tail + words > TAIL_WORDS: raise RuntimeError(f"tail ends inside row {row_idx}: tail={tail} words={words}")
            yield row_idx, obj
            cum += words; tail += words
            if tail == TAIL_WORDS: return
    raise RuntimeError(f"tail words {tail}!={TAIL_WORDS}")


def row_events(words: list[str], freqs: Counter[str], max_events: int) -> tuple[list[dict[str, Any]], int]:
    raw=[]
    for i in range(len(words)):
        cat=pivot_category(words,i)
        if cat is None: continue
        for t in collect_targets(words,i,cat):
            raw.append((i,t,cat))
    # Prefer rarer/stronger categories over high-frequency function-like categories.
    priority={"physical_change":0,"causal_connector":1,"spatial":2,"temporal":3,"comparative":4,"negation":5}
    raw.sort(key=lambda x:(priority.get(x[2],9), abs(x[1]-x[0]), x[0]))
    events=[]; used_targets=set(); used_controls=set(); no_control=0
    for p,t,cat in raw:
        if len(events) >= max_events: break
        if t in used_targets: continue
        ci, detail = choose_control(words,p,t,cat,freqs,used_controls)
        if ci is None:
            no_control += 1; continue
        pn, tn, cn = norm(words[p]), norm(words[t]), norm(words[ci])
        ev={
            "pivot_i":p,"target_i":t,"control_i":ci,"category":cat,
            "pivot":words[p],"target":words[t],"control":words[ci],
            "pivot_norm":pn,"target_norm":tn,"control_norm":cn,
            "pivot_anchor_class":anchor_class_for_pivot(cat,pn),"control_anchor_class":coarse_anchor_class(words[ci]),"target_class":target_class(words[t]),
            "pivot_freq_bin":freq_bin(freqs.get(pn,0)),"control_freq_bin":freq_bin(freqs.get(cn,0)),"target_freq_bin":freq_bin(freqs.get(tn,0)),
            "pivot_target_distance":abs(t-p),"control_target_distance":abs(t-ci),"distance_bin":dist_bin(abs(t-p)),"control_match":detail,
        }
        events.append(ev); used_targets.add(t); used_controls.add(ci)
    return events, no_control


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-events-per-row", type=int, default=4)
    ap.add_argument("--sample-limit", type=int, default=500)
    args=ap.parse_args()
    t0=time.time(); OUT.mkdir(parents=True, exist_ok=True)
    hashes={"stream":sha256_file(STREAM_100M),"pool":sha256_file(POOL_10M),"tokenizer":sha256_file(TOKENIZER_JSON)}
    if hashes["stream"] != EXPECTED_STREAM_SHA or hashes["pool"] != EXPECTED_POOL_SHA or hashes["tokenizer"] != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"hash mismatch {hashes}")
    print(json.dumps({"event":"strict_freq_start"}), flush=True)
    freqs=build_freqs(POOL_10M)
    print(json.dumps({"event":"strict_freq_done","types":len(freqs),"elapsed_sec":round(time.time()-t0,1)}), flush=True)
    labels_path=OUT/"pvdm_strict_tail_70M_100M_labels.jsonl"
    tail_path=OUT/"compact_tail_70M_100M.jsonl"
    sample_path=OUT/"pvdm_strict_event_samples.jsonl"
    stats=Counter(); cats=Counter(); target_classes=Counter(); target_bins=Counter(); pivot_bins=Counter(); control_bins=Counter(); dist_bins=Counter(); d_abs=Counter(); f_abs=Counter(); anchor_classes=Counter(); top_targets=Counter(); top_pivots=Counter(); top_controls=Counter(); samples=[]
    with labels_path.open("w",encoding="utf-8") as lf, tail_path.open("w",encoding="utf-8") as tf:
        for local_idx,(orig_idx,obj) in enumerate(stream_tail(STREAM_100M)):
            text=str(obj["text"]); words=text.split()
            tf.write(json.dumps(obj,ensure_ascii=False)+"\n")
            events,no_control=row_events(words,freqs,args.max_events_per_row)
            targets=sorted({e["target_i"] for e in events}); pivots=sorted({e["pivot_i"] for e in events}); controls=sorted({e["control_i"] for e in events})
            rec={"tail_row_idx":local_idx,"orig_row_idx":orig_idx,"example_id":int(obj.get("example_id",orig_idx)),"source":obj.get("source",""),"words":int(obj.get("words",len(words))),"text":text,"pvdm_pivot_indices":pivots,"dependent_target_indices":targets,"control_anchor_indices":controls,"events":events}
            lf.write(json.dumps(rec,ensure_ascii=False)+"\n")
            stats["rows"]+=1; stats["words"]+=len(words); stats["events"]+=len(events); stats["events_without_control"]+=no_control
            stats["target_positions_unique_sum"]+=len(targets); stats["pivot_positions_unique_sum"]+=len(pivots); stats["control_positions_unique_sum"]+=len(controls)
            if events:
                stats["eligible_rows"]+=1
                if len(samples)<args.sample_limit: samples.append({"tail_row_idx":local_idx,"orig_row_idx":orig_idx,"text":text,"events":events})
            for e in events:
                cats[e["category"]]+=1; target_classes[e["target_class"]]+=1; target_bins[e["target_freq_bin"]]+=1; pivot_bins[e["pivot_freq_bin"]]+=1; control_bins[e["control_freq_bin"]]+=1; dist_bins[e["distance_bin"]]+=1
                d_abs[str(e["control_match"].get("distance_abs_diff","NA"))]+=1; f_abs[str(e["control_match"].get("freq_bin_abs_diff","NA"))]+=1; anchor_classes[f"{e['pivot_anchor_class']}->{e['control_anchor_class']}"]+=1
                top_targets[e["target_norm"]]+=1; top_pivots[e["pivot_norm"]]+=1; top_controls[e["control_norm"]]+=1
            if (local_idx+1)%50000==0:
                print(json.dumps({"event":"strict_rows","rows":local_idx+1,"words":stats['words'],"events":stats['events'],"eligible_rows":stats['eligible_rows'],"elapsed_sec":round(time.time()-t0,1)}), flush=True)
    with sample_path.open("w",encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    target_hash=hashlib.sha256()
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); target_hash.update(json.dumps({"row":r["tail_row_idx"],"targets":r["dependent_target_indices"]},sort_keys=True).encode()+b"\n")
    summary={
        "status":"STRICT_DETERMINISTIC_PVDM_LABELS_READY",
        "compliance":{"no_learned_parser_or_pos_tagger":True,"rule_source":"hand-coded deterministic rules in this script; no model outputs","frequency_source":str(POOL_10M),"frequency_source_sha256":hashes['pool'],"stream_sha256":hashes['stream'],"tokenizer_sha256":hashes['tokenizer'],"excluded_from_training":"research spaCy pilot"},
        "tail":{"start_word_exposure":START_WORDS,"tail_words":TAIL_WORDS,"rows":stats['rows']},
        "counts":dict(stats),
        "rates":{"eligible_row_fraction":stats['eligible_rows']/max(1,stats['rows']),"events_per_eligible_row":stats['events']/max(1,stats['eligible_rows']),"unique_targets_per_row":stats['target_positions_unique_sum']/max(1,stats['rows']),"target_positions_per_1000_words":1000*stats['target_positions_unique_sum']/max(1,stats['words'])},
        "events_by_category":dict(cats),"target_class_distribution":dict(target_classes),"target_frequency_bins":dict(target_bins),"pivot_frequency_bins":dict(pivot_bins),"control_frequency_bins":dict(control_bins),"pivot_target_distance_bins":dict(dist_bins),
        "control_match_quality":{"anchor_class_pairs":dict(anchor_classes),"distance_abs_diff_counts":dict(d_abs),"freq_bin_abs_diff_counts":dict(f_abs)},
        "target_distribution_equality":{"treatment_and_control_use_identical_dependent_target_indices":True,"target_identity_sequence_sha256":target_hash.hexdigest(),"top_targets":top_targets.most_common(50),"top_pivots":top_pivots.most_common(50),"top_controls":top_controls.most_common(50)},
        "files":{"labels_jsonl":str(labels_path),"labels_sha256":sha256_file(labels_path),"tail_jsonl":str(tail_path),"tail_sha256":sha256_file(tail_path),"samples_jsonl":str(sample_path),"samples_sha256":sha256_file(sample_path)},
        "runtime_sec":round(time.time()-t0,2),
    }
    (OUT/"pvdm_strict_label_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    NOTE.write_text(f"""# research — strict deterministic PVDM label quality

The first deterministic label pass was legally clean but scientifically too permissive: nearly every row was eligible and top targets included speaker tags. This stricter training-facing pass excludes learned NLP tools and uses only fixed lexical/window rules plus frequency bins from the allowed 10M compact pool.

## Main facts

- rows/words: {stats['rows']:,} rows / {stats['words']:,} words in the 70M→100M tail
- eligible rows: {stats['eligible_rows']:,} ({summary['rates']['eligible_row_fraction']:.3f})
- accepted events: {stats['events']:,}; unique dependent targets: {stats['target_positions_unique_sum']:,}
- target positions per 1000 words: {summary['rates']['target_positions_per_1000_words']:.2f}
- identical dependent-target sequence for treatment/control: `{summary['target_distribution_equality']['target_identity_sequence_sha256']}`
- labels: `{labels_path}`
- tail JSONL for continuation: `{tail_path}`
- summary: `{OUT/'pvdm_strict_label_summary.json'}`
- samples: `{sample_path}`

Use these strict labels, not the permissive `pvdm_deterministic_labels`, for any training-facing PVDM experiment. The trainer must still normalize masking so PVDM and control have identical expected/actual masked mass per batch and must run a standard-masking parity check before any 30M GPU continuation.
""",encoding="utf-8")
    print(json.dumps({"status":summary['status'],"events":stats['events'],"eligible_rows":stats['eligible_rows'],"target_per_1000_words":summary['rates']['target_positions_per_1000_words'],"labels":str(labels_path),"tail":str(tail_path),"summary":str(OUT/'pvdm_strict_label_summary.json'),"runtime_sec":summary['runtime_sec']},indent=2),flush=True)

if __name__ == "__main__":
    main()
