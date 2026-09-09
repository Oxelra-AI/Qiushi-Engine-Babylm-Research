#!/usr/bin/env python3
"""research: Analyze Route 1/3 cheap screens and produce common-item comparisons."""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, re, statistics
from collections import defaultdict
from pathlib import Path

SCRIPT = _public_path('experiments/archive/representation_and_objectives/scripts/analyze_route_screens.py')
WS = _public_path('experiments/archive/representation_and_objectives')
EXT = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction')
SCORES = _public_path('experiments/archive/representation_and_objectives/data/cross_context_scores')
OUT = _public_path('experiments/archive/representation_and_objectives/data/route_screen_analysis')
OUT.mkdir(parents=True, exist_ok=True)

# Route 1 strict prose filtering
DIALOGUE_BAD = re.compile(r'\*(?:MOT|CHI|FAT|MAR|INV|EXP|BRO|SIS|GRA|AUN|UNC):|\b(?:yeah|yup|gonna|hafta|wanna)\b', re.I)
URL_BAD = re.compile(r'https?://|www\.|\.com\b', re.I)
# Require sentence-like prose with temporal/action structure but reject fragments and excessive markup.
with open(_public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/route1_procedural_passages.jsonl')) as f:
    raw_proc = [json.loads(x) for x in f]
strict_proc = []
for p in raw_proc:
    text = p["text"]
    if DIALOGUE_BAD.search(text) or URL_BAD.search(text):
        continue
    if text.count("*") > 1 or text.count("[") > 2 or text.count("=") > 2:
        continue
    if not text[0].isupper():
        continue
    if p["words"] < 25 or p["words"] > 180:
        continue
    # at least two clause/sentence boundaries
    if sum(text.count(x) for x in ".;:") < 2:
        continue
    strict_proc.append(p)
with open(_public_path('experiments/archive/representation_and_objectives/data/route_screen_analysis/route1_strict_procedural_passages.jsonl'), "w") as f:
    for p in strict_proc:
        f.write(json.dumps(p) + "\n")

# Load score payloads and align by stable item tuple.
files = {
    "chck82": _public_path('experiments/archive/representation_and_objectives/data/cross_context_scores/chck82.json'),
    "legal16k_base100": _public_path('experiments/archive/representation_and_objectives/data/cross_context_scores/legal16k_base100.json'),
    "legal40k_8x480_100": _public_path('experiments/archive/representation_and_objectives/data/cross_context_scores/legal40k_8x480_100.json'),
}
payloads = {k: json.load(open(v)) for k, v in files.items()}
def key(r):
    return (r["pair_type"], r["family"], r["target_a"], r["target_b"], r["row_a"], r["row_b"])
maps = {k: {key(r): r for r in p["records"]} for k,p in payloads.items()}
common = set.intersection(*(set(m.keys()) for m in maps.values()))

def agg(rs):
    return {
        "n": len(rs),
        "crossed_success": sum(r["crossed_success"] for r in rs)/len(rs),
        "mean_delta": statistics.mean(r["delta"] for r in rs),
        "median_delta": statistics.median(r["delta"] for r in rs),
        "delta_positive_frac": sum(r["delta"]>0 for r in rs)/len(rs),
    }
common_summary = {name: agg([mp[k] for k in common]) for name, mp in maps.items()}
# Break down common items by type.
by_type = {}
for typ in ["antonym", "same_entity"]:
    ks = [k for k in common if k[0] == typ]
    by_type[typ] = {name: agg([mp[k] for k in ks]) for name,mp in maps.items()}
# Pair family summary for chck82, retaining n>=10.
ch = maps["chck82"]
fams=defaultdict(list)
for k in common:
    fams[k[1]].append(ch[k])
family_summary={fam:agg(rs) for fam,rs in fams.items() if len(rs)>=10}

summary = {
    "status": "ROUTE_SCREEN_ANALYSIS",
    "route1_raw_candidates": len(raw_proc),
    "route1_strict_prose_candidates": len(strict_proc),
    "route1_strict_word_sum": sum(p["words"] for p in strict_proc),
    "route1_examples": strict_proc[:10],
    "route3_common_items": len(common),
    "route3_common_summary": common_summary,
    "route3_by_type": by_type,
    "route3_chck82_family_summary": family_summary,
    "score_files": {k:str(v) for k,v in files.items()},
}
with open(_public_path('experiments/archive/representation_and_objectives/data/route_screen_analysis/route_screen_analysis.json'), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ("route1_examples","route3_chck82_family_summary")},indent=2))
print("Route1 strict examples:")
for p in strict_proc[:5]:
    print("-",p["text"][:250])
print("chck82 family summaries:")
for fam,s in sorted(family_summary.items(), key=lambda x:x[1]["crossed_success"]):
    print(f"  {fam}: n={s['n']} crossed={s['crossed_success']:.3f} delta={s['mean_delta']:.3f}")
