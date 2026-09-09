#!/usr/bin/env python3
"""research transition/control sample quality audit.

Counts obvious noise markers and produces a bounded stratified sample note for the
research core-transition treatment and control assets.  CPU-only, no evaluation item
text, no training.
"""
from __future__ import annotations
import csv, importlib.util, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
CORE_SCRIPT = A01_WS / "scripts/core_transition_filter_and_controls.py"
CORE_DIR = A01_WS / "data/core_transition_filter"
CTRL_V2_DIR = A01_WS / "data/anchor_matched_controls_v2"
OUT = A01_WS / "data/transition_sample_quality"
NOTE = A01_WS / "notes/transition_sample_quality_audit.md"

spec = importlib.util.spec_from_file_location("core", CORE_SCRIPT)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)  # type: ignore[union-attr]

FILES = {
    "core_treatment_50k": CORE_DIR / "core_transition_treatment_50k.jsonl",
    "core_treatment_100k": CORE_DIR / "core_transition_treatment_100k.jsonl",
    "core_treatment_200k": CORE_DIR / "core_transition_treatment_200k.jsonl",
    "neutral_lengthmatched_50k": CORE_DIR / "core_transition_neutral_lengthmatched_50k.jsonl",
    "anchor_control_v2_50k": CTRL_V2_DIR / "core_transition_anchor_control_v2_50k.jsonl",
    "anchor_control_v2_100k": CTRL_V2_DIR / "core_transition_anchor_control_v2_100k.jsonl",
    "anchor_control_v2_200k": CTRL_V2_DIR / "core_transition_anchor_control_v2_200k.jsonl",
}

NOISE_PATTERNS = {
    "childes_path": re.compile(r"\bchildes/|\.cha\b", re.I),
    "transcript_speaker": re.compile(r"\*[A-Z0-9]{2,6}:", re.I),
    "subtitle_pos": re.compile(r"\\pos\(|\{\\", re.I),
    "ellipsis_artifact": re.compile(r"\.\.\.|…"),
    "bracket_stage_dir": re.compile(r"\[[^\]]{0,80}\]"),
    "underscore_markup": re.compile(r"_{2,}|\b_\w+_\b"),
    "equals_markup": re.compile(r"(?:=\s*){3,}"),
    "url_html": re.compile(r"https?://|www\.|</?\w+", re.I),
    "starts_lowercase_fragment": re.compile(r"^[a-z]"),
}
STRONG_RELATION_RE = re.compile(
    r"\b(if|when|whenever|unless|because|since|therefore|thus|so that|as a result|results? in|leads? to|causes?|prevent(?:s|ed)?|allows?|"
    r"became|becomes?|turn(?:s|ed)? into|change(?:d|s)?|increase(?:d|s)?|decrease(?:d|s)?|full|empty|hot|cold|wet|dry|open(?:ed|s)?|close(?:d|s)?|seal(?:ed|s)?|break(?:s|ing)?|broke|broken|melt(?:s|ed)?|freez(?:e|es|ing|en)|dissolv(?:e|es|ed)|fall(?:s|en|ing)?|fell|drop(?:s|ped)?|rise[sn]?|rose|"
    r"before|after|first|then|next|finally|instead|rather than|whereas|although|but|from\b.{0,45}\bto)\b",
    re.I,
)
ABSTRACT_EXTRA_RE = re.compile(r"\b(government|policy|election|parliament|minister|market|company|business|rights|law|court|argument|opinion|belief|idea|story|movie|religion|campaign|internet|website|software|data|research|study|competition|winning|love|happiness|music|dance|agency|report|political|army|party|finance|school|student|university)\b", re.I)


def load(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def flags(text: str) -> dict[str, int]:
    out = {name: int(bool(rx.search(text))) for name, rx in NOISE_PATTERNS.items()}
    out["strong_relation_marker_count"] = len(STRONG_RELATION_RE.findall(text))
    out["abstract_extra_count"] = len(ABSTRACT_EXTRA_RE.findall(text))
    return out


def bucket(row: dict[str,Any], label: str) -> str:
    if label.startswith("core_treatment"):
        return str(row.get("selected_route_bucket") or "unknown")
    if label.startswith("anchor_control"):
        return str(row.get("required_capability") or "unknown")
    return str(row.get("source_label") or "unknown")


def sample_rows(rows: list[dict[str,Any]], label: str, n_per_bucket=5):
    by=defaultdict(list)
    for r in rows:
        by[bucket(r,label)].append(r)
    out=[]
    for b, arr in sorted(by.items()):
        # sample deterministic top, middle, short/long diversity
        arr2=sorted(arr, key=lambda r: (-int(r.get('words',0)), str(r.get('text',''))))
        picks=[]
        for idx in [0, len(arr2)//4, len(arr2)//2, 3*len(arr2)//4, len(arr2)-1]:
            if 0 <= idx < len(arr2):
                r=arr2[idx]
                if id(r) not in {id(x) for x in picks}: picks.append(r)
        for r in picks[:n_per_bucket]:
            out.append((b,r))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    summary=[]
    sample_md=[]
    for label,path in FILES.items():
        rows=load(path)
        word_total=sum(int(r.get('words', len(core.words(r.get('text',''))))) for r in rows)
        cnt=Counter(); relation_counts=[]; abs_counts=[]; by_bucket=Counter(); words_by_bucket=Counter()
        for r in rows:
            text=str(r.get('text',''))
            fl=flags(text)
            for k,v in fl.items():
                if k.endswith('_count'):
                    if k=='strong_relation_marker_count': relation_counts.append(v)
                    elif k=='abstract_extra_count': abs_counts.append(v)
                elif v:
                    cnt[k]+=1
            b=bucket(r,label); by_bucket[b]+=1; words_by_bucket[b]+=int(r.get('words',0))
        summary.append({
            'label': label, 'path': str(path), 'sentences': len(rows), 'words': word_total,
            'noise_any_rows': sum(1 for r in rows if any(v for k,v in flags(str(r.get('text',''))).items() if not k.endswith('_count'))),
            'noise_any_frac': round(sum(1 for r in rows if any(v for k,v in flags(str(r.get('text',''))).items() if not k.endswith('_count')))/len(rows),6) if rows else 0,
            'strong_relation_rows': sum(1 for v in relation_counts if v>0),
            'strong_relation_frac': round(sum(1 for v in relation_counts if v>0)/len(rows),6) if rows else 0,
            'strong_relation_mean': round(statistics.mean(relation_counts),3) if relation_counts else 0,
            'abstract_extra_rows': sum(1 for v in abs_counts if v>0),
            'abstract_extra_frac': round(sum(1 for v in abs_counts if v>0)/len(rows),6) if rows else 0,
            **{f'noise_{k}': v for k,v in cnt.items()},
            'bucket_sentences': json.dumps(dict(by_bucket), sort_keys=True),
            'bucket_words': json.dumps(dict(words_by_bucket), sort_keys=True),
        })
        sample_md += [f"## {label}", f"Path: `{path}`", ""]
        for b,r in sample_rows(rows,label):
            text=str(r.get('text','')).replace('\n',' ')
            fl=flags(text)
            sample_md.append(f"- **{b}** w={r.get('words')} rel={fl['strong_relation_marker_count']} noise={[k for k,v in fl.items() if v and not k.endswith('_count')]} :: {text[:900]}")
        sample_md.append("")
    csv_path=OUT/'transition_sample_quality_summary.csv'
    fields=sorted({k for row in summary for k in row.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    json_path=OUT/'transition_sample_quality_summary.json'
    json_path.write_text(json.dumps({'status':'TRANSITION_SAMPLE_QUALITY_AUDIT_DONE','files':{k:str(v) for k,v in FILES.items()},'summary':summary,'csv':str(csv_path),'note':str(NOTE)}, indent=2, ensure_ascii=False), encoding='utf-8')
    NOTE.write_text("# research — transition/control sample quality audit\n\nThis bounded audit counts obvious noise markers and strong relation markers in the research treatment/control assets, then prints deterministic stratified samples. It is not semantic annotation, but it exposes residual corpus-artifact risk before any probe launch.\n\n## Summary CSV\n\n`"+str(csv_path)+"`\n\n"+"\n".join(sample_md)+"\n", encoding='utf-8')
    print(json.dumps({'status':'TRANSITION_SAMPLE_QUALITY_AUDIT_DONE','json':str(json_path),'csv':str(csv_path),'note':str(NOTE),'labels':len(summary)}, indent=2), flush=True)

if __name__=='__main__':
    main()
