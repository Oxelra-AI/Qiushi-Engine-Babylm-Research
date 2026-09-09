#!/usr/bin/env python3
"""research: compare the clean-Qwen control corpus and compact-view-reinvest corpus in
training-coordinate geometry, without using evaluation text and without launching GPU work.

The current route-selecting dependency is the within-legal-tokenizer clean-vs-reinvest
mature trajectory. This script prepares an independent interpretation asset: under the
same legal research tokenizer, what changed in the *training substrate* between the clean
control and the compact-view reinvest corpus?  It measures only corpus/tokenizer geometry
and source-class structure on the allowed training files, so it cannot decide route
selection by itself, but it helps interpret the mature treatment vector when it lands.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

SESSION_ROOT = _public_path('experiments/archive/frontier_consolidation')
SESSIONS_ROOT = _public_path('experiments/archive')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry')
A02_DATA = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard')
CLEAN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
CLEAN_100M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
REINVEST_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
REINVEST_100M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
REPEAT_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl')
LENGTHMATCHED_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_lengthmatched_compact_reinvest_10M.jsonl')
META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json')
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
TOKENIZER_JSON = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')
EXPECTED = {
    "clean_10m_sha256": "e0a3cdc20e39f2715fbdb0cfbc6c4aff61d51924c480a982049878272c5690b3",
    "clean_100m_sha256": "728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345",
    "reinvest_10m_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "reinvest_100m_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    "tokenizer_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
}

# Coarse linguistic/mechanistic cue lexicon. This is not a benchmark-derived prior;
# it is used only to compare training-corpus substrate composition.
CUE_SETS = {
    "spatial_state": {
        "above","across","against","along","around","at","away","behind","below","beneath","beside","between","beyond",
        "down","from","here","in","inside","into","near","next","off","on","onto","outside","over","there","through",
        "toward","towards","under","up","where","within","left","right","front","back","top","bottom","side","middle",
        "north","south","east","west","place","position","location","room","house","street","city","country","river",
    },
    "causal_temporal": {
        "after","again","already","always","before","because","cause","caused","causes","during","early","eventually",
        "finally","first","if","later","next","never","now","once","since","soon","then","therefore","until","when",
        "while","why","will","would","could","should","may","might","must","still","time","times","ago","today","tomorrow",
        "yesterday","happen","happened","happens","result","results","reason","change","changed","become","became",
    },
    "physical_dynamics": {
        "move","moves","moved","moving","fall","falls","fell","fallen","drop","dropped","push","pushed","pull","pulled",
        "hit","hits","break","broke","broken","turn","turned","run","ran","walk","walked","fly","flew","drive","driven",
        "open","opened","close","closed","stop","stopped","start","started","grow","grew","increase","increased","decrease",
        "decreased","accelerate","accelerating","slow","slowing","rise","rose","raise","raised","lower","lowered","carry",
        "carried","hold","held","put","take","took","give","gave","make","made","use","used","work","worked",
    },
    "material_property": {
        "hot","cold","warm","cool","hard","soft","heavy","light","big","small","large","little","long","short","high",
        "low","strong","weak","dry","wet","clean","dirty","red","blue","green","white","black","bright","dark","new",
        "old","young","same","different","good","bad","better","worse","full","empty","water","gas","air","wood","metal",
        "stone","paper","food","body","blood","ice","fire","energy","temperature","size","weight","shape","color",
    },
    "quantity_measure": {
        "one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand","million","billion",
        "many","much","more","most","less","least","few","several","number","amount","part","parts","percent","half",
        "year","years","month","months","day","days","hour","hours","minute","minutes","second","seconds","age","old",
    },
    "mental_social_dialogue": {
        "ask","asked","answer","answered","call","called","hear","heard","listen","look","looked","read","say","said",
        "says","see","saw","seen","speak","spoke","talk","talked","tell","told","think","thought","know","knew","believe",
        "believed","want","wanted","need","needed","like","liked","love","loved","feel","felt","friend","family","mother",
        "father","child","children","man","woman","people","person","boy","girl","sir","mr","mrs","miss","you","i","he",
        "she","we","they","me","him","her","us","them",
    },
    "negation_modality": {
        "no","not","never","nothing","none","nobody","cannot","can't","dont","don't","didn't","doesn't","isn't","aren't",
        "wasn't","weren't","won't","wouldn't","couldn't","shouldn't","may","might","must","should","would","could","perhaps",
        "maybe","if","unless","whether","possible","impossible","true","false",
    },
}
ALL_CUES = sorted(set().union(*CUE_SETS.values()))
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:]\d+)*")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_class(src: str) -> str:
    s = str(src)
    if s == "qwen_pair_packed":
        return "inherited_qwen_pair_packed"
    if s == "cleanqwen_fineweb_compact_view_reinvest":
        return "fineweb_source_compact_view_pair"
    if s == "cleanqwen_fineweb_repeat_compact_reinvest":
        return "fineweb_source_repeat_pair"
    if s.startswith("cleanqwen_lengthmatched_compact_reinvest::"):
        return "heldout_official_lengthmatched"
    if s.startswith("neutral_cleanqwen_topup"):
        return "neutral_topup"
    if "::" in s:
        head, tail = s.split("::", 1)
        if tail:
            return tail
    return s


def word_stats(text: str) -> tuple[int, Counter[str]]:
    toks = [m.group(0).lower().strip("'") for m in WORD_RE.finditer(text)]
    c = Counter()
    for w in toks:
        if not w:
            continue
        for cat, lex in CUE_SETS.items():
            if w in lex:
                c[cat] += 1
        if w in ALL_CUES:
            c["any_cue"] += 1
    return len(toks), c


def is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁")


def token_geometry(tok: Tokenizer, text: str, max_len: int = 256) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=False)
    ids = enc.ids
    toks = enc.tokens
    n_raw = len(ids)
    visible = min(n_raw, max_len)
    groups_visible = 0
    group_len = 0
    group_hist = Counter()
    for i, ts in enumerate(toks[:visible]):
        if groups_visible == 0 or is_word_start(ts) or i == 0:
            if group_len:
                group_hist[group_len] += 1
            groups_visible += 1
            group_len = 1
        else:
            group_len += 1
    if group_len:
        group_hist[group_len] += 1
    return {
        "raw_tokens": n_raw,
        "visible_tokens": visible,
        "over256": int(n_raw > max_len),
        "truncated_tokens": max(0, n_raw - max_len),
        "visible_groups": groups_visible,
        "group_hist": group_hist,
    }


def quantiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {k: None for k in ["mean", "median", "p10", "p90", "p99"]}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.quantile(a, 0.5)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
        "p99": float(np.quantile(a, 0.99)),
    }


def profile_file(path: Path, tokenizer: Tokenizer, label: str) -> dict[str, Any]:
    source_rows = Counter()
    source_words_declared = Counter()
    source_words_measured = Counter()
    cue_counts = Counter()
    cue_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    group_hist = Counter()
    n_rows = 0
    declared_words = 0
    measured_words = 0
    raw_tokens = 0
    visible_tokens = 0
    visible_groups = 0
    over256 = 0
    truncated_tokens = 0
    row_word_counts: list[int] = []
    row_raw_tokens: list[int] = []
    row_visible_groups: list[int] = []
    row_tpw: list[float] = []
    row_cue_frac: list[float] = []
    ids_seen = set()
    duplicate_ids = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            words = int(rec.get("words", 0))
            src = source_class(rec.get("source", ""))
            exid = rec.get("example_id")
            if exid in ids_seen:
                duplicate_ids += 1
            ids_seen.add(exid)
            n_rows += 1
            declared_words += words
            source_rows[src] += 1
            source_words_declared[src] += words
            wc, cues = word_stats(text)
            measured_words += wc
            source_words_measured[src] += wc
            cue_counts.update(cues)
            cue_by_source[src].update(cues)
            geom = token_geometry(tokenizer, text)
            raw_tokens += geom["raw_tokens"]
            visible_tokens += geom["visible_tokens"]
            visible_groups += geom["visible_groups"]
            over256 += geom["over256"]
            truncated_tokens += geom["truncated_tokens"]
            group_hist.update(geom["group_hist"])
            row_word_counts.append(words)
            row_raw_tokens.append(geom["raw_tokens"])
            row_visible_groups.append(geom["visible_groups"])
            if words > 0:
                row_tpw.append(geom["raw_tokens"] / words)
                row_cue_frac.append(float(cues.get("any_cue", 0)) / words)
    source_summary = {}
    for src in sorted(source_rows):
        dw = source_words_declared[src]
        mw = source_words_measured[src]
        source_summary[src] = {
            "rows": source_rows[src],
            "declared_words": dw,
            "measured_word_tokens": mw,
            "declared_word_frac": dw / declared_words if declared_words else 0.0,
            "any_cue_words": cue_by_source[src].get("any_cue", 0),
            "any_cue_frac_declared": cue_by_source[src].get("any_cue", 0) / dw if dw else 0.0,
        }
        for cat in sorted(CUE_SETS):
            source_summary[src][f"{cat}_words"] = cue_by_source[src].get(cat, 0)
    cue_summary = {
        cat: {
            "words": cue_counts.get(cat, 0),
            "frac_declared_words": cue_counts.get(cat, 0) / declared_words if declared_words else 0.0,
            "frac_measured_tokens": cue_counts.get(cat, 0) / measured_words if measured_words else 0.0,
        }
        for cat in ["any_cue"] + sorted(CUE_SETS)
    }
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": n_rows,
        "duplicate_example_ids": duplicate_ids,
        "unique_example_ids": len(ids_seen),
        "declared_words": declared_words,
        "measured_word_tokens": measured_words,
        "raw_tokens_step35": raw_tokens,
        "visible_tokens_seq256_step35": visible_tokens,
        "visible_groups_seq256_step35": visible_groups,
        "over256_rows_step35": over256,
        "truncated_tokens_step35": truncated_tokens,
        "raw_tokens_per_declared_word_step35": raw_tokens / declared_words if declared_words else None,
        "visible_tokens_per_declared_word_step35": visible_tokens / declared_words if declared_words else None,
        "visible_groups_per_declared_word_step35": visible_groups / declared_words if declared_words else None,
        "mean_tokens_per_visible_group_step35": visible_tokens / visible_groups if visible_groups else None,
        "row_declared_words": quantiles(row_word_counts),
        "row_raw_tokens_step35": quantiles(row_raw_tokens),
        "row_visible_groups_step35": quantiles(row_visible_groups),
        "row_raw_tokens_per_declared_word_step35": quantiles(row_tpw),
        "row_any_cue_frac": quantiles(row_cue_frac),
        "group_token_hist_visible": {str(k): int(v) for k, v in sorted(group_hist.items())},
        "source_summary": source_summary,
        "cue_summary": cue_summary,
    }


def diff_profile(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    # Return b-a (useful for treatment minus control if a=clean, b=reinvest)
    keys = [
        "rows", "declared_words", "measured_word_tokens", "raw_tokens_step35", "visible_tokens_seq256_step35",
        "visible_groups_seq256_step35", "over256_rows_step35", "truncated_tokens_step35",
        "raw_tokens_per_declared_word_step35", "visible_tokens_per_declared_word_step35",
        "visible_groups_per_declared_word_step35", "mean_tokens_per_visible_group_step35",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[k] = bv - av
    sources = sorted(set(a.get("source_summary", {})) | set(b.get("source_summary", {})))
    out["source_delta"] = {}
    for s in sources:
        ad = a.get("source_summary", {}).get(s, {})
        bd = b.get("source_summary", {}).get(s, {})
        out["source_delta"][s] = {
            "rows": bd.get("rows", 0) - ad.get("rows", 0),
            "declared_words": bd.get("declared_words", 0) - ad.get("declared_words", 0),
            "declared_word_frac": bd.get("declared_word_frac", 0.0) - ad.get("declared_word_frac", 0.0),
            "any_cue_frac_declared": bd.get("any_cue_frac_declared", 0.0) - ad.get("any_cue_frac_declared", 0.0),
        }
    cats = sorted(set(a.get("cue_summary", {})) | set(b.get("cue_summary", {})))
    out["cue_delta"] = {}
    for c in cats:
        ad = a.get("cue_summary", {}).get(c, {})
        bd = b.get("cue_summary", {}).get(c, {})
        out["cue_delta"][c] = {
            "words": bd.get("words", 0) - ad.get("words", 0),
            "frac_declared_words": bd.get("frac_declared_words", 0.0) - ad.get("frac_declared_words", 0.0),
        }
    return out


def write_md(payload: dict[str, Any], path: Path) -> None:
    profiles = payload["profiles"]
    clean = profiles["clean_qwen_10m"]
    reinv = profiles["compact_view_reinvest_10m"]
    repeat = profiles["compact_repeat_reinvest_10m"]
    lm = profiles["lengthmatched_compact_reinvest_10m"]
    diff = payload["treatment_minus_clean"]
    lines = []
    lines.append("# research training-corpus treatment geometry under the research tokenizer\n")
    lines.append("CPU-only substrate comparison. Uses only allowed training corpus files and the fixed research legal tokenizer; no evaluation text, no GPU work, no route selection by itself.\n")
    lines.append("## Integrity\n")
    for k, ok in payload["integrity"].items():
        lines.append(f"- {k}: `{ok}`")
    lines.append("\n## Headline corpus geometry\n")
    lines.append("| corpus | rows | words | raw tok/word | visible tok/word | visible groups/word | over256 rows | truncated tokens | any-cue frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for prof in [clean, lm, repeat, reinv]:
        lines.append(
            f"| {prof['label']} | {prof['rows']} | {prof['declared_words']} | "
            f"{prof['raw_tokens_per_declared_word_step35']:.6f} | {prof['visible_tokens_per_declared_word_step35']:.6f} | "
            f"{prof['visible_groups_per_declared_word_step35']:.6f} | {prof['over256_rows_step35']} | "
            f"{prof['truncated_tokens_step35']} | {prof['cue_summary']['any_cue']['frac_declared_words']:.6f} |"
        )
    lines.append("\n## Compact-view reinvest minus clean-Qwen\n")
    for k in ["rows", "raw_tokens_step35", "visible_tokens_seq256_step35", "visible_groups_seq256_step35", "over256_rows_step35", "truncated_tokens_step35"]:
        lines.append(f"- Δ {k}: `{diff.get(k)}`")
    for k in ["raw_tokens_per_declared_word_step35", "visible_tokens_per_declared_word_step35", "visible_groups_per_declared_word_step35", "mean_tokens_per_visible_group_step35"]:
        lines.append(f"- Δ {k}: `{diff.get(k):.9f}`")
    lines.append("\n## Source-class shift (reinvest minus clean)\n")
    lines.append("| source class | Δ rows | Δ words | clean frac | reinvest frac | Δ frac |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for src, d in sorted(diff["source_delta"].items(), key=lambda kv: abs(kv[1]["declared_words"]), reverse=True):
        csrc = clean["source_summary"].get(src, {})
        rsrc = reinv["source_summary"].get(src, {})
        if d["declared_words"] or csrc.get("declared_words") or rsrc.get("declared_words"):
            lines.append(
                f"| {src} | {d['rows']} | {d['declared_words']} | "
                f"{csrc.get('declared_word_frac',0.0):.6f} | {rsrc.get('declared_word_frac',0.0):.6f} | {d['declared_word_frac']:.6f} |"
            )
    lines.append("\n## Coarse cue-rate shift (reinvest minus clean)\n")
    lines.append("| cue class | clean frac | reinvest frac | Δ frac | Δ words |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, d in sorted(diff["cue_delta"].items(), key=lambda kv: abs(kv[1]["frac_declared_words"]), reverse=True):
        lines.append(
            f"| {cat} | {clean['cue_summary'].get(cat,{}).get('frac_declared_words',0.0):.6f} | "
            f"{reinv['cue_summary'].get(cat,{}).get('frac_declared_words',0.0):.6f} | {d['frac_declared_words']:.6f} | {d['words']} |"
        )
    lines.append("\n## Interpretation\n")
    lines.extend(payload["interpretation"])
    lines.append(f"\nFull JSON: `{payload['files']['json']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    actual_hashes = {
        "clean_10m_sha256": sha256_file(CLEAN_10M),
        "clean_100m_sha256": sha256_file(CLEAN_100M),
        "reinvest_10m_sha256": sha256_file(REINVEST_10M),
        "reinvest_100m_sha256": sha256_file(REINVEST_100M),
        "tokenizer_sha256": sha256_file(TOKENIZER_JSON),
    }
    integrity = {k: actual_hashes.get(k) == v for k, v in EXPECTED.items()}
    meta = json.loads(META.read_text(encoding="utf-8"))
    profiles = {
        "clean_qwen_10m": profile_file(CLEAN_10M, tokenizer, "clean_qwen_10m"),
        "lengthmatched_compact_reinvest_10m": profile_file(LENGTHMATCHED_10M, tokenizer, "lengthmatched_compact_reinvest_10m"),
        "compact_repeat_reinvest_10m": profile_file(REPEAT_10M, tokenizer, "compact_repeat_reinvest_10m"),
        "compact_view_reinvest_10m": profile_file(REINVEST_10M, tokenizer, "compact_view_reinvest_10m"),
    }
    payload = {
        "status": "CORPUS_TREATMENT_GEOMETRY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": None,
        "purpose": "interpret mature legal-tokenizer treatment vector by comparing training-corpus substrate under fixed research tokenizer, without evaluation-text mining or GPU work",
        "inputs": {k: str(v) for k, v in {
            "clean_10m": CLEAN_10M,
            "clean_100m": CLEAN_100M,
            "reinvest_10m": REINVEST_10M,
            "reinvest_100m": REINVEST_100M,
            "repeat_10m": REPEAT_10M,
            "lengthmatched_10m": LENGTHMATCHED_10M,
            "metadata": META,
            "tokenizer_json": TOKENIZER_JSON,
        }.items()},
        "expected_hashes": EXPECTED,
        "actual_hashes": actual_hashes,
        "integrity": integrity,
        "metadata_anchor": {
            "compact_reinvest_pair_rows": meta["families"]["compact_reinvest"]["pair_rows"],
            "compact_reinvest_pair_words": meta["families"]["compact_reinvest"]["pair_words"],
            "common_filler_words": meta["families"]["compact_reinvest"]["common_filler_words"],
            "changed_block_budget_words": meta["families"]["compact_reinvest"]["changed_block_budget_words"],
        },
        "profiles": profiles,
        "treatment_minus_clean": diff_profile(profiles["clean_qwen_10m"], profiles["compact_view_reinvest_10m"]),
        "repeat_minus_clean": diff_profile(profiles["clean_qwen_10m"], profiles["compact_repeat_reinvest_10m"]),
        "view_minus_repeat": diff_profile(profiles["compact_repeat_reinvest_10m"], profiles["compact_view_reinvest_10m"]),
        "interpretation": [
            "The mature clean-vs-reinvest vector, when it lands, should be read as a treatment changing two broad substrate properties: it replaces 423,520 official clean-Qwen words with 423,511 FineWeb source+compact-view pair words plus 9 neutral top-up words, and it changes tokenizer-visible target volume/context geometry under the same research tokenizer.",
            "This script does not turn official weak subtasks into a new objective. It compares only training-corpus composition and token geometry, without further endpoint microscopy.",
            "If reinvest beats clean broadly at 70M/80M, the source-diversity plus compact second-view mechanism survives in legal coordinates and should be combined with representation/optimization evidence from A01 rather than relation-only masking.",
            "If reinvest gains only where the corpus adds broad factual/physical/source diversity but loses in dialogue/syntax columns, the next mechanism may be how compact second views trade lexical/syntactic consolidation for semantic/source breadth, which source-view consistency or representation changes could address.",
            "If reinvest is broadly below clean despite old-tokenizer gains, the old mechanism depended on the inherited representation coordinate and broader hypotheses must be reopened instead of mining benchmark-specific token classes.",
        ],
        "files": {
            "json": str(_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.json')),
            "md": str(_public_path('research/documents/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.md')),
            "source_csv": str(_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/source_shift.csv')),
            "cue_csv": str(_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/cue_shift.csv')),
        },
    }
    payload["elapsed_sec"] = round(time.time() - start, 3)
    (_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Compact CSVs for later quick inspection.
    clean = profiles["clean_qwen_10m"]
    reinv = profiles["compact_view_reinvest_10m"]
    with (_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/source_shift.csv')).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "clean_words", "reinvest_words", "delta_words", "clean_frac", "reinvest_frac", "delta_frac"])
        w.writeheader()
        for src, d in sorted(payload["treatment_minus_clean"]["source_delta"].items()):
            w.writerow({
                "source": src,
                "clean_words": clean["source_summary"].get(src, {}).get("declared_words", 0),
                "reinvest_words": reinv["source_summary"].get(src, {}).get("declared_words", 0),
                "delta_words": d["declared_words"],
                "clean_frac": clean["source_summary"].get(src, {}).get("declared_word_frac", 0.0),
                "reinvest_frac": reinv["source_summary"].get(src, {}).get("declared_word_frac", 0.0),
                "delta_frac": d["declared_word_frac"],
            })
    with (_public_path('experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/cue_shift.csv')).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cue", "clean_frac", "reinvest_frac", "delta_frac", "clean_words", "reinvest_words", "delta_words"])
        w.writeheader()
        for cat, d in sorted(payload["treatment_minus_clean"]["cue_delta"].items()):
            w.writerow({
                "cue": cat,
                "clean_frac": clean["cue_summary"].get(cat, {}).get("frac_declared_words", 0.0),
                "reinvest_frac": reinv["cue_summary"].get(cat, {}).get("frac_declared_words", 0.0),
                "delta_frac": d["frac_declared_words"],
                "clean_words": clean["cue_summary"].get(cat, {}).get("words", 0),
                "reinvest_words": reinv["cue_summary"].get(cat, {}).get("words", 0),
                "delta_words": d["words"],
            })
    write_md(payload, _public_path('research/documents/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.md'))
    print(json.dumps({
        "status": payload["status"],
        "out_json": payload["files"]["json"],
        "out_md": payload["files"]["md"],
        "integrity_all_true": all(integrity.values()),
        "clean_rows_words": [clean["rows"], clean["declared_words"]],
        "reinvest_rows_words": [reinv["rows"], reinv["declared_words"]],
        "delta_raw_tpw": payload["treatment_minus_clean"]["raw_tokens_per_declared_word_step35"],
        "delta_visible_tpw": payload["treatment_minus_clean"]["visible_tokens_per_declared_word_step35"],
        "delta_over256": payload["treatment_minus_clean"]["over256_rows_step35"],
        "delta_truncated_tokens": payload["treatment_minus_clean"]["truncated_tokens_step35"],
        "largest_source_delta": max(payload["treatment_minus_clean"]["source_delta"].items(), key=lambda kv: abs(kv[1]["declared_words"])),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
