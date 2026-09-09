#!/usr/bin/env python3
"""research: relation-lexeme exposure audit for clean-Qwen vs compact-view corpora.

This is a cheap text-composition check only.  It does not infer EWoK competence
from word counts.  It asks whether the relation primitives highlighted by the
official-compatible margin audit were grossly absent/present in the 10M corpora
or the compact changed block, which helps separate obvious exposure explanations
from learning-dynamics/representation explanations.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import re
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01 = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = A01 / "data/relation_lexeme_corpus_audit"

CORPORA = {
    "clean_qwen_10M": ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
    "compact_view_reinvest_10M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "compact_repeat_reinvest_10M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl",
    "lengthmatched_compact_reinvest_10M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_lengthmatched_compact_reinvest_10M.jsonl",
}
CHANGED_META = {
    "compact_view_reinvest_changed_block": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
    "compact_repeat_reinvest_changed_block": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_changed_block_rows_meta.jsonl",
    "lengthmatched_compact_reinvest_changed_block": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_lengthmatched_compact_reinvest_changed_block_rows_meta.jsonl",
}

LEXEMES = [
    "above", "below", "left", "right", "front", "behind", "north", "south", "east", "west", "close", "far",
    "boss", "subordinate", "parent", "child", "teacher", "student", "tenant", "landlord", "friend", "enemy", "sibling", "colleague",
    "kick", "drop", "touch", "throw", "catch", "break", "fix", "push", "pull", "lift", "collide", "attract", "repel", "heat", "cool",
    "accelerate", "slow", "sink", "float", "roll", "slide", "fall", "rise", "grow", "shrink",
    "stir", "flap", "drip", "ripple", "drape", "pour", "rip", "squeeze", "fold", "wrinkle", "splash", "tap", "hang",
]
PAIR_PATTERNS = {
    "above_below": [r"\babove\b", r"\bbelow\b"],
    "left_right": [r"\bleft\b", r"\bright\b"],
    "front_behind": [r"\bfront\b", r"\bbehind\b"],
    "boss_subordinate": [r"\bboss\b", r"\bsubordinate\b"],
    "parent_child": [r"\bparent\b", r"\bchild\b"],
    "teacher_student": [r"\bteacher\b", r"\bstudent\b"],
    "landlord_tenant": [r"\blandlord\b", r"\btenant\b"],
    "kick_drop_touch": [r"\bkick(?:s|ed|ing)?\b", r"\bdrop(?:s|ped|ping)?\b", r"\btouch(?:es|ed|ing)?\b"],
    "throw_catch": [r"\bthrow(?:s|ing)?\b|\bthrew\b|\bthrown\b", r"\bcatch(?:es|ing)?\b|\bcaught\b"],
    "push_pull": [r"\bpush(?:es|ed|ing)?\b", r"\bpull(?:s|ed|ing)?\b"],
    "sink_float": [r"\bsink(?:s|ing)?\b|\bsank\b|\bsunk\b", r"\bfloat(?:s|ed|ing)?\b"],
    "rise_fall": [r"\brise(?:s|n)?\b|\brose\b|\brising\b", r"\bfall(?:s|en|ing)?\b|\bfell\b"],
    "grow_shrink": [r"\bgrow(?:s|n|ing)?\b|\bgrew\b", r"\bshrink(?:s|ing)?\b|\bshrank\b|\bshrunk\b"],
}

WORD_RE_CACHE = {lex: re.compile(r"(?<![A-Za-z])" + re.escape(lex) + r"(?![A-Za-z])", re.I) for lex in LEXEMES}
PAIR_RE_CACHE = {name: [re.compile(p, re.I) for p in pats] for name, pats in PAIR_PATTERNS.items()}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audit_corpus(name: str, path: Path) -> dict[str, Any]:
    lex_counts = collections.Counter()
    pair_row_counts = collections.Counter()
    pair_token_counts = collections.Counter()
    source_counts = collections.Counter()
    rows = 0
    words = 0
    for rec in iter_jsonl(path):
        txt = rec.get("text", "")
        low = txt.lower()
        rows += 1
        words += int(rec.get("words") or len(txt.split()))
        source_counts[str(rec.get("source", ""))] += 1
        for lex, cre in WORD_RE_CACHE.items():
            n = len(cre.findall(low))
            if n:
                lex_counts[lex] += n
        for pname, regs in PAIR_RE_CACHE.items():
            hits = [len(r.findall(low)) for r in regs]
            if all(h > 0 for h in hits):
                pair_row_counts[pname] += 1
                pair_token_counts[pname] += min(hits)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "rows": rows,
        "words": words,
        "lexeme_counts": dict(sorted(lex_counts.items())),
        "pair_row_counts": dict(sorted(pair_row_counts.items())),
        "pair_token_min_counts": dict(sorted(pair_token_counts.items())),
        "source_counts_top": source_counts.most_common(12),
    }


def audit_changed_meta(name: str, path: Path) -> dict[str, Any]:
    rows = 0
    words = 0
    component_word_counts = collections.Counter()
    pair_ids = 0
    for rec in iter_jsonl(path):
        rows += 1
        words += int(rec.get("words", 0))
        pair_ids += len(rec.get("pair_ids", []))
        for k, v in (rec.get("component_sources") or {}).items():
            component_word_counts[k] += int(v)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "rows": rows,
        "words": words,
        "pair_ids": pair_ids,
        "component_word_counts": dict(component_word_counts.most_common()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in list(CORPORA.values()) + list(CHANGED_META.values()) if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    corp = {name: audit_corpus(name, path) for name, path in CORPORA.items()}
    meta = {name: audit_changed_meta(name, path) for name, path in CHANGED_META.items()}

    # Compare to clean counts as ratios per 10M words.
    clean = corp["clean_qwen_10M"]
    lexeme_comparison: dict[str, Any] = {}
    for lex in LEXEMES:
        base = clean["lexeme_counts"].get(lex, 0)
        lexeme_comparison[lex] = {name: rec["lexeme_counts"].get(lex, 0) for name, rec in corp.items()}
        lexeme_comparison[lex]["compact_view_minus_clean"] = lexeme_comparison[lex]["compact_view_reinvest_10M"] - base
        lexeme_comparison[lex]["compact_view_vs_clean_ratio"] = (lexeme_comparison[lex]["compact_view_reinvest_10M"] / base) if base else None
    pair_comparison: dict[str, Any] = {}
    for pname in PAIR_PATTERNS:
        base = clean["pair_row_counts"].get(pname, 0)
        pair_comparison[pname] = {name: rec["pair_row_counts"].get(pname, 0) for name, rec in corp.items()}
        pair_comparison[pname]["compact_view_minus_clean"] = pair_comparison[pname]["compact_view_reinvest_10M"] - base
        pair_comparison[pname]["compact_view_vs_clean_ratio"] = (pair_comparison[pname]["compact_view_reinvest_10M"] / base) if base else None

    payload = {
        "status": "RELATION_LEXEME_CORPUS_AUDIT",
        "corpora": corp,
        "changed_block_meta": meta,
        "lexeme_comparison": lexeme_comparison,
        "pair_row_comparison": pair_comparison,
        "interpretation_limits": [
            "Counts are surface lexical exposure, not evidence that EWoK relation operations were learned.",
            "Many EWoK items use artificial templatic relations; natural corpus occurrences are much noisier than benchmark primitives.",
            "Use this only to identify gross absence/exposure differences while corrected-tokenizer runs proceed.",
        ],
    }
    out_json = OUT_DIR / "relation_lexeme_corpus_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # compact CSVs
    with (OUT_DIR / "relation_lexeme_counts.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["lexeme"] + list(CORPORA) + ["compact_view_minus_clean", "compact_view_vs_clean_ratio"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lex, row in lexeme_comparison.items():
            w.writerow({"lexeme": lex, **row})
    with (OUT_DIR / "relation_pair_row_counts.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["pair_pattern"] + list(CORPORA) + ["compact_view_minus_clean", "compact_view_vs_clean_ratio"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pname, row in pair_comparison.items():
            w.writerow({"pair_pattern": pname, **row})

    interesting_pairs = {k: pair_comparison[k] for k in ["above_below", "boss_subordinate", "parent_child", "teacher_student", "kick_drop_touch", "sink_float", "rise_fall", "grow_shrink"]}
    interesting_lex = {k: lexeme_comparison[k] for k in ["above", "below", "boss", "subordinate", "parent", "child", "teacher", "student", "kick", "drop", "touch", "break", "stir", "sink", "float"]}
    note = A01 / "notes/relation_lexeme_corpus_audit.md"
    lines = [
        "# research — relation lexeme corpus audit",
        "",
        f"JSON: `{out_json}`",
        "",
        "This is a surface text-composition audit for the relation words highlighted by the official-compatible EWoK margin analysis. It is not a competence result.",
        "",
        "## Corpus accounting",
    ]
    for name, rec in corp.items():
        lines.append(f"- {name}: rows={rec['rows']}, words={rec['words']}, top sources={rec['source_counts_top'][:5]}")
    lines.extend(["", "## Selected pair-pattern row counts"])
    for pname, row in interesting_pairs.items():
        lines.append(f"- {pname}: clean={row['clean_qwen_10M']}, compact_view={row['compact_view_reinvest_10M']}, compact_repeat={row['compact_repeat_reinvest_10M']}, lengthmatched={row['lengthmatched_compact_reinvest_10M']}, delta_view-clean={row['compact_view_minus_clean']}")
    lines.extend(["", "## Selected lexeme counts"])
    for lex, row in interesting_lex.items():
        lines.append(f"- {lex}: clean={row['clean_qwen_10M']}, compact_view={row['compact_view_reinvest_10M']}, compact_repeat={row['compact_repeat_reinvest_10M']}, lengthmatched={row['lengthmatched_compact_reinvest_10M']}, delta_view-clean={row['compact_view_minus_clean']}")
    lines.extend(["", "## Reading"])
    lines.append("The audit should only rule out or expose gross lexical exposure shifts. If relation margin failures persist despite many ordinary word occurrences, the likely bottleneck is learning relational direction/composition under templatic contrasts rather than simple absence of relation words.")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "note": str(note),
        "interesting_pairs": interesting_pairs,
        "interesting_lexemes": interesting_lex,
        "changed_block_words": {name: rec["words"] for name, rec in meta.items()},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
