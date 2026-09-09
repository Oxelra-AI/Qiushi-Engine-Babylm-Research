#!/usr/bin/env python3
"""Fast research relation-lexeme exposure audit.

Single-tokenization pass over each 10M corpus; keeps the scientific purpose of the
slower script but avoids per-lexeme regex scans over every row.
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
PAIR_TOKEN_SETS = {
    "above_below": [{"above"}, {"below"}],
    "left_right": [{"left"}, {"right"}],
    "front_behind": [{"front"}, {"behind"}],
    "boss_subordinate": [{"boss"}, {"subordinate"}],
    "parent_child": [{"parent", "parents"}, {"child", "children"}],
    "teacher_student": [{"teacher", "teachers"}, {"student", "students"}],
    "landlord_tenant": [{"landlord", "landlords"}, {"tenant", "tenants"}],
    "kick_drop_touch": [{"kick", "kicks", "kicked", "kicking"}, {"drop", "drops", "dropped", "dropping"}, {"touch", "touches", "touched", "touching"}],
    "throw_catch": [{"throw", "throws", "throwing", "threw", "thrown"}, {"catch", "catches", "catching", "caught"}],
    "push_pull": [{"push", "pushes", "pushed", "pushing"}, {"pull", "pulls", "pulled", "pulling"}],
    "sink_float": [{"sink", "sinks", "sinking", "sank", "sunk"}, {"float", "floats", "floated", "floating"}],
    "rise_fall": [{"rise", "rises", "rising", "rose", "risen"}, {"fall", "falls", "falling", "fell", "fallen"}],
    "grow_shrink": [{"grow", "grows", "growing", "grew", "grown"}, {"shrink", "shrinks", "shrinking", "shrank", "shrunk"}],
}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audit_corpus(name: str, path: Path) -> dict[str, Any]:
    token_counts = collections.Counter()
    pair_row_counts = collections.Counter()
    source_counts = collections.Counter()
    rows = 0
    words = 0
    for rec in iter_jsonl(path):
        txt = rec.get("text", "")
        toks = TOKEN_RE.findall(txt.lower())
        tset = set(toks)
        rows += 1
        words += int(rec.get("words") or len(txt.split()))
        source_counts[str(rec.get("source", ""))] += 1
        token_counts.update(toks)
        for pname, groups in PAIR_TOKEN_SETS.items():
            if all(tset.intersection(g) for g in groups):
                pair_row_counts[pname] += 1
    lex_counts = {lex: int(token_counts.get(lex, 0)) for lex in LEXEMES}
    return {
        "name": name,
        "path": str(path),
        "rows": rows,
        "words": words,
        "lexeme_counts": lex_counts,
        "pair_row_counts": dict(sorted(pair_row_counts.items())),
        "source_counts_top": source_counts.most_common(12),
    }


def audit_changed_meta(name: str, path: Path) -> dict[str, Any]:
    rows = words = pair_ids = 0
    comp = collections.Counter()
    for rec in iter_jsonl(path):
        rows += 1
        words += int(rec.get("words", 0))
        pair_ids += len(rec.get("pair_ids", []))
        for k, v in (rec.get("component_sources") or {}).items():
            comp[k] += int(v)
    return {"name": name, "path": str(path), "rows": rows, "words": words, "pair_ids": pair_ids, "component_word_counts": dict(comp.most_common())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in list(CORPORA.values()) + list(CHANGED_META.values()) if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    corp = {name: audit_corpus(name, path) for name, path in CORPORA.items()}
    meta = {name: audit_changed_meta(name, path) for name, path in CHANGED_META.items()}
    clean = corp["clean_qwen_10M"]
    lex_comp: dict[str, Any] = {}
    for lex in LEXEMES:
        row = {name: rec["lexeme_counts"].get(lex, 0) for name, rec in corp.items()}
        base = row["clean_qwen_10M"]
        row["compact_view_minus_clean"] = row["compact_view_reinvest_10M"] - base
        row["compact_view_vs_clean_ratio"] = row["compact_view_reinvest_10M"] / base if base else None
        lex_comp[lex] = row
    pair_comp: dict[str, Any] = {}
    for pname in PAIR_TOKEN_SETS:
        row = {name: rec["pair_row_counts"].get(pname, 0) for name, rec in corp.items()}
        base = row["clean_qwen_10M"]
        row["compact_view_minus_clean"] = row["compact_view_reinvest_10M"] - base
        row["compact_view_vs_clean_ratio"] = row["compact_view_reinvest_10M"] / base if base else None
        pair_comp[pname] = row
    payload = {
        "status": "RELATION_LEXEME_CORPUS_AUDIT_FAST",
        "method": "Single tokenization pass; pair counts are row-level co-occurrences of morphology sets, not syntactic relation examples.",
        "corpora": corp,
        "changed_block_meta": meta,
        "lexeme_comparison": lex_comp,
        "pair_row_comparison": pair_comp,
        "interpretation_limits": [
            "Surface lexical exposure does not establish EWoK relation learning.",
            "Many counts come from noisy natural contexts and not controlled relation contrasts.",
            "Useful mainly to rule out gross absence and compare corpus composition while corrected-tokenizer retraining continues.",
        ],
    }
    out_json = OUT_DIR / "relation_lexeme_corpus_audit_fast.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "relation_lexeme_counts_fast.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["lexeme"] + list(CORPORA) + ["compact_view_minus_clean", "compact_view_vs_clean_ratio"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for lex, row in lex_comp.items(): w.writerow({"lexeme": lex, **row})
    with (OUT_DIR / "relation_pair_row_counts_fast.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["pair_pattern"] + list(CORPORA) + ["compact_view_minus_clean", "compact_view_vs_clean_ratio"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for pname, row in pair_comp.items(): w.writerow({"pair_pattern": pname, **row})
    interesting_pairs = {k: pair_comp[k] for k in ["above_below", "boss_subordinate", "parent_child", "teacher_student", "kick_drop_touch", "sink_float", "rise_fall", "grow_shrink"]}
    interesting_lex = {k: lex_comp[k] for k in ["above", "below", "boss", "subordinate", "parent", "child", "teacher", "student", "kick", "drop", "touch", "break", "stir", "sink", "float"]}
    note = A01 / "notes/relation_lexeme_corpus_audit.md"
    lines = [
        "# research — relation lexeme corpus audit",
        "",
        f"JSON: `{out_json}`",
        "",
        "This is a surface text-composition audit for relation words highlighted by the official-compatible EWoK margin analysis. It is not a competence result.",
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
    lines.append("The compact-view corpus is not grossly devoid of the main relation words, but surface counts are not a relation-learning mechanism. Compare these counts with official-margin behavior only as a clue: relation failures with many ordinary occurrences point toward relation-direction/composition and seed-dependent representation dynamics, not simple word absence.")
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
