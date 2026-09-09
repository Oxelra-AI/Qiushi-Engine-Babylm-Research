#!/usr/bin/env python3
"""research: localize <unk> events in actual scored text fields.

The broad audit scanned many candidate text fields. This script narrows to files and
fields used by the official-compatible scorer: BLiMP/Supplement sentence fields,
EWoK candidate sentence fields, Entity/COMPS/GlobalPIQA text-like fields, Reading/AoA
sentence/context fields, and SuperGLUE fine-tuning text fields. It reports whether
unknowns live in actual scored strings rather than metadata.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

USER_ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
EVAL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
OUT = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_scored_text_unk_localization')
OUT.mkdir(parents=True, exist_ok=True)
NEW_TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')
OLD_TOK = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json')


def load_tok(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    # Disable padding/truncation in case the saved JSON contains active settings.
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def iter_strs(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, str):
                yield p, v
            else:
                yield from iter_strs(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_strs(v, f"{prefix}[{i}]")


def likely_scored_field(family: str, field: str) -> bool:
    fl = field.lower()
    tail = fl.split(".")[-1]
    if family in {"BLiMP", "Supplement"}:
        return any(x in tail for x in ["sentence_good", "sentence_bad", "good_sentence", "bad_sentence", "sentence", "one_prefix_prefix", "two_prefix_prefix"])
    if family == "EWoK":
        return any(x in tail for x in ["sentence", "context", "target", "candidate", "correct", "incorrect", "option", "choice"])
    if family == "Entity":
        return any(x in tail for x in ["sentence", "context", "query", "answer", "text"])
    if family == "COMPS":
        return any(x in tail for x in ["sentence", "context", "candidate", "option", "choice", "text"])
    if family == "GlobalPIQA":
        return any(x in tail for x in ["goal", "sol1", "sol2", "sentence", "context", "option", "choice", "text"])
    if family == "Reading_AoA":
        return any(x in tail for x in ["sentence", "context", "text", "utterance", "transcript", "word"])
    if family == "SuperGLUE":
        return any(x in tail for x in ["passage", "question", "answer", "premise", "hypothesis", "sentence", "sentence1", "sentence2", "paragraph", "text", "span", "query"])
    return any(x in tail for x in ["sentence", "text", "context", "question", "answer", "premise", "hypothesis"])


def family_for(path: pathlib.Path) -> str:
    s = str(path).lower()
    if "supplement_filtered" in s:
        return "Supplement"
    if "blimp_filtered" in s:
        return "BLiMP"
    if "ewok_filtered" in s:
        return "EWoK"
    if "entity_tracking" in s:
        return "Entity"
    if "/comps" in s or "comps" in path.name.lower():
        return "COMPS"
    if "globalpiqa" in s or "piqa" in s:
        return "GlobalPIQA"
    if "/aoa/" in s or "/reading/" in s:
        return "Reading_AoA"
    if "glue_filtered" in s:
        return "SuperGLUE"
    return "Other"


def iter_eval_records() -> Iterator[Tuple[str, str, str, str]]:
    # JSONL/JSON in exact official root. For CSV/TSV, use all text-like columns.
    for path in sorted(EVAL.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv"}:
            continue
        fam = family_for(path)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    for field, text in iter_strs(obj):
                        if text and likely_scored_field(fam, field):
                            yield fam, str(path), field, text
        elif path.suffix.lower() == ".json":
            try:
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            for field, text in iter_strs(obj):
                if text and likely_scored_field(fam, field):
                    yield fam, str(path), field, text
        else:
            delim = "\t" if path.suffix.lower() == ".tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter=delim)
                for row in reader:
                    for field, text in row.items():
                        if text and likely_scored_field(fam, field):
                            yield fam, str(path), field, text


def summarize(tok, unk_id):
    by_family = defaultdict(lambda: {"strings":0,"chars":0,"tokens":0,"unk_tokens":0,"strings_with_unk":0})
    by_path_field = defaultdict(lambda: {"strings":0,"chars":0,"tokens":0,"unk_tokens":0,"strings_with_unk":0})
    examples = []
    chars = Counter()
    for fam, path, field, text in iter_eval_records():
        enc = tok.encode(text)
        ids = enc.ids
        unk_pos = [] if unk_id is None else [i for i, x in enumerate(ids) if x == unk_id]
        for d in (by_family[fam], by_path_field[(path, field)]):
            d["strings"] += 1
            d["chars"] += len(text)
            d["tokens"] += len(ids)
            d["unk_tokens"] += len(unk_pos)
            d["strings_with_unk"] += bool(unk_pos)
        if unk_pos:
            for ch in text:
                if ord(ch) > 126 or ord(ch) < 32:
                    chars[ch] += 1
            if len(examples) < 100:
                examples.append({
                    "family": fam, "path": path, "field": field, "text_prefix": text[:500],
                    "n_tokens": len(ids), "n_unk": len(unk_pos), "unk_positions": unk_pos[:20],
                    "tokens_around_first_unk": enc.tokens[max(0, unk_pos[0]-5):unk_pos[0]+6],
                })
    def finish(d):
        o = dict(d)
        o["unk_rate"] = o["unk_tokens"]/o["tokens"] if o.get("tokens") else 0.0
        o["string_unk_rate"] = o["strings_with_unk"]/o["strings"] if o.get("strings") else 0.0
        return o
    total = {"strings":0,"chars":0,"tokens":0,"unk_tokens":0,"strings_with_unk":0}
    for d in by_family.values():
        for k in total:
            total[k] += d.get(k, 0)
    path_rows = []
    for (p, f), d in by_path_field.items():
        if d["unk_tokens"] or d["strings_with_unk"]:
            o = finish(d); o.update({"path": p, "field": f}); path_rows.append(o)
    path_rows.sort(key=lambda x: (-x["unk_tokens"], x["path"], x["field"]))
    return {
        "total": finish(total),
        "by_family": {k: finish(v) for k, v in sorted(by_family.items())},
        "by_path_field_with_unk": path_rows[:200],
        "top_nonascii_chars_in_unk_strings": [(repr(k), v) for k, v in chars.most_common(100)],
        "examples": examples,
    }


def main():
    t0 = time.time()
    old = load_tok(OLD_TOK)
    new = load_tok(NEW_TOK)
    payload = {
        "status": "SCORED_TEXT_UNK_LOCALIZATION",
        "purpose": "Localize <unk> events to likely official-scored text fields, with padding/truncation disabled in the low-level tokenizer objects.",
        "eval_root": str(EVAL),
        "tokenizers": {"old": str(OLD_TOK), "new": str(NEW_TOK)},
        "old": summarize(old, old.token_to_id("<unk>")),
        "new": summarize(new, new.token_to_id("<unk>")),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_scored_text_unk_localization/scored_text_unk_localization.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = []
    md.append("# research scored-text `<unk>` localization\n\n")
    md.append("This narrows the broad audit to likely scored text fields in the exact current-coordinate evaluation root, with low-level tokenizer padding/truncation disabled.\n\n")
    for label in ["old", "new"]:
        tot = payload[label]["total"]
        md.append(f"## {label} tokenizer total\n")
        md.append(f"strings={tot['strings']}, tokens={tot['tokens']}, unk_tokens={tot['unk_tokens']}, strings_with_unk={tot['strings_with_unk']}, unk_rate={tot['unk_rate']:.8g}, string_unk_rate={tot['string_unk_rate']:.8g}\n\n")
        md.append(f"### {label} by family\n")
        for fam, d in payload[label]["by_family"].items():
            md.append(f"- {fam}: strings={d['strings']}, tokens={d['tokens']}, unk_tokens={d['unk_tokens']}, strings_with_unk={d['strings_with_unk']}, unk_rate={d['unk_rate']:.8g}\n")
        md.append("\n")
    md.append("## New-tokenizer path/field unknown hotspots\n")
    for row in payload["new"]["by_path_field_with_unk"][:40]:
        md.append(f"- {row['unk_tokens']} unk / {row['strings']} strings, family path `{row['path']}`, field `{row['field']}`\n")
    md.append("\n## New-tokenizer example contexts\n")
    for ex in payload["new"]["examples"][:20]:
        clean = ex["text_prefix"].replace("\n", " ")[:240]
        md.append(f"- {ex['family']} field `{ex['field']}` n_unk={ex['n_unk']} tokens={ex['tokens_around_first_unk']} text=`{clean}`\n")
    md.append("\n## Interpretation\n")
    new_tot = payload["new"]["total"]
    supp = payload["new"]["by_family"].get("Supplement", {})
    blimp = payload["new"]["by_family"].get("BLiMP", {})
    ewok = payload["new"]["by_family"].get("EWoK", {})
    if new_tot["unk_tokens"] == 0:
        md.append("No `<unk>` events appeared in likely scored strings; the broad-audit warning was from metadata or unscored fields.\n")
    else:
        md.append(f"The compliant tokenizer produces `<unk>` in likely scored strings: total {new_tot['unk_tokens']}, Supplement {supp.get('unk_tokens',0)}, BLiMP {blimp.get('unk_tokens',0)}, EWoK {ewok.get('unk_tokens',0)}. This is a real tokenizer-construction vulnerability if confirmed by the official scorer's actual serialization.\n")
    md.append(f"\nFull JSON: `{out_json}`\n")
    out_md = _public_path('research/documents/frontier_consolidation/data/tokenizer_scored_text_unk_localization/scored_text_unk_localization.md')
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "out_json": str(out_json), "out_md": str(out_md),
        "old_unk_tokens": payload["old"]["total"]["unk_tokens"],
        "new_unk_tokens": payload["new"]["total"]["unk_tokens"],
        "new_by_family": {k: v["unk_tokens"] for k, v in payload["new"]["by_family"].items()},
        "elapsed_sec": payload["elapsed_sec"],
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
