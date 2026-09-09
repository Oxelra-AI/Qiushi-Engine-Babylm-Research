#!/usr/bin/env python3
"""research: audit byte-level alphabet coverage and <unk> behavior of the compliant tokenizer.

Scientific purpose
------------------
The compliant tokenizer was trained from the allowed 10M reinvest pool. The research
script copied the byte-level pre-tokenizer from the old tokenizer but did not pass
ByteLevel.alphabet() explicitly to BpeTrainer. This audit checks whether that choice
left unseen byte-level characters uncovered, causing <unk> tokens on official eval
text or the allowed training pool. If <unk> appears on score-bearing text, the
current compliant retrain would be scientifically suspect before any model score is
interpreted.

This is CPU-only tokenization/counting. It does not inspect running retrain dirs and
uses evaluation text only for auditing tokenizer coverage, not for selecting or
changing tokenizer vocabulary.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

USER_ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_byte_unk_audit')
OUT.mkdir(parents=True, exist_ok=True)

OLD_TOK = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json')
NEW_TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')
POOL_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
JSON = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.json')

# Candidate evaluation data roots used by the already-patched harness. The script
# searches JSON/JSONL/CSV/TSV/TXT fields recursively and tags examples by path.
EVAL_ROOTS = [
    # Exact current-coordinate root used by evaluate_compliant_endpoint.py.
    _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval'),
    _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data'),
    # Historical/local candidate roots retained for compatibility if the tree moves.
    _public_path('experiments/archive/frontier_consolidation/training/babylm_eval_current/babylm_eval/evaluation_data'),
    _public_path('experiments/archive/frontier_consolidation/training/babylm_eval_current/evaluation_data'),
    _public_path('experiments/archive/frontier_consolidation/training/babylm_eval_2026/babylm_eval/evaluation_data'),
    _public_path('experiments/archive/frontier_consolidation/training/babylm_eval_2026/evaluation_data'),
    _public_path('experiments/archive/representation_and_objectives/training/babylm_eval_current/babylm_eval/evaluation_data'),
    _public_path('experiments/archive/representation_and_objectives/training/babylm_eval_current/evaluation_data'),
]

TEXT_KEYS = {
    "text", "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "answer", "option", "option1", "option2", "premise", "hypothesis",
    "sentence1", "sentence2", "passage", "query", "word", "target", "target_word", "candidate",
    "choice", "label", "ending0", "ending1", "ending2", "ending3", "startphrase",
}

CONTROL_CHARS = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(path: pathlib.Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def byte_alphabet() -> List[str]:
    from tokenizers.pre_tokenizers import ByteLevel

    return list(ByteLevel.alphabet())


def vocab_set(path: pathlib.Path) -> set:
    tok = load_tokenizer(path)
    return set(tok.get_vocab().keys())


def pool_texts(limit: int | None = None) -> Iterator[Tuple[str, str]]:
    n = 0
    with open(POOL_10M, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                yield ("train_pool", text)
                n += 1
                if limit is not None and n >= limit:
                    return


def family_from_path(path: pathlib.Path) -> str:
    s = "/".join(path.parts).lower()
    name = path.name.lower()
    if "blimp_supp" in s or "supplement" in s:
        return "Supplement"
    if "blimp" in s:
        return "BLiMP"
    if "ewok" in s:
        return "EWoK"
    if "entity" in s:
        return "Entity"
    if "comps" in s:
        return "COMPS"
    if "globalpiqa" in s or "piqa" in s:
        return "GlobalPIQA"
    if "reading" in s or "aoa" in s or "cdi" in s:
        return "Reading_AoA"
    if "glue" in s or "superglue" in s or name in {"train.jsonl", "validation.jsonl", "test.jsonl"}:
        return "SuperGLUE_or_GLUE"
    return "OtherEval"


def iter_string_values(obj) -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                lk = str(k).lower()
                if lk in TEXT_KEYS or any(t in lk for t in ("sentence", "text", "context", "question", "answer", "premise", "hypothesis", "passage", "word")):
                    yield v
            else:
                yield from iter_string_values(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_string_values(x)


def eval_texts() -> Iterator[Tuple[str, str, str]]:
    roots = [p for p in EVAL_ROOTS if p.exists()]
    seen_files = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv", ".txt"}:
                continue
            rp = str(path.resolve())
            if rp in seen_files:
                continue
            seen_files.add(rp)
            family = family_from_path(path)
            try:
                if path.suffix.lower() == ".jsonl":
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            for text in iter_string_values(obj):
                                if text:
                                    yield family, str(path), text
                elif path.suffix.lower() == ".json":
                    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    for text in iter_string_values(obj):
                        if text:
                            yield family, str(path), text
                elif path.suffix.lower() in {".csv", ".tsv"}:
                    delim = "\t" if path.suffix.lower() == ".tsv" else ","
                    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                        reader = csv.DictReader(f, delimiter=delim)
                        for row in reader:
                            for k, v in row.items():
                                if not isinstance(v, str) or not v:
                                    continue
                                lk = str(k).lower()
                                if lk in TEXT_KEYS or any(t in lk for t in ("sentence", "text", "context", "question", "answer", "premise", "hypothesis", "passage", "word")):
                                    yield family, str(path), v
                else:
                    # TXT is included only for short line-based files; long raw text files
                    # are not expected in official evaluation, but line sampling is safe.
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            t = line.strip()
                            if t:
                                yield family, str(path), t
            except Exception as e:
                print(json.dumps({"warning": "eval_file_skipped", "path": str(path), "error": repr(e)}), flush=True)


def char_label(ch: str) -> str:
    if ch in CONTROL_CHARS:
        return CONTROL_CHARS[ch]
    if ch == " ":
        return "<space>"
    if ch == "\u00a0":
        return "<nbsp>"
    return ch


def audit_stream(tok, unk_id: int | None, stream: Iterable[Tuple[str, str]], max_examples: int = 30) -> Dict:
    stats = {
        "rows": 0,
        "chars": 0,
        "tokens": 0,
        "unk_tokens": 0,
        "rows_with_unk": 0,
        "unk_rate": 0.0,
        "row_unk_rate": 0.0,
        "top_unk_surface_chars": [],
        "examples": [],
    }
    unk_char_counter: Counter[str] = Counter()
    for fam, text in stream:
        stats["rows"] += 1
        stats["chars"] += len(text)
        enc = tok.encode(text)
        ids = enc.ids
        stats["tokens"] += len(ids)
        unk_positions = [] if unk_id is None else [i for i, x in enumerate(ids) if x == unk_id]
        if unk_positions:
            stats["unk_tokens"] += len(unk_positions)
            stats["rows_with_unk"] += 1
            # Capture a rough surface proxy by looking for non-ASCII/unusual chars in the text.
            for ch in text:
                if ord(ch) > 126 or ord(ch) < 32:
                    unk_char_counter[char_label(ch)] += 1
            if len(stats["examples"]) < max_examples:
                stats["examples"].append({
                    "family": fam,
                    "text_prefix": text[:500],
                    "n_tokens": len(ids),
                    "n_unk": len(unk_positions),
                    "unk_positions": unk_positions[:20],
                    "tokens_around_first_unk": enc.tokens[max(0, unk_positions[0]-5):unk_positions[0]+6],
                    "ids_around_first_unk": ids[max(0, unk_positions[0]-5):unk_positions[0]+6],
                    "non_ascii_chars": sorted({char_label(ch) for ch in text if ord(ch) > 126 or ord(ch) < 32})[:50],
                })
    if stats["tokens"]:
        stats["unk_rate"] = stats["unk_tokens"] / stats["tokens"]
    if stats["rows"]:
        stats["row_unk_rate"] = stats["rows_with_unk"] / stats["rows"]
    stats["top_unk_surface_chars"] = unk_char_counter.most_common(50)
    return stats


def audit_eval(tok, unk_id: int | None, max_examples: int = 50) -> Dict:
    by_family = defaultdict(lambda: {"rows": 0, "chars": 0, "tokens": 0, "unk_tokens": 0, "rows_with_unk": 0})
    by_file = defaultdict(lambda: {"rows": 0, "chars": 0, "tokens": 0, "unk_tokens": 0, "rows_with_unk": 0})
    examples = []
    char_counter: Counter[str] = Counter()
    scanned_any = False
    for fam, path, text in eval_texts():
        scanned_any = True
        enc = tok.encode(text)
        ids = enc.ids
        unk_positions = [] if unk_id is None else [i for i, x in enumerate(ids) if x == unk_id]
        for bucket in (by_family[fam], by_file[path]):
            bucket["rows"] += 1
            bucket["chars"] += len(text)
            bucket["tokens"] += len(ids)
            bucket["unk_tokens"] += len(unk_positions)
            bucket["rows_with_unk"] += 1 if unk_positions else 0
        if unk_positions:
            for ch in text:
                if ord(ch) > 126 or ord(ch) < 32:
                    char_counter[char_label(ch)] += 1
            if len(examples) < max_examples:
                examples.append({
                    "family": fam,
                    "path": path,
                    "text_prefix": text[:500],
                    "n_tokens": len(ids),
                    "n_unk": len(unk_positions),
                    "unk_positions": unk_positions[:20],
                    "tokens_around_first_unk": enc.tokens[max(0, unk_positions[0]-5):unk_positions[0]+6],
                    "ids_around_first_unk": ids[max(0, unk_positions[0]-5):unk_positions[0]+6],
                    "non_ascii_chars": sorted({char_label(ch) for ch in text if ord(ch) > 126 or ord(ch) < 32})[:50],
                })
    def finish(d):
        out = dict(d)
        out["unk_rate"] = (out["unk_tokens"] / out["tokens"]) if out["tokens"] else 0.0
        out["row_unk_rate"] = (out["rows_with_unk"] / out["rows"]) if out["rows"] else 0.0
        return out
    fam_out = {k: finish(v) for k, v in sorted(by_family.items())}
    file_out = {k: finish(v) for k, v in sorted(by_file.items()) if v["unk_tokens"] or v["rows_with_unk"]}
    total = {"rows": 0, "chars": 0, "tokens": 0, "unk_tokens": 0, "rows_with_unk": 0}
    for v in by_family.values():
        for kk, vv in v.items():
            total[kk] += vv
    return {
        "scanned_any_eval_text": scanned_any,
        "existing_eval_roots": [str(p) for p in EVAL_ROOTS if p.exists()],
        "total": finish(total),
        "by_family": fam_out,
        "files_with_unk": file_out,
        "top_unk_surface_chars": char_counter.most_common(80),
        "examples": examples,
    }


def main() -> None:
    t0 = time.time()
    for p in [OLD_TOK, NEW_TOK, POOL_10M]:
        if not p.exists():
            raise FileNotFoundError(p)
    old_tok = load_tokenizer(OLD_TOK)
    new_tok = load_tokenizer(NEW_TOK)
    old_vocab = vocab_set(OLD_TOK)
    new_vocab = vocab_set(NEW_TOK)
    alphabet = set(byte_alphabet())
    old_unk_id = old_tok.token_to_id("<unk>")
    new_unk_id = new_tok.token_to_id("<unk>")

    alphabet_report = {
        "bytelevel_alphabet_size": len(alphabet),
        "old_vocab_size": old_tok.get_vocab_size(),
        "new_vocab_size": new_tok.get_vocab_size(),
        "old_missing_bytelevel_alphabet": sorted(alphabet - old_vocab),
        "new_missing_bytelevel_alphabet": sorted(alphabet - new_vocab),
        "old_has_all_bytelevel_alphabet": len(alphabet - old_vocab) == 0,
        "new_has_all_bytelevel_alphabet": len(alphabet - new_vocab) == 0,
    }

    print(json.dumps({"event": "alphabet_checked", **{k: v if not isinstance(v, list) else len(v) for k, v in alphabet_report.items() if k.endswith("size") or k.startswith("old_missing") or k.startswith("new_missing")}}, ensure_ascii=False), flush=True)

    # Full allowed 10M-pool UNK audit for new tokenizer, and a small old-tokenizer sanity pass.
    new_pool_audit = audit_stream(new_tok, new_unk_id, pool_texts(), max_examples=40)
    # Old tokenizer over the same pool: should be zero, but count for direct contrast.
    old_pool_audit = audit_stream(old_tok, old_unk_id, pool_texts(), max_examples=10)
    print(json.dumps({"event": "pool_audit_done", "new_unk_tokens": new_pool_audit["unk_tokens"], "new_rows_with_unk": new_pool_audit["rows_with_unk"], "old_unk_tokens": old_pool_audit["unk_tokens"]}, ensure_ascii=False), flush=True)

    new_eval_audit = audit_eval(new_tok, new_unk_id, max_examples=60)
    old_eval_audit = audit_eval(old_tok, old_unk_id, max_examples=20)
    print(json.dumps({"event": "eval_audit_done", "new_eval_unk_tokens": new_eval_audit["total"]["unk_tokens"], "new_eval_rows_with_unk": new_eval_audit["total"]["rows_with_unk"], "old_eval_unk_tokens": old_eval_audit["total"]["unk_tokens"]}, ensure_ascii=False), flush=True)

    payload = {
        "status": "TOKENIZER_BYTE_UNK_AUDIT",
        "purpose": "Check whether the compliant 10M-trained byte-level BPE tokenizer actually preserves byte-level coverage and avoids <unk> on allowed training and official evaluation text.",
        "paths": {
            "old_tokenizer": str(OLD_TOK),
            "new_tokenizer": str(NEW_TOK),
            "pool_10m": str(POOL_10M),
            "json": str(JSON),
        },
        "sha256": {
            "old_tokenizer": sha256_file(OLD_TOK),
            "new_tokenizer": sha256_file(NEW_TOK),
            "pool_10m": sha256_file(POOL_10M),
        },
        "unk_ids": {"old": old_unk_id, "new": new_unk_id},
        "alphabet_report": alphabet_report,
        "pool_10m_unk_audit": {"old": old_pool_audit, "new": new_pool_audit},
        "eval_text_unk_audit": {"old": old_eval_audit, "new": new_eval_audit},
        "interpretation_rules": [
            "Any <unk> on the 10M training pool would indicate an internal tokenizer training defect.",
            "Any non-negligible <unk> on score-bearing official text would make tokenizer coverage a plausible cause of score loss and motivate retraining a legal same-pool tokenizer with explicit ByteLevel.alphabet(), not changing the data mechanism.",
            "Zero or negligible <unk> means future compliant-score movement should not be attributed to crude unseen-byte coverage; interpretation should focus on vocabulary merges, MLM target geometry, optimization, and the compact-view/reinvestment mechanism under a legal tokenizer."
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_byte_unk_audit/tokenizer_byte_unk_audit.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = []
    md_lines.append("# research tokenizer byte/alphabet and `<unk>` audit\n")
    md_lines.append("This CPU-only audit checks whether the compliant 10M-trained byte-level BPE tokenizer has a coverage defect from not explicitly passing `ByteLevel.alphabet()` to the trainer. It does not inspect active retrain directories and does not use evaluation text to tune vocabulary.\n")
    md_lines.append("## Tokenizer identities\n")
    md_lines.append(f"- Old tokenizer SHA256: `{payload['sha256']['old_tokenizer']}`\n")
    md_lines.append(f"- Compliant tokenizer SHA256: `{payload['sha256']['new_tokenizer']}`\n")
    md_lines.append(f"- 10M pool SHA256: `{payload['sha256']['pool_10m']}`\n")
    md_lines.append("## Byte alphabet coverage\n")
    md_lines.append(f"- ByteLevel alphabet size: {alphabet_report['bytelevel_alphabet_size']}\n")
    md_lines.append(f"- Old tokenizer missing byte-level alphabet entries: {len(alphabet_report['old_missing_bytelevel_alphabet'])}\n")
    md_lines.append(f"- Compliant tokenizer missing byte-level alphabet entries: {len(alphabet_report['new_missing_bytelevel_alphabet'])}\n")
    if alphabet_report['new_missing_bytelevel_alphabet']:
        preview = alphabet_report['new_missing_bytelevel_alphabet'][:80]
        md_lines.append(f"- Missing compliant alphabet preview: `{preview}`\n")
    md_lines.append("## `<unk>` counts\n")
    md_lines.append(f"- Existing eval roots scanned: `{new_eval_audit['existing_eval_roots']}`; scanned_any_eval_text={new_eval_audit['scanned_any_eval_text']}\n")
    for label, audit in [("old on 10M pool", old_pool_audit), ("compliant on 10M pool", new_pool_audit), ("old on eval text", old_eval_audit['total']), ("compliant on eval text", new_eval_audit['total'])]:
        md_lines.append(f"- {label}: rows={audit['rows']}, tokens={audit['tokens']}, unk_tokens={audit['unk_tokens']}, rows_with_unk={audit['rows_with_unk']}, unk_rate={audit['unk_rate']:.8g}, row_unk_rate={audit['row_unk_rate']:.8g}\n")
    md_lines.append("## Eval-family compliant `<unk>` counts\n")
    for fam, audit in new_eval_audit["by_family"].items():
        if audit["unk_tokens"] or audit["rows_with_unk"]:
            md_lines.append(f"- {fam}: rows={audit['rows']}, unk_tokens={audit['unk_tokens']}, rows_with_unk={audit['rows_with_unk']}, unk_rate={audit['unk_rate']:.8g}\n")
    if not any(v["unk_tokens"] or v["rows_with_unk"] for v in new_eval_audit["by_family"].values()):
        md_lines.append("- No compliant-tokenizer `<unk>` events found in scanned official evaluation text.\n")
    if new_eval_audit["examples"]:
        md_lines.append("## Example compliant eval `<unk>` contexts\n")
        for ex in new_eval_audit["examples"][:10]:
            md_lines.append(f"- {ex['family']} `{ex['path']}` n_unk={ex['n_unk']} chars={ex['non_ascii_chars']} text=`{ex['text_prefix'][:240]}`\n")
    md_lines.append("## Scientific interpretation\n")
    if new_pool_audit["unk_tokens"] == 0 and new_eval_audit["total"]["unk_tokens"] == 0:
        md_lines.append("The compliant tokenizer may lack some ByteLevel alphabet symbols in the raw vocabulary, but it produced zero `<unk>` tokens on the allowed 10M pool and the scanned official evaluation text. Thus a future compliant-score movement should not be explained by crude unseen-byte failure; the more likely mechanisms are changed BPE merges, MLM target/whole-word grouping geometry, and optimization under the legal vocabulary.\n")
    elif new_pool_audit["unk_tokens"] == 0:
        md_lines.append("The tokenizer produced no `<unk>` on the 10M training pool but did produce `<unk>` on official evaluation text. This is a coverage vulnerability of the legal tokenizer construction and should be considered if compliant scores fall in families carrying these characters. A repair would be to train a same-pool tokenizer with explicit byte alphabet, not to use evaluation text for vocabulary selection.\n")
    else:
        md_lines.append("The tokenizer produced `<unk>` even on its own 10M training pool. This would be a construction defect requiring immediate repair before interpreting the retrain scientifically.\n")
    md_lines.append(f"\nFull JSON: `{out_json}`\n")
    out_md = _public_path('research/documents/frontier_consolidation/data/tokenizer_byte_unk_audit/tokenizer_byte_unk_audit.md')
    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "new_missing_alphabet": len(alphabet_report["new_missing_bytelevel_alphabet"]),
        "new_pool_unk_tokens": new_pool_audit["unk_tokens"],
        "new_eval_unk_tokens": new_eval_audit["total"]["unk_tokens"],
        "elapsed_sec": payload["elapsed_sec"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
