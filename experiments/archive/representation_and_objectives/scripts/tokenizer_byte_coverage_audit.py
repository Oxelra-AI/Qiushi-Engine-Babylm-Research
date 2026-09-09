#!/usr/bin/env python3
"""Audit strict-small tokenizer byte coverage and scored-text <unk>.

One legal 10M-trained tokenizer was found to lack ByteLevel alphabet coverage and
create <unk> on official scored strings.  The tokenizer audited here was trained by a
different method (AutoTokenizer.train_new_from_iterator).  This CPU-only audit checks
its exact SHA 4a95... before interpreting the running full official evaluations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any, Iterator

USER_ROOT = pathlib.Path(".").resolve()
WS = USER_ROOT / "experiments/archive/representation_and_objectives"
OUT = WS / "data/tokenizer_byte_coverage_audit"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/tokenizer_byte_coverage_audit.md')
A01_TOK = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer/tokenizer.json"
OLD_TOK = USER_ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json"
A02_BYTE_TOK = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet/tokenizer.json"
POOL_10M = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
EVAL_ROOT = WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
EXPECTED_SHA = "4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738"
TEXT_KEYS = {
    "text", "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "answer", "option", "option1", "option2", "premise", "hypothesis",
    "sentence1", "sentence2", "passage", "query", "word", "target", "target_word", "candidate",
    "choice", "label", "ending0", "ending1", "ending2", "ending3", "startphrase",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tok(path: pathlib.Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    # Tokenizer JSONs saved by the training stack can carry padding/truncation
    # configuration. For coverage auditing we need the raw unpadded, untruncated
    # tokenization of each scored string; otherwise every record can appear to be
    # length 256 and unknowns after a truncation point can be hidden.
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def byte_alphabet() -> list[str]:
    from tokenizers.pre_tokenizers import ByteLevel
    return list(ByteLevel.alphabet())


def family_from_path(path: pathlib.Path) -> str:
    s = "/".join(path.parts).lower()
    name = path.name.lower()
    if "supplement" in s or "blimp_supp" in s:
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
    if "aoa" in s or "cdi" in s or "reading" in s:
        return "Reading_AoA"
    if "glue" in s or "superglue" in s or name in {"train.jsonl", "validation.jsonl", "test.jsonl"}:
        return "SuperGLUE"
    return "Other"


def iter_string_values(obj: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lk = str(k).lower()
            if isinstance(v, str):
                if lk in TEXT_KEYS or any(t in lk for t in ("sentence", "text", "context", "question", "answer", "premise", "hypothesis", "passage", "word")):
                    yield key, v
            else:
                yield from iter_string_values(v, key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_string_values(v, f"{prefix}[{i}]")


def eval_texts() -> Iterator[tuple[str, str, str, str]]:
    for path in sorted(EVAL_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv", ".txt"}:
            continue
        family = family_from_path(path)
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        for key, text in iter_string_values(obj):
                            if text:
                                yield family, str(path), f"line{i}.{key}", text
            elif path.suffix.lower() == ".json":
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                for key, text in iter_string_values(obj):
                    if text:
                        yield family, str(path), key, text
            elif path.suffix.lower() in {".csv", ".tsv"}:
                delim = "\t" if path.suffix.lower() == ".tsv" else ","
                with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f, delimiter=delim)
                    for i, row in enumerate(reader):
                        for k, text in row.items():
                            if not text:
                                continue
                            lk = str(k).lower()
                            if lk in TEXT_KEYS or any(t in lk for t in ("sentence", "text", "context", "question", "answer", "premise", "hypothesis", "passage", "word")):
                                yield family, str(path), f"row{i}.{k}", text
            else:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        text = line.strip()
                        if text:
                            yield family, str(path), f"line{i}", text
        except Exception as e:
            print(json.dumps({"warning": "eval_file_skipped", "path": str(path), "error": repr(e)}), flush=True)


def pool_texts() -> Iterator[tuple[str, str, str, str]]:
    with POOL_10M.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                yield "train_pool", str(POOL_10M), f"line{i}.text", text


def audit_stream(tok, unk_id: int | None, stream: Iterator[tuple[str, str, str, str]], max_examples: int = 40) -> dict[str, Any]:
    by_family = defaultdict(lambda: {"strings": 0, "chars": 0, "tokens": 0, "unk_tokens": 0, "strings_with_unk": 0})
    by_path_field = defaultdict(lambda: {"strings": 0, "tokens": 0, "unk_tokens": 0, "strings_with_unk": 0, "family": ""})
    examples = []
    for family, path, field, text in stream:
        enc = tok.encode(text, add_special_tokens=False)
        ids = enc.ids
        n_unk = 0 if unk_id is None else sum(1 for x in ids if x == unk_id)
        for d in (by_family[family], by_path_field[(path, field)]):
            d["strings"] += 1
            d["tokens"] += len(ids)
            d["unk_tokens"] += n_unk
            d["strings_with_unk"] += 1 if n_unk else 0
        by_path_field[(path, field)]["family"] = family
        by_family[family]["chars"] += len(text)
        if n_unk and len(examples) < max_examples:
            pos = [i for i, x in enumerate(ids) if x == unk_id]
            examples.append({
                "family": family,
                "path": path,
                "field": field,
                "n_unk": n_unk,
                "n_tokens": len(ids),
                "tokens_around_first_unk": enc.tokens[max(0, pos[0]-8):pos[0]+9],
                "text_prefix": text[:500],
                "non_ascii_or_control_chars": sorted({repr(ch)[1:-1] for ch in text if ord(ch) > 126 or ord(ch) < 32})[:50],
            })
    fam_out = {}
    for fam, d in sorted(by_family.items()):
        fam_out[fam] = dict(d)
        fam_out[fam]["unk_rate"] = d["unk_tokens"] / d["tokens"] if d["tokens"] else 0.0
        fam_out[fam]["string_unk_rate"] = d["strings_with_unk"] / d["strings"] if d["strings"] else 0.0
    total = {"strings": 0, "chars": 0, "tokens": 0, "unk_tokens": 0, "strings_with_unk": 0}
    for d in by_family.values():
        for k in total:
            total[k] += d.get(k, 0)
    total["unk_rate"] = total["unk_tokens"] / total["tokens"] if total["tokens"] else 0.0
    total["string_unk_rate"] = total["strings_with_unk"] / total["strings"] if total["strings"] else 0.0
    hotspots = []
    for (path, field), d in by_path_field.items():
        if d["unk_tokens"]:
            hotspots.append({"path": path, "field": field, **dict(d), "unk_rate": d["unk_tokens"] / d["tokens"] if d["tokens"] else 0.0})
    hotspots.sort(key=lambda x: (-x["unk_tokens"], -x["strings_with_unk"], x["path"], x["field"]))
    return {"total": total, "by_family": fam_out, "hotspots": hotspots[:80], "examples": examples}


def compare_tokenizers(a, b, stream_factory, label_a="a", label_b="b", max_examples: int = 20) -> dict[str, Any]:
    stats = {"strings": 0, "a_tokens": 0, "b_tokens": 0, "a_unk": 0, "b_unk": 0, "mean_b_minus_a": None, "mean_abs_diff": None, "by_family": {}}
    diffs = []
    byfam = defaultdict(lambda: {"strings": 0, "a_tokens": 0, "b_tokens": 0, "a_unk": 0, "b_unk": 0, "diffs": []})
    a_unk_id = a.token_to_id("<unk>")
    b_unk_id = b.token_to_id("<unk>")
    examples = []
    for family, path, field, text in stream_factory():
        ea = a.encode(text, add_special_tokens=False)
        eb = b.encode(text, add_special_tokens=False)
        au = 0 if a_unk_id is None else sum(1 for x in ea.ids if x == a_unk_id)
        bu = 0 if b_unk_id is None else sum(1 for x in eb.ids if x == b_unk_id)
        diff = len(eb.ids) - len(ea.ids)
        stats["strings"] += 1
        stats["a_tokens"] += len(ea.ids)
        stats["b_tokens"] += len(eb.ids)
        stats["a_unk"] += au
        stats["b_unk"] += bu
        diffs.append(diff)
        bf = byfam[family]
        bf["strings"] += 1
        bf["a_tokens"] += len(ea.ids)
        bf["b_tokens"] += len(eb.ids)
        bf["a_unk"] += au
        bf["b_unk"] += bu
        bf["diffs"].append(diff)
        if (au or bu or abs(diff) >= 5) and len(examples) < max_examples:
            examples.append({"family": family, "path": path, "field": field, "text_prefix": text[:300], f"{label_a}_tokens": len(ea.ids), f"{label_b}_tokens": len(eb.ids), f"{label_a}_unk": au, f"{label_b}_unk": bu, "diff_b_minus_a": diff})
    stats["token_ratio_b_over_a"] = stats["b_tokens"] / stats["a_tokens"] if stats["a_tokens"] else None
    stats["mean_b_minus_a"] = sum(diffs) / len(diffs) if diffs else None
    stats["mean_abs_diff"] = sum(abs(d) for d in diffs) / len(diffs) if diffs else None
    for fam, d in byfam.items():
        stats["by_family"][fam] = {
            "strings": d["strings"],
            "a_tokens": d["a_tokens"],
            "b_tokens": d["b_tokens"],
            "token_ratio_b_over_a": d["b_tokens"] / d["a_tokens"] if d["a_tokens"] else None,
            "a_unk": d["a_unk"],
            "b_unk": d["b_unk"],
            "mean_b_minus_a": sum(d["diffs"]) / len(d["diffs"]) if d["diffs"] else None,
        }
    stats["examples"] = examples
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a01 = load_tok(A01_TOK)
    old = load_tok(OLD_TOK)
    a02_byte = load_tok(A02_BYTE_TOK) if A02_BYTE_TOK.exists() else None
    a01_sha = sha256_file(A01_TOK)
    vocab = set(a01.get_vocab().keys())
    missing_alphabet = [x for x in byte_alphabet() if x not in vocab]
    unk_id = a01.token_to_id("<unk>")
    pool_audit = audit_stream(a01, unk_id, pool_texts(), max_examples=20)
    eval_audit = audit_stream(a01, unk_id, eval_texts(), max_examples=40)
    compare_old_a01_eval = compare_tokenizers(old, a01, eval_texts, label_a="old", label_b="a01", max_examples=30)
    compare_a01_byte_eval = None
    if a02_byte is not None:
        compare_a01_byte_eval = compare_tokenizers(a01, a02_byte, eval_texts, label_a="a01", label_b="a02_byte", max_examples=30)
    payload = {
        "status": "A01_TOKENIZER_BYTE_COVERAGE_AUDIT",
        "created_utc": now_utc(),
        "a01_tokenizer": str(A01_TOK.relative_to(USER_ROOT)),
        "a01_sha256": a01_sha,
        "expected_sha256": EXPECTED_SHA,
        "sha_matches_expected": a01_sha == EXPECTED_SHA,
        "vocab_size": a01.get_vocab_size(),
        "unk_id": unk_id,
        "missing_bytelevel_alphabet_count": len(missing_alphabet),
        "missing_bytelevel_alphabet_sample": missing_alphabet[:120],
        "pool_10m": str(POOL_10M.relative_to(USER_ROOT)),
        "pool_audit": pool_audit,
        "eval_root": str(EVAL_ROOT.relative_to(USER_ROOT)),
        "eval_audit": eval_audit,
        "compare_old_vs_a01_eval": compare_old_a01_eval,
        "compare_a01_vs_a02_byte_eval": compare_a01_byte_eval,
        "interpretation": [
            "Evaluation text is used only to audit tokenizer byte coverage and scored-input <unk>, not to train or select vocabulary.",
            "If A01 tokenizer has nonzero scored-text <unk>, the running evaluations remain useful as a flawed-tokenizer contrast but are not the cleanest submission endpoint; a byte-alphabet same-pool tokenizer repair would be justified.",
            "If A01 tokenizer has zero scored-text <unk> and missing byte alphabet is zero, the running evaluations remain submission-relevant with respect to tokenizer construction.",
        ],
    }
    out_json = OUT / "tokenizer_byte_coverage_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research A01 tokenizer byte coverage audit",
        "",
        f"A01 tokenizer: `{A01_TOK.relative_to(USER_ROOT)}` SHA `{a01_sha}` (expected match: {a01_sha == EXPECTED_SHA}).",
        f"Vocab size {a01.get_vocab_size()}, `<unk>` id {unk_id}, missing ByteLevel alphabet entries {len(missing_alphabet)}.",
        "",
        "## 10M pool `<unk>`",
        f"strings={pool_audit['total']['strings']}, tokens={pool_audit['total']['tokens']}, unk_tokens={pool_audit['total']['unk_tokens']}, strings_with_unk={pool_audit['total']['strings_with_unk']}, unk_rate={pool_audit['total']['unk_rate']:.8g}.",
        "",
        "## Official scored-text `<unk>`",
        f"strings={eval_audit['total']['strings']}, tokens={eval_audit['total']['tokens']}, unk_tokens={eval_audit['total']['unk_tokens']}, strings_with_unk={eval_audit['total']['strings_with_unk']}, unk_rate={eval_audit['total']['unk_rate']:.8g}.",
        "",
        "### By family",
    ]
    for fam, d in eval_audit["by_family"].items():
        lines.append(f"- {fam}: strings={d['strings']}, tokens={d['tokens']}, unk_tokens={d['unk_tokens']}, strings_with_unk={d['strings_with_unk']}, unk_rate={d['unk_rate']:.8g}")
    lines += [
        "",
        "## Comparison to old inherited tokenizer on official scored text",
        f"A01/old token ratio = {compare_old_a01_eval['token_ratio_b_over_a']:.8f}; old unk={compare_old_a01_eval['a_unk']}, A01 unk={compare_old_a01_eval['b_unk']}.",
    ]
    if compare_a01_byte_eval is not None:
        lines += [
            "",
            "## Comparison to A02 byte-alphabet tokenizer on official scored text",
            f"A02-byte/A01 token ratio = {compare_a01_byte_eval['token_ratio_b_over_a']:.8f}; A01 unk={compare_a01_byte_eval['a_unk']}, A02-byte unk={compare_a01_byte_eval['b_unk']}.",
        ]
    lines += [
        "",
        "## Interpretation",
        "This check decides whether the A01 corrected-tokenizer official evaluations are clean with respect to byte-level coverage or should be treated as a flawed-tokenizer contrast pending a byte-alphabet same-pool retrain.",
        f"Full JSON: `{out_json.relative_to(USER_ROOT)}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "note": str(NOTE.relative_to(USER_ROOT)),
        "a01_sha_matches": payload["sha_matches_expected"],
        "missing_bytelevel_alphabet_count": len(missing_alphabet),
        "pool_unk_tokens": pool_audit["total"]["unk_tokens"],
        "eval_unk_tokens": eval_audit["total"]["unk_tokens"],
        "eval_strings_with_unk": eval_audit["total"]["strings_with_unk"],
        "old_vs_a01_eval_token_ratio": compare_old_a01_eval["token_ratio_b_over_a"],
        "a02byte_vs_a01_eval_token_ratio": compare_a01_byte_eval["token_ratio_b_over_a"] if compare_a01_byte_eval else None,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
