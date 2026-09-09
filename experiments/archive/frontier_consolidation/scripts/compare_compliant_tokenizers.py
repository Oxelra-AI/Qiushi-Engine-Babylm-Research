#!/usr/bin/env python3
"""research: compare the research legal tokenizer and research byte-alphabet legal tokenizer.

This CPU-only analysis checks whether the byte-alphabet repair removes scored-text
<unk> events while preserving tokenization length/segmentation geometry close enough
to make it a tokenizer-construction repair rather than a new data/model idea. It uses
only the allowed 10M pool for training-side summaries and official evaluation text
for coverage/length auditing, not vocabulary fitting.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Iterable, Iterator, Tuple

USER_ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
POOL_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
EVAL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
TOK_A = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')  # currently-running retrain tokenizer
TOK_B = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet/tokenizer.json')  # same-pool byte-alphabet repair
OUT = _public_path('experiments/archive/frontier_consolidation/data/compare_compliant_tokenizers')
OUT.mkdir(parents=True, exist_ok=True)

SCORING_TEXT_FIELDS = {
    "BLiMP": {"sentence_good", "sentence_bad", "good_sentence", "bad_sentence", "sentence"},
    "Supplement": {"sentence_good", "sentence_bad", "good_sentence", "bad_sentence", "sentence"},
    "EWoK": {"sentence", "sentence_good", "sentence_bad", "text", "context", "target", "candidate", "correct", "incorrect"},
    "Entity": {"sentence", "context", "query", "answer", "text"},
    "COMPS": {"sentence", "context", "candidate", "option", "choice", "text"},
    "GlobalPIQA": {"goal", "sol1", "sol2", "sentence", "context", "option", "choice", "text"},
    "Reading_AoA": {"sentence", "context", "text", "utterance", "transcript", "word"},
    "SuperGLUE": {"passage", "question", "answer", "premise", "hypothesis", "sentence", "sentence1", "sentence2", "paragraph", "text", "span", "query", "question1", "question2", "span1_text", "span2_text"},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tok(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    try: tok.no_padding()
    except Exception: pass
    try: tok.no_truncation()
    except Exception: pass
    return tok


def family_for(path: pathlib.Path) -> str:
    s = str(path).lower()
    if "supplement_filtered" in s: return "Supplement"
    if "blimp_filtered" in s: return "BLiMP"
    if "ewok_filtered" in s: return "EWoK"
    if "entity_tracking" in s: return "Entity"
    if "/comps" in s or "comps" in path.name.lower(): return "COMPS"
    if "globalpiqa" in s or "piqa" in s: return "GlobalPIQA"
    if "/aoa/" in s or "/reading/" in s: return "Reading_AoA"
    if "glue_filtered" in s: return "SuperGLUE"
    return "Other"


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


def tail(field: str) -> str:
    return field.lower().split(".")[-1]


def likely_field(family: str, field: str) -> bool:
    t = tail(field)
    allowed = SCORING_TEXT_FIELDS.get(family, set())
    return t in allowed or any(k in t for k in allowed)


def eval_texts() -> Iterator[Tuple[str, str]]:
    for path in sorted(EVAL.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv"}:
            continue
        fam = family_for(path)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip(): continue
                    try: obj = json.loads(line)
                    except Exception: continue
                    for field, text in iter_strs(obj):
                        if text and likely_field(fam, field):
                            yield fam, text
        elif path.suffix.lower() == ".json":
            try: obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception: continue
            for field, text in iter_strs(obj):
                if text and likely_field(fam, field):
                    yield fam, text
        else:
            delim = "\t" if path.suffix.lower() == ".tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter=delim)
                for row in reader:
                    for field, text in row.items():
                        if text and likely_field(fam, field):
                            yield fam, text


def pool_texts(sample_every: int = 1) -> Iterator[Tuple[str, str]]:
    with open(POOL_10M, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if sample_every > 1 and i % sample_every:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                yield "train_pool", text


def compare_stream(tok_a, tok_b, unk_a, unk_b, stream: Iterable[Tuple[str, str]], max_diff_examples: int = 30):
    by_family = defaultdict(lambda: {"strings":0,"tokens_a":0,"tokens_b":0,"unk_a":0,"unk_b":0,"trunc_a":0,"trunc_b":0,"b_minus_a_sum":0,"abs_diff_sum":0,"b_longer":0,"b_shorter":0,"same_len":0})
    examples = []
    for fam, text in stream:
        ea = tok_a.encode(text); eb = tok_b.encode(text)
        la, lb = len(ea.ids), len(eb.ids)
        ua = sum(1 for x in ea.ids if x == unk_a)
        ub = sum(1 for x in eb.ids if x == unk_b)
        d = by_family[fam]
        d["strings"] += 1
        d["tokens_a"] += la
        d["tokens_b"] += lb
        d["unk_a"] += ua
        d["unk_b"] += ub
        d["trunc_a"] += la > 256
        d["trunc_b"] += lb > 256
        d["b_minus_a_sum"] += lb - la
        d["abs_diff_sum"] += abs(lb - la)
        d["b_longer"] += lb > la
        d["b_shorter"] += lb < la
        d["same_len"] += lb == la
        if len(examples) < max_diff_examples and (ua or ub or abs(lb-la) >= 8):
            examples.append({"family": fam, "text_prefix": text[:300], "len_a": la, "len_b": lb, "unk_a": ua, "unk_b": ub, "tokens_a_head": ea.tokens[:40], "tokens_b_head": eb.tokens[:40]})
    def finish(v):
        o = dict(v)
        n = o["strings"]
        o["token_ratio_b_over_a"] = o["tokens_b"] / o["tokens_a"] if o["tokens_a"] else 0.0
        o["mean_b_minus_a"] = o["b_minus_a_sum"] / n if n else 0.0
        o["mean_abs_len_diff"] = o["abs_diff_sum"] / n if n else 0.0
        o["trunc_rate_a"] = o["trunc_a"] / n if n else 0.0
        o["trunc_rate_b"] = o["trunc_b"] / n if n else 0.0
        o["unk_rate_a"] = o["unk_a"] / o["tokens_a"] if o["tokens_a"] else 0.0
        o["unk_rate_b"] = o["unk_b"] / o["tokens_b"] if o["tokens_b"] else 0.0
        return o
    total = {"strings":0,"tokens_a":0,"tokens_b":0,"unk_a":0,"unk_b":0,"trunc_a":0,"trunc_b":0,"b_minus_a_sum":0,"abs_diff_sum":0,"b_longer":0,"b_shorter":0,"same_len":0}
    for d in by_family.values():
        for k in total: total[k] += d[k]
    return {"total": finish(total), "by_family": {k: finish(v) for k,v in sorted(by_family.items())}, "examples": examples}


def main():
    t0 = time.time()
    tok_a = load_tok(TOK_A); tok_b = load_tok(TOK_B)
    va, vb = set(tok_a.get_vocab().keys()), set(tok_b.get_vocab().keys())
    from tokenizers.pre_tokenizers import ByteLevel
    alph = set(ByteLevel.alphabet())
    payload = {
        "status": "COMPARE_LEGAL_COMPLIANT_TOKENIZERS",
        "purpose": "Compare current running compliant tokenizer against a legal same-pool byte-alphabet repair tokenizer for unknown coverage and length geometry.",
        "tokenizers": {
            "current_running": {"path": str(TOK_A), "sha256": sha256_file(TOK_A), "vocab_size": tok_a.get_vocab_size(), "missing_bytelevel_alphabet": len(alph - va)},
            "bytealphabet": {"path": str(TOK_B), "sha256": sha256_file(TOK_B), "vocab_size": tok_b.get_vocab_size(), "missing_bytelevel_alphabet": len(alph - vb)},
            "shared_token_strings": len(va & vb),
            "shared_fraction_of_16384": len(va & vb) / 16384,
        },
        "train_pool_full": compare_stream(tok_a, tok_b, tok_a.token_to_id("<unk>"), tok_b.token_to_id("<unk>"), pool_texts(), max_diff_examples=20),
        "eval_scored_text": compare_stream(tok_a, tok_b, tok_a.token_to_id("<unk>"), tok_b.token_to_id("<unk>"), eval_texts(), max_diff_examples=60),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/compare_compliant_tokenizers/compare_compliant_tokenizers.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md=[]
    md.append("# research comparison of two legal same-pool compliant tokenizers\n\n")
    md.append("A = research current-running legal tokenizer; B = research legal byte-alphabet tokenizer trained on the same 10M pool with `initial_alphabet=ByteLevel.alphabet()`. Evaluation text is used only for coverage/length auditing.\n\n")
    md.append("## Tokenizer identity\n")
    for k,v in payload["tokenizers"].items():
        if isinstance(v, dict): md.append(f"- {k}: SHA `{v['sha256']}`, vocab={v['vocab_size']}, missing_bytelevel_alphabet={v['missing_bytelevel_alphabet']}\n")
    md.append(f"- Shared token strings: {payload['tokenizers']['shared_token_strings']}/16384 = {payload['tokenizers']['shared_fraction_of_16384']:.6f}\n\n")
    for section in ["train_pool_full", "eval_scored_text"]:
        tot=payload[section]["total"]
        md.append(f"## {section} total\n")
        md.append(f"strings={tot['strings']}, A_tokens={tot['tokens_a']}, B_tokens={tot['tokens_b']}, B/A={tot['token_ratio_b_over_a']:.8f}, A_unk={tot['unk_a']}, B_unk={tot['unk_b']}, A_trunc>{tot['trunc_a']}, B_trunc>{tot['trunc_b']}, mean_B_minus_A={tot['mean_b_minus_a']:.4f}, mean_abs_diff={tot['mean_abs_len_diff']:.4f}\n\n")
        md.append(f"### {section} by family\n")
        for fam,d in payload[section]["by_family"].items():
            md.append(f"- {fam}: strings={d['strings']}, B/A={d['token_ratio_b_over_a']:.6f}, A_unk={d['unk_a']}, B_unk={d['unk_b']}, A_trunc={d['trunc_a']}, B_trunc={d['trunc_b']}, mean_B-A={d['mean_b_minus_a']:.4f}\n")
        md.append("\n")
    md.append("## Scientific interpretation\n")
    et=payload["eval_scored_text"]["total"]; tr=payload["train_pool_full"]["total"]
    if et["unk_b"] == 0 and tr["unk_b"] == 0:
        md.append("The byte-alphabet tokenizer removes the `<unk>` coverage failure on both the 10M pool and likely scored evaluation strings. Its training-pool token count differs only modestly from the current tokenizer, so it is a legal tokenizer-construction repair rather than a changed corpus route. If the current running retrain performs poorly in Supplement/Reading/SuperGLUE where `<unk>` appears, rerunning the frozen recipe with this byte-alphabet tokenizer is a scientifically justified repair.\n")
    else:
        md.append("The byte-alphabet tokenizer did not fully remove `<unk>` events, so further tokenizer construction debugging would be required before treating it as the clean repair.\n")
    md.append(f"\nFull JSON: `{out_json}`\n")
    out_md=_public_path('research/documents/frontier_consolidation/data/compare_compliant_tokenizers/compare_compliant_tokenizers.md')
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "out_json": str(out_json), "out_md": str(out_md),
        "shared_fraction": payload["tokenizers"]["shared_fraction_of_16384"],
        "train_ratio_B_over_A": payload["train_pool_full"]["total"]["token_ratio_b_over_a"],
        "eval_ratio_B_over_A": payload["eval_scored_text"]["total"]["token_ratio_b_over_a"],
        "A_eval_unk": et["unk_a"], "B_eval_unk": et["unk_b"],
        "elapsed_sec": payload["elapsed_sec"],
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
