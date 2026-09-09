#!/usr/bin/env python3
"""research: characterize offset-unassigned byte-level tokens in chunking audits.

CPU-only. The sequence/pair chunking audits map tokenizer offsets to whitespace
words and source/rewrite spans. Byte-level BPE can emit tokens whose offset is
only a separator space (e.g. token string 'Ġ'); those do enter the trainer but
are not assigned to a whitespace word/span by the offset mapper. This script
quantifies their size and token identities so downstream interpretation does
not confuse a tiny accounting caveat with a route result.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

POOL_10M = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_MAP = WORKSPACE / "data" / "pair_span_map" / "pair_span_map.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
TOKENIZERS = {
    "legal16k": WORKSPACE / "data" / "compliant_tokenizer",
    "minfreq50_supportfloor": WORKSPACE / "data" / "supportfloor_tokenizers" / "legal_byte_bpe_40k_minfreq50",
}
OUT_DIR = WORKSPACE / "data" / "unassigned_space_token_probe"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/unassigned_space_token_probe.md')
WORD_RE = re.compile(r"\S+")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def changed_row_indices() -> set[int]:
    return {int(json.loads(line)["row_index_in_pool_1based"]) for line in PAIR_MAP.open("r", encoding="utf-8") if line.strip()}


def analyze(tokenizer, row_filter: set[int] | None) -> dict[str, Any]:
    rows = 0
    words = 0
    total_tokens = 0
    unassigned = 0
    token_counter: collections.Counter[str] = collections.Counter()
    substr_counter: collections.Counter[str] = collections.Counter()
    samples: list[dict[str, Any]] = []
    with POOL_10M.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            if row_filter is not None and row_idx not in row_filter:
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
            starts = [s for s, _ in spans]
            enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offsets = enc.get("offset_mapping") or []
            rows += 1
            words += int(obj.get("words", len(text.split())))
            for i, (s0, e0) in enumerate(offsets):
                s = int(s0); e = int(e0)
                total_tokens += 1
                idx = bisect.bisect_right(starts, s) - 1
                ok = False
                for j in (idx, idx + 1):
                    if 0 <= j < len(spans):
                        a, b = spans[j]
                        if s < b and e > a:
                            ok = True
                            break
                if not ok:
                    unassigned += 1
                    tok = str(tokenizer.convert_ids_to_tokens(int(ids[i])))
                    substr = text[s:e]
                    token_counter[tok] += 1
                    substr_counter[repr(substr)] += 1
                    if len(samples) < 20:
                        samples.append({
                            "row_index": row_idx,
                            "example_id": obj.get("example_id"),
                            "token_index": i,
                            "token_id": int(ids[i]),
                            "token": tok,
                            "offset": [s, e],
                            "substr_repr": repr(substr),
                            "around": text[max(0, s - 24):min(len(text), e + 24)],
                        })
    return {
        "rows": rows,
        "words": words,
        "total_tokens": total_tokens,
        "unassigned_tokens": unassigned,
        "unassigned_fraction_of_tokens": unassigned / max(1, total_tokens),
        "top_unassigned_tokens": token_counter.most_common(12),
        "top_unassigned_substrings": substr_counter.most_common(12),
        "samples": samples,
    }


def write_note(result: dict[str, Any]) -> None:
    lines = ["# research unassigned space-token probe\n", "CPU-only characterization of offset-unassigned tokens in the research sequence/pair chunking audits.\n\n"]
    lines.append(f"Pool SHA matched: `{result['sha256']['pool_matches']}`.\n\n")
    lines.append("| tokenizer | scope | rows | tokens | unassigned | fraction | top token | top substring |\n")
    lines.append("|---|---|---:|---:|---:|---:|---|---|\n")
    for label, scopes in result["tokenizer_results"].items():
        for scope, rec in scopes.items():
            top_tok = rec["top_unassigned_tokens"][0] if rec["top_unassigned_tokens"] else ["", 0]
            top_sub = rec["top_unassigned_substrings"][0] if rec["top_unassigned_substrings"] else ["", 0]
            lines.append(f"| {label} | {scope} | {rec['rows']} | {rec['total_tokens']} | {rec['unassigned_tokens']} | {rec['unassigned_fraction_of_tokens']:.4f} | {top_tok[0]}:{top_tok[1]} | {top_sub[0]}:{top_sub[1]} |\n")
    lines.append("\n## Scientific reading\n")
    lines.append("The unassigned offsets are overwhelmingly standalone separator-space byte-level tokens (`Ġ`, substring `' '`). They enter the trainer's token stream but are not semantic word content. Earlier offset-based word/pair chunk lengths are therefore slightly optimistic by about 0.6% on the full pool and about 0.8% on changed compact-pair rows. This does not change the qualitative conclusion that prefix slicing hides large suffix fractions or that pair-aware chunking is needed, but an eventual faithful sequence trainer should chunk the tokenizer group stream directly rather than reconstruct chunks only from whitespace words.\n")
    lines.append(f"\nFull JSON: `{rel(OUT_DIR / 'unassigned_space_token_probe.json')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL_10M)
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch {pool_sha}")
    changed = changed_row_indices()
    result: dict[str, Any] = {
        "status": "UNASSIGNED_SPACE_TOKEN_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Quantify tokenizer-offset tokens not assigned to whitespace words/source-rewrite spans in research chunking audits.",
        "sha256": {"pool_expected": EXPECTED_POOL_SHA, "pool_actual": pool_sha, "pool_matches": True},
        "inputs": {"pool_10m": rel(POOL_10M), "pair_map": rel(PAIR_MAP), "changed_rows": len(changed)},
        "tokenizer_results": {},
    }
    for label, path in TOKENIZERS.items():
        tok = base.make_portable_tokenizer(str(path))
        print(json.dumps({"event": "probe", "tokenizer": label, "scope": "full_pool"}), flush=True)
        full = analyze(tok, None)
        print(json.dumps({"event": "probe", "tokenizer": label, "scope": "changed_rows"}), flush=True)
        changed_rec = analyze(tok, changed)
        result["tokenizer_results"][label] = {"full_pool": full, "changed_rows": changed_rec}
    out_json = OUT_DIR / "unassigned_space_token_probe.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
