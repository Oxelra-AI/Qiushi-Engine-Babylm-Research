#!/usr/bin/env python3
"""research audit: can a low-diversity repeat view be BPE-matched to compact?

For each compact pair, compare the existing hash-rotated repeat view against the
best cyclic contiguous source segment of the same whitespace length, chosen to
minimize legal-tokenizer BPE-length difference from the compact rewrite.  This is
CPU-only design support for a future ordinary-WWM factorial; it does not alter any
existing corpus.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
DEFAULT_PAIRS = USER_ROOT / "experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
DEFAULT_TOKENIZER = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/repeat_bpe_match_audit"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def repeat_source_words(source_text: str, n_words: int, salt: str) -> tuple[str, int]:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return "", 0
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out), start


def bpe(tok: Any, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def best_bpe_segment(tok: Any, source_text: str, n_words: int, target_bpe: int) -> dict[str, Any]:
    toks = source_text.split()
    if n_words <= 0 or not toks:
        return {"text": "", "start": 0, "bpe": 0, "abs_delta": abs(target_bpe)}
    best: dict[str, Any] | None = None
    n = len(toks)
    # Same cyclic-contiguous family as the old repeat, but choose the start by BPE match.
    for start in range(n):
        seg = [toks[(start + k) % n] for k in range(n_words)]
        text = " ".join(seg)
        L = bpe(tok, text)
        rec = {"text": text, "start": start, "bpe": L, "abs_delta": abs(L - target_bpe), "delta": L - target_bpe}
        if best is None or (rec["abs_delta"], abs(start - n//2), start) < (best["abs_delta"], abs(best["start"] - n//2), best["start"]):
            best = rec
            if best["abs_delta"] == 0:
                # exact is optimal; deterministic tie-breaking not scientifically important for audit
                pass
    assert best is not None
    return best


def qstats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs)-1); lo = math.floor(pos); hi = math.ceil(pos)
        if lo == hi: return xs[lo]
        return xs[lo]*(hi-pos) + xs[hi]*(pos-lo)
    return {"n": len(xs), "mean": sum(xs)/len(xs), "median": q(0.5), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    pairs_path = Path(args.pairs)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    pairs = read_jsonl(pairs_path)
    records = []
    old_delta = []
    best_delta = []
    old_abs = []
    best_abs = []
    old_exact = 0
    best_exact = 0
    changed_start = 0
    examples = []
    for p in pairs:
        src = str(p["source_text"])
        view = str(p.get("view_text") or p.get("original_compact_text"))
        n_words = int(p.get("view_words", len(view.split())))
        view_bpe = bpe(tok, view)
        old_text, old_start = repeat_source_words(src, n_words, str(p["pair_id"]))
        old_bpe = bpe(tok, old_text)
        best = best_bpe_segment(tok, src, n_words, view_bpe)
        od = old_bpe - view_bpe
        bd = int(best["bpe"]) - view_bpe
        old_delta.append(od); best_delta.append(bd)
        old_abs.append(abs(od)); best_abs.append(abs(bd))
        if od == 0: old_exact += 1
        if bd == 0: best_exact += 1
        if int(best["start"]) != old_start: changed_start += 1
        rec = {
            "pair_id": p.get("pair_id"),
            "source_words": len(src.split()),
            "view_words": n_words,
            "view_bpe": view_bpe,
            "old_repeat_bpe": old_bpe,
            "best_repeat_bpe": int(best["bpe"]),
            "old_delta_repeat_minus_view": od,
            "best_delta_repeat_minus_view": bd,
            "old_start": old_start,
            "best_start": int(best["start"]),
        }
        records.append(rec)
        if abs(od) - abs(bd) >= 3 and len(examples) < 8:
            examples.append({**rec, "view_text": view[:160], "old_repeat": old_text[:160], "best_repeat": str(best["text"])[:160]})
    summary = {
        "status": "REPEAT_BPE_MATCH_AUDIT",
        "created_utc": now_utc(),
        "meaning": "CPU audit of whether a low-diversity source-repeat view can be chosen to match compact rewrite BPE mass more closely while keeping the same whitespace view length. No training or corpus mutation.",
        "inputs": {"pairs": str(pairs_path), "pairs_sha256": sha256_file(pairs_path), "tokenizer": args.tokenizer, "n_pairs": len(pairs)},
        "old_hash_repeat": {"delta_repeat_minus_view": qstats(old_delta), "abs_delta": qstats(old_abs), "exact_bpe_matches": old_exact, "fraction_exact": old_exact/len(pairs)},
        "best_cyclic_bpe_matched_repeat": {"delta_repeat_minus_view": qstats(best_delta), "abs_delta": qstats(best_abs), "exact_bpe_matches": best_exact, "fraction_exact": best_exact/len(pairs), "changed_start_fraction": changed_start/len(pairs)},
        "total_delta_old_repeat_minus_view": sum(old_delta),
        "total_delta_best_repeat_minus_view": sum(best_delta),
        "total_abs_delta_old": sum(old_abs),
        "total_abs_delta_best": sum(best_abs),
        "examples": examples,
        "interpretation": "A future low-diversity repeat control can greatly reduce BPE/supervised-mass mismatch by choosing the source repeat segment start to match compact BPE length, but this also changes which source positions are repeated; it should be audited for content/source-position coverage before replacing the historical hash-rotated repeat control.",
    }
    (out_dir / "repeat_bpe_match_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    with (out_dir / "repeat_bpe_match_records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    (out_dir / "repeat_bpe_match_audit.md").write_text(
        "# research repeat BPE-match audit\n\n"
        f"JSON: `{out_dir / 'repeat_bpe_match_audit.json'}`\n\n"
        f"Pairs: {len(pairs)}\n\n"
        f"Old hash repeat: mean delta repeat-view {sum(old_delta)/len(old_delta):.4f}; mean abs {sum(old_abs)/len(old_abs):.4f}; exact {old_exact/len(pairs):.4f}\n\n"
        f"Best cyclic repeat: mean delta {sum(best_delta)/len(best_delta):.4f}; mean abs {sum(best_abs)/len(best_abs):.4f}; exact {best_exact/len(pairs):.4f}; changed start {changed_start/len(pairs):.4f}\n\n"
        "Use this only as design support. BPE matching is not sufficient; source-position/content coverage must also be matched or interpreted.\n",
        encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_dir / "repeat_bpe_match_audit.json"), "old_mean_delta": sum(old_delta)/len(old_delta), "best_mean_delta": sum(best_delta)/len(best_delta), "old_abs": sum(old_abs)/len(old_abs), "best_abs": sum(best_abs)/len(best_abs), "best_exact_fraction": best_exact/len(pairs)}))

if __name__ == "__main__":
    main()
