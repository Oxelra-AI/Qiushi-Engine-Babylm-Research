#!/usr/bin/env python3
"""research CPU preflight for a context-diversity x anchor-recurrence factorial.

The research question is not local NLL: can ordinary-WWM training in an
independent MLM coordinate distinguish varied compact contexts from repeated
prediction of the same source-shared anchors?  This script checks whether the
existing 12,155 compact pair objects support a clean four-arm construction:

  HS: source_i + compact_i        (high-diversity second context, same source anchors)
  LS: source_i + repeat_i         (low-diversity repeated source words, same anchors)
  HD: source_i + compact_perm(i)  (high-diversity second context, different anchors)
  LD: source_i + repeat_perm(i)   (low-diversity repeated source words, different anchors)

The permutation is a length-preserving derangement within view-word-count bins
where possible.  This is CPU/file/tokenizer analysis only; it does not build
training streams or launch training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
DEFAULT_PAIRS = USER_ROOT / "experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
DEFAULT_TOKENIZER = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/context_anchor_factorial_preflight"

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if",
    "in", "into", "is", "it", "its", "itself", "may", "more", "most", "must", "no", "not", "of",
    "on", "or", "our", "she", "so", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "up", "was", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "within", "would", "you", "your", "we", "also", "all", "any",
    "because", "between", "through", "using", "use", "used", "uses", "like", "one", "two", "many",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-'’]*|\d+(?:\.\d+)?")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def content_set(text: str) -> set[str]:
    out: set[str] = set()
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower().strip("-'’")
        if len(w) <= 2:
            continue
        if w in STOP:
            continue
        # Keep numbers as content-like anchors but normalize decimals/integers literally.
        out.add(w)
    return out


def safe_mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def quantiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    xs = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "mean": float(sum(xs)/len(xs)), "median": q(0.5), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def length_derangement(pairs: list[dict[str, Any]]) -> dict[int, int]:
    """Return mapping i->j; exact within view_words bins where possible.

    Singleton bins are deranged globally among singletons sorted by view_words. This
    preserves zero self-pairs but may cause a few length mismatches.  The audit reports
    them so a later stream builder can either solve them exactly or quarantine them.
    """
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        by_len[int(p["view_words"])].append(i)
    mapping: dict[int, int] = {}
    singletons: list[int] = []
    for L, idxs in sorted(by_len.items()):
        if len(idxs) == 1:
            singletons.extend(idxs)
            continue
        # Stable pseudo-random order within length bin, then rotate one step.
        idxs = sorted(idxs, key=lambda i: hashlib.sha1(str(pairs[i]["pair_id"]).encode()).hexdigest())
        for a, b in zip(idxs, idxs[1:] + idxs[:1]):
            mapping[a] = b
    if singletons:
        if len(singletons) == 1:
            # Fallback: map to closest nonself globally. This should be extremely rare.
            i = singletons[0]
            best = min((j for j in range(len(pairs)) if j != i), key=lambda j: abs(int(pairs[j]["view_words"]) - int(pairs[i]["view_words"])))
            mapping[i] = best
        else:
            singletons = sorted(singletons, key=lambda i: (int(pairs[i]["view_words"]), str(pairs[i]["pair_id"])))
            for a, b in zip(singletons, singletons[1:] + singletons[:1]):
                mapping[a] = b
    return mapping


def bpe_len(tok: Any, text: str) -> int | None:
    if tok is None:
        return None
    return len(tok(text, add_special_tokens=False)["input_ids"])


def summarize_arm(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in records if r.get(key) is not None]
    return quantiles(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--sample", type=int, default=0, help="limit rows for debugging; 0 = all")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(pairs_path)
    if args.sample:
        pairs = pairs[: args.sample]
    for p in pairs:
        p["source_text"] = str(p["source_text"])
        p["view_text"] = str(p.get("view_text") or p.get("rewrite_text") or p.get("original_compact_text"))
        p["source_words"] = int(p.get("source_words", wc(p["source_text"])))
        p["view_words"] = int(p.get("view_words", wc(p["view_text"])))
        if p["source_words"] != wc(p["source_text"]):
            raise RuntimeError(f"source word mismatch {p.get('pair_id')}")
        if p["view_words"] != wc(p["view_text"]):
            raise RuntimeError(f"view word mismatch {p.get('pair_id')}")

    mapping = length_derangement(pairs)
    if len(mapping) != len(pairs):
        raise RuntimeError(f"mapping incomplete: {len(mapping)} vs {len(pairs)}")

    tok = None
    tokenizer_status: dict[str, Any] = {"loaded": False}
    if AutoTokenizer is not None and Path(args.tokenizer).exists():
        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        tokenizer_status = {"loaded": True, "path": args.tokenizer, "vocab_size": int(getattr(tok, "vocab_size", len(tok)))}

    records: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    self_pairs = 0
    exact_len = 0
    exact_total_pair_words = 0
    own_compact_overlaps: list[float] = []
    wrong_compact_overlaps: list[float] = []
    own_repeat_overlaps: list[float] = []
    wrong_repeat_overlaps: list[float] = []
    same_doc_derangements = 0
    length_deltas: list[int] = []
    bpe_deltas: dict[str, list[int]] = {k: [] for k in ["HS_minus_LS", "HD_minus_LD", "HS_minus_HD", "LS_minus_LD"]}
    arm_bpe: dict[str, list[int]] = {k: [] for k in ["HS", "LS", "HD", "LD"]}

    for i, p in enumerate(pairs):
        j = mapping[i]
        donor = pairs[j]
        if j == i:
            self_pairs += 1
        if str(p.get("key", "")) and str(p.get("key")) == str(donor.get("key")):
            same_doc_derangements += 1
        len_delta = int(donor["view_words"]) - int(p["view_words"])
        length_deltas.append(len_delta)
        if len_delta == 0:
            exact_len += 1
        if int(p["source_words"]) + int(donor["view_words"]) == int(p["source_words"]) + int(p["view_words"]):
            exact_total_pair_words += 1
        src_set = content_set(p["source_text"])
        own_comp = content_set(p["view_text"])
        own_rep_text = repeat_source_words(p["source_text"], int(p["view_words"]), str(p["pair_id"]))
        own_rep = content_set(own_rep_text)
        wrong_comp_text = donor["view_text"]
        wrong_rep_text = repeat_source_words(donor["source_text"], int(donor["view_words"]), str(donor["pair_id"]))
        wrong_comp = content_set(wrong_comp_text)
        wrong_rep = content_set(wrong_rep_text)
        den = max(1, len(src_set))
        own_c_ov = len(src_set & own_comp) / den
        wrong_c_ov = len(src_set & wrong_comp) / den
        own_r_ov = len(src_set & own_rep) / den
        wrong_r_ov = len(src_set & wrong_rep) / den
        own_compact_overlaps.append(own_c_ov)
        wrong_compact_overlaps.append(wrong_c_ov)
        own_repeat_overlaps.append(own_r_ov)
        wrong_repeat_overlaps.append(wrong_r_ov)
        hs = f"{p['source_text']} {p['view_text']}".strip()
        ls = f"{p['source_text']} {own_rep_text}".strip()
        hd = f"{p['source_text']} {wrong_comp_text}".strip()
        ld = f"{p['source_text']} {wrong_rep_text}".strip()
        bpes: dict[str, int | None] = {"HS": bpe_len(tok, hs), "LS": bpe_len(tok, ls), "HD": bpe_len(tok, hd), "LD": bpe_len(tok, ld)}
        for k, v in bpes.items():
            if v is not None:
                arm_bpe[k].append(int(v))
        if all(v is not None for v in bpes.values()):
            bpe_deltas["HS_minus_LS"].append(int(bpes["HS"]) - int(bpes["LS"]))
            bpe_deltas["HD_minus_LD"].append(int(bpes["HD"]) - int(bpes["LD"]))
            bpe_deltas["HS_minus_HD"].append(int(bpes["HS"]) - int(bpes["HD"]))
            bpe_deltas["LS_minus_LD"].append(int(bpes["LS"]) - int(bpes["LD"]))
        rec = {
            "i": i,
            "pair_id": p.get("pair_id"),
            "donor_i": j,
            "donor_pair_id": donor.get("pair_id"),
            "source_words": p["source_words"],
            "view_words": p["view_words"],
            "donor_view_words": donor["view_words"],
            "view_word_delta": len_delta,
            "own_compact_content_overlap": own_c_ov,
            "wrong_compact_content_overlap": wrong_c_ov,
            "own_repeat_content_overlap": own_r_ov,
            "wrong_repeat_content_overlap": wrong_r_ov,
            "bpe": bpes,
        }
        records.append(rec)
        if len_delta != 0 and len(mismatch_examples) < 10:
            mismatch_examples.append({
                "pair_id": p.get("pair_id"), "view_words": p["view_words"],
                "donor_pair_id": donor.get("pair_id"), "donor_view_words": donor["view_words"],
                "delta": len_delta,
                "source_prefix": p["source_text"][:120],
                "donor_view_prefix": donor["view_text"][:120],
            })

    summary: dict[str, Any] = {
        "status": "CONTEXT_ANCHOR_FACTORIAL_PREFLIGHT",
        "created_utc": now_utc(),
        "meaning": "CPU feasibility audit for HS/LS/HD/LD ordinary-WWM factorial: context diversity (compact vs repeat second context) crossed with own-source anchor recurrence (same vs deranged second context). No training launched.",
        "inputs": {
            "pairs": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
            "tokenizer": tokenizer_status,
            "n_pairs": len(pairs),
        },
        "derangement": {
            "self_pairs": self_pairs,
            "exact_view_word_length_matches": exact_len,
            "fraction_exact_view_word_length": exact_len / len(pairs) if pairs else None,
            "same_key_derangements": same_doc_derangements,
            "view_word_delta": quantiles([float(x) for x in length_deltas]),
            "total_abs_view_word_delta": int(sum(abs(x) for x in length_deltas)),
            "net_view_word_delta": int(sum(length_deltas)),
            "mismatch_examples": mismatch_examples,
        },
        "anchor_overlap_with_receiver_source": {
            "own_compact": quantiles(own_compact_overlaps),
            "wrong_compact": quantiles(wrong_compact_overlaps),
            "own_repeat": quantiles(own_repeat_overlaps),
            "wrong_repeat": quantiles(wrong_repeat_overlaps),
            "mean_drop_own_to_wrong_compact": safe_mean([a-b for a,b in zip(own_compact_overlaps, wrong_compact_overlaps)]),
            "mean_drop_own_to_wrong_repeat": safe_mean([a-b for a,b in zip(own_repeat_overlaps, wrong_repeat_overlaps)]),
        },
        "bpe_per_pair": {k: quantiles(v) for k, v in arm_bpe.items()},
        "bpe_deltas_per_pair": {k: quantiles(v) for k, v in bpe_deltas.items()},
        "interpretation": {
            "HS": "source_i + compact_i: high-diversity faithful second context with own-source shared anchors",
            "LS": "source_i + repeat_i: low-diversity source-word repetition with own-source anchors",
            "HD": "source_i + compact_deranged(i): high-diversity compact marginal with own-source anchor recurrence removed",
            "LD": "source_i + repeat_deranged(i): low-diversity repeat marginal with own-source anchor recurrence removed",
            "primary_estimands": {
                "diversity_main": "0.5*((HS-LS)+(HD-LD)) on downstream RoBERTa selected columns",
                "own_anchor_main": "0.5*((HS-HD)+(LS-LD))",
                "diversity_x_own_anchor": "(HS-LS)-(HD-LD): whether compact-context benefit requires same semantic anchors across source and second view",
            },
            "important_caveat": "Different-anchor arms are intentionally semantically mismatched source+second-context rows. They preserve ordinary WWM and second-view marginals but introduce incoherence; LD is required so the interaction is not merely mismatch cost. Downstream transfer, not local NLL, is the scientific outcome.",
        },
    }
    out_json = out_dir / "context_anchor_factorial_preflight.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep pair-level record for later stream construction, but bounded enough (~12k rows).
    out_jsonl = out_dir / "pair_derangement_records.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = out_dir / "context_anchor_factorial_preflight.md"
    md.write_text(
        "# research context-diversity × anchor-recurrence preflight\n\n"
        f"JSON: `{out_json}`\n\n"
        "## Derangement\n"
        f"- pairs: {len(pairs)}\n"
        f"- self-pairs: {self_pairs}\n"
        f"- exact view-word length matches: {exact_len}/{len(pairs)} ({exact_len/len(pairs):.6f})\n"
        f"- total abs view-word delta: {sum(abs(x) for x in length_deltas)}; net delta {sum(length_deltas)}\n\n"
        "## Mean receiver-source content overlap\n"
        f"- own compact: {safe_mean(own_compact_overlaps):.6f}; wrong compact: {safe_mean(wrong_compact_overlaps):.6f}\n"
        f"- own repeat: {safe_mean(own_repeat_overlaps):.6f}; wrong repeat: {safe_mean(wrong_repeat_overlaps):.6f}\n\n"
        "## Mean BPE deltas per pair\n"
        f"- HS-LS: {safe_mean(bpe_deltas['HS_minus_LS']):.6f}\n"
        f"- HD-LD: {safe_mean(bpe_deltas['HD_minus_LD']):.6f}\n"
        f"- HS-HD: {safe_mean(bpe_deltas['HS_minus_HD']):.6f}\n"
        f"- LS-LD: {safe_mean(bpe_deltas['LS_minus_LD']):.6f}\n\n"
        "## Scientific reading\n"
        "This supports a four-arm ordinary-WWM factorial if the next constructor can materialize exact 10M/100M streams with identical filler, row packing, tokenizer, initialization, optimizer and precomputed WWM manifests. The different-anchor arms preserve context-type marginals but introduce source/second-view mismatch, so only downstream interaction contrasts are interpretable.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(md), "n_pairs": len(pairs), "exact_length_fraction": summary["derangement"]["fraction_exact_view_word_length"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
