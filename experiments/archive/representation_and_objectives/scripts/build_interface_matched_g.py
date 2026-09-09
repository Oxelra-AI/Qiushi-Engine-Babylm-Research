#!/usr/bin/env python3
"""research: build interface-matched generic controls for S/G marginal factorization.

This is a CPU-only repair attempt; Stage-1 training was deferred.
It tries to remove the main learning-interface confounds in Arm G:
  * lighter BPE burden than S;
  * more fragmented span structure;
  * uncontrolled realized WWM burden.

For each source/rewrite pair, it keeps Arm S fixed and constructs two same-length source-
extract controls using the same span-count/skeleton family as S:
  * Gm: interface-matched generic. Candidate selection is based on content-count, BPE,
        span/position/gap matching, with no compact-overlap term in the score.
  * Ga: interface-matched anti-retained. Same matching terms, plus a compact-overlap
        penalty, so it more directly contrasts retained-proposition words with matched
        non-retained source words.

Both controls remain deterministic and legal: no language-learned tagger/model is used.
The output does not authorize H100 training; it is an audit artifact for deciding whether
learning-interface matching is now adequate or whether to move to same-text target selection.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
IN_PATH = ROOT / "data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl"
OUT_DIR = ROOT / "data/interface_matched_g"
TOK_PATH = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"

FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1


def stats(xs: list[float | int]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    def q(p: float):
        if n == 1: return s[0]
        return s[min(n-1, max(0, int(round(p*(n-1)))))]
    return {"n": n, "mean": round(float(statistics.mean(s)), 6),
            "median": round(float(statistics.median(s)), 6),
            "p05": round(float(q(0.05)), 6), "p10": round(float(q(0.10)), 6),
            "p25": round(float(q(0.25)), 6), "p75": round(float(q(0.75)), 6),
            "p90": round(float(q(0.90)), 6), "p95": round(float(q(0.95)), 6),
            "min": round(float(s[0]), 6), "max": round(float(s[-1]), 6)}


def read_jsonl(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spans_from_positions(pos: list[int]) -> list[tuple[int, int]]:
    if not pos:
        return []
    p = sorted(pos)
    spans = []
    st = prev = p[0]
    for x in p[1:]:
        if x == prev + 1:
            prev = x
        else:
            spans.append((st, prev))
            st = prev = x
    spans.append((st, prev))
    return spans


def positions_from_spans(spans: list[tuple[int, int]]) -> list[int]:
    out = []
    for st, en in spans:
        out.extend(range(st, en+1))
    return sorted(out)


def pos_features(pos: list[int], n_src: int) -> dict[str, float]:
    if not pos:
        return {"mean": 0.5, "spread": 0.0, "max_gap": 0.0, "mean_gap": 0.0, "span_count": 0}
    denom = max(1, n_src - 1)
    rel = [p / denom for p in pos]
    gaps = [pos[i]-pos[i-1]-1 for i in range(1, len(pos))]
    return {"mean": statistics.mean(rel), "spread": max(rel)-min(rel) if len(rel)>1 else 0.0,
            "max_gap": max(gaps) if gaps else 0.0,
            "mean_gap": statistics.mean(gaps) if gaps else 0.0,
            "span_count": len(spans_from_positions(pos))}


def compact_overlap(source_words: list[str], positions: list[int], c_text: str) -> float:
    rw = set(norm(w) for w in c_text.split() if norm(w))
    if not rw:
        return 0.0
    side = set(norm(source_words[j]) for j in positions if 0 <= j < len(source_words) and norm(source_words[j]))
    return len(rw & side) / len(rw)


def source_overlap(source_words: list[str], positions: list[int]) -> float:
    src = set(norm(w) for w in source_words if norm(w))
    side = set(norm(source_words[j]) for j in positions if 0 <= j < len(source_words) and norm(source_words[j]))
    return len(src & side) / len(src) if src else 0.0


def bpe_len(tok, words: list[str]) -> int:
    return len(tok.encode(" ".join(words), add_special_tokens=False)) if words else 0


def approx_word_piece_lengths(tok, words: list[str]) -> tuple[list[int], list[int]]:
    first = []
    cont = []
    for w in words:
        first.append(len(tok.encode(w, add_special_tokens=False)))
        cont.append(len(tok.encode(" " + w, add_special_tokens=False)))
    return first, cont


def approx_bpe_for_positions(first_lens: list[int], cont_lens: list[int], positions: list[int]) -> int:
    if not positions:
        return 0
    total = first_lens[positions[0]]
    for p in positions[1:]:
        total += cont_lens[p]
    return total


def candidate_random_spans(n_src: int, lengths: list[int], rng: random.Random, tries: int = 320) -> list[list[int]]:
    """Generate nonoverlapping sorted same-length span candidates."""
    candidates = []
    k = len(lengths)
    if k == 0:
        return candidates
    for _ in range(tries):
        order = list(range(k))
        # Use actual skeleton order but random starts; later sort by starts. This preserves multiset of span lengths.
        spans = []
        ok = True
        used = set()
        for L in lengths:
            if L > n_src:
                ok = False; break
            # try several starts with preference for broad coverage
            for _attempt in range(20):
                st = rng.randint(0, n_src - L)
                candidate = set(range(st, st+L))
                if not (candidate & used):
                    spans.append((st, st+L-1)); used |= candidate; break
            else:
                ok = False; break
        if not ok:
            continue
        pos = positions_from_spans(sorted(spans))
        if len(pos) == sum(lengths):
            candidates.append(pos)
    return candidates


def candidate_shift_skeleton(s_positions: list[int], n_src: int) -> list[list[int]]:
    if not s_positions:
        return []
    lo, hi = min(s_positions), max(s_positions)
    candidates = []
    for off in range(-lo, n_src-1-hi+1):
        if off == 0:
            continue
        pos = [p+off for p in s_positions]
        if min(pos) >= 0 and max(pos) < n_src:
            candidates.append(pos)
    return candidates


def candidate_local_replacements(source_words: list[str], s_positions: list[int], first_lens: list[int], cont_lens: list[int], rng: random.Random) -> list[list[int]]:
    """Replace S positions by same content/function and similar piece-count alternatives, preserving order approximately."""
    n = len(source_words)
    s_set = set(s_positions)
    out = []
    for radius in [4, 8, 16, n]:
        pos = []
        used = set()
        for idx, p in enumerate(s_positions):
            cls = is_content(source_words[p])
            target_piece = (first_lens[p] if idx == 0 else cont_lens[p])
            pool = [q for q in range(max(0, p-radius), min(n, p+radius+1))
                    if q not in used and q not in s_set and is_content(source_words[q]) == cls]
            if not pool:
                pool = [q for q in range(n) if q not in used and q not in s_set and is_content(source_words[q]) == cls]
            if not pool:
                pool = [p]
            pool.sort(key=lambda q: (abs((first_lens[q] if idx == 0 else cont_lens[q])-target_piece), abs(q-p), rng.random()))
            q = pool[0]
            pos.append(q); used.add(q)
        pos = sorted(pos)
        if len(pos) == len(s_positions):
            out.append(pos)
    return out


def make_candidates(row: dict[str, Any], tok, rng: random.Random) -> list[list[int]]:
    src_words = row["source_text"].split()
    n = len(src_words)
    s_pos = list(row.get("S_positions") or [])
    old_g = list(row.get("G_positions") or [])
    spans = spans_from_positions(s_pos)
    lengths = [en-st+1 for st, en in spans]
    cands: list[list[int]] = []
    if old_g: cands.append(sorted(old_g))
    if s_pos: cands.append(sorted(s_pos))
    cands.extend(candidate_shift_skeleton(s_pos, n))
    first_lens, cont_lens = approx_word_piece_lengths(tok, src_words)
    cands.extend(candidate_local_replacements(src_words, s_pos, first_lens, cont_lens, rng))
    cands.extend(candidate_random_spans(n, lengths, rng, tries=360))
    # Also generate content/function balanced random positions without span skeleton as a fallback.
    target_len = len(s_pos)
    s_nc = sum(1 for j in s_pos if is_content(src_words[j]))
    content = [j for j,w in enumerate(src_words) if is_content(w)]
    func = [j for j,w in enumerate(src_words) if not is_content(w)]
    for _ in range(120):
        if len(content) >= s_nc and len(func) >= target_len-s_nc:
            pos = sorted(rng.sample(content, s_nc) + rng.sample(func, target_len-s_nc))
            cands.append(pos)
    # Deduplicate and enforce target length.
    seen = set(); uniq = []
    for c in cands:
        if len(c) != target_len: continue
        if min(c, default=0) < 0 or max(c, default=-1) >= n: continue
        if len(set(c)) != len(c): continue
        t = tuple(sorted(c))
        if t in seen: continue
        seen.add(t); uniq.append(list(t))
    return uniq


def select_control(row: dict[str, Any], tok, mode: str) -> tuple[list[int], dict[str, Any]]:
    src_words = row["source_text"].split()
    s_pos = list(row.get("S_positions") or [])
    if not s_pos:
        return [], {"mode": mode, "n_candidates": 0, "fallback": "empty"}
    s_words = [src_words[j] for j in s_pos]
    s_bpe = bpe_len(tok, s_words)
    s_content = sum(1 for j in s_pos if is_content(src_words[j]))
    s_feat = pos_features(s_pos, len(src_words))
    s_compact = compact_overlap(src_words, s_pos, row["C_text"])
    rng = random.Random(int(hashlib.md5((mode + row["pair_id"]).encode()).hexdigest()[:8], 16))
    first_lens, cont_lens = approx_word_piece_lengths(tok, src_words)
    candidates = make_candidates(row, tok, rng)
    if not candidates:
        return list(row.get("G_positions") or []), {"mode": mode, "n_candidates": 0, "fallback": "old_G"}
    best = None
    best_meta = None
    for pos in candidates:
        words = [src_words[j] for j in pos]
        content = sum(1 for j in pos if is_content(src_words[j]))
        approx_bpe = approx_bpe_for_positions(first_lens, cont_lens, pos)
        # Exact BPE only for plausible candidates to save time.
        bdiff_approx = abs(approx_bpe - s_bpe)
        feat = pos_features(pos, len(src_words))
        span_diff = abs(feat["span_count"] - s_feat["span_count"])
        content_diff = abs(content - s_content)
        # compact overlap is measured for all candidates; used in Ga score, only reported in Gm.
        comp = compact_overlap(src_words, pos, row["C_text"])
        ident = 1 if tuple(pos) == tuple(s_pos) else 0
        # Matching score: heavily prioritize exact content count, BPE, span count, then position/gaps.
        match_score = (
            1000.0 * content_diff +
            4.0 * bdiff_approx +
            20.0 * span_diff +
            4.0 * abs(feat["mean"] - s_feat["mean"]) +
            2.0 * abs(feat["spread"] - s_feat["spread"]) +
            0.4 * abs(feat["max_gap"] - s_feat["max_gap"]) +
            0.2 * abs(feat["mean_gap"] - s_feat["mean_gap"]) +
            20.0 * ident
        )
        if mode == "Gm":
            # Do not optimize semantic overlap; deterministic tie-break uses position distance.
            semantic_term = 0.0
        else:
            # Explicit anti-retained control: prefer less compact overlap after interface match.
            semantic_term = 5.0 * comp
        score = match_score + semantic_term
        if best is None or score < best_meta["score"]:
            best = pos
            best_meta = {"score": score, "match_score": match_score, "semantic_term": semantic_term,
                         "approx_bpe": approx_bpe, "approx_bpe_diff": approx_bpe - s_bpe,
                         "content": content, "content_diff": content - s_content,
                         "span_count": feat["span_count"], "span_diff": feat["span_count"] - s_feat["span_count"],
                         "compact_overlap": comp, "s_compact_overlap": s_compact,
                         "n_candidates": len(candidates), "identical_to_S": bool(ident)}
    # Compute exact BPE for selected candidate.
    assert best is not None and best_meta is not None
    exact = bpe_len(tok, [src_words[j] for j in best])
    best_meta["exact_bpe"] = exact
    best_meta["exact_bpe_diff"] = exact - s_bpe
    return best, {"mode": mode, **best_meta}


def audit_rows(rows: list[dict[str, Any]], tok) -> dict[str, Any]:
    arms = ["C", "S", "G", "Gm", "Ga"]
    arm = {a: collections.defaultdict(list) for a in arms}
    pair_rows = []
    for r in rows:
        src_words = r["source_text"].split()
        rw_norms = set(norm(w) for w in r["C_text"].split() if norm(w))
        for a in arms:
            text = r[f"{a}_text"]
            words = text.split()
            ids = tok.encode(text, add_special_tokens=False) if text else []
            positions = r.get(f"{a}_positions")
            arm[a]["words"].append(len(words))
            arm[a]["bpe"].append(len(ids))
            arm[a]["bpe_per_word"].append(len(ids)/len(words) if words else 0.0)
            arm[a]["content_frac"].append(sum(1 for w in words if is_content(w))/len(words) if words else 0.0)
            if positions is not None:
                feat = pos_features(list(positions), len(src_words))
                arm[a]["span_count"].append(feat["span_count"])
                arm[a]["position_mean"].append(feat["mean"])
                arm[a]["position_spread"].append(feat["spread"])
                arm[a]["max_gap"].append(feat["max_gap"])
                arm[a]["mean_gap"].append(feat["mean_gap"])
                arm[a]["compact_overlap"].append(compact_overlap(src_words, list(positions), r["C_text"]))
                arm[a]["source_overlap"].append(source_overlap(src_words, list(positions)))
            else:
                arm[a]["span_count"].append(0)
                arm[a]["position_mean"].append(0.5)
                arm[a]["position_spread"].append(1.0)
                arm[a]["max_gap"].append(0.0)
                arm[a]["mean_gap"].append(0.0)
                side_norms = set(norm(w) for w in words if norm(w))
                src_norms = set(norm(w) for w in src_words if norm(w))
                arm[a]["compact_overlap"].append(1.0 if a == "C" else len(side_norms & rw_norms)/max(1,len(rw_norms)))
                arm[a]["source_overlap"].append(len(side_norms & src_norms)/max(1,len(src_norms)))
        pair_rows.append({
            "pair_id": r["pair_id"], "source_text": r["source_text"], "C_text": r["C_text"],
            "S_text": r["S_text"], "G_text": r["G_text"], "Gm_text": r["Gm_text"], "Ga_text": r["Ga_text"],
            "S_bpe": arm["S"]["bpe"][-1], "G_bpe": arm["G"]["bpe"][-1],
            "Gm_bpe": arm["Gm"]["bpe"][-1], "Ga_bpe": arm["Ga"]["bpe"][-1],
            "S_spans": arm["S"]["span_count"][-1], "G_spans": arm["G"]["span_count"][-1],
            "Gm_spans": arm["Gm"]["span_count"][-1], "Ga_spans": arm["Ga"]["span_count"][-1],
            "S_compact_overlap": arm["S"]["compact_overlap"][-1],
            "G_compact_overlap": arm["G"]["compact_overlap"][-1],
            "Gm_compact_overlap": arm["Gm"]["compact_overlap"][-1],
            "Ga_compact_overlap": arm["Ga"]["compact_overlap"][-1],
            "S_minus_Gm_compact_overlap": arm["S"]["compact_overlap"][-1] - arm["Gm"]["compact_overlap"][-1],
            "S_minus_Ga_compact_overlap": arm["S"]["compact_overlap"][-1] - arm["Ga"]["compact_overlap"][-1],
        })
    out: dict[str, Any] = {}
    for a in arms:
        out[f"arm_{a}"] = {
            "total_words": int(sum(arm[a]["words"])),
            "total_bpe": int(sum(arm[a]["bpe"])),
            "bpe_per_word": stats(arm[a]["bpe_per_word"]),
            "content_frac": stats(arm[a]["content_frac"]),
            "span_count": stats(arm[a]["span_count"]),
            "position_mean": stats(arm[a]["position_mean"]),
            "position_spread": stats(arm[a]["position_spread"]),
            "max_gap": stats(arm[a]["max_gap"]),
            "mean_gap": stats(arm[a]["mean_gap"]),
            "compact_overlap": stats(arm[a]["compact_overlap"]),
            "source_overlap": stats(arm[a]["source_overlap"]),
        }
    def diff_metric(a: str, b: str, key: str):
        return stats([x-y for x,y in zip(arm[a][key], arm[b][key])])
    out["matching"] = {
        "S_minus_G": {k: diff_metric("S", "G", k) for k in ["bpe", "bpe_per_word", "content_frac", "span_count", "position_mean", "position_spread", "compact_overlap"]},
        "S_minus_Gm": {k: diff_metric("S", "Gm", k) for k in ["bpe", "bpe_per_word", "content_frac", "span_count", "position_mean", "position_spread", "compact_overlap"]},
        "S_minus_Ga": {k: diff_metric("S", "Ga", k) for k in ["bpe", "bpe_per_word", "content_frac", "span_count", "position_mean", "position_spread", "compact_overlap"]},
        "C_minus_S": {k: diff_metric("C", "S", k) for k in ["bpe", "bpe_per_word", "content_frac", "compact_overlap"]},
    }
    out["examples"] = {
        "Gm_best_interface_high_sep": sorted(pair_rows, key=lambda x: (abs(x["S_bpe"]-x["Gm_bpe"]), abs(x["S_spans"]-x["Gm_spans"]), -x["S_minus_Gm_compact_overlap"]))[:12],
        "Gm_remaining_bad_bpe": sorted(pair_rows, key=lambda x: abs(x["S_bpe"]-x["Gm_bpe"]), reverse=True)[:12],
        "Ga_high_sep": sorted(pair_rows, key=lambda x: x["S_minus_Ga_compact_overlap"], reverse=True)[:12],
    }
    return out


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH), local_files_only=True, use_fast=True)
    rows = list(read_jsonl(IN_PATH))
    new_rows = []
    meta = {"Gm": collections.Counter(), "Ga": collections.Counter()}
    exact_bpe_diffs = {"Gm": [], "Ga": []}
    for idx, r in enumerate(rows):
        r = dict(r)
        src_words = r["source_text"].split()
        for mode in ["Gm", "Ga"]:
            pos, m = select_control(r, tok, mode)
            r[f"{mode}_positions"] = pos
            r[f"{mode}_text"] = " ".join(src_words[j] for j in pos)
            r[f"{mode}_meta"] = m
            meta[mode]["identical_to_S"] += int(m.get("identical_to_S", False))
            meta[mode]["fallback_old_G"] += int(m.get("fallback") == "old_G")
            meta[mode]["fallback_empty"] += int(m.get("fallback") == "empty")
            if "exact_bpe_diff" in m:
                exact_bpe_diffs[mode].append(m["exact_bpe_diff"])
        new_rows.append(r)
        if (idx+1) % 2000 == 0:
            print(f"processed {idx+1}/{len(rows)}", flush=True)
    out_jsonl = OUT_DIR / "marginal_corpora_csg_gmatched.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {
        "status": "INTERFACE_MATCHED_G_BUILT",
        "meaning": "Adds Gm/Ga controls to research C/S/G. This is a CPU repair attempt, not training evidence.",
        "input": str(IN_PATH),
        "input_sha256": sha256_file(IN_PATH),
        "output": str(out_jsonl),
        "output_sha256": sha256_file(out_jsonl),
        "tokenizer": str(TOK_PATH),
        "generation_meta": {
            mode: {**{k: int(v) for k,v in meta[mode].items()}, "exact_bpe_diff": stats(exact_bpe_diffs[mode])}
            for mode in ["Gm", "Ga"]
        },
        "audit": audit_rows(new_rows, tok),
        "elapsed_sec": round(time.time()-t0, 1),
    }
    out_audit = OUT_DIR / "interface_matched_g_audit.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    short = {
        "status": audit["status"], "output_sha256": audit["output_sha256"],
        "S_total_bpe": audit["audit"]["arm_S"]["total_bpe"],
        "G_total_bpe": audit["audit"]["arm_G"]["total_bpe"],
        "Gm_total_bpe": audit["audit"]["arm_Gm"]["total_bpe"],
        "Ga_total_bpe": audit["audit"]["arm_Ga"]["total_bpe"],
        "S_minus_G_bpe_mean": audit["audit"]["matching"]["S_minus_G"]["bpe"]["mean"],
        "S_minus_Gm_bpe_mean": audit["audit"]["matching"]["S_minus_Gm"]["bpe"]["mean"],
        "S_minus_Ga_bpe_mean": audit["audit"]["matching"]["S_minus_Ga"]["bpe"]["mean"],
        "S_minus_G_span_mean": audit["audit"]["matching"]["S_minus_G"]["span_count"]["mean"],
        "S_minus_Gm_span_mean": audit["audit"]["matching"]["S_minus_Gm"]["span_count"]["mean"],
        "S_minus_Ga_span_mean": audit["audit"]["matching"]["S_minus_Ga"]["span_count"]["mean"],
        "S_minus_Gm_compact_overlap_mean": audit["audit"]["matching"]["S_minus_Gm"]["compact_overlap"]["mean"],
        "S_minus_Ga_compact_overlap_mean": audit["audit"]["matching"]["S_minus_Ga"]["compact_overlap"]["mean"],
        "elapsed_sec": audit["elapsed_sec"],
    }
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
