#!/usr/bin/env python3
"""research: natural candidate-contrast mining pilot on legal corpus text.

Purpose: test whether the existing compliant 10M compact-view corpus contains
natural, non-synthetic positions where a globally plausible alternative competes
with the corpus token, and whether broader context changes that competition. This
is a readout/mining pilot only. It does not train, does not inspect official eval
rows, and must not be confused with a submission route.

The pilot samples content-word positions, masks the target token, uses the
scale1.75 80M model to select a clean top local-context alternative, then compares
logit margins target-minus-alternative under local and full context. When the
matched research legal16k base is included, it also asks whether scale1.75 weakened
natural target-vs-alternative margins relative to the base.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import random
import re
import statistics
import time
from collections import Counter
from typing import Any

# Set writable caches before importing transformers.
def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
OUT = ROOT / "experiments/archive/representation_and_objectives/data/natural_candidate_contrast_pilot"
CACHE = OUT / "hf_cache"
for key, sub in {
    "HF_HOME": "hf_home",
    "HF_HUB_CACHE": "hf_home/hub",
    "HUGGINGFACE_HUB_CACHE": "hf_home/hub",
    "TRANSFORMERS_CACHE": "transformers",
    "HF_MODULES_CACHE": "modules",
    "HF_DATASETS_CACHE": "datasets",
}.items():
    p = CACHE / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

CORPUS = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
SCALE80 = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M"
BASE80 = ROOT / "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M"
NOTE = ROOT / "research/notes/representation_and_objectives/natural_candidate_contrast_pilot.md"

STOPWORDS = set("""
a an the and or but if then than as at by for from in into is it its of on onto over under to with without within across after before during about above below between through this that these those there here when where while who whom whose which what why how be been being are was were will would should could can may might must do does did doing done have has had having you your we our they their he his she her i my me mine us them him himself herself itself themselves not no nor so such very more most less least other another same just only own out up down off back much many any each every some all both either neither
""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def clean_decoded(tok: str) -> str | None:
    s = tok.strip().lower()
    if not s:
        return None
    s = s.replace("Ġ", "").replace("▁", "").strip()
    if s in STOPWORDS:
        return None
    if not re.fullmatch(r"[a-z][a-z'\-]{2,14}", s):
        return None
    return s


def read_corpus_positions(tokenizer, max_rows: int, max_positions: int, row_stride: int, positions_per_row: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positions: list[dict[str, Any]] = []
    special = set(tokenizer.all_special_ids)
    with CORPUS.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no % row_stride != 0:
                continue
            if line_no >= max_rows:
                break
            if len(positions) >= max_positions:
                break
            rec = json.loads(line)
            text = rec["text"]
            # Fast tokenizers provide offsets; if unavailable this will fail loudly.
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256, add_special_tokens=True)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            if len(ids) < 16:
                continue
            # Identify single-token word spans. This keeps target and alternative one-token, simplifying contrast.
            candidates = []
            for m in WORD_RE.finditer(text):
                word = m.group(0)
                lw = word.lower().strip("-'")
                if lw in STOPWORDS or len(lw) < 4:
                    continue
                hits = []
                for ti, (a, b) in enumerate(offsets):
                    if ids[ti] in special:
                        continue
                    if a >= m.start() and b <= m.end() and b > a:
                        hits.append(ti)
                if len(hits) != 1:
                    continue
                ti = hits[0]
                # Require offset to cover most of the visible word, avoiding byte-BPE fragments.
                a, b = offsets[ti]
                piece = text[a:b]
                if piece.lower().strip("-'") != lw:
                    continue
                candidates.append((ti, m.start(), m.end(), word))
            if not candidates:
                continue
            # Prefer spread-out content tokens by hash/random but deterministic under seed.
            rng.shuffle(candidates)
            chosen = candidates[:positions_per_row]
            for ti, a, b, word in chosen:
                positions.append({
                    "line_no": line_no + 1,
                    "example_id": rec.get("example_id"),
                    "source": rec.get("source"),
                    "words_in_row": rec.get("words"),
                    "text": text,
                    "input_ids": ids,
                    "offsets": offsets,
                    "target_index": ti,
                    "target_id": ids[ti],
                    "target_word": word,
                    "char_span": [a, b],
                })
                if len(positions) >= max_positions:
                    break
    return positions


def make_local_ids(pos: dict[str, Any], tokenizer, radius: int) -> tuple[list[int], int]:
    ids = pos["input_ids"]
    ti = pos["target_index"]
    specials = set(tokenizer.all_special_ids)
    # Keep nearest non-special tokens around target; rewrap with model special tokens.
    left = max(1, ti - radius)
    right = min(len(ids) - 1, ti + radius + 1)
    window = ids[left:right]
    local_target = ti - left
    if tokenizer.cls_token_id is not None:
        window = [tokenizer.cls_token_id] + window
        local_target += 1
    if tokenizer.sep_token_id is not None:
        window = window + [tokenizer.sep_token_id]
    return window, local_target


def batch_logits(model, batch_ids: list[list[int]], mask_positions: list[int], tokenizer, device: torch.device, batch_size: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in range(0, len(batch_ids), batch_size):
        sub = batch_ids[i:i + batch_size]
        subpos = mask_positions[i:i + batch_size]
        maxlen = max(len(x) for x in sub)
        ids = torch.full((len(sub), maxlen), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(sub), maxlen), dtype=torch.long, device=device)
        for j, x in enumerate(sub):
            ids[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=device)
            attn[j, :len(x)] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=attn).logits
        for j, mp in enumerate(subpos):
            out.append(logits[j, mp].detach().cpu())
    return out


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    val = logits[token_id].item()
    return int((logits > val).sum().item()) + 1


def select_alt(local_logits: torch.Tensor, target_id: int, tokenizer, topk: int, special_ids: set[int]) -> tuple[int | None, str | None, int | None]:
    vals, idx = torch.topk(local_logits, k=min(topk + 20, local_logits.numel()))
    rank = 0
    for tid in idx.tolist():
        rank += 1
        if tid == target_id or tid in special_ids:
            continue
        dec = clean_decoded(tokenizer.decode([tid], clean_up_tokenization_spaces=False))
        if dec is None:
            continue
        target_dec = clean_decoded(tokenizer.decode([target_id], clean_up_tokenization_spaces=False))
        if target_dec is not None and dec == target_dec:
            continue
        return int(tid), dec, rank
    return None, None, None


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    sv = sorted(vals)
    def q(p: float) -> float:
        return sv[min(len(sv)-1, max(0, int(round(p*(len(sv)-1)))))]
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p10": q(0.1), "p90": q(0.9), "min": sv[0], "max": sv[-1]}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Keep only scalar/string columns.
    scalar_rows = []
    for r in rows:
        scalar_rows.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scalar_rows[0].keys()))
        w.writeheader()
        w.writerows(scalar_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-positions", type=int, default=160)
    ap.add_argument("--row-stride", type=int, default=97)
    ap.add_argument("--positions-per-row", type=int, default=1)
    ap.add_argument("--local-radius", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--include-base", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "load_start", "model": rel(SCALE80), "device": str(device), "utc": now()}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(SCALE80, trust_remote_code=True)
    scale = AutoModelForMaskedLM.from_pretrained(SCALE80, trust_remote_code=True).to(device)
    scale.eval()
    base = None
    if args.include_base:
        print(json.dumps({"event": "load_base_start", "model": rel(BASE80), "utc": now()}), flush=True)
        base = AutoModelForMaskedLM.from_pretrained(BASE80, trust_remote_code=True).to(device)
        base.eval()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask token")
    special_ids = set(tokenizer.all_special_ids)

    print(json.dumps({"event": "sample_positions_start", "corpus": rel(CORPUS), "utc": now()}), flush=True)
    positions = read_corpus_positions(tokenizer, args.max_rows, args.max_positions, args.row_stride, args.positions_per_row, seed=157)
    if not positions:
        raise RuntimeError("no positions sampled")
    print(json.dumps({"event": "positions_sampled", "n": len(positions), "utc": now()}), flush=True)

    full_ids = []
    full_pos = []
    local_ids = []
    local_pos = []
    for p in positions:
        ids = list(p["input_ids"])
        ti = int(p["target_index"])
        ids[ti] = mask_id
        full_ids.append(ids)
        full_pos.append(ti)
        loc, lti = make_local_ids(p, tokenizer, args.local_radius)
        loc[lti] = mask_id
        local_ids.append(loc)
        local_pos.append(lti)

    print(json.dumps({"event": "scale_forward_start", "pairs": len(positions), "utc": now()}), flush=True)
    full_logits = batch_logits(scale, full_ids, full_pos, tokenizer, device, args.batch_size)
    local_logits = batch_logits(scale, local_ids, local_pos, tokenizer, device, args.batch_size)
    base_full_logits = None
    if base is not None:
        print(json.dumps({"event": "base_forward_start", "pairs": len(positions), "utc": now()}), flush=True)
        base_full_logits = batch_logits(base, full_ids, full_pos, tokenizer, device, args.batch_size)

    rows: list[dict[str, Any]] = []
    for p, flog, llog, blog in zip(positions, full_logits, local_logits, base_full_logits or [None] * len(positions)):
        tid = int(p["target_id"])
        alt_id, alt_str, alt_local_rank = select_alt(llog, tid, tokenizer, args.topk, special_ids)
        if alt_id is None:
            continue
        target_dec = clean_decoded(tokenizer.decode([tid], clean_up_tokenization_spaces=False)) or p["target_word"].lower()
        local_margin = float(llog[tid].item() - llog[alt_id].item())
        full_margin = float(flog[tid].item() - flog[alt_id].item())
        rec = {
            "line_no": p["line_no"],
            "example_id": p.get("example_id"),
            "source": p.get("source"),
            "target_word": p["target_word"],
            "target_token_decoded": target_dec,
            "alternative_token_decoded": alt_str,
            "target_id": tid,
            "alternative_id": alt_id,
            "local_alt_rank": alt_local_rank,
            "scale_local_margin_target_minus_alt": local_margin,
            "scale_full_margin_target_minus_alt": full_margin,
            "context_gain_full_minus_local": full_margin - local_margin,
            "scale_full_target_rank": rank_of(flog, tid),
            "scale_local_target_rank": rank_of(llog, tid),
            "scale_full_alt_rank": rank_of(flog, alt_id),
            "is_scale_full_error_vs_alt": full_margin < 0.0,
            "is_unsaturated_full_abs_margin_le_0p5": abs(full_margin) <= 0.5,
            "is_context_decides_local_alt_full_gold": local_margin < 0.0 and full_margin > 0.5,
            "snippet": p["text"][:500],
        }
        if blog is not None:
            base_margin = float(blog[tid].item() - blog[alt_id].item())
            rec.update({
                "base_full_margin_target_minus_alt": base_margin,
                "scale_minus_base_full_margin": full_margin - base_margin,
                "base_full_target_rank": rank_of(blog, tid),
                "base_full_alt_rank": rank_of(blog, alt_id),
                "is_scale_weaker_than_base_by_0p5": (full_margin - base_margin) < -0.5,
            })
        rows.append(rec)

    margins = [r["scale_full_margin_target_minus_alt"] for r in rows]
    local_margins = [r["scale_local_margin_target_minus_alt"] for r in rows]
    cgains = [r["context_gain_full_minus_local"] for r in rows]
    summary = {
        "status": "NATURAL_CANDIDATE_CONTRAST_PILOT_DONE",
        "created_utc": now(),
        "description": "Natural corpus one-token hard-negative pilot using scale1.75 80M local top alternatives; not training and not official eval.",
        "corpus": rel(CORPUS),
        "scale_model": rel(SCALE80),
        "matched_base_model": rel(BASE80) if args.include_base else None,
        "device": str(device),
        "params": vars(args),
        "sampled_positions": len(positions),
        "usable_candidate_rows": len(rows),
        "counts": {
            "scale_full_errors_vs_alt": sum(r["is_scale_full_error_vs_alt"] for r in rows),
            "unsaturated_full_abs_margin_le_0p5": sum(r["is_unsaturated_full_abs_margin_le_0p5"] for r in rows),
            "context_decides_local_alt_full_gold": sum(r["is_context_decides_local_alt_full_gold"] for r in rows),
            "scale_weaker_than_base_by_0p5": sum(r.get("is_scale_weaker_than_base_by_0p5", False) for r in rows),
        },
        "margin_summary": {
            "scale_full_target_minus_alt": summarize(margins),
            "scale_local_target_minus_alt": summarize(local_margins),
            "context_gain_full_minus_local": summarize(cgains),
        },
        "local_alt_rank_counts": dict(Counter(r["local_alt_rank"] for r in rows)),
        "interpretation": "Useful evidence for a natural candidate-contrast fork would be many rows where a clean alternative is locally competitive, full-context margin is unsaturated or wrong for scale1.75, and the same pair is weaker than the matched base. If rows are mostly saturated or base is equally weak, this pilot argues against ordinary natural hard-negative contrast as the next expensive route.",
    }
    if args.include_base and rows:
        deltas = [r["scale_minus_base_full_margin"] for r in rows]
        summary["margin_summary"]["scale_minus_base_full_margin"] = summarize(deltas)

    rows_csv = OUT / "natural_candidate_rows.csv"
    hard_csv = OUT / "natural_candidate_hard_subset.csv"
    out_json = OUT / "natural_candidate_contrast_summary.json"
    write_csv(rows_csv, rows)
    hard = [r for r in rows if r["is_scale_full_error_vs_alt"] or r["is_unsaturated_full_abs_margin_le_0p5"] or r.get("is_scale_weaker_than_base_by_0p5", False)]
    hard = sorted(hard, key=lambda r: (r.get("scale_minus_base_full_margin", 0.0), r["scale_full_margin_target_minus_alt"]))[:80]
    write_csv(hard_csv, hard)
    summary["rows_csv"] = rel(rows_csv)
    summary["hard_subset_csv"] = rel(hard_csv)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# research natural candidate-contrast pilot\n\n",
        f"Summary JSON: `{rel(out_json)}`; row CSV: `{rel(rows_csv)}`; hard subset: `{rel(hard_csv)}`.\n\n",
        "This is a corpus readout/mining pilot only. It samples legal compact-view corpus tokens and never uses official evaluation text for training.\n\n",
        "## Key counts\n",
        json.dumps(summary["counts"], indent=2) + "\n\n",
        "## Margin summaries\n",
        json.dumps(summary["margin_summary"], indent=2) + "\n\n",
        "## Research implication\n",
        "A short repair fork is scientifically worth building only if hard natural rows are abundant and differ from ordinary MLM: local alternatives must be plausible, full-context scale1.75 margins should be unsaturated or wrong, and matched-base comparison should show scale1.75-specific weakening. Otherwise this points away from generic candidate contrast and toward a stronger representation or counterfactual source.\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(out_json), "rows": len(rows), "counts": summary["counts"], "note": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
