#!/usr/bin/env python3
"""research: relation-cue natural candidate-contrast pilot.

This stricter pilot addresses the noise in the first natural hard-negative readout.
It mines only legal-corpus rows containing simple relation/physical/comparative
cue words, samples single-token content targets near those cues, filters target and
alternative tokens by corpus word frequency, and scores scale1.75 80M vs matched
legal16k 80M. It is a readout/mining pilot only; it does not train and does not use
official evaluation rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
SCRIPT_DIR = ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import helper functions and model paths from the first pilot. Importing it sets
# writable HF cache variables before transformers are imported.
import natural_candidate_contrast_pilot as basepilot  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

OUT = ROOT / "experiments/archive/representation_and_objectives/data/relation_cue_candidate_contrast_pilot"
NOTE = ROOT / "research/notes/representation_and_objectives/relation_cue_candidate_contrast_pilot.md"
CORPUS = basepilot.CORPUS
SCALE80 = basepilot.SCALE80
BASE80 = basepilot.BASE80

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
CUE_RE = re.compile(
    r"\b(above|below|under|over|inside|outside|into|onto|near|beside|between|through|around|left|right|front|behind|"
    r"push|pull|open|close|enter|exit|fall|drop|throw|break|bounce|move|turn|face|facing|hold|holds|holding|"
    r"larger|smaller|bigger|shorter|longer|higher|lower|colder|hotter|warmer|heavier|lighter|"
    r"before|after|earlier|later|first|last|next|previous|more|less|same|different)\b",
    re.I,
)
EXTRA_STOP = set("""
well even additionally note notes take please value values one two three four five six seven eight nine ten yes yeah okay ok hmm uh um would could should also already really actually maybe probably just still rather however therefore whereas because although though since while into onto upon along within without among between through during before after next previous first last same different other another much many more most less least part parts thing things something anything everything nothing someone somebody anyone anybody everyone everybody there here then now today yesterday tomorrow monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december
""".split())
STOP = set(basepilot.STOPWORDS) | EXTRA_STOP


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def word_counts(max_lines: int = 0) -> Counter[str]:
    c: Counter[str] = Counter()
    with CORPUS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            text = json.loads(line)["text"].lower()
            c.update(m.group(0).strip("-'") for m in WORD_RE.finditer(text))
    return c


def acceptable_word(w: str, counts: Counter[str], min_freq: int) -> bool:
    s = w.lower().strip("-'")
    if s in STOP:
        return False
    if len(s) < 4 or len(s) > 14:
        return False
    if counts[s] < min_freq:
        return False
    # reject obvious speaker/transcript artifacts and possessive fragments
    if s in {"xxx", "www", "mhm", "gonna", "wanna", "gotta", "hafta", "yeah"}:
        return False
    return bool(re.fullmatch(r"[a-z][a-z'\-]{3,13}", s))


def sample_relation_positions(tokenizer, counts: Counter[str], max_rows: int, max_positions: int, cue_window_chars: int, min_word_freq: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positions: list[dict[str, Any]] = []
    special = set(tokenizer.all_special_ids)
    with CORPUS.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if max_rows and line_no >= max_rows:
                break
            if len(positions) >= max_positions:
                break
            rec = json.loads(line)
            text = rec["text"]
            cue_spans = [(m.start(), m.end(), m.group(0).lower()) for m in CUE_RE.finditer(text)]
            if not cue_spans:
                continue
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256, add_special_tokens=True)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            cands = []
            for m in WORD_RE.finditer(text):
                w = m.group(0)
                lw = w.lower().strip("-'")
                if not acceptable_word(lw, counts, min_word_freq):
                    continue
                # avoid selecting the cue word itself as target; targets are content around cue.
                if CUE_RE.fullmatch(lw):
                    continue
                nearest = min((abs(((m.start()+m.end())//2) - ((a+b)//2)), cue) for a, b, cue in cue_spans)
                if nearest[0] > cue_window_chars:
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
                a, b = offsets[ti]
                piece = text[a:b].lower().strip("-'")
                if piece != lw:
                    continue
                cands.append((ti, m.start(), m.end(), w, nearest[1], nearest[0]))
            if not cands:
                continue
            rng.shuffle(cands)
            for ti, a, b, w, cue, dist in cands[:2]:
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
                    "target_word": w,
                    "char_span": [a, b],
                    "nearest_cue": cue,
                    "distance_to_cue_chars": dist,
                    "target_word_freq": counts[w.lower().strip("-'")],
                })
                if len(positions) >= max_positions:
                    break
    return positions


def clean_alt(tokenizer, token_id: int, counts: Counter[str], min_freq: int) -> str | None:
    s = basepilot.clean_decoded(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))
    if s is None:
        return None
    if not acceptable_word(s, counts, min_freq):
        return None
    return s


def select_alt_freq(local_logits: torch.Tensor, target_id: int, tokenizer, counts: Counter[str], min_alt_freq: int, topk: int, special_ids: set[int]) -> tuple[int | None, str | None, int | None, int | None]:
    vals, idx = torch.topk(local_logits, k=min(topk + 80, local_logits.numel()))
    target_dec = clean_alt(tokenizer, target_id, counts, 1)
    for rank, tid in enumerate(idx.tolist(), start=1):
        if tid == target_id or tid in special_ids:
            continue
        dec = clean_alt(tokenizer, tid, counts, min_alt_freq)
        if dec is None:
            continue
        if target_dec is not None and dec == target_dec:
            continue
        return int(tid), dec, rank, counts[dec]
    return None, None, None, None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-rows", type=int, default=64740)
    ap.add_argument("--max-positions", type=int, default=160)
    ap.add_argument("--cue-window-chars", type=int, default=90)
    ap.add_argument("--min-word-freq", type=int, default=10)
    ap.add_argument("--min-alt-freq", type=int, default=10)
    ap.add_argument("--local-radius", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "word_count_start", "corpus": rel(CORPUS), "utc": now()}), flush=True)
    counts = word_counts()
    print(json.dumps({"event": "word_count_done", "types": len(counts), "tokens": sum(counts.values()), "utc": now()}), flush=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "load_models", "device": str(device), "utc": now()}), flush=True)
    tok = AutoTokenizer.from_pretrained(SCALE80, trust_remote_code=True)
    scale = AutoModelForMaskedLM.from_pretrained(SCALE80, trust_remote_code=True).to(device).eval()
    base = AutoModelForMaskedLM.from_pretrained(BASE80, trust_remote_code=True).to(device).eval()
    positions = sample_relation_positions(tok, counts, args.max_rows, args.max_positions, args.cue_window_chars, args.min_word_freq, seed=1571)
    print(json.dumps({"event": "positions", "n": len(positions), "utc": now()}), flush=True)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("mask token missing")
    full_ids=[]; full_pos=[]; local_ids=[]; local_pos=[]
    for p in positions:
        ids=list(p["input_ids"]); ti=int(p["target_index"]); ids[ti]=mask_id
        full_ids.append(ids); full_pos.append(ti)
        loc,lti=basepilot.make_local_ids(p,tok,args.local_radius); loc[lti]=mask_id
        local_ids.append(loc); local_pos.append(lti)
    print(json.dumps({"event": "forward_scale", "n": len(positions), "utc": now()}), flush=True)
    flogits=basepilot.batch_logits(scale, full_ids, full_pos, tok, device, args.batch_size)
    llogits=basepilot.batch_logits(scale, local_ids, local_pos, tok, device, args.batch_size)
    print(json.dumps({"event": "forward_base", "n": len(positions), "utc": now()}), flush=True)
    blogits=basepilot.batch_logits(base, full_ids, full_pos, tok, device, args.batch_size)
    rows=[]
    specials=set(tok.all_special_ids)
    for p,fl,ll,bl in zip(positions,flogits,llogits,blogits):
        tid=int(p["target_id"])
        alt_id, alt, alt_rank, alt_freq = select_alt_freq(ll, tid, tok, counts, args.min_alt_freq, args.topk, specials)
        if alt_id is None:
            continue
        scale_local=float(ll[tid].item()-ll[alt_id].item())
        scale_full=float(fl[tid].item()-fl[alt_id].item())
        base_full=float(bl[tid].item()-bl[alt_id].item())
        rows.append({
            "line_no": p["line_no"], "example_id": p.get("example_id"), "source": p.get("source"),
            "nearest_cue": p["nearest_cue"], "distance_to_cue_chars": p["distance_to_cue_chars"],
            "target_word": p["target_word"].lower(), "target_word_freq": p["target_word_freq"],
            "alternative_word": alt, "alternative_word_freq": alt_freq, "local_alt_rank": alt_rank,
            "scale_local_margin_target_minus_alt": scale_local,
            "scale_full_margin_target_minus_alt": scale_full,
            "base_full_margin_target_minus_alt": base_full,
            "context_gain_full_minus_local": scale_full-scale_local,
            "scale_minus_base_full_margin": scale_full-base_full,
            "scale_full_target_rank": basepilot.rank_of(fl,tid), "scale_full_alt_rank": basepilot.rank_of(fl,alt_id),
            "base_full_target_rank": basepilot.rank_of(bl,tid), "base_full_alt_rank": basepilot.rank_of(bl,alt_id),
            "scale_full_error_vs_alt": scale_full < 0,
            "unsaturated_abs_full_margin_le_0p5": abs(scale_full) <= 0.5,
            "context_decides_local_alt_full_gold": scale_local < 0 and scale_full > 0.5,
            "scale_weaker_than_base_by_0p5": scale_full-base_full < -0.5,
            "snippet": p["text"][:450],
        })
    def summar(vals): return basepilot.summarize(vals)
    summary={
        "status":"RELATION_CUE_CANDIDATE_CONTRAST_PILOT_DONE",
        "created_utc":now(),
        "description":"Relation-cue/frequency-filtered natural candidate contrast readout from legal corpus only; no training, no official eval text.",
        "corpus":rel(CORPUS), "scale_model":rel(SCALE80), "base_model":rel(BASE80),
        "params":vars(args), "sampled_positions":len(positions), "usable_candidate_rows":len(rows),
        "counts":{
            "scale_full_errors_vs_alt":sum(r["scale_full_error_vs_alt"] for r in rows),
            "unsaturated_full_abs_margin_le_0p5":sum(r["unsaturated_abs_full_margin_le_0p5"] for r in rows),
            "context_decides_local_alt_full_gold":sum(r["context_decides_local_alt_full_gold"] for r in rows),
            "scale_weaker_than_base_by_0p5":sum(r["scale_weaker_than_base_by_0p5"] for r in rows),
        },
        "margin_summary":{
            "scale_full_target_minus_alt":summar([r["scale_full_margin_target_minus_alt"] for r in rows]),
            "scale_local_target_minus_alt":summar([r["scale_local_margin_target_minus_alt"] for r in rows]),
            "context_gain_full_minus_local":summar([r["context_gain_full_minus_local"] for r in rows]),
            "scale_minus_base_full_margin":summar([r["scale_minus_base_full_margin"] for r in rows]),
        },
        "cue_counts":dict(Counter(r["nearest_cue"] for r in rows)),
        "source_counts":dict(Counter(r["source"] for r in rows)),
        "interpretation":"If this stricter relation-cue set still contains many scale-specific weak or wrong target-vs-alt margins, a short candidate-contrast fork is plausible. If rows are noisy or mostly target-token frequency artifacts, natural counterfactual/context transformation should be prioritized instead.",
    }
    rows_csv=OUT/"relation_cue_candidate_rows.csv"
    hard_csv=OUT/"relation_cue_candidate_hard_subset.csv"
    out_json=OUT/"relation_cue_candidate_contrast_summary.json"
    write_csv(rows_csv, rows)
    hard=[r for r in rows if r["scale_full_error_vs_alt"] or r["unsaturated_abs_full_margin_le_0p5"] or r["scale_weaker_than_base_by_0p5"]]
    hard=sorted(hard, key=lambda r:(r["scale_minus_base_full_margin"], r["scale_full_margin_target_minus_alt"]))[:100]
    write_csv(hard_csv, hard)
    summary["rows_csv"]=rel(rows_csv); summary["hard_subset_csv"]=rel(hard_csv)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join([
        "# research relation-cue natural candidate-contrast pilot\n\n",
        f"Summary JSON: `{rel(out_json)}`; rows: `{rel(rows_csv)}`; hard subset: `{rel(hard_csv)}`.\n\n",
        "This pilot filters the first natural candidate readout to relation/physical/comparative cue contexts and legal-corpus frequency-supported single-token words.\n\n",
        "## Counts\n", json.dumps(summary["counts"], indent=2), "\n\n",
        "## Margins\n", json.dumps(summary["margin_summary"], indent=2), "\n\n",
        "## Cue/source distribution\n", json.dumps({"cue_counts":summary["cue_counts"],"source_counts":summary["source_counts"]}, indent=2), "\n\n",
        "## Research implication\n", summary["interpretation"], "\n"
    ]), encoding="utf-8")
    print(json.dumps({"status":summary["status"],"summary":rel(out_json),"rows":len(rows),"counts":summary["counts"],"note":rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
