#!/usr/bin/env python3
"""research: what training material does word-mean MLM reweight?

The research objective gives each selected whole-word group equal credit instead of
each selected BPE token equal credit.  This CPU-only script samples actual legal
training batches with the same WWM selection path and reports which source classes,
word-length bins, and coarse lexical cue classes receive more or less loss credit
under word-mean than under token-mean.

It uses training text only.  No official evaluation text, model forward pass, GPU,
or corpus/tokenizer change is involved.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "data/wordmean_reweighting_profile"

CUE_SETS = {
    "spatial_state": {"above","across","against","along","around","at","away","behind","below","beneath","beside","between","beyond","down","from","here","in","inside","into","near","next","off","on","onto","outside","over","there","through","toward","towards","under","up","where","within","left","right","front","back","top","bottom","side","middle","north","south","east","west","place","position","location","room","house","street","city","country","river"},
    "causal_temporal": {"after","again","already","always","before","because","cause","caused","causes","during","early","eventually","finally","first","if","later","next","never","now","once","since","soon","then","therefore","until","when","while","why","will","would","could","should","may","might","must","still","time","times","ago","today","tomorrow","yesterday","happen","happened","happens","result","results","reason","change","changed","become","became"},
    "physical_dynamics": {"move","moves","moved","moving","fall","falls","fell","fallen","drop","dropped","push","pushed","pull","pulled","hit","hits","break","broke","broken","turn","turned","run","ran","walk","walked","fly","flew","drive","driven","open","opened","close","closed","stop","stopped","start","started","grow","grew","increase","increased","decrease","decreased","accelerate","accelerating","slow","slowing","rise","rose","raise","raised","lower","lowered","carry","carried","hold","held","put","take","took","give","gave","make","made","use","used","work","worked"},
    "material_property": {"hot","cold","warm","cool","hard","soft","heavy","light","big","small","large","little","long","short","high","low","strong","weak","dry","wet","clean","dirty","red","blue","green","white","black","bright","dark","new","old","young","same","different","good","bad","better","worse","full","empty","water","gas","air","wood","metal","stone","paper","food","body","blood","ice","fire","energy","temperature","size","weight","shape","color"},
    "quantity_measure": {"one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand","million","billion","many","much","more","most","less","least","few","several","number","amount","part","parts","percent","half","year","years","month","months","day","days","hour","hours","minute","minutes","second","seconds","age","old"},
    "mental_social_dialogue": {"ask","asked","answer","answered","call","called","hear","heard","listen","look","looked","read","say","said","says","see","saw","seen","speak","spoke","talk","talked","tell","told","think","thought","know","knew","believe","believed","want","wanted","need","needed","like","liked","love","loved","feel","felt","friend","family","mother","father","child","children","man","woman","people","person","boy","girl","sir","mr","mrs","miss","you","i","he","she","we","they","me","him","her","us","them"},
    "negation_modality": {"no","not","never","nothing","none","nobody","cannot","can't","dont","don't","didn't","doesn't","isn't","aren't","wasn't","weren't","won't","wouldn't","couldn't","shouldn't","may","might","must","should","would","could","perhaps","maybe","if","unless","whether","possible","impossible","true","false"},
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:]\d+)*")


def load_base_trainer():
    spec = importlib.util.spec_from_file_location("compact_experience_base_step062_reweight", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source_class(src: str) -> str:
    s = str(src)
    if s == "qwen_pair_packed":
        return "inherited_qwen_pair_packed"
    if s == "cleanqwen_fineweb_compact_view_reinvest":
        return "fineweb_source_compact_view_pair"
    if s.startswith("neutral_cleanqwen_topup"):
        return "neutral_topup"
    if "::" in s:
        head, tail = s.split("::", 1)
        return tail or head
    return s or "unknown"


def choose_batches(total_batches: int, front_batches: int, stride_batches: int) -> list[int]:
    out = set(range(min(front_batches, total_batches)))
    if stride_batches > 0 and total_batches > 0:
        out.update(int(round(x)) for x in np.linspace(0, total_batches - 1, stride_batches))
    return sorted(i for i in out if 0 <= i < total_batches)


def group_features_from_text(tokenizer, ex_text: str, source: str, max_len: int) -> dict[int, dict[str, Any]]:
    enc = tokenizer(ex_text, add_special_tokens=False, truncation=True, max_length=max_len, padding="max_length", return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    gid = -1
    special = set(tokenizer.all_special_ids)
    for i, tid in enumerate(ids):
        if tid in special or offsets[i] == (0, 0):
            continue
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if gid < 0 or (tok is not None and (str(tok).startswith("Ġ") or str(tok).startswith("▁"))) or i == 0:
            gid += 1
        groups[gid].append(tuple(offsets[i]))
    feats = {}
    src = source_class(source)
    for gid2, spans in groups.items():
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        text = ex_text[lo:hi].strip()
        low_words = [m.group(0).lower().strip("'") for m in WORD_RE.finditer(text)]
        cats = []
        for cat, lex in CUE_SETS.items():
            if any(w in lex for w in low_words):
                cats.append(cat)
        alpha = "".join(ch for ch in text if ch.isalpha())
        alen = len(alpha)
        if alen <= 3:
            len_bin = "short_0_3"
        elif alen <= 7:
            len_bin = "medium_4_7"
        elif alen <= 12:
            len_bin = "long_8_12"
        else:
            len_bin = "verylong_13plus"
        feats[gid2] = {
            "text": text[:80],
            "source_class": src,
            "cue_categories": cats,
            "has_any_cue": bool(cats),
            "has_digit": any(ch.isdigit() for ch in text),
            "has_upper": any(ch.isupper() for ch in text),
            "len_bin": len_bin,
        }
    return feats


def add_share(agg: dict[str, dict[str, float]], key: str, n_groups: int, n_tokens: int, G: int, T: int) -> None:
    if G <= 0 or T <= 0:
        return
    rec = agg.setdefault(key, {"wordmean_share_sum": 0.0, "tokenmean_share_sum": 0.0, "groups": 0.0, "tokens": 0.0, "batches_present": 0.0})
    rec["wordmean_share_sum"] += n_groups / G
    rec["tokenmean_share_sum"] += n_tokens / T
    rec["groups"] += n_groups
    rec["tokens"] += n_tokens
    rec["batches_present"] += 1.0


def summarize_agg(agg: dict[str, dict[str, float]], n_batches: int) -> list[dict[str, Any]]:
    rows = []
    for k, v in agg.items():
        wm = v["wordmean_share_sum"] / n_batches
        tm = v["tokenmean_share_sum"] / n_batches
        rows.append({
            "category": k,
            "wordmean_credit_share_avg_batch": wm,
            "tokenmean_credit_share_avg_batch": tm,
            "delta_wordmean_minus_tokenmean_share": wm - tm,
            "relative_ratio_wordmean_over_tokenmean": wm / tm if tm > 0 else None,
            "selected_groups": int(v["groups"]),
            "selected_tokens": int(v["tokens"]),
            "batches_present": int(v["batches_present"]),
        })
    rows.sort(key=lambda r: abs(r["delta_wordmean_minus_tokenmean_share"]), reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-word-exposure", type=int, default=80_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--front-batches", type=int, default=16)
    ap.add_argument("--stride-batches", type=int, default=64)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    examples, total_words, total_rows, sample_rows = base.load_examples_jsonl(TRAIN_FILE, args.max_word_exposure)
    total_batches = math.ceil(len(examples) / args.batch_size)
    batch_indices = choose_batches(total_batches, args.front_batches, args.stride_batches)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=total_batches)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    agg: dict[str, dict[str, float]] = {}
    examples_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    total_selected_groups = 0
    total_selected_tokens = 0
    batch_records = []
    t0 = time.time()
    for jj, bi in enumerate(batch_indices, 1):
        st = bi * args.batch_size
        en = min(len(examples), st + args.batch_size)
        items = [dataset[i] for i in range(st, en)]
        batch = base.collate(items)
        input_ids = batch["input_ids"][:, :args.seq_length].contiguous()
        attention_mask = batch["attention_mask"][:, :args.seq_length].contiguous()
        word_group = batch["word_group"][:, :args.seq_length].contiguous()
        state.current_step = bi
        masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
        del masked_inputs
        selected = labels != -100
        G = 0
        T = int(selected.sum().item())
        cat_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for rb, ex_idx in enumerate(range(st, en)):
            groups = word_group[rb]
            sel_groups = torch.unique(groups[selected[rb] & (groups >= 0)]).tolist()
            if not sel_groups:
                continue
            feats = group_features_from_text(tokenizer, examples[ex_idx].text, examples[ex_idx].source, args.max_seq_length)
            for gid in sel_groups:
                k = int(((groups == int(gid)) & selected[rb]).sum().item())
                if k <= 0:
                    continue
                G += 1
                feat = feats.get(int(gid), {"source_class": source_class(examples[ex_idx].source), "cue_categories": [], "has_any_cue": False, "has_digit": False, "has_upper": False, "len_bin": "unknown", "text": ""})
                keys = [
                    f"source::{feat['source_class']}",
                    f"bpe_len::{min(k, 8)}" if k < 8 else "bpe_len::8plus",
                    f"alpha_len::{feat['len_bin']}",
                    f"has_any_cue::{feat['has_any_cue']}",
                    f"has_digit::{feat['has_digit']}",
                    f"has_upper::{feat['has_upper']}",
                ]
                for cat in feat.get("cue_categories", []):
                    keys.append(f"cue::{cat}")
                if not feat.get("cue_categories"):
                    keys.append("cue::none")
                for key in keys:
                    cat_counts[key][0] += 1
                    cat_counts[key][1] += k
                    if len(examples_by_key[key]) < 2000:
                        txt = str(feat.get("text", "")).strip()
                        if txt:
                            examples_by_key[key][txt] += 1
        for key, (ng, nt) in cat_counts.items():
            add_share(agg, key, ng, nt, G, T)
        total_selected_groups += G
        total_selected_tokens += T
        batch_records.append({"batch_index0": bi, "rows": en-st, "selected_groups": G, "selected_tokens": T, "mean_tokens_per_selected_group": T/G if G else None})
        if jj == 1 or jj % 16 == 0 or jj == len(batch_indices):
            print(json.dumps({"event": "reweight_sample_progress", "sampled_batches": jj, "total": len(batch_indices), "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    rows = summarize_agg(agg, len(batch_indices))
    for row in rows:
        exs = examples_by_key.get(row["category"], Counter()).most_common(8)
        row["example_group_texts"] = [{"text": t, "count_in_sample": c} for t, c in exs]
    summary = {
        "status": "WORDMEAN_REWEIGHTING_PROFILE",
        "created_utc": now_utc(),
        "scope": "Training-text-only sampled actual WWM batches; no model forward, no official evaluation text, no GPU.",
        "train_file": str(TRAIN_FILE),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "max_word_exposure": args.max_word_exposure,
        "loaded_examples": len(examples),
        "loaded_words": sum(ex.words for ex in examples),
        "total_batches": total_batches,
        "sampled_batches": len(batch_indices),
        "batch_indices": batch_indices,
        "selected_groups": total_selected_groups,
        "selected_tokens": total_selected_tokens,
        "mean_tokens_per_selected_group": total_selected_tokens / total_selected_groups if total_selected_groups else None,
        "category_rows": rows,
        "batch_records": batch_records,
        "interpretation": "Word-mean credit favors categories concentrated in one-piece/short/common groups and reduces credit for categories carried by multi-piece/long/name-like groups. This profile helps decide whether a score change is broad late-benefit strengthening or a redistribution away from support-thin lexical material.",
    }
    out_json = out_dir / "wordmean_reweighting_profile.json"
    out_md = out_dir / "wordmean_reweighting_profile.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research word-mean reweighting profile", "", summary["scope"], "", f"Sampled `{len(batch_indices)}` batches, `{total_selected_groups}` selected groups, `{total_selected_tokens}` selected tokens; mean selected tokens/group `{summary['mean_tokens_per_selected_group']:.4f}`.", "", "## Largest average-batch credit-share shifts", "", "| category | wordmean share | tokenmean share | Δ share | ratio | groups | tokens | examples |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows[:60]:
        exs = "; ".join(x["text"] for x in r["example_group_texts"][:5])
        ratio = "NA" if r["relative_ratio_wordmean_over_tokenmean"] is None else f"{r['relative_ratio_wordmean_over_tokenmean']:.3f}"
        lines.append(f"| {r['category']} | {100*r['wordmean_credit_share_avg_batch']:.3f}% | {100*r['tokenmean_credit_share_avg_batch']:.3f}% | {100*r['delta_wordmean_minus_tokenmean_share']:.3f}% | {ratio} | {r['selected_groups']} | {r['selected_tokens']} | {exs} |")
    lines += ["", "## Interpretation", "", summary["interpretation"], "", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "sampled_batches": len(batch_indices), "selected_groups": total_selected_groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
