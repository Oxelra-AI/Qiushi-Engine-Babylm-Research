#!/usr/bin/env python3
"""research: fractional whole-word credit profile.

The global word-mean objective (alpha=1) failed as a wholesale replacement for
token-mean MLM (alpha=0), but it produced a distinctive COMPS/GlobalPIQA-nonparallel
movement.  This CPU-only analysis measures how intermediate group-credit exponents
would reallocate actual selected WWM training targets on the frozen legal compact-view
reinvest stream:

  token weight within a selected word group of k BPE pieces is proportional k^{-alpha}
  => selected group mass is proportional k^{1-alpha}.

It uses the same sampled batches and masking RNG style as research.  It performs no
model forward, training, official evaluation, corpus change, or route launch.  Its
purpose is to decide whether a later small/scheduled fractional credit branch would
be a materially different optimization hypothesis from the closed alpha=1 global
word-mean run.
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
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "data/fractional_credit_profile"

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
    spec = importlib.util.spec_from_file_location("compact_experience_base_step069_fractional_credit", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        alpha_text = "".join(ch for ch in text if ch.isalpha())
        alen = len(alpha_text)
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


def summarize(vals: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "min": float(min(vals)), "max": float(max(vals)), "std": float(statistics.pstdev(vals))}


def prefix_of(category: str) -> str:
    return category.split("::", 1)[0] if "::" in category else category


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
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = sha256_file(TRAIN_FILE)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if 0.0 not in alphas:
        alphas.insert(0, 0.0)
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    examples, total_words, total_rows, sample_rows = base.load_examples_jsonl(TRAIN_FILE, args.max_word_exposure)
    total_batches = math.ceil(len(examples) / args.batch_size)
    batch_indices = choose_batches(total_batches, args.front_batches, args.stride_batches)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=total_batches)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    category_mass = {a: defaultdict(float) for a in alphas}
    category_share_sum = {a: defaultdict(float) for a in alphas}
    category_batches = {a: defaultdict(int) for a in alphas}
    category_groups = defaultdict(int)
    category_tokens = defaultdict(int)
    examples_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    batch_distortion = {a: defaultdict(list) for a in alphas if a != 0.0}
    selected_groups = 0
    selected_tokens = 0
    selected_k_values: list[int] = []
    batch_records = []

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
        group_entries: list[tuple[int, list[str], str]] = []
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
                group_entries.append((k, keys, str(feat.get("text", ""))))
                selected_groups += 1
                selected_tokens += k
                selected_k_values.append(k)
                for key in keys:
                    category_groups[key] += 1
                    category_tokens[key] += k
                    if len(examples_by_key[key]) < 2000 and feat.get("text"):
                        examples_by_key[key][str(feat["text"])] += 1
        total_mass = {a: sum((k ** (1.0 - a)) for k, _, _ in group_entries) for a in alphas}
        per_alpha_cat_mass = {a: defaultdict(float) for a in alphas}
        for k, keys, _txt in group_entries:
            for a in alphas:
                w = k ** (1.0 - a)
                for key in keys:
                    per_alpha_cat_mass[a][key] += w
                    category_mass[a][key] += w
        # Shares are averaged by batch, following research rather than by one giant pass.
        all_keys = set().union(*[set(per_alpha_cat_mass[a].keys()) for a in alphas]) if group_entries else set()
        for a in alphas:
            den = total_mass[a]
            if den <= 0:
                continue
            for key in all_keys:
                sh = per_alpha_cat_mass[a].get(key, 0.0) / den
                category_share_sum[a][key] += sh
                category_batches[a][key] += 1
        # Family-level L1 distortion vs token-mean within each batch.
        den0 = total_mass[0.0]
        if den0 > 0:
            for a in alphas:
                if a == 0.0 or total_mass[a] <= 0:
                    continue
                by_prefix = defaultdict(lambda: defaultdict(float))
                keys2 = set(per_alpha_cat_mass[0.0].keys()) | set(per_alpha_cat_mass[a].keys())
                for key in keys2:
                    pref = prefix_of(key)
                    by_prefix[pref][key] = (per_alpha_cat_mass[a].get(key, 0.0) / total_mass[a]) - (per_alpha_cat_mass[0.0].get(key, 0.0) / den0)
                for pref, vals in by_prefix.items():
                    batch_distortion[a][pref].append(0.5 * sum(abs(v) for v in vals.values()))
        batch_records.append({
            "batch_index0": bi,
            "rows": en - st,
            "selected_groups": len(group_entries),
            "selected_tokens": sum(k for k, _, _ in group_entries),
            "mean_tokens_per_selected_group": (sum(k for k, _, _ in group_entries) / len(group_entries)) if group_entries else None,
        })
        if jj == 1 or jj % 16 == 0 or jj == len(batch_indices):
            print(json.dumps({"event": "fractional_credit_progress", "sampled_batches": jj, "total": len(batch_indices), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    n_batches = len(batch_indices)
    categories = sorted(set().union(*[set(category_share_sum[a].keys()) for a in alphas]))
    category_rows = []
    for key in categories:
        row: dict[str, Any] = {
            "category": key,
            "prefix": prefix_of(key),
            "selected_groups": int(category_groups[key]),
            "selected_tokens": int(category_tokens[key]),
            "example_group_texts": [{"text": t, "count_in_sample": c} for t, c in examples_by_key.get(key, Counter()).most_common(8)],
        }
        base_share = category_share_sum[0.0].get(key, 0.0) / n_batches
        for a in alphas:
            sh = category_share_sum[a].get(key, 0.0) / n_batches
            tag = str(a).replace(".", "p")
            row[f"share_alpha_{tag}"] = sh
            row[f"delta_vs_tokenmean_alpha_{tag}"] = sh - base_share
            row[f"ratio_vs_tokenmean_alpha_{tag}"] = (sh / base_share) if base_share > 0 else None
        category_rows.append(row)
    category_rows.sort(key=lambda r: abs(r.get("delta_vs_tokenmean_alpha_1p0", 0.0)), reverse=True)

    family_distortion = []
    for a in alphas:
        if a == 0.0:
            continue
        for pref, vals in sorted(batch_distortion[a].items()):
            family_distortion.append({"alpha": a, "family": pref, "half_l1_share_shift_vs_tokenmean_by_batch": summarize(vals)})

    key_categories = [
        "bpe_len::1", "bpe_len::2", "bpe_len::3", "bpe_len::4", "bpe_len::8plus",
        "has_upper::True", "has_digit::True", "source::childes", "source::fineweb_source_compact_view_pair",
        "source::inherited_qwen_pair_packed", "cue::mental_social_dialogue", "cue::spatial_state", "cue::physical_dynamics", "cue::material_property", "cue::none",
    ]
    key_rows = [r for r in category_rows if r["category"] in key_categories]
    summary = {
        "status": "FRACTIONAL_CREDIT_PROFILE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Training-text-only CPU analysis of actual selected WWM groups; no model forward, no official evaluation, no GPU.",
        "inputs": {
            "train_file": str(TRAIN_FILE),
            "train_sha256": train_sha,
            "tokenizer_dir": str(TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "max_word_exposure": args.max_word_exposure,
            "loaded_examples": len(examples),
            "loaded_words": sum(ex.words for ex in examples),
            "total_batches": total_batches,
            "sampled_batches": n_batches,
            "batch_indices": batch_indices,
            "train_rng_seed": args.train_rng_seed,
            "alphas": alphas,
        },
        "selected_summary": {
            "selected_groups": int(selected_groups),
            "selected_tokens": int(selected_tokens),
            "mean_tokens_per_selected_group": selected_tokens / selected_groups if selected_groups else None,
            "selected_group_piece_count": summarize([float(x) for x in selected_k_values]),
        },
        "family_distortion": family_distortion,
        "key_category_rows": key_rows,
        "category_rows": category_rows,
        "batch_records": batch_records,
        "scientific_interpretation": [
            "Alpha=0 is token-mean and alpha=1 is the closed global word-mean objective; intermediate alpha values are quantitatively different only if they materially reduce the content/name/digit downweighting while preserving a small amount of word-unit normalization.",
            "This profile cannot predict BabyLM scores. It is useful for deciding whether a later fractional/late credit fork is a targeted optimization-mechanism test rather than an unprincipled rerun of the failed global word-mean route.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "fractional_credit_profile.json"
    out_md = out_dir / "fractional_credit_profile.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research fractional whole-word credit profile",
        "",
        summary["scope"],
        "",
        f"Sampled `{n_batches}` batches, `{selected_groups}` selected groups, `{selected_tokens}` selected tokens; mean selected pieces/group `{summary['selected_summary']['mean_tokens_per_selected_group']:.4f}`.",
        "",
        "## Key category share shifts",
        "",
        "Shares are average-batch loss-credit shares. Alpha 0 is token-mean; alpha 1 is global word-mean.",
        "",
        "| category | tokenmean a=0 | a=0.25 Δ | a=0.5 Δ | a=0.75 Δ | wordmean a=1 Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in key_rows:
        lines.append(
            f"| {r['category']} | {100*r['share_alpha_0p0']:.3f}% | "
            f"{100*r.get('delta_vs_tokenmean_alpha_0p25', 0.0):+.3f}% | "
            f"{100*r.get('delta_vs_tokenmean_alpha_0p5', 0.0):+.3f}% | "
            f"{100*r.get('delta_vs_tokenmean_alpha_0p75', 0.0):+.3f}% | "
            f"{100*r.get('delta_vs_tokenmean_alpha_1p0', 0.0):+.3f}% |"
        )
    lines += ["", "## Family-level half-L1 share shift versus tokenmean", "", "| alpha | family | mean half-L1 shift |", "|---:|---|---:|"]
    for fd in family_distortion:
        if fd["family"] in {"bpe_len", "has_upper", "has_digit", "source", "cue"}:
            lines.append(f"| {fd['alpha']:.2f} | {fd['family']} | {100*fd['half_l1_share_shift_vs_tokenmean_by_batch']['mean']:.3f}% |")
    lines += ["", "## Interpretation"]
    for item in summary["scientific_interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "sampled_batches": n_batches,
        "selected_groups": selected_groups,
        "selected_tokens": selected_tokens,
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
