#!/usr/bin/env python3
"""research anchor-margin conditioned alpha-sweep census.

Scientific purpose
------------------
research found a coherent86 private residual whose item decisions across alpha
0/0.5/0.75/1.0 are almost all monotonic: thousands of anchor-wrong activations
and thousands of anchor-correct damages.  independent review identified the decisive
zero/few-training measurement: condition those activation/damage populations on
the frozen anchor's own log-likelihood margin.  If damage is concentrated at low
anchor margins while high-margin anchor decisions are protected, a label-free
anchor-confidence gate or Jacobian-complement mechanism is plausible.  If damage
is spread through the margin distribution, the separability premise is weak.

This script reconstructs the exact official-style MLM candidate scores for the
changed alpha items.  It can build a manifest only (no model inference) or load a
model and score the selected items.  It uses the same task data and MLM scoring
rule as the official sentence_zero_shot pipeline:
  - sum log P(completion tokens | sentence with one completion token masked at a time)
  - for GlobalPIQA, divide by number of completion tokens.

No benchmark labels are used for training; labels are used here only to read the
official evaluation item's gold alternative and classify already-observed alpha
patterns.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any

import numpy as np

ROOT = pathlib.Path(".").resolve()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, read_jsonl, norm_text, uid_group  # noqa: E402

ALPHA_PATHS = OrderedDict([
    ("a0", ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json"),
    ("a0p5", WORKSPACE / "data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json"),
    ("a0p75", WORKSPACE / "data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"),
    ("a1", WORKSPACE / "data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json"),
])
DEFAULT_MODEL = ROOT / "experiments/archive/representation_and_objectives/training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022/hf_model/chck_82M"
FALLBACK_MODEL = ROOT / "experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/hf_repo_bundle"
DEFAULT_OUT = WORKSPACE / "data/anchor_margin_alpha_census"
LENGTH_NORMALIZED_TASKS = {"GlobalPIQA"}
PATTERN_CLASSES = {
    "0001": "activation_mono_late1",
    "0011": "activation_mono_late075_1",
    "0111": "activation_mono_all_scaled",
    "0100": "activation_nonmono_0p5_only",
    "0110": "activation_nonmono_0p5_0p75_only",
    "0010": "activation_nonmono_0p75_only",
    "1110": "damage_mono_late1",
    "1100": "damage_mono_late075_1",
    "1000": "damage_mono_all_scaled",
    "1011": "damage_nonmono_0p5_only",
    "1101": "damage_nonmono_0p75_only",
    "1001": "damage_nonmono_0p5_0p75",
}
MONO_ACTIVATION = {"0001", "0011", "0111"}
MONO_DAMAGE = {"1110", "1100", "1000"}


@dataclass
class MarginItem:
    item_id: str
    column: str
    uid: str
    group: str
    pattern: str
    pattern_class: str
    anchor_correct: bool
    label_index: int
    gold_text: str
    candidates: list[str]
    completions: list[str]
    length_normalized: bool
    meta: dict[str, Any]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_rows(column: str) -> dict[str, dict[str, Any]]:
    out = {}
    for label, path in ALPHA_PATHS.items():
        rows, _ = PayloadLoader(path).load_column(column)
        out[label] = {r.item_id: r for r in rows}
    return out


def blimp_like_candidate_map(loader: PayloadLoader, column: str) -> dict[str, dict[str, Any]]:
    rec = loader.rec(column)
    data_dir = ROOT / rec["data_path"]
    out = {}
    counters = Counter()
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for raw in read_jsonl(file_path):
            if "field" in raw:
                uid = str(raw["UID"])
            else:
                uid = file_path.stem
            idx = counters[uid]; counters[uid] += 1
            good = norm_text(raw["sentence_good"])
            bad = norm_text(raw.get("sentence_bad", ""))
            out[f"{column}:{uid}:{idx}"] = {
                "uid": uid,
                "group": uid,
                "candidates": [good, bad],
                "completions": [good, bad],
                "label_index": 0,
                "gold_text": good,
                "meta": {"file": file_path.stem, "raw_keys": sorted(raw.keys())},
            }
    return out


def ewok_candidate_map(loader: PayloadLoader) -> dict[str, dict[str, Any]]:
    rec = loader.rec("EWoK")
    data_dir = ROOT / rec["data_path"]
    out = {}
    counters = Counter()
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for raw in read_jsonl(file_path):
            uid = str(raw["Domain"])
            idx = counters[uid]; counters[uid] += 1
            good = norm_text(" ".join([raw["Context1"], raw["Target1"]]))
            bad = norm_text(" ".join([raw["Context2"], raw["Target1"]]))
            # Official decode_ewok uses completion = " " + Target1; the leading
            # space is part of the suffix span used for MLM PLL masking.
            completion = " " + norm_text(raw["Target1"])
            out[f"EWoK:{uid}:{idx}"] = {
                "uid": uid,
                "group": uid,
                "candidates": [good, bad],
                "completions": [completion, completion],
                "label_index": 0,
                "gold_text": good,
                "meta": {"Domain": uid, "file": file_path.stem},
            }
    return out


def entity_candidate_map(loader: PayloadLoader) -> dict[str, dict[str, Any]]:
    rec = loader.rec("Entity")
    data_dir = ROOT / rec["data_path"]
    out = {}
    counters = Counter()
    for file_path in sorted(data_dir.glob("*.jsonl")):
        family = file_path.stem
        for raw in read_jsonl(file_path):
            if any("nothing" in str(opt) for opt in raw["options"]):
                continue
            uid = f"{family}_{int(raw['numops'])}_ops"
            idx = counters[uid]; counters[uid] += 1
            # Official decode_entity_tracking uses input_prefix + option as the
            # candidate sentence and the raw option string as the completion.
            prefix = str(raw["input_prefix"])
            opts = [str(o) for o in raw["options"]]
            candidates = [prefix + o for o in opts]
            out[f"Entity:{uid}:{idx}"] = {
                "uid": uid,
                "group": uid,
                "candidates": candidates,
                "completions": opts,
                "label_index": 0,
                "gold_text": norm_text(opts[0]),
                "meta": {"family": family, "numops": int(raw["numops"]), "raw_keys": sorted(raw.keys())},
            }
    return out


def comps_candidate_map(loader: PayloadLoader) -> dict[str, dict[str, Any]]:
    rec = loader.rec("COMPS")
    data_dir = ROOT / rec["data_path"]
    file_to_uid = {"comps_base.jsonl":"base", "comps_wugs.jsonl":"wugs", "comps_wugs_dist-before.jsonl":"wugs_dist_before", "comps_wugs_dist-in-between.jsonl":"wugs_dist_in_between"}
    out = {}
    counters = Counter()
    for filename, uid in file_to_uid.items():
        path = data_dir / filename
        for raw in read_jsonl(path):
            idx = counters[uid]; counters[uid] += 1
            good = norm_text(" ".join([raw["prefix_acceptable"], raw["property_phrase"]]))
            bad = norm_text(" ".join([raw["prefix_unacceptable"], raw["property_phrase"]]))
            completion = norm_text(raw["property_phrase"])
            out[f"COMPS:{uid}:{idx}"] = {
                "uid": uid,
                "group": uid,
                "candidates": [good, bad],
                "completions": [completion, completion],
                "label_index": 0,
                "gold_text": good,
                "meta": {"file": filename, "uid": uid},
            }
    return out


def globalpiqa_candidate_map(loader: PayloadLoader) -> dict[str, dict[str, Any]]:
    out = {}
    for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = loader.rec(sub)
        data_dir = ROOT / rec["data_path"]
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                uid = str(raw["example_id"])
                # Official GlobalPIQA decode uses a 4-way parallel task and a
                # 2-way nonparallel task: sentences = prompt + solution_i,
                # completions = " " + solution_i, label indexes the gold solution.
                num_solutions = 4 if sub == "GlobalPIQA_parallel" else 2
                solutions = [str(raw[f"solution{i}"]) for i in range(num_solutions)]
                label = int(raw["label"])
                if not (0 <= label < num_solutions):
                    raise ValueError(f"Bad GlobalPIQA label {label} for {sub} item {uid}")
                prompt = str(raw["prompt"])
                candidates = [" ".join([prompt, sol]) for sol in solutions]
                completions = [" " + sol for sol in solutions]
                out[f"GlobalPIQA:{sub}:{uid}"] = {
                    "uid": uid,
                    "group": sub,
                    "candidates": candidates,
                    "completions": completions,
                    "label_index": label,
                    "gold_text": norm_text(completions[label]),
                    "meta": {"sub": sub, "file": file_path.stem, "num_solutions": num_solutions, "raw_keys": sorted(raw.keys())},
                }
    return out


def build_candidate_maps(loader: PayloadLoader) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "BLiMP": blimp_like_candidate_map(loader, "BLiMP"),
        "Supplement": blimp_like_candidate_map(loader, "Supplement"),
        "EWoK": ewok_candidate_map(loader),
        "Entity": entity_candidate_map(loader),
        "COMPS": comps_candidate_map(loader),
        "GlobalPIQA": globalpiqa_candidate_map(loader),
    }


def build_manifest(limit_per_class: int | None = None) -> list[MarginItem]:
    loader0 = PayloadLoader(ALPHA_PATHS["a0"])
    cand_maps = build_candidate_maps(loader0)
    items: list[MarginItem] = []
    per_class_seen = Counter()
    for col in DISCRETE_COLUMNS:
        maps = alpha_rows(col)
        common = sorted(set.intersection(*(set(m) for m in maps.values())))
        for item_id in common:
            vals = {lab: bool(maps[lab][item_id].correct) for lab in ALPHA_PATHS}
            bits = "".join("1" if vals[lab] else "0" for lab in ALPHA_PATHS)
            if bits not in PATTERN_CLASSES:
                continue
            pclass = PATTERN_CLASSES[bits]
            if limit_per_class is not None and per_class_seen[pclass] >= limit_per_class:
                continue
            raw = cand_maps[col].get(item_id)
            if raw is None:
                continue
            base_row = maps["a0"][item_id]
            items.append(MarginItem(
                item_id=item_id,
                column=col,
                uid=base_row.uid,
                group=str(uid_group(col, base_row)),
                pattern=bits,
                pattern_class=pclass,
                anchor_correct=bool(vals["a0"]),
                label_index=int(raw["label_index"]),
                gold_text=str(raw["gold_text"]),
                candidates=list(raw["candidates"]),
                completions=list(raw["completions"]),
                length_normalized=(col in LENGTH_NORMALIZED_TASKS),
                meta=dict(raw.get("meta", {})),
            ))
            per_class_seen[pclass] += 1
    return items


def setup_env(cache_root: pathlib.Path, gpu: int | None) -> None:
    for k, p in {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TMPDIR": cache_root / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def score_sentence_mlm(model, tokenizer, text: str, completion: str, device: str) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    start_char_idx = len(text) - len(completion)
    phrase_indices = []
    targets = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_indices.append(i)
            targets.append(input_ids[i])
    if not phrase_indices:
        return float("nan"), 0
    scores = []
    bsz = 64
    mask_id = tokenizer.mask_token_id
    with torch.no_grad():
        for j in range(0, len(phrase_indices), bsz):
            batch_idx = phrase_indices[j:j+bsz]
            batch_targets = targets[j:j+bsz]
            toks = []
            masks = []
            for pos in batch_idx:
                cur = list(input_ids)
                cur[pos] = mask_id
                toks.append(cur)
                masks.append(list(attn))
            # pad manually through tokenizer.pad
            padded = tokenizer.pad({"input_ids": toks, "attention_mask": masks}, return_tensors="pt")
            inp = padded["input_ids"].to(device)
            am = padded["attention_mask"].to(device)
            logits = model(input_ids=inp, attention_mask=am).logits
            row = torch.arange(len(batch_idx), device=device)
            pos = torch.tensor(batch_idx, device=device)
            target = torch.tensor(batch_targets, device=device)
            lp = F.log_softmax(logits[row, pos], dim=-1)[row, target]
            scores.extend([float(x) for x in lp.detach().cpu().tolist()])
    return float(sum(scores)), len(scores)


def score_items(items: list[MarginItem], model_path: pathlib.Path, gpu: int | None, out_dir: pathlib.Path) -> list[dict[str, Any]]:
    # Set cache paths before importing transformers; otherwise dynamic-module
    # caching can bind to the shared read-only HF cache and fail for custom code.
    setup_env(out_dir / "runtime_cache", gpu)
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() and gpu is not None else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    results = []
    t0 = time.time()
    for n, item in enumerate(items, start=1):
        cand_scores = []
        token_counts = []
        for sent, comp in zip(item.candidates, item.completions):
            s, ntok = score_sentence_mlm(model, tokenizer, sent, comp, device)
            if item.length_normalized and ntok > 0:
                s = s / ntok
            cand_scores.append(s)
            token_counts.append(ntok)
        gold_score = cand_scores[item.label_index]
        other_scores = [s for i, s in enumerate(cand_scores) if i != item.label_index and math.isfinite(s)]
        best_other = max(other_scores) if other_scores else float("nan")
        margin = gold_score - best_other if math.isfinite(gold_score) and math.isfinite(best_other) else float("nan")
        pred_idx = int(np.nanargmax(cand_scores)) if any(math.isfinite(x) for x in cand_scores) else -1
        rec = asdict(item)
        rec.update({
            "anchor_candidate_scores": cand_scores,
            "completion_token_counts": token_counts,
            "anchor_gold_score": gold_score,
            "anchor_best_other_score": best_other,
            "anchor_margin_gold_minus_best_other": margin,
            "anchor_pred_index_from_rescore": pred_idx,
            "anchor_correct_from_rescore": bool(pred_idx == item.label_index),
        })
        results.append(rec)
        if n % 500 == 0:
            print(json.dumps({"event": "margin_progress", "n": n, "total": len(items), "elapsed_sec": round(time.time()-t0, 1)}), flush=True)
    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_margins(rows):
        return np.array([float(r["anchor_margin_gold_minus_best_other"]) for r in rows if math.isfinite(float(r["anchor_margin_gold_minus_best_other"]))], dtype=float)
    classes = defaultdict(list)
    for r in records:
        classes[r["pattern_class"]].append(r)
    groups = {
        "monotonic_activation": [r for r in records if r["pattern"] in MONO_ACTIVATION],
        "monotonic_damage": [r for r in records if r["pattern"] in MONO_DAMAGE],
        "nonmonotonic_changed": [r for r in records if r["pattern"] not in MONO_ACTIVATION and r["pattern"] not in MONO_DAMAGE],
    }
    out = {"n_records": len(records), "pattern_counts": dict(Counter(r["pattern"] for r in records)), "pattern_class_counts": {k: len(v) for k, v in classes.items()}}
    by_group = {}
    for name, rows in groups.items():
        m = finite_margins(rows)
        if len(m):
            by_group[name] = {
                "n": len(rows), "finite": int(len(m)),
                "mean": float(np.mean(m)), "median": float(np.median(m)), "std": float(np.std(m)),
                "q05_q25_q50_q75_q95": [float(np.percentile(m, q)) for q in [5, 25, 50, 75, 95]],
                "frac_margin_below_0": float(np.mean(m < 0.0)),
                "frac_abs_margin_below_0p1": float(np.mean(np.abs(m) < 0.1)),
                "frac_abs_margin_below_0p5": float(np.mean(np.abs(m) < 0.5)),
                "frac_abs_margin_below_1p0": float(np.mean(np.abs(m) < 1.0)),
            }
        else:
            by_group[name] = {"n": len(rows), "finite": 0}
    # simple effect sizes and separability: can |margin| or signed margin separate damage from activation?
    act = finite_margins(groups["monotonic_activation"])
    dmg = finite_margins(groups["monotonic_damage"])
    if len(act) and len(dmg):
        by_group["activation_vs_damage"] = {
            "damage_minus_activation_mean_margin": float(np.mean(dmg) - np.mean(act)),
            "damage_minus_activation_median_margin": float(np.median(dmg) - np.median(act)),
            "abs_damage_minus_abs_activation_mean": float(np.mean(np.abs(dmg)) - np.mean(np.abs(act))),
        }
        # AUC for using signed margin or |margin| to identify damage over activation.
        labels = np.concatenate([np.zeros(len(act)), np.ones(len(dmg))])
        for score_name, scores in {"signed_margin": np.concatenate([act, dmg]), "abs_margin": np.concatenate([np.abs(act), np.abs(dmg)])}.items():
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(scores)+1)
            # tie-average ranks
            # simple stable correction: use scipy if available
            try:
                from scipy.stats import rankdata
                ranks = rankdata(scores, method="average")
            except Exception:
                pass
            n1 = len(dmg); n0 = len(act)
            rank_sum_pos = float(np.sum(ranks[labels == 1]))
            auc = (rank_sum_pos - n1 * (n1 + 1) / 2) / (n0 * n1)
            by_group["activation_vs_damage"][f"auc_damage_by_{score_name}"] = float(auc)
    out["margin_summary"] = by_group

    # By column summaries to catch COMPS domination.
    by_col = {}
    for col in sorted(set(r["column"] for r in records)):
        rows = [r for r in records if r["column"] == col]
        col_act = finite_margins([r for r in rows if r["pattern"] in MONO_ACTIVATION])
        col_dmg = finite_margins([r for r in rows if r["pattern"] in MONO_DAMAGE])
        by_col[col] = {
            "n": len(rows),
            "pattern_counts": dict(Counter(r["pattern"] for r in rows)),
            "activation_n": int(len(col_act)),
            "damage_n": int(len(col_dmg)),
            "activation_median_margin": None if len(col_act) == 0 else float(np.median(col_act)),
            "damage_median_margin": None if len(col_dmg) == 0 else float(np.median(col_dmg)),
            "activation_abs_margin_lt_0p5": None if len(col_act) == 0 else float(np.mean(np.abs(col_act) < 0.5)),
            "damage_abs_margin_lt_0p5": None if len(col_dmg) == 0 else float(np.mean(np.abs(col_dmg) < 0.5)),
        }
    out["by_column"] = by_col
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", type=int, default=None, help="GPU index; omit for CPU or build-only")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None, help="small smoke subset per pattern_class")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_manifest(args.limit_per_class)
    manifest_path = out_dir / "anchor_margin_item_manifest.jsonl"
    if args.force or not manifest_path.exists() or args.limit_per_class is not None:
        with manifest_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    else:
        # preserve existing full manifest if present
        pass
    class_counts = dict(Counter(it.pattern_class for it in items))
    pattern_counts = dict(Counter(it.pattern for it in items))
    if args.build_only:
        out = {
            "status": "BUILD_ONLY",
            "created_utc": now(),
            "n_items": len(items),
            "pattern_counts": pattern_counts,
            "pattern_class_counts": class_counts,
            "manifest_path": rel(manifest_path),
            "note": "No model inference run.",
        }
    else:
        model_path = pathlib.Path(args.model_path)
        if not model_path.exists() and FALLBACK_MODEL.exists():
            model_path = FALLBACK_MODEL
        scored = score_items(items, model_path, args.gpu, out_dir)
        scored_path = out_dir / "anchor_margin_scored_items.jsonl"
        with scored_path.open("w", encoding="utf-8") as f:
            for rec in scored:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = summarize(scored)
        out = {
            "status": "COMPLETE",
            "created_utc": now(),
            "model_path": rel(model_path),
            "gpu": args.gpu,
            "n_items": len(items),
            "pattern_counts": pattern_counts,
            "pattern_class_counts": class_counts,
            "manifest_path": rel(manifest_path),
            "scored_items_path": rel(scored_path),
            **summary,
        }
    out_json = out_dir / "anchor_margin_alpha_census.json"
    out_md = out_dir / "anchor_margin_alpha_census.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research anchor-margin conditioned alpha-sweep census",
        "",
        f"Status: **{out['status']}**",
        f"n_items: `{out['n_items']}`",
        "",
        "Pattern counts: `" + json.dumps(pattern_counts, sort_keys=True) + "`",
        "",
    ]
    if out["status"] == "COMPLETE":
        mg = out["margin_summary"]
        lines += [
            "## Margin summaries",
            "",
            "Anchor margin = official-style anchor log-score(gold) − max log-score(non-gold).",
            "",
            "| group | n | median | mean | q05/q25/q50/q75/q95 | frac margin<0 | frac |margin|<0.5 |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
        for name in ["monotonic_activation", "monotonic_damage", "nonmonotonic_changed"]:
            r = mg.get(name, {})
            if r.get("finite", 0):
                qs = r["q05_q25_q50_q75_q95"]
                lines.append(f"| {name} | {r['n']} | {r['median']:+.4f} | {r['mean']:+.4f} | {[round(x,4) for x in qs]} | {r['frac_margin_below_0']:.3f} | {r['frac_abs_margin_below_0p5']:.3f} |")
        avd = mg.get("activation_vs_damage", {})
        if avd:
            lines += [
                "",
                "Activation-vs-damage separability:",
                f"- damage minus activation mean signed margin: `{avd.get('damage_minus_activation_mean_margin'):+.6f}`",
                f"- damage minus activation median signed margin: `{avd.get('damage_minus_activation_median_margin'):+.6f}`",
                f"- AUC(damage by signed margin): `{avd.get('auc_damage_by_signed_margin'):.6f}`",
                f"- AUC(damage by abs margin): `{avd.get('auc_damage_by_abs_margin'):.6f}`",
            ]
        lines += ["", "## By column", "", "| column | n | act n | dmg n | act median margin | dmg median margin | act |margin|<0.5 | dmg |margin|<0.5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for col, r in out["by_column"].items():
            lines.append(f"| {col} | {r['n']} | {r['activation_n']} | {r['damage_n']} | {r['activation_median_margin']} | {r['damage_median_margin']} | {r['activation_abs_margin_lt_0p5']} | {r['damage_abs_margin_lt_0p5']} |")
    else:
        lines += ["Build-only manifest produced; no margins scored."]
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "n_items": out["n_items"], "out_json": rel(out_json), "manifest": rel(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
