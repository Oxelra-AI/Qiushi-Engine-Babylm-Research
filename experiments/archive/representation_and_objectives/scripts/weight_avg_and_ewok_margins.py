#!/usr/bin/env python3
"""research: Cross-seed weight averaging + EWoK per-item PLL margin extraction.

Produces:
  1. Cross-seed averaged reinvest model (CPU)
  2. Tail-averaged seed43022 and seed43122 models (CPU)
  3. Per-item EWoK PLL margins for all 7 models (GPU)
  4. Pattern-aligned margin analysis for the decision (CPU)

Decisive outputs:
  - Does cross-seed averaging produce a better EWoK score than either seed alone?
  - Are the DiD-negative items near-zero margin (noise) or confidently wrong (systematic)?
  - What is the recommended stability action?
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import statistics
import time
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

USER_ROOT = pathlib.Path(".").resolve()
SESS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
OUT = SESS / "data" / "weight_avg_and_margins"

# ── Model checkpoint paths ──────────────────────────────────────────
ORIGINAL_MODELS = {
    "clean430": USER_ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M",
    "clean431": USER_ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122/hf_model/chck_100M",
    "reinv430": USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    "reinv431": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_100M",
}
REINVEST_43022_HF = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model"
REINVEST_43122_HF = USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model"

# ── EWoK gold data (pristine 7618 rows) ────────────────────────────
EWOK_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"


# ═══════════════════════════════════════════════════════════════════
# Weight Averaging
# ═══════════════════════════════════════════════════════════════════

def average_state_dicts(paths: list[pathlib.Path], out_path: pathlib.Path, label: str) -> pathlib.Path:
    """Average model weights from multiple safetensor checkpoint paths."""
    out_path.mkdir(parents=True, exist_ok=True)
    sds = [load_file(str(p / "model.safetensors")) for p in paths]
    n = len(sds)
    averaged = {}
    for key in sds[0]:
        s = sds[0][key].clone().float()
        for i in range(1, n):
            s += sds[i][key].float()
        averaged[key] = (s / n).to(sds[0][key].dtype)
    save_file(averaged, str(out_path / "model.safetensors"))
    for fname in ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        src = paths[0] / fname
        if src.exists():
            shutil.copy2(str(src), str(out_path / fname))
    info = {"label": label, "n": n, "sources": [str(p) for p in paths]}
    (out_path / "averaging_info.json").write_text(json.dumps(info, indent=2))
    del sds, averaged
    print(f"  Averaged {n} models -> {out_path.name}", flush=True)
    return out_path


# ═══════════════════════════════════════════════════════════════════
# EWoK PLL Margin Scorer
# ═══════════════════════════════════════════════════════════════════

def load_ewok_gold(ewok_dir: pathlib.Path) -> list[dict]:
    """Load all EWoK gold items following the official decode_ewok logic."""
    items = []
    for f in sorted(ewok_dir.glob("*.jsonl")):
        for line in f.read_text().strip().split("\n"):
            if not line.strip():
                continue
            raw = json.loads(line)
            items.append({
                "domain": raw["Domain"],
                "context_type": raw["ContextType"],
                "context_diff": raw["ContextDiff"],
                "target_diff": raw["TargetDiff"],
                "sentence_good": raw["Context1"] + " " + raw["Target1"],
                "sentence_bad": raw["Context2"] + " " + raw["Target1"],
            })
    return items


def compute_pll(model, tokenizer, sentence: str, device, mask_id: int,
                special_ids: set[int], max_sub_batch: int = 96) -> float:
    """Pseudo-log-likelihood via mask-each-token-and-sum for an MLM model."""
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=True,
                    truncation=True, max_length=512)
    input_ids = enc["input_ids"][0]       # [seq_len]
    attn_mask = enc["attention_mask"][0]   # [seq_len]
    seq_len = input_ids.shape[0]

    mask_positions = [i for i in range(seq_len) if input_ids[i].item() not in special_ids]
    if not mask_positions:
        return float("-inf")

    n = len(mask_positions)
    batch_ids = input_ids.unsqueeze(0).expand(n, -1).clone()
    batch_attn = attn_mask.unsqueeze(0).expand(n, -1).clone()
    targets = torch.zeros(n, dtype=torch.long)

    for i, pos in enumerate(mask_positions):
        targets[i] = batch_ids[i, pos]
        batch_ids[i, pos] = mask_id

    total_lp = 0.0
    for s in range(0, n, max_sub_batch):
        e = min(s + max_sub_batch, n)
        sub_ids = batch_ids[s:e].to(device)
        sub_attn = batch_attn[s:e].to(device)
        sub_tgt = targets[s:e].to(device)
        sub_pos = mask_positions[s:e]

        with torch.no_grad():
            logits = model(input_ids=sub_ids, attention_mask=sub_attn).logits

        for j, pos in enumerate(sub_pos):
            lp = torch.log_softmax(logits[j, pos], dim=-1)
            total_lp += lp[sub_tgt[j]].item()

    return total_lp


def score_ewok(model, tokenizer, items: list[dict], device, label: str):
    """Score all EWoK items; return per-item margin records and accuracy."""
    mask_id = tokenizer.mask_token_id
    special_ids = set()
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    margins = []
    correct = 0
    t0 = time.time()

    for idx, item in enumerate(items):
        pll_good = compute_pll(model, tokenizer, item["sentence_good"],
                               device, mask_id, special_ids)
        pll_bad = compute_pll(model, tokenizer, item["sentence_bad"],
                              device, mask_id, special_ids)
        margin = pll_good - pll_bad
        is_correct = margin > 0
        if is_correct:
            correct += 1
        margins.append({
            "idx": idx,
            "domain": item["domain"],
            "context_type": item["context_type"],
            "context_diff": item["context_diff"],
            "target_diff": item["target_diff"],
            "pll_good": round(pll_good, 6),
            "pll_bad": round(pll_bad, 6),
            "margin": round(margin, 6),
            "correct": is_correct,
        })
        if (idx + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  [{label}] {idx+1}/{len(items)}  {el:.0f}s  acc={correct/(idx+1)*100:.2f}%",
                  flush=True)

    el = time.time() - t0
    accuracy = correct / len(items) * 100
    print(f"  [{label}] {len(items)} items  {el:.1f}s  accuracy={accuracy:.4f}%", flush=True)
    return margins, accuracy


# ═══════════════════════════════════════════════════════════════════
# Decision Analysis
# ═══════════════════════════════════════════════════════════════════

def analyse(items, all_margins, results):
    """Compute pattern-aligned margin statistics and decision recommendation."""
    n = len(items)

    # Build per-item patterns
    patterns_data: dict[str, dict] = {}
    for idx in range(n):
        c430 = int(all_margins["clean430"][idx]["correct"])
        r430 = int(all_margins["reinv430"][idx]["correct"])
        c431 = int(all_margins["clean431"][idx]["correct"])
        r431 = int(all_margins["reinv431"][idx]["correct"])
        pat = f"{c430}{r430}{c431}{r431}"

        if pat not in patterns_data:
            patterns_data[pat] = {k: [] for k in
                ["r430_m", "r431_m", "cross_m", "tail022_m", "tail122_m"]}
            patterns_data[pat]["count"] = 0
        patterns_data[pat]["count"] += 1
        patterns_data[pat]["r430_m"].append(all_margins["reinv430"][idx]["margin"])
        patterns_data[pat]["r431_m"].append(all_margins["reinv431"][idx]["margin"])
        patterns_data[pat]["cross_m"].append(all_margins["cross_seed_avg"][idx]["margin"])
        patterns_data[pat]["tail022_m"].append(all_margins["tail_avg_022"][idx]["margin"])
        patterns_data[pat]["tail122_m"].append(all_margins["tail_avg_122"][idx]["margin"])

    def safe_stats(arr):
        if not arr:
            return {}
        arr_abs = [abs(x) for x in arr]
        return {
            "mean": round(statistics.mean(arr), 4),
            "median": round(statistics.median(arr), 4),
            "abs_mean": round(statistics.mean(arr_abs), 4),
            "near_zero_frac": round(sum(1 for x in arr_abs if x < 2.0) / len(arr), 4),
            "pct_correct": round(sum(1 for x in arr if x > 0) / len(arr) * 100, 2),
        }

    pattern_summary = {}
    for pat in sorted(patterns_data, key=lambda p: -patterns_data[p]["count"]):
        d = patterns_data[pat]
        pattern_summary[pat] = {
            "count": d["count"],
            "reinv430": safe_stats(d["r430_m"]),
            "reinv431": safe_stats(d["r431_m"]),
            "cross_seed_avg": safe_stats(d["cross_m"]),
            "tail_avg_022": safe_stats(d["tail022_m"]),
            "tail_avg_122": safe_stats(d["tail122_m"]),
        }

    results["pattern_margin_summary"] = pattern_summary

    # Domain-level cross-seed averaged model accuracy
    domain_cross = {}
    for idx in range(n):
        dom = items[idx]["domain"]
        if dom not in domain_cross:
            domain_cross[dom] = {"correct": 0, "total": 0}
        domain_cross[dom]["total"] += 1
        if all_margins["cross_seed_avg"][idx]["correct"]:
            domain_cross[dom]["correct"] += 1
    domain_summary = {d: round(v["correct"]/v["total"]*100, 2)
                      for d, v in sorted(domain_cross.items())}
    results["cross_seed_avg_ewok_by_domain"] = domain_summary

    # Overall margin statistics for original seeds
    r430_all = [m["margin"] for m in all_margins["reinv430"]]
    r431_all = [m["margin"] for m in all_margins["reinv431"]]
    cross_all = [m["margin"] for m in all_margins["cross_seed_avg"]]

    # Margin correlation between seeds
    mean430 = statistics.mean(r430_all)
    mean431 = statistics.mean(r431_all)
    cov = sum((a - mean430) * (b - mean431) for a, b in zip(r430_all, r431_all)) / n
    std430 = (sum((a - mean430)**2 for a in r430_all) / n) ** 0.5
    std431 = (sum((b - mean431)**2 for b in r431_all) / n) ** 0.5
    corr = cov / (std430 * std431) if std430 > 0 and std431 > 0 else 0.0

    results["decision_metrics"] = {
        "reinv430_mean_margin": round(mean430, 4),
        "reinv431_mean_margin": round(mean431, 4),
        "cross_seed_avg_mean_margin": round(statistics.mean(cross_all), 4),
        "margin_pearson_r430_r431": round(corr, 4),
        "reinv430_near_zero_frac_2": round(
            sum(1 for x in r430_all if abs(x) < 2.0) / n, 4),
        "reinv431_near_zero_frac_2": round(
            sum(1 for x in r431_all if abs(x) < 2.0) / n, 4),
        "cross_seed_avg_near_zero_frac_2": round(
            sum(1 for x in cross_all if abs(x) < 2.0) / n, 4),
    }

    # Decision recommendation
    ewok_accuracies = {k: v for k, v in results.items() if k.startswith("ewok_accuracy_")}
    cross_acc = results.get("ewok_accuracy_cross_seed_avg", 0)
    r430_acc = results.get("ewok_accuracy_reinv430", 0)
    r431_acc = results.get("ewok_accuracy_reinv431", 0)
    tail022_acc = results.get("ewok_accuracy_tail_avg_022", 0)

    decision = {}
    decision["cross_seed_avg_vs_seeds"] = {
        "cross_seed_avg": round(cross_acc, 4),
        "reinv430": round(r430_acc, 4),
        "reinv431": round(r431_acc, 4),
        "individual_mean": round((r430_acc + r431_acc) / 2, 4),
        "avg_improves_over_mean": cross_acc > (r430_acc + r431_acc) / 2,
    }
    decision["tail_avg_vs_individual"] = {
        "tail_avg_022": round(tail022_acc, 4),
        "reinv430_individual": round(r430_acc, 4),
        "tail_improves": tail022_acc > r430_acc,
    }

    # Key diagnostic: pattern 0110 margins
    pat0110 = pattern_summary.get("0110", {})
    if pat0110:
        r430_nz = pat0110.get("reinv430", {}).get("near_zero_frac", 0)
        r431_nz = pat0110.get("reinv431", {}).get("near_zero_frac", 0)
        cross_correct_pct = pat0110.get("cross_seed_avg", {}).get("pct_correct", 0)
        if r430_nz > 0.5 and r431_nz > 0.5:
            decision["instability_type"] = "near_zero_margins"
            decision["recommended_action"] = "weight_averaging_or_tail_consolidation"
        elif r430_nz < 0.3 and r431_nz < 0.3:
            decision["instability_type"] = "confident_disagreement"
            decision["recommended_action"] = "third_seed_or_relation_repair"
        else:
            decision["instability_type"] = "mixed"
            decision["recommended_action"] = "evaluate_cross_seed_avg_full_surface"
        decision["pat0110_cross_avg_recovered_pct"] = cross_correct_pct

    results["decision"] = decision


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {"status": "WEIGHT_AVG_AND_MARGINS"}

    # ── Phase 1: Weight Averaging (CPU) ─────────────────────────────
    print("=== Phase 1: Weight averaging (CPU) ===", flush=True)
    t0 = time.time()

    avg_models: dict[str, pathlib.Path] = {}

    # Cross-seed: reinv430 + reinv431 at chck_100M
    avg_models["cross_seed_avg"] = average_state_dicts(
        [ORIGINAL_MODELS["reinv430"], ORIGINAL_MODELS["reinv431"]],
        OUT / "models" / "cross_seed_avg_100M",
        "cross_seed_avg_reinv_100M",
    )

    # Tail average seed43022: chck_90M-100M (11 checkpoints)
    avg_models["tail_avg_022"] = average_state_dicts(
        [REINVEST_43022_HF / f"chck_{i}M" for i in range(90, 101)],
        OUT / "models" / "tail_avg_seed43022_90_100M",
        "tail_avg_seed43022_90_100M",
    )

    # Tail average seed43122: chck_90M-100M
    avg_models["tail_avg_122"] = average_state_dicts(
        [REINVEST_43122_HF / f"chck_{i}M" for i in range(90, 101)],
        OUT / "models" / "tail_avg_seed43122_90_100M",
        "tail_avg_seed43122_90_100M",
    )

    results["weight_averaging_elapsed_sec"] = round(time.time() - t0, 1)
    results["averaged_model_paths"] = {k: str(v) for k, v in avg_models.items()}
    print(f"  Done in {results['weight_averaging_elapsed_sec']}s\n", flush=True)

    # ── Phase 2: EWoK Margin Extraction (GPU) ──────────────────────
    print(f"=== Phase 2: EWoK PLL margin extraction (device={device}) ===", flush=True)

    items = load_ewok_gold(EWOK_DIR)
    results["ewok_gold_items"] = len(items)
    print(f"  Loaded {len(items)} EWoK gold items\n", flush=True)
    assert len(items) == 7618, f"Expected 7618, got {len(items)}"

    all_models = dict(ORIGINAL_MODELS)
    all_models.update(avg_models)

    all_margins: dict[str, list] = {}

    for label, model_path in all_models.items():
        print(f"--- {label}: {model_path} ---", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(
            str(model_path), torch_dtype=torch.float16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        margins, accuracy = score_ewok(model, tokenizer, items, device, label)
        all_margins[label] = margins
        results[f"ewok_accuracy_{label}"] = accuracy

        del model
        torch.cuda.empty_cache()

    # Save full per-item margins (large file)
    margins_path = OUT / "ewok_margins_all_models.json"
    with open(margins_path, "w") as f:
        json.dump(all_margins, f)
    results["margins_path"] = str(margins_path)

    # ── Phase 3: Decision Analysis (CPU) ───────────────────────────
    print("\n=== Phase 3: Analysis ===", flush=True)
    analyse(items, all_margins, results)

    # ── Save ───────────────────────────────────────────────────────
    out_json = OUT / "weight_avg_and_margins_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save note
    note = (USER_ROOT / 'research/notes/representation_and_objectives/46_weight_avg_and_ewok_margins.md')
    note.parent.mkdir(parents=True, exist_ok=True)
    accs = {k.replace("ewok_accuracy_", ""): f"{v:.2f}" for k, v in results.items()
            if k.startswith("ewok_accuracy_")}
    dm = results.get("decision_metrics", {})
    dec = results.get("decision", {})
    note.write_text(
        f"# research — Weight averaging + EWoK margin analysis\n\n"
        f"## EWoK accuracies (standalone PLL scorer, temp=1)\n"
        + "\n".join(f"- {k}: {v}%" for k, v in sorted(accs.items()))
        + f"\n\n## Decision metrics\n"
        f"- reinv430 mean margin: {dm.get('reinv430_mean_margin')}\n"
        f"- reinv431 mean margin: {dm.get('reinv431_mean_margin')}\n"
        f"- cross-seed avg mean margin: {dm.get('cross_seed_avg_mean_margin')}\n"
        f"- Pearson(r430, r431 margins): {dm.get('margin_pearson_r430_r431')}\n"
        f"\n## Decision\n"
        f"- Instability type: {dec.get('instability_type', 'TBD')}\n"
        f"- Recommended action: {dec.get('recommended_action', 'TBD')}\n"
        f"- Pattern 0110 cross-avg recovered: {dec.get('pat0110_cross_avg_recovered_pct', 'TBD')}%\n"
        f"- Cross-seed avg improves: {dec.get('cross_seed_avg_vs_seeds', {}).get('avg_improves_over_mean')}\n"
        f"\nJSON: `{out_json}`\n"
    )

    # Print compact summary
    print(json.dumps({
        "status": results["status"],
        "ewok_accuracies": accs,
        "decision_metrics": dm,
        "decision": dec,
        "out_json": str(out_json),
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
