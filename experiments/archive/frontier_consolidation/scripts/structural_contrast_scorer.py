#!/usr/bin/env python3
"""research: score structural contrast-margin probe across checkpoints.

For each (coherent, perturbed) pair, computes a focused pseudo-log-likelihood
margin: NLL_focused(perturbed) - NLL_focused(coherent).  Positive margin means
the model assigns higher probability to the coherent structural binding.

Aggregation follows the frozen plan weights:
  R_t = 0.30 R_state + 0.25 R_temporal + 0.20 R_directed + 0.15 R_belief
        + 0.10 R_polarity - 0.25 L_t

Usage:
  python structural_contrast_scorer.py \
      --probe-json <probe_path> \
      --run-hf <checkpoint_root> \
      --checkpoints chck_77M chck_78M ... chck_100M \
      --gpu 0 \
      --tag <output_tag>
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from collections import defaultdict

# Writable HF caches for custom adapter code
SESSION_HF_CACHE = Path("experiments/archive/frontier_consolidation/.hf_cache_scorer")
os.environ["HF_HOME"] = str(SESSION_HF_CACHE / "home")
os.environ["HF_MODULES_CACHE"] = str(SESSION_HF_CACHE / "modules")
os.environ["TRANSFORMERS_CACHE"] = str(SESSION_HF_CACHE / "transformers")
for d in [SESSION_HF_CACHE / "home", SESSION_HF_CACHE / "modules",
          SESSION_HF_CACHE / "transformers"]:
    d.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

ROOT = Path("experiments/archive/frontier_consolidation")
DEFAULT_RUN_HF = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
DEFAULT_PROBE = ROOT / "data/structural_contrast_probe/structural_contrast_probe.json"
OUT_DIR = ROOT / "data/structural_contrast_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known official cheap7 for correlation (scale1.75 trajectory)
CHEAP7_SCALE1P75 = {
    "chck_77M": 43.28214285714286, "chck_78M": 43.70214285714286,
    "chck_79M": 43.57857142857143, "chck_80M": 43.81214285714286,
    "chck_81M": 43.64928571428572, "chck_82M": 43.95944987645173,
    "chck_83M": 43.80785714285714, "chck_100M": 43.543159919261925,
}

# Plan scoring weights (frozen before seeing any checkpoint scores)
FAMILY_WEIGHTS = {
    "entity_state": 0.30,
    "temporal_order": 0.25,
    "directed_relation": 0.20,
    "belief_report": 0.15,
    "polarity_relation": 0.10,
}
SURFACE_WEIGHT = -0.25


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def checkpoint_key(name: str) -> int:
    return int(name.replace("chck_", "").replace("M", ""))


# ---------- Focus token mapping -------------------------------------------

def map_focus_to_tokens(text: str, focus_char_spans: list[list[int]],
                        tokenizer, max_length: int = 256) -> tuple[list, list, list]:
    """Map character-level focus spans to token indices.

    Returns: (input_ids, attention_mask, focus_token_indices)
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, max_length=max_length, return_attention_mask=True)
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    focus_indices = set()
    for cs, ce in focus_char_spans:
        for tok_idx, (ts, te) in enumerate(offsets):
            if ts < ce and te > cs and ts != te:
                focus_indices.add(tok_idx)

    return input_ids, attn_mask, sorted(focus_indices)


# ---------- Focused PLL for one text --------------------------------------

def focused_nll(model, input_ids: list[int], attn_mask: list[int],
                focus_indices: list[int], mask_token_id: int,
                device: str, amp_dtype) -> tuple[float, int]:
    """Compute focused NLL by masking each focus token one at a time (batched)."""
    if not focus_indices:
        return 0.0, 0

    # Build batch: one copy per focus token, each with that token masked
    batch_ids = []
    targets = []  # (batch_idx, tok_idx, true_id)
    for fi in focus_indices:
        if input_ids[fi] in (0, mask_token_id):  # skip pad or already mask
            continue
        row = list(input_ids)
        true_id = row[fi]
        row[fi] = mask_token_id
        batch_ids.append(row)
        targets.append((len(batch_ids) - 1, fi, true_id))

    if not targets:
        return 0.0, 0

    inp = torch.tensor(batch_ids, dtype=torch.long, device=device)
    attn = torch.tensor([attn_mask] * len(batch_ids), dtype=torch.long, device=device)

    with torch.no_grad():
        if amp_dtype and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_ids=inp, attention_mask=attn).logits
        else:
            logits = model(input_ids=inp, attention_mask=attn).logits

    total_nll = 0.0
    for bi, ti, true_id in targets:
        lp = F.log_softmax(logits[bi, ti].float(), dim=-1)
        total_nll += -lp[true_id].item()

    return total_nll, len(targets)


# ---------- Score one pair ------------------------------------------------

def score_pair(model, tokenizer, pair: dict, device: str, amp_dtype,
               max_length: int = 256) -> dict | None:
    """Compute focused margin for one (coherent, perturbed) pair.

    margin = NLL_focused(perturbed) - NLL_focused(coherent)
    Positive margin means the model prefers the coherent version.
    """
    coh_ids, coh_attn, coh_focus = map_focus_to_tokens(
        pair["coherent_text"], pair["focus_spans_coherent"], tokenizer, max_length)
    pert_ids, pert_attn, pert_focus = map_focus_to_tokens(
        pair["perturbed_text"], pair["focus_spans_perturbed"], tokenizer, max_length)

    mask_id = tokenizer.mask_token_id

    coh_nll, coh_n = focused_nll(model, coh_ids, coh_attn, coh_focus,
                                 mask_id, device, amp_dtype)
    pert_nll, pert_n = focused_nll(model, pert_ids, pert_attn, pert_focus,
                                   mask_id, device, amp_dtype)

    if coh_n == 0 and pert_n == 0:
        return None

    # Normalize by number of focus tokens (average per-token NLL)
    coh_avg = coh_nll / max(coh_n, 1)
    pert_avg = pert_nll / max(pert_n, 1)
    margin = pert_avg - coh_avg  # positive = model prefers coherent

    return {
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "sub_type": pair.get("sub_type", ""),
        "corpus_source": pair.get("corpus_source", ""),
        "coherent_nll": round(coh_nll, 6),
        "coherent_n_tokens": coh_n,
        "perturbed_nll": round(pert_nll, 6),
        "perturbed_n_tokens": pert_n,
        "margin": round(margin, 6),
        "correct": margin > 0,
    }


# ---------- Aggregation ---------------------------------------------------

def trimmed_mean(xs: list[float], trim_frac: float = 0.2) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(len(xs) * trim_frac)
    trimmed = xs[k:len(xs) - k] if k > 0 else xs
    return sum(trimmed) / len(trimmed) if trimmed else 0.0


def aggregate_scores(scored_pairs: list[dict]) -> dict:
    """Compute family-level and composite structural margin scores."""
    by_family = defaultdict(list)
    for sp in scored_pairs:
        by_family[sp["family"]].append(sp)

    family_results = {}
    for fam, pairs in sorted(by_family.items()):
        margins = [p["margin"] for p in pairs]
        corrects = [p["correct"] for p in pairs]
        family_results[fam] = {
            "n_pairs": len(pairs),
            "mean_margin": round(sum(margins) / len(margins), 6) if margins else None,
            "trimmed_mean_margin": round(trimmed_mean(margins), 6),
            "median_margin": round(sorted(margins)[len(margins) // 2], 6) if margins else None,
            "hard_accuracy": round(sum(corrects) / len(corrects), 6) if corrects else None,
            "q25_margin": round(sorted(margins)[len(margins) // 4], 6) if len(margins) >= 4 else None,
        }

    # Composite score R_t using plan weights
    composite = 0.0
    for fam, w in FAMILY_WEIGHTS.items():
        if fam in family_results and family_results[fam]["trimmed_mean_margin"] is not None:
            composite += w * family_results[fam]["trimmed_mean_margin"]
    # Subtract surface control baseline
    if "surface_control" in family_results and family_results["surface_control"]["trimmed_mean_margin"] is not None:
        composite += SURFACE_WEIGHT * family_results["surface_control"]["trimmed_mean_margin"]

    # Entity state by depth
    es_pairs = by_family.get("entity_state", [])
    depth_results = {}
    for depth in [2, 3, 4]:
        d_margins = [p["margin"] for p in es_pairs
                     if p.get("sub_type", "").startswith(f"depth{depth}")]
        if d_margins:
            depth_results[f"depth{depth}"] = {
                "n": len(d_margins),
                "mean_margin": round(sum(d_margins) / len(d_margins), 6),
                "hard_accuracy": round(sum(1 for m in d_margins if m > 0) / len(d_margins), 6),
            }

    return {
        "composite_score": round(composite, 6),
        "family_results": family_results,
        "entity_state_by_depth": depth_results,
    }


# ---------- Checkpoint evaluation -----------------------------------------

def eval_one_checkpoint(ckpt_dir: Path, tokenizer, pairs: list[dict],
                        device: str = "cuda", amp_dtype=torch.bfloat16) -> dict:
    t0 = time.time()
    print(json.dumps({"event": "load_model", "checkpoint": ckpt_dir.name,
                       "path": rel(ckpt_dir)}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    model.eval().to(device)

    scored = []
    n_skipped = 0
    for i, pair in enumerate(pairs):
        result = score_pair(model, tokenizer, pair, device, amp_dtype)
        if result is None:
            n_skipped += 1
        else:
            scored.append(result)
        if (i + 1) % 200 == 0:
            print(json.dumps({"event": "progress", "checkpoint": ckpt_dir.name,
                               "scored": len(scored), "total": i + 1}), flush=True)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    agg = aggregate_scores(scored)
    elapsed = round(time.time() - t0, 2)
    print(json.dumps({"event": "checkpoint_done", "checkpoint": ckpt_dir.name,
                       "scored": len(scored), "skipped": n_skipped,
                       "composite": agg["composite_score"],
                       "elapsed_sec": elapsed}), flush=True)

    return {
        "checkpoint": ckpt_dir.name,
        "checkpoint_M": checkpoint_key(ckpt_dir.name),
        "cheap7": CHEAP7_SCALE1P75.get(ckpt_dir.name),
        "elapsed_sec": elapsed,
        "n_scored": len(scored),
        "n_skipped": n_skipped,
        "aggregation": agg,
        "per_pair": scored,
    }


# ---------- Analysis across checkpoints -----------------------------------

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs_ = [p[0] for p in pairs]
    ys_ = [p[1] for p in pairs]
    mx = sum(xs_) / len(xs_)
    my = sum(ys_) / len(ys_)
    vx = sum((x - mx) ** 2 for x in xs_)
    vy = sum((y - my) ** 2 for y in ys_)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def analyze(records: list[dict], tag: str):
    """Compute cross-checkpoint correlations and write output."""
    records = sorted(records, key=lambda r: r["checkpoint_M"])

    # Extract series
    ckpts = [r["checkpoint"] for r in records]
    cheap7s = [r["cheap7"] for r in records]
    composites = [r["aggregation"]["composite_score"] for r in records]

    # Per-family series
    fam_series = defaultdict(list)
    for r in records:
        for fam, res in r["aggregation"]["family_results"].items():
            fam_series[fam].append(res["trimmed_mean_margin"])

    # Correlations: positive correlation means higher structural margin ↔ higher cheap7
    corr_composite = pearson(composites, cheap7s)
    fam_corrs = {}
    for fam, series in fam_series.items():
        fam_corrs[fam] = pearson(series, cheap7s)

    # Depth analysis
    depth_series = defaultdict(list)
    for r in records:
        for dk, dv in r["aggregation"].get("entity_state_by_depth", {}).items():
            depth_series[dk].append(dv["mean_margin"])
    depth_corrs = {}
    for dk, series in depth_series.items():
        depth_corrs[dk] = pearson(series, cheap7s)

    # Selector: which checkpoint has the highest composite?
    best_composite_idx = max(range(len(composites)), key=lambda i: composites[i])
    best_composite_ckpt = ckpts[best_composite_idx]

    analysis = {
        "tag": tag,
        "checkpoints": ckpts,
        "cheap7_series": cheap7s,
        "composite_series": composites,
        "composite_vs_cheap7_pearson": round(corr_composite, 6) if corr_composite else None,
        "family_vs_cheap7_pearson": {k: round(v, 6) if v else None
                                      for k, v in fam_corrs.items()},
        "depth_vs_cheap7_pearson": {k: round(v, 6) if v else None
                                     for k, v in depth_corrs.items()},
        "best_composite_checkpoint": best_composite_ckpt,
        "best_composite_score": round(composites[best_composite_idx], 6),
        "official_best_cheap7_checkpoint": "chck_82M",
    }

    # Result object
    result = {
        "status": "STRUCTURAL_CONTRAST_SCORED",
        "tag": tag,
        "probe_id": "structural_contrast_margin_v1",
        "analysis": analysis,
        "per_checkpoint": [{k: v for k, v in r.items() if k != "per_pair"}
                           for r in records],
    }

    out_json = OUT_DIR / f"{tag}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md = [
        f"# Structural contrast-margin scores: {tag}",
        "",
        f"Probe: structural_contrast_margin_v1",
        f"Composite-vs-cheap7 Pearson: **{analysis['composite_vs_cheap7_pearson']}**",
        f"Best composite checkpoint: **{best_composite_ckpt}** (official best: chck_82M)",
        "",
        "## Per-checkpoint summary",
        "",
        "| Checkpoint | cheap7 | composite | entity_state | temporal | directed | belief | polarity | surface |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        fam_r = r["aggregation"]["family_results"]
        def fv(fam):
            return f"{fam_r[fam]['trimmed_mean_margin']:.4f}" if fam in fam_r else "-"
        md.append(
            f"| {r['checkpoint']} | {r['cheap7'] or '-':.4f} | "
            f"{r['aggregation']['composite_score']:.4f} | "
            f"{fv('entity_state')} | {fv('temporal_order')} | "
            f"{fv('directed_relation')} | {fv('belief_report')} | "
            f"{fv('polarity_relation')} | {fv('surface_control')} |"
        )

    md.extend([
        "",
        "## Family correlations with cheap7",
        "",
    ])
    for fam, corr in sorted(fam_corrs.items()):
        md.append(f"- {fam}: {corr:.4f}" if corr else f"- {fam}: N/A")

    md.extend([
        "",
        "## Entity state depth correlations",
        "",
    ])
    for dk, corr in sorted(depth_corrs.items()):
        md.append(f"- {dk}: {corr:.4f}" if corr else f"- {dk}: N/A")

    out_md = OUT_DIR / f"{tag}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    return result


# ---------- Main ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", type=str, default=str(DEFAULT_PROBE))
    ap.add_argument("--run-hf", type=str, default=str(DEFAULT_RUN_HF))
    ap.add_argument("--checkpoints", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", type=str, default="scale1p75_late_panel")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: only 2 checkpoints and 20 pairs per family")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

    probe = json.load(open(args.probe_json))
    all_pairs = probe["pairs"]
    print(json.dumps({"event": "probe_loaded", "n_pairs": len(all_pairs),
                       "probe_id": probe.get("probe_id")}), flush=True)

    if args.smoke:
        # Limit to 20 pairs per family for smoke test
        by_fam = defaultdict(list)
        for p in all_pairs:
            by_fam[p["family"]].append(p)
        all_pairs = []
        for fam, fps in by_fam.items():
            all_pairs.extend(fps[:20])
        print(json.dumps({"event": "smoke_limit", "n_pairs": len(all_pairs)}), flush=True)

    run_hf = Path(args.run_hf)
    if args.checkpoints:
        ckpt_names = args.checkpoints
    else:
        ckpt_names = ["chck_77M", "chck_78M", "chck_79M", "chck_80M",
                       "chck_81M", "chck_82M", "chck_83M", "chck_100M"]

    # Load tokenizer from first checkpoint
    first_ckpt = run_hf / ckpt_names[0]
    tokenizer = AutoTokenizer.from_pretrained(str(first_ckpt), trust_remote_code=True)
    print(json.dumps({"event": "tokenizer_loaded", "vocab_size": tokenizer.vocab_size,
                       "mask_token_id": tokenizer.mask_token_id}), flush=True)

    records = []
    for ckpt_name in ckpt_names:
        ckpt_dir = run_hf / ckpt_name
        if not ckpt_dir.exists():
            print(json.dumps({"event": "skip_missing", "checkpoint": ckpt_name}), flush=True)
            continue
        rec = eval_one_checkpoint(ckpt_dir, tokenizer, all_pairs, device, amp_dtype)
        records.append(rec)

    if records:
        result = analyze(records, args.tag)
        print(json.dumps({
            "status": result["status"],
            "tag": args.tag,
            "composite_vs_cheap7_pearson": result["analysis"]["composite_vs_cheap7_pearson"],
            "best_composite_checkpoint": result["analysis"]["best_composite_checkpoint"],
            "out_json": str(OUT_DIR / f"{args.tag}.json"),
            "out_md": str(OUT_DIR / f"{args.tag}.md"),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
