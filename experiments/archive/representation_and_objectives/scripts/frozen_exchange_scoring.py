#!/usr/bin/env python3
"""research: Frozen exchange scoring for role-switch packets.

Scores all role-switch pairs on frozen checkpoints to test:
  1. Whether current models are unsaturated (fail context-conditioned binding)
  2. Whether failure is context-dependent (not target-prior)
  3. Transfer across held-out families/entities

For each pair and context (AB, BA, erased), masks the target position in
  context + consequence_template.replace(TARGET, alternative)
and extracts log P(alternative tokens | masked text).

Four-cell margin: M = margin_AB + margin_BA
  where margin_AB = s(AB, correct_AB) - s(AB, wrong_AB)
Context-conditioned: M - 2*erased_bias
"""

import json, pathlib, sys, os, time, math
import torch
import torch.nn.functional as F
from collections import defaultdict

USER_ROOT = pathlib.Path(".")
PAIRS_PATH = USER_ROOT / "experiments/archive/representation_and_objectives/data/role_switch_packets/scoring_pairs.jsonl"
OUT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/frozen_exchange_scoring"
TOK_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"

CHECKPOINTS = {
    "legal40k_100M": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M",
    "reheat_100M": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M",
}

MAX_LEN = 128

def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def find_last_subseq(full_ids, sub_ids):
    """Find last occurrence of sub_ids as contiguous subsequence in full_ids."""
    n = len(sub_ids)
    for i in range(len(full_ids) - n, -1, -1):
        if full_ids[i:i+n] == sub_ids:
            return i
    return None

def score_one(model, tokenizer, text, target_word, device, mask_id):
    """Score logP(target_word) at its last occurrence in text, masked."""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN, add_special_tokens=True)
    input_ids = enc["input_ids"][0].tolist()

    # Try with leading space first (in-context token form)
    target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
    start = find_last_subseq(input_ids, target_ids)
    if start is None:
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        start = find_last_subseq(input_ids, target_ids)
    if start is None:
        return float('nan'), 0

    # Mask target positions
    masked = list(input_ids)
    for j in range(len(target_ids)):
        masked[start + j] = mask_id

    masked_t = torch.tensor([masked], device=device)
    attn_t = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=masked_t, attention_mask=attn_t).logits[0]

    log_probs = F.log_softmax(logits, dim=-1)
    total_lp = sum(log_probs[start + j, tid].item() for j, tid in enumerate(target_ids))
    return total_lp, len(target_ids)

def score_all_pairs(model, tokenizer, pairs, device):
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(f"Tokenizer at {TOK_DIR} has no mask_token_id")
    print(f"  mask_token_id={mask_id}, mask_token='{tokenizer.mask_token}'", flush=True)

    # Validate a few target tokenizations
    sample_words = set()
    for p in pairs[:20]:
        sample_words.add(p["alt_0"])
        sample_words.add(p["alt_1"])
    for w in sorted(sample_words)[:8]:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        print(f"    ' {w}' -> {ids} ({len(ids)} tok)", flush=True)

    results = []
    t0 = time.time()
    for idx, p in enumerate(pairs):
        con_tmpl = p["consequence_masked"]
        alt0, alt1 = p["alt_0"], p["alt_1"]
        cor_AB, cor_BA = p["correct_AB"], p["correct_BA"]

        scores = {}
        for ctx_key, ctx_text in [("AB", p["context_AB"]),
                                   ("BA", p["context_BA"]),
                                   ("erased", "")]:
            for alt_key, alt_word in [("alt0", alt0), ("alt1", alt1)]:
                # Build text: context + consequence with alt_word at TARGET
                con_filled = con_tmpl.replace("__TARGET__", alt_word)
                if ctx_text:
                    text = ctx_text + " " + con_filled
                else:
                    text = con_filled
                lp, ntok = score_one(model, tokenizer, text, alt_word,
                                     device, mask_id)
                scores[f"s_{ctx_key}_{alt_key}"] = lp
                scores[f"ntok_{ctx_key}_{alt_key}"] = ntok

        # Extract scores
        s_AB_0 = scores["s_AB_alt0"]
        s_AB_1 = scores["s_AB_alt1"]
        s_BA_0 = scores["s_BA_alt0"]
        s_BA_1 = scores["s_BA_alt1"]
        s_er_0 = scores["s_erased_alt0"]
        s_er_1 = scores["s_erased_alt1"]

        # Compute margins: correct - wrong under each context
        if cor_AB == alt0:
            margin_AB = s_AB_0 - s_AB_1
            margin_BA = s_BA_1 - s_BA_0  # cor_BA == alt1
            erased_bias = s_er_0 - s_er_1  # bias toward AB-correct
        else:
            margin_AB = s_AB_1 - s_AB_0
            margin_BA = s_BA_0 - s_BA_1
            erased_bias = s_er_1 - s_er_0

        M = margin_AB + margin_BA
        context_M = M - 2 * erased_bias

        acc_AB = 1.0 if margin_AB > 0 else 0.0
        acc_BA = 1.0 if margin_BA > 0 else 0.0
        both = 1.0 if (acc_AB > 0 and acc_BA > 0) else 0.0

        r = {
            "pair_id": p["pair_id"], "family": p["family"],
            "template_id": p["template_id"], "style": p["style"],
            "entity_split": p["entity_split"], "alt_type": p["alt_type"],
            "alt_0": alt0, "alt_1": alt1,
            "correct_AB": cor_AB, "correct_BA": cor_BA,
            **scores,
            "margin_AB": margin_AB, "margin_BA": margin_BA,
            "four_cell_M": M, "erased_bias": erased_bias,
            "context_conditioned_M": context_M,
            "acc_AB": acc_AB, "acc_BA": acc_BA, "both_correct": both,
        }
        results.append(r)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {idx+1}/{len(pairs)} pairs, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Scored {len(results)} pairs in {elapsed:.1f}s", flush=True)
    return results

def _safe_mean(vals):
    v = [x for x in vals if not (math.isnan(x) if isinstance(x, float) else False)]
    return sum(v) / len(v) if v else float('nan')

def summarize(results, label):
    summary = {"target": label, "n": len(results)}

    def _grp_stats(grp):
        Ms = [r["four_cell_M"] for r in grp]
        CMs = [r["context_conditioned_M"] for r in grp]
        biases = [r["erased_bias"] for r in grp]
        boths = [r["both_correct"] for r in grp]
        return {
            "n": len(grp),
            "M_mean": _safe_mean(Ms),
            "M_pos_frac": _safe_mean([1.0 if m > 0 else 0.0 for m in Ms]),
            "CM_mean": _safe_mean(CMs),
            "bias_mean": _safe_mean(biases),
            "bias_abs_mean": _safe_mean([abs(b) for b in biases]),
            "acc_AB": _safe_mean([r["acc_AB"] for r in grp]),
            "acc_BA": _safe_mean([r["acc_BA"] for r in grp]),
            "both_correct": _safe_mean(boths),
        }

    summary["overall"] = _grp_stats(results)

    # By family
    fam_grps = defaultdict(list)
    for r in results:
        fam_grps[r["family"]].append(r)
    summary["by_family"] = {k: _grp_stats(v) for k, v in sorted(fam_grps.items())}

    # By style
    style_grps = defaultdict(list)
    for r in results:
        style_grps[r["style"]].append(r)
    summary["by_style"] = {k: _grp_stats(v) for k, v in sorted(style_grps.items())}

    # By entity_split
    esplit_grps = defaultdict(list)
    for r in results:
        esplit_grps[r["entity_split"]].append(r)
    summary["by_entity_split"] = {k: _grp_stats(v) for k, v in sorted(esplit_grps.items())}

    # By (family, style, entity_split) — full cross
    cross_grps = defaultdict(list)
    for r in results:
        cross_grps[f"{r['family']}|{r['style']}|{r['entity_split']}"].append(r)
    summary["by_cross"] = {k: _grp_stats(v) for k, v in sorted(cross_grps.items())}

    return summary

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs(PAIRS_PATH)
    print(f"Loaded {len(pairs)} scoring pairs", flush=True)

    if args.checkpoint:
        if args.checkpoint in CHECKPOINTS:
            targets = {args.checkpoint: CHECKPOINTS[args.checkpoint]}
        else:
            targets = {args.checkpoint: pathlib.Path(args.checkpoint)}
    else:
        targets = CHECKPOINTS

    all_summaries = {}
    for label, ckpt in targets.items():
        print(f"\n{'='*60}\nScoring: {label} ({ckpt})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))
        model = AutoModelForMaskedLM.from_pretrained(str(ckpt), trust_remote_code=True)
        model = model.to(device).eval()

        results = score_all_pairs(model, tokenizer, pairs, device)

        # Save per-pair results
        rpath = OUT_DIR / f"{label}_pair_results.jsonl"
        with open(rpath, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        s = summarize(results, label)
        all_summaries[label] = s

        ov = s["overall"]
        print(f"\n  Overall ({ov['n']}): M={ov['M_mean']:.4f} pos={ov['M_pos_frac']:.3f} "
              f"CM={ov['CM_mean']:.4f} bias={ov['bias_mean']:.4f} |bias|={ov['bias_abs_mean']:.4f} "
              f"accAB={ov['acc_AB']:.3f} accBA={ov['acc_BA']:.3f} both={ov['both_correct']:.3f}",
              flush=True)
        for fam, fv in sorted(s["by_family"].items()):
            print(f"    {fam:14s}: n={fv['n']:3d} M={fv['M_mean']:+.4f} CM={fv['CM_mean']:+.4f} "
                  f"|bias|={fv['bias_abs_mean']:.4f} both={fv['both_correct']:.3f}", flush=True)
        for sty, sv in sorted(s["by_style"].items()):
            print(f"    style={sty:9s}: n={sv['n']:3d} M={sv['M_mean']:+.4f} both={sv['both_correct']:.3f}", flush=True)
        for es, ev in sorted(s["by_entity_split"].items()):
            print(f"    entity={es:9s}: n={ev['n']:3d} M={ev['M_mean']:+.4f} both={ev['both_correct']:.3f}", flush=True)

        del model
        torch.cuda.empty_cache()

    spath = OUT_DIR / "frozen_exchange_summary.json"
    with open(spath, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {spath}", flush=True)
    print(json.dumps({"event": "frozen_exchange_done",
        "checkpoints": list(targets.keys()), "summary": str(spath)}), flush=True)

if __name__ == "__main__":
    main()
