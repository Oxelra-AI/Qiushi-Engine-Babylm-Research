#!/usr/bin/env python3
"""research Phase1: Run an existing DeBERTa checkpoint on the counterbalanced binding eval.

Confirms the entity-event-state binding failure on the exact counterbalanced substrate
before any memory integration. Expected: affected_accuracy≈0, unaffected_accuracy≈1.

Usage:
  python3 deberta_binding_baseline.py \
    --model-path <path-to-chck_82M-or-similar> \
    --data-dir data/annotated_corpus \
    --out-dir data/deberta_binding_baseline
"""
import argparse, json, hashlib, time
from pathlib import Path
from collections import defaultdict, Counter
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out_dir or "data/deberta_binding_baseline")
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model
    print(f"loading model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    mask_id = tok.mask_token_id
    print(f"mask_token_id={mask_id}, vocab_size={tok.vocab_size}")

    # Load eval data
    eval_path = Path(args.data_dir) / "eval.jsonl"
    rows = [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]
    print(f"eval records: {len(rows)}")

    # Pre-encode answer/foil tokens
    for r in rows:
        aid = tok.encode(" " + r["answer"], add_special_tokens=False)
        fid = tok.encode(" " + r["foil"], add_special_tokens=False)
        # Use first subword if multi-piece, or skip the space prefix piece
        r["_answer_id"] = aid[1] if len(aid) > 1 else aid[0]
        r["_foil_id"] = fid[1] if len(fid) > 1 else fid[0]

    # Evaluate in batches
    results = []
    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [r["text"] for r in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**enc).logits  # (B, L, V)

        for j, r in enumerate(batch):
            # Find mask position
            ids = enc["input_ids"][j]
            mask_positions = (ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                results.append({**r, "correct": False, "margin": -999, "error": "no_mask"})
                continue
            mp = mask_positions[0].item()
            ans_logit = logits[j, mp, r["_answer_id"]].item()
            foil_logit = logits[j, mp, r["_foil_id"]].item()
            margin = ans_logit - foil_logit
            results.append({
                "id": r.get("id", ""),
                "kind": r["kind"],
                "split": r["split"],
                "family": r["family"],
                "is_affected": r.get("is_affected_query"),
                "answer": r["answer"],
                "foil": r["foil"],
                "event_order": r.get("event_order", "single"),
                "correct": margin > 0,
                "margin": margin,
            })

    elapsed = time.time() - t0
    print(f"inference: {elapsed:.1f}s over {len(rows)} items")

    # Aggregate by split
    by_split = defaultdict(lambda: {"correct": 0, "n": 0, "margin_sum": 0.0})
    for r in results:
        s = r["split"]
        by_split[s]["correct"] += int(r["correct"])
        by_split[s]["n"] += 1
        by_split[s]["margin_sum"] += r["margin"]

    # Selective updating for held_recomb
    held = [r for r in results if r["split"] == "eval_held_recomb" and r["kind"] == "binding"]
    affected = [r for r in held if r["is_affected"] == True]
    unaffected = [r for r in held if r["is_affected"] == False]
    aff_acc = sum(r["correct"] for r in affected) / max(len(affected), 1)
    unaff_acc = sum(r["correct"] for r in unaffected) / max(len(unaffected), 1)

    # By family
    by_family = defaultdict(lambda: {"affected_correct": 0, "affected_n": 0,
                                      "unaffected_correct": 0, "unaffected_n": 0})
    for r in held:
        fam = r["family"]
        if r["is_affected"]:
            by_family[fam]["affected_correct"] += int(r["correct"])
            by_family[fam]["affected_n"] += 1
        else:
            by_family[fam]["unaffected_correct"] += int(r["correct"])
            by_family[fam]["unaffected_n"] += 1

    # Multi-event
    multi = [r for r in results if r["kind"] == "multi_event"]
    multi_acc = sum(r["correct"] for r in multi) / max(len(multi), 1) if multi else None

    # Quartet consistency
    held_rows_by_qbase = defaultdict(list)
    for r in results:
        if r["split"] == "eval_held_recomb" and "quartet_id" in r:
            # The original row didn't carry quartet_id into results; re-fetch from source
            pass

    summary = {
        "status": "DEBERTA_BINDING_BASELINE",
        "model_path": args.model_path,
        "device": device,
        "eval_records": len(rows),
        "elapsed_sec": round(elapsed, 1),
        "by_split": {k: {"accuracy": v["correct"] / v["n"],
                         "n": v["n"],
                         "margin_mean": v["margin_sum"] / v["n"]}
                     for k, v in by_split.items()},
        "selective_updating": {
            "affected_accuracy": aff_acc,
            "affected_n": len(affected),
            "unaffected_accuracy": unaff_acc,
            "unaffected_n": len(unaffected),
            "EEBF_composite": (aff_acc + unaff_acc) / 2,
        },
        "by_family": {fam: {
            "affected_acc": v["affected_correct"] / max(v["affected_n"], 1),
            "unaffected_acc": v["unaffected_correct"] / max(v["unaffected_n"], 1),
            "affected_n": v["affected_n"],
            "unaffected_n": v["unaffected_n"],
        } for fam, v in sorted(by_family.items())},
        "multi_event_accuracy": multi_acc,
        "multi_event_n": len(multi),
    }

    (out / "deberta_binding_baseline.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
