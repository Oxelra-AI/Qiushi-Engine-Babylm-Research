#!/usr/bin/env python3
"""research: Cross-loss analysis for fixed-budget persistence mechanism.

The research persistence analysis measured each trained arm on its own changed vs
unchanged rows. That showed repeat changed rows become very low-loss, breadth rows
remain high-loss, and view rows are intermediate/low. This script asks the more
causal question: how hard are the same admitted/displaced text sets for each
trained model?

It evaluates small deterministic samples from four text sets:
  - view_changed: compact-view admitted rows
  - repeat_changed: duplicate/repeat admitted rows
  - breadth_changed: independent FineWeb breadth rows
  - clean_displaced: original clean rows at the corresponding replacement positions
under four models:
  - clean, view, repeat, breadth
at selected checkpoints. CPU-only, no new training, no official benchmark scoring.

Key readouts:
  * clean-prior difficulty of each admitted text set.
  * arm specialization: clean_loss(textset) - arm_loss(textset).
  * displacement cost: intervention_loss(clean_displaced) - clean_loss(clean_displaced).
  * whether repeat's low own loss is already present in the clean model, indicating
    redundancy rather than transferable learning signal.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"
OUT_DIR = WS / "data" / "persistence_crossloss_analysis"

POOL_FILES = {
    "view_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_10M.jsonl",
    "repeat_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_10M.jsonl",
    "breadth_changed": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_10M.jsonl",
    "clean_displaced": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
}

META_FILES = {
    "view_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
    "breadth_changed": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_changed_block_rows_meta.jsonl",
    "clean_displaced": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl",
}

MODEL_RUNS = {
    "view": WS / "training" / "runs" / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "repeat": WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "breadth": WS / "training" / "runs" / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "clean": WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
}

DEFAULT_CKS = ["chck_40M", "chck_80M", "chck_100M"]
MASK_PROB = 0.15
MAX_SEQ_LEN = 256
BASE_MASK_SEED = 293293


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_indices(path: pathlib.Path) -> list[int]:
    out: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(int(json.loads(line)["row_index"]))
    return out


def load_texts_at_indices(pool: pathlib.Path, indices: list[int], n: int, seed: int) -> list[dict[str, Any]]:
    wanted = set(indices)
    rows: list[dict[str, Any]] = []
    with open(pool, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in wanted:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text.strip():
                rows.append({"row_index": idx, "text": text})
    rng = random.Random(seed)
    if len(rows) > n:
        rows = rng.sample(rows, n)
    rows.sort(key=lambda r: r["row_index"])
    return rows


def deterministic_positions(input_ids, special_ids: set[int], text: str, mask_prob: float, tokenizer) -> list[int]:
    maskable = [i for i in range(input_ids.shape[1]) if int(input_ids[0, i]) not in special_ids]
    if not maskable:
        return []
    n_mask = max(1, int(len(maskable) * mask_prob))
    h = hashlib.sha256((str(BASE_MASK_SEED) + "\n" + text).encode("utf-8", errors="ignore")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return rng.sample(maskable, min(n_mask, len(maskable)))


def mean_loss(model, tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    total_loss = 0.0
    total_masked = 0
    n_rows_used = 0
    n_subword_tokens = 0
    with torch.no_grad():
        for r in rows:
            text = r["text"]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=False)
            input_ids = enc["input_ids"].clone()
            positions = deterministic_positions(input_ids, special_ids, text, MASK_PROB, tokenizer)
            if not positions:
                continue
            labels = torch.full_like(input_ids, -100)
            for pos in positions:
                labels[0, pos] = input_ids[0, pos]
                input_ids[0, pos] = tokenizer.mask_token_id
            outputs = model(input_ids=input_ids, attention_mask=enc["attention_mask"], labels=labels)
            nm = int((labels != -100).sum().item())
            total_loss += float(outputs.loss.item()) * nm
            total_masked += nm
            n_rows_used += 1
            n_subword_tokens += int(enc["attention_mask"].sum().item())
    return {
        "mean_loss": round(total_loss / total_masked, 6) if total_masked else None,
        "n_rows": n_rows_used,
        "n_tokens_masked": total_masked,
        "n_subword_tokens": n_subword_tokens,
    }


def summarize(results: list[dict[str, Any]], checkpoints: list[str]) -> dict[str, Any]:
    # index[(model, ck, textset)] = loss
    idx: dict[tuple[str, str, str], float] = {}
    for r in results:
        if r.get("mean_loss") is not None:
            idx[(r["model_arm"], r["checkpoint"], r["text_set"])] = float(r["mean_loss"])

    derived: list[dict[str, Any]] = []
    for ck in checkpoints:
        for textset in ["view_changed", "repeat_changed", "breadth_changed", "clean_displaced"]:
            clean_loss = idx.get(("clean", ck, textset))
            if clean_loss is None:
                continue
            # owning model for this text set
            owner = {
                "view_changed": "view",
                "repeat_changed": "repeat",
                "breadth_changed": "breadth",
                "clean_displaced": "clean",
            }[textset]
            owner_loss = idx.get((owner, ck, textset))
            if owner_loss is not None:
                derived.append({
                    "checkpoint": ck,
                    "text_set": textset,
                    "metric": "clean_minus_owner_loss",
                    "clean_loss": round(clean_loss, 6),
                    "owner_arm": owner,
                    "owner_loss": round(owner_loss, 6),
                    "delta": round(clean_loss - owner_loss, 6),
                    "interpretation": "positive means admitted arm learned/predicts this text set better than clean",
                })
            if textset == "clean_displaced":
                for arm in ["view", "repeat", "breadth"]:
                    loss = idx.get((arm, ck, textset))
                    if loss is not None:
                        derived.append({
                            "checkpoint": ck,
                            "text_set": textset,
                            "metric": f"{arm}_minus_clean_loss_on_displaced",
                            "intervention_loss": round(loss, 6),
                            "clean_loss": round(clean_loss, 6),
                            "delta": round(loss - clean_loss, 6),
                            "interpretation": "positive means intervention model predicts sacrificed clean rows worse",
                        })

    # Compact checkpoint-level summary for the 100M mechanism reading.
    by_ck: dict[str, dict[str, Any]] = {}
    for ck in checkpoints:
        entry: dict[str, Any] = {"checkpoint": ck}
        for textset, owner in [("view_changed", "view"), ("repeat_changed", "repeat"), ("breadth_changed", "breadth"), ("clean_displaced", "clean")]:
            entry[f"clean_loss_on_{textset}"] = idx.get(("clean", ck, textset))
            entry[f"owner_loss_on_{textset}"] = idx.get((owner, ck, textset))
            if entry[f"clean_loss_on_{textset}"] is not None and entry[f"owner_loss_on_{textset}"] is not None:
                entry[f"clean_minus_owner_{textset}"] = round(entry[f"clean_loss_on_{textset}"] - entry[f"owner_loss_on_{textset}"], 6)
        by_ck[ck] = entry
    return {"derived_records": derived, "checkpoint_summary": by_ck}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--text-sets", nargs="*", default=["view_changed", "repeat_changed", "breadth_changed", "clean_displaced"])
    ap.add_argument("--model-arms", nargs="*", default=["clean", "view", "repeat", "breadth"])
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for ts in args.text_sets:
        if ts not in POOL_FILES:
            missing.append(f"unknown_text_set:{ts}")
            continue
        if not POOL_FILES[ts].exists():
            missing.append(f"pool:{ts}")
        if not META_FILES[ts].exists():
            missing.append(f"meta:{ts}")
    for arm in args.model_arms:
        if arm not in MODEL_RUNS:
            missing.append(f"unknown_model_arm:{arm}")
            continue
        for ck in args.checkpoints:
            if not (MODEL_RUNS[arm] / "hf_model" / ck).exists():
                missing.append(f"model:{arm}/{ck}")

    plan = {
        "status": "CROSSLOSS_PLAN",
        "created_utc": now(),
        "out_dir": rel(OUT_DIR),
        "model_arms": args.model_arms,
        "text_sets": args.text_sets,
        "checkpoints": args.checkpoints,
        "n_per_text_set": args.n,
        "measurements": len(args.model_arms) * len(args.text_sets) * len(args.checkpoints),
        "cpu_only": True,
        "no_training_no_official_eval_no_upload": True,
        "missing": missing,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if missing:
        raise SystemExit("Missing inputs: " + "; ".join(missing[:20]))

    print(f"Loading tokenizer from {rel(TOKENIZER_DIR)}", flush=True)
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    text_samples: dict[str, list[dict[str, Any]]] = {}
    for i, ts in enumerate(args.text_sets):
        indices = load_indices(META_FILES[ts])
        rows = load_texts_at_indices(POOL_FILES[ts], indices, args.n, seed=29300 + i)
        text_samples[ts] = rows
        print(f"Text set {ts}: sampled {len(rows)} rows from {len(indices)} changed indices", flush=True)

    results: list[dict[str, Any]] = []
    total_models = len(args.model_arms) * len(args.checkpoints)
    done_models = 0
    for arm in args.model_arms:
        for ck in args.checkpoints:
            done_models += 1
            model_path = MODEL_RUNS[arm] / "hf_model" / ck
            print(f"\n[{done_models}/{total_models}] Loading {arm} {ck} from {rel(model_path)}", flush=True)
            t0 = time.time()
            model = AutoModelForMaskedLM.from_pretrained(str(model_path))
            model.eval()
            print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
            for ts, rows in text_samples.items():
                t1 = time.time()
                info = mean_loss(model, tokenizer, rows)
                rec = {
                    "model_arm": arm,
                    "checkpoint": ck,
                    "text_set": ts,
                    **info,
                    "eval_time_sec": round(time.time() - t1, 1),
                }
                results.append(rec)
                print(f"  {ts:<16} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
            del model
            import gc
            gc.collect()

    summary = summarize(results, args.checkpoints)
    output = {
        "status": "CROSSLOSS_DONE",
        "finished_utc": now(),
        "parameters": {
            "n_per_text_set": args.n,
            "checkpoints": args.checkpoints,
            "model_arms": args.model_arms,
            "text_sets": args.text_sets,
            "mask_prob": MASK_PROB,
            "base_mask_seed": BASE_MASK_SEED,
            "max_seq_len": MAX_SEQ_LEN,
            "cpu_only": True,
        },
        "samples": {ts: [{"row_index": r["row_index"], "sha12": hashlib.sha256(r["text"].encode("utf-8", errors="ignore")).hexdigest()[:12]} for r in rows] for ts, rows in text_samples.items()},
        "results": results,
        **summary,
    }
    out_json = OUT_DIR / "crossloss_results.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / "crossloss_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["model_arm", "checkpoint", "text_set", "mean_loss", "n_rows", "n_tokens_masked", "n_subword_tokens", "eval_time_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})

    md_lines = [
        "# research persistence cross-loss analysis",
        "",
        "CPU-only cross evaluation of admitted/displaced text sets under clean/view/repeat/breadth models.",
        "Positive `clean_minus_owner` means the intervention model predicts its admitted text better than the clean model; near-zero means the clean model already had the text distribution covered.",
        "",
        "## Checkpoint summary",
        "",
    ]
    for ck in args.checkpoints:
        entry = summary["checkpoint_summary"].get(ck, {})
        md_lines.append(f"### {ck}")
        for ts in ["view_changed", "repeat_changed", "breadth_changed", "clean_displaced"]:
            cl = entry.get(f"clean_loss_on_{ts}")
            ol = entry.get(f"owner_loss_on_{ts}")
            de = entry.get(f"clean_minus_owner_{ts}")
            md_lines.append(f"- {ts}: clean_loss={cl}, owner_loss={ol}, clean_minus_owner={de}")
        md_lines.append("")
    md_lines += [
        "## Files",
        f"- JSON: `{rel(out_json)}`",
        f"- CSV: `{rel(csv_path)}`",
    ]
    out_md = OUT_DIR / "crossloss_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nSaved {rel(out_json)}", flush=True)
    print(f"Saved {rel(csv_path)}", flush=True)
    print(f"Saved {rel(out_md)}", flush=True)
    print(json.dumps({"status": output["status"], "n_results": len(results), "out_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
