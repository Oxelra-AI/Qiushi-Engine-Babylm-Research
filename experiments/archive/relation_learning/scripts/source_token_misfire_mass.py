#!/usr/bin/env python3
"""research: source-token misfire mass on held-out compact rewrite probes.

The research/14 decomposition showed that REPEAT can be better than CLEAN with an
unrelated source yet worse with the true source. This script tests a mechanistic
alternative: when the target token is deliberately non-overlapping with the true
source, does the model place excess probability mass on tokens occurring in the
source span? If yes, the true source may actively attract probability toward the
wrong identity tokens. If no, the cost is more consistent with displacement of
content reading or another non-misfire mechanism.

It rebuilds the same held-out rewrite true-source records used by the research
probe and scores all token-nonoverlap masks for selected arms/checkpoints. The
main quantity is source_mass = sum_{v in source token set} p(v) at the masked
position, excluding special tokens. Target tokens are non-overlap by construction
of the selected subset.
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
import pathlib
import random
import re
import statistics
import time
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = WS / "training" / "runs"
TOKENIZER = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "compliant_tokenizer"
ALL_ACCEPTED_PAIRS = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "expansion_analysis" / "combined_all_accepted_pairs.jsonl"
SELECTED_MAX_PAIRS = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_distribution_select" / "selected_matched_max_pairs.jsonl"
OUT_DEFAULT = WS / "data" / "source_token_misfire_mass"
CKS = ["chck_80M", "chck_90M", "chck_100M"]

ARM_CONFIGS = {
    "D_V_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_V_43222": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "D_C_43222": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel",
    "D_R_43222": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "RBT_V_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "roberta_view_dose2p64x_matched_rowholdout_100M_seed43022",
    "RBT_C_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022",
    "RBT_R_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len(str(text).split())


def mean(xs) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float("nan")


def pstdev(xs) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def pair_key(p: dict[str, Any]) -> str:
    pid = p.get("pair_id") or p.get("id")
    if pid is not None:
        return str(pid)
    return norm_text(str(p.get("source_text", ""))) + "\n" + norm_text(str(p.get("rewrite_text", "")))


def load_unselected_rewrite_pairs(max_pairs: int | None) -> list[dict[str, Any]]:
    selected_keys = {pair_key(p) for p in read_jsonl(SELECTED_MAX_PAIRS)}
    out: list[dict[str, Any]] = []
    for p in read_jsonl(ALL_ACCEPTED_PAIRS):
        src = str(p.get("source_text") or "").strip()
        rew = str(p.get("rewrite_text") or p.get("compact_rewrite") or "").strip()
        if not src or not rew:
            continue
        q = dict(p)
        q["source_text"] = src
        q["rewrite_text"] = rew
        if pair_key(q) in selected_keys:
            continue
        q.setdefault("pair_id", pair_key(q)[:80])
        out.append(q)
        if max_pairs is not None and len(out) >= max_pairs:
            break
    return out


def token_content_class(piece: str) -> bool:
    s = re.sub(r"[^A-Za-z0-9']+", "", str(piece)).lower()
    if len(s) < 2:
        return False
    if s in STOPWORDS:
        return False
    if all(ch.isdigit() for ch in s):
        return False
    return True


def start_end_mask(tokenizer) -> tuple[int, int, int]:
    start_tok = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end_tok = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask_tok = tokenizer.mask_token_id
    if start_tok is None or end_tok is None or mask_tok is None:
        raise RuntimeError("tokenizer lacks start/end/mask ids")
    return int(start_tok), int(end_tok), int(mask_tok)


def build_records(tokenizer, max_pairs: int | None, tokens_per_pair: int, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = load_unselected_rewrite_pairs(max_pairs)
    records: list[dict[str, Any]] = []
    stats = defaultdict(int)
    start_tok, end_tok, mask_tok = start_end_mask(tokenizer)
    special = set(int(x) for x in tokenizer.all_special_ids)
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = [int(x) for x in tokenizer(src_text, add_special_tokens=False)["input_ids"]]
        rew_enc = tokenizer(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = [int(x) for x in rew_enc["input_ids"]]
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > max_len:
            stats["too_long_or_empty"] += 1
            continue
        src_set = {x for x in src_ids if x not in special}
        if not src_set:
            stats["empty_source_set"] += 1
            continue
        chosen: list[int] = []
        for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a:
                continue
            piece = rew_text[a:b]
            target = int(rew_ids[j])
            if target in src_set:
                continue
            if not token_content_class(piece):
                continue
            chosen.append(j)
        if not chosen:
            stats["no_nonoverlap_content_tokens"] += 1
            continue
        if len(chosen) > tokens_per_pair:
            picks = [chosen[round(k * (len(chosen) - 1) / (tokens_per_pair - 1))] for k in range(tokens_per_pair)] if tokens_per_pair > 1 else [chosen[len(chosen)//2]]
        else:
            picks = chosen
        body = list(src_ids) + list(rew_ids)
        ids = [start_tok] + body + [end_tok]
        for j in picks:
            target = int(rew_ids[j])
            body_pos = len(src_ids) + j
            pos = body_pos + 1
            masked = list(ids)
            masked[pos] = mask_tok
            records.append({
                "input_ids": masked,
                "attention_mask": [1] * len(masked),
                "position": pos,
                "target": target,
                "probe_id": f"misfire:{i}:{j}:{p.get('pair_id')}",
                "pair_index": i,
                "pair_id": p.get("pair_id"),
                "sentence_id": p.get("sentence_id"),
                "doc_id": p.get("doc_id"),
                "source_words": int(p.get("source_words", wc(src_text))),
                "rewrite_words": int(p.get("rewrite_words", wc(rew_text))),
                "source_token_len": len(src_ids),
                "rewrite_token_len": len(rew_ids),
                "target_token_id": target,
                "source_token_ids": sorted(src_set),
                "source_type_count": len(src_set),
                "seq_len": len(masked),
            })
        stats["pairs_used"] += 1
        stats["records"] += len(picks)
    stats["unselected_pairs_loaded"] = len(pairs)
    return records, dict(stats)


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
        ids = ids.to(device)
        att = att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            pos = int(r["position"])
            target = int(r["target"])
            pr = probs[i, pos]
            src_ids = torch.tensor(r["source_token_ids"], dtype=torch.long, device=device)
            source_mass = float(pr.index_select(0, src_ids).sum().detach().cpu())
            target_prob = float(pr[target].detach().cpu())
            topk_vals, topk_idx = torch.topk(pr, k=min(10, pr.shape[-1]))
            top_ids = [int(x) for x in topk_idx.detach().cpu().tolist()]
            top_probs = [float(x) for x in topk_vals.detach().cpu().tolist()]
            rank = int((pr > pr[target]).sum().detach().cpu()) + 1
            meta = {k: v for k, v in r.items() if k not in {"input_ids", "attention_mask", "source_token_ids"}}
            meta.update({
                "source_mass": source_mass,
                "target_prob": target_prob,
                "source_to_target_mass_ratio": source_mass / max(target_prob, 1e-12),
                "target_rank": rank,
                "top10_ids": " ".join(str(x) for x in top_ids),
                "top10_probs": " ".join(f"{x:.8g}" for x in top_probs),
                "top1_in_source": int(top_ids[0] in set(r["source_token_ids"])),
                "top10_source_hits": sum(int(x in set(r["source_token_ids"])) for x in top_ids),
            })
            rows.append(meta)
    return rows


def parse_arm(arm: str) -> tuple[str, str, str]:
    parts = arm.split("_")
    seed = parts[-1] if parts and parts[-1].isdigit() else "NA"
    role = parts[-2] if len(parts) >= 2 else ""
    arch = "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
    return arch, role, seed


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    arm_terms = []
    d = defaultdict(list)
    for r in rows:
        d[(r["arch"], r["seed"], r["role"], r["checkpoint"])].append(r)
    for (arch, seed, role, ck), vals in sorted(d.items()):
        arm_terms.append({
            "arch": arch,
            "seed": seed,
            "role": role,
            "checkpoint": ck,
            "n": len(vals),
            "mean_source_mass": mean([v["source_mass"] for v in vals]),
            "mean_target_prob": mean([v["target_prob"] for v in vals]),
            "mean_source_to_target_mass_ratio": mean([v["source_to_target_mass_ratio"] for v in vals]),
            "top1_source_rate": mean([v["top1_in_source"] for v in vals]),
            "mean_top10_source_hits": mean([v["top10_source_hits"] for v in vals]),
            "mean_target_rank": mean([v["target_rank"] for v in vals]),
        })
    late_d = defaultdict(list)
    for r in arm_terms:
        if r["checkpoint"] in CKS:
            late_d[(r["arch"], r["seed"], r["role"])].append(r)
    late_terms = []
    for (arch, seed, role), vals in sorted(late_d.items()):
        late_terms.append({
            "arch": arch,
            "seed": seed,
            "role": role,
            "n_checkpoints": len(vals),
            "n": int(vals[0]["n"]),
            "mean_source_mass": mean([v["mean_source_mass"] for v in vals]),
            "mean_target_prob": mean([v["mean_target_prob"] for v in vals]),
            "mean_source_to_target_mass_ratio": mean([v["mean_source_to_target_mass_ratio"] for v in vals]),
            "top1_source_rate": mean([v["top1_source_rate"] for v in vals]),
            "mean_top10_source_hits": mean([v["mean_top10_source_hits"] for v in vals]),
            "mean_target_rank": mean([v["mean_target_rank"] for v in vals]),
        })
    idx = {(r["arch"], r["seed"], r["role"]): r for r in late_terms}
    contrasts = []
    for arch, seed in sorted({(r["arch"], r["seed"]) for r in late_terms}):
        for a, b in [("R", "C"), ("V", "C"), ("V", "R")]:
            ra = idx.get((arch, seed, a)); rb = idx.get((arch, seed, b))
            if not ra or not rb:
                continue
            contrasts.append({
                "arch": arch,
                "seed": seed,
                "contrast": f"{a}minus{b}",
                "n_min": min(int(ra["n"]), int(rb["n"])),
                "source_mass_delta": ra["mean_source_mass"] - rb["mean_source_mass"],
                "target_prob_delta": ra["mean_target_prob"] - rb["mean_target_prob"],
                "ratio_delta": ra["mean_source_to_target_mass_ratio"] - rb["mean_source_to_target_mass_ratio"],
                "top1_source_rate_delta": ra["top1_source_rate"] - rb["top1_source_rate"],
                "top10_source_hits_delta": ra["mean_top10_source_hits"] - rb["mean_top10_source_hits"],
                "target_rank_delta": ra["mean_target_rank"] - rb["mean_target_rank"],
            })
    return arm_terms, late_terms, contrasts


def write_note(out_dir: pathlib.Path, late_terms: list[dict], contrasts: list[dict], plan: dict[str, Any]) -> pathlib.Path:
    lines = []
    lines.append("# research source-token misfire mass")
    lines.append("")
    lines.append("For each held-out compact-rewrite token-nonoverlap target, the true source is present and the masked target token is absent from the source token set. The readout measures the probability mass placed on source-span token types at the masked target position. Elevated REPEAT source mass would support an identity-misfire account of the active recurrence cost; no elevation would support displacement or another mechanism.")
    lines.append("")
    lines.append("## Late means over 80M/90M/100M")
    lines.append("")
    lines.append("| arch | seed | role | source mass | target prob | source/target ratio | top1 source rate | mean top10 source hits | n |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in late_terms:
        lines.append(f"| {r['arch']} | {r['seed']} | {r['role']} | {r['mean_source_mass']:.5f} | {r['mean_target_prob']:.5f} | {r['mean_source_to_target_mass_ratio']:.3f} | {r['top1_source_rate']:.4f} | {r['mean_top10_source_hits']:.3f} | {int(r['n'])} |")
    lines.append("")
    lines.append("## Key contrasts")
    lines.append("")
    lines.append("| arch | seed | contrast | source-mass Δ | target-prob Δ | ratio Δ | top1-source-rate Δ | target-rank Δ |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in contrasts:
        if r["contrast"] in {"RminusC", "VminusC", "VminusR"}:
            lines.append(f"| {r['arch']} | {r['seed']} | {r['contrast']} | {r['source_mass_delta']:+.5f} | {r['target_prob_delta']:+.5f} | {r['ratio_delta']:+.3f} | {r['top1_source_rate_delta']:+.4f} | {r['target_rank_delta']:+.1f} |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"Output directory: `{rel(out_dir)}`")
    lines.append(f"Plan: `{rel(out_dir / 'misfire_plan.json')}`")
    note = WS / "notes" / "source_token_misfire_mass.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["D_V_43022", "D_C_43022", "D_R_43022", "D_V_43122", "D_C_43122", "D_R_43122", "D_V_43222", "D_C_43222", "D_R_43222"])
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-pairs", type=int, default=1626)
    ap.add_argument("--tokens-per-pair", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    records, stats = build_records(tokenizer, None if args.max_pairs <= 0 else args.max_pairs, args.tokens_per_pair, args.max_len)
    plan = {
        "status": "MISFIRE_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "record_count": len(records),
        "record_stats": stats,
        "scoring": "true-source compact rewrite token-nonoverlap masks; source_mass=sum probability on source span token types",
        "mechanism_prediction": {
            "identity_misfire": "REPEAT should have clearly higher source_mass than CLEAN and the elevation should align with its excess true-source cost.",
            "displacement_without_misfire": "REPEAT should not have higher source_mass; cost arises from failing to transform/use content rather than assigning probability to source tokens."
        },
    }
    (out_dir / "misfire_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    all_rows: list[dict] = []
    for arm in args.arms:
        run_dir = ARM_CONFIGS.get(arm)
        if run_dir is None:
            raise KeyError(f"unknown arm {arm}")
        if not run_dir.exists():
            print(f"[SKIP] {arm}: missing run dir {run_dir}", flush=True)
            continue
        arch, role, seed = parse_arm(arm)
        for ck in args.checkpoints:
            model_path = run_dir / "hf_model" / ck
            if not model_path.exists():
                print(f"[SKIP] {arm} {ck}: missing {model_path}", flush=True)
                continue
            print(f"[LOAD] {arm} {ck} {rel(model_path)}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32).to(device)
            model.eval()
            rows = score_records(model, records, device, int(pad_id), args.batch_size)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for r in rows:
                r.update({"arm": arm, "arch": arch, "role": role, "seed": seed, "checkpoint": ck})
            out_path = out_dir / f"misfire_rows_{arm}_{ck}.csv"
            write_csv(out_path, rows)
            all_rows.extend(rows)
            print(f"[DONE] {arm} {ck}: rows={len(rows)}", flush=True)
    write_csv(out_dir / "misfire_rows_all.csv", all_rows)
    arm_terms, late_terms, contrasts = summarize(all_rows)
    write_csv(out_dir / "misfire_arm_terms_by_checkpoint.csv", arm_terms)
    write_csv(out_dir / "misfire_late_terms.csv", late_terms)
    write_csv(out_dir / "misfire_late_contrasts.csv", contrasts)
    note = write_note(out_dir, late_terms, contrasts, plan)
    result = {"status": "SOURCE_TOKEN_MISFIRE_DONE", "finished_utc": now(), "note": rel(note), "outputs": rel(out_dir), "rows": len(all_rows)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
