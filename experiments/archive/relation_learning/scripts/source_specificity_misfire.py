#!/usr/bin/env python3
"""research: source-specificity misfire control.

research showed REPEAT places ~+0.10 more mass on source-span tokens than CLEAN at
nonoverlap rewrite targets.  But CLEAN already places 0.37 on source tokens, largely
because source spans share most function-word vocabulary.

This script separates true in-window copy misfire from frequency hedging by measuring
source CONTENT token mass under two conditions:
  T  – true source in window:       [CLS] TRUE_SRC  REWRITE [SEP]
  U  – unrelated source in window:  [CLS] UNREL_SRC REWRITE [SEP]

At each masked nonoverlap target, measure mass on:
  true_src_content_tokens   – content tokens from the TRUE source
  unrel_src_content_tokens  – content tokens from the UNRELATED source

This yields a 2×2 (token_set × condition) where:

  tokens \ condition | T (true src) | U (unrel src)
  ─────────────────────────────────────────────────
  true_src_content   | IN-WINDOW    | OUT-OF-WINDOW
  unrel_src_content  | OUT-OF-WINDOW| IN-WINDOW

Copy mechanism:  R−C positive for IN-WINDOW cells, near zero for OUT-OF-WINDOW cells.
Frequency hedge: R−C positive for ALL cells regardless of in-window status.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, csv, json, math, pathlib, random, re, statistics, time
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
OUT_DEFAULT = WS / "data" / "source_specificity_misfire"
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
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "just", "so", "very", "also", "about", "more", "some", "any", "all",
    "each", "every", "both", "few", "many", "much", "such", "own", "other", "up", "out",
}


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except: return str(p)

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def wc(text: str) -> int:
    return len(str(text).split())

def mean(xs) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float("nan")

def read_jsonl(path, limit=None):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                yield json.loads(line)

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")

def pair_key(p: dict) -> str:
    pid = p.get("pair_id") or p.get("id")
    if pid is not None: return str(pid)
    return norm_text(str(p.get("source_text", ""))) + "\n" + norm_text(str(p.get("rewrite_text", "")))


def token_content_class(piece: str) -> bool:
    s = re.sub(r"[^A-Za-z0-9']+", "", str(piece)).lower()
    return len(s) >= 2 and s not in STOPWORDS and not s.isdigit()


def start_end_mask(tokenizer):
    start = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask = tokenizer.mask_token_id
    assert start is not None and end is not None and mask is not None
    return int(start), int(end), int(mask)


def build_content_set(token_ids: list[int], tokenizer, special: set[int]) -> set[int]:
    """Content tokens only: exclude stopwords, short, digits, special."""
    out = set()
    for tid in token_ids:
        if tid in special:
            continue
        piece = tokenizer.decode([tid])
        if token_content_class(piece):
            out.add(tid)
    return out


def load_unselected_rewrite_pairs(max_pairs=None):
    selected_keys = {pair_key(p) for p in read_jsonl(SELECTED_MAX_PAIRS)}
    out = []
    for p in read_jsonl(ALL_ACCEPTED_PAIRS):
        src = str(p.get("source_text") or "").strip()
        rew = str(p.get("rewrite_text") or p.get("compact_rewrite") or "").strip()
        if not src or not rew:
            continue
        q = dict(p); q["source_text"] = src; q["rewrite_text"] = rew
        if pair_key(q) in selected_keys:
            continue
        q.setdefault("pair_id", pair_key(q)[:80])
        out.append(q)
        if max_pairs and len(out) >= max_pairs:
            break
    return out


def build_records(tokenizer, max_pairs, tokens_per_pair, max_len):
    """Build dual-condition records for source-specificity test."""
    pairs = load_unselected_rewrite_pairs(max_pairs)
    N = len(pairs)
    if N < 10:
        raise RuntimeError(f"too few pairs: {N}")

    start_tok, end_tok, mask_tok = start_end_mask(tokenizer)
    special = set(int(x) for x in tokenizer.all_special_ids)
    stats = defaultdict(int)
    records = []

    # Pre-tokenize all sources for unrelated assignment
    src_ids_all = []
    for p in pairs:
        ids = [int(x) for x in tokenizer(p["source_text"], add_special_tokens=False)["input_ids"]]
        src_ids_all.append(ids)

    for i, p in enumerate(pairs):
        src_text = " ".join(p["source_text"].split())
        rew_text = " ".join(p["rewrite_text"].split())

        src_ids = src_ids_all[i]
        rew_enc = tokenizer(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = [int(x) for x in rew_enc["input_ids"]]

        if not src_ids or not rew_ids:
            stats["empty"] += 1; continue
        if len(src_ids) + len(rew_ids) + 2 > max_len:
            stats["too_long"] += 1; continue

        # Find length-matched unrelated source (different pair)
        target_len = len(src_ids)
        best_j, best_diff = None, float("inf")
        for offset in range(N // 4, N // 4 + N // 2):
            j = (i + offset) % N
            if j == i: continue
            diff = abs(len(src_ids_all[j]) - target_len)
            if diff < best_diff:
                best_j, best_diff = j, diff
                if diff == 0: break
        if best_j is None:
            stats["no_unrel"] += 1; continue

        unrel_ids = list(src_ids_all[best_j])
        # Truncate/trim to exactly match true source length for position consistency
        if len(unrel_ids) > len(src_ids):
            unrel_ids = unrel_ids[:len(src_ids)]
        elif len(unrel_ids) < len(src_ids):
            # Pad with pad token to match length; positions will be masked by attention
            pass  # accept length mismatch if close enough

        if len(unrel_ids) + len(rew_ids) + 2 > max_len:
            unrel_ids = unrel_ids[:max_len - len(rew_ids) - 2]

        # Content token sets
        src_content = build_content_set(src_ids, tokenizer, special)
        unrel_content = build_content_set(unrel_ids, tokenizer, special)
        if not src_content or not unrel_content:
            stats["no_content"] += 1; continue

        # Source token set (all non-special) for compatibility with research
        src_set = {x for x in src_ids if x not in special}

        # Find nonoverlap content positions in rewrite
        chosen = []
        for j_r, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a: continue
            piece = rew_text[a:b]
            target_tok = int(rew_ids[j_r])
            if target_tok in src_set: continue  # overlap with true source
            if not token_content_class(piece): continue
            chosen.append(j_r)

        if not chosen:
            stats["no_nonoverlap"] += 1; continue

        # Subsample positions
        if len(chosen) > tokens_per_pair:
            step = (len(chosen) - 1) / max(tokens_per_pair - 1, 1)
            picks = [chosen[round(k * step)] for k in range(tokens_per_pair)]
        else:
            picks = chosen

        for j_r in picks:
            target_tok = int(rew_ids[j_r])

            # Condition T: [CLS] src_ids rew_ids [SEP]
            body_T = list(src_ids) + list(rew_ids)
            pos_T = 1 + len(src_ids) + j_r
            ids_T = [start_tok] + body_T + [end_tok]
            ids_T_m = list(ids_T); ids_T_m[pos_T] = mask_tok

            # Condition U: [CLS] unrel_ids rew_ids [SEP]
            body_U = list(unrel_ids) + list(rew_ids)
            pos_U = 1 + len(unrel_ids) + j_r
            ids_U = [start_tok] + body_U + [end_tok]
            ids_U_m = list(ids_U); ids_U_m[pos_U] = mask_tok

            meta = {
                "pair_index": i, "pair_id": p.get("pair_id"),
                "target_token_id": target_tok, "rewrite_pos": j_r,
                "n_true_src_content": len(src_content),
                "n_unrel_src_content": len(unrel_content),
                "src_len": len(src_ids), "unrel_len": len(unrel_ids),
            }

            for cond, ids_m, pos in [("T", ids_T_m, pos_T), ("U", ids_U_m, pos_U)]:
                records.append({
                    "input_ids": ids_m, "attention_mask": [1] * len(ids_m),
                    "position": pos, "target": target_tok,
                    "condition": cond,
                    "true_src_content_ids": sorted(src_content),
                    "unrel_src_content_ids": sorted(unrel_content),
                    **meta,
                })

        stats["pairs_used"] += 1
        stats["records"] += len(picks) * 2

    stats["total_pairs_loaded"] = N
    return records, dict(stats)


@torch.no_grad()
def score_records(model, records, device, pad_id, batch_size):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        mx = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), mx), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), mx), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        probs = torch.nn.functional.softmax(logits, dim=-1)

        for i, r in enumerate(batch):
            pr = probs[i, r["position"]]
            target = r["target"]
            target_prob = float(pr[target].cpu())

            tsc = torch.tensor(r["true_src_content_ids"], dtype=torch.long, device=device)
            true_src_content_mass = float(pr.index_select(0, tsc).sum().cpu())

            usc = torch.tensor(r["unrel_src_content_ids"], dtype=torch.long, device=device)
            unrel_src_content_mass = float(pr.index_select(0, usc).sum().cpu())

            meta = {k: v for k, v in r.items()
                    if k not in {"input_ids", "attention_mask", "true_src_content_ids", "unrel_src_content_ids"}}
            meta.update({
                "true_src_content_mass": true_src_content_mass,
                "unrel_src_content_mass": unrel_src_content_mass,
                "target_prob": target_prob,
            })
            rows.append(meta)
    return rows


def parse_arm(arm: str):
    parts = arm.split("_")
    seed = parts[-1] if parts and parts[-1].isdigit() else "NA"
    role = parts[-2] if len(parts) >= 2 else ""
    arch = "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
    return arch, role, seed


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n"); return
    flds = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds); w.writeheader(); w.writerows(rows)


def summarize(rows):
    """Aggregate by arch × seed × role × condition, compute late means."""
    d = defaultdict(list)
    for r in rows:
        d[(r["arch"], r["seed"], r["role"], r["condition"], r["checkpoint"])].append(r)

    ck_terms = []
    for (arch, seed, role, cond, ck), vals in sorted(d.items()):
        ck_terms.append({
            "arch": arch, "seed": seed, "role": role, "condition": cond, "checkpoint": ck,
            "n": len(vals),
            "mean_true_src_content_mass": mean([v["true_src_content_mass"] for v in vals]),
            "mean_unrel_src_content_mass": mean([v["unrel_src_content_mass"] for v in vals]),
            "mean_target_prob": mean([v["target_prob"] for v in vals]),
        })

    # Late means
    late_d = defaultdict(list)
    for r in ck_terms:
        if r["checkpoint"] in CKS:
            late_d[(r["arch"], r["seed"], r["role"], r["condition"])].append(r)
    late_terms = []
    for (arch, seed, role, cond), vals in sorted(late_d.items()):
        late_terms.append({
            "arch": arch, "seed": seed, "role": role, "condition": cond,
            "n_checkpoints": len(vals), "n": vals[0]["n"],
            "mean_true_src_content_mass": mean([v["mean_true_src_content_mass"] for v in vals]),
            "mean_unrel_src_content_mass": mean([v["mean_unrel_src_content_mass"] for v in vals]),
            "mean_target_prob": mean([v["mean_target_prob"] for v in vals]),
        })

    # Contrasts: R−C, V−C, V−R within each condition
    idx = {(r["arch"], r["seed"], r["role"], r["condition"]): r for r in late_terms}
    contrasts = []
    for (arch, seed) in sorted({(r["arch"], r["seed"]) for r in late_terms}):
        for cond in ["T", "U"]:
            for a_name, b_name in [("R", "C"), ("V", "C"), ("V", "R")]:
                ra = idx.get((arch, seed, a_name, cond))
                rb = idx.get((arch, seed, b_name, cond))
                if not ra or not rb:
                    continue
                contrasts.append({
                    "arch": arch, "seed": seed, "contrast": f"{a_name}minus{b_name}",
                    "condition": cond, "n_min": min(int(ra["n"]), int(rb["n"])),
                    "true_src_content_mass_delta": ra["mean_true_src_content_mass"] - rb["mean_true_src_content_mass"],
                    "unrel_src_content_mass_delta": ra["mean_unrel_src_content_mass"] - rb["mean_unrel_src_content_mass"],
                    "target_prob_delta": ra["mean_target_prob"] - rb["mean_target_prob"],
                })

    return ck_terms, late_terms, contrasts


def write_note(out_dir, late_terms, contrasts, plan):
    lines = ["# research source-specificity misfire control", "",
        "Tests whether REPEAT's source-token mass elevation is specific to having the TRUE source "
        "in-window (copy mechanism) or persists with an unrelated source (frequency hedging).", "",
        "## Design", "",
        "At each nonoverlap rewrite mask, content-token mass is measured under two conditions:", "",
        "| token_set \\ condition | T (true source in window) | U (unrelated source in window) |",
        "|---|---|---|",
        "| true_src_content | **IN-WINDOW** | OUT-OF-WINDOW |",
        "| unrel_src_content | OUT-OF-WINDOW | **IN-WINDOW** |", "",
        "Copy mechanism: R−C elevated for IN-WINDOW cells only.",
        "Frequency hedge: R−C elevated for all cells.", "",
        "## Late means (80M/90M/100M)", ""]

    lines.append("| arch | seed | role | cond | true_src_content_mass | unrel_src_content_mass | target_prob | n |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for r in late_terms:
        lines.append(f"| {r['arch']} | {r['seed']} | {r['role']} | {r['condition']} | "
                     f"{r['mean_true_src_content_mass']:.5f} | {r['mean_unrel_src_content_mass']:.5f} | "
                     f"{r['mean_target_prob']:.5f} | {r['n']} |")

    lines += ["", "## Key contrasts (R−C is decisive)", ""]
    lines.append("| arch | seed | contrast | cond | true_src_content_Δ | unrel_src_content_Δ | target_prob_Δ |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for r in contrasts:
        lines.append(f"| {r['arch']} | {r['seed']} | {r['contrast']} | {r['condition']} | "
                     f"{r['true_src_content_mass_delta']:+.5f} | {r['unrel_src_content_mass_delta']:+.5f} | "
                     f"{r['target_prob_delta']:+.5f} |")

    lines += ["", "## Interpretation", "",
        "If R−C `true_src_content_Δ` is positive under T but near zero under U, the copy mechanism "
        "fires specifically when the true source is in-window. If positive under both, frequency hedging "
        "or generic model uncertainty explains the elevation.", "",
        f"Output: `{rel(out_dir)}`"]

    note = WS / "notes" / "source_specificity_misfire.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS.keys()))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=1626)
    ap.add_argument("--tokens-per-pair", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    records, stats = build_records(tokenizer, args.max_pairs, args.tokens_per_pair, args.max_len)
    plan = {
        "status": "SPECIFICITY_PLAN", "created_utc": now(),
        "arms": args.arms, "checkpoints": args.checkpoints,
        "record_count": len(records), "record_stats": stats,
        "conditions": {"T": "true source in window", "U": "unrelated source in window"},
        "decisive_test": "R−C true_src_content_mass_delta: positive under T but near zero under U → copy mechanism; positive under both → frequency",
    }
    (out_dir / "specificity_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_rows = []

    for arm in args.arms:
        run_dir = ARM_CONFIGS.get(arm)
        if not run_dir or not run_dir.exists():
            print(f"[SKIP] {arm}", flush=True); continue
        arch, role, seed = parse_arm(arm)
        for ck in args.checkpoints:
            mp = run_dir / "hf_model" / ck
            if not mp.exists():
                print(f"[SKIP] {arm} {ck}", flush=True); continue
            print(f"[LOAD] {arm} {ck} {rel(mp)}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32).to(device)
            model.eval()
            rows = score_records(model, records, device, int(pad_id), args.batch_size)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            for r in rows:
                r.update({"arm": arm, "arch": arch, "role": role, "seed": seed, "checkpoint": ck})
            write_csv(out_dir / f"specificity_{arm}_{ck}.csv", rows)
            all_rows.extend(rows)
            print(f"[DONE] {arm} {ck}: rows={len(rows)}", flush=True)

    write_csv(out_dir / "specificity_all.csv", all_rows)
    ck_terms, late_terms, contrasts = summarize(all_rows)
    write_csv(out_dir / "specificity_ck_terms.csv", ck_terms)
    write_csv(out_dir / "specificity_late_terms.csv", late_terms)
    write_csv(out_dir / "specificity_late_contrasts.csv", contrasts)
    note = write_note(out_dir, late_terms, contrasts, plan)

    result = {"status": "SOURCE_SPECIFICITY_DONE", "finished_utc": now(),
              "note": rel(note), "outputs": rel(out_dir), "rows": len(all_rows)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
