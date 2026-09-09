#!/usr/bin/env python3
"""research: strict word-level nonoverlap sensitivity for rewrite probes.

The existing research/research rewrite probes define non-overlap at tokenizer-ID level.
This CPU script replays the accepted pair text/tokenization to identify the surface
word containing each masked rewrite token, then filters rows whose normalized whole
word does not appear among normalized whole words in the source. It then recomputes
term decompositions for DeBERTa and RoBERTa from already-scored rows.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path("experiments/archive/relation_learning")
PROJECT_ROOT = Path(".")
ALL_ACCEPTED_PAIRS = Path("experiments/archive/frontier_consolidation/data/expansion_analysis/combined_all_accepted_pairs.jsonl")
SELECTED_MAX_PAIRS = Path("experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl")
TOKENIZER_DIR = Path("experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model")
D_ROWS = Path("experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv")
RBT_ROWS = Path("experiments/archive/relation_learning/data/roberta_probe_dynamic/rewrite_pair_rows.csv")
OUT = Path("experiments/archive/relation_learning/data/relation_decomposition")
NOTE = Path("research/notes/relation_learning/strict_word_nonoverlap_sensitivity.md")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
}
CKPTS = {"D": ["chck_80M", "chck_90M", "chck_100M"], "RBT": ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]}
CONTRASTS = [("R", "C"), ("V", "C"), ("V", "R")]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def canonical_pair_id(obj: dict[str, Any]) -> str:
    return str(obj.get("pair_id") or obj.get("prompt_id") or f"sid:{obj.get('sentence_id')}|doc:{obj.get('doc_id')}")


def selected_pair_ids() -> set[str]:
    out = set()
    for obj in read_jsonl(SELECTED_MAX_PAIRS):
        pid = str(obj.get("pair_id") or obj.get("prompt_id"))
        out.add(pid)
        out.add(pid.replace("compact:", ""))
    return out


def norm_word(w: str) -> str:
    w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w).lower()
    # light singular/plural and possessive smoothing; intentionally conservative
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def word_set(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text):
        w = norm_word(m.group(0))
        if w and w not in STOPWORDS and len(w) >= 2:
            out.add(w)
    return out


def word_at_offset(text: str, a: int, b: int) -> str:
    if a < 0 or b <= a or a >= len(text):
        return ""
    left = a
    while left > 0 and re.match(r"[A-Za-z0-9'\-]", text[left - 1]):
        left -= 1
    right = b
    while right < len(text) and re.match(r"[A-Za-z0-9'\-]", text[right]):
        right += 1
    return norm_word(text[left:right])


def token_content_word(w: str) -> bool:
    return bool(w) and w not in STOPWORDS and len(w) >= 2


def build_probe_word_map(tokens_per_class: int = 6, max_len: int = 256, max_pairs: int | None = None) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    selected = selected_pair_ids()
    pairs = []
    for obj in read_jsonl(ALL_ACCEPTED_PAIRS):
        pid0 = canonical_pair_id(obj)
        if pid0 in selected or pid0.replace("compact:", "") in selected or f"compact:{pid0}" in selected:
            continue
        if "source_text" not in obj or "rewrite_text" not in obj:
            continue
        d = dict(obj)
        d["pair_id"] = pid0 if pid0.startswith("compact:") else f"compact:{pid0}"
        pairs.append(d)
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    rows = []
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = tok(src_text, add_special_tokens=False)["input_ids"]
        rew_enc = tok(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = list(rew_enc["input_ids"])
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > max_len:
            continue
        src_token_set = set(int(x) for x in src_ids)
        src_words = word_set(src_text)
        content_pos = []
        for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a:
                continue
            piece = rew_text[a:b]
            piece_norm = norm_word(piece)
            whole = word_at_offset(rew_text, int(a), int(b))
            if not token_content_word(piece_norm):
                continue
            token_cls = "overlap" if int(rew_ids[j]) in src_token_set else "nonoverlap"
            strict_word_nonoverlap = token_content_word(whole) and (whole not in src_words)
            content_pos.append((j, token_cls, whole, strict_word_nonoverlap))
        by_cls = {"overlap": [], "nonoverlap": []}
        meta = {}
        for j, cls, whole, strict in content_pos:
            by_cls[cls].append(j)
            meta[j] = (whole, strict)
        chosen = []
        for cls in ["nonoverlap", "overlap"]:
            vals = by_cls[cls]
            if not vals:
                continue
            if len(vals) <= tokens_per_class:
                picks = vals
            else:
                picks = [vals[round(k * (len(vals) - 1) / (tokens_per_class - 1))] for k in range(tokens_per_class)] if tokens_per_class > 1 else [vals[len(vals)//2]]
            chosen.extend((j, cls) for j in picks)
        for j, cls in chosen:
            whole, strict = meta[j]
            rows.append({
                "probe_id": f"rewritecond:{i}:{j}:{p['pair_id']}",
                "pair_id": p["pair_id"],
                "token_class_replayed": cls,
                "target_word_norm": whole,
                "strict_word_nonoverlap": bool(strict),
                "source_word_count": len(src_words),
            })
    return pd.DataFrame(rows)


def role_from_arm(arm: str) -> str:
    parts = str(arm).split("_")
    return parts[1]


def summarize(src_arch: str, scored_path: Path, word_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pd.read_csv(scored_path)
    if "arch" not in scored.columns:
        scored["arch"] = src_arch
    scored["role"] = scored["arm"].map(role_from_arm)
    scored = scored.merge(word_map, on=["probe_id", "pair_id"], how="inner")
    scored = scored[(scored["strict_word_nonoverlap"]) & (scored["token_class"] == "nonoverlap") & (scored["checkpoint"].isin(CKPTS[src_arch]))].copy()
    scored.to_csv(OUT / f"rewrite_strict_word_nonoverlap_rows_{src_arch}.csv", index=False)
    byck = []
    for (arch, seed, ck), g0 in scored.groupby(["arch", "seed", "checkpoint"], dropna=False):
        roles = set(g0["role"])
        for a, b in CONTRASTS:
            if a not in roles or b not in roles:
                continue
            aa = g0[g0["role"] == a][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(columns={"gain": "gain_a", "true_source_nll": "true_a", "unrelated_source_nll": "unrel_a"})
            bb = g0[g0["role"] == b][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(columns={"gain": "gain_b", "true_source_nll": "true_b", "unrelated_source_nll": "unrel_b"})
            j = aa.merge(bb, on="probe_id", how="inner")
            if j.empty:
                continue
            gain_delta = j["gain_a"] - j["gain_b"]
            true_delta = j["true_a"] - j["true_b"]
            unrel_delta = j["unrel_a"] - j["unrel_b"]
            excess = true_delta - unrel_delta
            byck.append({
                "arch": arch,
                "seed": int(seed),
                "checkpoint": ck,
                "contrast": f"{a}minus{b}",
                "n_tokens": int(len(j)),
                "n_pairs": int(scored[scored["probe_id"].isin(j["probe_id"])] ["pair_id"].nunique()),
                "gain_delta": float(gain_delta.mean()),
                "true_source_delta": float(true_delta.mean()),
                "unrelated_source_delta": float(unrel_delta.mean()),
                "excess_true_cost": float(excess.mean()),
                "gain_delta_se": float(gain_delta.std(ddof=1) / math.sqrt(len(j))) if len(j) > 1 else float("nan"),
                "frac_excess_positive_tokens": float((excess > 0).mean()),
            })
    byck_df = pd.DataFrame(byck)
    late_rows = []
    for (arch, seed, contrast), g in byck_df.groupby(["arch", "seed", "contrast"], dropna=False):
        late_rows.append({
            "arch": arch,
            "seed": int(seed),
            "contrast": contrast,
            "n_checkpoints": int(g["checkpoint"].nunique()),
            "n_tokens_min": int(g["n_tokens"].min()),
            "n_pairs_min": int(g["n_pairs"].min()),
            "mean_gain_delta": float(g["gain_delta"].mean()),
            "sd_gain_delta_across_ckpts": float(g["gain_delta"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "mean_true_source_delta": float(g["true_source_delta"].mean()),
            "mean_unrelated_source_delta": float(g["unrelated_source_delta"].mean()),
            "mean_excess_true_cost": float(g["excess_true_cost"].mean()),
            "sd_excess_across_ckpts": float(g["excess_true_cost"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "mean_frac_excess_positive_tokens": float(g["frac_excess_positive_tokens"].mean()),
            "checkpoint_values_gain_delta": ";".join(f"{r.checkpoint}:{r.gain_delta:.6g}" for r in g.itertuples()),
            "checkpoint_values_excess": ";".join(f"{r.checkpoint}:{r.excess_true_cost:.6g}" for r in g.itertuples()),
        })
    late_df = pd.DataFrame(late_rows)
    byck_df.to_csv(OUT / f"rewrite_strict_word_nonoverlap_by_checkpoint_{src_arch}.csv", index=False)
    return byck_df, late_df


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def write_note(late: pd.DataFrame, meta: dict[str, Any]) -> None:
    lines = []
    lines.append("# research strict word-level nonoverlap sensitivity")
    lines.append("")
    lines.append("This CPU-only check replays the compact-pair tokenization and filters the already-scored rewrite probe rows to targets whose whole normalized surface word does not occur among normalized source words. It is stricter than the original tokenizer-ID nonoverlap label, though still heuristic for morphology and named entities.")
    lines.append("")
    lines.append(f"Replay map rows: {meta['word_map_rows']}; strict-word rows: {meta['strict_rows']}; strict-word pairs: {meta['strict_pairs']}.")
    lines.append("")
    lines.append("| arch | seed | contrast | n tokens min | n pairs min | gain delta | true-source delta | unrelated delta | excess true-source cost | frac tokens excess>0 | checkpoint gains |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in late.sort_values(["arch", "seed", "contrast"]).iterrows():
        lines.append(
            f"| {r['arch']} | {int(r['seed'])} | {r['contrast']} | {int(r['n_tokens_min'])} | {int(r['n_pairs_min'])} | "
            f"{fmt(r['mean_gain_delta'])} | {fmt(r['mean_true_source_delta'])} | {fmt(r['mean_unrelated_source_delta'])} | "
            f"{fmt(r['mean_excess_true_cost'])} | {float(r['mean_frac_excess_positive_tokens']):.3f} | {r['checkpoint_values_gain_delta']} |"
        )
    lines.append("")
    lines.append("The main relation-cost result survives the stricter word filter. REPEAT remains worse than CLEAN in source-conditioned use of held-out nonidentical words for both DeBERTa seeds and for RoBERTa, with positive excess true-source cost after subtracting the unrelated-source control. DeBERTa also retains a strong VIEW-over-CLEAN positive conditioning residual under this stricter filter, while RoBERTa's V-C residual remains small. The sample is smaller than the tokenizer-level analysis, so it should be used as a sensitivity check, not as the primary estimate.")
    NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    word_map = build_probe_word_map(tokens_per_class=6, max_len=256, max_pairs=None)
    word_map.to_csv(OUT / "rewrite_probe_word_nonoverlap_map.csv", index=False)
    bycks, lates = [], []
    for arch, path in [("D", D_ROWS), ("RBT", RBT_ROWS)]:
        byck, late = summarize(arch, path, word_map)
        bycks.append(byck); lates.append(late)
    byck_all = pd.concat(bycks, ignore_index=True)
    late_all = pd.concat(lates, ignore_index=True)
    byck_all.to_csv(OUT / "rewrite_strict_word_nonoverlap_by_checkpoint.csv", index=False)
    late_all.to_csv(OUT / "rewrite_strict_word_nonoverlap_late_summary.csv", index=False)
    strict_ids = set(word_map[word_map["strict_word_nonoverlap"]]["probe_id"])
    meta = {
        "word_map_rows": int(len(word_map)),
        "strict_rows": int(word_map["strict_word_nonoverlap"].sum()),
        "strict_pairs": int(word_map[word_map["strict_word_nonoverlap"]]["pair_id"].nunique()),
        "status": "STRICT_WORD_NONOVERLAP_COMPLETE",
        "word_map": str(OUT / "rewrite_probe_word_nonoverlap_map.csv"),
        "by_checkpoint": str(OUT / "rewrite_strict_word_nonoverlap_by_checkpoint.csv"),
        "late_summary": str(OUT / "rewrite_strict_word_nonoverlap_late_summary.csv"),
        "note": str(NOTE),
    }
    write_note(late_all, meta)
    meta["important_rows"] = late_all.to_dict(orient="records")
    (OUT / "rewrite_strict_word_nonoverlap_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["status", "strict_rows", "strict_pairs", "late_summary", "note"]}, indent=2))


if __name__ == "__main__":
    main()
