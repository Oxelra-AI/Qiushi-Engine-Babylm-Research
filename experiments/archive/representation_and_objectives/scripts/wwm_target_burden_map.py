#!/usr/bin/env python3
"""research corrected: WWM target-burden map for legal16k vs legal40k.

CPU-only; does not inspect or poll active training tasks.

Scientific question
-------------------
The legal-40k route keeps fixed whole-word masking (WWM) but changes token inventory and
segmentation. Under WWM the *word/group selection probability* is nominally the same, but
selected words expand to different numbers of predicted subword targets and different rows
cross the seq256 visibility boundary. This script quantifies that representation-interface
change on the exact 10M compact_view_reinvest pool.

Correction relative to the first research run
-------------------------------------------
The initial draft tokenized with truncation and therefore reported zero over_seq256 rows;
that was a bug for raw sequence-capacity interpretation. This corrected script computes:
  1. raw token length / raw WWM group count / raw overflow before truncation;
  2. visible seq256 token length and exact visible WWM group count after the trainer's
     prefix truncation.
Expected masked target tokens under WWM are 0.15 * visible_tokens, while expected masked
word groups are 0.15 * visible_groups.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import statistics
from typing import Any

from transformers import AutoTokenizer

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT_DIR = WS / "data/wwm_target_burden_map"
POOL10 = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
CHANGED_META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl")
TOKENS = {
    "legal16k": WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer",
    "legal40k": WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
}

SOURCE_CLASS_MAP = {
    "cleanqwen_fineweb_compact_view_reinvest": "fineweb_source_qwen_compact_rewrite_pair_row",
    "neutral_cleanqwen_topup_compact_reinvest::open_subtitles": "neutral_topup_from_heldout_official_row",
    "qwen_pair_packed": "inherited_official_source_qwen_paraphrase_pair_row",
    "childes": "official_babylm_source_row",
    "gutenberg": "official_babylm_source_row",
    "open_subtitles": "official_babylm_source_row",
    "simple_wiki": "official_babylm_source_row",
    "bnc_spoken": "official_babylm_source_row",
    "switchboard": "official_babylm_source_row",
}


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class TokSurface:
    def __init__(self, tok):
        self.tok = tok
        self.special_ids = set(tok.all_special_ids)
        self.unk_id = tok.unk_token_id
        self._start_cache: dict[int, bool] = {}

    def word_start_flag(self, tid: int) -> bool:
        if tid not in self._start_cache:
            s = self.tok.convert_ids_to_tokens(int(tid))
            self._start_cache[tid] = bool(s is not None and is_word_start(str(s)))
        return self._start_cache[tid]

    def group_count(self, ids: list[int]) -> int:
        gid = -1
        for i, tid in enumerate(ids):
            if tid in self.special_ids:
                continue
            if gid < 0 or self.word_start_flag(tid) or i == 0:
                gid += 1
        return gid + 1 if gid >= 0 else 0

    def stat_ids(self, ids: list[int]) -> dict[str, int]:
        return {
            "tokens": len([x for x in ids if x not in self.special_ids]),
            "groups": self.group_count(ids),
            "unk": sum(1 for x in ids if self.unk_id is not None and x == self.unk_id),
        }


class Agg:
    def __init__(self):
        self.rows = 0
        self.words = 0
        self.raw_toks = collections.Counter()
        self.vis_toks = collections.Counter()
        self.raw_groups = collections.Counter()
        self.vis_groups = collections.Counter()
        self.over256 = collections.Counter()
        self.unk = collections.Counter()
        self.raw_tpw_vals = collections.defaultdict(list)
        self.vis_tpw_vals = collections.defaultdict(list)
        self.raw_tpg_vals = collections.defaultdict(list)
        self.vis_tpg_vals = collections.defaultdict(list)
        self.row_raw_delta_vals = []
        self.row_visible_delta_vals = []

    def add(self, word_count: int, stats: dict[str, dict[str, int | float]]):
        self.rows += 1
        self.words += word_count
        for lab, st in stats.items():
            raw = int(st["raw_tokens"])
            vis = int(st["visible_tokens"])
            rg = int(st["raw_groups"])
            vg = int(st["visible_groups"])
            self.raw_toks[lab] += raw
            self.vis_toks[lab] += vis
            self.raw_groups[lab] += rg
            self.vis_groups[lab] += vg
            self.over256[lab] += int(raw > 256)
            self.unk[lab] += int(st["unk"])
            self.raw_tpw_vals[lab].append(raw / max(1, word_count))
            self.vis_tpw_vals[lab].append(vis / max(1, word_count))
            self.raw_tpg_vals[lab].append(raw / max(1, rg))
            self.vis_tpg_vals[lab].append(vis / max(1, vg))
        if "legal16k" in stats and "legal40k" in stats:
            self.row_raw_delta_vals.append(1.0 - int(stats["legal40k"]["raw_tokens"]) / max(1, int(stats["legal16k"]["raw_tokens"])))
            self.row_visible_delta_vals.append(1.0 - int(stats["legal40k"]["visible_tokens"]) / max(1, int(stats["legal16k"]["visible_tokens"])))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rows": self.rows, "declared_whitespace_words": self.words}
        for lab in sorted(self.raw_toks):
            out[lab] = {
                "raw_tokens": int(self.raw_toks[lab]),
                "visible_seq256_tokens": int(self.vis_toks[lab]),
                "raw_groups": int(self.raw_groups[lab]),
                "visible_seq256_groups": int(self.vis_groups[lab]),
                "raw_tokens_per_declared_word_weighted": self.raw_toks[lab] / max(1, self.words),
                "visible_tokens_per_declared_word_weighted": self.vis_toks[lab] / max(1, self.words),
                "raw_groups_per_declared_word_weighted": self.raw_groups[lab] / max(1, self.words),
                "visible_groups_per_declared_word_weighted": self.vis_groups[lab] / max(1, self.words),
                "raw_tokens_per_group_weighted": self.raw_toks[lab] / max(1, self.raw_groups[lab]),
                "visible_tokens_per_group_weighted": self.vis_toks[lab] / max(1, self.vis_groups[lab]),
                "expected_visible_masked_target_tokens_per_epoch_p015": 0.15 * self.vis_toks[lab],
                "expected_visible_masked_word_groups_per_epoch_p015": 0.15 * self.vis_groups[lab],
                "raw_over_seq256_rows": int(self.over256[lab]),
                "unk_tokens_raw": int(self.unk[lab]),
                "row_raw_tpw_mean": statistics.mean(self.raw_tpw_vals[lab]) if self.raw_tpw_vals[lab] else None,
                "row_raw_tpw_median": statistics.median(self.raw_tpw_vals[lab]) if self.raw_tpw_vals[lab] else None,
                "row_raw_tpw_p90": q(self.raw_tpw_vals[lab], 0.9),
                "row_visible_tpw_mean": statistics.mean(self.vis_tpw_vals[lab]) if self.vis_tpw_vals[lab] else None,
                "row_visible_tpw_median": statistics.median(self.vis_tpw_vals[lab]) if self.vis_tpw_vals[lab] else None,
                "row_visible_tpw_p90": q(self.vis_tpw_vals[lab], 0.9),
                "row_raw_tpg_mean": statistics.mean(self.raw_tpg_vals[lab]) if self.raw_tpg_vals[lab] else None,
                "row_visible_tpg_mean": statistics.mean(self.vis_tpg_vals[lab]) if self.vis_tpg_vals[lab] else None,
            }
        if "legal16k" in self.raw_toks and "legal40k" in self.raw_toks:
            out["legal40k_vs_legal16k"] = {
                "raw_token_ratio_weighted": self.raw_toks["legal40k"] / max(1, self.raw_toks["legal16k"]),
                "raw_target_token_reduction_fraction_weighted": 1.0 - self.raw_toks["legal40k"] / max(1, self.raw_toks["legal16k"]),
                "visible_token_ratio_weighted": self.vis_toks["legal40k"] / max(1, self.vis_toks["legal16k"]),
                "visible_target_token_reduction_fraction_weighted": 1.0 - self.vis_toks["legal40k"] / max(1, self.vis_toks["legal16k"]),
                "visible_target_token_reduction_count_per_epoch_p015": 0.15 * (self.vis_toks["legal16k"] - self.vis_toks["legal40k"]),
                "visible_target_token_reduction_count_over_10_epochs_p015": 10 * 0.15 * (self.vis_toks["legal16k"] - self.vis_toks["legal40k"]),
                "visible_group_ratio_weighted": self.vis_groups["legal40k"] / max(1, self.vis_groups["legal16k"]),
                "visible_masked_word_group_reduction_fraction_weighted": 1.0 - self.vis_groups["legal40k"] / max(1, self.vis_groups["legal16k"]),
                "raw_over256_row_reduction": int(self.over256["legal16k"] - self.over256["legal40k"]),
                "row_raw_reduction_mean": statistics.mean(self.row_raw_delta_vals) if self.row_raw_delta_vals else None,
                "row_raw_reduction_median": statistics.median(self.row_raw_delta_vals) if self.row_raw_delta_vals else None,
                "row_visible_reduction_mean": statistics.mean(self.row_visible_delta_vals) if self.row_visible_delta_vals else None,
                "row_visible_reduction_median": statistics.median(self.row_visible_delta_vals) if self.row_visible_delta_vals else None,
                "row_visible_reduction_p10": q(self.row_visible_delta_vals, 0.1),
                "row_visible_reduction_p90": q(self.row_visible_delta_vals, 0.9),
            }
        return out


def load_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = []
    with POOL10.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            obj["row_index"] = idx
            obj["words"] = int(obj.get("words", len(str(obj.get("text", "")).split())))
            obj["source_class"] = SOURCE_CLASS_MAP.get(str(obj.get("source", "")), "unknown")
            rows.append(obj)
    changed = {}
    with CHANGED_META.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            changed[int(obj["row_index"])] = obj
    return rows, changed


def token_stats(tok, texts: list[str], batch_size: int = 512) -> list[dict[str, int]]:
    # Avoid model_max_length warnings when tokenizing long raw rows for analysis only.
    tok.model_max_length = 10**9
    surface = TokSurface(tok)
    out: list[dict[str, int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encs = tok(batch, add_special_tokens=False, truncation=False, padding=False)
        for ids in encs["input_ids"]:
            raw = surface.stat_ids(ids)
            visible = surface.stat_ids(ids[:256])
            out.append({
                "raw_tokens": raw["tokens"],
                "raw_groups": raw["groups"],
                "visible_tokens": visible["tokens"],
                "visible_groups": visible["groups"],
                "unk": raw["unk"],
            })
    return out


def allocated_stats(stats: dict[str, dict[str, int | float]], frac: float) -> dict[str, dict[str, int]]:
    keys = ["raw_tokens", "raw_groups", "visible_tokens", "visible_groups", "unk"]
    return {lab: {k: int(round(float(st[k]) * frac)) for k in keys} for lab, st in stats.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, changed = load_rows()
    texts = [str(r.get("text", "")) for r in rows]
    tokenizer_stats: dict[str, list[dict[str, int]]] = {}
    tok_meta = {}
    for lab, path in TOKENS.items():
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        tok_meta[lab] = {"path": str(path), "len": len(tok), "vocab_size": tok.vocab_size, "special_ids": tok.all_special_ids}
        tokenizer_stats[lab] = token_stats(tok, texts)

    by_source = collections.defaultdict(Agg)
    by_source_class = collections.defaultdict(Agg)
    changed_all = Agg()
    changed_by_domain = collections.defaultdict(Agg)
    global_agg = Agg()
    top_visible_reduction_rows = []
    rows_16_over_40_not = []

    for i, r in enumerate(rows):
        stats = {lab: tokenizer_stats[lab][i] for lab in TOKENS}
        global_agg.add(r["words"], stats)
        by_source[str(r.get("source", ""))].add(r["words"], stats)
        by_source_class[str(r.get("source_class", ""))].add(r["words"], stats)
        if i in changed:
            changed_all.add(r["words"], stats)
            cm = changed[i]
            for dom, dom_words in dict(cm.get("component_sources", {})).items():
                frac = float(dom_words) / max(1, int(cm.get("words", r["words"])))
                changed_by_domain[str(dom)].add(int(dom_words), allocated_stats(stats, frac))
        l16_raw = int(stats["legal16k"]["raw_tokens"])
        l40_raw = int(stats["legal40k"]["raw_tokens"])
        l16_vis = int(stats["legal16k"]["visible_tokens"])
        l40_vis = int(stats["legal40k"]["visible_tokens"])
        red = 1 - (l40_vis / max(1, l16_vis))
        row_rec = {
            "row_index": i,
            "example_id": r.get("example_id"),
            "source": r.get("source"),
            "source_class": r.get("source_class"),
            "words": r["words"],
            "legal16_raw_tokens": l16_raw,
            "legal40_raw_tokens": l40_raw,
            "legal16_visible_tokens": l16_vis,
            "legal40_visible_tokens": l40_vis,
            "legal16_visible_groups": int(stats["legal16k"]["visible_groups"]),
            "legal40_visible_groups": int(stats["legal40k"]["visible_groups"]),
            "visible_target_reduction_fraction": red,
            "raw_over256_16k": l16_raw > 256,
            "raw_over256_40k": l40_raw > 256,
            "text_head": str(r.get("text", ""))[:240].replace("\n", " "),
        }
        if len(top_visible_reduction_rows) < 200 or red > min(x["visible_target_reduction_fraction"] for x in top_visible_reduction_rows):
            top_visible_reduction_rows.append(row_rec)
            top_visible_reduction_rows = sorted(top_visible_reduction_rows, key=lambda x: x["visible_target_reduction_fraction"], reverse=True)[:200]
        if l16_raw > 256 and l40_raw <= 256 and len(rows_16_over_40_not) < 100:
            rows_16_over_40_not.append(row_rec)

    global_summary = global_agg.summary()
    changed_summary = changed_all.summary()
    payload = {
        "status": "WWM_TARGET_BURDEN_MAP_CORRECTED_V2",
        "supersedes": "Earlier research target-burden run that mistakenly computed over_seq256 after truncation.",
        "purpose": "Quantify raw seq256 visibility and visible WWM target-token burden change from legal16k to legal40k on the exact compact_view_reinvest 10M pool.",
        "pool10": str(POOL10),
        "changed_meta": str(CHANGED_META),
        "tokenizers": tok_meta,
        "global": global_summary,
        "by_source_class": {k: v.summary() for k, v in sorted(by_source_class.items())},
        "by_source": {k: v.summary() for k, v in sorted(by_source.items())},
        "changed_block_all": changed_summary,
        "changed_block_by_domain_allocated": {k: v.summary() for k, v in sorted(changed_by_domain.items())},
        "top_visible_target_reduction_rows": top_visible_reduction_rows[:50],
        "examples_16k_over256_but_40k_not": rows_16_over_40_not[:30],
        "interpretation": [
            "Under fixed WWM, expected visible target-token count is p times visible non-special tokens, whereas expected selected word groups are p times visible WWM groups.",
            "Legal40k strongly lowers subword targets per selected word and rescues some rows from seq256 overflow; this is the intended representation-interface hypothesis, not a score by itself.",
            "If official legal40k recovers Supplement/EWoK, the strongest mechanism is likely reduced fragmentation plus better visible context for relation/diagnostic strings; if it loses GlobalPIQA/Entity/COMPS, the larger vocabulary may have weakened useful subword regularization or rare-token learning under 10M words.",
        ],
    }
    out_json = OUT_DIR / "wwm_target_burden_map.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (OUT_DIR / "source_class_target_burden.csv").open("w", encoding="utf-8") as f:
        f.write("source_class,words,raw_tok16,raw_tok40,raw_reduction,vis_tok16,vis_tok40,visible_reduction,vis_groups16,vis_groups40,over256_16,over256_40\n")
        for k, agg in sorted(by_source_class.items()):
            s = agg.summary(); cmp = s["legal40k_vs_legal16k"]
            f.write(f"{k},{s['declared_whitespace_words']},{s['legal16k']['raw_tokens']},{s['legal40k']['raw_tokens']},{cmp['raw_target_token_reduction_fraction_weighted']},{s['legal16k']['visible_seq256_tokens']},{s['legal40k']['visible_seq256_tokens']},{cmp['visible_target_token_reduction_fraction_weighted']},{s['legal16k']['visible_seq256_groups']},{s['legal40k']['visible_seq256_groups']},{s['legal16k']['raw_over_seq256_rows']},{s['legal40k']['raw_over_seq256_rows']}\n")
    with (OUT_DIR / "changed_domain_target_burden_allocated.csv").open("w", encoding="utf-8") as f:
        f.write("domain,words,raw_tok16,raw_tok40,raw_reduction,vis_tok16,vis_tok40,visible_reduction,vis_groups16,vis_groups40,over256_16,over256_40\n")
        for k, agg in sorted(changed_by_domain.items()):
            s = agg.summary(); cmp = s["legal40k_vs_legal16k"]
            f.write(f"{k},{s['declared_whitespace_words']},{s['legal16k']['raw_tokens']},{s['legal40k']['raw_tokens']},{cmp['raw_target_token_reduction_fraction_weighted']},{s['legal16k']['visible_seq256_tokens']},{s['legal40k']['visible_seq256_tokens']},{cmp['visible_target_token_reduction_fraction_weighted']},{s['legal16k']['visible_seq256_groups']},{s['legal40k']['visible_seq256_groups']},{s['legal16k']['raw_over_seq256_rows']},{s['legal40k']['raw_over_seq256_rows']}\n")
    note = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/wwm_target_burden_map/wwm_target_burden_map_summary.md')
    g = global_summary; c = changed_summary
    note.write_text(
        "# research corrected WWM target-burden map\n\n"
        "This corrected map separates raw row length/overflow from the actual visible seq256 MLM target surface. "
        "It supersedes the first research target-burden run, which computed overflow after truncation.\n\n"
        f"## Global 10M pool\n"
        f"- Raw tokens/word: legal16k {g['legal16k']['raw_tokens_per_declared_word_weighted']:.4f}; legal40k {g['legal40k']['raw_tokens_per_declared_word_weighted']:.4f}; raw reduction {g['legal40k_vs_legal16k']['raw_target_token_reduction_fraction_weighted']:.2%}.\n"
        f"- Visible seq256 tokens/word: legal16k {g['legal16k']['visible_tokens_per_declared_word_weighted']:.4f}; legal40k {g['legal40k']['visible_tokens_per_declared_word_weighted']:.4f}; visible target-token reduction {g['legal40k_vs_legal16k']['visible_target_token_reduction_fraction_weighted']:.2%}.\n"
        f"- Raw rows above 256 tokens: legal16k {g['legal16k']['raw_over_seq256_rows']}; legal40k {g['legal40k']['raw_over_seq256_rows']}; rescued rows {g['legal40k_vs_legal16k']['raw_over256_row_reduction']}.\n"
        f"- Visible WWM groups/word: legal16k {g['legal16k']['visible_groups_per_declared_word_weighted']:.4f}; legal40k {g['legal40k']['visible_groups_per_declared_word_weighted']:.4f}.\n\n"
        f"## Changed compact-view block\n"
        f"- Raw tokens/word: legal16k {c['legal16k']['raw_tokens_per_declared_word_weighted']:.4f}; legal40k {c['legal40k']['raw_tokens_per_declared_word_weighted']:.4f}; raw reduction {c['legal40k_vs_legal16k']['raw_target_token_reduction_fraction_weighted']:.2%}.\n"
        f"- Visible seq256 tokens/word: legal16k {c['legal16k']['visible_tokens_per_declared_word_weighted']:.4f}; legal40k {c['legal40k']['visible_tokens_per_declared_word_weighted']:.4f}; visible target-token reduction {c['legal40k_vs_legal16k']['visible_target_token_reduction_fraction_weighted']:.2%}.\n"
        f"- Raw rows above 256 tokens: legal16k {c['legal16k']['raw_over_seq256_rows']}; legal40k {c['legal40k']['raw_over_seq256_rows']}; rescued rows {c['legal40k_vs_legal16k']['raw_over256_row_reduction']}.\n"
        f"- Visible WWM groups/word: legal16k {c['legal16k']['visible_groups_per_declared_word_weighted']:.4f}; legal40k {c['legal40k']['visible_groups_per_declared_word_weighted']:.4f}.\n\n"
        "Detailed files: `wwm_target_burden_map.json`, `source_class_target_burden.csv`, `changed_domain_target_burden_allocated.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "global_raw_reduction": g["legal40k_vs_legal16k"]["raw_target_token_reduction_fraction_weighted"],
        "global_visible_reduction": g["legal40k_vs_legal16k"]["visible_target_token_reduction_fraction_weighted"],
        "changed_raw_reduction": c["legal40k_vs_legal16k"]["raw_target_token_reduction_fraction_weighted"],
        "changed_visible_reduction": c["legal40k_vs_legal16k"]["visible_target_token_reduction_fraction_weighted"],
        "global_over256_16k": g["legal16k"]["raw_over_seq256_rows"],
        "global_over256_40k": g["legal40k"]["raw_over_seq256_rows"],
        "changed_over256_16k": c["legal16k"]["raw_over_seq256_rows"],
        "changed_over256_40k": c["legal40k"]["raw_over_seq256_rows"],
        "note": str(note),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
