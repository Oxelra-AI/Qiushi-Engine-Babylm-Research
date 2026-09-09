#!/usr/bin/env python3
"""research: deterministic expected-budget analysis for static-prior WWM.

The research verifier pointed out that the original static-prior patch preserved
WWM group rate but not necessarily MLM token supervision.  This analyzer imports
the actual generated trainer, uses its real tokenizer dataset/WWM grouping, and
computes the exact expected group and token budgets under fixed WWM and static
prior schemes on the training-visible seq256 stream.  It is CPU-only and reads no
evaluation examples.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER_PATH = WORKSPACE / "scripts/masking_curriculum_trainer_static_prior.py"
POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
PRIOR = WORKSPACE / "data/static_token_mask_prior/static_token_mask_prior.json"
OUT_DIR = WORKSPACE / "data/static_prior_expected_budget"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
RELATION_WORDS = {
    "above", "across", "against", "along", "among", "around", "at", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "down", "from", "in", "inside",
    "into", "near", "off", "on", "onto", "opposite", "outside", "over", "through",
    "throughout", "to", "toward", "towards", "under", "underneath", "up", "within",
    "because", "cause", "caused", "causes", "causing", "due", "effect", "effects",
    "impact", "impacts", "therefore", "thus", "hence", "result", "resulted", "results",
    "resulting", "lead", "leads", "led", "allow", "allows", "allowed", "prevent",
    "prevents", "prevented", "reduce", "reduces", "reduced", "increase", "increases",
    "increased", "risk", "risks", "so", "since", "thereby",
    "act", "acts", "acted", "acting", "arrive", "arrived", "begin", "began", "become",
    "became", "build", "built", "carry", "carried", "change", "changed", "changes",
    "collide", "collided", "come", "came", "create", "created", "develop", "developed",
    "drive", "drove", "enter", "entered", "fall", "fell", "falling", "flow", "flowed",
    "flows", "form", "formed", "forms", "grow", "grew", "hit", "hits", "leave", "left",
    "make", "made", "move", "moved", "moves", "moving", "produce", "produced",
    "reach", "reached", "rise", "rose", "run", "ran", "running", "send", "sent",
    "start", "started", "stop", "stopped", "turn", "turned", "use", "used", "work", "worked",
    "after", "before", "during", "while", "when", "whenever", "until", "then",
    "later", "earlier", "first", "last", "next", "year", "years", "month", "months",
    "day", "days", "century", "centuries", "annual", "currently", "eventually",
}


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def tok_core(token: str) -> str:
    s = str(token).replace("Ġ", "").replace("▁", "")
    s = s.strip(" \t\r\n.,;:!?()[]{}<>\"'`“”‘’-/\\")
    return s.lower()


class SplitStats:
    def __init__(self) -> None:
        self.rows = 0
        self.words = 0
        self.candidate_tokens = 0.0
        self.candidate_groups = 0.0
        self.fixed_exp_groups = 0.0
        self.fixed_exp_tokens = 0.0
        self.static_exp_groups = 0.0
        self.static_exp_tokens = 0.0
        self.candidate_prior_sum = 0.0
        self.static_exp_prior_sum = 0.0
        self.fixed_exp_prior_sum = 0.0
        self.candidate_high125 = 0.0
        self.static_exp_high125 = 0.0
        self.fixed_exp_high125 = 0.0
        self.candidate_high150 = 0.0
        self.static_exp_high150 = 0.0
        self.fixed_exp_high150 = 0.0
        self.candidate_relation = 0.0
        self.static_exp_relation = 0.0
        self.fixed_exp_relation = 0.0
        self.clip_hits = 0
        self.clip_mass_loss = 0.0
        self.max_group_prob = 0.0
        self.min_group_prob = 1.0
        self.sum_len = 0.0
        self.sum_weight = 0.0
        self.sum_len2 = 0.0
        self.sum_weight2 = 0.0
        self.sum_len_weight = 0.0
        self.group_obs = 0

    def add_row(self, words: int, fixed_prob: float, group_lengths: torch.Tensor, group_weights: torch.Tensor, group_relation_counts: torch.Tensor, group_high125_counts: torch.Tensor, group_high150_counts: torch.Tensor, group_prior_sums: torch.Tensor, group_probs: torch.Tensor, raw_group_probs: torch.Tensor) -> None:
        n_groups = int(group_lengths.numel())
        if n_groups == 0:
            self.rows += 1
            self.words += words
            return
        self.rows += 1
        self.words += words
        lengths = group_lengths.float()
        probs = group_probs.float()
        raw_probs = raw_group_probs.float()
        self.candidate_groups += float(n_groups)
        self.candidate_tokens += float(lengths.sum().item())
        self.fixed_exp_groups += float(fixed_prob * n_groups)
        self.fixed_exp_tokens += float(fixed_prob * lengths.sum().item())
        self.static_exp_groups += float(probs.sum().item())
        self.static_exp_tokens += float((probs * lengths).sum().item())
        self.candidate_prior_sum += float(group_prior_sums.sum().item())
        self.fixed_exp_prior_sum += float(fixed_prob * group_prior_sums.sum().item())
        self.static_exp_prior_sum += float((probs * group_prior_sums).sum().item())
        self.candidate_high125 += float(group_high125_counts.sum().item())
        self.fixed_exp_high125 += float(fixed_prob * group_high125_counts.sum().item())
        self.static_exp_high125 += float((probs * group_high125_counts).sum().item())
        self.candidate_high150 += float(group_high150_counts.sum().item())
        self.fixed_exp_high150 += float(fixed_prob * group_high150_counts.sum().item())
        self.static_exp_high150 += float((probs * group_high150_counts).sum().item())
        self.candidate_relation += float(group_relation_counts.sum().item())
        self.fixed_exp_relation += float(fixed_prob * group_relation_counts.sum().item())
        self.static_exp_relation += float((probs * group_relation_counts).sum().item())
        clipped = raw_probs > 0.95
        self.clip_hits += int(clipped.sum().item())
        if clipped.any():
            self.clip_mass_loss += float(((raw_probs - 0.95).clamp_min(0) * lengths).sum().item())
        self.max_group_prob = max(self.max_group_prob, float(probs.max().item()))
        self.min_group_prob = min(self.min_group_prob, float(probs.min().item()))
        self.sum_len += float(lengths.sum().item())
        self.sum_weight += float(group_weights.sum().item())
        self.sum_len2 += float((lengths * lengths).sum().item())
        self.sum_weight2 += float((group_weights * group_weights).sum().item())
        self.sum_len_weight += float((lengths * group_weights).sum().item())
        self.group_obs += n_groups

    def corr_len_weight(self) -> float | None:
        n = self.group_obs
        if n <= 1:
            return None
        mx = self.sum_len / n
        my = self.sum_weight / n
        cov = self.sum_len_weight / n - mx * my
        vx = self.sum_len2 / n - mx * mx
        vy = self.sum_weight2 / n - my * my
        if vx <= 0 or vy <= 0:
            return None
        return cov / math.sqrt(vx * vy)

    def as_dict(self) -> dict[str, Any]:
        cand_tok = max(self.candidate_tokens, 1.0)
        fixed_tok = max(self.fixed_exp_tokens, 1.0)
        static_tok = max(self.static_exp_tokens, 1.0)
        cand_grp = max(self.candidate_groups, 1.0)
        fixed_grp = max(self.fixed_exp_groups, 1.0)
        static_grp = max(self.static_exp_groups, 1.0)
        return {
            "rows": self.rows,
            "words": self.words,
            "candidate_tokens": self.candidate_tokens,
            "candidate_groups": self.candidate_groups,
            "fixed_expected_group_rate": self.fixed_exp_groups / cand_grp,
            "static_expected_group_rate": self.static_exp_groups / cand_grp,
            "group_rate_delta_static_minus_fixed": self.static_exp_groups / cand_grp - self.fixed_exp_groups / cand_grp,
            "fixed_expected_token_rate": self.fixed_exp_tokens / cand_tok,
            "static_expected_token_rate": self.static_exp_tokens / cand_tok,
            "token_rate_delta_static_minus_fixed": self.static_exp_tokens / cand_tok - self.fixed_exp_tokens / cand_tok,
            "candidate_mean_prior": self.candidate_prior_sum / cand_tok,
            "fixed_expected_selected_mean_prior": self.fixed_exp_prior_sum / fixed_tok,
            "static_expected_selected_mean_prior": self.static_exp_prior_sum / static_tok,
            "static_prior_lift_vs_candidate": (self.static_exp_prior_sum / static_tok) / max(self.candidate_prior_sum / cand_tok, 1e-12),
            "fixed_prior_lift_vs_candidate": (self.fixed_exp_prior_sum / fixed_tok) / max(self.candidate_prior_sum / cand_tok, 1e-12),
            "candidate_high125_rate": self.candidate_high125 / cand_tok,
            "static_high125_rate": self.static_exp_high125 / static_tok,
            "static_high125_lift": (self.static_exp_high125 / static_tok) / max(self.candidate_high125 / cand_tok, 1e-12),
            "candidate_high150_rate": self.candidate_high150 / cand_tok,
            "static_high150_rate": self.static_exp_high150 / static_tok,
            "static_high150_lift": (self.static_exp_high150 / static_tok) / max(self.candidate_high150 / cand_tok, 1e-12),
            "candidate_relation_rate": self.candidate_relation / cand_tok,
            "static_relation_rate": self.static_exp_relation / static_tok,
            "static_relation_lift": (self.static_exp_relation / static_tok) / max(self.candidate_relation / cand_tok, 1e-12),
            "clip_hits": self.clip_hits,
            "clip_mass_loss_token_expectation": self.clip_mass_loss,
            "max_group_prob": self.max_group_prob,
            "min_group_prob": self.min_group_prob if self.group_obs else None,
            "group_length_weight_corr": self.corr_len_weight(),
        }


def read_pool(limit_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit_rows and len(rows) >= limit_rows:
                break
    return rows


def make_examples(trainer, rows: list[dict[str, Any]]) -> list[Any]:
    examples = []
    for i, obj in enumerate(rows):
        text = str(obj["text"])
        words = int(obj.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"word mismatch in row {i}")
        examples.append(trainer.Example(text=text, words=words, example_id=int(obj.get("example_id", i)), source=str(obj.get("source", ""))))
    return examples


def analyze_scheme(trainer, tokenizer, rows: list[dict[str, Any]], weights: torch.Tensor, relation_ids: torch.Tensor, scheme: str, seq_length: int, mask_prob: float) -> dict[str, Any]:
    ds = trainer.MaskedChunkDataset(make_examples(trainer, rows), tokenizer, seq_length=seq_length)
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), dtype=torch.long)
    splits = {"all": SplitStats(), "changed": SplitStats(), "other": SplitStats()}
    source_counts: dict[str, int] = {}
    n_rows_at_max = 0
    for i, row in enumerate(rows):
        item = ds[i]
        input_ids = item["input_ids"]
        attention = item["attention_mask"]
        groups = item["word_group"]
        candidate = attention.bool() & ~torch.isin(input_ids, special_ids)
        if int(candidate.sum().item()) >= seq_length:
            n_rows_at_max += 1
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        group_lengths = []
        group_weights = []
        group_prior_sums = []
        group_relation_counts = []
        group_high125_counts = []
        group_high150_counts = []
        for gid in valid_groups:
            g_mask = (groups == gid) & candidate
            ids = input_ids[g_mask]
            if ids.numel() == 0:
                continue
            w = weights[ids]
            group_lengths.append(float(ids.numel()))
            group_weights.append(float(w.mean().item()))
            group_prior_sums.append(float(w.sum().item()))
            group_relation_counts.append(float(relation_ids[ids].sum().item()))
            group_high125_counts.append(float((w >= 1.25).sum().item()))
            group_high150_counts.append(float((w >= 1.50).sum().item()))
        if not group_lengths:
            continue
        gl = torch.tensor(group_lengths, dtype=torch.float32)
        gw = torch.tensor(group_weights, dtype=torch.float32)
        gps = torch.tensor(group_prior_sums, dtype=torch.float32)
        grel = torch.tensor(group_relation_counts, dtype=torch.float32)
        gh125 = torch.tensor(group_high125_counts, dtype=torch.float32)
        gh150 = torch.tensor(group_high150_counts, dtype=torch.float32)
        token_weighted_mean = ((gw * gl).sum() / gl.sum().clamp_min(1.0)).clamp_min(1e-8)
        raw_probs = gw / token_weighted_mean * mask_prob
        probs = raw_probs.clamp(0.0, 0.95)
        src = str(row.get("source", ""))
        source_counts[src] = source_counts.get(src, 0) + 1
        changed = src == CHANGED_SOURCE
        words = int(row.get("words", 0))
        splits["all"].add_row(words, mask_prob, gl, gw, grel, gh125, gh150, gps, probs, raw_probs)
        splits["changed" if changed else "other"].add_row(words, mask_prob, gl, gw, grel, gh125, gh150, gps, probs, raw_probs)
    return {
        "scheme": scheme,
        "rows_analyzed": len(rows),
        "rows_at_candidate_max_length": n_rows_at_max,
        "source_counts": source_counts,
        "splits": {k: v.as_dict() for k, v in splits.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--schemes", nargs="*", default=[])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()
    trainer = load_module(TRAINER_PATH, "expected_budget_trainer")
    tokenizer = trainer.make_portable_tokenizer(str(TOKENIZER))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    schemes = args.schemes or sorted(prior["schemes"].keys())
    # report narrow first when present
    schemes = sorted(schemes, key=lambda x: (x != "relation_only_v1", x != "relation_info_v1", x))
    token_cores = [tok_core(tokenizer.convert_ids_to_tokens(i)) for i in range(len(tokenizer))]
    relation_ids = torch.tensor([core in RELATION_WORDS for core in token_cores], dtype=torch.bool)
    rows = read_pool(args.limit_rows)
    results = []
    for scheme in schemes:
        weights = torch.tensor(prior["schemes"][scheme]["weights"], dtype=torch.float32)
        results.append(analyze_scheme(trainer, tokenizer, rows, weights, relation_ids, scheme, args.seq_length, args.mask_prob))
    summary = {
        "status": "STATIC_PRIOR_EXPECTED_BUDGET",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Exact expected-budget analysis for static-prior WWM on the trainer-visible legal-pool seq256 stream; verifies whether token-level MLM supervision is preserved after token-length normalization.",
        "inputs": {
            "trainer": str(TRAINER_PATH),
            "pool_10m": str(POOL),
            "tokenizer": str(TOKENIZER),
            "prior_json": str(PRIOR),
            "limit_rows": args.limit_rows,
            "seq_length": args.seq_length,
            "mask_prob": args.mask_prob,
            "schemes": schemes,
        },
        "results": results,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "static_prior_expected_budget.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research static-prior expected-budget analysis",
        "",
        summary["purpose"],
        "",
        "This deterministic CPU analysis uses the actual generated trainer dataset and WWM grouping. It is not a model-training result.",
        "",
    ]
    for res in results:
        lines.append(f"## Scheme `{res['scheme']}`")
        lines.append(f"- rows analyzed: {res['rows_analyzed']}; rows at candidate max length: {res['rows_at_candidate_max_length']}")
        lines.append(f"- source_counts: {res['source_counts']}")
        lines.append("")
        lines.append("| split | fixed token rate | static token rate | Δ token | fixed group rate | static group rate | Δ group | selected prior lift | relation lift | high1.50 lift | clip hits | max p | len-weight corr |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split in ["all", "changed", "other"]:
            s = res["splits"][split]
            def fmt(x: Any) -> str:
                if x is None:
                    return "NA"
                return f"{float(x):.6f}"
            lines.append("| {split} | {ftr} | {str_} | {dtr} | {fgr} | {sgr} | {dgr} | {pl} | {rl} | {hl} | {ch} | {mp} | {corr} |".format(
                split=split,
                ftr=fmt(s["fixed_expected_token_rate"]),
                str_=fmt(s["static_expected_token_rate"]),
                dtr=fmt(s["token_rate_delta_static_minus_fixed"]),
                fgr=fmt(s["fixed_expected_group_rate"]),
                sgr=fmt(s["static_expected_group_rate"]),
                dgr=fmt(s["group_rate_delta_static_minus_fixed"]),
                pl=fmt(s["static_prior_lift_vs_candidate"]),
                rl=fmt(s["static_relation_lift"]),
                hl=fmt(s["static_high150_lift"]),
                ch=s["clip_hits"],
                mp=fmt(s["max_group_prob"]),
                corr=fmt(s["group_length_weight_corr"]),
            ))
        lines.append("")
    lines += ["## Input files", ""]
    for k, v in summary["inputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "static_prior_expected_budget.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "schemes": schemes,
        "rows": len(rows),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
