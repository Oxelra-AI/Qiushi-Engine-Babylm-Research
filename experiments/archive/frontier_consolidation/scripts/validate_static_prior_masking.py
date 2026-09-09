#!/usr/bin/env python3
"""research: validate the static-prior masking trainer on real legal-pool batches.

This is CPU-only.  It imports the generated research trainer and uses its real
MaskedChunkDataset, collate, MaskingCurriculumState, and apply_masking_curriculum
functions to compare standard fixed WWM with the two corpus-derived static-prior
schemes.  The goal is not to prove score improvement, but to establish whether
the patch preserves the effective WWM mask budget while actually reallocating
prediction pressure toward the intended relation/information-bearing tokens.
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
OUT_DIR = WORKSPACE / "data/static_prior_masking_validation"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"

# Same lexical classes used to build the research static prior.  These are used
# only for interpretable validation of selected-token enrichment, not to rebuild
# the prior.
SPATIAL = {
    "above", "across", "against", "along", "among", "around", "at", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "down", "from", "in", "inside",
    "into", "near", "off", "on", "onto", "opposite", "outside", "over", "through",
    "throughout", "to", "toward", "towards", "under", "underneath", "up", "within",
}
CAUSAL = {
    "because", "cause", "caused", "causes", "causing", "due", "effect", "effects",
    "impact", "impacts", "therefore", "thus", "hence", "result", "resulted", "results",
    "resulting", "lead", "leads", "led", "allow", "allows", "allowed", "prevent",
    "prevents", "prevented", "reduce", "reduces", "reduced", "increase", "increases",
    "increased", "risk", "risks", "so", "since", "thereby",
}
DYNAMIC = {
    "act", "acts", "acted", "acting", "arrive", "arrived", "begin", "began", "become",
    "became", "build", "built", "carry", "carried", "change", "changed", "changes",
    "collide", "collided", "come", "came", "create", "created", "develop", "developed",
    "drive", "drove", "enter", "entered", "fall", "fell", "falling", "flow", "flowed",
    "flows", "form", "formed", "forms", "grow", "grew", "hit", "hits", "leave", "left",
    "make", "made", "move", "moved", "moves", "moving", "produce", "produced",
    "reach", "reached", "rise", "rose", "run", "ran", "running", "send", "sent",
    "start", "started", "stop", "stopped", "turn", "turned", "use", "used", "work", "worked",
}
TEMPORAL = {
    "after", "before", "during", "while", "when", "whenever", "until", "since", "then",
    "later", "earlier", "first", "last", "next", "year", "years", "month", "months",
    "day", "days", "century", "centuries", "annual", "currently", "eventually",
}
RELATION_LEX = SPATIAL | CAUSAL | DYNAMIC | TEMPORAL


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick_samples(rows: list[dict[str, Any]], rows_front: int, rows_stride: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    out["front_changed_plus_early_filler"] = rows[: min(rows_front, len(rows))]
    if rows_stride > 0:
        if rows_stride >= len(rows):
            out["stride_full_pool"] = list(rows)
        else:
            # Deterministic near-uniform sample over the complete legal 10M pool.
            idxs = sorted({round(i * (len(rows) - 1) / max(1, rows_stride - 1)) for i in range(rows_stride)})
            out["stride_full_pool"] = [rows[int(i)] for i in idxs]
    return out


def make_examples(trainer, rows: list[dict[str, Any]]) -> list[Any]:
    examples = []
    for i, obj in enumerate(rows):
        text = str(obj["text"])
        words = int(obj.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"word mismatch in validation row {i}: field={words}")
        examples.append(trainer.Example(text=text, words=words, example_id=int(obj.get("example_id", i)), source=str(obj.get("source", ""))))
    return examples


def tok_core(token: str) -> str:
    # Byte-level BPE word-start marker and SentencePiece marker handling; keep
    # alphabetic cores so relation prepositions/verbs can be read.
    s = str(token).replace("Ġ", "").replace("▁", "")
    s = s.strip(" \t\r\n.,;:!?()[]{}<>\"'`“”‘’-/\\")
    return s.lower()


def zeros() -> dict[str, float]:
    return {
        "candidate_tokens": 0.0,
        "selected_tokens": 0.0,
        "candidate_groups": 0.0,
        "selected_groups": 0.0,
        "candidate_prior_sum": 0.0,
        "selected_prior_sum": 0.0,
        "candidate_high125": 0.0,
        "selected_high125": 0.0,
        "candidate_high150": 0.0,
        "selected_high150": 0.0,
        "candidate_relation_lex": 0.0,
        "selected_relation_lex": 0.0,
        "rows": 0.0,
    }


def add_counts(acc: dict[str, float], *, input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor, selected: torch.Tensor, prior_weights: torch.Tensor, relation_ids: torch.Tensor, special_ids: torch.Tensor) -> None:
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    acc["candidate_tokens"] += float(candidate.sum().item())
    acc["selected_tokens"] += float((selected & candidate).sum().item())
    if candidate.any():
        w = prior_weights[input_ids]
        acc["candidate_prior_sum"] += float(w[candidate].sum().item())
        acc["selected_prior_sum"] += float(w[selected & candidate].sum().item())
        acc["candidate_high125"] += float(((w >= 1.25) & candidate).sum().item())
        acc["selected_high125"] += float(((w >= 1.25) & selected & candidate).sum().item())
        acc["candidate_high150"] += float(((w >= 1.50) & candidate).sum().item())
        acc["selected_high150"] += float(((w >= 1.50) & selected & candidate).sum().item())
        rel = relation_ids[input_ids]
        acc["candidate_relation_lex"] += float((rel & candidate).sum().item())
        acc["selected_relation_lex"] += float((rel & selected & candidate).sum().item())
    bsz = input_ids.shape[0]
    acc["rows"] += float(bsz)
    for b in range(bsz):
        cg = set(int(x) for x in torch.unique(word_group[b][candidate[b] & (word_group[b] >= 0)]).cpu().tolist())
        sg = set(int(x) for x in torch.unique(word_group[b][selected[b] & candidate[b] & (word_group[b] >= 0)]).cpu().tolist())
        acc["candidate_groups"] += float(len(cg))
        acc["selected_groups"] += float(len(sg))


def summarize_acc(acc: dict[str, float]) -> dict[str, float]:
    cand_tok = max(acc["candidate_tokens"], 1.0)
    sel_tok = max(acc["selected_tokens"], 1.0)
    cand_grp = max(acc["candidate_groups"], 1.0)
    sel_grp = max(acc["selected_groups"], 1.0)
    return {
        "rows": acc["rows"],
        "candidate_tokens": acc["candidate_tokens"],
        "selected_tokens": acc["selected_tokens"],
        "token_mask_rate": acc["selected_tokens"] / cand_tok,
        "candidate_groups": acc["candidate_groups"],
        "selected_groups": acc["selected_groups"],
        "group_mask_rate": acc["selected_groups"] / cand_grp,
        "candidate_mean_prior": acc["candidate_prior_sum"] / cand_tok,
        "selected_mean_prior": acc["selected_prior_sum"] / sel_tok,
        "prior_selection_lift": (acc["selected_prior_sum"] / sel_tok) / max(acc["candidate_prior_sum"] / cand_tok, 1e-12),
        "candidate_high125_rate": acc["candidate_high125"] / cand_tok,
        "selected_high125_rate": acc["selected_high125"] / sel_tok,
        "high125_lift": (acc["selected_high125"] / sel_tok) / max(acc["candidate_high125"] / cand_tok, 1e-12),
        "candidate_high150_rate": acc["candidate_high150"] / cand_tok,
        "selected_high150_rate": acc["selected_high150"] / sel_tok,
        "high150_lift": (acc["selected_high150"] / sel_tok) / max(acc["candidate_high150"] / cand_tok, 1e-12),
        "candidate_relation_lex_rate": acc["candidate_relation_lex"] / cand_tok,
        "selected_relation_lex_rate": acc["selected_relation_lex"] / sel_tok,
        "relation_lex_lift": (acc["selected_relation_lex"] / sel_tok) / max(acc["candidate_relation_lex"] / cand_tok, 1e-12),
    }


def mean_std(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs), "min": min(xs), "max": max(xs)}


def run_one_sample(trainer, tokenizer, sample_name: str, rows: list[dict[str, Any]], prior: dict[str, Any], schemes: list[str], seed0: int, num_seeds: int, batch_size: int, seq_length: int) -> dict[str, Any]:
    examples = make_examples(trainer, rows)
    ds = trainer.MaskedChunkDataset(examples, tokenizer, seq_length=seq_length)
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), dtype=torch.long)
    vocab_size = len(tokenizer)
    token_cores = [tok_core(tokenizer.convert_ids_to_tokens(i)) for i in range(vocab_size)]
    relation_ids = torch.tensor([core in RELATION_LEX for core in token_cores], dtype=torch.bool)

    source_counts: dict[str, int] = {}
    for r in rows:
        source_counts[str(r.get("source", ""))] = source_counts.get(str(r.get("source", "")), 0) + 1

    modes: list[tuple[str, str, str | None]] = (
        [(f"wwm_fixed:{s}", s, None) for s in schemes]
        + [(f"wwm_static_prior:{s}", s, s) for s in schemes]
    )
    per_mode: dict[str, Any] = {}
    for mode_name, scheme_for_metrics, scheme_for_curriculum in modes:
        weights = torch.tensor(prior["schemes"][scheme_for_metrics]["weights"], dtype=torch.float32)
        seed_summaries = []
        for si in range(num_seeds):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed0 + si)
            state = trainer.MaskingCurriculumState(
                curriculum="wwm_fixed" if scheme_for_curriculum is None else "wwm_static_prior",
                mask_prob_start=0.15,
                mask_prob_end=0.15,
                switch_frac=0.7,
                amlm_window=10,
                amlm_lambda=0.2,
                static_prior_path="" if scheme_for_curriculum is None else str(PRIOR),
                static_prior_scheme=scheme_for_curriculum or scheme_for_metrics,
            )
            state.initialize(vocab_size=vocab_size, total_steps=1)
            acc_all = zeros()
            acc_changed = zeros()
            acc_other = zeros()
            for start in range(0, len(ds), batch_size):
                end = min(start + batch_size, len(ds))
                batch_rows = rows[start:end]
                batch = trainer.collate([ds[i] for i in range(start, end)])
                _, labels = trainer.apply_masking_curriculum(batch["input_ids"], batch["attention_mask"], batch["word_group"], tokenizer, state, gen)
                selected = labels.ne(-100)
                add_counts(acc_all, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], word_group=batch["word_group"], selected=selected, prior_weights=weights, relation_ids=relation_ids, special_ids=special_ids)
                # Split batch rows without changing the trainer tensors.
                changed_mask = torch.tensor([str(r.get("source", "")) == CHANGED_SOURCE for r in batch_rows], dtype=torch.bool)
                if changed_mask.any():
                    add_counts(acc_changed, input_ids=batch["input_ids"][changed_mask], attention_mask=batch["attention_mask"][changed_mask], word_group=batch["word_group"][changed_mask], selected=selected[changed_mask], prior_weights=weights, relation_ids=relation_ids, special_ids=special_ids)
                if (~changed_mask).any():
                    add_counts(acc_other, input_ids=batch["input_ids"][~changed_mask], attention_mask=batch["attention_mask"][~changed_mask], word_group=batch["word_group"][~changed_mask], selected=selected[~changed_mask], prior_weights=weights, relation_ids=relation_ids, special_ids=special_ids)
            seed_summaries.append({"seed": seed0 + si, "all": summarize_acc(acc_all), "changed": summarize_acc(acc_changed), "other": summarize_acc(acc_other)})
        aggregate = {}
        for split in ["all", "changed", "other"]:
            keys = list(seed_summaries[0][split].keys())
            aggregate[split] = {k: mean_std([float(s[split][k]) for s in seed_summaries]) for k in keys}
        per_mode[mode_name] = {"seed_runs": seed_summaries, "aggregate": aggregate}

    # Pair each static mode against the fixed mode for the most important rates.
    comparisons: dict[str, Any] = {}
    for s in schemes:
        fixed = per_mode[f"wwm_fixed:{s}"]["aggregate"]
        mode = f"wwm_static_prior:{s}"
        comparisons[mode] = {}
        for split in ["all", "changed", "other"]:
            rec = per_mode[mode]["aggregate"][split]
            base = fixed[split]
            comparisons[mode][split] = {
                "group_mask_rate_delta_vs_fixed": rec["group_mask_rate"]["mean"] - base["group_mask_rate"]["mean"],
                "token_mask_rate_delta_vs_fixed": rec["token_mask_rate"]["mean"] - base["token_mask_rate"]["mean"],
                "selected_mean_prior_delta_vs_fixed": rec["selected_mean_prior"]["mean"] - base["selected_mean_prior"]["mean"],
                "relation_lex_lift_delta_vs_fixed": rec["relation_lex_lift"]["mean"] - base["relation_lex_lift"]["mean"],
                "high150_lift_delta_vs_fixed": rec["high150_lift"]["mean"] - base["high150_lift"]["mean"],
            }
    return {
        "sample_name": sample_name,
        "rows": len(rows),
        "words": sum(int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows),
        "source_counts": source_counts,
        "modes": per_mode,
        "comparisons_vs_fixed": comparisons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-front", type=int, default=4096)
    ap.add_argument("--rows-stride", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=43023)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()

    for p in [TRAINER_PATH, POOL, TOKENIZER / "tokenizer.json", PRIOR]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    trainer = load_module(TRAINER_PATH, "static_prior_trainer_module")
    tokenizer = trainer.make_portable_tokenizer(str(TOKENIZER))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    schemes = sorted(prior.get("schemes", {}).keys())
    if schemes != ["relation_info_v1", "relation_only_v1"]:
        raise RuntimeError(f"unexpected schemes {schemes}")
    # Keep relation_only first in reports because it is the narrower candidate.
    schemes = ["relation_only_v1", "relation_info_v1"]

    rows = load_rows(POOL)
    samples = pick_samples(rows, args.rows_front, args.rows_stride)
    results = [run_one_sample(trainer, tokenizer, name, sample_rows, prior, schemes, args.seed0, args.num_seeds, args.batch_size, args.seq_length) for name, sample_rows in samples.items()]

    summary = {
        "status": "STATIC_PRIOR_MASKING_VALIDATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU validation that generated static-prior trainer preserves WWM mask budget and reallocates mask pressure on real legal-pool examples before any GPU training is considered.",
        "inputs": {
            "trainer": str(TRAINER_PATH),
            "pool_10m": str(POOL),
            "tokenizer": str(TOKENIZER),
            "prior_json": str(PRIOR),
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "num_seeds": args.num_seeds,
            "seed0": args.seed0,
            "rows_front": args.rows_front,
            "rows_stride": args.rows_stride,
        },
        "interpretation": [
            "A valid execution asset should keep group/token mask rates close to fixed WWM while raising selected prior weight and relation/high-prior-token enrichment; this validates the intervention mechanism, not BabyLM task improvement.",
            "The validation uses only the legal training pool and corpus-derived prior; it does not read evaluation examples or labels.",
            "If future mature clean-vs-reinvest evidence selects a learning-signal repair, this validation supports a single-arm static-prior screen; it does not support mixing static prior with pair consistency in the first GPU test.",
        ],
        "results": results,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "static_prior_masking_validation.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: float) -> str:
        return f"{x:.4f}"
    lines = [
        "# research static-prior masking validation",
        "",
        summary["purpose"],
        "",
        "This is a CPU-only validation of the generated trainer's actual masking function; it is not a model-training result.",
        "",
        "## Samples and mode comparisons",
        "",
    ]
    for res in results:
        lines.append(f"### {res['sample_name']}")
        lines.append(f"- rows: {res['rows']}; words: {res['words']}; source_counts: {res['source_counts']}")
        lines.append("")
        lines.append("| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode, mrec in res["modes"].items():
            for split in ["all", "changed", "other"]:
                ag = mrec["aggregate"][split]
                comp = res["comparisons_vs_fixed"].get(mode, {}).get(split, {})
                lines.append("| {mode} | {split} | {g} | {t} | {smp} | {pl} | {rl} | {hl} | {dg} | {dt} |".format(
                    mode=mode,
                    split=split,
                    g=fmt(ag["group_mask_rate"]["mean"]),
                    t=fmt(ag["token_mask_rate"]["mean"]),
                    smp=fmt(ag["selected_mean_prior"]["mean"]),
                    pl=fmt(ag["prior_selection_lift"]["mean"]),
                    rl=fmt(ag["relation_lex_lift"]["mean"]),
                    hl=fmt(ag["high150_lift"]["mean"]),
                    dg=fmt(comp.get("group_mask_rate_delta_vs_fixed", 0.0)),
                    dt=fmt(comp.get("token_mask_rate_delta_vs_fixed", 0.0)),
                ))
        lines.append("")
    lines += ["## Input files", ""]
    for k, v in summary["inputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "static_prior_masking_validation.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "samples": [{"name": r["sample_name"], "rows": r["rows"], "words": r["words"]} for r in results],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
