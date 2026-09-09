#!/usr/bin/env python3
"""Audit hybrid source overlap and WWM-to-token training dynamics.

This script is CPU-side evidence while managed GPU jobs run. It answers two
questions that affect the next expensive BabyLM Strict-Small run:

1. Does the research clean-Qwen + semantic-view hybrid add mostly new source text
   relative to COMPACT_EXPERIENCE selected clean-Qwen originals, or does it duplicate the same
   source sentences?
2. Does the completed WWM-to-token run differ from fixed WWM mainly as an easier
   MLM loss, or as a training pathology visible before downstream evaluation?

It writes JSON and Markdown notes; it does not launch training or evaluation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
OUT_JSON = ROOT / "training/data/hybrid_cleanqwen_semantic_view/hybrid_overlap_and_training_dynamics.json"
OUT_MD = (ROOT.parents[2] / 'research/notes/representation_and_objectives/hybrid_overlap_and_recipe_training_dynamics.md')
SELECTED_PAIRS = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/selected_pairs.jsonl"
SEM_PROMPTS = ROOT / "training/data/factual_prompts_shard1_simpara.jsonl"
SEM_ACCEPTED = ROOT / "training/data/semantic_view/full_contrast_capped1/accepted_views.jsonl"
SEM_META = ROOT / "training/data/semantic_view/full_contrast_capped1/semantic_view_materialization_metadata.json"
HYBRID_AUDIT = ROOT / "training/data/hybrid_cleanqwen_semantic_view/hybrid_materialization_audit.json"
WWM_TOKEN_METRICS = ROOT / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/scientific_metrics.json"
WWM_TOKEN_LOG = ROOT / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/training_log.jsonl"
BASE_WWM_METRICS = COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022/scientific_metrics.json"
BASE_WWM_LOG = COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022/training_log.jsonl"


def norm_text(s: str) -> str:
    return " ".join(str(s).replace("\u00a0", " ").split())


def text_key(s: str) -> str:
    """Case-insensitive normalized text key for cross-corpus overlap."""
    return hashlib.sha256(norm_text(s).lower().encode("utf-8")).hexdigest()


def materializer_source_key(s: str) -> str:
    """Match materialize_semantic_view_contrast.sha256_text exactly."""
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()


def words(s: str) -> int:
    return len(norm_text(s).split())


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(vals: Iterable[float | int]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {"n": len(xs), "min": min(xs), "p05": q(0.05), "mean": round(statistics.mean(xs), 4), "median": statistics.median(xs), "p95": q(0.95), "max": max(xs), "sum": sum(xs)}


def parse_log(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("{"):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("event") == "train" or ("step" in o and "loss" in o):
                rows.append(o)
    return rows


def summarize_training(metrics_path: pathlib.Path, log_path: pathlib.Path) -> dict[str, Any]:
    m = load_json(metrics_path)
    logs = parse_log(log_path)
    by_mode = collections.defaultdict(list)
    for r in logs:
        by_mode[str(r.get("mask_mode", "unknown"))].append(float(r.get("loss")))
    mode_loss_stats = {k: stats(v) for k, v in by_mode.items()}
    first_last_by_mode = {}
    for k, rs in collections.defaultdict(list, {mode: [r for r in logs if str(r.get('mask_mode', 'unknown')) == mode] for mode in by_mode}).items():
        if rs:
            first_last_by_mode[k] = {"first": rs[0], "last": rs[-1]}
    return {
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "variant": m.get("variant"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "masking_curriculum": m.get("masking_curriculum"),
        "switch_frac": m.get("switch_frac"),
        "parameter_count": m.get("parameter_count"),
        "saved_checkpoints": [c.get("name") for c in m.get("saved_checkpoints", [])],
        "log_train_events": len(logs),
        "log_first": logs[0] if logs else None,
        "log_last": logs[-1] if logs else None,
        "mask_modes_in_log": list(by_mode.keys()),
        "mode_loss_stats": mode_loss_stats,
        "first_last_by_mode": first_last_by_mode,
    }


def main() -> None:
    selected = load_jsonl(SELECTED_PAIRS)
    sem_prompts = load_jsonl(SEM_PROMPTS)
    accepted = load_jsonl(SEM_ACCEPTED)
    sem_meta = load_json(SEM_META)
    hybrid = load_json(HYBRID_AUDIT)

    selected_by_key = {text_key(p.get("original", "")): p for p in selected}
    selected_keys_by_source = collections.defaultdict(set)
    selected_words_by_source = collections.Counter()
    for p in selected:
        k = text_key(p.get("original", ""))
        selected_keys_by_source[p.get("source", "unknown")].add(k)
        selected_words_by_source[p.get("source", "unknown")] += int(p.get("original_words", words(p.get("original", ""))))

    accepted_keys = {a.get("source_key") for a in accepted}
    prompt_by_source_key = {}
    for p in sem_prompts:
        source_text = p.get("source_text", "")
        source_key = p.get("source_key") or materializer_source_key(source_text)
        if source_key in accepted_keys:
            pp = dict(p)
            pp["source_key"] = source_key
            prompt_by_source_key.setdefault(source_key, pp)
    semantic_source_texts = list(prompt_by_source_key.values())
    sem_text_keys = {text_key(p.get("source_text", "")) for p in semantic_source_texts}
    overlap_all = sem_text_keys & set(selected_by_key)
    overlap_simplewiki = sem_text_keys & selected_keys_by_source.get("simple_wiki", set())

    overlap_examples = []
    for k in list(overlap_all)[:20]:
        p = selected_by_key[k]
        overlap_examples.append({"source": p.get("source"), "example_id": p.get("example_id"), "words": p.get("original_words"), "text": norm_text(p.get("original", ""))[:300]})

    sem_articles = collections.Counter(p.get("source_article", "") for p in semantic_source_texts)
    qwen_sources = collections.Counter(p.get("source", "unknown") for p in selected)

    wwm_token = summarize_training(WWM_TOKEN_METRICS, WWM_TOKEN_LOG)
    base_wwm = summarize_training(BASE_WWM_METRICS, BASE_WWM_LOG)
    loss_delta = None
    try:
        loss_delta = float(wwm_token["loss_last"]) - float(base_wwm["loss_last"])
    except Exception:
        pass

    payload = {
        "status": "HYBRID_OVERLAP_AND_TRAINING_DYNAMICS",
        "hybrid_audit": str(HYBRID_AUDIT),
        "semantic_materialization": str(SEM_META),
        "selected_cleanqwen_pairs": str(SELECTED_PAIRS),
        "overlap": {
            "selected_cleanqwen_pairs": len(selected),
            "selected_cleanqwen_unique_original_texts": len(selected_by_key),
            "selected_cleanqwen_simplewiki_unique_original_texts": len(selected_keys_by_source.get("simple_wiki", set())),
            "semantic_accepted_views": len(accepted),
            "semantic_unique_source_texts_from_prompts": len(sem_text_keys),
            "exact_overlap_semantic_sources_vs_all_selected_cleanqwen_originals": len(overlap_all),
            "exact_overlap_semantic_sources_vs_selected_simplewiki_cleanqwen_originals": len(overlap_simplewiki),
            "exact_overlap_word_mass_in_selected_originals": sum(int(selected_by_key[k].get("original_words", words(selected_by_key[k].get("original", "")))) for k in overlap_all),
            "overlap_examples": overlap_examples,
        },
        "source_distributions": {
            "selected_cleanqwen_pair_sources": dict(qwen_sources),
            "selected_cleanqwen_original_words_by_source": dict(selected_words_by_source),
            "semantic_source_articles_top20": sem_articles.most_common(20),
            "semantic_source_articles_total": len(sem_articles),
        },
        "hybrid": {
            "qwen_pair_words": hybrid.get("qwen_pair_words"),
            "semantic_packet_words": hybrid.get("semantic_packet_words"),
            "prefix_word_fraction": hybrid.get("prefix_word_fraction"),
            "hybrid_rows": hybrid.get("hybrid_rows"),
            "source_word_counts_treatment": hybrid.get("source_word_counts_treatment"),
            "required_checks": hybrid.get("required_checks"),
            "all_required_checks_pass": hybrid.get("all_required_checks_pass"),
        },
        "training_dynamics": {
            "wwm_to_token": wwm_token,
            "clean_qwen_fixed_wwm_seed43022": base_wwm,
            "loss_last_delta_wwm_to_token_minus_fixed_wwm": loss_delta,
            "interpretation": "The WWM-to-token run's lower final MLM loss is not by itself evidence of better downstream language competence because token-level subword masking changes target difficulty; official-compatible evaluation is still decisive.",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research hybrid overlap and WWM-to-token training dynamics\n\n"]
    ov = payload["overlap"]
    lines.append("## Source overlap\n\n")
    lines.append(f"- COMPACT_EXPERIENCE selected clean-Qwen originals: {ov['selected_cleanqwen_pairs']:,} pairs, {ov['selected_cleanqwen_unique_original_texts']:,} unique original texts; SimpleWiki subset {ov['selected_cleanqwen_simplewiki_unique_original_texts']:,}.\n")
    lines.append(f"- REPRESENTATION_FRONTIER_STUDIES semantic-view selected sources recovered from prompts: {ov['semantic_unique_source_texts_from_prompts']:,} unique source texts for {ov['semantic_accepted_views']:,} accepted generated views.\n")
    lines.append(f"- Exact normalized overlap between semantic sources and all clean-Qwen selected originals: {ov['exact_overlap_semantic_sources_vs_all_selected_cleanqwen_originals']:,} texts, word mass {ov['exact_overlap_word_mass_in_selected_originals']:,}.\n")
    lines.append(f"- Exact normalized overlap against the clean-Qwen SimpleWiki subset: {ov['exact_overlap_semantic_sources_vs_selected_simplewiki_cleanqwen_originals']:,} texts.\n")
    lines.append("This means the hybrid candidate mostly adds new SimpleWiki source texts relative to the selected COMPACT_EXPERIENCE clean-Qwen originals at exact-string level, while still reweighting SimpleWiki rather than broadening outside official sources.\n\n")
    hy = payload["hybrid"]
    lines.append("## Hybrid candidate scale\n\n")
    lines.append(f"- Clean-Qwen pair words: {hy['qwen_pair_words']:,}; semantic packet words: {hy['semantic_packet_words']:,}; combined prefix fraction {100*float(hy['prefix_word_fraction']):.2f}%.\n")
    lines.append(f"- Required construction checks pass: {hy['all_required_checks_pass']}; rows {hy['hybrid_rows']}.\n\n")
    td = payload["training_dynamics"]
    wt = td["wwm_to_token"]
    bw = td["clean_qwen_fixed_wwm_seed43022"]
    lines.append("## Training dynamics: fixed WWM vs WWM→token\n\n")
    lines.append(f"- WWM→token completed {wt['word_exposure']:,} words in {wt['actual_training_steps']} steps; loss {wt['loss_first']:.4f} → {wt['loss_last']:.4f}; log modes {wt['mask_modes_in_log']}.\n")
    lines.append(f"- Clean-Qwen fixed WWM reference completed {bw['word_exposure']:,} words in {bw['actual_training_steps']} steps; loss {bw['loss_first']:.4f} → {bw['loss_last']:.4f}.\n")
    if loss_delta is not None:
        lines.append(f"- Final loss delta WWM→token minus fixed WWM: {loss_delta:.4f}. This lower loss is not by itself a downstream competence result because token masking changes target granularity.\n")
    lines.append("- The managed no-AoA evaluation task will decide whether the recipe change helps BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading.\n\n")
    lines.append(f"JSON: `{OUT_JSON}`\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "overlap_exact_all": ov["exact_overlap_semantic_sources_vs_all_selected_cleanqwen_originals"],
        "overlap_exact_simplewiki": ov["exact_overlap_semantic_sources_vs_selected_simplewiki_cleanqwen_originals"],
        "semantic_unique_source_texts": ov["semantic_unique_source_texts_from_prompts"],
        "loss_last_delta_wwm_to_token_minus_fixed_wwm": loss_delta,
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
