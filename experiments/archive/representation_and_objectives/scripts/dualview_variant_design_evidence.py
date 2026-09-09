#!/usr/bin/env python3
"""research: evidence for broad-preserving coupled dual-view variant design.

CPU/file-only analysis. It does not score models, train models, or touch the
chck_82M endpoint. It prepares the next mechanism decision while the matched
coupled shuffled control is running.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
A02 = Path("experiments/archive/frontier_consolidation")
OUT_DIR = ROOT / "data" / "dualview_variant_design_evidence"
OUT_JSON = OUT_DIR / "dualview_variant_design_evidence.json"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/dualview_variant_design_evidence.md')

AUX_JSON = A02 / "data" / "sparse_aux_pair_data" / "top20" / "sparse_aux_pair_data_top20.json"
TOKENIZER_PATH = A02 / "data" / "compliant_tokenizer"
ANATOMY_JSON = ROOT / "data" / "dualview_broad_damage_anatomy" / "dualview_broad_damage_anatomy.json"

RUNS = {
    "mlm_only_20M": A02 / "training" / "runs" / "dualview_mlm_only_20M_seed43022",
    "coupled_sparse20_aligned_20M": A02 / "training" / "runs" / "coupled_sparse20_aligned_20M_seed43022",
    "sep_sparse20_aligned_20M": A02 / "training" / "runs" / "sep_sparse20_aligned_20M_seed43022",
    "sep_sparse20_shuffled_20M": A02 / "training" / "runs" / "sep_sparse20_shuffled_20M_seed43022",
}

SCRIPT_PATHS = {
    "coupled_trainer": A02 / "scripts" / "dual_view_corrected_trainer.py",
    "separated_trainer": A02 / "scripts" / "detached_private_trainer.py",
    "unscaled_adapter_modeling": A02 / "scripts" / "adapter_modeling.py",
    "scaled_adapter_modeling_chck82": A02 / "training" / "runs" / "adapter128_scale1p75_h100M100M_seed43022_official_ladder" / "hf_model" / "chck_82M" / "adapter_scaled_modeling.py",
}

FUNCTION_WORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","because","so","for","to","of","in","on","at","by","with","from","as","is","are","was","were","be","been","being","am","do","does","did","done","have","has","had","having","will","would","can","could","may","might","must","should","shall","this","that","these","those","it","its","he","she","they","them","we","us","you","your","i","me","my","his","her","their","our","not","no","yes","than","into","about","over","under","between","before","after","up","down","out","off","again","more","most","less","least","some","any","all","each","every","one","two","three","which","what","who","whom","whose","where","why","how"
}

REL_MARKERS = {
    "negation": {"not","no","never","none","without","cannot","can't","won't","doesn't","don't","didn't","isn't","aren't","wasn't","weren't"},
    "quantifier_number": {"all","each","every","some","many","few","most","least","more","less","one","two","three","four","five","first","second","third","single","multiple","several","both","either","neither"},
    "aux_modal": {"is","are","was","were","be","been","being","am","do","does","did","have","has","had","will","would","can","could","may","might","must","should","shall"},
    "spatial": {"in","on","under","over","above","below","inside","outside","between","near","far","left","right","behind","front","across","around","through","into","onto"},
    "causal_dynamic": {"cause","causes","caused","make","makes","made","lead","leads","led","result","results","force","forces","forced","allow","allows","prevent","prevents","increase","decrease","reduce","change","changes","move","moves","moved","push","pull","fall","drop","break","open","close","heat","cool","melt","freeze"},
    "discourse_dialogue": {"said","says","tell","tells","told","ask","asks","asked","answer","answers","replied","reply","conversation","question"},
}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "training_log.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def basic_stats(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        pos = p * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - pos) + s[hi] * (pos - lo)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": q(0.5),
        "p05": q(0.05),
        "p25": q(0.25),
        "p75": q(0.75),
        "p95": q(0.95),
        "min": s[0],
        "max": s[-1],
    }


def log_summary(run_dir: Path) -> dict[str, Any]:
    cfg = read_json(run_dir / "train_config.json") if (run_dir / "train_config.json").exists() else {}
    met = read_json(run_dir / "scientific_metrics.json") if (run_dir / "scientific_metrics.json").exists() else {}
    rows = load_training_log(run_dir)
    aux_rows = [r for r in rows if float(r.get("aux_loss") or 0.0) > 0.0]
    d = {
        "config": cfg,
        "metrics": met,
        "n_log_rows": len(rows),
        "aux_active_rows": len(aux_rows),
    }
    if rows:
        first, last = rows[0], rows[-1]
        d["first_row"] = {k: first.get(k) for k in ["update","loader_step","loss","aux_loss","neutral_loss","batch_words","aux_words","cumulative_main_words","cumulative_aux_words","cumulative_charged_words","masked_tokens","aux_targets","aux_units","aux_conditioned_views","aux_free_views"] if k in first}
        d["last_row"] = {k: last.get(k) for k in ["update","loader_step","loss","aux_loss","neutral_loss","batch_words","aux_words","cumulative_main_words","cumulative_aux_words","cumulative_charged_words","masked_tokens","aux_targets","aux_units","aux_conditioned_views","aux_free_views"] if k in last}
        d["loss_stats"] = basic_stats([float(r.get("loss")) for r in rows if r.get("loss") is not None])
        d["batch_words_stats"] = basic_stats([float(r.get("batch_words")) for r in rows if r.get("batch_words") is not None])
        d["masked_tokens_stats"] = basic_stats([float(r.get("masked_tokens")) for r in rows if r.get("masked_tokens") is not None])
    if aux_rows:
        d["aux_loss_stats_active"] = basic_stats([float(r.get("aux_loss")) for r in aux_rows])
        d["aux_words_stats_active"] = basic_stats([float(r.get("aux_words")) for r in aux_rows])
        d["aux_targets_stats_active"] = basic_stats([float(r.get("aux_targets")) for r in aux_rows])
        d["aux_units_stats_active"] = basic_stats([float(r.get("aux_units")) for r in aux_rows])
        d["aux_conditioned_views_stats_active"] = basic_stats([float(r.get("aux_conditioned_views")) for r in aux_rows])
        d["aux_free_views_stats_active"] = basic_stats([float(r.get("aux_free_views")) for r in aux_rows])
    return d


def decode_tokens(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return ""


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def token_piece_strings(tokenizer, ids: list[int]) -> list[str]:
    toks = tokenizer.convert_ids_to_tokens(ids)
    out = []
    for t in toks:
        s = str(t)
        s = s.replace("Ġ", " ").replace("▁", " ")
        out.append(s)
    return out


def content_markers(ws: list[str]) -> dict[str, Any]:
    c = Counter()
    unique = set(ws)
    for cat, vocab in REL_MARKERS.items():
        hit = sorted(unique & vocab)
        if hit:
            c[cat] = len(hit)
    return dict(c)


def aux_content_summary() -> dict[str, Any]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True, use_fast=True)
    aux = read_json(AUX_JSON)
    pair_data = aux.get("pair_data", {})
    rows = []
    per_eid_counts = []
    pair_ids = set()
    decoy_ids = set()
    source_words = []
    rewrite_words = []
    source_tokens = []
    rewrite_tokens = []
    compression = []
    rw_groups = []
    src_word_counter = Counter()
    rw_word_counter = Counter()
    marker_counter_src = Counter()
    marker_counter_rw = Counter()
    pairs_with_marker = Counter()
    named_or_number_pairs = 0
    function_heavy_pairs = 0
    source_eq_rewrite_pairs = 0
    source_contains_all_rw_words = 0
    source_overlap_fracs = []
    rw_content_fracs = []
    piece_len_counter = Counter()
    example_ids = []
    total_pairs = 0
    for eid_str, rec in pair_data.items():
        pairs = rec.get("pairs", []) or []
        per_eid_counts.append(len(pairs))
        example_ids.append(int(eid_str))
        for pr in pairs:
            total_pairs += 1
            pair_ids.add(pr.get("pair_id"))
            decoy_ids.add(pr.get("decoy_pair_id"))
            sids = [int(x) for x in pr.get("source_ids", [])]
            rids = [int(x) for x in pr.get("rw_ids", [])]
            sw = int(pr.get("source_words") or 0)
            rw = int(pr.get("rewrite_words") or 0)
            source_words.append(sw); rewrite_words.append(rw)
            source_tokens.append(len(sids)); rewrite_tokens.append(len(rids))
            if sw:
                compression.append(rw / sw)
            rw_g = pr.get("rw_word_group") or []
            if rw_g:
                rw_groups.append(len(set(int(x) for x in rw_g if int(x) >= 0)))
            stext = decode_tokens(tok, sids)
            rtext = decode_tokens(tok, rids)
            if stext.strip().lower() == rtext.strip().lower():
                source_eq_rewrite_pairs += 1
            sws = words(stext); rws = words(rtext)
            src_word_counter.update(sws)
            rw_word_counter.update(rws)
            sset = set(sws); rset = set(rws)
            if rws:
                source_overlap_fracs.append(sum(1 for w in rws if w in sset) / len(rws))
                content_count = sum(1 for w in rws if w not in FUNCTION_WORDS)
                rw_content_fracs.append(content_count / len(rws))
                if content_count / len(rws) < 0.45:
                    function_heavy_pairs += 1
            if rset and rset <= sset:
                source_contains_all_rw_words += 1
            if any(w[:1].isupper() for w in re.findall(r"\b[A-Z][A-Za-z]+\b", rtext)) or any(ch.isdigit() for ch in rtext):
                named_or_number_pairs += 1
            ms = content_markers(sws); mr = content_markers(rws)
            marker_counter_src.update(ms)
            marker_counter_rw.update(mr)
            for cat in set(ms) | set(mr):
                pairs_with_marker[cat] += 1
            for ps in token_piece_strings(tok, rids):
                clean = ps.strip()
                if clean:
                    piece_len_counter[len(clean)] += 1
            if len(rows) < 60:
                rows.append({
                    "eid": int(eid_str),
                    "pair_id": pr.get("pair_id"),
                    "source_words": sw,
                    "rewrite_words": rw,
                    "source_tokens": len(sids),
                    "rewrite_tokens": len(rids),
                    "rewrite_over_source_words": (rw / sw if sw else None),
                    "rewrite_word_source_overlap_frac": (source_overlap_fracs[-1] if source_overlap_fracs else None),
                    "rewrite_text": rtext[:300],
                    "source_text": stext[:300],
                })
    summary = aux.get("summary", {})
    return {
        "aux_file_summary": summary,
        "n_example_ids": len(pair_data),
        "example_id_minmax": [min(example_ids), max(example_ids)] if example_ids else None,
        "n_aux_pair_records_seen": total_pairs,
        "n_unique_pair_ids": len(pair_ids),
        "n_unique_decoy_pair_ids": len(decoy_ids),
        "pairs_per_example_stats": basic_stats([float(x) for x in per_eid_counts]),
        "source_words_stats": basic_stats([float(x) for x in source_words]),
        "rewrite_words_stats": basic_stats([float(x) for x in rewrite_words]),
        "source_tokens_stats": basic_stats([float(x) for x in source_tokens]),
        "rewrite_tokens_stats": basic_stats([float(x) for x in rewrite_tokens]),
        "rewrite_over_source_words_stats": basic_stats([float(x) for x in compression]),
        "rw_word_group_count_stats": basic_stats([float(x) for x in rw_groups]),
        "rewrite_word_source_overlap_frac_stats": basic_stats([float(x) for x in source_overlap_fracs]),
        "rewrite_content_word_frac_stats": basic_stats([float(x) for x in rw_content_fracs]),
        "source_contains_all_rewrite_words_pairs": source_contains_all_rw_words,
        "source_eq_rewrite_pairs": source_eq_rewrite_pairs,
        "named_or_number_pairs": named_or_number_pairs,
        "function_heavy_pairs_rw_content_frac_lt_0p45": function_heavy_pairs,
        "top_source_words": src_word_counter.most_common(40),
        "top_rewrite_words": rw_word_counter.most_common(40),
        "marker_counts_source": dict(marker_counter_src),
        "marker_counts_rewrite": dict(marker_counter_rw),
        "pairs_with_marker": dict(pairs_with_marker),
        "rewrite_piece_length_top": piece_len_counter.most_common(20),
        "sample_rows": rows,
    }


def inspect_source_text(path: Path, needles: list[str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    out = {"path": str(path), "exists": path.exists(), "line_count": text.count("\n") + 1}
    for needle in needles:
        out[needle] = (needle in text)
    return out


def compact_delta(anatomy: dict[str, Any]) -> dict[str, Any]:
    score_deltas = anatomy.get("score_deltas_vs_mlm_only", {})
    broad = score_deltas.get("coupled_sparse20_aligned_20M_minus_mlm_only", {})
    sep = score_deltas.get("sep_sparse20_aligned_20M_minus_mlm_only", {})
    sep_shuf = score_deltas.get("sep_sparse20_shuffled_20M_minus_mlm_only", {})
    # Extract most negative coupled UID/report deltas, excluding row-level GlobalPIQA UID lines.
    report = anatomy.get("report_deltas_vs_mlm_only", {}).get("coupled_sparse20_aligned_20M_minus_mlm_only", [])
    worst = []
    best = []
    for item in report:
        if item.get("section") == "UNSECTIONED":
            continue
        if str(item.get("metric")) == "average" and item.get("section") == "AVERAGE ACCURACY":
            continue
        task = item.get("task")
        # GlobalPIQA per-example 0/100 rows are too granular here; use fixed hard readout instead.
        if task in {"GlobalPIQA_parallel", "GlobalPIQA_nonparallel"} and str(item.get("section")) == "UID ACCURACY":
            continue
        delta = item.get("delta")
        if isinstance(delta, (int, float)):
            if delta < -2.0:
                worst.append(item)
            if delta > 2.0:
                best.append(item)
    worst = sorted(worst, key=lambda x: x["delta"])[:20]
    best = sorted(best, key=lambda x: -x["delta"])[:20]
    hard = anatomy.get("hard_surface_deltas_vs_mlm_only", {}).get("coupled_sparse20_aligned_20M_minus_mlm_only", {})
    return {"score_deltas_coupled": broad, "score_deltas_sep_aligned": sep, "score_deltas_sep_shuffled": sep_shuf, "worst_coupled_report_deltas": worst, "best_coupled_report_deltas": best, "hard_surface_coupled_vs_mlm_only": hard}


def training_geometry(logs: dict[str, Any]) -> dict[str, Any]:
    m = {}
    mlm = logs.get("mlm_only_20M", {})
    coupled = logs.get("coupled_sparse20_aligned_20M", {})
    sep = logs.get("sep_sparse20_aligned_20M", {})
    for name, x in logs.items():
        met = x.get("metrics", {})
        main = met.get("total_main_word_exposure")
        aux = met.get("total_aux_word_exposure")
        charged = met.get("total_charged_words")
        m[name] = {
            "main_words": main,
            "aux_words": aux,
            "charged_words": charged,
            "aux_fraction_of_charged": (aux / charged if aux and charged else 0.0),
            "main_word_shortfall_vs_mlm_only": (main - mlm.get("metrics", {}).get("total_main_word_exposure", 0) if main is not None else None),
            "updates": met.get("updates"),
            "first_loss": met.get("first_loss"),
            "final_loss": met.get("final_loss"),
            "mean_loss": met.get("mean_loss"),
            "mean_aux_loss": met.get("mean_aux_loss"),
        }
    if coupled and sep:
        m["coupled_minus_separated_aligned"] = {
            "mean_main_loss_delta": coupled.get("metrics", {}).get("mean_loss", 0) - sep.get("metrics", {}).get("mean_loss", 0),
            "final_main_loss_delta": coupled.get("metrics", {}).get("final_loss", 0) - sep.get("metrics", {}).get("final_loss", 0),
            "mean_aux_loss_delta": coupled.get("metrics", {}).get("mean_aux_loss", 0) - sep.get("metrics", {}).get("mean_aux_loss", 0),
        }
    return m


def make_payload() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = {name: log_summary(path) for name, path in RUNS.items()}
    aux = aux_content_summary()
    anatomy = read_json(ANATOMY_JSON)
    script_facts = {
        "coupled_trainer": inspect_source_text(SCRIPT_PATHS["coupled_trainer"], [
            "main_loss.backward()", "param.requires_grad_(False)", "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "optimizer.step()", "source_free", "conditioned_plus_source_free", "batch_derangement_preserves_source_multiset_per_batch"
        ]),
        "separated_trainer": inspect_source_text(SCRIPT_PATHS["separated_trainer"], [
            "Phase A: Main MLM", "adapter OFF", "stock_optimizer.step()", "adapter_optimizer.step()", "KL(P_stock || P_adapter)", "neutral_lambda", "pathway\": \"separated\""
        ]),
        "unscaled_adapter_modeling": inspect_source_text(SCRIPT_PATHS["unscaled_adapter_modeling"], [
            "self.scale", "adapter_scale", "return update", "layer_output = layer_output + self.adapter(layer_output)"
        ]),
        "scaled_adapter_modeling_chck82": inspect_source_text(SCRIPT_PATHS["scaled_adapter_modeling_chck82"], [
            "self.scale", "adapter_scale", "* self.scale", "return update", "layer_output = layer_output + self.adapter(layer_output)"
        ]),
    }
    geom = training_geometry(logs)
    deltas = compact_delta(anatomy)

    # Mechanistic interpretation kept as structured hypotheses, not conclusions beyond evidence.
    evidence_reading = {
        "aux_exposure_volume": {
            "reading": "Coupled broad loss is unlikely to be explained by simple word-exposure debit alone: coupled aligned charges only about 183k aux words (<1% of 20M) and loses ~188k main words versus MLM-only, yet broad cheap7 drops ~0.99 while separated aligned with the same debit gains cheap7 ~0.51.",
            "support": {
                "coupled_aux_fraction": geom.get("coupled_sparse20_aligned_20M", {}).get("aux_fraction_of_charged"),
                "coupled_main_shortfall": geom.get("coupled_sparse20_aligned_20M", {}).get("main_word_shortfall_vs_mlm_only"),
                "separated_main_shortfall": geom.get("sep_sparse20_aligned_20M", {}).get("main_word_shortfall_vs_mlm_only"),
                "coupled_cheap7_delta": deltas.get("score_deltas_coupled", {}).get("cheap7"),
                "separated_cheap7_delta": deltas.get("score_deltas_sep_aligned", {}).get("cheap7"),
            },
        },
        "interference_source": {
            "reading": "The distinguishing geometry is whether the aux-trained adapter is in the ordinary MLM path. In research coupled training, main MLM runs with adapters enabled and one optimizer steps stock+adapter together after main+aux gradients; aux freezes stock only during the aux pass, so adapter gradients are shaped by both ordinary MLM and source/free rewrite loss. In research separated training, ordinary MLM disables adapters and updates stock only; the adapter is trained on aux plus neutrality against detached stock logits. This makes the broad collapse a coupled-path interference/co-adaptation problem, not simply the presence of the source-free/conditioned auxiliary data.",
            "support": script_facts,
        },
        "source_free_double_pressure": {
            "reading": "Each aux unit contributes both source-conditioned and source-free masked-rewrite views. If true correspondence is confirmed by the pending shuffled control, the future variant should keep the cross-view shared representation pressure but prevent it from rewriting the ordinary MLM path everywhere; if shuffled repairs equally, then source-free/coupled perturbation rather than source correspondence is the active ingredient.",
            "support": {"aux_views": "conditioned_plus_source_free", "pending_decisive_control": "coupled_sparse20_shuffled_20M"},
        },
        "amplitude_warning": {
            "reading": "Amplitude rescaling is not automatically available for the sparse20 research/130 models: their copied `adapter_modeling.py` records adapter_scale but does not multiply updates by it, unlike the scale1.75 endpoint's `adapter_scaled_modeling.py`. Any amplitude experiment for sparse20 requires a code-level scaled modeling repair or explicit adapter disable/toggle, not just editing config.adapter_scale.",
            "support": {
                "unscaled_step102_has_scale_attr": script_facts["unscaled_adapter_modeling"].get("self.scale"),
                "unscaled_step102_has_adapter_scale": script_facts["unscaled_adapter_modeling"].get("adapter_scale"),
                "scaled_step104_multiplies_scale": script_facts["scaled_adapter_modeling_chck82"].get("* self.scale"),
            },
        },
    }

    future_minimal_variants = [
        {
            "name": "coupled_with_main_adapter_dropout_or_stopgrad_screen",
            "purpose": "If aligned-over-shuffled is large, test whether hard-surface repair survives when the adapter is prevented from dominating every ordinary MLM update. The smallest reliable screen is 20M or shorter only after the matched control lands; use broad cheap7 plus fixed EWoK/GlobalPIQA hard readout.",
            "not_launch_now": "pending control must first decide true-correspondence specificity",
            "expected_decision": "continue only if hard EWoK repair remains alignment-specific and cheap7 loss is materially smaller than -0.99 by 20M",
        },
        {
            "name": "separated_plus_coupled_bridge_hybrid",
            "purpose": "Use separated stock-only MLM to preserve broad competence but periodically let a small, explicitly scaled coupled adapter influence main MLM on pair rows or selected source-absent rewrite targets. This targets the observed failure that fully separated preserves broad score but loses the EWoK hard repair.",
            "not_launch_now": "needs correspondence result and perhaps A02 implementation ownership",
            "expected_decision": "worth promoting only if it beats MLM-only on fixed EWoK stable rows, not merely on cheap7 or GlobalPIQA aggregate",
        },
        {
            "name": "source_conditioned_only_vs_source_free_only_micro_screen",
            "purpose": "Decompose whether broad damage is driven by source-free rewrite reconstruction pressure, conditioned source copying, or the necessity of sharing both views. Must be a tiny matched screen/control, not endpoint-scale.",
            "not_launch_now": "the matched shuffled control is the current single decisive H100 job",
            "expected_decision": "if source-free-only damages broad without hard repair, keep source-free only as a representation anchor with neutralization; if conditioned-only loses hard repair, cross-view sharing is load-bearing",
        },
        {
            "name": "explicit_scaled_adapter_modeling_for_sparse20",
            "purpose": "Repair the sparse20 modeling code so config.adapter_scale actually controls branch amplitude, then run a zero-training or short continuation amplitude map on existing 20M checkpoints before any training from scratch. This avoids assuming config edits affect research models.",
            "not_launch_now": "CPU/code design first; do not alter existing evidence checkpoints",
            "expected_decision": "if disabling/scaling the coupled adapter recovers broad but erases hard repair, the repair is in live branch; if broad remains damaged, stock co-adaptation is already embedded",
        },
    ]

    return {
        "status": "PASS",
        "created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "boundary": "CPU-only evidence synthesis while the shuffled-coupled comparison remains pending; no model scoring/training, chck_82M untouched",
        "inputs": {"aux_json": str(AUX_JSON), "anatomy_json": str(ANATOMY_JSON), "runs": {k: str(v) for k, v in RUNS.items()}},
        "logs": logs,
        "training_geometry": geom,
        "aux_content_summary": aux,
        "broad_and_hard_anatomy_compact": deltas,
        "script_pathway_facts": script_facts,
        "evidence_reading": evidence_reading,
        "future_minimal_variants_after_control_only": future_minimal_variants,
    }


def write_note(payload: dict[str, Any]) -> None:
    geom = payload["training_geometry"]
    aux = payload["aux_content_summary"]
    deltas = payload["broad_and_hard_anatomy_compact"]
    reading = payload["evidence_reading"]
    lines: list[str] = []
    lines.append("# research dual-view broad-preserving variant evidence\n\n")
    lines.append("CPU/file-only synthesis while the matched `coupled_sparse20_shuffled_20M` control runs. No polling, no training, no model scoring, and the reproduced `chck_82M` endpoint is untouched.\n\n")
    lines.append("## What changed in the mechanism picture\n\n")
    lines.append("The broad collapse of `coupled_sparse20_aligned_20M` is not plausibly a simple exposure-volume effect. The coupled run charges only ")
    c = geom["coupled_sparse20_aligned_20M"]
    s = geom["sep_sparse20_aligned_20M"]
    lines.append(f"{c['aux_words']:,} aux words out of {c['charged_words']:,} charged words ({100*c['aux_fraction_of_charged']:.3f}%) and has a main-word shortfall of {c['main_word_shortfall_vs_mlm_only']:,} versus MLM-only. Separated aligned has essentially the same debit ({s['aux_words']:,} aux words, {s['main_word_shortfall_vs_mlm_only']:,} main-word shortfall) but gains cheap7 +{deltas['score_deltas_sep_aligned']['cheap7']:.4f}; coupled aligned loses cheap7 {deltas['score_deltas_coupled']['cheap7']:.4f}.\n\n")
    lines.append("The distinguishing implementation fact is coupling: research runs ordinary MLM with adapters enabled and steps one optimizer after main+aux gradients, freezing stock only during the auxiliary pass. research separated runs main MLM with adapters off and stock-only updates, then trains the private adapter separately with aux plus KL neutrality to detached stock logits. Therefore the damage source to repair is coupled-path adapter/main co-adaptation, not just the existence of a source/free rewrite auxiliary object.\n\n")
    lines.append("## Auxiliary object shape\n\n")
    lines.append(f"Top20 aux object: {aux['n_example_ids']:,} example ids, {aux['n_aux_pair_records_seen']:,} pair records, selection `{aux['aux_file_summary'].get('selection')}`. Summary reports selected aux charge per full activation {aux['aux_file_summary'].get('selected_aux_charge_per_full_activation'):,} and selected changed source-absent pieces {aux['aux_file_summary'].get('selected_changed_source_absent_pieces'):,}.\n\n")
    lines.append(f"Rewrite/source compression (words) mean {aux['rewrite_over_source_words_stats']['mean']:.3f}, median {aux['rewrite_over_source_words_stats']['median']:.3f}; rewrite word overlap with source mean {aux['rewrite_word_source_overlap_frac_stats']['mean']:.3f}, median {aux['rewrite_word_source_overlap_frac_stats']['median']:.3f}. Pairs where all rewrite words appear in source: {aux['source_contains_all_rewrite_words_pairs']:,}/{aux['n_aux_pair_records_seen']:,}; exact source=rewrite pairs: {aux['source_eq_rewrite_pairs']:,}.\n\n")
    lines.append(f"Rewrite marker counts include {aux['marker_counts_rewrite']}; pair records with markers include {aux['pairs_with_marker']}. This confirms the selected object is a compact source-to-rewrite transformation set with some relation/function markers, but still mostly compression and lexical reuse rather than dense bidirectional operator orbits.\n\n")
    lines.append("## Broad damage and hard repair to preserve\n\n")
    bd = deltas["score_deltas_coupled"]
    lines.append(f"Coupled aligned vs MLM-only broad score deltas: BLiMP {bd.get('BLiMP'):+.2f}, Supplement {bd.get('Supplement'):+.2f}, Reading {bd.get('Reading'):+.2f}, COMPS {bd.get('COMPS'):+.2f}, Entity {bd.get('Entity'):+.2f}, EWoK aggregate {bd.get('EWoK'):+.2f}, GlobalPIQA aggregate {bd.get('GlobalPIQA'):+.2f}, cheap7 {bd.get('cheap7'):+.4f}.\n\n")
    lines.append("Worst coupled report deltas (excluding per-example GlobalPIQA rows) include:\n\n")
    for item in deltas["worst_coupled_report_deltas"][:12]:
        lines.append(f"- {item['task']} / {item['section']} / {item['metric']}: {item['delta']:+.2f}\n")
    lines.append("\nBest coupled report deltas include:\n\n")
    for item in deltas["best_coupled_report_deltas"][:10]:
        lines.append(f"- {item['task']} / {item['section']} / {item['metric']}: {item['delta']:+.2f}\n")
    lines.append("\nStep178 already localized hard EWoK repair mainly to agent-properties, physical-relations, physical-interactions/social/spatial domains and variable-swap ContextDiff rows. These are the surfaces the pending matched shuffled control must read before any variant is launched.\n\n")
    lines.append("## Important implementation warning\n\n")
    lines.append("The sparse20 research/130 models use `adapter_modeling.py`, where `config.adapter_scale` is recorded but the adapter output is returned unmultiplied. In contrast, `chck_82M` scale1.75 uses `adapter_scaled_modeling.py`, where the update is explicitly multiplied by `self.scale`. Thus future sparse20 amplitude tests require a code-level scaled-modeling repair or explicit adapter toggling; changing `config.adapter_scale` alone is not evidence.\n\n")
    lines.append("## How to use this after the pending control lands\n\n")
    lines.append("If `coupled_sparse20_aligned` strongly beats matched `coupled_sparse20_shuffled` on the 1,471-row EWoK stable-reversal surface with coherent GlobalPIQA hard52 rank/margin movement, preserve coupled true-correspondence as the mechanism and test the smallest broad-preserving coupled variant. The first variants should target coupled-path interference: adapter influence on ordinary MLM, shared source-free/conditioned pressure, and explicit adapter scaling/neutrality. If shuffled repairs EWoK near aligned, the hard-row movement is not true-correspondence-specific and the proposed mechanism requires reconsideration.\n\n")
    lines.append("Proposed minimal variants are recorded in the JSON under `future_minimal_variants_after_control_only`; none should be launched before the matched control readout.\n\n")
    lines.append(f"JSON: `{OUT_JSON}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    payload = make_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
