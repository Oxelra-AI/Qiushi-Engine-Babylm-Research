#!/usr/bin/env python3
"""research: one-command post-training readout for packed target-selective 100M arms.

Run only after the runtime has delivered both research training tasks as terminal.
This script intentionally separates three pieces of evidence:

1. training integrity and realized deleted-label mass;
2. official-compatible endpoint task movement;
3. local fixed-event compact-side denoising movement.

The primary causal comparison is always arm-to-arm:
    drop_abs_content  vs  drop_copied_content_wholeword
because those endpoints are trained by the same research fast implementation.
The historical full compact endpoint is retained as a shape/scale reference for the
original compact-view triangle, not as the strict counterfactual arm for the fast run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DROP_ABS_RUN = ROOT / "training/runs/packed_drop_abs_content_100M_fast"
DROP_COPIED_RUN = ROOT / "training/runs/packed_drop_copied_content_wholeword_100M_fast"
EXPECTED_INIT_SHA = "f13f1f85923a6e04d033f755a180f8755be7247005dca327be01c7461493d509"
EXPECTED_EXPOSURE = 100_000_000
EXPECTED_STEPS = 2529
EXPECTED_DROPPED_BPE = {
    "drop_abs": ("dropped_abs_content", 48_105),
    "drop_copied_word": ("dropped_copied_wholeword", 48_387),
}
EVAL_SCRIPT = ROOT / "scripts/eval_packed_targetselect_100M.py"
SIGNATURE_V2 = ROOT / "scripts/packed_targetselect_endpoint_signature_v2.py"
DENOISING = ROOT / "scripts/packed_targetselect_denoising_probe_100M.py"
FULL_JSON = ROOT / "data/compact_triangle_noaoa_eval/per_target/compact_view_reinvest.json"
DEFAULT_OUT = ROOT / "data/packed_targetselect_postrun_readout"
RELATIONAL = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
ADJ_INDEPENDENT = ["material-properties", "social-interactions"]


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def tail(path: pathlib.Path, n: int = 6000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-n:]


def run_cmd(cmd: list[str], log_path: pathlib.Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("RUN " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    rec = {
        "cmd": cmd,
        "returncode": int(p.returncode),
        "elapsed_sec": round(time.time() - t0, 1),
        "log": str(log_path),
    }
    if p.returncode != 0:
        rec["log_tail"] = tail(log_path)
    return rec


def validate_one(name: str, run_dir: pathlib.Path) -> dict[str, Any]:
    metrics_path = run_dir / "scientific_metrics.json"
    model_dir = run_dir / "hf_model/chck_100M"
    rec: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "model_dir": str(model_dir),
        "model_dir_exists": model_dir.exists(),
        "config_exists": (model_dir / "config.json").exists(),
        "model_safetensors_exists": (model_dir / "model.safetensors").exists(),
        "tokenizer_json_exists": (model_dir / "tokenizer.json").exists(),
        "ok": False,
        "problems": [],
    }
    if not metrics_path.exists():
        rec["problems"].append("missing scientific_metrics.json")
        return rec
    m = load_json(metrics_path)
    rec["metrics"] = m
    if int(m.get("word_exposure", -1)) != EXPECTED_EXPOSURE:
        rec["problems"].append(f"word_exposure {m.get('word_exposure')} != {EXPECTED_EXPOSURE}")
    if int(m.get("actual_training_steps", -1)) != EXPECTED_STEPS:
        rec["problems"].append(f"actual_training_steps {m.get('actual_training_steps')} != {EXPECTED_STEPS}")
    if str(m.get("init_sha")) != EXPECTED_INIT_SHA:
        rec["problems"].append(f"init_sha {m.get('init_sha')} != {EXPECTED_INIT_SHA}")
    if not model_dir.exists() or not (model_dir / "model.safetensors").exists() or not (model_dir / "config.json").exists():
        rec["problems"].append("missing final chck_100M model files")
    drop_key, drop_expected = EXPECTED_DROPPED_BPE[name]
    actual_drop = int((m.get("total_dropped_counts") or {}).get(drop_key, -1))
    rec["expected_drop_key"] = drop_key
    rec["expected_drop_bpe"] = drop_expected
    rec["actual_drop_bpe"] = actual_drop
    rec["drop_bpe_delta_actual_minus_expected"] = None if actual_drop < 0 else actual_drop - drop_expected
    if actual_drop != drop_expected:
        rec["problems"].append(f"{drop_key} {actual_drop} != realized-WWM expected {drop_expected}")
    saved = m.get("saved_checkpoints") or []
    if not saved or str(saved[-1].get("name")) != "chck_100M":
        rec["problems"].append("last saved checkpoint is not chck_100M")
    rec["ok"] = len(rec["problems"]) == 0
    return rec


def bootstrap_good(rec: dict[str, Any], positive: bool = True, field: str = "p025") -> bool:
    boot = (rec or {}).get("pair_cluster_bootstrap") or (rec or {}).get("bootstrap_accuracy_delta_b_minus_a_pp") or {}
    v = boot.get(field)
    if v is None:
        return False
    return float(v) > 0 if positive else float(v) < 0


def get_path(d: dict[str, Any], parts: list[str]) -> Any:
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def summarize_mechanism(signature: dict[str, Any], denoise: dict[str, Any]) -> dict[str, Any]:
    """Extract the few numbers that determine the scientific reading.

    For local losses, positive drop_abs_minus_drop_copied_word means removing
    source-absent labels hurts source-absent content prediction.

    For endpoint accuracies, positive drop_copied_word_minus_drop_abs means the
    arm that retained source-absent labels outperformed the arm that removed them.
    """
    cheap = (signature.get("cheap7") or {}).get("deltas") or {}
    item = signature.get("item_transitions") or {}
    rev_key = "drop_copied_word_minus_drop_abs"
    fwd_key = "drop_abs_minus_drop_copied_word"
    supp_rev = get_path(item, ["Supplement", rev_key])
    rel_rev = get_path(item, ["EWoK_groups", "relational_domains", "contrasts", rev_key])
    adj_rev = get_path(item, ["EWoK_groups", "adjacency_independent_domains", "contrasts", rev_key])
    doms = get_path(item, ["EWoK_domain_vectors"]) or {}
    rel_domain_reverse = {d: get_path(doms, [d, rev_key, "accuracy_delta_b_minus_a_pp"]) for d in RELATIONAL}
    adj_domain_reverse = {d: get_path(doms, [d, rev_key, "accuracy_delta_b_minus_a_pp"]) for d in ADJ_INDEPENDENT}

    local = (denoise.get("eval_summary") or {}).get("contrasts") or {}
    local_fwd = local.get(fwd_key, {})
    local_summary = {}
    for es, cats in sorted(local_fwd.items()):
        local_summary[es] = {cat: cats.get(cat) for cat in ["source_absent_content", "retained_content", "function_other"]}

    # Extract compact, fixed scientific reading fields without forcing a route decision.
    local_source_absent = {
        es: rec.get("piece_weighted_delta") if isinstance(rec, dict) else None
        for es, rec in ((es, cats.get("source_absent_content")) for es, cats in local_fwd.items())
    }
    local_source_absent_positive_sets = [
        es for es, rec in ((es, cats.get("source_absent_content")) for es, cats in local_fwd.items())
        if isinstance(rec, dict) and rec.get("piece_weighted_delta") is not None and float(rec["piece_weighted_delta"]) > 0
    ]
    local_source_absent_strong_sets = [
        es for es, rec in ((es, cats.get("source_absent_content")) for es, cats in local_fwd.items())
        if isinstance(rec, dict) and bootstrap_good(rec, positive=True, field="p025")
    ]
    rel_val = None if rel_rev is None else rel_rev.get("accuracy_delta_b_minus_a_pp")
    adj_val = None if adj_rev is None else adj_rev.get("accuracy_delta_b_minus_a_pp")
    supp_val = None if supp_rev is None else supp_rev.get("accuracy_delta_b_minus_a_pp")
    rel_pos_count = sum(1 for v in rel_domain_reverse.values() if v is not None and float(v) > 0)
    adj_pos_count = sum(1 for v in adj_domain_reverse.values() if v is not None and float(v) > 0)

    return {
        "meaning": {
            "local_loss_positive": "drop_abs_minus_drop_copied_word > 0 means source-absent labels were more useful for that compact-side loss category than copied labels.",
            "endpoint_accuracy_positive": "drop_copied_word_minus_drop_abs > 0 means retaining source-absent labels improved official-task accuracy relative to retaining copied labels.",
        },
        "cheap7_deltas": {
            fwd_key: cheap.get(fwd_key),
            rev_key: cheap.get(rev_key),
        },
        "endpoint_reverse_accuracy_fields": {
            "Supplement_pp": supp_val,
            "EWoK_step211_relational_pp": rel_val,
            "EWoK_step211_adjacency_independent_pp": adj_val,
            "relational_minus_adjacency_independent_pp": None if rel_val is None or adj_val is None else round(float(rel_val) - float(adj_val), 6),
            "relational_domain_reverse_pp": rel_domain_reverse,
            "adjacency_independent_domain_reverse_pp": adj_domain_reverse,
            "relational_positive_domain_count": rel_pos_count,
            "adjacency_independent_positive_domain_count": adj_pos_count,
            "Supplement_pair_interval_positive": bootstrap_good(supp_rev or {}, positive=True, field="p025"),
            "relational_pair_interval_positive": bootstrap_good(rel_rev or {}, positive=True, field="p025"),
        },
        "local_loss_fields": {
            "drop_abs_minus_drop_copied_word_by_set": local_summary,
            "source_absent_delta_by_set": local_source_absent,
            "source_absent_positive_sets": local_source_absent_positive_sets,
            "source_absent_pair_interval_positive_sets": local_source_absent_strong_sets,
        },
        "vector_similarity_reverse_to_historical": (signature.get("vector_similarity_to_historical") or {}).get(rev_key),
        "interpretation_note": "A load-bearing historical mediation reading needs both the local source_absent_content loss movement and the endpoint reverse-accuracy pattern on Supplement plus the research relational EWoK domains; local loss alone is a within-coordinate learning channel, not endpoint mediation.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--device_for_denoising", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    integrity = {
        "drop_abs": validate_one("drop_abs", DROP_ABS_RUN),
        "drop_copied_word": validate_one("drop_copied_word", DROP_COPIED_RUN),
    }
    waiting = [k for k, v in integrity.items() if not v.get("metrics_exists") or not v.get("model_dir_exists")]
    failed = [k for k, v in integrity.items() if v.get("metrics_exists") and v.get("model_dir_exists") and not v.get("ok")]
    if waiting or failed or args.validate_only:
        status = "PACKED_TARGETSELECT_POSTRUN_VALIDATE_ONLY" if args.validate_only else "PACKED_TARGETSELECT_POSTRUN_WAITING_OR_NEEDS_REPAIR"
        summary = {
            "status": status,
            "meaning": "Training endpoints must be complete and match the realized WWM deleted-label counts before endpoint/local readout is run.",
            "integrity": integrity,
            "waiting_for_complete_runs": waiting,
            "runs_with_integrity_problems": failed,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        (out / "postrun_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        if failed:
            raise SystemExit(1)
        return

    logs = out / "logs"
    endpoint_out = out / "endpoint_eval"
    eval_cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--drop_abs_run", str(DROP_ABS_RUN),
        "--drop_copied_run", str(DROP_COPIED_RUN),
        "--out_root", str(endpoint_out),
        "--gpus", args.gpus,
        "--skip-signature",
    ]
    if args.force:
        eval_cmd.append("--force")
    eval_rec = run_cmd(eval_cmd, logs / "endpoint_eval.log")
    if eval_rec["returncode"] != 0:
        summary = {"status": "PACKED_TARGETSELECT_ENDPOINT_EVAL_FAILED", "integrity": integrity, "eval": eval_rec}
        (out / "postrun_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    drop_abs_json = endpoint_out / "drop_abs/per_target/dropabs100.json"
    drop_copied_json = endpoint_out / "drop_copied_word/per_target/dropcopied_word100.json"
    sig_out = out / "signature_v2"
    sig_cmd = [
        sys.executable, "-B", str(SIGNATURE_V2),
        "--full-json", str(FULL_JSON),
        "--arm-json", f"drop_abs={drop_abs_json}",
        "--arm-json", f"drop_copied_word={drop_copied_json}",
        "--output_dir", str(sig_out),
    ]
    sig_rec = run_cmd(sig_cmd, logs / "signature_v2.log")
    if sig_rec["returncode"] != 0:
        summary = {"status": "PACKED_TARGETSELECT_SIGNATURE_V2_FAILED", "integrity": integrity, "eval": eval_rec, "signature_v2": sig_rec}
        (out / "postrun_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    denoise_out = out / "denoising_probe"
    denoise_cmd = [
        sys.executable, "-B", str(DENOISING),
        "--output_dir", str(denoise_out),
        "--device", args.device_for_denoising,
        "--force",
    ]
    denoise_rec = run_cmd(denoise_cmd, logs / "denoising_probe.log")
    if denoise_rec["returncode"] != 0:
        summary = {"status": "PACKED_TARGETSELECT_DENOISING_PROBE_FAILED", "integrity": integrity, "eval": eval_rec, "signature_v2": sig_rec, "denoising_probe": denoise_rec}
        (out / "postrun_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    sig_json = sig_out / "packed_targetselect_endpoint_signature_v2.json"
    denoise_json = denoise_out / "packed_targetselect_denoising_probe_100M.json"
    sig_payload = load_json(sig_json)
    denoise_payload = load_json(denoise_json)
    mechanism = summarize_mechanism(sig_payload, denoise_payload)
    summary = {
        "status": "PACKED_TARGETSELECT_POSTRUN_READOUT_DONE",
        "meaning": "Complete post-training readout for the packed historical source-absent versus copied-content target-selective 100M intervention.",
        "integrity": integrity,
        "endpoint_eval": eval_rec,
        "signature_v2": {**sig_rec, "json": str(sig_json)},
        "denoising_probe": {**denoise_rec, "json": str(denoise_json)},
        "mechanism_readout": mechanism,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "postrun_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research packed target-selective post-training readout",
        "",
        f"Summary JSON: `{out / 'postrun_readout_summary.json'}`",
        f"Endpoint signature JSON: `{sig_json}`",
        f"Local denoising JSON: `{denoise_json}`",
        "",
        "## Core reading fields",
        "",
        "Positive local `drop_abs_minus_drop_copied_word` loss means source-absent compact labels carried more predictive value than matched copied labels.",
        "Positive endpoint `drop_copied_word_minus_drop_abs` accuracy means retaining source-absent labels improved task accuracy relative to retaining copied labels.",
        "",
        "```json",
        json.dumps(mechanism, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (out / "postrun_readout_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
