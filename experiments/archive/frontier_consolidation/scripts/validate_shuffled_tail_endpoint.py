#!/usr/bin/env python3
"""research validation for the frozen-82M shuffled private-tail endpoint hypothesis.

This script is deliberately static / low-cost: it does not train or re-evaluate the
model.  It verifies whether the research shuffled 4M private-tail function is a real,
legal, loadable endpoint candidate before spending GPU on deterministic replay and
score repetition.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/shuffled_tail_endpoint_validation')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/shuffled_tail_endpoint_validation/shuffled_tail_endpoint_validation.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/shuffled_tail_endpoint_validation/shuffled_tail_endpoint_validation.md')

SHUFFLED_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022')
SHUFFLED_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final')
PROTECTED_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
LEGAL_TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
LEGAL_POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
LEGAL_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
TRAIN_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/scientific_metrics.json')
CHEAP_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json')
SUPERGLUE_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_superglue_summary/frozen82_tail4M_shuffled_superglue_superglue_summary.json')
SCORE_DECISION = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_score_decision/frozen82_tail_score_decision.json')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
LIVE_CHECK_VALIDITY = _public_path('experiments/archive/frontier_consolidation/data/live_leaderboard_space_repo/src/submission/check_validity.py')

EXPECTED = {
    "protected_model_sha256": "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3",
    "legal_pool_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "legal_stream_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    "legal_tokenizer_json_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "initial_words": 82_012_495,
    "full_cap_words": 100_000_000,
    "shuffled_tail_charged_words": 3_992_918,
    "shuffled_total_consumed_words": 86_005_413,
    "protected_overall": 41.942481167385985,
    "protected_cheap7": 43.95944987645173,
    "protected_superglue": 69.7661813713118,
    "shuffled_cheap7": 44.01285714285714,
    "shuffled_superglue": 69.7661813713118,
    "shuffled_projected_overall_aoa0": 41.984020152367975,
    "shuffled_margin_vs_protected": 0.04153898498199027,
    "trusted_total_params": 36_458_592,
    "slow_frozen_params": 35_463_008,
    "private_params": 995_584,
}


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_env() -> dict[str, str]:
    cache = _public_path('experiments/archive/frontier_consolidation/.hf_cache_shuffled_tail_validation')
    env = {
        "HF_HOME": str(cache / "hf_home"),
        "HF_HUB_CACHE": str(cache / "hf_home/hub"),
        "TRANSFORMERS_CACHE": str(cache / "hf_home/transformers"),
        "HF_MODULES_CACHE": str(cache / "hf_modules_cache"),
        "HF_DATASETS_CACHE": str(cache / "hf_datasets"),
        "TMPDIR": str(cache / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for k, v in env.items():
        os.environ[k] = v
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
    return env


def param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def tokenizer_equivalence() -> dict[str, Any]:
    from transformers import AutoTokenizer as _AT
    legal = _AT.from_pretrained(LEGAL_TOKENIZER, trust_remote_code=True)
    protected = _AT.from_pretrained(PROTECTED_MODEL, trust_remote_code=True)
    tail = _AT.from_pretrained(SHUFFLED_MODEL, trust_remote_code=True)
    texts = [
        "Hello world!",
        "The child put the red cup on the table because it was clean.",
        "A plastic bag is filled with air and then sealed.",
        "weird token test",
        "mixed 123 numbers and CAPS",
    ]
    enc_equal = True
    for t in texts:
        a = legal(t, add_special_tokens=False)["input_ids"]
        b = protected(t, add_special_tokens=False)["input_ids"]
        c = tail(t, add_special_tokens=False)["input_ids"]
        if not (a == b == c):
            enc_equal = False
    return {
        "vocab_equal_legal_vs_tail": legal.get_vocab() == tail.get_vocab(),
        "vocab_equal_legal_vs_protected": legal.get_vocab() == protected.get_vocab(),
        "encoding_equal_across_legal_protected_tail": enc_equal,
        "tail_special_ids": tail.all_special_ids,
        "protected_special_ids": protected.all_special_ids,
        "tail_tokenizer_json_matches_protected_serialization": (
            sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/tokenizer.json')) == sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/tokenizer.json'))
        ),
        "note": "The tail checkpoint tokenizer.json serialization is byte-identical to the verified protected chck_82M endpoint tokenizer.json; both differ from the original tokenizer-training JSON only by HF re-serialization while preserving identical vocabulary and encoding.",
    }


def tensor_checks() -> dict[str, Any]:
    tail_state = load_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/model.safetensors'), device="cpu")
    protected_state = load_file(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors'), device="cpu")
    common_non_private = []
    non_private_max_abs = 0.0
    non_private_numel = 0
    missing_in_protected = []
    unexpected_tail_non_private = []
    private_norm2 = 0.0
    private_numel = 0
    private_abs_max = 0.0
    private_keys = []
    for k, v in tail_state.items():
        if ".private_adapter." in k:
            private_keys.append(k)
            vf = v.float()
            private_norm2 += float((vf * vf).sum().item())
            private_numel += int(v.numel())
            private_abs_max = max(private_abs_max, float(vf.abs().max().item()))
            continue
        if k in protected_state:
            d = (v.float() - protected_state[k].float()).abs()
            if d.numel():
                non_private_max_abs = max(non_private_max_abs, float(d.max().item()))
                non_private_numel += int(d.numel())
            common_non_private.append(k)
        else:
            unexpected_tail_non_private.append(k)
    for k in protected_state.keys():
        if k not in tail_state:
            missing_in_protected.append(k)
    return {
        "tail_model_safetensors_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/model.safetensors')),
        "protected_model_safetensors_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')),
        "tail_config_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/config.json')),
        "tail_tokenizer_json_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/tokenizer.json')),
        "legal_tokenizer_json_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')),
        "legal_pool_sha256": sha256_file(LEGAL_POOL),
        "legal_stream_sha256": sha256_file(LEGAL_STREAM),
        "common_non_private_tensor_count": len(common_non_private),
        "non_private_max_abs_diff_vs_protected": non_private_max_abs,
        "non_private_numel_compared": non_private_numel,
        "missing_in_tail_from_protected": missing_in_protected[:20],
        "missing_in_tail_from_protected_count": len(missing_in_protected),
        "unexpected_tail_non_private_keys": unexpected_tail_non_private[:20],
        "unexpected_tail_non_private_key_count": len(unexpected_tail_non_private),
        "private_tensor_count": len(private_keys),
        "private_numel": private_numel,
        "private_rms": math.sqrt(private_norm2 / private_numel) if private_numel else None,
        "private_abs_max": private_abs_max,
        "private_keys_sample": private_keys[:8],
    }


def model_load_checks() -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(SHUFFLED_MODEL, trust_remote_code=True)
    batch = tok([
        "The child put the red cup on the table because it was clean.",
        "A plastic bag is filled with air and then sealed.",
    ], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        trusted = AutoModelForMaskedLM.from_pretrained(SHUFFLED_MODEL, trust_remote_code=True).eval()
        protected = AutoModelForMaskedLM.from_pretrained(PROTECTED_MODEL, trust_remote_code=True).eval()
        out_on = trusted(**batch).logits.detach().float()
        # Verify the private branch is active by comparing to the same model with private off.
        if hasattr(trusted, "set_private_enabled"):
            trusted.set_private_enabled(False)
        out_private_off = trusted(**batch).logits.detach().float()
        out_protected = protected(**batch).logits.detach().float()
        # Restore enabled for class/config reporting.
        if hasattr(trusted, "set_private_enabled"):
            trusted.set_private_enabled(True)
        out_on_again = trusted(**batch).logits.detach().float()
    native_info: dict[str, Any] = {}
    try:
        native = AutoModelForMaskedLM.from_pretrained(SHUFFLED_MODEL, trust_remote_code=False).eval()
        native_info = {
            "load_ok": True,
            "class": native.__class__.__name__,
            "param_count": param_count(native),
            "is_scored_function": False,
        }
    except Exception as e:  # pragma: no cover - diagnostic only
        native_info = {"load_ok": False, "error_type": type(e).__name__, "error": str(e)[:500], "is_scored_function": False}
    return {
        "trusted_class": trusted.__class__.__name__,
        "trusted_param_count": param_count(trusted),
        "protected_class": protected.__class__.__name__,
        "protected_param_count": param_count(protected),
        "private_enabled_config_after_restore": bool(getattr(trusted.config, "private_adapter_enabled", False)),
        "private_scale": float(getattr(trusted.config, "private_adapter_scale", float("nan"))),
        "slow_adapter_scale": float(getattr(trusted.config, "adapter_scale", float("nan"))),
        "finite_logits": bool(torch.isfinite(out_on).all().item()),
        "max_abs_logits_private_on_vs_off": float((out_on - out_private_off).abs().max().item()),
        "mean_abs_logits_private_on_vs_off": float((out_on - out_private_off).abs().mean().item()),
        "max_abs_logits_private_off_vs_protected": float((out_private_off - out_protected).abs().max().item()),
        "mean_abs_logits_private_off_vs_protected": float((out_private_off - out_protected).abs().mean().item()),
        "max_abs_logits_private_on_restore_diff": float((out_on - out_on_again).abs().max().item()),
        "native_or_nontrust_load": native_info,
    }


def score_and_legal_checks() -> dict[str, Any]:
    metrics = load_json(TRAIN_METRICS)
    cheap = load_json(CHEAP_SUMMARY)
    sg = load_json(SUPERGLUE_SUMMARY)
    decision = load_json(SCORE_DECISION)
    chck82 = load_json(CHCK82_VERIFY)
    scores = dict(cheap["scores"])
    scores["SuperGLUE"] = float(sg["superglue"])
    scores["AoA"] = 0.0
    overall = sum(float(v) for v in scores.values()) / 9.0
    protected_overall = float(decision["protected_chck82"]["overall"])
    return {
        "training_metrics_path": rel(TRAIN_METRICS),
        "cheap_summary_path": rel(CHEAP_SUMMARY),
        "superglue_summary_path": rel(SUPERGLUE_SUMMARY),
        "score_decision_path": rel(SCORE_DECISION),
        "mode": metrics.get("mode"),
        "updates": metrics.get("updates"),
        "initial_consumed_words": metrics.get("initial_consumed_words"),
        "skip_rows": metrics.get("skip_rows"),
        "tail_main_word_exposure": metrics.get("tail_main_word_exposure"),
        "tail_aux_word_exposure": metrics.get("tail_aux_word_exposure"),
        "tail_charged_words": metrics.get("tail_charged_words"),
        "total_consumed_words": metrics.get("total_consumed_words"),
        "full_cap_words": 100_000_000,
        "within_legal_cap": int(metrics.get("total_consumed_words", 10**18)) <= 100_000_000,
        "charged_words_match_expected": metrics.get("tail_charged_words") == EXPECTED["shuffled_tail_charged_words"],
        "total_words_match_expected": metrics.get("total_consumed_words") == EXPECTED["shuffled_total_consumed_words"],
        "trainable": metrics.get("trainable"),
        "final_aux_loss": metrics.get("final_aux_loss"),
        "mean_neutral_loss": metrics.get("mean_neutral_loss"),
        "cheap_scores": cheap["scores"],
        "cheap7": float(cheap["cheap7"]),
        "superglue": float(sg["superglue"]),
        "aoa_verified_as_scalar_zero": 0.0,
        "nine_column_scores_used_for_projection": scores,
        "projected_overall_with_aoa0_recomputed": overall,
        "protected_overall": protected_overall,
        "margin_vs_protected_recomputed": overall - protected_overall,
        "decision_route_read": decision.get("route_read"),
        "intended_correspondence_supported": decision.get("panel_decisions_plus_probe", {}).get("intended_correspondence_source_free_supported"),
        "chck82_verification_status": chck82.get("status"),
        "chck82_reproduced_score": chck82.get("score_arithmetic", {}).get("overall"),
    }


def validator_scalar_aoa_check() -> dict[str, Any]:
    # We do not need a full prediction carrier here.  We test the exact current live
    # validator branch: if `aoa` is a scalar-score dict rather than raw `results`,
    # the validator accepts it; eval_submission maps it to the same scalar.  This is
    # the way the candidate is ranked with AoA=0 until a positive AoA ladder is worth
    # computing.
    text = LIVE_CHECK_VALIDITY.read_text(encoding="utf-8")
    has_scalar_escape = 'if "aoa" in predictions and predictions["aoa"] is not None and "results" in predictions["aoa"]' in text
    return {
        "live_check_validity_path": rel(LIVE_CHECK_VALIDITY),
        "scalar_aoa_accepted_by_current_validator_logic": bool(has_scalar_escape),
        "aoa_score_used": 0.0,
        "interpretation": "Current submit validator only performs 8005-row checkpoint checks when aoa contains raw `results`; a scalar {'aoa': 0.0} is accepted and eval_submission scores it as 0.0. This verifies the ranking assumption relative to protected chck82, which also has AoA 0.0.",
    }


def make_md(payload: dict[str, Any]) -> str:
    sc = payload["score_and_legal"]
    tc = payload["tensor_checks"]
    lc = payload["trusted_load_checks"]
    lines = [
        "# research shuffled frozen-tail endpoint validation",
        "",
        f"Status: **{payload['status']}**",
        f"Route read: `{payload['route_read']}`",
        "",
        "## Score arithmetic and legal exposure",
        f"- Projected Overall with AoA=0: `{sc['projected_overall_with_aoa0_recomputed']}`",
        f"- Protected chck_82M Overall: `{sc['protected_overall']}`",
        f"- Margin vs protected: `{sc['margin_vs_protected_recomputed']}`",
        f"- Cheap7: `{sc['cheap7']}`; SuperGLUE: `{sc['superglue']}`; AoA scalar: `0.0`",
        f"- Total consumed words: `{sc['total_consumed_words']}` / `{sc['full_cap_words']}`; within cap: `{sc['within_legal_cap']}`",
        f"- Tail charged words: `{sc['tail_charged_words']}` (main `{sc['tail_main_word_exposure']}`, auxiliary `{sc['tail_aux_word_exposure']}`)",
        "",
        "## Function and loading checks",
        f"- Trusted class: `{lc['trusted_class']}`; parameters: `{lc['trusted_param_count']}`",
        f"- Protected class: `{lc['protected_class']}`; parameters: `{lc['protected_param_count']}`",
        f"- Non-private tensor max abs diff vs protected: `{tc['non_private_max_abs_diff_vs_protected']}` across `{tc['non_private_numel_compared']}` elements",
        f"- Private tensors: `{tc['private_tensor_count']}` tensors, `{tc['private_numel']}` params, RMS `{tc['private_rms']}`, abs max `{tc['private_abs_max']}`",
        f"- Private ON vs OFF max logit diff on smoke batch: `{lc['max_abs_logits_private_on_vs_off']}`",
        f"- Private OFF vs protected max logit diff on smoke batch: `{lc['max_abs_logits_private_off_vs_protected']}`",
        f"- Non-trust AutoModel load: `{lc['native_or_nontrust_load']}`",
        "",
        "## AoA ranking assumption",
        f"- Scalar AoA=0 accepted by current validator logic: `{payload['aoa_validator']['scalar_aoa_accepted_by_current_validator_logic']}`",
        "",
        "Scientific reading: the shuffled tail is a real generic private-tail endpoint hypothesis if and only if trusted load, legal exposure, frozen slow-path equality, active private-path evidence, scalar AoA=0 acceptance, and score arithmetic all pass. It is not evidence for source-correspondence transfer; that route remains closed by aligned<shuffled score and worse source-free NLL.",
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = setup_env()
    tensor = tensor_checks()
    tok_equiv = tokenizer_equivalence()
    load = model_load_checks()
    score = score_and_legal_checks()
    aoa = validator_scalar_aoa_check()

    pass_core = all([
        tensor["protected_model_safetensors_sha256"] == EXPECTED["protected_model_sha256"],
        tensor["legal_pool_sha256"] == EXPECTED["legal_pool_sha256"],
        tensor["legal_stream_sha256"] == EXPECTED["legal_stream_sha256"],
        tensor["legal_tokenizer_json_sha256"] == EXPECTED["legal_tokenizer_json_sha256"],
        tok_equiv["encoding_equal_across_legal_protected_tail"],
        tok_equiv["vocab_equal_legal_vs_tail"],
        tok_equiv["tail_tokenizer_json_matches_protected_serialization"],
        tensor["non_private_max_abs_diff_vs_protected"] == 0.0,
        tensor["missing_in_tail_from_protected_count"] == 0,
        tensor["unexpected_tail_non_private_key_count"] == 0,
        tensor["private_numel"] == EXPECTED["private_params"],
        tensor["private_abs_max"] > 0.0,
        load["trusted_class"] == "FrozenSlowPrivateDebertaV2ForMaskedLM",
        load["trusted_param_count"] == EXPECTED["trusted_total_params"],
        load["protected_param_count"] == EXPECTED["slow_frozen_params"],
        load["finite_logits"],
        load["max_abs_logits_private_on_vs_off"] > 0.0,
        load["max_abs_logits_private_off_vs_protected"] == 0.0,
        load["max_abs_logits_private_on_restore_diff"] == 0.0,
        score["within_legal_cap"],
        score["charged_words_match_expected"],
        score["total_words_match_expected"],
        abs(score["projected_overall_with_aoa0_recomputed"] - EXPECTED["shuffled_projected_overall_aoa0"]) < 1e-12,
        abs(score["margin_vs_protected_recomputed"] - EXPECTED["shuffled_margin_vs_protected"]) < 1e-12,
        aoa["scalar_aoa_accepted_by_current_validator_logic"],
    ])
    status = "PASS_REAL_LOADABLE_LEGAL_ENDPOINT_HYPOTHESIS" if pass_core else "FAIL_ENDPOINT_HYPOTHESIS_STATIC_CHECK"
    route_read = "launch_deterministic_replay_and_score_repeat" if pass_core and score["margin_vs_protected_recomputed"] > 0 else "do_not_replay_until_static_issue_repaired"
    payload = {
        "status": status,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "route_read": route_read,
        "purpose": "Verify the research shuffled 4M frozen-82M private-tail candidate as a real, legal, trusted-loadable endpoint hypothesis before spending GPU on deterministic replay and repeated evaluation.",
        "paths": {
            "shuffled_model": rel(SHUFFLED_MODEL),
            "protected_model": rel(PROTECTED_MODEL),
            "legal_tokenizer": rel(LEGAL_TOKENIZER),
            "score_decision": rel(SCORE_DECISION),
        },
        "tensor_checks": tensor,
        "tokenizer_equivalence": tok_equiv,
        "trusted_load_checks": load,
        "score_and_legal": score,
        "aoa_validator": aoa,
        "expected_constants": EXPECTED,
        "env_cache": {k: rel(Path(v)) for k, v in env.items() if k.startswith("HF") or k in {"TMPDIR"}},
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "route_read": route_read, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
