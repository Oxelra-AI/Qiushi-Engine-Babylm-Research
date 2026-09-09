#!/usr/bin/env python3
"""research: AoA estimator provenance and saved-surprisal rescoring.

Purpose
-------
The research shared-ancestry AoA extraction produces one raw measurement object:
`surprisal.json` with context-level target surprisals at each genuine checkpoint.
The frontier comparison must not accidentally turn a change in AoA scoring
mathematics into a claimed model improvement.  This script therefore separates:

1. The platform-matching participant-side AoA scorer: the current public
   babylm-org/babylm-eval repository at commit 6f825c291e2c4..., whose
   `AoAEvaluator` is the scorer invoked by `strict/evaluation_pipeline/AoA_word/run.py`.
2. Explicit sensitivity estimators that change one scoring choice at a time
   (context averaging, subword-scaled chance ceiling, bounded vs unbounded
   sigmoid fit).  These are not platform scores unless future evidence shows
   that a platform revision used them.

The script consumes saved surprisals and a tokenizer; it repeats no checkpoint
inference and does not modify model/evaluation repositories.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
import typing as t
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
WORKSPACE = _public_path('experiments/archive/functional_learning')
STRICT_REPO = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
UTILS = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/utils.py')
AOA_RUN = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/AoA_word/run.py')
COLLATE = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/collate_preds.py')
CDI_HUMAN = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
LEADERBOARD_DIR = _public_path('experiments/archive/initial_model_studies/data/leaderboard_probe')
LEADERBOARD_READ = _public_path('experiments/archive/initial_model_studies/data/leaderboard_probe/src__leaderboard__read_evals.py')
SUBMISSION_EVAL = _public_path('experiments/archive/initial_model_studies/data/leaderboard_probe/src__submission__eval_submission.py')
SUBMISSION_VALIDITY = _public_path('experiments/archive/initial_model_studies/data/leaderboard_probe/src__submission__check_validity.py')
KNOWLEDGE_META = _public_path('data/external/metadata.json')
KNOWLEDGE_ARCHIVE = _public_path('data/external/babylm-org-babylm-eval-6f825c291e2c4c78ad33b1935fd64d45f52642dc.zip')
DEFAULT_TOKENIZER = _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/aoa_estimator_provenance')

PLATFORM_COMMIT = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str], cwd: pathlib.Path | None = None, timeout: int = 20) -> dict[str, t.Any]:
    try:
        p = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:  # bounded provenance only
        return {"ok": False, "error": repr(e)}


def archive_utils_identity() -> dict[str, t.Any]:
    out: dict[str, t.Any] = {"archive_path": str(KNOWLEDGE_ARCHIVE.relative_to(ROOT)), "archive_exists": KNOWLEDGE_ARCHIVE.exists()}
    local_bytes = UTILS.read_bytes() if UTILS.exists() else b""
    out["local_utils_sha256"] = hashlib.sha256(local_bytes).hexdigest() if local_bytes else None
    out["local_utils_size_bytes"] = len(local_bytes) if local_bytes else None
    if not KNOWLEDGE_ARCHIVE.exists():
        return out
    try:
        with zipfile.ZipFile(KNOWLEDGE_ARCHIVE) as z:
            names = [n for n in z.namelist() if n.endswith("strict/evaluation_pipeline/utils.py")]
            out["archive_utils_names"] = names
            if names:
                b = z.read(names[0])
                out["archive_utils_sha256"] = hashlib.sha256(b).hexdigest()
                out["archive_utils_size_bytes"] = len(b)
                out["archive_equals_local"] = b == local_bytes
    except Exception as e:
        out["archive_error"] = repr(e)
    return out


def line_hits(path: pathlib.Path, patterns: list[str], context: int = 0, max_hits: int = 20) -> list[dict[str, t.Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            start = max(1, i - context)
            end = min(len(lines), i + context)
            hits.append({"line": i, "text": line, "context": "\n".join(f"{j}: {lines[j-1]}" for j in range(start, end + 1))})
            if len(hits) >= max_hits:
                break
    return hits


def platform_provenance(include_remote: bool = False) -> dict[str, t.Any]:
    meta = None
    if KNOWLEDGE_META.exists():
        try:
            meta = json.loads(KNOWLEDGE_META.read_text(encoding="utf-8"))
        except Exception as e:
            meta = {"error": repr(e)}
    local_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=STRICT_REPO)
    status = run_cmd(["git", "status", "--short"], cwd=STRICT_REPO)
    remotes = run_cmd(["git", "remote", "-v"], cwd=STRICT_REPO)
    remote_main = run_cmd(["git", "ls-remote", "https://github.com/babylm-org/babylm-eval.git", "refs/heads/main"], timeout=30) if include_remote else {"skipped": True}

    utils_text = UTILS.read_text(encoding="utf-8", errors="replace") if UTILS.exists() else ""
    eval_text = SUBMISSION_EVAL.read_text(encoding="utf-8", errors="replace") if SUBMISSION_EVAL.exists() else ""
    read_text = LEADERBOARD_READ.read_text(encoding="utf-8", errors="replace") if LEADERBOARD_READ.exists() else ""
    valid_text = SUBMISSION_VALIDITY.read_text(encoding="utf-8", errors="replace") if SUBMISSION_VALIDITY.exists() else ""

    current_features = {
        "model_fit_bounded": "bounds=(lower, upper)" in utils_text and "maxfev=20000" in utils_text,
        "model_fit_mentions_old_unbounded": "old unbounded maxfev=1000" in utils_text,
        "context_average_within_checkpoint": "mean_surprisals" in utils_text and "steps_arr == s" in utils_text,
        "subword_scaled_random_ceiling": "n_subword_tokens * np.log(vocab_size)" in utils_text,
        "p_value_gates_zero": "if p_value > 0.1" in utils_text,
        "extract_step_number_accepts_decimal_units": "(\\d+(?:\\.\\d+)?)" in utils_text,
    }
    leaderboard_features = {
        "space_submission_eval_reads_numeric_aoa_object": "aoa = results_dict.get(\"aoa\")" in eval_text and "float(aoa[\"aoa\"])" in eval_text,
        "space_submission_eval_carries_aoa_surprisals": "'aoa_surprisals'" in eval_text,
        "space_read_evals_zeroes_if_no_aoa_surprisals": "if not data[\"results\"].get(\"aoa_surprisals\")" in read_text,
        "space_validity_checks_raw_aoa_if_legacy_results_key": "\"results\" in predictions[\"aoa\"]" in valid_text,
        "space_validity_expected_aoa_size_symbol": "AOA_SIZE" in valid_text,
    }

    return {
        "status": "AOA_ESTIMATOR_PLATFORM_PROVENANCE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform_commit_expected": PLATFORM_COMMIT,
        "local_repository": {
            "path": str(STRICT_REPO.relative_to(ROOT)),
            "head": local_head,
            "status_short": status,
            "remotes": remotes,
            "remote_main": remote_main,
        },
        "knowledge_repository_object": {
            "metadata_path": str(KNOWLEDGE_META.relative_to(ROOT)) if KNOWLEDGE_META.exists() else None,
            "metadata_version": meta.get("object_version") if isinstance(meta, dict) else None,
            "metadata_commit_sha": (meta.get("metadata", {}) or {}).get("commit_sha") if isinstance(meta, dict) else None,
            "metadata_source_updated_at": meta.get("source_updated_at") if isinstance(meta, dict) else None,
            "metadata_official_source": meta.get("official_source") if isinstance(meta, dict) else None,
        },
        "file_identities": {
            "utils_py": {"path": str(UTILS.relative_to(ROOT)), "sha256": sha256_file(UTILS), "size_bytes": UTILS.stat().st_size if UTILS.exists() else None},
            "aoa_run_py": {"path": str(AOA_RUN.relative_to(ROOT)), "sha256": sha256_file(AOA_RUN), "size_bytes": AOA_RUN.stat().st_size if AOA_RUN.exists() else None},
            "collate_preds_py": {"path": str(COLLATE.relative_to(ROOT)), "sha256": sha256_file(COLLATE), "size_bytes": COLLATE.stat().st_size if COLLATE.exists() else None},
            "leaderboard_submission_eval_py": {"path": str(SUBMISSION_EVAL.relative_to(ROOT)) if SUBMISSION_EVAL.exists() else None, "sha256": sha256_file(SUBMISSION_EVAL)},
            "leaderboard_read_evals_py": {"path": str(LEADERBOARD_READ.relative_to(ROOT)) if LEADERBOARD_READ.exists() else None, "sha256": sha256_file(LEADERBOARD_READ)},
            "leaderboard_check_validity_py": {"path": str(SUBMISSION_VALIDITY.relative_to(ROOT)) if SUBMISSION_VALIDITY.exists() else None, "sha256": sha256_file(SUBMISSION_VALIDITY)},
        },
        "archive_identity": archive_utils_identity(),
        "current_official_main_aoa_features": current_features,
        "space_side_submission_features_from_initial_model_studies_snapshot": leaderboard_features,
        "source_line_evidence": {
            "utils_features": line_hits(UTILS, ["old unbounded", "n_subword_tokens * np.log", "mean_surprisals", "p_value > 0.1", "maxfev=20000", "extract_step_number"], context=1, max_hits=16),
            "aoa_run_scores_with_aoaevaluator": line_hits(AOA_RUN, ["AoAEvaluator", "aoa_score.json", "AutoTokenizer.from_pretrained"], context=2, max_hits=8),
            "collate_loads_aoa_score_and_surprisals": line_hits(COLLATE, ["aoa_surprisals", "aoa_score.json", "AOA_SIZE"], context=1, max_hits=12),
            "space_reads_numeric_aoa": line_hits(SUBMISSION_EVAL, ["aoa = results_dict.get", "aoa_surprisals", "float(aoa"], context=2, max_hits=10),
            "space_display_requires_aoa_surprisals": line_hits(LEADERBOARD_READ, ["aoa_surprisals", "results[\"aoa\"] = 0.0"], context=2, max_hits=8),
        },
        "interpretation": {
            "platform_matching_scorer": "current_official_main_6f825c2_AoAEvaluator",
            "why": "The local working repository is at the public babylm-org/babylm-eval main commit, the Knowledge archive for that commit has byte-identical utils.py, and the current AoA run.py writes aoa_score.json using AoAEvaluator. The saved Space-side code reads the submitted numeric aoa object and requires aoa_surprisals presence for display rather than recomputing the curve server-side.",
            "frontier_use": "Use current_official_main_6f825c2 scores for the BabyLM platform-style frontier coordinate unless a newer platform revision is explicitly identified. Do not mix scores from different estimator variants.",
            "scientific_use": "Use sensitivity estimators only as attributed developmental analyses of fitting/aggregation/baseline choices; if they differ from platform scores, report both rather than crediting estimator mathematics as learning progress.",
        },
    }


# ---- AoA scoring variants -------------------------------------------------


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


class VariantAoAScorer:
    def __init__(self, cdi_data_path: pathlib.Path):
        self.cdi_data = pd.read_csv(cdi_data_path)
        age_columns = [str(age) for age in range(16, 31)]
        available_age_columns = [c for c in age_columns if c in self.cdi_data.columns]
        if not available_age_columns:
            raise ValueError("No age columns found in CDI data")
        self.ages = np.array([int(c) for c in available_age_columns])
        self.age_data = self.cdi_data[available_age_columns].values
        self.words = self.cdi_data["word"].values

    def compute_child_aoa(self, word_idx: int, threshold: float = 0.5) -> float | None:
        proportions = self.age_data[word_idx]
        valid_mask = ~np.isnan(proportions)
        if not np.any(valid_mask):
            return None
        valid_ages = self.ages[valid_mask]
        valid_proportions = proportions[valid_mask]
        if len(valid_proportions) < 3:
            return None
        try:
            initial_guess = [1.0, 0.1, float(np.mean(valid_ages)), 0.0]
            bounds = ([0, 0, valid_ages[0], -0.5], [2.0, 1.0, valid_ages[-1], 0.5])
            popt, _ = curve_fit(sigmoid_function, valid_ages, valid_proportions, p0=initial_guess, bounds=bounds, maxfev=1000)
            a, b, c, d = popt
            if b <= 0 or a <= 0:
                return None
            if threshold <= d or threshold >= a + d:
                return None
            aoa = c - np.log((a / (threshold - d)) - 1) / b
            if aoa < valid_ages[0] or aoa > valid_ages[-1]:
                return None
            return float(aoa)
        except Exception:
            return None

    @staticmethod
    def extract_step_number(step_name: t.Any) -> float | None:
        if isinstance(step_name, (int, float)) and not isinstance(step_name, bool):
            return float(step_name)
        match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
        if not match:
            return None
        number = float(match.group(1))
        unit = match.group(2).upper() if match.group(2) else ""
        return number * {"K": 1000, "M": 1000000, "B": 1000000000}.get(unit, 1)

    @staticmethod
    def compute_model_aoa(
        surprisal_data: list[float],
        training_steps: list[float],
        vocab_size: int,
        n_subword_tokens: int = 1,
        subword_scaled_baseline: bool = True,
        bounded_fit: bool = True,
    ) -> float | None:
        if len(surprisal_data) != len(training_steps):
            raise ValueError("Surprisal data and training steps must have same length")
        if len(surprisal_data) < 3:
            return None
        steps = np.array(training_steps, dtype=float)
        surprisals = np.array(surprisal_data, dtype=float)
        valid_mask = ~np.isnan(surprisals)
        if not np.any(valid_mask):
            return None
        valid_steps = steps[valid_mask]
        valid_surprisals = surprisals[valid_mask]
        ceiling_tokens = n_subword_tokens if subword_scaled_baseline else 1
        random_chance_surprisal = ceiling_tokens * np.log(vocab_size)
        min_surprisal = np.min(valid_surprisals)
        threshold_surprisal = random_chance_surprisal - 0.5 * (random_chance_surprisal - min_surprisal)
        try:
            neg_surprisals = -valid_surprisals
            log_steps = np.log10(valid_steps + 1)
            rng = float(np.max(neg_surprisals) - np.min(neg_surprisals))
            rng_for_fit = rng if rng > 0 else 1e-8
            p0 = [rng_for_fit, 1.0, float(np.mean(log_steps)), float(np.min(neg_surprisals))]
            if bounded_fit:
                lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg_surprisals) - 2 * rng_for_fit - 1)]
                upper = [10 * rng_for_fit + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg_surprisals) + 1)]
                popt, _ = curve_fit(sigmoid_function, log_steps, neg_surprisals, p0=p0, bounds=(lower, upper), maxfev=20000)
            else:
                popt, _ = curve_fit(sigmoid_function, log_steps, neg_surprisals, p0=p0, maxfev=1000)
            a, b, c, d = popt
            neg_threshold = -threshold_surprisal
            if b <= 1e-6 or a <= 1e-6:
                return None
            if neg_threshold <= d or neg_threshold >= a + d:
                return None
            log_aoa_step = c - np.log((a / (neg_threshold - d)) - 1) / b
            aoa_step = 10**log_aoa_step - 1
            if aoa_step < np.min(valid_steps) or aoa_step > np.max(valid_steps):
                return None
            return float(log_aoa_step)
        except Exception:
            return None

    def compute_curve_fitness(
        self,
        model_results: dict[str, t.Any],
        tokenizer: t.Any,
        *,
        aggregate_contexts: bool = True,
        subword_scaled_baseline: bool = True,
        bounded_fit: bool = True,
        target_words: list[str] | None = None,
    ) -> dict[str, t.Any]:
        results = model_results.get("results", [])
        if not results:
            raise ValueError("No results found in model data")
        vocab_size = int(tokenizer.vocab_size)
        prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]

        def subword_len(word: str) -> int:
            ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
            return max(1, len(ids) - len(prefix_ids))

        word_data: dict[str, dict[str, list[float]]] = {}
        unparsable_steps = 0
        bad_surprisals = 0
        allowed = set(target_words) if target_words else None
        for result in results:
            word = result.get("target_word")
            if allowed is not None and word not in allowed:
                continue
            step_val = self.extract_step_number(result.get("step"))
            if step_val is None:
                unparsable_steps += 1
                continue
            try:
                surprisal_val = float(result.get("surprisal"))
            except Exception:
                bad_surprisals += 1
                continue
            word_data.setdefault(str(word), {"steps": [], "surprisals": []})
            word_data[str(word)]["steps"].append(float(step_val))
            word_data[str(word)]["surprisals"].append(surprisal_val)

        model_aoas: list[float] = []
        child_aoas: list[float] = []
        valid_words: list[str] = []
        invalid_reasons = collections.Counter()
        for word, wd in word_data.items():
            word_mask = self.cdi_data["word"] == word
            if not np.any(word_mask):
                invalid_reasons["not_in_cdi_human"] += 1
                continue
            word_idx = int(np.where(word_mask)[0][0])
            child_aoa = self.compute_child_aoa(word_idx)
            if child_aoa is None:
                invalid_reasons["child_aoa_none"] += 1
                continue
            steps_arr = np.array(wd["steps"], dtype=float)
            surp_arr = np.array(wd["surprisals"], dtype=float)
            if aggregate_contexts:
                uniq_steps = np.unique(steps_arr)
                fit_steps = uniq_steps.tolist()
                fit_surps = [float(surp_arr[steps_arr == s].mean()) for s in uniq_steps]
            else:
                # Sensitivity only: keep context-level points as repeated x values.
                order = np.argsort(steps_arr)
                fit_steps = steps_arr[order].tolist()
                fit_surps = surp_arr[order].tolist()
            model_aoa = self.compute_model_aoa(
                fit_surps,
                fit_steps,
                vocab_size,
                n_subword_tokens=subword_len(word),
                subword_scaled_baseline=subword_scaled_baseline,
                bounded_fit=bounded_fit,
            )
            if model_aoa is None:
                invalid_reasons["model_aoa_none"] += 1
                continue
            model_aoas.append(model_aoa)
            child_aoas.append(float(child_aoa))
            valid_words.append(word)

        if len(model_aoas) < 3:
            return {
                "curve_fitness": 0.0,
                "n_words": len(model_aoas),
                "p_value": None,
                "valid_words": valid_words,
                "invalid_reasons": dict(invalid_reasons),
                "unparsable_steps": unparsable_steps,
                "bad_surprisals": bad_surprisals,
                "model_aoas_preview": model_aoas[:10],
                "child_aoas_preview": child_aoas[:10],
            }
        correlation, p_value = pearsonr(model_aoas, child_aoas)
        score = 0.0 if p_value > 0.1 else float(correlation)
        return {
            "curve_fitness": score,
            "raw_correlation": float(correlation),
            "p_value": float(p_value),
            "n_words": len(valid_words),
            "valid_words": valid_words,
            "valid_words_preview": valid_words[:20],
            "invalid_reasons": dict(invalid_reasons),
            "unparsable_steps": unparsable_steps,
            "bad_surprisals": bad_surprisals,
            "model_aoas_preview": [float(x) for x in model_aoas[:10]],
            "child_aoas_preview": [float(x) for x in child_aoas[:10]],
        }


VARIANTS: dict[str, dict[str, t.Any]] = {
    "current_official_main_6f825c2": {
        "frontier_platform_candidate": True,
        "aggregate_contexts": True,
        "subword_scaled_baseline": True,
        "bounded_fit": True,
        "meaning": "Current babylm-org/babylm-eval main AoAEvaluator: context means per checkpoint, subword-scaled chance ceiling, bounded model sigmoid fit with maxfev=20000, p-gated Pearson correlation.",
    },
    "sensitivity_no_context_mean": {
        "frontier_platform_candidate": False,
        "aggregate_contexts": False,
        "subword_scaled_baseline": True,
        "bounded_fit": True,
        "meaning": "Sensitivity only: fit the context-level cloud directly instead of averaging contexts within each checkpoint.",
    },
    "sensitivity_unscaled_random_ceiling": {
        "frontier_platform_candidate": False,
        "aggregate_contexts": True,
        "subword_scaled_baseline": False,
        "bounded_fit": True,
        "meaning": "Sensitivity only: use ln(vocab) chance ceiling for every word rather than scaling by target subword length.",
    },
    "sensitivity_unbounded_fit": {
        "frontier_platform_candidate": False,
        "aggregate_contexts": True,
        "subword_scaled_baseline": True,
        "bounded_fit": False,
        "meaning": "Sensitivity only: use an unbounded model sigmoid fit with maxfev=1000; this corresponds to the behavior referenced in the current source comments as old/unbounded, but is not identified here as a platform revision.",
    },
    "sensitivity_combined_legacy_like": {
        "frontier_platform_candidate": False,
        "aggregate_contexts": False,
        "subword_scaled_baseline": False,
        "bounded_fit": False,
        "meaning": "Sensitivity only: combines no context averaging, unscaled chance ceiling, and unbounded model fit. It is a stress comparison, not a verified historical platform scorer.",
    },
}


def summarize_surprisal(data: dict[str, t.Any]) -> dict[str, t.Any]:
    results = data.get("results", [])
    steps = collections.Counter()
    words = set()
    finite = 0
    for r in results:
        steps[str(r.get("step"))] += 1
        words.add(str(r.get("target_word")))
        try:
            v = float(r.get("surprisal"))
            if math.isfinite(v):
                finite += 1
        except Exception:
            pass
    values = list(steps.values())
    return {
        "n_rows": len(results),
        "n_finite_surprisals": finite,
        "n_steps": len(steps),
        "n_words": len(words),
        "row_count_values": sorted(set(values)),
        "steps_preview": list(steps.items())[:20],
        "metadata": data.get("metadata", {}),
    }


def load_official_import_score(data: dict[str, t.Any], tokenizer: t.Any) -> dict[str, t.Any]:
    old_path = list(sys.path)
    sys.path.insert(0, str(STRICT))
    try:
        from evaluation_pipeline.utils import AoAEvaluator as OfficialAoAEvaluator  # type: ignore
        score = OfficialAoAEvaluator(CDI_HUMAN).compute_curve_fitness(data, tokenizer=tokenizer)
        return {"ok": True, "score": score}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    finally:
        sys.path[:] = old_path


def rescore_surprisal(surprisal_path: pathlib.Path, tokenizer_path: pathlib.Path, out_dir: pathlib.Path, label: str, variants: list[str]) -> dict[str, t.Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(surprisal_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    scorer = VariantAoAScorer(CDI_HUMAN)
    provenance = platform_provenance(include_remote=False)
    summaries: dict[str, t.Any] = {}
    t0 = time.time()
    for name in variants:
        spec = VARIANTS[name]
        v0 = time.time()
        score = scorer.compute_curve_fitness(
            data,
            tokenizer,
            aggregate_contexts=bool(spec["aggregate_contexts"]),
            subword_scaled_baseline=bool(spec["subword_scaled_baseline"]),
            bounded_fit=bool(spec["bounded_fit"]),
        )
        summaries[name] = {
            "estimator_name": name,
            "spec": spec,
            "score": score,
            "aoa_leaderboard_score": float(score.get("curve_fitness", 0.0)) * 100.0,
            "elapsed_sec": round(time.time() - v0, 3),
        }

    official_import = load_official_import_score(data, tokenizer)
    official_custom = summaries.get("current_official_main_6f825c2", {}).get("score")
    official_import_check = {"ok": False}
    if official_import.get("ok") and official_custom:
        imported_curve = float(official_import["score"].get("curve_fitness", 0.0))
        custom_curve = float(official_custom.get("curve_fitness", 0.0))
        official_import_check = {
            "ok": abs(imported_curve - custom_curve) < 1e-12,
            "imported_curve_fitness": imported_curve,
            "custom_curve_fitness": custom_curve,
            "abs_diff": abs(imported_curve - custom_curve),
            "imported_summary_keys": sorted(official_import["score"].keys()),
        }
    else:
        official_import_check = {"ok": False, "official_import": official_import}

    result = {
        "status": "AOA_SAVED_SURPRISAL_RESCORE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "surprisal_path": str(surprisal_path.relative_to(ROOT) if surprisal_path.is_relative_to(ROOT) else surprisal_path),
        "surprisal_sha256": sha256_file(surprisal_path),
        "tokenizer_path": str(tokenizer_path.relative_to(ROOT) if tokenizer_path.is_relative_to(ROOT) else tokenizer_path),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "cdi_human_path": str(CDI_HUMAN.relative_to(ROOT)),
        "cdi_human_sha256": sha256_file(CDI_HUMAN),
        "surprisal_summary": summarize_surprisal(data),
        "platform_provenance_summary": {
            "platform_commit_expected": provenance["platform_commit_expected"],
            "local_head": provenance["local_repository"]["head"].get("stdout"),
            "utils_sha256": provenance["file_identities"]["utils_py"].get("sha256"),
            "archive_equals_local": provenance["archive_identity"].get("archive_equals_local"),
            "current_official_features": provenance["current_official_main_aoa_features"],
            "space_side_features": provenance["space_side_submission_features_from_initial_model_studies_snapshot"],
        },
        "official_import_check": official_import_check,
        "variant_scores": summaries,
        "interpretation": {
            "frontier_coordinate": "Use `current_official_main_6f825c2` for platform-matching BabyLM comparison unless a newer platform scorer is explicitly identified.",
            "score_units": "curve_fitness is raw Pearson r after the p-value rule; aoa_leaderboard_score is 100*curve_fitness for leaderboard column units.",
            "sensitivity_use": "Non-platform variants show whether fitting/aggregation/baseline choices alone can change AoA. They must not be credited as model learning improvements.",
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / f"{label}_aoa_estimator_rescore.json"
    out_md = out_dir / f"{label}_aoa_estimator_rescore.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# research AoA estimator rescore: {label}",
        "",
        f"Surprisal: `{result['surprisal_path']}`",
        f"Tokenizer: `{result['tokenizer_path']}` (vocab {result['tokenizer_vocab_size']})",
        f"Rows/steps/words: `{result['surprisal_summary']['n_rows']}` / `{result['surprisal_summary']['n_steps']}` / `{result['surprisal_summary']['n_words']}`; row counts per step `{result['surprisal_summary']['row_count_values']}`",
        "",
        "## Estimator provenance",
        "",
        f"- Current public BabyLM evaluation commit expected: `{PLATFORM_COMMIT}`; local head: `{result['platform_provenance_summary']['local_head']}`.",
        f"- `utils.py` SHA256: `{result['platform_provenance_summary']['utils_sha256']}`; Knowledge archive byte-equal to local: `{result['platform_provenance_summary']['archive_equals_local']}`.",
        f"- Official import check for current estimator: `{official_import_check.get('ok')}`; diff `{official_import_check.get('abs_diff')}`.",
        "- The saved Space-side submission code reads the numeric `aoa` object and carries `aoa_surprisals`; display code zeroes AoA when `aoa_surprisals` is absent. Thus estimator identity belongs to the participant-side evaluation commit that produced `aoa_score.json`.",
        "",
        "## Scores from the same surprisal measurements",
        "",
        "| Estimator | Platform coordinate? | curve_fitness | leaderboard units | n fitted words | p value | raw corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summaries.items():
        sc = item["score"]
        lines.append(
            f"| `{name}` | `{item['spec']['frontier_platform_candidate']}` | {float(sc.get('curve_fitness', 0.0)):.12g} | {float(item.get('aoa_leaderboard_score', 0.0)):.12g} | {sc.get('n_words')} | {sc.get('p_value')} | {sc.get('raw_correlation')} |"
        )
    lines.extend([
        "",
        "Non-platform rows are estimator-sensitivity analyses. They are useful for developmental interpretation but must not be mixed into the BabyLM frontier coordinate.",
        "",
        f"Full JSON: `{out_json.relative_to(ROOT)}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "label": label, "out_json": str(out_json.relative_to(ROOT)), "official_import_check": official_import_check, "scores": {k: v["score"].get("curve_fitness") for k, v in summaries.items()}}, indent=2, ensure_ascii=False))
    return result


def write_provenance(out_dir: pathlib.Path, include_remote: bool = False) -> dict[str, t.Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = platform_provenance(include_remote=include_remote)
    out_json = out_dir / "aoa_estimator_platform_provenance.json"
    out_md = out_dir / "aoa_estimator_platform_provenance.md"
    out_json.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    feat = p["current_official_main_aoa_features"]
    space = p["space_side_submission_features_from_initial_model_studies_snapshot"]
    lines = [
        "# research AoA estimator platform provenance",
        "",
        f"Created: {p['created_utc']}",
        "",
        "## Repository identity",
        "",
        f"- Local repo: `{p['local_repository']['path']}`",
        f"- Local HEAD: `{p['local_repository']['head'].get('stdout')}`",
        f"- Expected public main commit: `{p['platform_commit_expected']}`",
        f"- Remote main probe: `{p['local_repository']['remote_main'].get('stdout', p['local_repository']['remote_main'])}`",
        f"- Knowledge archive byte-equal for `utils.py`: `{p['archive_identity'].get('archive_equals_local')}`",
        f"- `utils.py` SHA256: `{p['file_identities']['utils_py']['sha256']}`",
        "",
        "## Current official-main AoA scorer features",
        "",
    ]
    for k, v in feat.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(["", "## Space-side scoring snapshot", ""])
    for k, v in space.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        p["interpretation"]["why"],
        "",
        "For the frontier comparison, the AoA column should be attributed to `current_official_main_6f825c2_AoAEvaluator` unless a newer platform scorer is explicitly identified. The same saved `surprisal.json` can be rescored under sensitivity estimators for scientific analysis, but those results must remain separate from the platform-matching coordinate.",
        "",
        f"Full JSON: `{out_json.relative_to(ROOT)}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": p["status"], "out_json": str(out_json.relative_to(ROOT)), "local_head": p["local_repository"]["head"].get("stdout"), "archive_equals_local": p["archive_identity"].get("archive_equals_local")}, indent=2))
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surprisal", type=pathlib.Path, default=None, help="Saved AoA_word/surprisal.json to rescore. If omitted, only writes provenance.")
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER, help="Tokenizer/model path used only for vocab size and subword lengths.")
    ap.add_argument("--label", default=None, help="Output label; defaults to surprisal parent name.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--include-remote", action="store_true", help="Probe GitHub main in the provenance record.")
    ap.add_argument("--variants", default="all", choices=["all", "official", "sensitivity"], help="Which estimator variants to run when --surprisal is given.")
    ap.add_argument("--variant-names", default=None, help="Optional comma-separated exact estimator names; overrides --variants. Useful for targeted sensitivity without slow context-level fits.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.surprisal is None:
        write_provenance(args.out_dir, include_remote=args.include_remote)
        return
    if args.variant_names:
        variants = [x.strip() for x in args.variant_names.split(",") if x.strip()]
        unknown = [x for x in variants if x not in VARIANTS]
        if unknown:
            raise SystemExit(f"Unknown --variant-names entries: {unknown}; known={list(VARIANTS)}")
    elif args.variants == "official":
        variants = ["current_official_main_6f825c2"]
    elif args.variants == "sensitivity":
        variants = [k for k in VARIANTS if k != "current_official_main_6f825c2"]
    else:
        variants = list(VARIANTS)
    label = args.label or args.surprisal.parent.parent.name
    rescore_surprisal(args.surprisal, args.tokenizer, args.out_dir, label, variants)


if __name__ == "__main__":
    main()
