#!/usr/bin/env python3
"""research: matched true-correspondence test for the coupled dual-view hard-surface signal.

Scientific question
--------------------
The `coupled_sparse20_aligned_20M` experiment was the first tested intervention to
strongly move the load-bearing context-conditioned alternative-binding
surface: on the fixed research EWoK stable-reversal subset it cuts stable failures
922 -> 430 (accuracy 0.26581 -> 0.51258) and moves the fixed research GlobalPIQA
hard52 slice 3.85% -> 9.62%. But that repair could come from true source-rewrite
CORRESPONDENCE, or merely from the coupled auxiliary perturbation of the training
trajectory. The decisive control is a matched `coupled_sparse20_shuffled_20M` run,
identical to the aligned run except that the source-conditioned auxiliary view uses a
batch-level derangement of the same source texts, so the source multiset and charged
exposure are identical per batch and only true correspondence differs.

This script reuses the research posthoc hard-surface machinery verbatim (same fixed
research GlobalPIQA hard52 all-option length-normalized reader and same fixed research
EWoK stable-failure subset), extends the target set with the new coupled shuffled
control, and computes the decisive delta `coupled_aligned_minus_coupled_shuffled`.

No training here; CPU only; existing checkpoints only; fixed hard surfaces only.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout')


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import the research readout module and reuse its readers/summarizers.
S175 = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/detached_private_hard_surface_readout.py'), "readout_for_step177")


# New coupled shuffled control target, matched to the coupled aligned run.
COUPLED_SHUFFLED = {
    "label": "A01 coupled sparse20 SHUFFLED-correspondence matched control at 20M (research)",
    "model_path": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final'),
    "metrics_path": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/scientific_metrics.json'),
    "eval_payload_path": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/does_not_exist.json'),
    "family": "coupled_shuffled_alignment",
}


def build_step177_deltas(globalpiqa: dict[str, Any], ewok: dict[str, Any]) -> dict[str, Any]:
    """The decisive pairs for the coupled correspondence test."""
    pairs = [
        # Decisive: does true correspondence uniquely produce the hard-surface repair?
        ("coupled_aligned_minus_coupled_shuffled", "coupled_sparse20_shuffled_20M", "coupled_sparse20_aligned_20M"),
        # Context: each coupled arm vs the exact MLM-only payload baseline.
        ("coupled_aligned_minus_mlm_only", "mlm_only_20M", "coupled_sparse20_aligned_20M"),
        ("coupled_shuffled_minus_mlm_only", "mlm_only_20M", "coupled_sparse20_shuffled_20M"),
    ]
    out: dict[str, Any] = {"globalpiqa": {}, "ewok_stable_subset": {}}
    for label, base, cand in pairs:
        if base in globalpiqa and cand in globalpiqa:
            out["globalpiqa"][label] = S175.gp_delta(globalpiqa[base], globalpiqa[cand])
        if base in ewok and cand in ewok:
            out["ewok_stable_subset"][label] = S175.ewok_delta(ewok[base], ewok[cand])
    return out


def _num(x: Any) -> float | None:
    return x if isinstance(x, (int, float)) else None


def _classify_correspondence(summary: dict[str, Any]) -> dict[str, Any]:
    """Read true-correspondence specificity at each surface's natural resolution.

    the 1,471-row EWoK reversal set is the cleaner interaction
    measure and should be read as a graded effect (accuracy, stable-failure count,
    and the interaction-sum distribution shift), while the 52-row GlobalPIQA set
    should be read through paired ranks and margins as well as discrete accuracy.
    A large aligned-over-shuffled EWoK effect with coherent GlobalPIQA margin/rank
    movement preserves the mechanism even if only a few hard52 choices flip; broad
    movement without alignment-specific hard-row transitions does not.
    """
    dgp = summary.get("deltas", {}).get("globalpiqa", {})
    dew = summary.get("deltas", {}).get("ewok_stable_subset", {})
    ca_cs_gp = dgp.get("coupled_aligned_minus_coupled_shuffled", {}).get("parallel", {}) or {}
    ca_cs_ew = dew.get("coupled_aligned_minus_coupled_shuffled", {}) or {}
    cs_mlm_ew = dew.get("coupled_shuffled_minus_mlm_only", {}) or {}
    ca_mlm_ew = dew.get("coupled_aligned_minus_mlm_only", {}) or {}

    # EWoK primary graded interaction surface (aligned minus shuffled).
    ew_acc = _num(ca_cs_ew.get("accuracy"))
    ew_stable = _num(ca_cs_ew.get("stable_failure"))          # negative = aligned has fewer stable failures
    ew_frac = _num(ca_cs_ew.get("stable_failure_frac_all"))
    ew_int_mean = _num(ca_cs_ew.get("interaction_sum_wrong_mean"))
    ew_int_median = _num(ca_cs_ew.get("interaction_sum_wrong_median"))

    # GlobalPIQA hard52: ranks + margins + discrete accuracy (aligned minus shuffled).
    gp_hard_acc = _num(ca_cs_gp.get("hard52_accuracy"))
    gp_hard_margin = _num(ca_cs_gp.get("hard52_mean_top_minus_correct"))  # negative = correct closer to top
    gp_hard_rank1 = _num(ca_cs_gp.get("hard52_rank1"))                    # positive = more correct-at-rank1
    gp_all_acc = _num(ca_cs_gp.get("accuracy"))
    gp_all_margin = _num(ca_cs_gp.get("all_mean_top_minus_correct"))

    # How much of the coupled hard-surface EWoK repair survives when only true
    # correspondence is removed. Aligned-vs-MLM is the total coupled repair;
    # shuffled-vs-MLM is the correspondence-free part.
    total_repair = _num(ca_mlm_ew.get("stable_failure"))       # negative = repair
    residual_repair = _num(cs_mlm_ew.get("stable_failure"))    # negative = repair even without correspondence
    correspondence_share = None
    if isinstance(total_repair, (int, float)) and total_repair < 0:
        # fraction of the stable-failure reduction attributable to true correspondence
        aligned_specific = (residual_repair - total_repair) if isinstance(residual_repair, (int, float)) else None
        # aligned_specific >=0 means shuffled repaired less than aligned (correspondence adds repair)
        if aligned_specific is not None:
            correspondence_share = aligned_specific / abs(total_repair)

    # Graded EWoK verdict: meaningful aligned-over-shuffled reduction in stable
    # failures and/or a coherent negative interaction shift.
    ewok_alignment_specific = None
    if ew_stable is not None:
        ewok_alignment_specific = bool(ew_stable <= -50 or (isinstance(ew_acc, (int, float)) and ew_acc >= 0.03))
    # GlobalPIQA coherent-margin verdict: aligned pushes correct options toward top
    # (negative margin delta and/or more rank1), even if discrete flips are few.
    gp_coherent_margin = None
    if gp_hard_margin is not None:
        gp_coherent_margin = bool(gp_hard_margin <= -0.05 or (isinstance(gp_hard_rank1, (int, float)) and gp_hard_rank1 >= 2) or (isinstance(gp_hard_acc, (int, float)) and gp_hard_acc >= 0.03))

    return {
        "ewok_aligned_minus_shuffled": {
            "accuracy_delta": ew_acc,
            "stable_failure_count_delta": ew_stable,
            "stable_failure_frac_delta": ew_frac,
            "interaction_sum_wrong_mean_delta": ew_int_mean,
            "interaction_sum_wrong_median_delta": ew_int_median,
            "alignment_specific_effect": ewok_alignment_specific,
        },
        "globalpiqa_hard52_aligned_minus_shuffled": {
            "hard52_accuracy_delta": gp_hard_acc,
            "hard52_mean_margin_delta_nats": gp_hard_margin,
            "hard52_rank1_delta": gp_hard_rank1,
            "parallel_accuracy_delta": gp_all_acc,
            "parallel_mean_margin_delta_nats": gp_all_margin,
            "coherent_margin_movement": gp_coherent_margin,
        },
        "coupled_stable_failure_repair_decomposition": {
            "total_repair_aligned_minus_mlm_stable_delta": total_repair,
            "correspondence_free_repair_shuffled_minus_mlm_stable_delta": residual_repair,
            "fraction_attributable_to_true_correspondence": correspondence_share,
        },
        "verdict_inputs": {
            "ewok_alignment_specific": ewok_alignment_specific,
            "globalpiqa_coherent_margin": gp_coherent_margin,
        },
    }


def interpret_step177(summary: dict[str, Any]) -> list[str]:
    items: list[str] = []
    cls = _classify_correspondence(summary)
    summary["correspondence_classification"] = cls

    ew = cls["ewok_aligned_minus_shuffled"]
    gp = cls["globalpiqa_hard52_aligned_minus_shuffled"]
    dec = cls["coupled_stable_failure_repair_decomposition"]

    items.append(
        "EWoK research stable subset (primary graded interaction surface), coupled aligned minus coupled shuffled: "
        f"accuracy delta {ew['accuracy_delta']}, stable-failure count delta {ew['stable_failure_count_delta']}, "
        f"stable-failure fraction delta {ew['stable_failure_frac_delta']}, "
        f"interaction_sum_wrong mean delta {ew['interaction_sum_wrong_mean_delta']} nats, "
        f"median delta {ew['interaction_sum_wrong_median_delta']} nats."
    )
    items.append(
        "GlobalPIQA hard52 (read through ranks and margins, not only discrete accuracy), coupled aligned minus coupled shuffled: "
        f"hard52 accuracy delta {gp['hard52_accuracy_delta']}, hard52 mean top-minus-correct margin delta {gp['hard52_mean_margin_delta_nats']} nats, "
        f"hard52 rank1 count delta {gp['hard52_rank1_delta']}, whole-set parallel accuracy delta {gp['parallel_accuracy_delta']}, "
        f"whole-set mean margin delta {gp['parallel_mean_margin_delta_nats']} nats."
    )
    items.append(
        "Stable-failure repair decomposition on the fixed EWoK subset: total coupled repair (aligned minus MLM-only) "
        f"stable-failure delta {dec['total_repair_aligned_minus_mlm_stable_delta']}; correspondence-free part "
        f"(shuffled minus MLM-only) stable-failure delta {dec['correspondence_free_repair_shuffled_minus_mlm_stable_delta']}; "
        f"fraction of the repair attributable to TRUE correspondence {dec['fraction_attributable_to_true_correspondence']}."
    )

    ew_spec = cls["verdict_inputs"]["ewok_alignment_specific"]
    gp_spec = cls["verdict_inputs"]["globalpiqa_coherent_margin"]
    if ew_spec is True and (gp_spec is True or gp_spec is None):
        items.append(
            "Reading: the coupled hard-surface repair is at least partly TRUE-CORRESPONDENCE-SPECIFIC. Aligned removes "
            "materially more EWoK stable reversals than the shuffled control at the natural resolution of the 1,471-row "
            "interaction set, and GlobalPIQA margins/ranks move coherently. This preserves the coupled dual-view mechanism "
            "as the strongest open representation-forming route, and the next work is a minimal broad-preserving coupled "
            "variant that keeps this cross-view true-alignment pressure while restoring BLiMP/Supplement/Entity/Reading/COMPS."
        )
    elif ew_spec is False:
        items.append(
            "Reading: the coupled EWoK repair is NOT true-correspondence-specific at 20M. The shuffled control removes "
            "comparably many stable reversals, so the effect is dominated by coupled auxiliary perturbation of the "
            "trajectory rather than by learning true source-rewrite correspondence. The route must be rethought rather "
            "than scaled; do not promote coupled aligned on the basis of the research hard-surface readout alone."
        )
    else:
        items.append(
            "Reading: mixed/graded signal. Judge by whether the aligned-over-shuffled EWoK interaction shift is large and "
            "coherent with GlobalPIQA margin/rank movement; a few discrete hard52 flips alone are not decisive either way."
        )
    items.append(
        "Resolution principle: the 1,471-row EWoK reversal set is the cleaner interaction measure and is read "
        "as a graded effect; the 52-row GlobalPIQA set is read through paired ranks and margins in addition to discrete "
        "accuracy. Broad-score movement without alignment-specific hard-row transitions does not preserve the mechanism."
    )
    items.append(
        "Fixed research GlobalPIQA hard52 and research EWoK stable rows are diagnostics only; they must not be tuned or "
        "used to define a submission-time scoring rule. chck_82M endpoint is untouched by this readout."
    )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="research coupled shuffled matched control readout")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--skip-globalpiqa", action="store_true")
    ap.add_argument("--skip-ewok", action="store_true")
    ap.add_argument("--gp-max-items", type=int, default=0)
    ap.add_argument("--ewok-max-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ewok-row-batch-size", type=int, default=64)
    ap.add_argument("--ewok-masked-batch-size", type=int, default=160)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    # Register the new control target inside the research module namespace so its
    # readers resolve model paths identically.
    S175.TARGETS["coupled_sparse20_shuffled_20M"] = COUPLED_SHUFFLED

    # The four targets that matter for the coupled correspondence test.
    target_names = [
        "mlm_only_20M",
        "coupled_sparse20_aligned_20M",
        "coupled_sparse20_shuffled_20M",
    ]
    selected = {name: S175.TARGETS[name] for name in target_names}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "hf_cache"
    for d in [out_root / "home", out_root / "xdg_cache", cache_root, cache_root / "modules",
              out_root / "torch_cache", out_root / "tmp"]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((out_root / "home").resolve())
    os.environ["XDG_CACHE_HOME"] = str((out_root / "xdg_cache").resolve())
    os.environ["HF_HOME"] = str(cache_root.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root.resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache_root / "modules").resolve())
    os.environ["TORCH_HOME"] = str((out_root / "torch_cache").resolve())
    os.environ["TMPDIR"] = str((out_root / "tmp").resolve())

    target_status: dict[str, Any] = {}
    missing = []
    for name, meta in selected.items():
        mp = Path(meta["model_path"])
        ready = mp.exists() and (mp / "model.safetensors").exists() and (mp / "config.json").exists()
        if not ready:
            missing.append(name)
        target_status[name] = {
            "label": meta["label"],
            "family": meta.get("family"),
            "model_path": S175.rel(mp),
            "ready": ready,
            "training_metrics": S175.read_json(Path(meta["metrics_path"])),
        }
    preflight = {
        "status": "READY" if not missing else "NOT_READY",
        "created_utc": S175.now_utc(),
        "boundary": "posthoc coupled-correspondence hard-surface readout on existing checkpoints; no training; CPU only",
        "targets": target_status,
        "missing_targets": missing,
        "out_root": S175.rel(out_root),
    }
    (out_root / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    if args.ready_only:
        return
    if missing:
        raise FileNotFoundError(json.dumps(preflight, ensure_ascii=False))

    summary: dict[str, Any] = {
        "status": "COUPLED_SHUFFLED_CONTROL_READOUT_DONE",
        "created_utc": S175.now_utc(),
        "boundary": preflight["boundary"],
        "out_root": S175.rel(out_root),
        "summary_json": str(out_root / "coupled_shuffled_control_summary.json"),
        "targets": target_status,
        "globalpiqa_fixed_hard_set": None,
        "ewok_stable_subset_definition": None,
        "globalpiqa": {},
        "ewok_stable_subset": {},
        "deltas": {},
        "interpretation": [],
    }

    gp_max = args.gp_max_items if args.gp_max_items and args.gp_max_items > 0 else None
    ewok_max = args.ewok_max_rows if args.ewok_max_rows and args.ewok_max_rows > 0 else None
    if not args.skip_globalpiqa:
        gp_results, gp_meta = S175.run_globalpiqa(selected, out_root / "globalpiqa", gp_max, args.threads)
        summary["globalpiqa"] = gp_results
        summary["globalpiqa_fixed_hard_set"] = gp_meta
    if not args.skip_ewok:
        ew_results, ew_meta = S175.run_ewok_subset(
            selected, out_root / "ewok_stable_subset", ewok_max, args.threads,
            args.ewok_row_batch_size, args.ewok_masked_batch_size)
        summary["ewok_stable_subset"] = ew_results
        summary["ewok_stable_subset_definition"] = ew_meta
    summary["deltas"] = build_step177_deltas(summary["globalpiqa"], summary["ewok_stable_subset"])
    summary["interpretation"] = interpret_step177(summary)

    out_json = out_root / "coupled_shuffled_control_summary.json"
    summary["summary_json"] = str(out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_json": S175.rel(out_json)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
