#!/usr/bin/env python3
"""research: trusted relation-face scoring for seed43122 base/dose21/dose25.

This is the seed43122 analogue of research's `--preset dose43022` run.  It reuses
the validated compact and Wikipedia source-conditioned probes from
`score_paired_context_relation_design.py`, but binds the arms to the seed43122
compact-view-reinvest base and the now-completed research dose trainings.

Outputs mirror research: compact_TUN_*, wikipedia_*, copy_*, score_meta, identity
preamble and summary.json.  Entity aggregation is skipped because these dose arms
have no official per-target JSON yet.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_seed43122_relation_faces.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive/frontier_consolidation"
SCRIPTS = WS / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Critical: configure writable Hugging Face caches before importing the research
# scorer, because that module imports Transformers and dynamic-module cache paths
# are fixed early.  A previous research launch configured caches only after import
# and failed by trying to write under data/external/modules.
EARLY_CACHE = pathlib.Path(os.environ.get("REL_FACE_CACHE", str(WS / "data/seed43122_trusted_relation_faces/hf_cache")))
if not EARLY_CACHE.is_absolute():
    EARLY_CACHE = ROOT / EARLY_CACHE
for _sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (EARLY_CACHE / _sub).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(EARLY_CACHE / "hf_home")
os.environ["HF_HUB_CACHE"] = str(EARLY_CACHE / "hf_home/hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(EARLY_CACHE / "hf_home/hub")
os.environ["TRANSFORMERS_CACHE"] = str(EARLY_CACHE / "transformers")
os.environ["HF_MODULES_CACHE"] = str(EARLY_CACHE / "modules")
os.environ["HF_DATASETS_CACHE"] = str(EARLY_CACHE / "datasets")
os.environ["TMPDIR"] = str(EARLY_CACHE / "tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import score_paired_context_relation_design as base  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def set_dose43122_preset() -> None:
    base.ARMS = {
        "BASE0": {
            "role": "BASE0",
            "description": "compact-view-reinvest adapter-scaled base seed43122",
            "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/adapter128_scale1p75_seed43122_dense100M",
        },
        "DOSE21": {
            "role": "DOSE21",
            "description": "probe-clean nested aligned-restatement dose21 seed43122",
            "run": WS / "training/runs/probe_clean_restatement_dose21_seed43122",
        },
        "DOSE25": {
            "role": "DOSE25",
            "description": "probe-clean nested aligned-restatement dose25 seed43122",
            "run": WS / "training/runs/probe_clean_restatement_dose25_seed43122",
        },
    }
    base.CONTRASTS = [
        ("DOSE21minusBASE0", "DOSE21", "BASE0"),
        ("DOSE25minusBASE0", "DOSE25", "BASE0"),
        ("DOSE25minusDOSE21", "DOSE25", "DOSE21"),
    ]


def pick_compact(compact_con, token_class: str, contrast: str) -> dict[str, Any] | None:
    sub = compact_con[(compact_con.token_class == token_class) & (compact_con.contrast == contrast)]
    if len(sub) != 1:
        return None
    r = sub.iloc[0]
    return {
        "token_class": token_class,
        "contrast": contrast,
        "delta_A_T": float(r.delta_A_T),
        "delta_A_U": float(r.delta_A_U),
        "delta_G": float(r.delta_G),
    }


def pick_wiki(wiki_con, token_class: str, overlap_bin: str, contrast: str, estimand: str) -> dict[str, Any] | None:
    sub = wiki_con[(wiki_con.token_class == token_class) & (wiki_con.overlap_bin == overlap_bin) & (wiki_con.contrast == contrast) & (wiki_con.estimand == estimand)]
    if len(sub) != 1:
        return None
    r = sub.iloc[0]
    return {
        "overlap_bin": overlap_bin,
        "token_class": token_class,
        "contrast": contrast,
        "estimand": estimand,
        "mean_difference": float(r.mean_difference),
        "se_pair_difference": float(r.se_pair_difference),
        "n_pairs": int(r.n_pairs),
        "fraction_positive": float(r.fraction_positive),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=["chck_80M", "chck_90M", "chck_100M"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--copy-n-per-span", type=int, default=1000)
    ap.add_argument("--copy-span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--copy-scan-rows", type=int, default=6992)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0)
    ap.add_argument("--rewrite-tokens-per-class", type=int, default=2)
    ap.add_argument("--max-len-copy-rewrite", type=int, default=256)
    ap.add_argument("--wiki-max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=930030)
    ap.add_argument("--out-dir", type=pathlib.Path, default=WS / "data/seed43122_trusted_relation_faces")
    ap.add_argument("--hf-cache-dir", type=pathlib.Path, default=WS / "data/seed43122_trusted_relation_faces/hf_cache")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    set_dose43122_preset()
    base.CKPTS = list(args.checkpoints)
    base.configure_hf_cache(args.hf_cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(args.arms) if args.arms else list(base.ARMS.keys())

    missing = []
    for role in arms:
        for ck in base.CKPTS:
            mp = base.model_path(role, ck)
            if not mp.exists() or not (mp / "config.json").exists():
                missing.append(rel(mp))
    plan = {
        "status": "SEED43122_RELATION_FACE_PLAN",
        "created_utc": now(),
        "arms": {k: {"description": base.ARMS[k]["description"], "run": rel(base.ARMS[k]["run"])} for k in arms},
        "checkpoints": base.CKPTS,
        "missing": missing,
        "out_dir": rel(args.out_dir),
        "trust_remote_code": True,
        "local_files_only": True,
        "hf_cache_dir": rel(args.hf_cache_dir),
    }
    (args.out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if missing:
        raise FileNotFoundError(missing)
    if args.plan_only:
        return

    # Build records with the seed43122/base tokenizer; it is the same family as the dose arms.
    tokenizer = base.AutoTokenizer.from_pretrained(str(base.model_path(arms[0], base.CKPTS[-1]).parent), use_fast=True)
    records, rec_stats = base.build_probe_records(tokenizer, args)
    plan["records"] = rec_stats
    plan["total_records_per_model"] = len(records)
    (args.out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    device = base.torch.device(args.device if args.device == "cpu" or base.torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    scored_all: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for role in arms:
        for ck in base.CKPTS:
            t0 = time.time()
            mp = base.model_path(role, ck)
            print(f"[LOAD] {role} {ck} {mp}", flush=True)
            # Minimal namespace object matching base.load_masked_lm_for_scoring args.
            load_args = argparse.Namespace(trust_remote_code=True, local_files_only=True)
            model, ident = base.load_masked_lm_for_scoring(mp, load_args, device)
            ident.update({"role": role, "checkpoint": ck})
            identities.append(ident)
            print(json.dumps({"event": "model_identity", **ident}, ensure_ascii=False), flush=True)
            if int(ident.get("adapter_params_loaded", 0)) != 995584:
                raise RuntimeError(f"Expected 995584 adapter params for {role} {ck}, got {ident}")
            rows = base.score_records(model, records, device, pad_id, args.batch_size)
            for r in rows:
                r["role"] = role; r["arm"] = role; r["checkpoint"] = ck; r["seed"] = 43122
            scored_all.extend(rows)
            elapsed = time.time() - t0
            meta.append({"role": role, "checkpoint": ck, "records": len(rows), "elapsed_sec": round(elapsed, 2), "device": str(device), "loaded_class": ident.get("loaded_class"), "total_params_loaded": ident.get("total_params_loaded"), "adapter_params_loaded": ident.get("adapter_params_loaded"), "adapter_scale_config": ident.get("adapter_scale_config")})
            del model
            if device.type == "cuda":
                base.torch.cuda.empty_cache()
            print(f"[DONE] {role} {ck}: {len(rows)} records in {elapsed:.1f}s", flush=True)

    base.write_csv(args.out_dir / "score_meta.csv", meta)
    base.write_csv(args.out_dir / "model_identity_preamble.csv", identities)
    with (args.out_dir / "model_identity_preamble.jsonl").open("w", encoding="utf-8") as f:
        for ident in identities:
            f.write(json.dumps(ident, ensure_ascii=False) + "\n")

    dfs = base.aggregate_all(scored_all, args.out_dir)
    _compact_rows, compact_late, compact_con = base.integrate_compact(dfs, args.out_dir)
    copy_late, copy_con = base.integrate_copy(dfs, args.out_dir)
    _wiki_target, wiki_late, wiki_con = base.integrate_wikipedia(dfs, args.out_dir)

    key_compact = [x for x in [
        pick_compact(compact_con, "nonoverlap", "DOSE21minusBASE0"),
        pick_compact(compact_con, "overlap", "DOSE21minusBASE0"),
        pick_compact(compact_con, "nonoverlap", "DOSE25minusBASE0"),
        pick_compact(compact_con, "overlap", "DOSE25minusBASE0"),
        pick_compact(compact_con, "nonoverlap", "DOSE25minusDOSE21"),
        pick_compact(compact_con, "overlap", "DOSE25minusDOSE21"),
    ] if x is not None]
    key_wiki = []
    for con in ["DOSE21minusBASE0", "DOSE25minusBASE0", "DOSE25minusDOSE21"]:
        for token_class in ["overlap", "nonoverlap", "ALL"]:
            for est in ["gain_T_vs_N", "gain_T_vs_U", "gain_U_vs_N"]:
                item = pick_wiki(wiki_con, token_class, "ALL", con, est)
                if item is not None:
                    key_wiki.append(item)

    summary = {
        "status": "SEED43122_RELATION_FACE_DONE",
        "created_utc": now(),
        "out_dir": rel(args.out_dir),
        "records_per_model": len(records),
        "score_rows": len(scored_all),
        "arms": arms,
        "checkpoints": base.CKPTS,
        "model_identity_preamble": rel(args.out_dir / "model_identity_preamble.jsonl"),
        "key_compact_contrasts": key_compact,
        "key_wikipedia_contrasts": key_wiki,
        "outputs": {
            "compact_contrasts": rel(args.out_dir / "compact_TUN_late_contrasts.csv"),
            "wikipedia_contrasts": rel(args.out_dir / "wikipedia_late_contrasts.csv"),
            "copy_contrasts": rel(args.out_dir / "copy_late_contrasts.csv"),
            "score_meta": rel(args.out_dir / "score_meta.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
