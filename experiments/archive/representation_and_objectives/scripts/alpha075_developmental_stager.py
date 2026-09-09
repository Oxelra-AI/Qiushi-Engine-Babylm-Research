#!/usr/bin/env python3
"""research: stage the truthful developmental checkpoint lineage for coherent86 alpha=0.75.

This script is deliberately CPU/file-system only.  It does not evaluate or train.
It encodes the evaluation constraint for the alpha0.75 submission-tail branch:

* revisions before the private intervention use the true ancestral slow trajectory;
* 83--85M use the real coherent-tail states re-materialized at alpha=0.75;
* 86M uses the exact alpha0.75 candidate final model;
* missing 90M/100M states are reported as missing and never faked.

The output is a staged `hf_model/` tree plus a manifest that later AoA/fast pilots
can read.  Existing checkpoints are symlinked except scale-materialized private-tail
states, whose configs must be changed from alpha=1.0 to alpha=0.75 while the exact
weights are preserved.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
A01_WORK = A01
A02_WORK = A02
SLOW_LADDER = A02_WORK / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
COHERENT_TAIL = A02_WORK / "training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model"
ALPHA075_FINAL = A02_WORK / "training/runs/coherent86_private_scale_0p75/hf_model/final"
ALPHA075_CARRIER_MANIFEST = A02_WORK / "data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json"
DEFAULT_OUT = A01_WORK / "data/alpha075_developmental_staging"
FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
TAIL_TOTAL_MAP = {
    "chck_83M": "chck_total_83012495w",
    "chck_84M": "chck_total_84012495w",
    "chck_85M": "chck_total_85012495w",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    """Root-relative display path without resolving final symlinks.

    For staged checkpoint aliases, resolving the path would collapse the staged
    destination back onto the source and erase the provenance of the staging tree.
    """
    p = Path(path)
    try:
        q = p if p.is_absolute() else USER_ROOT / p
        return str(q.absolute().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remove_existing(dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()


def symlink_dir(src: Path, dst: Path, force: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if not (src / "model.safetensors").exists():
        raise FileNotFoundError(src / "model.safetensors")
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(dst)
        remove_existing(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst, target_is_directory=True)
    return {"mode": "symlink", "src": rel(src), "dst": rel(dst)}


def copy_with_scaled_config(src: Path, dst: Path, scale: float, force: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if not (src / "model.safetensors").exists():
        raise FileNotFoundError(src / "model.safetensors")
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(dst)
        remove_existing(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
            copied[item.name] = "copytree"
        elif item.name == "config.json":
            continue
        elif item.suffix in {".safetensors", ".bin"}:
            try:
                os.link(item, target)
                copied[item.name] = "hardlink"
            except Exception:
                shutil.copy2(item, target)
                copied[item.name] = "copy"
        else:
            shutil.copy2(item, target)
            copied[item.name] = "copy"
    cfg = read_json(src / "config.json")
    old_scale = cfg.get("private_adapter_scale")
    old_enabled = cfg.get("private_adapter_enabled")
    cfg["private_adapter_scale"] = float(scale)
    cfg["private_adapter_enabled"] = True
    cfg.setdefault("architectures", ["FrozenSlowPrivateDebertaV2ForMaskedLM"])
    cfg.setdefault("auto_map", {"AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"})
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rec = {
        "mode": "copy_or_hardlink_scaled_config",
        "src": rel(src),
        "dst": rel(dst),
        "scale": float(scale),
        "old_private_adapter_scale": old_scale,
        "old_private_adapter_enabled": old_enabled,
        "copied": copied,
        "model_safetensors_sha256": sha256_file(dst / "model.safetensors"),
        "config_sha256": sha256_file(dst / "config.json"),
    }
    (dst / "alpha075_materialization_manifest.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rec


def classify_revision(rev: str) -> dict[str, Any]:
    n = int(rev.split("_")[1].rstrip("M"))
    if n <= 82:
        return {
            "revision": rev,
            "phase": "ancestral_slow_before_private_intervention",
            "source": SLOW_LADDER / rev,
            "operation": "symlink",
            "exposure_meaning": f"true scale1.75 slow trajectory after approximately {n}M word exposure; no private adapter exists before 82M intervention",
        }
    if rev in TAIL_TOTAL_MAP:
        return {
            "revision": rev,
            "phase": "real_coherent_private_tail_materialized_alpha0p75",
            "source": COHERENT_TAIL / TAIL_TOTAL_MAP[rev],
            "operation": "scale_config_copy",
            "true_tail_checkpoint_name": TAIL_TOTAL_MAP[rev],
            "exposure_meaning": f"real coherent private-tail state at total {TAIL_TOTAL_MAP[rev].split('_total_')[1].rstrip('w')} words, named {rev} only for official fast/AoA revision slots",
        }
    if rev == "chck_90M" or rev == "chck_100M":
        return {
            "revision": rev,
            "phase": "missing_requires_genuine_continuation",
            "source": None,
            "operation": "missing_do_not_fake",
            "exposure_meaning": "not available from current coherent86 alpha0.75 lineage; must be produced by genuine continuation from unchanged coherent private-tail state, not duplicated or borrowed",
        }
    return {
        "revision": rev,
        "phase": "post_candidate_missing_nonfast_revision",
        "source": None,
        "operation": "missing_do_not_fake",
        "exposure_meaning": "not part of strict-small fast revision list beyond 86M candidate unless genuine continuation creates it",
    }


def model_identity(path: Path) -> dict[str, Any]:
    cfg = read_json(path / "config.json")
    return {
        "path": rel(path),
        "exists": path.exists(),
        "model_sha256": sha256_file(path / "model.safetensors") if (path / "model.safetensors").exists() else None,
        "config_sha256": sha256_file(path / "config.json") if (path / "config.json").exists() else None,
        "architectures": cfg.get("architectures") if (path / "config.json").exists() else None,
        "auto_map": cfg.get("auto_map") if (path / "config.json").exists() else None,
        "adapter_scale": cfg.get("adapter_scale") if (path / "config.json").exists() else None,
        "private_adapter_scale": cfg.get("private_adapter_scale") if (path / "config.json").exists() else None,
        "private_adapter_enabled": cfg.get("private_adapter_enabled") if (path / "config.json").exists() else None,
        "vocab_size": cfg.get("vocab_size") if (path / "config.json").exists() else None,
    }


def build_staging(out_dir: Path, force: bool, make_partial: bool) -> dict[str, Any]:
    if not ALPHA075_CARRIER_MANIFEST.exists():
        raise FileNotFoundError(ALPHA075_CARRIER_MANIFEST)
    carrier = read_json(ALPHA075_CARRIER_MANIFEST)
    expected_final_sha = carrier["model_identity"]["model_safetensors_sha256"]
    out_hf = out_dir / "hf_model"
    out_hf.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    missing_required: list[str] = []
    staged_revisions: list[str] = []

    for rev in FAST_REVISIONS:
        spec = classify_revision(rev)
        rec: dict[str, Any] = dict(spec)
        if rec.get("source") is not None:
            rec["source"] = rel(Path(rec["source"]))
        if spec["source"] is not None:
            src = Path(spec["source"])
            dst = out_hf / rev
            if spec["operation"] == "symlink":
                rec["materialization"] = symlink_dir(src, dst, force=force)
            elif spec["operation"] == "scale_config_copy":
                rec["materialization"] = copy_with_scaled_config(src, dst, scale=0.75, force=force)
            else:
                raise ValueError(spec["operation"])
            rec["identity"] = model_identity(dst)
            staged_revisions.append(rev)
        else:
            missing_required.append(rev)
        records.append(rec)

    # Preserve the real saved 83--85M coherent-tail states as optional nonofficial
    # aliases.  Strict-Small --fast/AoA does not request these revisions, but they
    # are the developmental bridge between the 82M private intervention and the
    # 86M full-eval endpoint; preserving them prevents later fake-history repairs.
    optional_tail_records: list[dict[str, Any]] = []
    for rev, src_name in TAIL_TOTAL_MAP.items():
        src = COHERENT_TAIL / src_name
        dst = out_hf / rev
        mat = copy_with_scaled_config(src, dst, scale=0.75, force=force)
        optional_tail_records.append({
            "revision": rev,
            "phase": "optional_nonofficial_real_coherent_private_tail_alpha0p75",
            "source": rel(src),
            "operation": "scale_config_copy",
            "true_tail_checkpoint_name": src_name,
            "materialization": mat,
            "identity": model_identity(dst),
            "exposure_meaning": f"real coherent private-tail state at total {src_name.split('_total_')[1].rstrip('w')} words, not an official strict-small fast slot",
        })

    # Also stage a main alias and an 86M alias for the exact alpha0.75 candidate.  chck_86M is
    # not one of the 19 strict-small fast slots but is the measured full-eval endpoint.
    main_rec = symlink_dir(ALPHA075_FINAL, out_hf / "main", force=force)
    chck86_rec = symlink_dir(ALPHA075_FINAL, out_hf / "chck_86M", force=force)
    final_identity = model_identity(ALPHA075_FINAL)
    if final_identity["model_sha256"] != expected_final_sha:
        raise RuntimeError({"final_sha_mismatch": final_identity, "expected": expected_final_sha})

    manifest = {
        "status": "ALPHA075_DEVELOPMENTAL_STAGING_PARTIAL" if missing_required else "ALPHA075_DEVELOPMENTAL_STAGING_COMPLETE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "hf_model_root": rel(out_hf),
        "strict_small_fast_revisions": FAST_REVISIONS,
        "staged_fast_revisions": staged_revisions,
        "missing_fast_revisions": missing_required,
        "can_run_full_official_fast_or_aoa": not missing_required,
        "can_run_pilots_on_existing_revisions": bool(staged_revisions) and make_partial,
        "truthful_developmental_policy": {
            "before_private_intervention": "chck_1M..chck_82M are true scale1.75 slow ancestor checkpoints; no private weights are retrofitted into early history",
            "private_tail_existing": "chck_83M..chck_85M are real coherent-tail checkpoint states, re-materialized only by changing config private_adapter_scale from 1.0 to 0.75",
            "candidate_endpoint": "main and chck_86M point to the exact alpha0.75 candidate final model; chck_86M is not an official fast slot but preserves endpoint identity",
            "after_candidate": "chck_90M and chck_100M are absent until genuine continuation from the unchanged coherent private tail creates them; this script refuses duplicates or ordinary-trajectory borrowing",
        },
        "source_roots": {
            "slow_ladder": rel(SLOW_LADDER),
            "coherent_tail_alpha1": rel(COHERENT_TAIL),
            "alpha075_final": rel(ALPHA075_FINAL),
            "alpha075_carrier_manifest": rel(ALPHA075_CARRIER_MANIFEST),
        },
        "final_candidate_identity": final_identity,
        "main_alias": main_rec,
        "chck86_alias": chck86_rec,
        "optional_nonofficial_tail_revisions": optional_tail_records,
        "records": records,
    }
    write_json(out_dir / "alpha075_developmental_staging_manifest.json", manifest)
    lines = [
        "# research alpha0.75 developmental staging",
        "",
        f"Status: **{manifest['status']}**",
        f"HF model root: `{manifest['hf_model_root']}`",
        f"Staged fast revisions: {len(staged_revisions)}/{len(FAST_REVISIONS)}",
        f"Missing fast revisions: `{missing_required}`",
        "",
        "## Interpretation",
        "This is a truthful partial lineage, not a completed submission ladder. Early checkpoints remain the true ancestral slow trajectory; private-tail checkpoints are only the real saved coherent-tail states re-materialized at alpha=0.75; 90M/100M are intentionally absent until genuine continuation creates them.",
        "",
        f"Manifest: `{rel(out_dir / 'alpha075_developmental_staging_manifest.json')}`",
    ]
    (out_dir / "alpha075_developmental_staging_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--make-partial", action="store_true", help="Allow truthful partial staging with missing 90M/100M reported")
    args = ap.parse_args()
    manifest = build_staging(Path(args.out_dir), force=args.force, make_partial=args.make_partial)
    print(json.dumps({
        "status": manifest["status"],
        "hf_model_root": manifest["hf_model_root"],
        "staged_fast_revisions": len(manifest["staged_fast_revisions"]),
        "missing_fast_revisions": manifest["missing_fast_revisions"],
        "can_run_full_official_fast_or_aoa": manifest["can_run_full_official_fast_or_aoa"],
        "manifest": rel(Path(args.out_dir) / "alpha075_developmental_staging_manifest.json"),
    }, indent=2), flush=True)
    if manifest["missing_fast_revisions"] and not args.make_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
