#!/usr/bin/env python3
"""research: compare the from-corpus scale1.75 reproduction to the original ladder.

Run after the from-corpus reproduction finishes.  This script checks whether the independent
from-corpus reproduction regenerated the original `chck_82M` endpoint under the
fixed selection rule.  It does not run official evaluation; if hashes differ,
launch the hardened evaluator on the reproduced `chck_82M` instead of transferring
research/166 scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
A02 = ROOT / "experiments/archive/frontier_consolidation"
ORIG_RUN = A02 / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
REPRO_RUN = STUDY / "training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022"
OUT = STUDY / "data/from_corpus_reproduction_compare"
EXPECTED = {
    "chck_20M": "b37e243b5fc9c9459efe0b6b4f2abce40f4597d82c9621949e0d458e85e7b031",
    "chck_50M": "995c167832d83a721e1cccb6c3d4f348d11443283845ed5d72ba39bb8e802034",
    "chck_80M": "c37f6665df84109a266428e606e1bf665db70d9a5e7842816b5717d36922edda",
    "chck_82M": "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3",
    "chck_100M": "7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52",
}
CHECKPOINTS = ["chck_1M", "chck_2M", "chck_3M", "chck_4M", "chck_5M", "chck_6M", "chck_7M", "chck_8M", "chck_9M"] + [f"chck_{i}M" for i in range(10, 101)]
KEY_CHECKPOINTS = ["chck_20M", "chck_50M", "chck_80M", "chck_82M", "chck_100M"]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_hash(run: Path, ckpt: str) -> dict[str, Any]:
    p = run / "hf_model" / ckpt / "model.safetensors"
    if not p.exists():
        return {"path": rel(p), "exists": False}
    return {"path": rel(p), "exists": True, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}


def checkpoint_record(metrics: dict[str, Any], name: str) -> dict[str, Any] | None:
    for rec in metrics.get("saved_checkpoints", []):
        if isinstance(rec, dict) and rec.get("name") == name:
            return rec
    return None


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    if not REPRO_RUN.exists():
        out = {
            "status": "PENDING",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": "Reproduction run directory does not exist yet.",
            "repro_run": rel(REPRO_RUN),
        }
        write_json(OUT / "from_corpus_reproduction_compare.json", out)
        print(json.dumps(out, indent=2), flush=True)
        return

    metrics_path = REPRO_RUN / "scientific_metrics.json"
    if not metrics_path.exists():
        out = {
            "status": "PENDING",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": "Reproduction run exists but scientific_metrics.json is not present yet.",
            "repro_run": rel(REPRO_RUN),
            "training_log_exists": (REPRO_RUN / "training_log.jsonl").exists(),
        }
        write_json(OUT / "from_corpus_reproduction_compare.json", out)
        print(json.dumps(out, indent=2), flush=True)
        return

    metrics = read_json(metrics_path)
    orig_metrics = read_json(ORIG_RUN / "scientific_metrics.json")
    key_compare = {}
    for ckpt in KEY_CHECKPOINTS:
        orig = model_hash(ORIG_RUN, ckpt)
        repro = model_hash(REPRO_RUN, ckpt)
        key_compare[ckpt] = {
            "original": orig,
            "reproduction": repro,
            "hash_equal": orig.get("sha256") == repro.get("sha256") and bool(orig.get("exists")) and bool(repro.get("exists")),
            "matches_expected_original_hash": repro.get("sha256") == EXPECTED.get(ckpt),
            "original_record": checkpoint_record(orig_metrics, ckpt),
            "reproduction_record": checkpoint_record(metrics, ckpt),
        }
    missing = [ck for ck in CHECKPOINTS if not (REPRO_RUN / "hf_model" / ck / "model.safetensors").exists()]
    if metrics.get("word_exposure") != 100_000_000:
        errors.append(f"reproduction word_exposure {metrics.get('word_exposure')} != 100000000")
    if metrics.get("actual_training_steps") != 2529:
        errors.append(f"reproduction steps {metrics.get('actual_training_steps')} != 2529")
    if len(metrics.get("saved_checkpoints", [])) != 100:
        errors.append(f"reproduction saved_checkpoints count {len(metrics.get('saved_checkpoints', []))} != 100")
    if missing:
        errors.append(f"missing checkpoint model files count {len(missing)}")
    if not key_compare.get("chck_82M", {}).get("hash_equal"):
        errors.append("reproduced chck_82M is not bit-identical to original; evaluate reproduced chck_82M directly before any score transfer")
    if not key_compare.get("chck_80M", {}).get("hash_equal"):
        warnings.append("chck_80M differs; deterministic-prefix identity did not reproduce in this run")

    first_loss = metrics.get("loss_first")
    loss_first_ok = isinstance(first_loss, (int, float)) and abs(float(first_loss) - 9.837543487548828) < 1e-6
    if not loss_first_ok:
        errors.append(f"first loss mismatch: {first_loss}")

    out = {
        "status": "PASS" if not errors else "CHECK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_run": rel(ORIG_RUN),
        "reproduction_run": rel(REPRO_RUN),
        "errors": errors,
        "warnings": warnings,
        "metrics_core": {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "example_jsonl": metrics.get("example_jsonl"),
        },
        "missing_checkpoint_model_files": missing,
        "key_checkpoint_compare": key_compare,
        "decision": "scores_transfer_if_hash_identical_else_evaluate_reproduced_chck82",
    }
    out_json = OUT / "from_corpus_reproduction_compare.json"
    out_md = OUT / "from_corpus_reproduction_compare.md"
    write_json(out_json, out)
    lines = [
        "# research from-corpus reproduction comparison",
        "",
        f"Status: **{out['status']}**",
        f"Reproduction run: `{rel(REPRO_RUN)}`",
        "",
        "| checkpoint | original sha | reproduced sha | equal |",
        "|---|---|---|---:|",
    ]
    for ckpt in KEY_CHECKPOINTS:
        rec = key_compare[ckpt]
        lines.append(f"| {ckpt} | `{rec['original'].get('sha256')}` | `{rec['reproduction'].get('sha256')}` | {rec['hash_equal']} |")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    if errors:
        lines += ["", "Errors:"] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "Warnings:"] + [f"- {w}" for w in warnings]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": rel(out_json), "chck82_equal": key_compare.get("chck_82M", {}).get("hash_equal"), "errors": errors}, indent=2), flush=True)

if __name__ == "__main__":
    main()
