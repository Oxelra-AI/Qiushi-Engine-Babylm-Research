#!/usr/bin/env python3
"""research: context-gain coordinate for legal format/private endpoints.

This reuses the research Strict-complement sentence coordinate for arbitrary trusted
endpoints such as chck82, coherent86, coherent-special, isolated, and half-format
private branches.  The coordinate is

    context_gain = isolation_loss - row_context_loss

where row_context scores a target sentence inside its 160-word Strict-complement row
and isolation scores the same sentence alone.  Positive endpoint-minus-anchor
context_gain means the endpoint benefits more from adjacent row context on the same
masked target spans.

No training is performed here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import statistics
import time
from typing import Any

import pandas as pd
import torch

ROOT = _public_path('.')
WS = _public_path('experiments/archive/relation_learning')
PATH = _public_path('experiments/archive/relation_learning/scripts/context_isolation_coordinate.py')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/format_context_gain_coordinate')
AXIS_DEFAULT = _public_path('experiments/archive/relation_learning/data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl')

DEFAULT_ENDPOINTS = {
    "chck82_slow_scale1p75": {
        "path": "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M",
        "trust_remote_code": True,
        "description": "protected slow-adapter chck82 reference",
    },
    "coherent86_alpha075": {
        "path": "models/frontier",
        "trust_remote_code": True,
        "description": "historical coherent86 alpha0.75 private endpoint",
    },
    "coherent_special_98097_alpha075": {
        "path": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/alpha_0.75",
        "trust_remote_code": True,
        "description": "research exact coherent suffix trained with special tokens, seed98097 alpha0.75",
    },
    "coherent_special_98098_alpha075": {
        "path": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/alpha_0.75",
        "trust_remote_code": True,
        "description": "research exact coherent suffix trained with special tokens, seed98098 alpha0.75",
    },
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def se(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) <= 1:
        return float("nan")
    return statistics.stdev(vals) / math.sqrt(len(vals))


def prepare_cache(out_dir: pathlib.Path) -> None:
    cache = out_dir / "hf_cache"
    env_map = {
        "CONTEXT_HF_CACHE": cache,
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }
    for k, v in env_map.items():
        os.environ[k] = str(v)
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)


def load_step076_module():
    spec = importlib.util.spec_from_file_location("context_isolation_coordinate", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_endpoint_spec(path: str) -> dict[str, Any]:
    if not path:
        return DEFAULT_ENDPOINTS
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "endpoints" in data:
        data = data["endpoints"]
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = {"path": v, "trust_remote_code": True, "description": ""}
        else:
            out[k] = dict(v)
            out[k].setdefault("trust_remote_code", True)
            out[k].setdefault("description", "")
    return out


def model_file_ready(path: pathlib.Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(score_rows: list[dict[str, Any]], anchors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(score_rows)
    if df.empty:
        return [], [], []
    # Filter as in research: remove sentence/endpoints whose row-context target was
    # truncated or whose whole target span was selected for masking.
    df["target_truncated_right_bool"] = df["target_truncated_right"].astype(str).str.lower().isin(["true", "1"])
    row_bad = df[(df["mode"] == "row_context") & (df["target_truncated_right_bool"] | (df["n_masked"] >= df["n_target_tokens"]))]
    bad = set(zip(row_bad["endpoint"], row_bad["sentence_uid"]))
    keep = [(e, s) not in bad for e, s in zip(df["endpoint"], df["sentence_uid"])]
    filt = df.loc[keep].copy()
    piv = filt.pivot_table(index=["endpoint", "sentence_uid", "source", "sentence_words"], columns="mode", values="loss", aggfunc="mean").reset_index()
    piv = piv.dropna(subset=["row_context", "isolation"])
    piv["context_gain"] = piv["isolation"] - piv["row_context"]
    summaries: list[dict[str, Any]] = []
    for endpoint, g in piv.groupby("endpoint", dropna=False):
        cg = [float(x) for x in g["context_gain"].tolist()]
        summaries.append({
            "endpoint": endpoint,
            "n_sentences_filtered": int(len(g)),
            "row_context_loss": float(g["row_context"].mean()),
            "isolation_loss": float(g["isolation"].mean()),
            "context_gain": float(g["context_gain"].mean()),
            "se_context_gain": se(cg),
            "removed_bad_sentences": int(len(set(df[df["endpoint"] == endpoint]["sentence_uid"])) - len(set(g["sentence_uid"]))),
        })
    contrast_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in set(piv["endpoint"]):
            continue
        a = piv[piv["endpoint"] == anchor].set_index("sentence_uid")
        for endpoint in sorted(set(piv["endpoint"])):
            if endpoint == anchor:
                continue
            b = piv[piv["endpoint"] == endpoint].set_index("sentence_uid")
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            da = a.loc[common]
            db = b.loc[common]
            d_row = (db["row_context"].to_numpy(dtype=float) - da["row_context"].to_numpy(dtype=float)).tolist()
            d_iso = (db["isolation"].to_numpy(dtype=float) - da["isolation"].to_numpy(dtype=float)).tolist()
            d_gain = (db["context_gain"].to_numpy(dtype=float) - da["context_gain"].to_numpy(dtype=float)).tolist()
            contrast_rows.append({
                "endpoint": endpoint,
                "anchor": anchor,
                "n_common_sentences_filtered": int(len(common)),
                "delta_row_context_loss": float(sum(d_row) / len(d_row)),
                "delta_isolation_loss": float(sum(d_iso) / len(d_iso)),
                "delta_context_gain": float(sum(d_gain) / len(d_gain)),
                "se_delta_context_gain": se(d_gain),
                "fraction_endpoint_higher_context_gain": float(sum(1 for x in d_gain if x > 0) / len(d_gain)),
            })
    source_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in set(piv["endpoint"]):
            continue
        a = piv[piv["endpoint"] == anchor].set_index("sentence_uid")
        for endpoint in sorted(set(piv["endpoint"])):
            if endpoint == anchor:
                continue
            b = piv[piv["endpoint"] == endpoint].set_index("sentence_uid")
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            tmp = pd.DataFrame({
                "source": b.loc[common]["source"].tolist(),
                "d_row": (b.loc[common]["row_context"].to_numpy(dtype=float) - a.loc[common]["row_context"].to_numpy(dtype=float)).tolist(),
                "d_iso": (b.loc[common]["isolation"].to_numpy(dtype=float) - a.loc[common]["isolation"].to_numpy(dtype=float)).tolist(),
                "d_gain": (b.loc[common]["context_gain"].to_numpy(dtype=float) - a.loc[common]["context_gain"].to_numpy(dtype=float)).tolist(),
            })
            for src, gg in tmp.groupby("source"):
                vals = [float(x) for x in gg["d_gain"].tolist()]
                source_rows.append({
                    "endpoint": endpoint,
                    "anchor": anchor,
                    "source": src,
                    "n_common_sentences_filtered": int(len(gg)),
                    "delta_row_context_loss": float(gg["d_row"].mean()),
                    "delta_isolation_loss": float(gg["d_iso"].mean()),
                    "delta_context_gain": float(gg["d_gain"].mean()),
                    "se_delta_context_gain": se(vals),
                })
    return summaries, contrast_rows, source_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--endpoints-json", default="")
    ap.add_argument("--axis", default=str(AXIS_DEFAULT))
    ap.add_argument("--max-sentences", type=int, default=720)
    ap.add_argument("--min-sentence-words", type=int, default=8)
    ap.add_argument("--max-sentence-words", type=int, default=45)
    ap.add_argument("--sample-seed", type=int, default=7601)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--anchors", nargs="*", default=["chck82_slow_scale1p75", "coherent86_alpha075"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_cache(out_dir)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    research = load_step076_module()
    endpoints = read_endpoint_spec(args.endpoints_json)
    sentence_records = research.load_sentence_records(pathlib.Path(args.axis), max_sentences=args.max_sentences, min_words=args.min_sentence_words, max_words=args.max_sentence_words, seed=args.sample_seed)
    plan = {
        "status": "FORMAT_CONTEXT_GAIN_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "axis": rel(args.axis),
        "n_sentences": len(sentence_records),
        "endpoints": {k: {**v, "path": rel(ROOT / v["path"])} for k, v in endpoints.items()},
        "missing": [rel(ROOT / v["path"]) for v in endpoints.values() if not model_file_ready(ROOT / v["path"])],
        "anchors": args.anchors,
        "device": args.device,
        "batch_size": args.batch_size,
        "filtering": "research filter: remove endpoint/sentence pairs whose row-context target span is truncated or whose entire target span is masked.",
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if plan["missing"]:
        raise FileNotFoundError(plan["missing"][0])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    score_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for label, spec in endpoints.items():
        endpoint = ROOT / spec["path"]
        t0 = time.time()
        print(json.dumps({"event": "load", "endpoint": label, "path": rel(endpoint)}, ensure_ascii=False), flush=True)
        tok = research.AutoTokenizer.from_pretrained(str(endpoint), use_fast=True, trust_remote_code=bool(spec.get("trust_remote_code", True)), local_files_only=True)
        model = research.load_model(endpoint, bool(spec.get("trust_remote_code", True)), device)
        ident = research.model_identity(endpoint, model, bool(spec.get("trust_remote_code", True)))
        ident.update({"endpoint": label, "description": spec.get("description", "")})
        identities.append(ident)
        scored = research.score_model(model, tok, sentence_records, device, args.batch_size)
        for r in scored:
            q = dict(r)
            q.update({"endpoint": label, "endpoint_path": rel(endpoint), "description": spec.get("description", "")})
            score_rows.append(q)
        meta_rows.append({
            "endpoint": label,
            "endpoint_path": rel(endpoint),
            "n_sentence_records": len(sentence_records),
            "n_scored_mode_rows": len(scored),
            "elapsed_sec": round(time.time() - t0, 1),
            "loaded_class": ident.get("loaded_class"),
            "total_params_loaded": ident.get("total_params_loaded"),
            "adapter_params_loaded": ident.get("adapter_params_loaded"),
        })
        print(json.dumps({"event": "done", **meta_rows[-1]}, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries, contrasts, source_contrasts = summarize(score_rows, args.anchors)
    write_csv(out_dir / "context_gain_scores.csv", score_rows)
    write_csv(out_dir / "endpoint_summary.csv", summaries)
    write_csv(out_dir / "endpoint_contrasts.csv", contrasts)
    write_csv(out_dir / "source_contrasts.csv", source_contrasts)
    write_csv(out_dir / "score_meta.csv", meta_rows)
    with (out_dir / "model_identity.jsonl").open("w", encoding="utf-8") as f:
        for r in identities:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "status": "FORMAT_CONTEXT_GAIN_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "n_sentences": len(sentence_records),
        "endpoints": list(endpoints),
        "summaries": summaries,
        "contrasts": contrasts,
        "source_contrasts": source_contrasts,
        "outputs": {
            "plan": rel(out_dir / "score_plan.json"),
            "scores": rel(out_dir / "context_gain_scores.csv"),
            "endpoint_summary": rel(out_dir / "endpoint_summary.csv"),
            "endpoint_contrasts": rel(out_dir / "endpoint_contrasts.csv"),
            "source_contrasts": rel(out_dir / "source_contrasts.csv"),
            "model_identity": rel(out_dir / "model_identity.jsonl"),
        },
        "interpretation_note": "context_gain = isolation_loss - row_context_loss. Negative endpoint-minus-anchor delta_context_gain means the endpoint relies less on adjacent row context than the anchor on the same Strict-complement spans.",
    }
    out_json = out_dir / "format_context_gain_summary.json"
    out_md = out_dir / "format_context_gain_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research format endpoint context-gain coordinate",
        "",
        summary["interpretation_note"],
        "",
        f"Scored {len(sentence_records)} Strict-complement sentence spans.",
        "",
        "## Endpoint summaries",
        "",
        "| endpoint | n | row loss | isolated loss | context gain | se | removed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(f"| {r['endpoint']} | {r['n_sentences_filtered']} | {r['row_context_loss']:.4f} | {r['isolation_loss']:.4f} | {r['context_gain']:.4f} | {r['se_context_gain']:.4f} | {r['removed_bad_sentences']} |")
    lines += ["", "## Paired contrasts", "", "| endpoint | anchor | n | d_row | d_iso | d_context_gain | se | frac higher |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {r['endpoint']} | {r['anchor']} | {r['n_common_sentences_filtered']} | {r['delta_row_context_loss']:+.4f} | {r['delta_isolation_loss']:+.4f} | {r['delta_context_gain']:+.4f} | {r['se_delta_context_gain']:.4f} | {r['fraction_endpoint_higher_context_gain']:.3f} |")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
