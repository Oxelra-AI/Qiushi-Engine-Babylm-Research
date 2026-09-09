#!/usr/bin/env python3
"""research: token-geometry KL bins for special-token format movement.

The research fast readout showed that private/slow KL is several times larger on
short isolated rows with special tokens than on long coherent rows.  This script
keeps the same matched content but records where that KL appears: token distance
from <s>/</s>, row length, and form.  It performs no training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import pathlib
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
MANIFEST = _public_path('experiments/archive/relation_learning/data/short_format_readout/held_short_readout_manifest.json')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/special_geometry_kl_bins')

DEFAULT_ENDPOINTS = {
    "coherent86_alpha075": "models/frontier",
    "coherent_special_98097_alpha075": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/alpha_0.75",
    "coherent_special_98098_alpha075": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/alpha_0.75",
}

DIST_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 10_000]
LEN_BINS = [0, 16, 32, 64, 128, 256, 10_000]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_rows(path: pathlib.Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in iter_jsonl(path):
        rows.append(obj)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def set_private_enabled(model, enabled: bool) -> None:
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(bool(enabled))
    elif hasattr(model, "config") and hasattr(model.config, "private_adapter_enabled"):
        model.config.private_adapter_enabled = bool(enabled)


def token_rows(tokenizer, rows: list[dict[str, Any]], add_special_tokens: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    for ri, obj in enumerate(rows):
        enc = tokenizer(
            str(obj.get("text", "")),
            truncation=True,
            max_length=256,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        ids = enc["input_ids"].view(-1)
        att = enc["attention_mask"].view(-1)
        active_positions = [int(i) for i, v in enumerate(att.tolist()) if int(v) == 1]
        special_positions = [i for i in active_positions if int(ids[i]) in special_ids]
        if special_positions:
            dists = {i: min(abs(i - sp) for sp in special_positions) for i in active_positions}
        else:
            dists = {i: -1 for i in active_positions}
        out.append({
            "row_index": ri,
            "source_id": obj.get("example_id", obj.get("id", ri)),
            "text_words": int(obj.get("words", len(str(obj.get("text", "")).split()))),
            "ids": ids,
            "att": att,
            "n_active_tokens": len(active_positions),
            "special_positions": special_positions,
            "distance_to_nearest_special": dists,
            "add_special_tokens": add_special_tokens,
        })
    return out


def collate(tok_rows: list[dict[str, Any]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(r["ids"].numel()) for r in tok_rows)
    ids = torch.full((len(tok_rows), max_len), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((len(tok_rows), max_len), dtype=torch.long, device=device)
    for i, r in enumerate(tok_rows):
        v = r["ids"].to(device)
        a = r["att"].to(device)
        ids[i, : v.numel()] = v
        att[i, : a.numel()] = a
    return ids, att


def bucket(value: int, bins: list[int]) -> str:
    if value < 0:
        return "none"
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            if hi >= 10_000:
                return f">={lo}"
            if hi == lo + 1:
                return str(lo)
            return f"{lo}-{hi-1}"
    return f">={bins[-2]}"


def init_acc() -> dict[str, float]:
    return {"tokens": 0.0, "kl_sum": 0.0, "kl_sq_sum": 0.0, "kl_max": 0.0}


def add_acc(acc: dict[str, float], x: float) -> None:
    acc["tokens"] += 1.0
    acc["kl_sum"] += x
    acc["kl_sq_sum"] += x * x
    if x > acc["kl_max"]:
        acc["kl_max"] = x


def finish_acc(acc: dict[str, float]) -> dict[str, Any]:
    n = int(acc["tokens"])
    mean = acc["kl_sum"] / max(1, n)
    var = max(0.0, acc["kl_sq_sum"] / max(1, n) - mean * mean)
    return {"tokens": n, "kl_mean": mean, "kl_sd_pop": var ** 0.5, "kl_max": acc["kl_max"]}


def load_model_and_tokenizer(endpoint: pathlib.Path, device: torch.device, cache_root: pathlib.Path):
    cache_items = {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TMPDIR": cache_root / "tmp",
    }
    for k, p in cache_items.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def measure_endpoint(label: str, endpoint: pathlib.Path, coherent_rows: list[dict[str, Any]], short_rows: list[dict[str, Any]], device: torch.device, batch_size: int, out_dir: pathlib.Path) -> dict[str, Any]:
    t0 = time.time()
    model, tok = load_model_and_tokenizer(endpoint, device, out_dir / "hf_cache" / label)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    special_ids = set(int(x) for x in getattr(tok, "all_special_ids", []) if x is not None)
    forms = {
        "coherent_add_special": token_rows(tok, coherent_rows, True),
        "short_add_special": token_rows(tok, short_rows, True),
        "coherent_no_special": token_rows(tok, coherent_rows, False),
    }
    form_acc: dict[str, dict[str, float]] = {}
    dist_acc: dict[tuple[str, str, str, str], dict[str, float]] = {}
    len_acc: dict[tuple[str, str], dict[str, float]] = {}
    top_tokens: list[dict[str, Any]] = []

    for form, rows_tok in forms.items():
        form_acc[form] = init_acc()
        for st in range(0, len(rows_tok), batch_size):
            rb = rows_tok[st:st+batch_size]
            ids, att = collate(rb, pad_id, device)
            set_private_enabled(model, False)
            slow = model(input_ids=ids, attention_mask=att).logits.detach()
            set_private_enabled(model, True)
            priv = model(input_ids=ids, attention_mask=att).logits.detach()
            kl = F.kl_div(F.log_softmax(priv, dim=-1), F.softmax(slow, dim=-1), reduction="none").sum(-1)
            for bi, r in enumerate(rb):
                row_len_bin = bucket(int(r["n_active_tokens"]), LEN_BINS)
                for pos in range(int(r["ids"].numel())):
                    if int(r["att"][pos]) != 1:
                        continue
                    tid = int(r["ids"][pos])
                    is_special = tid in special_ids
                    token_kind = "special" if is_special else "content"
                    dist = int(r["distance_to_nearest_special"].get(pos, -1))
                    dist_bin = bucket(dist, DIST_BINS)
                    x = float(kl[bi, pos].cpu())
                    add_acc(form_acc[form], x)
                    add_acc(len_acc.setdefault((form, row_len_bin), init_acc()), x)
                    add_acc(dist_acc.setdefault((form, token_kind, dist_bin, row_len_bin), init_acc()), x)
                    if len(top_tokens) < 40 or x > min(t["kl"] for t in top_tokens):
                        top_tokens.append({
                            "endpoint": label,
                            "form": form,
                            "source_id": r["source_id"],
                            "row_index": r["row_index"],
                            "position": pos,
                            "token_id": tid,
                            "token": tok.convert_ids_to_tokens([tid])[0],
                            "token_kind": token_kind,
                            "distance_to_nearest_special": dist,
                            "row_active_tokens": int(r["n_active_tokens"]),
                            "row_length_bin": row_len_bin,
                            "kl": x,
                        })
                        top_tokens = sorted(top_tokens, key=lambda z: z["kl"], reverse=True)[:40]
            del ids, att, slow, priv, kl
    dist_rows: list[dict[str, Any]] = []
    for (form, token_kind, dist_bin, row_len_bin), acc in sorted(dist_acc.items()):
        rec = {"endpoint": label, "form": form, "token_kind": token_kind, "distance_bin": dist_bin, "row_length_bin": row_len_bin}
        rec.update(finish_acc(acc))
        dist_rows.append(rec)
    len_rows: list[dict[str, Any]] = []
    for (form, row_len_bin), acc in sorted(len_acc.items()):
        rec = {"endpoint": label, "form": form, "row_length_bin": row_len_bin}
        rec.update(finish_acc(acc))
        len_rows.append(rec)
    rec = {
        "status": "measured",
        "endpoint": rel(endpoint),
        "model_class": type(model).__name__,
        "private_scale": float(getattr(model.config, "private_adapter_scale", -1.0)) if hasattr(model, "config") else None,
        "forms": {k: finish_acc(v) for k, v in form_acc.items()},
        "by_distance": dist_rows,
        "by_row_length": len_rows,
        "top_tokens": top_tokens,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--endpoints-json", default="")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(MANIFEST)
    coherent_rows = load_rows(ROOT / manifest["coherent_readout_path"], args.rows)
    short_rows = load_rows(ROOT / manifest["short_isolated_readout_path"], args.rows)
    endpoints = DEFAULT_ENDPOINTS if not args.endpoints_json else read_json(pathlib.Path(args.endpoints_json))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    for label, p in endpoints.items():
        endpoint = ROOT / p
        if not (endpoint / "model.safetensors").exists():
            results[label] = {"status": "missing_model", "endpoint": rel(endpoint)}
            continue
        rec = measure_endpoint(label, endpoint, coherent_rows, short_rows, device, args.batch_size, out_dir)
        results[label] = rec
        c = rec["forms"]["coherent_add_special"]["kl_mean"]
        s = rec["forms"]["short_add_special"]["kl_mean"]
        print(json.dumps({"event": "measured", "label": label, "coherent_kl": c, "short_kl": s, "ratio": s / c if c else None}, ensure_ascii=False), flush=True)

    dist_rows = [r for rec in results.values() for r in rec.get("by_distance", [])]
    len_rows = [r for rec in results.values() for r in rec.get("by_row_length", [])]
    top_rows = [r for rec in results.values() for r in rec.get("top_tokens", [])]
    write_csv(out_dir / "kl_by_distance.csv", dist_rows)
    write_csv(out_dir / "kl_by_row_length.csv", len_rows)
    write_csv(out_dir / "top_kl_tokens.csv", top_rows)
    summary = {
        "status": "SPECIAL_GEOMETRY_KL_BINS_DONE",
        "created_utc": now(),
        "rows_per_form": args.rows,
        "source_manifest": rel(MANIFEST),
        "device": str(device),
        "batch_size": args.batch_size,
        "results": results,
        "tables": {
            "by_distance": rel(out_dir / "kl_by_distance.csv"),
            "by_row_length": rel(out_dir / "kl_by_row_length.csv"),
            "top_tokens": rel(out_dir / "top_kl_tokens.csv"),
        },
        "interpretation_note": "Distance is token-index distance to the nearest special token among active positions. Concentration at low distances and in short-row bins supports a special-token geometry channel; broad elevation across distances supports more global short-format movement.",
    }
    out_json = out_dir / "special_geometry_kl_bins.json"
    out_md = out_dir / "special_geometry_kl_bins.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research special-token geometry KL bins",
        "",
        f"Rows per form: {args.rows}.",
        "",
        "| endpoint | scale | coherent+special KL | short+special KL | coherent no-special KL | ratio | max short token KL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in results.items():
        if r.get("status") != "measured":
            lines.append(f"| {label} | NA | NA | NA | NA | NA | NA |")
            continue
        c = r["forms"]["coherent_add_special"]["kl_mean"]
        s = r["forms"]["short_add_special"]["kl_mean"]
        n = r["forms"]["coherent_no_special"]["kl_mean"]
        mx = r["forms"]["short_add_special"]["kl_max"]
        lines.append(f"| {label} | {r.get('private_scale')} | {c:.8f} | {s:.8f} | {n:.8f} | {s/c if c else 0:.3f} | {mx:.4f} |")
    lines += ["", "## Largest KL tokens", "", "| endpoint | form | token | kind | dist | row tokens | KL |", "|---|---|---|---|---:|---:|---:|"]
    for r in sorted(top_rows, key=lambda z: z.get("kl", 0), reverse=True)[:30]:
        tok = str(r.get("token", "")).replace("|", "¦")
        lines.append(f"| {r['endpoint']} | {r['form']} | `{tok}` | {r['token_kind']} | {r['distance_to_nearest_special']} | {r['row_active_tokens']} | {r['kl']:.5f} |")
    lines += ["", f"Distance table: `{rel(out_dir / 'kl_by_distance.csv')}`", f"Row-length table: `{rel(out_dir / 'kl_by_row_length.csv')}`", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
