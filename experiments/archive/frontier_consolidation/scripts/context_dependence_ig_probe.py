#!/usr/bin/env python3
"""research: forward-only context-dependence information-gain probe.

For legal-corpus target pieces, compare masked-token NLL under full visible context
against masked-token NLL under local windows. The output is meant to test whether
ordinary MLM spends supervision on locally-solvable targets while a wider-context
underlearned tail remains. It does not train, upload, or submit anything.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
DEFAULT_CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/context_dependence_ig_probe')

# Cache roots are set inside main() before model import because output path is an argument.


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(p: Path | str) -> str:
    q = Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_checkpoint_args(xs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for x in xs:
        if "=" not in x:
            raise ValueError(f"checkpoint must be label=path, got {x!r}")
        label, path = x.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty checkpoint label in {x!r}")
        out[label] = Path(path)
    return out


def parse_radii(s: str) -> list[int]:
    vals = [int(x) for x in s.replace(",", " ").split() if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"bad radii: {s!r}")
    return sorted(set(vals))


def token_is_interesting(tok: str) -> bool:
    # Byte-level BPE tokens may contain Ġ/Ċ or unicode byte escapes; keep tokens that
    # carry at least one alphanumeric character after stripping common markers.
    cleaned = tok.replace("Ġ", "").replace("▁", "").replace("Ċ", "")
    return any(ch.isalnum() for ch in cleaned)


def sample_rows(corpus: Path, sample_rows_n: int, seed: int, max_scan_rows: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    total = 0
    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if max_scan_rows is not None and total >= max_scan_rows:
                break
            total += 1
            obj = json.loads(line)
            obj["_row_index"] = total - 1
            if len(rows) < sample_rows_n:
                rows.append(obj)
            else:
                j = rng.randrange(total)
                if j < sample_rows_n:
                    rows[j] = obj
    rows.sort(key=lambda r: int(r["_row_index"]))
    return rows, {"scan_rows": total, "sample_rows": len(rows), "seed": seed, "max_scan_rows": max_scan_rows}


def build_items(rows: list[dict[str, Any]], tokenizer: Any, *, seed: int, max_seq_len: int, max_targets_per_row: int) -> tuple[list[dict[str, Any]], Counter[int], dict[str, Any]]:
    rng = random.Random(seed)
    special_ids = set(x for x in [
        getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None),
        getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None),
        tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.unk_token_id,
    ] if x is not None)
    items: list[dict[str, Any]] = []
    freq: Counter[int] = Counter()
    row_stats = []
    content_max = max_seq_len - 2
    for row in rows:
        text = row.get("text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > content_max:
            token_ids = token_ids[:content_max]
        freq.update(token_ids)
        candidates = []
        for i, tid in enumerate(token_ids):
            if tid in special_ids:
                continue
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            if token_is_interesting(tok):
                candidates.append(i)
        rng.shuffle(candidates)
        chosen = sorted(candidates[:max_targets_per_row])
        for i in chosen:
            tid = int(token_ids[i])
            target_tok = tokenizer.convert_ids_to_tokens(tid)
            context_count = int(sum(1 for j, x in enumerate(token_ids) if j != i and int(x) == tid))
            items.append({
                "item_id": f"row{row['_row_index']}:tok{i}:id{tid}",
                "row_index": int(row["_row_index"]),
                "example_id": row.get("example_id"),
                "source": row.get("source"),
                "row_words": row.get("words"),
                "position": int(i),
                "seq_len": int(len(token_ids)),
                "position_frac": float(i / max(1, len(token_ids) - 1)),
                "target_id": tid,
                "target_token": target_tok,
                "target_context_count": context_count,
                "content_ids": [int(x) for x in token_ids],
            })
        row_stats.append({"row_index": row["_row_index"], "seq_len": len(token_ids), "candidate_count": len(candidates), "chosen": len(chosen)})
    return items, freq, {"row_stats": row_stats, "n_items": len(items), "n_rows": len(rows)}


def make_context(content_ids: list[int], pos: int, target_id: int, tokenizer: Any, mode: str, radius: int | None, max_seq_len: int) -> tuple[list[int], int]:
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)
    mask_id = tokenizer.mask_token_id
    if cls_id is None or sep_id is None or mask_id is None:
        raise RuntimeError("tokenizer is missing usable bos/cls, eos/sep, or mask token ids")
    if mode == "full":
        left = content_ids[:pos]
        right = content_ids[pos + 1:]
    elif mode == "local":
        assert radius is not None
        start = max(0, pos - radius)
        end = min(len(content_ids), pos + radius + 1)
        local = content_ids[start:end]
        local_pos = pos - start
        left = local[:local_pos]
        right = local[local_pos + 1:]
    else:
        raise ValueError(mode)
    seq = [int(cls_id)] + [int(x) for x in left] + [int(mask_id)] + [int(x) for x in right] + [int(sep_id)]
    mask_index = 1 + len(left)
    if len(seq) > max_seq_len:
        raise RuntimeError(f"constructed sequence too long: {len(seq)} > {max_seq_len}")
    return seq, mask_index


def batch_nll(model: Any, tokenizer: Any, contexts: list[dict[str, Any]], *, device: str, batch_size: int) -> list[float]:
    import torch
    import torch.nn.functional as F

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad token id")
    out: list[float] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start:start + batch_size]
        max_len = max(len(c["input_ids"]) for c in batch)
        input_ids = []
        attention_mask = []
        mask_indices = []
        target_ids = []
        for c in batch:
            ids = list(c["input_ids"])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            mask_indices.append(int(c["mask_index"]))
            target_ids.append(int(c["target_id"]))
        with torch.no_grad():
            x = torch.tensor(input_ids, dtype=torch.long, device=device)
            am = torch.tensor(attention_mask, dtype=torch.long, device=device)
            logits = model(input_ids=x, attention_mask=am).logits
            mi = torch.tensor(mask_indices, dtype=torch.long, device=device)
            ti = torch.tensor(target_ids, dtype=torch.long, device=device)
            row_idx = torch.arange(len(batch), device=device)
            selected = logits[row_idx, mi, :]
            nll = F.cross_entropy(selected, ti, reduction="none")
            out.extend(float(v) for v in nll.detach().cpu().tolist())
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs); my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        r = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def residual_against_one_covariate(vals: list[float], cov: list[float]) -> list[float] | None:
    if len(vals) != len(cov) or len(vals) < 3:
        return None
    mc = statistics.fmean(cov); mv = statistics.fmean(vals)
    vc = sum((c - mc) ** 2 for c in cov)
    if vc <= 0:
        return None
    beta = sum((c - mc) * (v - mv) for c, v in zip(cov, vals)) / vc
    alpha = mv - beta * mc
    return [v - (alpha + beta * c) for v, c in zip(vals, cov)]


def summarize_values(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)
    def q(p: float) -> float:
        if len(xs2) == 1:
            return float(xs2[0])
        idx = p * (len(xs2) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return float(xs2[lo])
        return float(xs2[lo] * (hi - idx) + xs2[hi] * (idx - lo))
    return {"n": len(xs), "mean": float(statistics.fmean(xs)), "median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": float(xs2[0]), "max": float(xs2[-1])}


def set_cache_env(out_dir: Path) -> None:
    cache = out_dir / "runtime_cache"
    for key, sub in {
        "HF_HOME": "home",
        "HF_HUB_CACHE": "hub",
        "HUGGINGFACE_HUB_CACHE": "hub",
        "HF_DATASETS_CACHE": "datasets",
        "TRANSFORMERS_CACHE": "transformers",
        "HF_MODULES_CACHE": "modules",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        p = cache / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p.resolve())


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cache_env(out_dir)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    checkpoints = parse_checkpoint_args(args.checkpoint)
    radii = parse_radii(args.radii)
    corpus = Path(args.corpus)
    rows, sample_info = sample_rows(corpus, args.sample_rows, args.seed, args.max_scan_rows)

    first_ckpt = next(iter(checkpoints.values()))
    tokenizer = AutoTokenizer.from_pretrained(first_ckpt, trust_remote_code=True, local_files_only=True)
    items, freq, item_info = build_items(rows, tokenizer, seed=args.seed + 17, max_seq_len=args.max_seq_len, max_targets_per_row=args.max_targets_per_row)
    if not items:
        raise RuntimeError("no probe items built")

    contexts_by_kind: dict[str, list[dict[str, Any]]] = {"full": []}
    for r in radii:
        contexts_by_kind[f"local_r{r}"] = []
    for it in items:
        content_ids = it["content_ids"]
        pos = int(it["position"])
        tid = int(it["target_id"])
        ids, mi = make_context(content_ids, pos, tid, tokenizer, "full", None, args.max_seq_len)
        contexts_by_kind["full"].append({"input_ids": ids, "mask_index": mi, "target_id": tid, "item_id": it["item_id"]})
        for r in radii:
            ids, mi = make_context(content_ids, pos, tid, tokenizer, "local", r, args.max_seq_len)
            contexts_by_kind[f"local_r{r}"].append({"input_ids": ids, "mask_index": mi, "target_id": tid, "item_id": it["item_id"]})

    records: list[dict[str, Any]] = []
    per_ckpt_nll: dict[str, dict[str, list[float]]] = {}
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    for label, ckpt in checkpoints.items():
        t0 = time.time()
        tok2 = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
        if tok2.mask_token_id != tokenizer.mask_token_id or tok2.vocab_size != tokenizer.vocab_size:
            raise RuntimeError(f"tokenizer mismatch at {label}")
        model = AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
        model.to(device)
        model.eval()
        nlls: dict[str, list[float]] = {}
        for kind, contexts in contexts_by_kind.items():
            nlls[kind] = batch_nll(model, tokenizer, contexts, device=device, batch_size=args.batch_size)
        per_ckpt_nll[label] = nlls
        if device == "cuda":
            torch.cuda.empty_cache()
        del model
        ckpt_time = time.time() - t0
        for idx, it in enumerate(items):
            base = {
                k: v for k, v in it.items()
                if k != "content_ids"
            }
            base.update({
                "checkpoint": label,
                "full_nll": nlls["full"][idx],
                "log_sample_freq": math.log1p(freq[int(it["target_id"])]),
                "sample_freq": int(freq[int(it["target_id"])]),
                "checkpoint_elapsed_sec": ckpt_time,
            })
            for r in radii:
                local = nlls[f"local_r{r}"][idx]
                base[f"local_r{r}_nll"] = local
                base[f"ig_r{r}"] = local - nlls["full"][idx]
            records.append(base)

    # Add between-checkpoint deltas for same item if at least two checkpoints exist.
    labels = list(checkpoints)
    by_key = {(r["checkpoint"], r["item_id"]): r for r in records}
    if len(labels) >= 2:
        ref = labels[0]
        for label in labels[1:]:
            for it in items:
                a = by_key.get((ref, it["item_id"]))
                b = by_key.get((label, it["item_id"]))
                if a and b:
                    for target in (a, b):
                        target[f"full_nll_delta_{label}_minus_{ref}"] = b["full_nll"] - a["full_nll"]
                    for r in radii:
                        for target in (a, b):
                            target[f"ig_r{r}_delta_{label}_minus_{ref}"] = b[f"ig_r{r}"] - a[f"ig_r{r}"]

    # Summary by checkpoint/radius.
    summary: dict[str, Any] = {
        "status": "CONTEXT_DEPENDENCE_IG_PROBE",
        "created_utc": utc_now(),
        "corpus": rel(corpus),
        "corpus_sha256": sha256_file(corpus),
        "sample_info": sample_info,
        "item_info": {k: v for k, v in item_info.items() if k != "row_stats"},
        "checkpoints": {k: rel(v) for k, v in checkpoints.items()},
        "radii": radii,
        "device": device,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "summaries": {},
    }
    rows_by_ckpt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        rows_by_ckpt[str(r["checkpoint"])].append(r)
    for label, rs in rows_by_ckpt.items():
        s: dict[str, Any] = {
            "full_nll": summarize_values([float(r["full_nll"]) for r in rs]),
            "log_sample_freq": summarize_values([float(r["log_sample_freq"]) for r in rs]),
        }
        for rad in radii:
            ig = [float(r[f"ig_r{rad}"]) for r in rs]
            full = [float(r["full_nll"]) for r in rs]
            logfreq = [float(r["log_sample_freq"]) for r in rs]
            pos = [float(r["position_frac"]) for r in rs]
            dup = [float(r["target_context_count"]) for r in rs]
            ig_res = residual_against_one_covariate(ig, logfreq)
            nll_res = residual_against_one_covariate(full, logfreq)
            partial = pearson(ig_res, nll_res) if ig_res is not None and nll_res is not None else None
            s[f"radius_{rad}"] = {
                "ig": summarize_values(ig),
                "corr_ig_full_nll_pearson": pearson(ig, full),
                "corr_ig_full_nll_spearman": spearman(ig, full),
                "corr_ig_logfreq_pearson": pearson(ig, logfreq),
                "corr_full_nll_logfreq_pearson": pearson(full, logfreq),
                "partial_corr_ig_full_nll_control_logfreq": partial,
                "corr_ig_position_frac_pearson": pearson(ig, pos),
                "corr_ig_target_context_count_pearson": pearson(ig, dup),
                "mean_full_nll_high_ig_top20pct": None,
                "mean_full_nll_low_ig_bottom20pct": None,
            }
            order = sorted(range(len(ig)), key=lambda i: ig[i])
            q = max(1, len(order) // 5)
            low = order[:q]; high = order[-q:]
            s[f"radius_{rad}"]["mean_full_nll_high_ig_top20pct"] = float(statistics.fmean(full[i] for i in high))
            s[f"radius_{rad}"]["mean_full_nll_low_ig_bottom20pct"] = float(statistics.fmean(full[i] for i in low))
        summary["summaries"][label] = s

    # Between checkpoint relation: does IG at first checkpoint predict later full-NLL change?
    if len(labels) >= 2:
        ref = labels[0]
        for label in labels[1:]:
            ref_rows = [by_key[(ref, it["item_id"])] for it in items if (ref, it["item_id"]) in by_key and (label, it["item_id"]) in by_key]
            comp_rows = [by_key[(label, it["item_id"])] for it in items if (ref, it["item_id"]) in by_key and (label, it["item_id"]) in by_key]
            summary.setdefault("between_checkpoint", {})[f"{label}_minus_{ref}"] = {}
            delta = [float(b["full_nll"] - a["full_nll"]) for a, b in zip(ref_rows, comp_rows)]
            logfreq = [float(a["log_sample_freq"]) for a in ref_rows]
            for rad in radii:
                ig_ref = [float(a[f"ig_r{rad}"]) for a in ref_rows]
                ig_res = residual_against_one_covariate(ig_ref, logfreq)
                delta_res = residual_against_one_covariate(delta, logfreq)
                summary["between_checkpoint"][f"{label}_minus_{ref}"][f"radius_{rad}"] = {
                    "full_nll_delta": summarize_values(delta),
                    "corr_ref_ig_with_full_nll_delta_pearson": pearson(ig_ref, delta),
                    "corr_ref_ig_with_full_nll_delta_spearman": spearman(ig_ref, delta),
                    "partial_corr_ref_ig_with_delta_control_logfreq": pearson(ig_res, delta_res) if ig_res is not None and delta_res is not None else None,
                }

    # Write row-level CSV and JSON/MD summary.
    csv_path = out_dir / "context_dependence_items.csv"
    fieldnames = sorted({k for r in records for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)
    summary["csv"] = rel(csv_path)
    out_json = out_dir / "context_dependence_summary.json"
    out_md = out_dir / "context_dependence_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research context-dependence information-gain probe",
        "",
        "Forward-only masked-token NLL comparison on legal-corpus text. No training, upload, or leaderboard submission was performed.",
        "",
        f"Status: `{summary['status']}`",
        f"Corpus: `{summary['corpus']}`",
        f"Rows sampled: `{sample_info['sample_rows']}` from scan `{sample_info['scan_rows']}`; items `{item_info['n_items']}`",
        f"Checkpoints: `{summary['checkpoints']}`",
        f"Radii: `{radii}`; device `{device}`; max_seq_len `{args.max_seq_len}`",
        "",
        "## Per-checkpoint summaries",
    ]
    for label, s in summary["summaries"].items():
        lines.append(f"### {label}")
        lines.append(f"- full NLL mean/median: `{s['full_nll'].get('mean')}` / `{s['full_nll'].get('median')}`")
        for rad in radii:
            rr = s[f"radius_{rad}"]
            lines.append(
                f"- r={rad}: IG mean `{rr['ig'].get('mean')}`, corr IG~fullNLL pearson `{rr['corr_ig_full_nll_pearson']}`, "
                f"partial control logfreq `{rr['partial_corr_ig_full_nll_control_logfreq']}`, "
                f"highIG top20 fullNLL `{rr['mean_full_nll_high_ig_top20pct']}`, lowIG bottom20 fullNLL `{rr['mean_full_nll_low_ig_bottom20pct']}`"
            )
    if summary.get("between_checkpoint"):
        lines.append("")
        lines.append("## Between-checkpoint relation")
        for comp, ss in summary["between_checkpoint"].items():
            lines.append(f"### {comp}")
            for rad, rr in ss.items():
                lines.append(
                    f"- {rad}: mean full-NLL delta `{rr['full_nll_delta'].get('mean')}`, "
                    f"corr refIG~delta `{rr['corr_ref_ig_with_full_nll_delta_pearson']}`, "
                    f"partial control logfreq `{rr['partial_corr_ref_ig_with_delta_control_logfreq']}`"
                )
    lines.extend(["", f"CSV: `{rel(csv_path)}`", f"JSON: `{rel(out_json)}`"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "items": item_info["n_items"], "checkpoints": list(checkpoints), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--checkpoint", action="append", required=True, help="label=path; first checkpoint is the reference for between-checkpoint deltas")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--sample-rows", type=int, default=256)
    ap.add_argument("--max-scan-rows", type=int, default=None)
    ap.add_argument("--max-targets-per-row", type=int, default=4)
    ap.add_argument("--seed", type=int, default=16201)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--radii", default="4,8,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.sample_rows < 1 or args.max_targets_per_row < 1:
        raise ValueError("sample rows and targets must be positive")
    run(args)


if __name__ == "__main__":
    main()
