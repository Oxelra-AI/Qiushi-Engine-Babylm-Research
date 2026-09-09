#!/usr/bin/env python3
"""research: exact DeBERTa-loader realized-context audit for factorial view arms.

CPU-only.  Reconstructs the research changed block with research candidate views and
measures what the inherited DeBERTa MLM loader actually sees before any training:
row truncation, source/view co-visibility, WWM group/candidate mass, a CPU proxy of
the fixed-seed WWM mask stream for the changed block, and source-to-view token
distances.  This is a pre-training design audit, not model evaluation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoTokenizer

ROOT = Path("experiments/archive/frontier_consolidation")
PAIR_DIR = ROOT / "data/factorial_view_candidate_audit"
META_PATH = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
BASE_CHANGED_PATH = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOKENIZER_PATH = ROOT / "data/compliant_tokenizer"
DEFAULT_OUT = ROOT / "data/deberta_loader_realized_context_audit"
DEFAULT_VARIANTS = ["compact", "compact_scrambled", "sourcewide_onegap", "sourcewide_onegap_scrambled", "prefix_fluent", "prefix_scrambled"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path, limit: int | None = None) -> List[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def wc(text: str) -> int:
    return len(text.split())


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def build_groups(ids: List[int], tokenizer, special_ids: set[int]) -> List[int]:
    groups: List[int] = []
    gid = -1
    cache: Dict[int, bool] = {}
    for i, tid in enumerate(ids):
        if tid in special_ids:
            groups.append(-1)
            continue
        v = cache.get(tid)
        if v is None:
            s = tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            cache[tid] = v
        if gid < 0 or v or i == 0:
            gid += 1
        groups.append(gid)
    return groups


@dataclass
class Segment:
    pair_id: str | None
    segment: str
    char_start: int
    char_end: int


def locate_segment(offset: Tuple[int, int], segs: List[Segment]) -> Segment | None:
    a, b = offset
    mid = (a + b - 1) / 2.0 if b > a else a
    best: Segment | None = None
    best_ov = -1
    for s in segs:
        if s.char_start <= mid < s.char_end:
            return s
        ov = max(0, min(b, s.char_end) - max(a, s.char_start))
        if ov > best_ov:
            best_ov = ov
            best = s
    return best if best_ov > 0 else None


def reconstruct_row(meta: dict[str, Any], view_by_pid: Dict[str, str], source_by_pid: Dict[str, str], existing_changed: Dict[int, dict[str, Any]]) -> Tuple[str, List[Segment]]:
    pair_ids = [str(x) for x in meta.get("pair_ids", [])]
    row_index = int(meta.get("row_index", 0))
    if not pair_ids:
        ex = existing_changed.get(row_index)
        if ex is None:
            raise ValueError(f"missing existing top-up row {row_index}")
        t = str(ex.get("text", "")).strip()
        return t, [Segment(None, "other", 0, len(t))] if t else []
    parts: List[Tuple[str, str, str]] = []
    for pid in pair_ids:
        parts.append((pid, "source", source_by_pid[pid].strip()))
        parts.append((pid, "view", view_by_pid[pid].strip()))
    text_parts: List[str] = []
    segments: List[Segment] = []
    cur = 0
    for pid, seg, text in parts:
        if text_parts:
            cur += 1  # joining space
        start = cur
        end = start + len(text)
        segments.append(Segment(pid, seg, start, end))
        text_parts.append(text)
        cur = end
    text = " ".join(text_parts)
    return text, segments


def summarize_values(xs: Iterable[float]) -> dict[str, float | None]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    n = len(vals)
    def q(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = (n - 1) * p
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"n": n, "mean": sum(vals) / n, "p50": q(0.5), "p95": q(0.95), "max": vals[-1]}


def load_variant_pairs(variant: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    path = PAIR_DIR / f"{variant}_candidate_pairs.jsonl"
    view_by_pid: Dict[str, str] = {}
    source_by_pid: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r["pair_id"])
            view_by_pid[pid] = str(r["view_text"])
            source_by_pid[pid] = str(r["source_text"])
    return view_by_pid, source_by_pid


def row_analysis(meta_rows: List[dict[str, Any]], variant: str, tokenizer, seq_len: int, special_ids: set[int], existing_changed: Dict[int, dict[str, Any]]) -> dict[str, Any]:
    view_by_pid, source_by_pid = load_variant_pairs(variant)
    rows_for_mask: List[dict[str, Any]] = []
    total = collections.Counter()
    seg_raw = collections.Counter()
    seg_active = collections.Counter()
    seg_groups = collections.Counter()
    pair_visibility_counts = collections.Counter()
    pair_view_visible_fracs: List[float] = []
    pair_source_visible_fracs: List[float] = []
    source_to_view_min_gaps: List[float] = []
    row_records: List[dict[str, Any]] = []
    group_count_hist = collections.Counter()
    trunc_rows = 0
    word_mismatches = 0

    for row_pos, meta in enumerate(meta_rows):
        text, segments = reconstruct_row(meta, view_by_pid, source_by_pid, existing_changed)
        words = wc(text)
        if words != int(meta.get("words", -1)):
            word_mismatches += 1
        enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc.get("offset_mapping", [])]
        raw = len(ids)
        active_ids = ids[:seq_len]
        active = len(active_ids)
        trunc = max(0, raw - seq_len)
        if trunc:
            trunc_rows += 1
        groups = build_groups(active_ids, tokenizer, special_ids)
        valid_groups = sorted({g for g in groups if g >= 0})
        group_count_hist[len(valid_groups)] += 1

        # Token segment labels and pair-level visible positions.
        token_seg: List[str] = []
        token_pid: List[str | None] = []
        per_pair: Dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"source_raw": 0, "source_vis": 0, "view_raw": 0, "view_vis": 0, "source_pos": [], "view_pos": []})
        for ti, off in enumerate(offsets):
            seg = locate_segment(off, segments)
            sname = seg.segment if seg is not None else "unknown"
            pid = seg.pair_id if seg is not None else None
            if sname not in {"source", "view", "other"}:
                sname = "unknown"
            seg_raw[sname] += 1
            if ti < seq_len:
                seg_active[sname] += 1
            token_seg.append(sname)
            token_pid.append(pid)
            if pid is not None and sname in {"source", "view"}:
                per_pair[pid][f"{sname}_raw"] += 1
                if ti < seq_len:
                    per_pair[pid][f"{sname}_vis"] += 1
                    per_pair[pid][f"{sname}_pos"].append(ti)

        # Group segment by visible tokens in the group.
        group_seg: Dict[int, str] = {}
        group_pid: Dict[int, str | None] = {}
        group_token_count: Dict[int, int] = collections.Counter()
        for ti, g in enumerate(groups):
            if g < 0:
                continue
            group_token_count[g] += 1
            sname = token_seg[ti] if ti < len(token_seg) else "unknown"
            pid = token_pid[ti] if ti < len(token_pid) else None
            if g not in group_seg:
                group_seg[g] = sname
                group_pid[g] = pid
            elif group_seg[g] != sname or group_pid[g] != pid:
                group_seg[g] = "mixed"
                group_pid[g] = None
        for g, sname in group_seg.items():
            seg_groups[sname] += 1

        for pid, d in per_pair.items():
            src_raw, src_vis = d["source_raw"], d["source_vis"]
            view_raw, view_vis = d["view_raw"], d["view_vis"]
            if src_raw and view_raw and src_vis == src_raw and view_vis == view_raw:
                pair_visibility_counts["pair_full_visible"] += 1
            if src_raw and view_raw and src_vis and view_vis:
                pair_visibility_counts["pair_co_visible_any"] += 1
            if src_raw and view_raw and (src_vis < src_raw or view_vis < view_raw):
                pair_visibility_counts["pair_truncated_or_partial"] += 1
            if view_raw:
                pair_view_visible_fracs.append(view_vis / view_raw)
            if src_raw:
                pair_source_visible_fracs.append(src_vis / src_raw)
            if d["source_pos"] and d["view_pos"]:
                sp = d["source_pos"]
                for vp in d["view_pos"]:
                    source_to_view_min_gaps.append(min(abs(vp - s) for s in sp))

        total["rows"] += 1
        total["words"] += words
        total["raw_tokens"] += raw
        total["active_tokens"] += active
        total["truncated_tokens"] += trunc
        total["candidate_tokens"] += sum(1 for tid in active_ids if tid not in special_ids)
        total["candidate_groups"] += len(valid_groups)
        total["pair_units"] += len(per_pair)
        row_records.append({
            "row_pos": row_pos,
            "row_index": int(meta.get("row_index", row_pos)),
            "words": words,
            "raw_tokens": raw,
            "active_tokens": active,
            "truncated_tokens": trunc,
            "candidate_tokens": sum(1 for tid in active_ids if tid not in special_ids),
            "candidate_groups": len(valid_groups),
            "pair_count": len(per_pair),
            "source_active_tokens": int(sum(1 for i, s in enumerate(token_seg[:seq_len]) if s == "source")),
            "view_active_tokens": int(sum(1 for i, s in enumerate(token_seg[:seq_len]) if s == "view")),
        })
        rows_for_mask.append({
            "groups": groups,
            "group_token_count": group_token_count,
            "group_seg": group_seg,
            "input_len": active,
            "candidate_tokens": row_records[-1]["candidate_tokens"],
        })

    return {
        "variant": variant,
        "rows": rows_for_mask,
        "row_records": row_records,
        "summary": {
            **{k: int(v) for k, v in total.items()},
            "word_mismatches": word_mismatches,
            "truncated_rows": trunc_rows,
            "active_segment_tokens": dict(seg_active),
            "raw_segment_tokens": dict(seg_raw),
            "candidate_groups_by_segment": dict(seg_groups),
            "group_count_histogram_preview": dict(sorted(group_count_hist.items())[:20]),
            "pair_visibility_counts": dict(pair_visibility_counts),
            "pair_full_visible_fraction": pair_visibility_counts.get("pair_full_visible", 0) / max(1, total.get("pair_units", 0)),
            "pair_co_visible_any_fraction": pair_visibility_counts.get("pair_co_visible_any", 0) / max(1, total.get("pair_units", 0)),
            "view_visible_fraction_summary": summarize_values(pair_view_visible_fracs),
            "source_visible_fraction_summary": summarize_values(pair_source_visible_fracs),
            "source_to_view_min_token_gap_summary": summarize_values(source_to_view_min_gaps),
        },
    }


def simulate_cpu_wwm(rows: List[dict[str, Any]], seed: int, batch_size: int, seq_len: int, vocab_size: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    totals = collections.Counter()
    per_batch = []
    per_row = []
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        batch_selected = 0
        batch_seg = collections.Counter()
        selected_by_row: List[set[int]] = []
        for r in batch:
            valid_groups = sorted({g for g in r["groups"] if g >= 0})
            chosen: set[int] = set()
            if valid_groups:
                gp = torch.rand(len(valid_groups), generator=gen)
                for i, g in enumerate(valid_groups):
                    if float(gp[i]) < 0.15:
                        chosen.add(int(g))
            selected_by_row.append(chosen)
        # Fallback mirrors the trainer at batch level if no selected token anywhere.
        if not any(chosen for chosen in selected_by_row):
            for ridx, r in enumerate(batch):
                valid_groups = sorted({g for g in r["groups"] if g >= 0})
                if valid_groups:
                    selected_by_row[ridx].add(valid_groups[0])
                    break
        for local_i, (r, chosen) in enumerate(zip(batch, selected_by_row)):
            row_tokens = 0
            row_seg = collections.Counter()
            for g in chosen:
                mass = int(r["group_token_count"].get(g, 0))
                seg = str(r["group_seg"].get(g, "unknown"))
                row_tokens += mass
                row_seg[seg] += mass
            batch_selected += row_tokens
            batch_seg.update(row_seg)
            per_row.append({
                "row_pos": b0 + local_i,
                "masked_tokens_cpu_proxy": row_tokens,
                "masked_source_tokens_cpu_proxy": int(row_seg.get("source", 0)),
                "masked_view_tokens_cpu_proxy": int(row_seg.get("view", 0)),
                "masked_other_tokens_cpu_proxy": int(row_seg.get("other", 0) + row_seg.get("unknown", 0) + row_seg.get("mixed", 0)),
                "selected_groups_cpu_proxy": len(chosen),
            })
        totals["masked_tokens_cpu_proxy"] += batch_selected
        for k, v in batch_seg.items():
            totals[f"masked_{k}_tokens_cpu_proxy"] += int(v)
        # Advance generator for the 80/10/10 replacement draw and random-token ids,
        # because the real trainer uses the same generator for these operations.
        rmat = torch.rand(len(batch), seq_len, generator=gen)
        # Approximate random-token count from selected token positions by count only.
        # We cannot map selected token positions without storing full token masks here;
        # the draw advances by fixed batch*seq_len regardless, and randint advances by
        # the number of selected tokens whose replacement draw is in [0.8,0.9).  Use
        # the first selected-token-count draws from the flattened rmat only as a stable
        # CPU proxy.  This is sufficient for a design audit and is labelled as proxy.
        if batch_selected:
            flat = rmat.flatten()
            rand_tok_count = int(((flat[:batch_selected] >= 0.8) & (flat[:batch_selected] < 0.9)).sum().item())
            if rand_tok_count:
                _ = torch.randint(0, vocab_size, (rand_tok_count,), generator=gen)
        per_batch.append({
            "batch_index": b0 // batch_size,
            "rows": len(batch),
            "masked_tokens_cpu_proxy": int(batch_selected),
            **{f"masked_{k}_tokens_cpu_proxy": int(v) for k, v in batch_seg.items()},
        })
    return {"summary": {k: int(v) for k, v in totals.items()}, "per_batch": per_batch, "per_row": per_row}


def contrast_rows(row_records_a: List[dict[str, Any]], row_records_b: List[dict[str, Any]], mask_a: List[dict[str, Any]], mask_b: List[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    deltas = []
    for ra, rb, ma, mb in zip(row_records_a, row_records_b, mask_a, mask_b):
        d = {"row_pos": ra["row_pos"], "row_index": ra["row_index"]}
        for fld in ["raw_tokens", "active_tokens", "truncated_tokens", "candidate_tokens", "candidate_groups", "source_active_tokens", "view_active_tokens"]:
            d[f"delta_{fld}"] = float(ra[fld] - rb[fld])
        for fld in ["masked_tokens_cpu_proxy", "masked_source_tokens_cpu_proxy", "masked_view_tokens_cpu_proxy", "selected_groups_cpu_proxy"]:
            d[f"delta_{fld}"] = float(ma[fld] - mb[fld])
        deltas.append(d)
    def sum_field(fld: str) -> float:
        return float(sum(d[fld] for d in deltas))
    def count_nonzero(fld: str) -> int:
        return sum(1 for d in deltas if abs(d[fld]) > 1e-12)
    out = {"contrast": f"{a}_minus_{b}", "rows": len(deltas)}
    for fld in [k for k in deltas[0] if k.startswith("delta_")]:
        vals = [d[fld] for d in deltas]
        out[f"{fld}_sum"] = sum_field(fld)
        out[f"{fld}_mean"] = sum_field(fld) / max(1, len(deltas))
        out[f"{fld}_nonzero_rows"] = count_nonzero(fld)
        out[f"{fld}_max_abs"] = max(abs(v) for v in vals) if vals else 0.0
    return out


def write_csv(path: Path, rows: List[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt_i(x: Any) -> str:
    return f"{int(round(float(x))):,}"


def fmt_f(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.{nd}f}"


def markdown(result: dict[str, Any]) -> str:
    lines = []
    lines.append("# research exact DeBERTa-loader realized-context audit")
    lines.append("")
    lines.append("CPU/tokenizer audit of the research changed block reconstructed with research candidate views.  It measures the actual row-level DeBERTa MLM loader geometry and a CPU proxy of fixed-seed WWM masking; no model was loaded and no training was run.")
    lines.append("")
    lines.append("## Variant summaries")
    lines.append("| variant | rows | words | active tokens | trunc tokens | trunc rows | source/view active | candidate groups | pair full visible | mean source→view min gap | CPU masked tokens | CPU masked view/source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in result["variants"]:
        s = result["variant_summaries"][v]
        m = result["mask_proxy_summaries"][v]
        seg = s.get("active_segment_tokens", {})
        gap = s.get("source_to_view_min_token_gap_summary", {}).get("mean")
        lines.append("| {v} | {rows} | {words} | {active} | {trunc} | {trunc_rows} | {src}/{view} | {groups} | {pfv}% | {gap} | {masked} | {mview}/{msrc} |".format(
            v=v,
            rows=fmt_i(s["rows"]),
            words=fmt_i(s["words"]),
            active=fmt_i(s["active_tokens"]),
            trunc=fmt_i(s["truncated_tokens"]),
            trunc_rows=fmt_i(s["truncated_rows"]),
            src=fmt_i(seg.get("source", 0)),
            view=fmt_i(seg.get("view", 0)),
            groups=fmt_i(s["candidate_groups"]),
            pfv=fmt_f(100 * s["pair_full_visible_fraction"], 2),
            gap=fmt_f(gap, 2),
            masked=fmt_i(m.get("masked_tokens_cpu_proxy", 0)),
            mview=fmt_i(m.get("masked_view_tokens_cpu_proxy", 0)),
            msrc=fmt_i(m.get("masked_source_tokens_cpu_proxy", 0)),
        ))
    lines.append("")
    lines.append("## Fixed-word-multiset contrast equality")
    for name in ["compact_minus_compact_scrambled", "sourcewide_onegap_minus_sourcewide_onegap_scrambled", "prefix_fluent_minus_prefix_scrambled"]:
        c = result["contrasts"].get(name)
        if not c:
            continue
        lines.append(f"- `{name}`: Δ active {fmt_i(c['delta_active_tokens_sum'])}, Δ candidate groups {fmt_i(c['delta_candidate_groups_sum'])}, Δ view active {fmt_i(c['delta_view_active_tokens_sum'])}, Δ trunc tokens {fmt_i(c['delta_truncated_tokens_sum'])}, CPU-proxy Δ masked tokens {fmt_i(c['delta_masked_tokens_cpu_proxy_sum'])}, CPU-proxy Δ masked view/source {fmt_i(c['delta_masked_view_tokens_cpu_proxy_sum'])}/{fmt_i(c['delta_masked_source_tokens_cpu_proxy_sum'])}. Nonzero-row counts for active/group/mask: {c['delta_active_tokens_nonzero_rows']}/{c['delta_candidate_groups_nonzero_rows']}/{c['delta_masked_tokens_cpu_proxy_nonzero_rows']}.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("- The compact ordered/scrambled arm remains mechanically close under the actual row loader: legal words and pair positions are identical, active/candidate masses differ only at the token-level boundary induced by tokenization/truncation. The CPU mask proxy reveals the size of realized WWM-target drift that would remain even with identical legal words.")
    lines.append("- Sourcewide and prefix ordered/scrambled controls are similarly clean within each word multiset, but their cross-family comparison still changes copy composition, tail mass, and active target exposure; this audit does not make them pure tail-coverage or semantic-transformation tests.")
    lines.append("- Because this is changed-block-only and CPU-mask-proxy evidence, it supports design hygiene only. It must be combined with delivered DeBERTa grids and A01 compact directional evidence before deciding whether any H100 training screen is worth its cost.")
    lines.append("")
    lines.append(f"JSON: `{result['json_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mask-seed", type=int, default=43023)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    meta_rows = read_jsonl(META_PATH)
    existing_changed = {i: r for i, r in enumerate(read_jsonl(BASE_CHANGED_PATH, limit=len(meta_rows)))}

    variant_data: Dict[str, dict[str, Any]] = {}
    variant_summaries: Dict[str, dict[str, Any]] = {}
    mask_summaries: Dict[str, dict[str, Any]] = {}
    per_row_paths: Dict[str, str] = {}
    mask_row_paths: Dict[str, str] = {}

    for v in args.variants:
        analysis = row_analysis(meta_rows, v, tokenizer, args.seq_len, special_ids, existing_changed)
        mask = simulate_cpu_wwm(analysis["rows"], seed=args.mask_seed, batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=len(tokenizer))
        variant_data[v] = {"analysis": analysis, "mask": mask}
        variant_summaries[v] = analysis["summary"]
        mask_summaries[v] = mask["summary"]
        row_path = args.out_dir / f"{v}_row_loader_context.csv"
        mask_path = args.out_dir / f"{v}_row_mask_proxy.csv"
        write_csv(row_path, analysis["row_records"])
        write_csv(mask_path, mask["per_row"])
        per_row_paths[v] = str(row_path)
        mask_row_paths[v] = str(mask_path)

    contrast_pairs = [
        ("compact", "compact_scrambled"),
        ("sourcewide_onegap", "sourcewide_onegap_scrambled"),
        ("prefix_fluent", "prefix_scrambled"),
        ("compact", "sourcewide_onegap"),
        ("sourcewide_onegap", "prefix_fluent"),
    ]
    contrasts: Dict[str, dict[str, Any]] = {}
    for a, b in contrast_pairs:
        if a not in variant_data or b not in variant_data:
            continue
        c = contrast_rows(
            variant_data[a]["analysis"]["row_records"],
            variant_data[b]["analysis"]["row_records"],
            variant_data[a]["mask"]["per_row"],
            variant_data[b]["mask"]["per_row"],
            a,
            b,
        )
        contrasts[f"{a}_minus_{b}"] = c

    result = {
        "status": "DEBERTA_LOADER_REALIZED_CONTEXT_AUDIT",
        "scope": "changed_block_only_first_pass_cpu_mask_proxy",
        "variants": args.variants,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH / "tokenizer.json"),
        "meta_path": str(META_PATH),
        "meta_sha256": sha256_file(META_PATH),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "mask_seed": args.mask_seed,
        "variant_summaries": variant_summaries,
        "mask_proxy_summaries": mask_summaries,
        "contrasts": contrasts,
        "per_row_paths": per_row_paths,
        "mask_row_paths": mask_row_paths,
        "interpretation_boundary": "CPU/tokenizer design audit only; no model training or official evaluation was run.",
    }
    json_path = args.out_dir / "deberta_loader_realized_context_audit.json"
    result["json_path"] = str(json_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "deberta_loader_realized_context_audit.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "variants": args.variants}, indent=2))


if __name__ == "__main__":
    main()
