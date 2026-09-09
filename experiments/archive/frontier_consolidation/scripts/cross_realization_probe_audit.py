#!/usr/bin/env python3
"""research: audit cross-realization event balance and doc/pair-cluster robustness.

This is file-only. It reads research preflight events plus research event losses,
checks event construction/balance, and recomputes selected all-event contrasts with
both pair and document clustering. No model loading, no training, no official eval.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
EVENTS = ROOT / "data/cross_realization_probe_preflight/cross_realization_probe_preflight_events.jsonl"
LOSSES = {
    "full_compact_100M": ROOT / "data/cross_realization_probe_gpu0/full_compact_100M_event_losses.jsonl",
    "drop_abs_100M": ROOT / "data/cross_realization_probe_gpu0/drop_abs_100M_event_losses.jsonl",
    "drop_copied_word_100M": ROOT / "data/cross_realization_probe_gpu0/drop_copied_word_100M_event_losses.jsonl",
    "repeat_100M": ROOT / "data/cross_realization_probe_gpu1/repeat_100M_event_losses.jsonl",
    "adjbreak_100M": ROOT / "data/cross_realization_probe_gpu1/adjbreak_100M_event_losses.jsonl",
}
OUT_DEFAULT = ROOT / "data/cross_realization_probe_audit"
ARM_ORDER = ["full_compact_100M", "drop_abs_100M", "drop_copied_word_100M", "repeat_100M", "adjbreak_100M"]
CONTRASTS = [
    ("drop_abs_minus_repeat", "drop_abs_100M", "repeat_100M"),
    ("drop_copied_word_minus_repeat", "drop_copied_word_100M", "repeat_100M"),
    ("full_minus_drop_abs", "full_compact_100M", "drop_abs_100M"),
    ("full_minus_drop_copied_word", "full_compact_100M", "drop_copied_word_100M"),
    ("full_minus_repeat", "full_compact_100M", "repeat_100M"),
    ("adjbreak_minus_repeat", "adjbreak_100M", "repeat_100M"),
]
FIELDS = ["compact_context_advantage_nats", "rep_source_compact_advantage"]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        return xs[min(len(xs)-1, max(0, int(round(p*(len(xs)-1)))))]
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": sum(xs)/len(xs), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest()[:12], 16) % (2**31 - 1)


def boot(rows: list[dict[str, Any]], cluster_field: str, seed: int, n_boot: int) -> dict[str, Any]:
    byc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cid = str(r.get(cluster_field) or r.get("pair_id") or r.get("event_uid"))
        byc[cid].append(r)
    clusters = list(byc.items())
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = den = 0.0
        for _j in range(len(clusters)):
            _cid, rs = clusters[rng.randrange(len(clusters))]
            for r in rs:
                w = float(r["target_pieces"])
                num += float(r["diff"]) * w
                den += w
        vals.append(num / den if den else 0.0)
    vals.sort()
    def q(p: float) -> float:
        return vals[min(len(vals)-1, max(0, int(round(p*(len(vals)-1)))))]
    return {"cluster_field": cluster_field, "n_clusters": len(clusters), "n_boot": len(vals), "p025": q(0.025), "p50": q(0.5), "p975": q(0.975), "fraction_gt0": sum(x > 0 for x in vals)/len(vals), "fraction_lt0": sum(x < 0 for x in vals)/len(vals)}


def contrast_summary(event_index: dict[str, dict[str, Any]], loss_index: dict[str, dict[str, dict[str, Any]]], cname: str, a: str, b: str, field: str, cluster_fields: list[str], n_boot: int) -> dict[str, Any]:
    rows = []
    for uid, ev in event_index.items():
        if uid not in loss_index[a] or uid not in loss_index[b]:
            continue
        da = float(loss_index[a][uid][field])
        db = float(loss_index[b][uid][field])
        rows.append({
            "event_uid": uid,
            "pair_id": ev.get("pair_id"),
            "doc_id": ev.get("doc_id"),
            "eval_set": ev.get("eval_set"),
            "target_pieces": ev.get("target_pieces"),
            "diff": da - db,
        })
    den = sum(float(r["target_pieces"]) for r in rows)
    mean = sum(float(r["diff"]) * float(r["target_pieces"]) for r in rows) / den if den else None
    return {
        "contrast": cname,
        "field": field,
        "n_events": len(rows),
        "n_pairs": len({str(r.get("pair_id")) for r in rows}),
        "n_docs": len({str(r.get("doc_id")) for r in rows}),
        "target_pieces": int(den),
        "mean": mean,
        "by_cluster": {cf: boot(rows, cf, stable_seed(cname, field, cf), n_boot) for cf in cluster_fields},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(EVENTS)
    event_index = {str(e["event_uid"]): e for e in events}
    losses = {arm: read_jsonl(path) for arm, path in LOSSES.items()}
    loss_index = {arm: {str(r["event_uid"]): r for r in rows} for arm, rows in losses.items()}
    event_sets = {arm: set(idx) for arm, idx in loss_index.items()}
    common = set.intersection(*event_sets.values())

    diag_counts = collections.Counter()
    for e in events:
        for d in e.get("match_diagnostics", []):
            diag_counts[str(d)] += 1
    construction_checks = {
        "events_path": str(EVENTS),
        "events_sha256": sha256_file(EVENTS),
        "event_count": len(events),
        "unique_event_uid_count": len(event_index),
        "all_counterfactual_context_no_target_before_insert": all(bool(e.get("counterfactual_context_no_target_before_insert")) for e in events),
        "all_bpe_ids_identical_flag_present": diag_counts.get("bpe_ids_identical_across_source_compact_counterfactual", 0) == len(events),
        "all_target_absent_flag_present": diag_counts.get("target_absent_from_counterfactual_context_before_insert", 0) == len(events),
        "same_doc_rejected_flag_count": diag_counts.get("same_doc_rejected", 0),
        "diag_counts": dict(diag_counts),
        "eval_set_counts": dict(collections.Counter(str(e.get("eval_set")) for e in events)),
        "pair_count": len({str(e.get("pair_id")) for e in events}),
        "doc_count": len({str(e.get("doc_id")) for e in events}),
        "counterfactual_pair_count": len({str(e.get("counterfactual_pair_id")) for e in events}),
        "counterfactual_doc_count": len({str(e.get("counterfactual_doc_id")) for e in events}),
        "doc_overlap_target_vs_counterfactual_count": sum(1 for e in events if str(e.get("doc_id")) == str(e.get("counterfactual_doc_id"))),
        "pair_overlap_target_vs_counterfactual_count": sum(1 for e in events if str(e.get("pair_id")) == str(e.get("counterfactual_pair_id"))),
    }
    balance = {
        "compact_minus_counterfactual_context_tokens": stats([float(e["compact_context_tokens"]) - float(e["counterfactual_context_tokens"]) for e in events]),
        "source_minus_compact_context_tokens": stats([float(e["source_context_tokens"]) - float(e["compact_context_tokens"]) for e in events]),
        "compact_counterfactual_abs_relpos_delta": stats([abs(float(e["compact_rel_pos"]) - float(e["counterfactual_rel_pos"])) for e in events]),
        "compact_counterfactual_abs_word_index_delta": stats([abs(float(e["compact_word_index"]) - float(e["counterfactual_word_index"])) for e in events]),
        "target_pieces": stats([float(e["target_pieces"]) for e in events]),
        "counterfactual_match_score": stats([float(e.get("counterfactual_match_score", 0.0)) for e in events]),
    }
    by_eval_balance = {}
    for es in sorted({str(e.get("eval_set")) for e in events}):
        es_events = [e for e in events if str(e.get("eval_set")) == es]
        by_eval_balance[es] = {
            "n": len(es_events),
            "pair_count": len({str(e.get("pair_id")) for e in es_events}),
            "doc_count": len({str(e.get("doc_id")) for e in es_events}),
            "compact_minus_counterfactual_tokens_mean": stats([float(e["compact_context_tokens"]) - float(e["counterfactual_context_tokens"]) for e in es_events]).get("mean"),
            "compact_counterfactual_abs_relpos_delta_mean": stats([abs(float(e["compact_rel_pos"]) - float(e["counterfactual_rel_pos"])) for e in es_events]).get("mean"),
        }

    loss_checks = {
        "loss_paths": {arm: str(path) for arm, path in LOSSES.items()},
        "loss_sha256": {arm: sha256_file(path) for arm, path in LOSSES.items()},
        "loss_event_counts": {arm: len(rows) for arm, rows in losses.items()},
        "loss_unique_event_counts": {arm: len(set(loss_index[arm])) for arm in ARM_ORDER},
        "common_loss_event_count": len(common),
        "all_losses_cover_all_events": all(set(event_index) == event_sets[arm] for arm in ARM_ORDER),
    }
    robust_contrasts: dict[str, Any] = {}
    for cname, a, b in CONTRASTS:
        robust_contrasts[cname] = {field: contrast_summary(event_index, loss_index, cname, a, b, field, ["pair_id", "doc_id"], args.n_boot) for field in FIELDS}

    out = {
        "status": "CROSS_REALIZATION_PROBE_AUDIT",
        "meaning": "Construction/balance audit plus pair/doc-cluster robustness for research saved-checkpoint probe.",
        "construction_checks": construction_checks,
        "counterfactual_balance": balance,
        "by_eval_set_balance": by_eval_balance,
        "loss_checks": loss_checks,
        "robust_contrasts": robust_contrasts,
        "scientific_reading": "The audit can strengthen or weaken the local-mechanistic interpretation, but cannot turn local NLL/cosine into selected downstream evidence.",
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }
    out_json = out_dir / "cross_realization_probe_audit.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(s: dict[str, Any], cf: str) -> str:
        b = s["by_cluster"][cf]
        return f"mean={s['mean']:.6f}, {cf}95=[{b['p025']:.6f},{b['p975']:.6f}], P>0={b['fraction_gt0']:.3f}, clusters={b['n_clusters']}"
    md = [
        "# research cross-realization probe audit",
        "",
        f"Events: {len(events)}, pairs: {construction_checks['pair_count']}, docs: {construction_checks['doc_count']}",
        f"Losses cover all events: {loss_checks['all_losses_cover_all_events']}; common loss events: {loss_checks['common_loss_event_count']}",
        f"Counterfactual target-absence flag all true: {construction_checks['all_counterfactual_context_no_target_before_insert']}; BPE-identity flag all present: {construction_checks['all_bpe_ids_identical_flag_present']}",
        f"Target/counterfactual doc overlaps: {construction_checks['doc_overlap_target_vs_counterfactual_count']}; pair overlaps: {construction_checks['pair_overlap_target_vs_counterfactual_count']}",
        "",
        "## Balance",
        f"compact-counterfactual context tokens mean: {balance['compact_minus_counterfactual_context_tokens']['mean']:.6f}, p95 {balance['compact_minus_counterfactual_context_tokens']['p95']}, min/max {balance['compact_minus_counterfactual_context_tokens']['min']}/{balance['compact_minus_counterfactual_context_tokens']['max']}",
        f"abs compact-counterfactual relpos delta mean: {balance['compact_counterfactual_abs_relpos_delta']['mean']:.6f}, p95 {balance['compact_counterfactual_abs_relpos_delta']['p95']}",
        "",
        "## Robust all-event contrasts",
    ]
    for cname in ["drop_abs_minus_repeat", "drop_copied_word_minus_repeat", "full_minus_drop_abs", "full_minus_drop_copied_word", "full_minus_repeat", "adjbreak_minus_repeat"]:
        c = robust_contrasts[cname]
        md.append(f"- {cname} compact_adv pair: {fmt(c['compact_context_advantage_nats'], 'pair_id')}")
        md.append(f"  - {cname} compact_adv doc: {fmt(c['compact_context_advantage_nats'], 'doc_id')}")
        md.append(f"  - {cname} rep_A pair: {fmt(c['rep_source_compact_advantage'], 'pair_id')}")
        md.append(f"  - {cname} rep_A doc: {fmt(c['rep_source_compact_advantage'], 'doc_id')}")
    md.extend(["", "JSON: `" + str(out_json) + "`", ""])
    out_md = out_dir / "cross_realization_probe_audit.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "events": len(events), "no_training_official_eval_upload_aoa_or_leaderboard": True}, indent=2), flush=True)

if __name__ == "__main__":
    main()
