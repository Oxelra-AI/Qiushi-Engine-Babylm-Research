#!/usr/bin/env python3
"""research: saved-checkpoint cross-realization probe.

Evaluation-only mechanistic probe. For source-shared/copied content targets, mask
the identical BPE target under three realizations:
  source context, genuine compact rewrite context, matched counterfactual compact context.
Across existing checkpoints, measure target NLL and last-layer masked-target
representation agreement. No training, official evaluation, upload, AoA, or leaderboard action.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT_A02 = pathlib.Path("experiments/archive/frontier_consolidation")
EVENTS_DEFAULT = ROOT_A02 / "data/cross_realization_probe_preflight/cross_realization_probe_preflight_events.jsonl"
TOKENIZER_PATH = ROOT_A02 / "data/compliant_tokenizer"
OUT_DEFAULT = ROOT_A02 / "data/cross_realization_probe"

ARMS = {
    "full_compact_100M": pathlib.Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"),
    "drop_abs_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M"),
    "drop_copied_word_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M"),
    "repeat_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_100M"),
    "adjbreak_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2/hf_model/chck_100M"),
}

CONTEXTS = ("source", "compact", "counterfactual")


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_events(path: pathlib.Path, max_events: int | None = None) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if max_events is not None and len(out) >= max_events:
                break
    return out


def norm_space(text: str) -> str:
    return " ".join(str(text).split())


def encode_text(tok, text: str) -> list[int]:
    return tok(norm_space(text), add_special_tokens=False, truncation=True, max_length=256)["input_ids"]


def make_context_records(events: list[dict[str, Any]], tok) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for e in events:
        target_ids = [int(x) for x in e.get("target_ids", [])]
        if not target_ids:
            skipped.append({"event_uid": e.get("event_uid"), "reason": "missing_target_ids"})
            continue
        specs = [
            ("source", norm_space(e.get("source_text", "")), e.get("source_target_span")),
            ("compact", norm_space(e.get("compact_text", "")), e.get("compact_target_span")),
            ("counterfactual", norm_space(e.get("counterfactual_text_with_target", "")), e.get("counterfactual_target_span")),
        ]
        ok_records = []
        bad = None
        for ctype, text, span in specs:
            if not text or not isinstance(span, list) or len(span) != 2:
                bad = f"bad_{ctype}_text_or_span"
                break
            ids = encode_text(tok, text)
            st, en = int(span[0]), int(span[1])
            if st < 0 or en <= st or en > len(ids):
                bad = f"{ctype}_span_out_of_range"
                break
            if ids[st:en] != target_ids:
                bad = f"{ctype}_target_ids_mismatch"
                break
            ok_records.append({
                "event_uid": str(e["event_uid"]),
                "context_type": ctype,
                "text": text,
                "ids": ids,
                "target_span": [st, en],
                "target_ids": target_ids,
                "target_pieces": len(target_ids),
            })
        if bad:
            skipped.append({"event_uid": e.get("event_uid"), "reason": bad})
            continue
        contexts.extend(ok_records)
    return contexts, skipped


def batchify(records: list[dict[str, Any]], tok):
    max_len = max(len(r["ids"]) for r in records)
    pad_id = int(tok.pad_token_id)
    mask_id = int(tok.mask_token_id)
    input_ids = torch.full((len(records), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(records), max_len), -100, dtype=torch.long)
    for i, r in enumerate(records):
        ids = list(r["ids"])
        st, en = int(r["target_span"][0]), int(r["target_span"][1])
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, st:en] = torch.tensor(r["target_ids"], dtype=torch.long)
        input_ids[i, st:en] = mask_id
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, labels


def eval_one_arm(arm_name: str, model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    contexts, skipped = make_context_records(events, tok)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    by_uid: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            input_ids, attention_mask, labels = batchify(batch, tok)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
            hidden = outputs.hidden_states[-1]
            for b, rec in enumerate(batch):
                st, en = rec["target_span"]
                mask = labels[b] != -100
                vals = per_tok[b][mask]
                if vals.numel() == 0:
                    skipped.append({"event_uid": rec["event_uid"], "reason": f"zero_labels_{rec['context_type']}"})
                    continue
                h = hidden[b, st:en].mean(dim=0).detach().float().cpu()
                er = by_uid.setdefault(rec["event_uid"], {
                    "event_uid": rec["event_uid"],
                    "target_pieces": int(rec["target_pieces"]),
                    "contexts": {},
                    "_h": {},
                })
                er["contexts"][rec["context_type"]] = {
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "loss_mean": float(vals.mean().detach().cpu()),
                    "pieces": int(vals.numel()),
                    "context_tokens": int(attention_mask[b].sum().detach().cpu()),
                }
                er["_h"][rec["context_type"]] = h
    records: list[dict[str, Any]] = []
    incomplete = 0
    for e in events:
        uid = str(e.get("event_uid"))
        r = by_uid.get(uid)
        if not r or any(c not in r["contexts"] for c in CONTEXTS) or any(c not in r["_h"] for c in CONTEXTS):
            incomplete += 1
            continue
        hs = r.pop("_h")
        cos_sc = float(F.cosine_similarity(hs["source"], hs["compact"], dim=0).item())
        cos_scf = float(F.cosine_similarity(hs["source"], hs["counterfactual"], dim=0).item())
        ctx = r["contexts"]
        p = int(r["target_pieces"])
        rec = {
            **{k: e.get(k) for k in ["event_uid", "eval_set", "source_name", "pair_id", "key", "doc_id", "target_word", "target_norm"]},
            "target_pieces": p,
            "arm": arm_name,
            "loss_source_sum": ctx["source"]["loss_sum"],
            "loss_compact_sum": ctx["compact"]["loss_sum"],
            "loss_counterfactual_sum": ctx["counterfactual"]["loss_sum"],
            "loss_source_mean": ctx["source"]["loss_mean"],
            "loss_compact_mean": ctx["compact"]["loss_mean"],
            "loss_counterfactual_mean": ctx["counterfactual"]["loss_mean"],
            "compact_context_advantage_nats": (ctx["counterfactual"]["loss_sum"] - ctx["compact"]["loss_sum"]) / p,
            "source_context_advantage_nats": (ctx["counterfactual"]["loss_sum"] - ctx["source"]["loss_sum"]) / p,
            "compact_minus_source_loss_nats": (ctx["compact"]["loss_sum"] - ctx["source"]["loss_sum"]) / p,
            "rep_cos_source_compact": cos_sc,
            "rep_cos_source_counterfactual": cos_scf,
            "rep_source_compact_advantage": cos_sc - cos_scf,
            "source_context_tokens": ctx["source"]["context_tokens"],
            "compact_context_tokens": ctx["compact"]["context_tokens"],
            "counterfactual_context_tokens": ctx["counterfactual"]["context_tokens"],
        }
        records.append(rec)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    meta = {
        "arm": arm_name,
        "model_path": str(model_path),
        "loaded_context_records": len(contexts),
        "input_events": len(events),
        "complete_events": len(records),
        "incomplete_events": incomplete,
        "skipped_context_build": skipped[:50],
        "skipped_count": len(skipped),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    return records, meta


def cluster_bootstrap(rows: list[dict[str, Any]], field_num: str, field_den: str | None, cluster_field: str, seed: int, n_boot: int) -> dict[str, Any] | None:
    if not rows:
        return None
    by_cluster: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cid = str(r.get(cluster_field) or r.get("pair_id") or r.get("event_uid"))
        by_cluster[cid].append(r)
    clusters = list(by_cluster.items())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        num = 0.0
        den = 0.0
        for _j in range(len(clusters)):
            _cid, rs = clusters[rng.randrange(len(clusters))]
            for r in rs:
                if field_den is None:
                    num += float(r[field_num])
                    den += 1.0
                else:
                    num += float(r[field_num])
                    den += float(r[field_den])
        if den > 0:
            boots.append(num / den)
    boots.sort()
    def q(p: float):
        if not boots:
            return None
        return boots[min(len(boots) - 1, max(0, int(round(p * (len(boots) - 1)))))]
    return {
        "n_boot": len(boots),
        "n_clusters": len(clusters),
        "p025": q(0.025),
        "p50": q(0.5),
        "p975": q(0.975),
        "fraction_gt0": sum(1 for x in boots if x > 0) / len(boots) if boots else None,
    }


def summarize_records(records: list[dict[str, Any]], seed: int, n_boot: int) -> dict[str, Any]:
    groups: dict[str, Any] = {
        "all": lambda r: "all",
        "eval_set": lambda r: str(r.get("eval_set", "unknown")),
        "target_piece_count": lambda r: f"pieces_{r.get('target_pieces')}",
    }
    out = {}
    for gname, gfn in groups.items():
        out[gname] = {}
        buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in records:
            buckets[gfn(r)].append(r)
        for key, rows in sorted(buckets.items()):
            den = sum(float(r["target_pieces"]) for r in rows)
            def wmean(field: str):
                return sum(float(r[field]) * float(r["target_pieces"]) for r in rows) / den if den else None
            # For loss advantages, recompute from sums to avoid double weighting mistakes.
            compact_num = sum(float(r["loss_counterfactual_sum"]) - float(r["loss_compact_sum"]) for r in rows)
            source_num = sum(float(r["loss_counterfactual_sum"]) - float(r["loss_source_sum"]) for r in rows)
            comp_src_num = sum(float(r["loss_compact_sum"]) - float(r["loss_source_sum"]) for r in rows)
            for r in rows:
                r["_compact_adv_sum"] = float(r["loss_counterfactual_sum"]) - float(r["loss_compact_sum"])
                r["_source_adv_sum"] = float(r["loss_counterfactual_sum"]) - float(r["loss_source_sum"])
                r["_comp_minus_source_sum"] = float(r["loss_compact_sum"]) - float(r["loss_source_sum"])
                r["_rep_A_weighted"] = float(r["rep_source_compact_advantage"]) * float(r["target_pieces"])
            out[gname][key] = {
                "n_events": len(rows),
                "n_clusters_pair": len({str(r.get("pair_id")) for r in rows}),
                "target_pieces": int(den),
                "compact_context_advantage_nats": compact_num / den if den else None,
                "source_context_advantage_nats": source_num / den if den else None,
                "compact_minus_source_loss_nats": comp_src_num / den if den else None,
                "rep_source_compact_advantage": wmean("rep_source_compact_advantage"),
                "rep_cos_source_compact": wmean("rep_cos_source_compact"),
                "rep_cos_source_counterfactual": wmean("rep_cos_source_counterfactual"),
                "bootstrap": {
                    "compact_context_advantage_nats": cluster_bootstrap(rows, "_compact_adv_sum", "target_pieces", "pair_id", seed + 11, n_boot),
                    "source_context_advantage_nats": cluster_bootstrap(rows, "_source_adv_sum", "target_pieces", "pair_id", seed + 13, n_boot),
                    "compact_minus_source_loss_nats": cluster_bootstrap(rows, "_comp_minus_source_sum", "target_pieces", "pair_id", seed + 17, n_boot),
                    "rep_source_compact_advantage": cluster_bootstrap(rows, "_rep_A_weighted", "target_pieces", "pair_id", seed + 19, n_boot),
                },
            }
    return out


def make_arm_contrasts(summary_by_arm: dict[str, Any]) -> dict[str, Any]:
    # Use all/all group only for a compact top-level decision surface.
    vals = {}
    for arm, summ in summary_by_arm.items():
        vals[arm] = summ.get("all", {}).get("all", {})
    fields = [
        "compact_context_advantage_nats",
        "source_context_advantage_nats",
        "compact_minus_source_loss_nats",
        "rep_source_compact_advantage",
        "rep_cos_source_compact",
        "rep_cos_source_counterfactual",
    ]
    pairs = [
        ("full_minus_drop_abs", "full_compact_100M", "drop_abs_100M"),
        ("full_minus_drop_copied_word", "full_compact_100M", "drop_copied_word_100M"),
        ("drop_abs_minus_drop_copied_word", "drop_abs_100M", "drop_copied_word_100M"),
        ("drop_abs_minus_repeat", "drop_abs_100M", "repeat_100M"),
        ("full_minus_repeat", "full_compact_100M", "repeat_100M"),
        ("adjbreak_minus_repeat", "adjbreak_100M", "repeat_100M"),
        ("full_minus_adjbreak", "full_compact_100M", "adjbreak_100M"),
    ]
    out = {}
    for name, a, b in pairs:
        if a not in vals or b not in vals:
            continue
        out[name] = {}
        for f in fields:
            av = vals[a].get(f)
            bv = vals[b].get(f)
            if av is not None and bv is not None:
                out[name][f] = av - bv
    if all(k in vals for k in ["full_compact_100M", "drop_abs_100M", "drop_copied_word_100M", "repeat_100M", "adjbreak_100M"]):
        ca = {k: vals[k].get("compact_context_advantage_nats") for k in vals}
        ra = {k: vals[k].get("rep_source_compact_advantage") for k in vals}
        out["qualitative_reading_fields"] = {
            "compact_context_advantage_by_arm": ca,
            "rep_source_compact_advantage_by_arm": ra,
            "input_enrichment_pattern_to_check": "drop_abs retains a genuine compact-over-counterfactual advantage above repeat, adjbreak retains much of it, and full adds little beyond drop_abs",
            "joint_complementarity_pattern_to_check": "full_compact is uniquely above both deletion arms on compact_context_advantage and representation agreement, not merely equal to drop_abs or copied-supervised arms",
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(EVENTS_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-events", type=int, default=0, help="0 means all events")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=22243024)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; valid {sorted(ARMS)}")
    events = load_events(pathlib.Path(args.events), max_events=args.max_events or None)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)

    all_meta = []
    summary_by_arm = {}
    per_arm_paths = {}
    for arm in arm_names:
        records, meta = eval_one_arm(arm, ARMS[arm], events, tok, args.device, args.batch_size)
        rec_path = out_dir / f"{arm}_event_losses.jsonl"
        with rec_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stable_arm_seed = int(hashlib.sha256(arm.encode("utf-8")).hexdigest()[:8], 16) % 100000
        summary = summarize_records(records, seed=args.seed + stable_arm_seed, n_boot=args.n_boot)
        summ_path = out_dir / f"{arm}_summary.json"
        with summ_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        meta["event_losses_path"] = str(rec_path)
        meta["event_losses_sha256"] = sha256_file(rec_path)
        meta["summary_path"] = str(summ_path)
        all_meta.append(meta)
        summary_by_arm[arm] = summary
        per_arm_paths[arm] = {"event_losses": str(rec_path), "summary": str(summ_path)}
        print(json.dumps({"event": "arm_done", "arm": arm, "complete_events": len(records), "elapsed_sec": meta["elapsed_sec"]}, indent=2), flush=True)

    integrated = {
        "status": "CROSS_REALIZATION_PROBE",
        "meaning": "Saved-checkpoint no-training probe of whether genuine compact context and source context give a cross-realization advantage for copied/shared content targets relative to matched counterfactual compact contexts.",
        "events_path": str(args.events),
        "events_sha256": sha256_file(pathlib.Path(args.events)),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_json_sha256": sha256_file(TOKENIZER_PATH / "tokenizer.json"),
        "arms": {a: str(ARMS[a]) for a in arm_names},
        "device": args.device,
        "batch_size": args.batch_size,
        "input_events": len(events),
        "arm_meta": all_meta,
        "per_arm_paths": per_arm_paths,
        "summary_by_arm_all": {a: summary_by_arm[a].get("all", {}).get("all", {}) for a in arm_names},
        "arm_contrasts_all": make_arm_contrasts(summary_by_arm),
        "interpretation_contract": {
            "positive_compact_context_advantage_nats": "counterfactual target NLL minus genuine compact-context target NLL; positive means genuine compact context predicts the copied/shared target better than matched counterfactual compact context.",
            "positive_rep_source_compact_advantage": "cos(masked-target hidden under source, compact) minus cos(source, counterfactual); positive means source and genuine compact target representations align more than source and counterfactual.",
            "input_exposure_contextual_enrichment_signature": "drop_abs keeps compact_context_advantage and representation agreement above repeat despite no source-absent target supervision; adjbreak retains much; full adds little beyond drop_abs.",
            "joint_target_complementarity_signature": "full_compact uniquely exceeds both deletion arms on compact advantage/representation, not merely both copied-supervised models.",
            "not_downstream_evidence": "This probe is mechanistic inference only. Any new training or endpoint claim still requires official-compatible selected scores on stable families.",
        },
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }
    integ_path = out_dir / "cross_realization_probe_integrated.json"
    with integ_path.open("w", encoding="utf-8") as f:
        json.dump(integrated, f, indent=2, ensure_ascii=False)
    md = [
        "# research cross-realization saved-checkpoint probe",
        "",
        f"Input events: {len(events)}",
        f"Arms: {', '.join(arm_names)}",
        "",
        "## All-event summaries",
    ]
    for arm in arm_names:
        s = integrated["summary_by_arm_all"][arm]
        md.append(f"- {arm}: compact_adv={s.get('compact_context_advantage_nats'):.6f}, source_adv={s.get('source_context_advantage_nats'):.6f}, compact_minus_source_loss={s.get('compact_minus_source_loss_nats'):.6f}, rep_A={s.get('rep_source_compact_advantage'):.6f}, n={s.get('n_events')}")
    md.extend(["", "## Arm contrasts (all events)"])
    for name, vals in integrated["arm_contrasts_all"].items():
        if not isinstance(vals, dict) or name == "qualitative_reading_fields":
            continue
        md.append(f"- {name}: " + ", ".join(f"{k}={v:.6f}" for k, v in vals.items()))
    md.extend(["", "JSON: `" + str(integ_path) + "`", ""])
    with (out_dir / "cross_realization_probe_integrated.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(json.dumps({
        "status": integrated["status"],
        "out_json": str(integ_path),
        "arms": arm_names,
        "input_events": len(events),
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
