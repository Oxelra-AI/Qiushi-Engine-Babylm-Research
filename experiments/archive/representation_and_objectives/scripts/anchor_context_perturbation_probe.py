#!/usr/bin/env python3
"""research: context-relevance probe for compact-view coupling.

Question
--------
research rejected direct source-absent TARGET LABEL mediation at the official
surface: deleting source-absent compact-content labels did not hurt relative to
deleting matched copied-content labels.  However, all target-selective arms still
SAW the intact compact input.  This probe asks whether source-absent compact words
serve as visible context that conditions predictions of copied / retained content
anchors.

For retained-content target events in compact rewrites, evaluate target loss under:
  intact:              only the retained target word is masked
  mask_abs_context:    target masked + all other source-absent content words in
                       the compact rewrite are masked as visible context
  mask_function_match: target masked + same BPE mass of function/other compact
                       words masked (low-level extra-mask control)
  mask_copied_match:   target masked + same BPE mass of other copied-content
                       compact words masked (anchor/copy-context control)

Labels are only on the retained target span.  Positive sensitivity = perturbing
that context raises the target loss, i.e. the model used that context for anchor
prediction.

This is an evaluation-only saved-model analysis; no training.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
TRAIN_FIXED_EVENTS = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
SOURCE_DISJOINT_EVENTS = ROOT / "data/source_disjoint_target_probe/source_disjoint_probe_events.jsonl"
OUT_DIR = ROOT / "data/anchor_context_perturbation_probe"
SEQ_LEN = 256

ARMS = {
    "full_compact_100M": pathlib.Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"),
    "drop_abs_100M": ROOT / "training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M",
    "drop_copied_word_100M": ROOT / "training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M",
}

FUNC_WORDS = frozenset("""
a an the this that these those my your his her its our their is am are was were be been being
have has had do does did will would shall should can could may might must need ought to of in on
at by for with from into through during before after above below between under over about against
along across around behind beside beyond down near off since toward upon within without and but or
nor so yet both either neither not no if then else than as which who whom whose what when where how
while until because although though even also just only still already very much more most less least
too quite really rather such same other another each every all some any few many i me we us you he
him she her it they them one ones there here up out away again back now however therefore thus hence
moreover furthermore nevertheless meanwhile otherwise instead indeed perhaps maybe probably
""".split())
RELATIONAL_WORDS = set("""
able about across action active activity actor actors actually added adds affects after allows also among another appears area areas around associated because become becomes became before being between based called cause caused causes change changes changing common commonly compared connected consists contain contained contains created creates defined described different during effects either especially events example examples formed found from function functions general generally gives group groups having including indicates involves known leads made makes means method methods more most occurs often part parts process processes produces provides related relation relations relationship result results same separate shows similar since specific structure system systems through used using usually while within without works
""".split())
GENERIC_ENTITY_WORDS = set("""
animal animals area areas body building buildings city cities country countries culture cultures family families group groups language languages material materials member members people person place places plant plants region regions species state states system systems thing things time times water work works world year years
""".split())
EVENT_STATE_WORDS = set("""
add added adds become becomes became broken built carried changed changes changing closed created creates damaged developed discovered done drop dropped dry eaten fall falls fell filled found frozen gave gives got grow grows has held hidden increased left located made make makes moved moves named opened placed produced removed said sent set shown stuck taken turn turned used went won wrote
""".split())
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def norm(w: str) -> str:
    m = WORD_RE.search(w)
    return m.group(0).lower() if m else re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return bool(n) and n not in FUNC_WORDS and len(n) > 1


def word_flags(surface: str) -> set[str]:
    nw = norm(surface)
    fs: set[str] = set()
    if any(c.isdigit() for c in surface):
        fs.add("number_or_year")
    if surface[:1].isupper() and not surface.isupper():
        fs.add("capitalized")
    if "-" in surface:
        fs.add("hyphenated")
    if nw in RELATIONAL_WORDS:
        fs.add("relational_or_abstracting_word")
    if nw in GENERIC_ENTITY_WORDS:
        fs.add("generic_entity_word")
    if nw in EVENT_STATE_WORDS:
        fs.add("event_or_state_word")
    if len(nw) <= 3:
        fs.add("short_norm")
    if not fs:
        fs.add("other_content")
    return fs


def rel_event(surface: str) -> bool:
    fs = word_flags(surface)
    return bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_count(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


def target_span(tok, source_text: str, side_words: list[str], word_i: int) -> tuple[int, int] | None:
    before_side = " ".join(side_words[:word_i])
    through_side = " ".join(side_words[: word_i + 1])
    prefix_before = source_text if not before_side else source_text + " " + before_side
    prefix_after = source_text + " " + through_side
    st = token_count(tok, prefix_before)
    en = token_count(tok, prefix_after)
    if st < en and st < SEQ_LEN:
        return st, min(en, SEQ_LEN)
    return None


def load_events(path: pathlib.Path, source_name: str, keep_sets: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if str(d.get("category")) != "retained_content":
                continue
            eval_set = str(d.get("eval_set", source_name))
            if keep_sets is not None and eval_set not in keep_sets:
                continue
            d = dict(d)
            d.setdefault("eval_set", eval_set)
            d.setdefault("event_uid", f"{source_name}|{i}|{d.get('pair_id')}|wi{d.get('word_index')}|retained_content")
            d["source_name"] = source_name
            rows.append(d)
    return rows


def exact_or_greedy_subset(spans: list[dict[str, Any]], target_pieces: int) -> tuple[list[dict[str, Any]], bool]:
    """Return subset whose total pieces equals target when feasible, else a close greedy subset <= target."""
    if target_pieces <= 0 or not spans:
        return [], target_pieces == 0
    # DP exact subset up to moderate target; spans are tiny in this probe.
    dp: dict[int, list[int]] = {0: []}
    for i, sp in enumerate(spans):
        w = int(sp["pieces"])
        if w <= 0 or w > target_pieces:
            continue
        for s, inds in list(dp.items())[::-1]:
            ns = s + w
            if ns > target_pieces or ns in dp:
                continue
            dp[ns] = inds + [i]
            if ns == target_pieces:
                return [spans[j] for j in dp[ns]], True
    if target_pieces in dp:
        return [spans[j] for j in dp[target_pieces]], True
    # Greedy close under target, prefer spans near target but not overlapping already.
    selected: list[dict[str, Any]] = []
    total = 0
    for sp in sorted(spans, key=lambda x: (-int(x["pieces"]), abs(float(x.get("rel_pos", 0.5)) - 0.5))):
        w = int(sp["pieces"])
        if total + w <= target_pieces:
            selected.append(sp)
            total += w
        if total == target_pieces:
            break
    return selected, total == target_pieces


def build_prepared_events(tok, max_per_eval_set: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_sets = {"source_disjoint_quality", "doc_disjoint_all_accepted", "doc_disjoint_quality"}
    raw = load_events(TRAIN_FIXED_EVENTS, "train_fixed_probe")
    raw.extend(load_events(SOURCE_DISJOINT_EVENTS, "source_disjoint", keep_sets=keep_sets))
    # deterministic per-eval-set cap after shuffling to avoid only early rows if requested
    rng = random.Random(seed)
    by_set: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in raw:
        by_set[str(e.get("eval_set", "unknown"))].append(e)
    capped: list[dict[str, Any]] = []
    for es, rows in sorted(by_set.items()):
        rng.shuffle(rows)
        capped.extend(rows if max_per_eval_set is None else rows[:max_per_eval_set])

    prepared: list[dict[str, Any]] = []
    skip = collections.Counter()
    meta_counts = collections.Counter()
    for e in capped:
        src = str(e.get("source_text", ""))
        side = str(e.get("side_text", e.get("rewrite_text", "")))
        if not src or not side:
            skip["missing_text"] += 1
            continue
        full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
        span = e.get("span")
        if not isinstance(span, list) or len(span) != 2:
            skip["bad_target_span"] += 1
            continue
        target_st, target_en = int(span[0]), int(span[1])
        target_en = min(target_en, len(full_ids), SEQ_LEN)
        if target_st < 0 or target_st >= target_en or target_en > len(full_ids):
            skip["target_out_of_range"] += 1
            continue
        side_words = side.split()
        source_norms = set(norm(w) for w in src.split() if norm(w))
        abs_spans: list[dict[str, Any]] = []
        copied_spans: list[dict[str, Any]] = []
        function_spans: list[dict[str, Any]] = []
        abs_flags = collections.Counter()
        for wi, w in enumerate(side_words):
            sp = target_span(tok, src, side_words, wi)
            if sp is None:
                continue
            st, en = sp
            if st >= en or en > len(full_ids):
                continue
            # Never perturb the target itself.
            if not (en <= target_st or st >= target_en):
                continue
            nw = norm(w)
            pieces = en - st
            rec = {"word_index": wi, "word": w, "span": [st, en], "pieces": pieces,
                   "rel_pos": wi / max(1, len(side_words) - 1), "flags": sorted(word_flags(w))}
            if is_content(w) and nw not in source_norms:
                abs_spans.append(rec)
                for fl in rec["flags"]:
                    abs_flags[fl] += 1
            elif is_content(w) and nw in source_norms:
                copied_spans.append(rec)
            else:
                function_spans.append(rec)
        if not abs_spans:
            skip["no_abs_context"] += 1
            continue
        abs_pieces = sum(int(s["pieces"]) for s in abs_spans)
        fn_sel, fn_exact = exact_or_greedy_subset(function_spans, abs_pieces)
        cp_sel, cp_exact = exact_or_greedy_subset(copied_spans, abs_pieces)
        if not fn_sel:
            skip["no_function_control"] += 1
            continue
        # copied control is scientifically useful but not required for the event to be included.
        pe = dict(e)
        pe.update({
            "input_ids": full_ids,
            "target_span": [target_st, target_en],
            "target_pieces": target_en - target_st,
            "target_word": str(e.get("word", "")),
            "target_rel_event": rel_event(str(e.get("word", ""))),
            "target_flags": sorted(word_flags(str(e.get("word", "")))),
            "context_abs_spans": abs_spans,
            "context_function_match_spans": fn_sel,
            "context_copied_match_spans": cp_sel,
            "function_match_exact": bool(fn_exact),
            "copied_match_exact": bool(cp_exact and cp_sel),
            "context_abs_pieces": int(abs_pieces),
            "function_match_pieces": int(sum(int(s["pieces"]) for s in fn_sel)),
            "copied_match_pieces": int(sum(int(s["pieces"]) for s in cp_sel)),
            "context_abs_words": len(abs_spans),
            "context_abs_rel_event_words": int(sum(1 for s in abs_spans if {"relational_or_abstracting_word", "event_or_state_word"} & set(s["flags"]))),
            "context_abs_capnum_words": int(sum(1 for s in abs_spans if {"capitalized", "number_or_year"} & set(s["flags"]))),
            "context_abs_flags": dict(abs_flags),
        })
        pe["context_group"] = "abs_context_has_rel_event" if pe["context_abs_rel_event_words"] > 0 else "abs_context_no_rel_event"
        pe["context_capnum_group"] = "abs_context_has_capnum" if pe["context_abs_capnum_words"] > 0 else "abs_context_no_capnum"
        prepared.append(pe)
        meta_counts[str(pe.get("eval_set", "unknown"))] += 1
    meta = {
        "raw_retained_events": len(raw),
        "after_cap": len(capped),
        "prepared_events": len(prepared),
        "skips": dict(skip),
        "by_eval_set": dict(meta_counts),
        "max_per_eval_set": max_per_eval_set,
        "seed": seed,
    }
    return prepared, meta


def apply_spans(ids: list[int], spans: list[dict[str, Any]], mask_id: int) -> list[int]:
    x = list(ids)
    for sp in spans:
        st, en = int(sp["span"][0]), int(sp["span"][1])
        st = max(0, min(st, len(x)))
        en = max(st, min(en, len(x)))
        for j in range(st, en):
            x[j] = mask_id
    return x


def make_batch(events: list[dict[str, Any]], variant: str, tok) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    max_len = min(SEQ_LEN, max(len(e["input_ids"]) for e in events))
    input_batch = torch.full((len(events), max_len), int(tok.pad_token_id), dtype=torch.long)
    labels = torch.full((len(events), max_len), -100, dtype=torch.long)
    kept: list[dict[str, Any]] = []
    for i, e in enumerate(events):
        ids = list(e["input_ids"][:max_len])
        st, en = int(e["target_span"][0]), int(e["target_span"][1])
        en = min(en, len(ids), max_len)
        if st >= en:
            continue
        if variant == "intact":
            pert_spans: list[dict[str, Any]] = []
        elif variant == "mask_abs_context":
            pert_spans = e["context_abs_spans"]
        elif variant == "mask_function_match":
            pert_spans = e["context_function_match_spans"]
        elif variant == "mask_copied_match":
            pert_spans = e["context_copied_match_spans"]
        else:
            raise ValueError(variant)
        ids = apply_spans(ids, pert_spans, int(tok.mask_token_id))
        input_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, st:en] = torch.tensor(e["input_ids"][st:en], dtype=torch.long)
        input_batch[i, st:en] = int(tok.mask_token_id)
        kept.append(e)
    return input_batch, labels, kept


def eval_arm(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> list[dict[str, Any]]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    variants = ["intact", "mask_abs_context", "mask_function_match", "mask_copied_match"]
    # Accumulate per event uid then write wide records.
    acc: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for variant in variants:
            # For copied variant, skip events without any copied matched pieces.
            evs = [e for e in events if variant != "mask_copied_match" or e.get("copied_match_pieces", 0) > 0]
            for i in range(0, len(evs), batch_size):
                batch = evs[i : i + batch_size]
                input_ids, labels, kept = make_batch(batch, variant, tok)
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attn = (input_ids != int(tok.pad_token_id)).long().to(device)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                vocab = logits.shape[-1]
                per_tok = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
                mask = labels != -100
                for b, e in enumerate(kept):
                    vals = per_tok[b][mask[b]]
                    if vals.numel() == 0:
                        continue
                    uid = str(e["event_uid"])
                    rec = acc.setdefault(uid, {
                        "event_uid": uid,
                        "eval_set": str(e.get("eval_set", "unknown")),
                        "source_name": str(e.get("source_name", "unknown")),
                        "pair_id": str(e.get("pair_id", "unknown")),
                        "target_word": str(e.get("target_word", "")),
                        "target_pieces": int(e.get("target_pieces", vals.numel())),
                        "target_rel_event": bool(e.get("target_rel_event", False)),
                        "target_flags": e.get("target_flags", []),
                        "context_group": str(e.get("context_group", "unknown")),
                        "context_capnum_group": str(e.get("context_capnum_group", "unknown")),
                        "context_abs_pieces": int(e.get("context_abs_pieces", 0)),
                        "context_abs_words": int(e.get("context_abs_words", 0)),
                        "context_abs_rel_event_words": int(e.get("context_abs_rel_event_words", 0)),
                        "context_abs_capnum_words": int(e.get("context_abs_capnum_words", 0)),
                        "function_match_pieces": int(e.get("function_match_pieces", 0)),
                        "copied_match_pieces": int(e.get("copied_match_pieces", 0)),
                        "function_match_exact": bool(e.get("function_match_exact", False)),
                        "copied_match_exact": bool(e.get("copied_match_exact", False)),
                        "losses": {},
                    })
                    rec["losses"][variant] = {
                        "loss_sum": float(vals.sum().detach().cpu()),
                        "pieces": int(vals.numel()),
                        "loss_mean": float(vals.mean().detach().cpu()),
                    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return list(acc.values())


def cluster_bootstrap(vals: list[tuple[str, float, int]], seed: int, n_boot: int) -> dict[str, Any] | None:
    # vals: (cluster_id, delta_loss_sum, target_pieces)
    if not vals:
        return None
    by_cluster: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for cid, d, p in vals:
        by_cluster[cid].append((float(d), int(p)))
    clusters = list(by_cluster.items())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        num = 0.0
        den = 0
        for _j in range(len(clusters)):
            _cid, rows = clusters[rng.randrange(len(clusters))]
            for d, p in rows:
                num += d
                den += p
        if den:
            boots.append(num / den)
    boots.sort()
    def q(p: float) -> float | None:
        if not boots:
            return None
        idx = min(len(boots) - 1, max(0, int(round(p * (len(boots) - 1)))))
        return round(boots[idx], 6)
    return {
        "n_boot": len(boots),
        "n_clusters": len(clusters),
        "p025": q(0.025),
        "p50": q(0.5),
        "p975": q(0.975),
        "fraction_gt0": round(sum(1 for x in boots if x > 0) / len(boots), 6) if boots else None,
    }


def summarize_arm_records(records: list[dict[str, Any]], seed: int, n_boot: int) -> dict[str, Any]:
    variants = [
        ("abs_sensitivity", "mask_abs_context", "intact"),
        ("function_sensitivity", "mask_function_match", "intact"),
        ("copied_sensitivity", "mask_copied_match", "intact"),
        ("abs_minus_function_sensitivity", "mask_abs_context", "mask_function_match"),
        ("abs_minus_copied_sensitivity", "mask_abs_context", "mask_copied_match"),
    ]
    groupers = {
        "all": lambda r: "all",
        "eval_set": lambda r: str(r.get("eval_set", "unknown")),
        "context_group": lambda r: str(r.get("context_group", "unknown")),
        "context_capnum_group": lambda r: str(r.get("context_capnum_group", "unknown")),
        "target_rel_event": lambda r: "target_rel_event" if r.get("target_rel_event") else "target_not_rel_event",
        "exact_function_match": lambda r: "function_exact" if r.get("function_match_exact") else "function_inexact",
        "exact_copied_match": lambda r: "copied_exact" if r.get("copied_match_exact") else "copied_inexact_or_missing",
    }
    out: dict[str, Any] = {}
    for contrast_name, a, b in variants:
        out[contrast_name] = {}
        for gname, fn in groupers.items():
            buckets: dict[str, list[tuple[str, float, int]]] = collections.defaultdict(list)
            for r in records:
                losses = r.get("losses", {})
                if a not in losses or b not in losses:
                    continue
                # Require exact matching for matched-control contrasts in their intended readings.
                if "function" in contrast_name and not bool(r.get("function_match_exact")):
                    continue
                if "copied" in contrast_name and not bool(r.get("copied_match_exact")):
                    continue
                pieces = int(losses[a]["pieces"])
                if pieces <= 0:
                    continue
                delta_sum = float(losses[a]["loss_sum"]) - float(losses[b]["loss_sum"])
                buckets[fn(r)].append((str(r.get("pair_id", r.get("event_uid"))), delta_sum, pieces))
            out[contrast_name][gname] = {}
            for key, vals in sorted(buckets.items()):
                den = sum(p for _, _, p in vals)
                num = sum(d for _, d, _ in vals)
                if not den:
                    continue
                out[contrast_name][gname][key] = {
                    "n_events": len(vals),
                    "n_pairs": len(set(cid for cid, _, _ in vals)),
                    "target_pieces": int(den),
                    "piece_weighted_delta_nats": round(num / den, 6),
                    "bootstrap": cluster_bootstrap(vals, seed + abs(hash((contrast_name, gname, key))) % 1000000, n_boot),
                }
    # Also summarize raw intact losses.
    raw_loss: dict[str, list[tuple[str, float, int]]] = collections.defaultdict(list)
    for r in records:
        losses = r.get("losses", {})
        if "intact" in losses:
            p = int(losses["intact"]["pieces"])
            raw_loss[str(r.get("eval_set", "unknown"))].append((str(r.get("pair_id")), float(losses["intact"]["loss_sum"]), p))
    out["intact_losses_by_eval_set"] = {}
    for es, vals in sorted(raw_loss.items()):
        den = sum(p for _, _, p in vals)
        num = sum(d for _, d, _ in vals)
        out["intact_losses_by_eval_set"][es] = {
            "n_events": len(vals),
            "n_pairs": len(set(cid for cid, _, _ in vals)),
            "target_pieces": int(den),
            "piece_weighted_loss": round(num / den, 6) if den else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_per_eval_set", type=int, default=1536, help="cap retained-anchor events per eval set; 0 means no cap")
    ap.add_argument("--n_boot", type=int, default=600)
    ap.add_argument("--seed", type=int, default=23643023)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "anchor_context_perturbation_probe.json"
    if out_json.exists() and not args.force:
        print(json.dumps({"status": "exists", "out": str(out_json)}))
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    max_per = None if args.max_per_eval_set == 0 else int(args.max_per_eval_set)
    events, event_meta = build_prepared_events(tok, max_per, args.seed)
    event_path = out_dir / "anchor_context_events.jsonl"
    with event_path.open("w", encoding="utf-8") as f:
        for e in events:
            safe = {k: v for k, v in e.items() if k != "input_ids"}
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")

    arm_records: dict[str, list[dict[str, Any]]] = {}
    arm_summary: dict[str, Any] = {}
    for arm, path in ARMS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing model for {arm}: {path}")
        print(json.dumps({"event": "eval_start", "arm": arm, "model": str(path), "n_events": len(events)}), flush=True)
        recs = eval_arm(path, events, tok, args.device, args.batch_size)
        arm_records[arm] = recs
        rec_path = out_dir / f"{arm}_anchor_context_losses.jsonl"
        with rec_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        arm_summary[arm] = summarize_arm_records(recs, args.seed, args.n_boot)
        print(json.dumps({"event": "eval_done", "arm": arm, "records": len(recs), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # Between-arm contrast of context sensitivities: does target deletion change use of SA context?
    between: dict[str, Any] = {}
    # Build uid maps for sensitivity values.
    sens_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, recs in arm_records.items():
        sens_by_arm[arm] = {}
        for r in recs:
            losses = r.get("losses", {})
            if "mask_abs_context" in losses and "intact" in losses:
                p = int(losses["intact"]["pieces"])
                d = float(losses["mask_abs_context"]["loss_sum"]) - float(losses["intact"]["loss_sum"])
                sens_by_arm[arm][str(r["event_uid"])] = {"delta_sum": d, "pieces": p, "pair_id": str(r.get("pair_id")), "eval_set": str(r.get("eval_set")), "context_group": str(r.get("context_group"))}
    for left, right in [("full_compact_100M", "drop_abs_100M"), ("full_compact_100M", "drop_copied_word_100M"), ("drop_abs_100M", "drop_copied_word_100M")]:
        vals_by_group: dict[str, list[tuple[str, float, int]]] = collections.defaultdict(list)
        common = sorted(set(sens_by_arm[left]) & set(sens_by_arm[right]))
        for uid in common:
            a = sens_by_arm[left][uid]
            b = sens_by_arm[right][uid]
            p = int(a["pieces"])
            delta = float(a["delta_sum"]) - float(b["delta_sum"])
            vals_by_group["all"].append((a["pair_id"], delta, p))
            vals_by_group[a["eval_set"]].append((a["pair_id"], delta, p))
            vals_by_group[a["context_group"]].append((a["pair_id"], delta, p))
        key = f"{left}_minus_{right}_abs_context_sensitivity"
        between[key] = {}
        for group, vals in sorted(vals_by_group.items()):
            den = sum(p for _, _, p in vals)
            num = sum(d for _, d, _ in vals)
            between[key][group] = {
                "n_events": len(vals),
                "n_pairs": len(set(cid for cid, _, _ in vals)),
                "target_pieces": int(den),
                "piece_weighted_delta_nats": round(num / den, 6) if den else None,
                "bootstrap": cluster_bootstrap(vals, args.seed + abs(hash(key + group)) % 1000000, args.n_boot),
            }

    result = {
        "status": "ANCHOR_CONTEXT_PERTURBATION_PROBE",
        "meaning": "Evaluation-only saved-model probe: for retained/copy anchor targets, positive abs_sensitivity means masking visible source-absent compact-content context worsens copied-anchor prediction. Matched function/copied controls test generic extra-mask versus anchor/copy-context dependence.",
        "inputs": {
            "tokenizer": str(TOKENIZER),
            "train_fixed_events": str(TRAIN_FIXED_EVENTS),
            "train_fixed_events_sha256": sha256_file(TRAIN_FIXED_EVENTS),
            "source_disjoint_events": str(SOURCE_DISJOINT_EVENTS),
            "source_disjoint_events_sha256": sha256_file(SOURCE_DISJOINT_EVENTS),
            "arms": {k: str(v) for k, v in ARMS.items()},
        },
        "event_meta": event_meta,
        "event_file": str(event_path),
        "arm_summary": arm_summary,
        "between_arm_abs_context_sensitivity": between,
        "interpretation_keys": {
            "ACA_support": "source-absent context perturbation should hurt copied-anchor prediction more than matched function perturbation, especially for contexts containing relational/event source-absent words; if full/drop_abs share sensitivity, input exposure rather than source-absent target loss is sufficient for this context role.",
            "copy_anchor_dominance": "copied-context matched perturbation much larger than source-absent perturbation means anchor prediction is dominated by other copied lexical anchors, not novel compact context.",
            "negative_result": "near-zero source-absent sensitivity, or sensitivity no larger than matched function masking, weakens anchor-conditioned abstraction and points back to generic input distribution / supervision mass.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_lines = [
        "# research anchor-context perturbation probe",
        "",
        f"JSON: `{out_json}`",
        "",
        "Positive sensitivity = extra context masking raises retained-anchor target loss.",
        "",
        "## Event preparation",
        f"- prepared_events: {event_meta.get('prepared_events')}",
        f"- by_eval_set: {event_meta.get('by_eval_set')}",
        f"- skips: {event_meta.get('skips')}",
        "",
        "## Main sensitivities (all events)",
    ]
    for arm in ARMS:
        md_lines.append(f"### {arm}")
        summ = arm_summary[arm]
        for cname in ["abs_sensitivity", "function_sensitivity", "copied_sensitivity", "abs_minus_function_sensitivity", "abs_minus_copied_sensitivity"]:
            rec = summ.get(cname, {}).get("all", {}).get("all")
            if rec:
                boot = rec.get("bootstrap") or {}
                md_lines.append(f"- {cname}: n={rec['n_events']}, delta={rec['piece_weighted_delta_nats']}, boot[{boot.get('p025')},{boot.get('p975')}], frac_gt0={boot.get('fraction_gt0')}")
        cg = summ.get("abs_sensitivity", {}).get("context_group", {})
        for g in ["abs_context_has_rel_event", "abs_context_no_rel_event"]:
            if g in cg:
                rec = cg[g]; boot = rec.get("bootstrap") or {}
                md_lines.append(f"  - abs_sensitivity/{g}: n={rec['n_events']}, delta={rec['piece_weighted_delta_nats']}, boot[{boot.get('p025')},{boot.get('p975')}]")
        md_lines.append("")
    md_lines.append("## Between-arm source-absent context sensitivity")
    for key, groups in between.items():
        rec = groups.get("all")
        if rec:
            boot = rec.get("bootstrap") or {}
            md_lines.append(f"- {key}: n={rec['n_events']}, delta={rec['piece_weighted_delta_nats']}, boot[{boot.get('p025')},{boot.get('p975')}]")
    (out_dir / "anchor_context_perturbation_probe.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(out_json), "prepared_events": len(events), "elapsed_sec": result["elapsed_sec"], "main_all": {arm: {c: arm_summary[arm].get(c, {}).get("all", {}).get("all", {}) for c in ["abs_sensitivity", "function_sensitivity", "abs_minus_function_sensitivity"]} for arm in ARMS}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
