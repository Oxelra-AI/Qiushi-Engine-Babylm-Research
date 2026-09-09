#!/usr/bin/env python3
"""research: decompose the research interference boundary.

No-training analysis and small fixed inference on existing checkpoints.  The goal is
to test whether the prior comparison conflated state supersession with
evidence form: explicit override used direct state words, while contradiction rows
required action verbs to imply the final state.

This script:
  1. builds a frozen decomposition probe extending research base frames;
  2. scores selected existing checkpoints;
  3. decomposes margins into target prior, prior-only, action-only, and combined
     prior+action terms, using additive residuals;
  4. completes an update-sensitive official EWoK comparison from existing row
     records and research legal40k by-domain tables.

No training, no endpoint mutation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition')
PROBE_DIR = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/probe')
SCORE_DIR = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/scores')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/overwrite_decomposition.md')
CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "default")
CACHE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/hf_cache') / CACHE_SUFFIX
for name, path in {
    "HOME": CACHE_ROOT / "home",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "HF_HOME": CACHE_ROOT / "hf_home",
    "HF_MODULES_CACHE": CACHE_ROOT / "hf_modules",
    "TRANSFORMERS_CACHE": CACHE_ROOT / "transformers",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(name, str(path))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

FRAMES = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_frames.jsonl')
MANIFEST = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_manifest.json')
BY_DOMAIN = _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_by_domain_fullcpu.csv')

TARGETS = {
    **{f"scale1p75_{m}M": A02_WS / f"training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_{m}M"
       for m in [77, 78, 79, 80, 81, 82, 83, 100]},
    "legal16k_base_80M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
    "legal16k_base_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M'),
    "legal16k_80M_seed43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_80M'),
    "legal16k_100M_seed43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_100M'),
    "legal40k_8x480_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "legal40k_depth12_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "fw_compact_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M'),
    "fw_rowblock_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M'),
    "mlm_only_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/hf_model/final'),
    "coupled_aligned_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final'),
    "coupled_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final'),
}
DEFAULT_TARGETS = [
    "legal16k_base_100M_seed43022",
    "scale1p75_82M",
    "scale1p75_100M",
    "legal40k_8x480_100M_seed43022",
    "legal40k_depth12_100M_seed43022",
    "fw_rowblock_100M_seed43022",
    "fw_compact_100M_seed43022",
    "mlm_only_20M",
    "coupled_shuffled_20M",
]

EWOK_RECORDS = {
    "legal16k_base_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/legal16k_100M/ewok_interaction_records.csv'),
    "scale1p75_100M": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/scale1p75_100M/ewok_interaction_records.csv'),
    "fw_compact_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_compact_fullbatch_seed43022/ewok_interaction_records.csv'),
    "fw_rowblock_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_breadth_rowblock_fullbatch_seed43022/ewok_interaction_records.csv'),
    "mlm_only_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/mlm_only_20M/ewok_interaction_records.csv'),
    "coupled_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/coupled_aligned_20M/ewok_interaction_records.csv'),
    "coupled_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/coupled_shuffled_20M/ewok_interaction_records.csv'),
}
LEGAL40_MODEL_LABELS = {
    "legal40k_8x480_100M_seed43022": "legal40_8x480_43022",
    "legal40k_depth12_100M_seed43022": "legal40_depth_12x384_43022",
}

ORDER = [
    "none",
    "prior_only_contradict",
    "action_only",
    "action_distance",
    "contradict_adjacent",
    "contradict_then",
    "contradict_distance",
    "consistent_adjacent",
    "explicit_override_direct",
    "tf_prior_only_contradict",
    "tf_action_only",
    "tf_contradict_adjacent",
    "tf_contradict_distance",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(rows: list[dict[str, Any]]) -> str:
    s = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def qstats(vals: Iterable[Any]) -> dict[str, Any]:
    xs = []
    for v in vals:
        try:
            x = float(v)
            if math.isfinite(x):
                xs.append(x)
        except Exception:
            pass
    xs.sort()
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        "n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25),
        "mean": statistics.fmean(xs), "median": statistics.median(xs),
        "p75": q(0.75), "p95": q(0.95), "max": xs[-1],
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (len(xs) - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (len(ys) - 1))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((len(xs) - 1) * sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    def ranks(vs: list[float]) -> list[float]:
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i + 1
            while j < len(vs) and vs[idx[j]] == vs[idx[i]]:
                j += 1
            r = (i + j - 1) / 2.0
            for k in range(i, j):
                out[idx[k]] = r
            i = j
        return out
    return pearson(ranks(xs), ranks(ys))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_base_from_step186() -> list[dict[str, Any]]:
    frames = load_jsonl(FRAMES)
    by_base: dict[str, dict[str, Any]] = {}
    for r in frames:
        if r["condition"] == "last_event":
            by_base[r["base_id"]] = {
                "base_id": r["base_id"], "family": r["family"], "object": r["object"],
                "actor": r["actor"], "alt_a": r["alt_a"], "alt_b": r["alt_b"],
            }
    return [by_base[k] for k in sorted(by_base)]


def tf_contexts(obj: str, actor: str, family: str, state_a: str, state_b: str) -> dict[str, tuple[str, str]]:
    # Contexts intentionally avoid literal target words open/closed/empty/full.
    # They preserve the final state implication using ordinary paraphrases.
    if family == "empty_full":
        prior_a = f"The {obj} had nothing in it."
        prior_b = f"The {obj} held water up to the top."
        action_a = f"{actor} poured everything out of the {obj}."
        action_b = f"{actor} poured water into the {obj} until it could hold no more."
    elif family == "open_closed":
        # Use forms that are grammatical for box/bag/window/door/drawer and avoid open/closed substrings.
        prior_a = f"The {obj} was not shut."
        prior_b = f"The {obj} was shut."
        action_a = f"{actor} pulled the {obj} wide."
        action_b = f"{actor} pushed the {obj} shut."
    else:
        raise ValueError(family)
    return {
        "tf_prior_only_contradict": (prior_b, prior_a),
        "tf_action_only": (action_a, action_b),
        "tf_contradict_adjacent": (f"{prior_b} {action_a}", f"{prior_a} {action_b}"),
        "tf_contradict_distance": (
            f"{prior_b} A red pencil was beside the table. A small book was on the shelf. {action_a}",
            f"{prior_a} A red pencil was beside the table. A small book was on the shelf. {action_b}",
        ),
    }


def direct_contexts(obj: str, actor: str, family: str, state_a: str, state_b: str) -> dict[str, tuple[str, str]]:
    if family == "empty_full":
        ea, eb = "emptied", "filled"
    elif family == "open_closed":
        ea, eb = "opened", "closed"
    else:
        raise ValueError(family)
    neutral = "A red pencil was beside the table. A small book was on the shelf."
    return {
        "none": ("", ""),
        "prior_only_contradict": (f"The {obj} was {state_b}.", f"The {obj} was {state_a}."),
        "action_only": (f"{actor} {ea} the {obj}.", f"{actor} {eb} the {obj}."),
        "action_distance": (f"{neutral} {actor} {ea} the {obj}.", f"{neutral} {actor} {eb} the {obj}."),
        "contradict_adjacent": (f"The {obj} was {state_b}. {actor} {ea} the {obj}.", f"The {obj} was {state_a}. {actor} {eb} the {obj}."),
        "contradict_then": (f"The {obj} was {state_b}. Then {actor} {ea} the {obj}.", f"The {obj} was {state_a}. Then {actor} {eb} the {obj}."),
        "contradict_distance": (f"The {obj} was {state_b}. {neutral} {actor} {ea} the {obj}.", f"The {obj} was {state_a}. {neutral} {actor} {eb} the {obj}."),
        "consistent_adjacent": (f"The {obj} was already {state_a}. Then {actor} {ea} the {obj}.", f"The {obj} was already {state_b}. Then {actor} {eb} the {obj}."),
        "explicit_override_direct": (f"The {obj} was {state_b}. The {obj} is now {state_a}.", f"The {obj} was {state_a}. The {obj} is now {state_b}."),
    }


def freeze_probe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    base = get_base_from_step186()
    frames: list[dict[str, Any]] = []
    counter = 0
    target_word_hits = []
    for b in base:
        ctxs = {}
        ctxs.update(direct_contexts(b["object"], b["actor"], b["family"], b["alt_a"], b["alt_b"]))
        ctxs.update(tf_contexts(b["object"], b["actor"], b["family"], b["alt_a"], b["alt_b"]))
        for cond in ORDER:
            c1, c2 = ctxs[cond]
            counter += 1
            row = {
                "frame_id": f"OD_{counter:05d}",
                "base_id": b["base_id"],
                "condition": cond,
                "family": b["family"],
                "object": b["object"],
                "actor": b["actor"],
                "alt_a": b["alt_a"],
                "alt_b": b["alt_b"],
                "context1": c1,
                "context2": c2,
                "query_template": f"The {b['object']} is now {{target}}.",
                "correct_c1": "a",
                "correct_c2": "b",
                "target_word_free_context": cond.startswith("tf_"),
            }
            if cond.startswith("tf_"):
                low = (c1 + " " + c2).lower()
                for w in ["open", "closed", "empty", "full"]:
                    if re.search(rf"\b{re.escape(w)}\b", low):
                        target_word_hits.append({"frame_id": row["frame_id"], "condition": cond, "word": w, "context1": c1, "context2": c2})
            frames.append(row)
    frame_path = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/probe/overwrite_decomposition_frames.jsonl')
    with frame_path.open("w", encoding="utf-8") as fh:
        for r in frames:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "created_utc": now(),
        "source_step186_frames": rel(FRAMES),
        "source_step186_manifest": rel(MANIFEST),
        "scientific_object": "prior/action/combined overwrite decomposition under direct and target-word-free lexical evidence",
        "n_base_frames": len(base),
        "conditions": ORDER,
        "condition_counts": dict(collections.Counter(r["condition"] for r in frames)),
        "n_frames": len(frames),
        "frame_path": rel(frame_path),
        "frame_sha256": sha256_path(frame_path),
        "content_sha256": sha256_lines(frames),
        "target_word_free_hits": target_word_hits[:20],
        "n_target_word_free_hits": len(target_word_hits),
    }
    (_public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/probe/overwrite_decomposition_manifest.json')).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return frames, manifest


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


class Scorer:
    def __init__(self, model, tok, device, batch_size: int):
        self.model = model
        self.tok = tok
        self.device = device
        self.batch_size = batch_size
        self.mask_id = tok.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("tokenizer has no mask token")

    def score_examples(self, examples: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
        vals: dict[tuple[int, str], float] = {}
        todo = []
        for ex in examples:
            enc = self.tok(ex["sentence"], return_offsets_mapping=True, return_tensors=None)
            ids = list(enc["input_ids"])
            att = list(enc["attention_mask"])
            offs = list(enc["offset_mapping"])
            s, e = ex["span"]
            selected = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
            if len(selected) != 1:
                vals[(ex["rec_i"], ex["key"])] = float("nan")
                continue
            pos = selected[0]
            cur = list(ids)
            cur[pos] = self.mask_id
            todo.append({"sid": (ex["rec_i"], ex["key"]), "ids": cur, "att": att,
                         "pos": pos, "tid": ids[pos], "length": len(cur)})
        todo.sort(key=lambda x: x["length"])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.inference_mode():
            for st in range(0, len(todo), self.batch_size):
                batch = todo[st:st + self.batch_size]
                ml = max(x["length"] for x in batch)
                ids_t = torch.tensor([x["ids"] + [pad] * (ml - x["length"]) for x in batch], dtype=torch.long, device=self.device)
                att_t = torch.tensor([x["att"] + [0] * (ml - x["length"]) for x in batch], dtype=torch.long, device=self.device)
                pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=self.device)
                tid_t = torch.tensor([x["tid"] for x in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=ids_t, attention_mask=att_t)
                logits = out.logits if hasattr(out, "logits") else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                lp = torch.gather(F.log_softmax(logits[mb, pos_t], dim=-1), -1, tid_t.unsqueeze(-1)).squeeze(-1)
                for sid, val in zip([x["sid"] for x in batch], lp.detach().cpu().tolist()):
                    vals[sid] = float(val)
        return vals


def make_examples(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exs = []
    for i, r in enumerate(frames):
        for ck, ctx in [("c1", r["context1"]), ("c2", r["context2"] )]:
            for ak, alt in [("a", r["alt_a"]), ("b", r["alt_b"] )]:
                sent, span = render(ctx, r["query_template"], alt)
                exs.append({"rec_i": i, "key": f"{ck}_{ak}", "sentence": sent, "span": span})
    return exs


def attach_scores(frames: list[dict[str, Any]], scores: dict[tuple[int, str], float]) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(frames):
        c1a = scores.get((i, "c1_a"), float("nan"))
        c1b = scores.get((i, "c1_b"), float("nan"))
        c2a = scores.get((i, "c2_a"), float("nan"))
        c2b = scores.get((i, "c2_b"), float("nan"))
        d1 = c1a - c1b
        d2 = c2a - c2b
        rows.append({
            "frame_id": r["frame_id"], "base_id": r["base_id"], "condition": r["condition"],
            "family": r["family"], "object": r["object"], "alt_a": r["alt_a"], "alt_b": r["alt_b"],
            "delta1": d1, "delta2": d2,
            "signed_c1": d1, "signed_c2": -d2,
            "interaction": d1 - d2,
            "min_signed_margin": min(d1, -d2),
            "crossed_success": d1 > 0 and d2 < 0,
            "same_a_bias": d1 > 0 and d2 > 0,
            "same_b_bias": d1 < 0 and d2 < 0,
            "target_word_free_context": bool(r.get("target_word_free_context")),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    out = {}
    for cond in ORDER:
        rs = groups.get(cond, [])
        if not rs:
            continue
        out[cond] = {
            "n": len(rs),
            "crossed": sum(1 for r in rs if r["crossed_success"]) / len(rs),
            "min_signed_margin": qstats(r["min_signed_margin"] for r in rs),
            "signed_c1": qstats(r["signed_c1"] for r in rs),
            "signed_c2": qstats(r["signed_c2"] for r in rs),
            "interaction": qstats(r["interaction"] for r in rs),
        }
    return out


def decompose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_base_cond: dict[tuple[str, str], dict[str, Any]] = {(r["base_id"], r["condition"]): r for r in rows}
    residual_rows = []
    comparison_specs = [
        ("direct_adjacent", "none", "prior_only_contradict", "action_only", "contradict_adjacent"),
        ("direct_then", "none", "prior_only_contradict", "action_only", "contradict_then"),
        ("direct_distance", "none", "prior_only_contradict", "action_distance", "contradict_distance"),
        ("targetfree_adjacent", "none", "tf_prior_only_contradict", "tf_action_only", "tf_contradict_adjacent"),
        ("targetfree_distance", "none", "tf_prior_only_contradict", "tf_action_only", "tf_contradict_distance"),
    ]
    for base in sorted({r["base_id"] for r in rows}):
        meta = next(r for r in rows if r["base_id"] == base)
        for label, none_c, prior_c, action_c, combined_c in comparison_specs:
            needed = [by_base_cond.get((base, c)) for c in [none_c, prior_c, action_c, combined_c]]
            if any(x is None for x in needed):
                continue
            none, prior, action, combined = needed  # type: ignore[misc]
            pred_c1 = prior["signed_c1"] + action["signed_c1"] - none["signed_c1"]
            pred_c2 = prior["signed_c2"] + action["signed_c2"] - none["signed_c2"]
            res_c1 = combined["signed_c1"] - pred_c1
            res_c2 = combined["signed_c2"] - pred_c2
            residual_rows.append({
                "base_id": base, "family": meta["family"], "object": meta["object"],
                "label": label,
                "none_c1": none["signed_c1"], "none_c2": none["signed_c2"],
                "prior_c1": prior["signed_c1"], "prior_c2": prior["signed_c2"],
                "action_c1": action["signed_c1"], "action_c2": action["signed_c2"],
                "combined_c1": combined["signed_c1"], "combined_c2": combined["signed_c2"],
                "pred_c1": pred_c1, "pred_c2": pred_c2,
                "residual_c1": res_c1, "residual_c2": res_c2,
                "residual_mean": 0.5 * (res_c1 + res_c2),
                "action_effect_mean": 0.5 * ((action["signed_c1"] - none["signed_c1"]) + (action["signed_c2"] - none["signed_c2"])),
                "prior_effect_mean": 0.5 * ((prior["signed_c1"] - none["signed_c1"]) + (prior["signed_c2"] - none["signed_c2"])),
                "combined_effect_mean": 0.5 * ((combined["signed_c1"] - none["signed_c1"]) + (combined["signed_c2"] - none["signed_c2"])),
                "combined_crossed": combined["crossed_success"],
                "action_crossed": action["crossed_success"],
            })
    by_label = {}
    for label, rs in collections.defaultdict(list, {k: [] for k in []}).items():
        pass
    grouped = collections.defaultdict(list)
    for r in residual_rows:
        grouped[r["label"]].append(r)
    for label, rs in sorted(grouped.items()):
        by_label[label] = {
            "n": len(rs),
            "residual_mean": qstats(r["residual_mean"] for r in rs),
            "residual_c1": qstats(r["residual_c1"] for r in rs),
            "residual_c2": qstats(r["residual_c2"] for r in rs),
            "prior_effect_mean": qstats(r["prior_effect_mean"] for r in rs),
            "action_effect_mean": qstats(r["action_effect_mean"] for r in rs),
            "combined_effect_mean": qstats(r["combined_effect_mean"] for r in rs),
            "combined_crossed_frac": sum(1 for r in rs if r["combined_crossed"]) / len(rs),
            "action_crossed_frac": sum(1 for r in rs if r["action_crossed"]) / len(rs),
            "negative_nonadditive_frac": sum(1 for r in rs if r["residual_mean"] < -0.25) / len(rs),
            "positive_nonadditive_frac": sum(1 for r in rs if r["residual_mean"] > 0.25) / len(rs),
        }
    return {"by_label": by_label, "residual_rows": residual_rows}


def score_target(target: str, frames: list[dict[str, Any]], device_name: str, batch_size: int, threads: int) -> dict[str, Any]:
    path = TARGETS[target]
    if not path.exists():
        raise FileNotFoundError(path)
    if threads > 0:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "score_start", "target": target, "device": str(device), "model": rel(path)}), flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True).eval().to(device)
    scorer = Scorer(model, tok, device, batch_size)
    examples = make_examples(frames)
    scores = scorer.score_examples(examples)
    rows = attach_scores(frames, scores)
    out_dir = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/scores/targets') / target
    write_csv(out_dir / "overwrite_decomposition_records.csv", rows)
    decomp = decompose(rows)
    write_csv(out_dir / "overwrite_decomposition_residual_rows.csv", decomp["residual_rows"])
    summary = {
        "target": target, "model_path": rel(path), "elapsed_sec": round(time.time() - t0, 2),
        "n_frames": len(frames), "n_examples": len(examples),
        "by_condition": summarize_condition(rows),
        "decomposition": {"by_label": decomp["by_label"]},
    }
    (out_dir / "overwrite_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "score_done", "target": target,
                      "direct_resid_med": summary["decomposition"]["by_label"].get("direct_adjacent", {}).get("residual_mean", {}).get("median"),
                      "tf_resid_med": summary["decomposition"]["by_label"].get("targetfree_adjacent", {}).get("residual_mean", {}).get("median"),
                      "elapsed_sec": summary["elapsed_sec"]}), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def fnum(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def summarize_ewok_rows(path: Path, domain: str | None = None, update_filter: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"n": 0, "missing": rel(path)}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if domain is not None and r.get("domain") != domain:
                continue
            if update_filter:
                text = " ".join(r.get(k, "") for k in ["ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB", "context_diff_c1_texts_joined", "context_diff_c2_texts_joined", "target1", "target2"]).lower()
                # update-sensitive official rows: material/physical dynamics and rows with state-change lexical material.
                if not (r.get("domain") in {"material-dynamics", "physical-dynamics"} or re.search(r"\b(open|closed|empty|full|wet|dry|break|broken|melt|melted|cook|cooked|freeze|frozen|burn|burned|clean|dirty|fall|fell|dropped|filled|emptied|opened|shut)\b", text)):
                    continue
            rows.append(r)
    if not rows:
        return {"n": 0}
    acc = sum(1 for r in rows if r.get("saved_model_correct_flag") == "True") / len(rows)
    sf = sum(1 for r in rows if r.get("conditional_reversal_failure_stable") == "True") / len(rows)
    inter = [fnum(r.get("interaction_sum")) for r in rows]
    inter = [x for x in inter if x is not None]
    return {
        "n": len(rows), "accuracy": acc, "stable_failure_frac": sf,
        "interaction_sum_mean": statistics.fmean(inter) if inter else None,
        "interaction_sum_median": statistics.median(inter) if inter else None,
    }


def legal40_domain_table() -> dict[str, dict[str, Any]]:
    out = {}
    if not BY_DOMAIN.exists():
        return out
    by_model_domain = {}
    with BY_DOMAIN.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            by_model_domain[(r["model"], r["by_domain"])] = r
    for public, label in LEGAL40_MODEL_LABELS.items():
        all_rows = [r for (m, d), r in by_model_domain.items() if m == label]
        mat = by_model_domain.get((label, "material-dynamics"))
        phys = by_model_domain.get((label, "physical-dynamics"))
        agent = by_model_domain.get((label, "agent-properties"))
        def conv(r):
            if not r:
                return {"n": 0}
            return {
                "n": int(float(r["n"])),
                "accuracy": fnum(r["saved_model_accuracy"]),
                "stable_failure_frac": fnum(r["stable_nonpositive_interaction_frac"]),
                "interaction_sum_median": fnum(r["interaction_sum_median"]),
                "interaction_sum_positive_frac": fnum(r["interaction_sum_positive_frac"]),
            }
        # exact all-EWoK aggregate exists in the full JSON refs, but compute weighted accuracy from domains for local use.
        if all_rows:
            n_total = sum(int(float(r["n"])) for r in all_rows)
            acc = sum(int(float(r["n"])) * float(r["saved_model_accuracy"]) for r in all_rows) / n_total
            sf = sum(int(float(r["n"])) * float(r["stable_nonpositive_interaction_frac"]) for r in all_rows) / n_total
        else:
            n_total, acc, sf = 0, None, None
        out[public] = {
            "all_ewok": {"n": n_total, "accuracy": acc, "stable_failure_frac": sf},
            "material_dynamics": conv(mat),
            "physical_dynamics": conv(phys),
            "agent_properties": conv(agent),
            "source": rel(BY_DOMAIN),
        }
    return out


def official_surface_link(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"targets": {}, "correlations": {}}
    legal40 = legal40_domain_table()
    for s in summaries:
        t = s["target"]
        direct = s["decomposition"]["by_label"].get("direct_adjacent", {})
        tf = s["decomposition"]["by_label"].get("targetfree_adjacent", {})
        metrics = {
            "last_action_crossed": s["by_condition"].get("action_only", {}).get("crossed"),
            "direct_combined_crossed": s["by_condition"].get("contradict_adjacent", {}).get("crossed"),
            "tf_action_crossed": s["by_condition"].get("tf_action_only", {}).get("crossed"),
            "tf_combined_crossed": s["by_condition"].get("tf_contradict_adjacent", {}).get("crossed"),
            "direct_nonadditive_median": direct.get("residual_mean", {}).get("median"),
            "tf_nonadditive_median": tf.get("residual_mean", {}).get("median"),
            "direct_negative_nonadditive_frac": direct.get("negative_nonadditive_frac"),
            "tf_negative_nonadditive_frac": tf.get("negative_nonadditive_frac"),
        }
        if t in EWOK_RECORDS:
            surf = {
                "all_ewok": summarize_ewok_rows(EWOK_RECORDS[t]),
                "material_dynamics": summarize_ewok_rows(EWOK_RECORDS[t], domain="material-dynamics"),
                "physical_dynamics": summarize_ewok_rows(EWOK_RECORDS[t], domain="physical-dynamics"),
                "update_sensitive_rows": summarize_ewok_rows(EWOK_RECORDS[t], update_filter=True),
                "agent_properties": summarize_ewok_rows(EWOK_RECORDS[t], domain="agent-properties"),
                "source": rel(EWOK_RECORDS[t]),
            }
        elif t in legal40:
            surf = legal40[t]
            # For legal40, use material+physical dynamics as the update-sensitive official slice available in research table.
            md = surf.get("material_dynamics", {"n": 0})
            pd = surf.get("physical_dynamics", {"n": 0})
            if md.get("n", 0) and pd.get("n", 0):
                n = md["n"] + pd["n"]
                surf["update_sensitive_rows"] = {
                    "n": n,
                    "accuracy": (md["n"] * md["accuracy"] + pd["n"] * pd["accuracy"]) / n,
                    "stable_failure_frac": (md["n"] * md["stable_failure_frac"] + pd["n"] * pd["stable_failure_frac"]) / n,
                }
            else:
                surf["update_sensitive_rows"] = {"n": 0}
        else:
            surf = {}
        out["targets"][t] = {"synthetic_metrics": metrics, "official_surfaces": surf}

    # Cross-target associations.  These are not row-identical but use row-level official subsets aggregated per target.
    syn_keys = ["direct_nonadditive_median", "tf_nonadditive_median", "last_action_crossed", "tf_action_crossed"]
    surf_specs = [
        ("update_acc", ["update_sensitive_rows", "accuracy"]),
        ("update_stable_frac", ["update_sensitive_rows", "stable_failure_frac"]),
        ("material_acc", ["material_dynamics", "accuracy"]),
        ("material_stable_frac", ["material_dynamics", "stable_failure_frac"]),
        ("all_ewok_acc", ["all_ewok", "accuracy"]),
        ("all_ewok_stable_frac", ["all_ewok", "stable_failure_frac"]),
    ]
    for sk in syn_keys:
        for surf_name, path in surf_specs:
            xs: list[float] = []
            ys: list[float] = []
            pts = []
            for t, entry in out["targets"].items():
                x = entry["synthetic_metrics"].get(sk)
                cur = entry.get("official_surfaces", {})
                for key in path:
                    cur = cur.get(key, {}) if isinstance(cur, dict) else {}
                y = cur if isinstance(cur, (int, float)) else None
                if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)):
                    xs.append(float(x)); ys.append(float(y)); pts.append(t)
            out["correlations"][f"{sk}__{surf_name}"] = {
                "n": len(xs), "pearson": pearson(xs, ys), "spearman": spearman(xs, ys), "targets": pts,
            }
    return out


def write_summary(panel: dict[str, Any], manifest: dict[str, Any]) -> None:
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json')
    out_json.write_text(json.dumps(panel, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    targets = panel["targets"]
    lines = [
        "# research overwrite decomposition",
        "",
        "Status: **NO_TRAINING_DECOMPOSITION**",
        "",
        "This analysis addresses a potential confound: research contrasted explicit final-state language with action-mediated contradiction, so it could have conflated state supersession with evidence form.",
        "",
        f"Frozen probe: {manifest['n_frames']} frames = {manifest['n_base_frames']} base frames × {len(manifest['conditions'])} conditions.",
        f"Content SHA256: `{manifest['content_sha256']}`; frame SHA256: `{manifest['frame_sha256']}`.",
        f"Target-word-free context hit count: {manifest['n_target_word_free_hits']} (should be 0 for literal open/closed/empty/full words).",
        "",
        "## Synthetic decomposition by target",
        "",
        "| target | action crossed | direct combined crossed | direct residual median | direct neg-resid frac | tf action crossed | tf combined crossed | tf residual median | tf neg-resid frac | update-sensitive EWoK acc | update-sensitive stable frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(v):
        return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{float(v):.4f}"
    for s in targets:
        t = s["target"]
        byc = s["by_condition"]
        dec = s["decomposition"]["by_label"]
        link = panel.get("official_surface_link", {}).get("targets", {}).get(t, {})
        surf = link.get("official_surfaces", {}).get("update_sensitive_rows", {}) if link else {}
        lines.append(
            f"| {t} | {fmt(byc.get('action_only',{}).get('crossed'))} | {fmt(byc.get('contradict_adjacent',{}).get('crossed'))} | "
            f"{fmt(dec.get('direct_adjacent',{}).get('residual_mean',{}).get('median'))} | {fmt(dec.get('direct_adjacent',{}).get('negative_nonadditive_frac'))} | "
            f"{fmt(byc.get('tf_action_only',{}).get('crossed'))} | {fmt(byc.get('tf_contradict_adjacent',{}).get('crossed'))} | "
            f"{fmt(dec.get('targetfree_adjacent',{}).get('residual_mean',{}).get('median'))} | {fmt(dec.get('targetfree_adjacent',{}).get('negative_nonadditive_frac'))} | "
            f"{fmt(surf.get('accuracy'))} | {fmt(surf.get('stable_failure_frac'))} |"
        )
    lines.extend(["", "## Cross-target association with update-sensitive official rows", "", "| synthetic metric vs official surface | n | Pearson | Spearman |", "|---|---:|---:|---:|"])
    for k, c in sorted(panel.get("official_surface_link", {}).get("correlations", {}).items()):
        if "update" in k or "material" in k:
            lines.append(f"| {k} | {c['n']} | {fmt(c['pearson'])} | {fmt(c['spearman'])} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Direct-state contradiction rows remain extremely nonadditive if the combined prior+action margin is far below prior-only + action-only minus no-context. A robust training route would require the same negative residual under target-word-free paraphrases and a meaningful association with update-sensitive official rows. If target-word-free residuals vanish or official association is weak/inverted, the research universal zero should be treated as a synthetic boundary rather than a sufficient training target.",
        "",
        f"JSON: `{rel(out_json)}`",
    ])
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    frames, manifest = freeze_probe()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    summaries = []
    for t in targets:
        summaries.append(score_target(t, frames, args.device, args.batch_size, args.threads))
    link = official_surface_link(summaries)
    panel = {
        "status": "OVERWRITE_DECOMPOSITION_DONE",
        "created_utc": now(),
        "probe_manifest": manifest,
        "targets": summaries,
        "official_surface_link": link,
    }
    write_summary(panel, manifest)
    print(json.dumps({"status": panel["status"], "summary": rel(_public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json')), "note": rel(NOTE_PATH)}), flush=True)


if __name__ == "__main__":
    main()
