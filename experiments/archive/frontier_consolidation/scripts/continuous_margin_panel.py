#!/usr/bin/env python3
"""research: continuous per-item margin panel for fixed-budget substitution arms.

Scientific role
---------------
The remaining BabyLM Strict-Small substitution contrasts are near the measured
accuracy noise floor.  Thresholded family accuracy can hide a small but signed
movement because it discards margin size.  This script scores the same official
stable-family stimuli (or deterministic per-subtask samples of them) with
continuous masked-LM log-probability margins:

* BLiMP / Supplement / COMPS: pseudo-log-probability(good sentence)
  minus pseudo-log-probability(bad sentence), with both summed and per-token
  averaged margins saved.
* EWoK: target-context margins and the symmetric 2x2 interaction
  (C1T1+C2T2)-(C1T2+C2T1), saved as the main EWoK margin.
* Entity Tracking: correct option log-probability minus best distractor.

The output is not a leaderboard score and does not touch GlobalPIQA, SuperGLUE,
AoA, upload, or submission.  It is an instrument for reading signs and itemwise
stability when family accuracies are too coarse.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
PRISTINE_FULL = STRICT / "evaluation_data/full_eval"
OUT_DEFAULT = WS / "data/continuous_margin_panel"

CHOICE_FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
DEFAULT_CKS = [f"chck_{i}M" for i in range(10, 101, 10)]

ARM_DEFS: dict[str, dict[str, Any]] = {
    "clean_maxgeom": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "MAX-geometry clean seed43022 reference",
        "rho": 0.0,
        "kind": "clean",
    },
    "max_view": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "MAX view: FineWeb source + compact view, distinct paired content",
        "rho": 0.111872,
        "kind": "fineweb_distinct_view",
    },
    "max_repeat": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "MAX repeat: duplicated admitted source content",
        "rho": 0.111872,
        "kind": "fineweb_duplicate_repeat",
    },
    "max_breadth": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "MAX breadth: same source rows plus independent FineWeb companion sentences",
        "rho": 0.111872,
        "kind": "fineweb_distinct_breadth",
    },
    "regmax_adultprose": {
        "run_dir": WS / "training/runs/regmax_adultprose_samefw_deberta100M_seed43022",
        "description": "MAX register arm: identical FineWeb admission, adult-prose clean rows removed",
        "rho": 0.111872,
        "kind": "same_fineweb_adultprose_removed",
    },
    "regmax_childspeech": {
        "run_dir": WS / "training/runs/regmax_childspeech_samefw_deberta100M_seed43022",
        "description": "MAX register arm: identical FineWeb admission, child/speech rows removed",
        "rho": 0.111872,
        "kind": "same_fineweb_childspeech_removed",
    },
    "subdose_quarter_1x": {
        "run_dir": WS / "training/runs/subdose_quarter_1x_deberta100M_seed43022",
        "description": "Sub-dose view quarter of 1x changed block",
        "rho": 0.0106,
        "kind": "subdose_view",
    },
    "subdose_half_1x": {
        "run_dir": WS / "training/runs/subdose_half_1x_deberta100M_seed43022",
        "description": "Sub-dose view half of 1x changed block",
        "rho": 0.0212,
        "kind": "subdose_view",
    },
    "subdose_full_1x": {
        "run_dir": WS / "training/runs/subdose_full_1x_deberta100M_seed43022",
        "description": "Sub-dose view full 1x changed block, comparator for in-corpus adult prose",
        "rho": 0.0424,
        "kind": "subdose_view",
    },
    "incorpus_adultprose": {
        "run_dir": WS / "training/runs/incorpus_adultprose_deberta100M_seed43022",
        "description": "In-corpus surplus Gutenberg/SimpleWiki adult prose, no FineWeb",
        "rho": 0.044299,
        "kind": "incorpus_adultprose_surplus",
    },
}


@dataclass(frozen=True)
class ChoiceRow:
    family: str
    uid: str
    subtask: str
    good: str
    bad: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class EwokRow:
    uid: str
    domain: str
    index: int
    row: dict[str, Any]


@dataclass(frozen=True)
class EntityRow:
    uid: str
    split: str
    numops: int
    index: int
    input_prefix: str
    options: list[str]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFF


def sample_rows(rows: list[Any], n: int, seed: int, key: str) -> list[Any]:
    if n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random(stable_seed(seed, key))
    return sorted(rng.sample(rows, n), key=lambda r: getattr(r, "uid", str(r)))


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_blimp_like(family: str, data_dir: pathlib.Path, per_subtask: int, seed: int) -> list[ChoiceRow]:
    rows: list[ChoiceRow] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        subtask = path.stem
        local: list[ChoiceRow] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                good = str(d.get("sentence_good", ""))
                bad = str(d.get("sentence_bad", ""))
                if not good or not bad:
                    continue
                uid = f"{subtask}_{i}"
                local.append(ChoiceRow(family=family, uid=uid, subtask=subtask, good=good, bad=bad, meta={k: d.get(k) for k in ["field", "linguistics_term", "contrast", "UID", "pair_id", "row"] if k in d}))
        rows.extend(sample_rows(local, per_subtask, seed, f"{family}:{subtask}"))
    return rows


def load_comps(per_file: int, seed: int) -> list[ChoiceRow]:
    rows: list[ChoiceRow] = []
    for path in sorted((PRISTINE_FULL / "comps").glob("*.jsonl")):
        # Keep the official files but avoid letting the huge base file dominate a
        # small-margin readout; deterministic per-file sampling preserves breadth.
        subtask = path.stem.replace("comps_", "")
        local: list[ChoiceRow] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                prop = str(d.get("property_phrase", "")).strip()
                pa = str(d.get("prefix_acceptable", "")).strip()
                pu = str(d.get("prefix_unacceptable", "")).strip()
                if not (prop and pa and pu):
                    continue
                good = f"{pa} {prop}".strip()
                bad = f"{pu} {prop}".strip()
                uid = f"{subtask}_{i}"
                local.append(ChoiceRow(family="COMPS", uid=uid, subtask=subtask, good=good, bad=bad, meta={k: d.get(k) for k in ["negative_sample_type", "similarity", "property", "acceptable_concept", "unacceptable_concept"] if k in d}))
        rows.extend(sample_rows(local, per_file, seed, f"COMPS:{subtask}"))
    return rows


def load_ewok(per_domain: int, seed: int) -> list[EwokRow]:
    rows: list[EwokRow] = []
    for path in sorted((PRISTINE_FULL / "ewok_filtered").glob("*.jsonl")):
        domain = path.stem
        local: list[EwokRow] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                uid = f"{domain}_{i}"
                local.append(EwokRow(uid=uid, domain=domain, index=i, row=d))
        rows.extend(sample_rows(local, per_domain, seed, f"EWoK:{domain}"))
    return rows


def load_entity(per_group: int, seed: int) -> list[EntityRow]:
    rows: list[EntityRow] = []
    groups: dict[tuple[str, int], list[EntityRow]] = {}
    group_counts: dict[str, int] = {}
    for path in sorted((PRISTINE_FULL / "entity_tracking").glob("*.jsonl")):
        split = path.stem
        with path.open("r", encoding="utf-8") as f:
            for raw_i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                opts = [str(x) for x in d.get("options", [])]
                # Official entity scoring filters examples whose gold option contains
                # "nothing"; reuse the research convention so item ids align.
                if any("nothing" in o for o in opts):
                    continue
                numops = int(d.get("numops", -1))
                subset = f"{split}_{numops}_ops"
                j = group_counts.get(subset, 0)
                group_counts[subset] = j + 1
                row = EntityRow(uid=f"{subset}_{j}", split=split, numops=numops, index=raw_i, input_prefix=str(d.get("input_prefix", "")), options=opts)
                groups.setdefault((split, numops), []).append(row)
    for (split, numops), local in sorted(groups.items()):
        rows.extend(sample_rows(local, per_group, seed, f"Entity:{split}:{numops}"))
    return rows


def build_rows(args: argparse.Namespace) -> dict[str, list[Any]]:
    rows = {
        "BLiMP": load_blimp_like("BLiMP", PRISTINE_FULL / "blimp_filtered", args.blimp_per_subtask, args.seed),
        "Supplement": load_blimp_like("Supplement", PRISTINE_FULL / "supplement_filtered", args.supp_per_subtask, args.seed),
        "EWoK": load_ewok(args.ewok_per_domain, args.seed),
        "Entity": load_entity(args.entity_per_group, args.seed),
        "COMPS": load_comps(args.comps_per_file, args.seed),
    }
    return {k: v for k, v in rows.items() if k in args.families}


def row_counts(rows: dict[str, list[Any]]) -> dict[str, int]:
    return {k: len(v) for k, v in rows.items()}


# ------------------------------ scoring child ------------------------------

def prepare_cpu_env(out_dir: pathlib.Path, worker_id: str, cpu_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, cpu_threads))
    cache = out_dir / f"hf_cache_margin_{worker_id}"
    tmp = out_dir / f"tmp_margin_{worker_id}"
    for p in [cache, cache / "hub", cache / "transformers", cache / "modules", cache / "datasets", tmp]:
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    os.environ["TMPDIR"] = str(tmp.resolve())


def load_model_and_tokenizer(model_path: pathlib.Path, cpu_threads: int):
    import torch  # noqa: PLC0415
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU safety failure: torch can see CUDA")
    torch.set_num_threads(max(1, cpu_threads))
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast  # noqa: PLC0415
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), padding_side="right", trust_remote_code=True)
    except Exception:
        tok = PreTrainedTokenizerFast.from_pretrained(str(model_path), padding_side="right")
    if tok.pad_token_id is None:
        if tok.cls_token_id is not None:
            tok.pad_token_id = tok.cls_token_id
        elif tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token_id is None:
        raise RuntimeError(f"Tokenizer at {model_path} has no mask token")
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(torch.device("cpu"))
    model.eval()
    return torch, model, tok


def span_encoding(tokenizer: Any, text: str, completion: str | None, max_length: int) -> dict[str, Any]:
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    tokens = list(enc["input_ids"])
    attn = list(enc["attention_mask"])
    offsets = enc["offset_mapping"]
    if completion is None:
        start_char = 0
    else:
        start_char = len(text) - len(completion)
        if start_char < 0:
            start_char = 0
    indices: list[int] = []
    targets: list[int] = []
    for i, (start, end) in enumerate(offsets):
        # Exclude specials and padding (usually offset 0,0); score every token with
        # positive character support that overlaps the requested completion span.
        if attn[i] and end > max(0, start_char) and end > start:
            indices.append(i)
            targets.append(int(tokens[i]))
    if not indices:
        return {"valid": False, "reason": "no_scored_tokens", "tokens": tokens, "attn": attn, "indices": [], "targets": []}
    return {"valid": True, "reason": "ok", "tokens": tokens, "attn": attn, "indices": indices, "targets": targets}


def score_encodings(model: Any, tokenizer: Any, encs: list[dict[str, Any]], torch: Any, batch_size: int) -> list[dict[str, float | int | bool | str]]:
    records: list[tuple[int, list[int], list[int], int, int]] = []
    out = [{"valid": bool(e.get("valid")), "n_tokens": 0, "sum_logprob": float("nan"), "mean_logprob": float("nan"), "mean_p_target": float("nan"), "mean_logit_grad_l2": float("nan"), "top1_acc": float("nan")} for e in encs]
    for ex_i, enc in enumerate(encs):
        if not enc.get("valid"):
            continue
        for idx, tgt in zip(enc["indices"], enc["targets"]):
            toks = list(enc["tokens"])
            toks[int(idx)] = int(tokenizer.mask_token_id)
            records.append((ex_i, toks, list(enc["attn"]), int(idx), int(tgt)))
    if not records:
        return out
    import torch.nn.functional as F  # noqa: PLC0415
    pad_id = int(tokenizer.pad_token_id)
    sums = [0.0 for _ in encs]
    counts = [0 for _ in encs]
    p_sums = [0.0 for _ in encs]
    grad_sums = [0.0 for _ in encs]
    top1_sums = [0 for _ in encs]
    device = torch.device("cpu")
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(r[1]) for r in batch)
        ids, masks, idxs, tgts, exs = [], [], [], [], []
        for ex_i, toks, attn, idx, tgt in batch:
            pad_n = max_len - len(toks)
            ids.append(toks + [pad_id] * pad_n)
            masks.append(attn + [0] * pad_n)
            idxs.append(idx)
            tgts.append(tgt)
            exs.append(ex_i)
        with torch.no_grad():
            output = model(input_ids=torch.tensor(ids, dtype=torch.long, device=device), attention_mask=torch.tensor(masks, dtype=torch.long, device=device))
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if logits.size(1) != max_len:
                logits = logits[:, -max_len:]
            mb = torch.arange(logits.shape[0], device=device)
            masked_logits = logits[mb, torch.tensor(idxs, dtype=torch.long, device=device)]
            lp = F.log_softmax(masked_logits, dim=-1)
            tgt_t = torch.tensor(tgts, dtype=torch.long, device=device).unsqueeze(-1)
            chosen_lp = torch.gather(lp, -1, tgt_t).squeeze(-1)
            p = torch.exp(lp)
            p_target = torch.exp(chosen_lp)
            grad_l2 = torch.sqrt(torch.clamp((p * p).sum(dim=-1) - 2.0 * p_target + 1.0, min=0.0))
            top1 = (masked_logits.argmax(dim=-1) == torch.tensor(tgts, dtype=torch.long, device=device))
            vals = chosen_lp.detach().cpu().tolist()
            pvals = p_target.detach().cpu().tolist()
            gvals = grad_l2.detach().cpu().tolist()
            tvals = top1.detach().cpu().tolist()
        for ex_i, val, pv, gv, tv in zip(exs, vals, pvals, gvals, tvals):
            sums[ex_i] += float(val)
            p_sums[ex_i] += float(pv)
            grad_sums[ex_i] += float(gv)
            top1_sums[ex_i] += int(bool(tv))
            counts[ex_i] += 1
    for i, n in enumerate(counts):
        if n:
            out[i] = {
                "valid": True,
                "n_tokens": n,
                "sum_logprob": sums[i],
                "mean_logprob": sums[i] / n,
                "mean_p_target": p_sums[i] / n,
                "mean_logit_grad_l2": grad_sums[i] / n,
                "top1_acc": top1_sums[i] / n,
            }
    return out


def score_choice_rows(model: Any, tok: Any, torch: Any, rows: list[ChoiceRow], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.row_batch_size):
        batch = rows[start:start + args.row_batch_size]
        encs: list[dict[str, Any]] = []
        for r in batch:
            # Score the full candidate sentence.  Save token-mean and token-sum;
            # token-mean is the default readout to reduce length sensitivity.
            encs.append(span_encoding(tok, r.good, r.good, args.max_length))
            encs.append(span_encoding(tok, r.bad, r.bad, args.max_length))
        vals = score_encodings(model, tok, encs, torch, args.mask_batch_size)
        for i, r in enumerate(batch):
            vg, vb = vals[2 * i], vals[2 * i + 1]
            rec = {
                "family": r.family,
                "uid": r.uid,
                "subtask": r.subtask,
                "main_margin": (float(vg["mean_logprob"]) - float(vb["mean_logprob"])) if finite(vg.get("mean_logprob")) and finite(vb.get("mean_logprob")) else None,
                "sum_margin": (float(vg["sum_logprob"]) - float(vb["sum_logprob"])) if finite(vg.get("sum_logprob")) and finite(vb.get("sum_logprob")) else None,
                "good_mean_logprob": vg.get("mean_logprob"),
                "bad_mean_logprob": vb.get("mean_logprob"),
                "good_sum_logprob": vg.get("sum_logprob"),
                "bad_sum_logprob": vb.get("sum_logprob"),
                "good_tokens": vg.get("n_tokens"),
                "bad_tokens": vb.get("n_tokens"),
                "good_grad_proxy": vg.get("mean_logit_grad_l2"),
                "bad_grad_proxy": vb.get("mean_logit_grad_l2"),
                "correct_by_mean": bool(finite(vg.get("mean_logprob")) and finite(vb.get("mean_logprob")) and float(vg["mean_logprob"]) > float(vb["mean_logprob"])),
                "correct_by_sum": bool(finite(vg.get("sum_logprob")) and finite(vb.get("sum_logprob")) and float(vg["sum_logprob"]) > float(vb["sum_logprob"])),
            }
            rec.update({f"meta_{k}": v for k, v in r.meta.items() if isinstance(v, (str, int, float, bool)) or v is None})
            records.append(rec)
    return records


def score_ewok_rows(model: Any, tok: Any, torch: Any, rows: list[EwokRow], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.row_batch_size):
        batch = rows[start:start + args.row_batch_size]
        encs: list[dict[str, Any]] = []
        for r in batch:
            d = r.row
            c1, c2 = str(d["Context1"]), str(d["Context2"])
            t1, t2 = str(d["Target1"]), str(d["Target2"])
            for c, t in [(c1, t1), (c2, t1), (c1, t2), (c2, t2)]:
                sent = " ".join([c, t])
                encs.append(span_encoding(tok, sent, " " + t, args.max_length))
        vals = score_encodings(model, tok, encs, torch, args.mask_batch_size)
        for i, r in enumerate(batch):
            s11, s21, s12, s22 = vals[4 * i], vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]
            def f(v, k): return float(v[k]) if finite(v.get(k)) else float("nan")
            m_sum = f(s11, "sum_logprob") - f(s21, "sum_logprob")
            inter_sum = (f(s11, "sum_logprob") + f(s22, "sum_logprob")) - (f(s12, "sum_logprob") + f(s21, "sum_logprob"))
            m_avg = f(s11, "mean_logprob") - f(s21, "mean_logprob")
            inter_avg = (f(s11, "mean_logprob") + f(s22, "mean_logprob")) - (f(s12, "mean_logprob") + f(s21, "mean_logprob"))
            records.append({
                "family": "EWoK",
                "uid": r.uid,
                "subtask": r.domain,
                "domain": r.domain,
                "index": r.index,
                "main_margin": inter_avg if finite(inter_avg) else None,
                "sum_margin": inter_sum if finite(inter_sum) else None,
                "target1_context_margin_avg": m_avg if finite(m_avg) else None,
                "target1_context_margin_sum": m_sum if finite(m_sum) else None,
                "interaction_avg": inter_avg if finite(inter_avg) else None,
                "interaction_sum": inter_sum if finite(inter_sum) else None,
                "correct_by_interaction_avg": bool(finite(inter_avg) and inter_avg > 0),
                "correct_by_target1_margin_avg": bool(finite(m_avg) and m_avg > 0),
                "ContextType": r.row.get("ContextType"),
                "ContextDiff": r.row.get("ContextDiff"),
                "TargetDiff": r.row.get("TargetDiff"),
            })
    return records


def score_entity_rows(model: Any, tok: Any, torch: Any, rows: list[EntityRow], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.row_batch_size):
        batch = rows[start:start + args.row_batch_size]
        encs: list[dict[str, Any]] = []
        keys: list[tuple[int, int]] = []
        for bi, r in enumerate(batch):
            for oi, opt in enumerate(r.options):
                encs.append(span_encoding(tok, r.input_prefix + opt, opt, args.max_length))
                keys.append((bi, oi))
        vals = score_encodings(model, tok, encs, torch, args.mask_batch_size)
        by_row: dict[int, dict[int, dict[str, Any]]] = {}
        for (bi, oi), val in zip(keys, vals):
            by_row.setdefault(bi, {})[oi] = val
        for bi, r in enumerate(batch):
            d = by_row.get(bi, {})
            if not d:
                continue
            # Save both sum and token-average margins.  Token-average is the main
            # readout because option lengths differ, but the historical research
            # sum margin remains in sum_margin for comparison.
            avg_scores = {oi: float(v["mean_logprob"]) for oi, v in d.items() if finite(v.get("mean_logprob"))}
            sum_scores = {oi: float(v["sum_logprob"]) for oi, v in d.items() if finite(v.get("sum_logprob"))}
            if not avg_scores or 0 not in avg_scores:
                main = None
                sum_m = None
                pred_avg = -1
                pred_sum = -1
            else:
                pred_avg = max(avg_scores.items(), key=lambda kv: kv[1])[0]
                distract_avg = max([v for k, v in avg_scores.items() if k != 0], default=float("nan"))
                main = avg_scores[0] - distract_avg if finite(distract_avg) else None
                pred_sum = max(sum_scores.items(), key=lambda kv: kv[1])[0] if sum_scores else -1
                distract_sum = max([v for k, v in sum_scores.items() if k != 0], default=float("nan"))
                sum_m = sum_scores.get(0, float("nan")) - distract_sum if finite(distract_sum) and 0 in sum_scores else None
            records.append({
                "family": "Entity",
                "uid": r.uid,
                "subtask": f"{r.split}_{r.numops}_ops",
                "split": r.split,
                "numops": r.numops,
                "index": r.index,
                "option_count": len(r.options),
                "main_margin": main,
                "sum_margin": sum_m,
                "correct_by_mean": bool(pred_avg == 0),
                "correct_by_sum": bool(pred_sum == 0),
                "gold_mean_logprob": avg_scores.get(0),
                "gold_sum_logprob": sum_scores.get(0),
                "best_distractor_mean_logprob": (max([v for k, v in avg_scores.items() if k != 0], default=None) if avg_scores else None),
                "best_distractor_sum_logprob": (max([v for k, v in sum_scores.items() if k != 0], default=None) if sum_scores else None),
            })
    return records


def summarize_vals(vals: Iterable[Any]) -> dict[str, Any]:
    xs = [float(v) for v in vals if finite(v)]
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "se": (statistics.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0,
        "p10": ys[max(0, int(0.10 * (len(ys) - 1)))],
        "p90": ys[min(len(ys) - 1, int(0.90 * (len(ys) - 1)))],
        "frac_positive": sum(x > 0 for x in xs) / len(xs),
    }


def score_model_job(job: dict[str, Any]) -> dict[str, Any]:
    out_dir = pathlib.Path(job["out_dir"])
    arm = job["arm"]
    ck = job["checkpoint"]
    worker_id = f"{arm}_{ck}_{os.getpid()}"
    args_dict = job["args"]
    class Args:  # minimal namespace for reused scoring functions
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    prepare_cpu_env(out_dir, worker_id, args.cpu_threads)
    model_path = pathlib.Path(job["model_path"])
    t0 = time.time()
    torch, model, tok = load_model_and_tokenizer(model_path, args.cpu_threads)
    rows_by_family = job["rows_by_family"]
    all_records: list[dict[str, Any]] = []
    family_timings: dict[str, float] = {}
    for fam in args.families:
        ft0 = time.time()
        rows = rows_by_family.get(fam, [])
        if not rows:
            continue
        if fam in {"BLiMP", "Supplement", "COMPS"}:
            recs = score_choice_rows(model, tok, torch, rows, args)
        elif fam == "EWoK":
            recs = score_ewok_rows(model, tok, torch, rows, args)
        elif fam == "Entity":
            recs = score_entity_rows(model, tok, torch, rows, args)
        else:
            continue
        for r in recs:
            r["arm"] = arm
            r["checkpoint"] = ck
            r["words"] = int(ck.split("_")[1].replace("M", "")) * 1_000_000
            r["rho"] = ARM_DEFS[arm]["rho"]
            r["arm_kind"] = ARM_DEFS[arm]["kind"]
        all_records.extend(recs)
        family_timings[fam] = round(time.time() - ft0, 2)
    del model

    family_summary: dict[str, Any] = {}
    for fam in sorted({r["family"] for r in all_records}):
        fam_rows = [r for r in all_records if r["family"] == fam]
        family_summary[fam] = {
            "main_margin": summarize_vals([r.get("main_margin") for r in fam_rows]),
            "sum_margin": summarize_vals([r.get("sum_margin") for r in fam_rows]),
            "correct_by_mean_pct": 100.0 * sum(bool(r.get("correct_by_mean") or r.get("correct_by_interaction_avg")) for r in fam_rows) / max(1, len(fam_rows)),
            "n_records": len(fam_rows),
        }
    result = {
        "status": "CONTINUOUS_MARGIN_MODEL_DONE",
        "created_utc": now(),
        "arm": arm,
        "checkpoint": ck,
        "model_path": rel(model_path),
        "description": ARM_DEFS[arm]["description"],
        "rho": ARM_DEFS[arm]["rho"],
        "families": args.families,
        "row_counts": {k: len(v) for k, v in rows_by_family.items()},
        "elapsed_sec": round(time.time() - t0, 2),
        "family_timings_sec": family_timings,
        "cpu_safety": "CUDA_VISIBLE_DEVICES=-1 before torch/transformers import; torch.cuda unavailable asserted",
        "summary": family_summary,
        "records": all_records,
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    return result


def selected_jobs(args: argparse.Namespace, rows_by_family: dict[str, list[Any]], out_dir: pathlib.Path) -> list[dict[str, Any]]:
    jobs = []
    arg_keys = ["families", "row_batch_size", "mask_batch_size", "max_length", "cpu_threads"]
    args_payload = {k: getattr(args, k) for k in arg_keys}
    for arm in args.arms:
        if arm not in ARM_DEFS:
            raise ValueError(f"Unknown arm {arm}; choices={sorted(ARM_DEFS)}")
        run_dir = pathlib.Path(ARM_DEFS[arm]["run_dir"])
        for ck in args.checkpoints:
            model_path = run_dir / "hf_model" / ck
            if not model_path.exists():
                continue
            cache = out_dir / "per_model" / f"{arm}_{ck}.json"
            if cache.exists() and not args.force:
                continue
            jobs.append({
                "arm": arm,
                "checkpoint": ck,
                "model_path": str(model_path),
                "out_dir": str(out_dir),
                "rows_by_family": rows_by_family,
                "args": args_payload,
            })
    return jobs


def write_per_model(result: dict[str, Any], out_dir: pathlib.Path) -> None:
    p = out_dir / "per_model" / f"{result['arm']}_{result['checkpoint']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def read_all_per_model(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    objs = []
    for p in sorted((out_dir / "per_model").glob("*.json")):
        try:
            objs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return objs


def window_of_ck(ck: str) -> list[str]:
    w = int(ck.split("_")[1].replace("M", ""))
    out = ["all"]
    if 10 <= w <= 80:
        out.append("common10_80")
    if 80 <= w <= 100:
        out.append("late80_100")
    if w == 80:
        out.append("endpoint80")
    return out


def write_readout(out_dir: pathlib.Path) -> dict[str, Any]:
    objs = read_all_per_model(out_dir)
    all_records: list[dict[str, Any]] = []
    model_summary_rows = []
    for obj in objs:
        model_summary_rows.append({
            "arm": obj.get("arm"),
            "checkpoint": obj.get("checkpoint"),
            "elapsed_sec": obj.get("elapsed_sec"),
            "families": ";".join(obj.get("families") or []),
            "n_records": len(obj.get("records") or []),
        })
        all_records.extend(obj.get("records") or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_csv = out_dir / "continuous_margin_records.csv"
    fields = sorted({k for r in all_records for k in r.keys()}) if all_records else ["arm", "checkpoint", "family", "uid", "main_margin"]
    with flat_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    model_csv = out_dir / "model_summaries.csv"
    with model_csv.open("w", encoding="utf-8", newline="") as f:
        fields2 = ["arm", "checkpoint", "elapsed_sec", "families", "n_records"]
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader(); w.writerows(model_summary_rows)

    # Model-level family summaries.
    family_rows = []
    for obj in objs:
        for fam, s in (obj.get("summary") or {}).items():
            mm = s.get("main_margin") or {}
            sm = s.get("sum_margin") or {}
            family_rows.append({
                "arm": obj.get("arm"), "checkpoint": obj.get("checkpoint"), "family": fam,
                "n": mm.get("n", 0), "main_mean": mm.get("mean"), "main_se": mm.get("se"),
                "main_frac_positive": mm.get("frac_positive"), "sum_mean": sm.get("mean"),
                "correct_by_mean_pct": s.get("correct_by_mean_pct"),
            })
    fam_csv = out_dir / "family_margin_summaries.csv"
    with fam_csv.open("w", encoding="utf-8", newline="") as f:
        fields3 = ["arm", "checkpoint", "family", "n", "main_mean", "main_se", "main_frac_positive", "sum_mean", "correct_by_mean_pct"]
        w = csv.DictWriter(f, fieldnames=fields3)
        w.writeheader(); w.writerows(family_rows)

    by_key = {(r.get("arm"), r.get("checkpoint"), r.get("family"), r.get("uid")): r for r in all_records}
    pairs = [
        ("max_view_minus_clean", "max_view", "clean_maxgeom"),
        ("max_repeat_minus_clean", "max_repeat", "clean_maxgeom"),
        ("max_breadth_minus_clean", "max_breadth", "clean_maxgeom"),
        ("max_view_minus_repeat", "max_view", "max_repeat"),
        ("max_view_minus_breadth", "max_view", "max_breadth"),
        ("reg_childspeech_minus_adultprose", "regmax_childspeech", "regmax_adultprose"),
        ("reg_childspeech_minus_clean", "regmax_childspeech", "clean_maxgeom"),
        ("reg_adultprose_minus_clean", "regmax_adultprose", "clean_maxgeom"),
        ("subdose_quarter_minus_clean", "subdose_quarter_1x", "clean_maxgeom"),
        ("subdose_half_minus_clean", "subdose_half_1x", "clean_maxgeom"),
        ("subdose_full_minus_clean", "subdose_full_1x", "clean_maxgeom"),
        ("incorpus_minus_clean", "incorpus_adultprose", "clean_maxgeom"),
        ("incorpus_minus_subdose_full", "incorpus_adultprose", "subdose_full_1x"),
    ]
    delta_rows: list[dict[str, Any]] = []
    arms = sorted({r.get("arm") for r in all_records})
    cks = sorted({r.get("checkpoint") for r in all_records}, key=lambda x: int(str(x).split("_")[1].replace("M", "")) if x else -1)
    families = sorted({r.get("family") for r in all_records})
    for label, a, b in pairs:
        if a not in arms or b not in arms:
            continue
        for ck in cks:
            for fam in families:
                uids = sorted({uid for (arm, c, f, uid) in by_key if arm == a and c == ck and f == fam} & {uid for (arm, c, f, uid) in by_key if arm == b and c == ck and f == fam})
                for uid in uids:
                    ra = by_key[(a, ck, fam, uid)]
                    rb = by_key[(b, ck, fam, uid)]
                    ma = ra.get("main_margin")
                    mb = rb.get("main_margin")
                    sa = ra.get("sum_margin")
                    sb = rb.get("sum_margin")
                    delta_rows.append({
                        "contrast": label,
                        "arm_a": a,
                        "arm_b": b,
                        "checkpoint": ck,
                        "family": fam,
                        "uid": uid,
                        "subtask": ra.get("subtask"),
                        "main_delta": float(ma) - float(mb) if finite(ma) and finite(mb) else None,
                        "sum_delta": float(sa) - float(sb) if finite(sa) and finite(sb) else None,
                        "a_main_margin": ma,
                        "b_main_margin": mb,
                    })
    delta_csv = out_dir / "continuous_margin_contrast_records.csv"
    dfields = sorted({k for r in delta_rows for k in r.keys()}) if delta_rows else ["contrast", "checkpoint", "family", "main_delta"]
    with delta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dfields, extrasaction="ignore")
        w.writeheader(); w.writerows(delta_rows)

    contrast_summaries: list[dict[str, Any]] = []
    for label in sorted({r["contrast"] for r in delta_rows}):
        for window in ["common10_80", "late80_100", "endpoint80", "all"]:
            win_rows = [r for r in delta_rows if r["contrast"] == label and window in window_of_ck(str(r["checkpoint"]))]
            for fam in sorted({r["family"] for r in win_rows}):
                group = [r for r in win_rows if r["family"] == fam]
                rec = {"contrast": label, "window": window, "family": fam}
                rec.update(summarize_vals([r.get("main_delta") for r in group]))
                rec["sum_delta"] = summarize_vals([r.get("sum_delta") for r in group])
                contrast_summaries.append(rec)
            ex = [r for r in win_rows if r.get("family") != "Entity"]
            if ex:
                rec = {"contrast": label, "window": window, "family": "exEntity_choice4"}
                rec.update(summarize_vals([r.get("main_delta") for r in ex]))
                rec["sum_delta"] = summarize_vals([r.get("sum_delta") for r in ex])
                contrast_summaries.append(rec)
    summ_csv = out_dir / "continuous_margin_contrast_summaries.csv"
    sfields = sorted({k for r in contrast_summaries for k in r.keys() if k != "sum_delta"}) + ["sum_delta_json"] if contrast_summaries else ["contrast", "window", "family"]
    with summ_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
        w.writeheader()
        for r in contrast_summaries:
            rr = dict(r)
            rr["sum_delta_json"] = json.dumps(rr.pop("sum_delta", {}), ensure_ascii=False)
            w.writerow(rr)

    summary = {
        "status": "CONTINUOUS_MARGIN_READOUT",
        "created_utc": now(),
        "scientific_reading": "Continuous itemwise masked-LM margins for the stable choice families. This complements, but does not replace, official accuracy scoring. Reading is not included because its official output is already continuous and is not a choice-margin task.",
        "model_count": len(objs),
        "record_count": len(all_records),
        "delta_record_count": len(delta_rows),
        "available_arms": sorted({r.get("arm") for r in all_records}),
        "available_checkpoints": cks,
        "files": {
            "flat_csv": rel(flat_csv),
            "model_summaries_csv": rel(model_csv),
            "family_summaries_csv": rel(fam_csv),
            "contrast_records_csv": rel(delta_csv),
            "contrast_summaries_csv": rel(summ_csv),
            "summary_json": rel(out_dir / "continuous_margin_summary.json"),
            "summary_md": rel(out_dir / "continuous_margin_summary.md"),
        },
        "contrast_summaries": contrast_summaries[:500],
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "continuous_margin_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        "# research continuous per-item margin panel",
        "",
        summary["scientific_reading"],
        "",
        f"Models read: {len(objs)}; records: {len(all_records)}; contrast records: {len(delta_rows)}.",
        "",
        "## Selected contrast summaries",
    ]
    interesting = ["reg_childspeech_minus_adultprose", "incorpus_minus_clean", "incorpus_minus_subdose_full", "max_view_minus_clean", "max_repeat_minus_clean", "max_breadth_minus_clean", "max_view_minus_repeat"]
    for rec in contrast_summaries:
        if rec.get("contrast") in interesting and rec.get("family") in {"exEntity_choice4", "EWoK", "Entity", "BLiMP", "Supplement", "COMPS"}:
            mean = rec.get("mean")
            se = rec.get("se")
            lines.append(f"- {rec['contrast']} {rec['window']} {rec['family']}: n={rec.get('n')} mean={mean if mean is None else round(float(mean), 6)} se={se if se is None else round(float(se), 6)} frac_pos={rec.get('frac_positive')}")
    lines += ["", "## Files"]
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (out_dir / "continuous_margin_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def cmd_plan(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    rows = build_rows(args)
    jobs = selected_jobs(args, rows, out_dir)
    available = []
    missing = []
    for arm in args.arms:
        run_dir = pathlib.Path(ARM_DEFS[arm]["run_dir"])
        for ck in args.checkpoints:
            p = run_dir / "hf_model" / ck
            (available if p.exists() else missing).append({"arm": arm, "checkpoint": ck, "path": rel(p)})
    plan = {
        "status": "CONTINUOUS_MARGIN_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "families": args.families,
        "row_counts": row_counts(rows),
        "sample_policy": {
            "blimp_per_subtask": args.blimp_per_subtask,
            "supp_per_subtask": args.supp_per_subtask,
            "ewok_per_domain": args.ewok_per_domain,
            "entity_per_group": args.entity_per_group,
            "comps_per_file": args.comps_per_file,
            "seed": args.seed,
        },
        "available_model_count": len(available),
        "missing_model_count": len(missing),
        "jobs_not_cached": len(jobs),
        "workers": args.workers,
        "cpu_threads_per_worker": args.cpu_threads,
        "available_first12": available[:12],
        "missing_first12": missing[:12],
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)
    (out_dir / "plans" / "latest_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    jobs = selected_jobs(args, rows, out_dir)
    print(json.dumps({
        "event": "continuous_margin_run_start",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "row_counts": row_counts(rows),
        "jobs": len(jobs),
        "workers": args.workers,
        "cpu_threads_per_worker": args.cpu_threads,
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "families": args.families,
    }, indent=2, ensure_ascii=False), flush=True)
    t0 = time.time()
    ok = 0
    fail = 0
    if args.workers <= 1:
        for job in jobs:
            try:
                res = score_model_job(job)
                write_per_model(res, out_dir)
                ok += 1
                print(json.dumps({"event": "model_done", "arm": res["arm"], "checkpoint": res["checkpoint"], "records": len(res.get("records", [])), "elapsed_sec": res.get("elapsed_sec")}), flush=True)
            except Exception as e:
                fail += 1
                print(json.dumps({"event": "model_failed", "arm": job["arm"], "checkpoint": job["checkpoint"], "error": repr(e)}), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(score_model_job, j) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    res = fut.result()
                    write_per_model(res, out_dir)
                    ok += 1
                    print(json.dumps({"event": "model_done", "arm": res["arm"], "checkpoint": res["checkpoint"], "records": len(res.get("records", [])), "elapsed_sec": res.get("elapsed_sec")}), flush=True)
                except Exception as e:
                    fail += 1
                    print(json.dumps({"event": "model_failed", "error": repr(e)}), flush=True)
    summary = write_readout(out_dir)
    print(json.dumps({"status": "CONTINUOUS_MARGIN_RUN_FINISHED", "ok_models": ok, "failed_models": fail, "elapsed_sec": round(time.time() - t0, 2), "summary_json": summary["files"]["summary_json"]}, indent=2, ensure_ascii=False), flush=True)


def cmd_readout(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    summary = write_readout(out_dir)
    print(json.dumps({"status": summary["status"], "model_count": summary["model_count"], "record_count": summary["record_count"], "summary_json": summary["files"]["summary_json"], "summary_md": summary["files"]["summary_md"]}, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["plan", "run", "readout"], default="plan")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arms", nargs="+", default=["clean_maxgeom", "max_view", "max_repeat", "max_breadth", "regmax_adultprose", "regmax_childspeech", "subdose_quarter_1x", "subdose_half_1x", "subdose_full_1x", "incorpus_adultprose"])
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CKS)
    ap.add_argument("--families", nargs="+", default=CHOICE_FAMILIES, choices=CHOICE_FAMILIES)
    ap.add_argument("--blimp-per-subtask", type=int, default=16, help="0 means full BLiMP; default is deterministic per-subtask panel")
    ap.add_argument("--supp-per-subtask", type=int, default=0, help="0 means full Supplement")
    ap.add_argument("--ewok-per-domain", type=int, default=64, help="0 means full EWoK")
    ap.add_argument("--entity-per-group", type=int, default=64, help="0 means full Entity after official nothing-filter")
    ap.add_argument("--comps-per-file", type=int, default=256, help="0 means full COMPS; default keeps base file from dominating")
    ap.add_argument("--row-batch-size", type=int, default=32)
    ap.add_argument("--mask-batch-size", type=int, default=96)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=291)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.mode == "plan":
        cmd_plan(args)
    elif args.mode == "run":
        cmd_run(args)
    else:
        cmd_readout(args)


if __name__ == "__main__":
    main()
