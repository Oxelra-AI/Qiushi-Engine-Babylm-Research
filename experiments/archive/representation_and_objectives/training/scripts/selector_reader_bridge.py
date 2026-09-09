#!/usr/bin/env python3
"""research: Decomposed selector-reader bridge.

Final small bridge test for the supplied-address line.  Earlier monolithic
role-query state training failed, but that does not imply an architectural law.
This script separates:
  R(k, state): direct tag/address -> state reader, trained then held fixed.
  M(role, k): role-language -> tag/address selector, trained with explicit
              positive/negative candidate tags and no state labels.
  M∘R: select a tag with M, then query frozen R for state.

Scientific boundary: small pretrained ATP temporal-change bridge only; no
BabyLM-scale pretraining or official evaluation.
"""
from __future__ import annotations

import argparse, collections, copy, json, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

import importlib.util
STUDY = Path("experiments/archive/representation_and_objectives")
ST270_PATH = STUDY / "training/scripts/prefix_warmstart_probe.py"
spec270 = importlib.util.spec_from_file_location("st270", ST270_PATH)
st270 = importlib.util.module_from_spec(spec270)
assert spec270.loader is not None
sys.modules[spec270.name] = st270
spec270.loader.exec_module(st270)

s269 = st270.s269
base = st270.base
DEFAULT_MODEL = st270.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/selector_reader_bridge"

ALL_Q = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
SPARSE_Q = ["focal_before", "focal_after"]


def _role_words(qf: str, qmode: str) -> tuple[str, str]:
    scope = s269.SCOPE[qf]
    if qmode == "role_inline":
        time_word = s269.TNAME[qf]
    elif qmode == "role_para":
        time_word = s269.TPARA[s269.TNAME[qf]]
    else:
        raise ValueError(qmode)
    return scope, time_word


def selector_hyp(qf: str, cand_tag: str, *, qmode: str, style: str = "train") -> str:
    scope, time_word = _role_words(qf, qmode)
    if qmode == "role_inline":
        if style == "train":
            return f"The {scope} {time_word} entry is {cand_tag}."
        return f"The {scope} {time_word} record is named {cand_tag}."
    if qmode == "role_para":
        return f"The {scope} {time_word} record is named {cand_tag}."
    raise ValueError(qmode)


def tag_map_for_context(tmap: dict[str, str], *, swap: bool = False) -> dict[str, str]:
    """Role phrase -> tag mapping implied by inline-role context text."""
    if not swap:
        return dict(tmap)
    # s269.build_ctx with swap=True swaps both background/update labels in the
    # focal and secondary role text while leaving tmap as normal record identity.
    return {
        "focal_before": tmap["focal_after"],
        "focal_after": tmap["focal_before"],
        "secondary_before": tmap["secondary_after"],
        "secondary_after": tmap["secondary_before"],
    }


def role_state_label(qf: str, f, s, hd: str, *, swap: bool = False) -> int:
    if swap:
        return s269.lab_swap(qf, f, s, hd)
    return s269.lab_normal(qf, f, s, hd)


def _make_selector_row(*, ns: str, mode: str, qmode: str, qf: str, cand_tag: str,
                       expected_tag: str, context: str, f, s, am: dict[str, str],
                       split: str, eset: str, tkind: str, style: str,
                       swap: bool, group_key: str, vi: int, vl: int) -> dict[str, Any]:
    lab = int(cand_tag == expected_tag)
    hyp = selector_hyp(qf, cand_tag, qmode=qmode, style=style)
    qw = f if qf.startswith("focal") else s
    # Reader hypotheses for composition if this candidate is selected.
    h_ab = s269.hyp_for(qf=qf, qw=qw, am=am, hd="AB", qmode="direct_tag",
                        tag=cand_tag, style="held")
    h_ba = s269.hyp_for(qf=qf, qw=qw, am=am, hd="BA", qmode="direct_tag",
                        tag=cand_tag, style="held")
    row = base.mk(
        row_id=f"{ns}|selector|{mode}|{qmode}|{qf}|cand={cand_tag}|vi={vi}|vl={vl}|sw={swap}",
        context=context, hypothesis=hyp, label=lab, split=split, eval_set=eset,
        arm="selector", query_family=qf, focal_world=f.world_id,
        secondary_world=s.world_id, changed_focal=bool(f.changed),
        stable_secondary=not bool(s.changed),
        template_group=f"ctx={mode}|q={qmode}|vi={vi}|vl={vl}|sw={swap}",
        train_kind=f"{tkind}|ctx={mode}|q={qmode}", hyp_dir="SELECT",
        pair_key=group_key, hyp_wording=style, ctx_wording=mode,
        label_mode="selector")
    row.update({
        "candidate_tag": cand_tag,
        "expected_tag": expected_tag,
        "select_group": group_key,
        "selector_qmode": qmode,
        "role_swap": bool(swap),
        "reader_hyp_AB": h_ab,
        "reader_hyp_BA": h_ba,
        "role_label_AB": role_state_label(qf, f, s, "AB", swap=swap),
        "role_label_BA": role_state_label(qf, f, s, "BA", swap=swap),
        "vi": int(vi),
        "vl": int(vl),
    })
    return row


def selector_rows(focals, seconds, *, ns: str, mode: str, qmode: str, split: str,
                  eset: str, queries: list[str], tkind: str, style: str = "train",
                  swap: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                ctx, tmap, _tpl = s269.build_ctx(f=f, s=s, am=am, mode=mode, ns=ns,
                                                  vi=vi, vl=vl, swap=swap)
                rolemap = tag_map_for_context(tmap, swap=swap)
                all_tags = list(dict.fromkeys(tmap.values()))
                assert len(all_tags) == 4, all_tags
                for qf in queries:
                    g = f"{ns}|{mode}|{qmode}|{split}|{eset}|{f.world_id}|{s.world_id}|vi={vi}|vl={vl}|qf={qf}|sw={swap}"
                    for cand in all_tags:
                        rows.append(_make_selector_row(
                            ns=ns, mode=mode, qmode=qmode, qf=qf, cand_tag=cand,
                            expected_tag=rolemap[qf], context=ctx, f=f, s=s, am=am,
                            split=split, eset=eset, tkind=tkind, style=style,
                            swap=swap, group_key=g, vi=vi, vl=vl))
    return rows


def selector_train_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]
    mode = args.mode
    qmode = "role_inline"
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, Any, Any, str, list[str]]] = []
    specs.append(("base_stable_anchor", sel["base_stable"], sel["base_stable2"],
                  st270._addr_ns(args, mode, "b"), ALL_Q))
    if getattr(args, "base_train_wording", False):
        specs.append(("base_stable_train", sel["base_stable"], sel["base_stable2"],
                      st270._addr_ns(args, mode, "bw"), ALL_Q))
    specs.append(("sparse_changed_focal", sel["sparse_changed"], secs,
                  st270._addr_ns(args, mode, "c"), SPARSE_Q))
    specs.append(("sparse_stable_focal", sel["sparse_stable"], secs,
                  st270._addr_ns(args, mode, "s"), SPARSE_Q))
    for tkind, focals, seconds, ns, qs in specs:
        rows += selector_rows(focals, seconds, ns=ns, mode=mode, qmode=qmode,
                              split="train", eset=tkind, queries=qs, tkind=tkind,
                              style="train", swap=False)
    con = {"inv": inv, "selection": {k: base.world_summary(v) for k, v in sel.items()},
           "train": base.describe_rows(rows), "selector_groups": count_groups(rows)}
    return rows, con


def selector_eval_suite(args) -> dict[str, list[dict[str, Any]]]:
    sel, _ = base.select_splits(args)
    mode = args.mode
    foc_h, sec_h = sel["eval_held_changed"], sel["eval_held_stable"]
    foc_t = sel["eval_train_changed"]
    return {
        "held_exact_nsA": selector_rows(
            foc_h, sec_h, ns="e272_nsA_hC_hS", mode=mode, qmode="role_inline",
            split="held", eset="hC_hS", queries=ALL_Q, tkind="eval", style="held"),
        "held_exact_nsB": selector_rows(
            foc_h, sec_h, ns="e272_nsB_hC_hS", mode=mode, qmode="role_inline",
            split="held", eset="hC_hS", queries=ALL_Q, tkind="eval", style="held"),
        "trainChanged_exact_nsA": selector_rows(
            foc_t, sec_h, ns="e272_nsA_tC_hS", mode=mode, qmode="role_inline",
            split="held", eset="tC_hS", queries=ALL_Q, tkind="eval", style="held"),
        "held_role_swap_nsA": selector_rows(
            foc_h, sec_h, ns="e272_nsA_hC_hS", mode=mode, qmode="role_inline",
            split="held", eset="hC_hS_swap", queries=ALL_Q, tkind="eval", style="held", swap=True),
        "held_para_nsA": selector_rows(
            foc_h, sec_h, ns="e272_nsA_hC_hS", mode=mode, qmode="role_para",
            split="held", eset="hC_hS_para", queries=ALL_Q, tkind="eval", style="held"),
    }


def count_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r.get("select_group", r.get("pair_key", ""))].append(r)
    sizes = [len(v) for v in groups.values()]
    pos = [sum(int(x["label"]) for x in v) for v in groups.values()]
    return {"rows": len(rows), "groups": len(groups),
            "group_size_min": int(min(sizes)) if sizes else 0,
            "group_size_max": int(max(sizes)) if sizes else 0,
            "positive_per_group_min": int(min(pos)) if pos else 0,
            "positive_per_group_max": int(max(pos)) if pos else 0}


def token_audit(tok, rows: list[dict[str, Any]], max_len: int) -> dict[str, Any]:
    lens = [len(tok.encode(r["text"], add_special_tokens=True)) for r in rows]
    arr = np.array(lens, dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {"n": len(lens), "min": int(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "max": int(arr.max()),
            "frac_gt_max_len": float(np.mean(arr > max_len))}


def encode(tok, rows, max_len, dev):
    return base.encode_rows(tok, rows, max_len, dev)


def make_opt(model, args, *, head_lr=None, encoder_lr=None):
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    return torch.optim.AdamW([
        {"params": head_p, "lr": args.head_lr if head_lr is None else head_lr},
        {"params": enc_p, "lr": args.encoder_lr if encoder_lr is None else encoder_lr},
    ], weight_decay=args.weight_decay)


def get_logits(model, tok, rows, args, dev):
    return base.get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size)


def selector_binary_breakdown(logits: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = logits.argmax(1).cpu().numpy()
    lab = np.array([int(r["label"]) for r in rows])
    out: dict[str, Any] = {"row_acc": float(np.mean(pred == lab)),
                           "pos_rate_pred": float(np.mean(pred == 1)),
                           "pos_rate_true": float(np.mean(lab == 1))}
    for key in ["train_kind", "query_family", "selector_qmode", "role_swap"]:
        d = collections.defaultdict(list)
        for p, y, r in zip(pred, lab, rows):
            d[str(r.get(key))].append(int(p == y))
        out[f"by_{key}"] = {k: float(np.mean(v)) for k, v in sorted(d.items())}
    return out


def selector_top1(logits: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
    groups: dict[str, list[tuple[float, dict[str, Any]]]] = collections.defaultdict(list)
    for m, r in zip(margins, rows):
        groups[r["select_group"]].append((float(m), r))
    recs = []
    for g, vals in groups.items():
        vals = sorted(vals, key=lambda x: x[0], reverse=True)
        top = vals[0][1]
        ok = int(top["candidate_tag"] == top["expected_tag"])
        recs.append((ok, top))
    out: dict[str, Any] = {"top1": float(np.mean([x[0] for x in recs])) if recs else float("nan"),
                           "groups": len(recs)}
    for key in ["query_family", "selector_qmode", "role_swap", "eval_set"]:
        d = collections.defaultdict(list)
        for ok, r in recs:
            d[str(r.get(key))].append(ok)
        out[f"by_{key}"] = {k: float(np.mean(v)) for k, v in sorted(d.items())}
    return out


def train_selector(model, tok, rows, args, dev, *, seed: int, epochs: int,
                   label: str, track_evals: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    torch.manual_seed(seed); np.random.seed(seed)
    opt = make_opt(model, args)
    x, mask, y = encode(tok, rows, args.max_len, dev)
    loader = DataLoader(TensorDataset(x, mask, y), batch_size=args.selector_batch_size, shuffle=True)
    class_weight = torch.tensor([1.0, float(args.selector_pos_weight)], dtype=torch.float32, device=dev)
    best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    best_top1 = -1.0
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb, weight=class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss.item()))
        logits = get_logits(model, tok, rows, args, dev)
        top = selector_top1(logits, rows)
        binb = selector_binary_breakdown(logits, rows)
        ev_snap = {}
        if track_evals:
            for en, evrows in track_evals.items():
                evlog = get_logits(model, tok, evrows, args, dev)
                ev_snap[en] = selector_top1(evlog, evrows)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)),
                     "train_top1": top, "train_binary": binb,
                     "eval_top1": ev_snap})
        print(json.dumps({"event": "epoch", "phase": label, "epoch": ep,
                          "loss": hist[-1]["loss"], "train_top1": top.get("top1"),
                          "row_acc": binb.get("row_acc"),
                          "eval_top1": {k: v.get("top1") for k, v in ev_snap.items()}}), flush=True)
        if float(top.get("top1", -1.0)) > best_top1:
            best_top1 = float(top.get("top1", -1.0))
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    final_logits = get_logits(model, tok, rows, args, dev)
    return {"phase": label, "best_train_top1": float(best_top1),
            "final_top1": selector_top1(final_logits, rows),
            "final_binary": selector_binary_breakdown(final_logits, rows),
            "history": hist}


def selector_eval(model, tok, suites: dict[str, list[dict[str, Any]]], args, dev) -> dict[str, Any]:
    out = {}
    for name, rows in suites.items():
        logits = get_logits(model, tok, rows, args, dev)
        out[name] = {"top1": selector_top1(logits, rows),
                     "binary": selector_binary_breakdown(logits, rows)}
    return out


def selected_reader_rows(selector_model, tok, selector_rows_eval: list[dict[str, Any]], args, dev,
                         *, use_oracle: bool, eval_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logits = get_logits(selector_model, tok, selector_rows_eval, args, dev)
    margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
    groups: dict[str, list[tuple[float, dict[str, Any]]]] = collections.defaultdict(list)
    for m, r in zip(margins, selector_rows_eval):
        groups[r["select_group"]].append((float(m), r))
    state_rows: list[dict[str, Any]] = []
    selected = []
    for g, vals in groups.items():
        if use_oracle:
            poss = [r for _m, r in vals if r["candidate_tag"] == r["expected_tag"]]
            assert len(poss) == 1
            chosen = poss[0]
            score = float("nan")
        else:
            score, chosen = max(vals, key=lambda x: x[0])
        ok_sel = int(chosen["candidate_tag"] == chosen["expected_tag"])
        selected.append({"group": g, "qf": chosen["query_family"],
                         "chosen": chosen["candidate_tag"], "expected": chosen["expected_tag"],
                         "ok": ok_sel, "score": score,
                         "role_swap": bool(chosen.get("role_swap", False)),
                         "selector_qmode": chosen.get("selector_qmode")})
        for hd in ["AB", "BA"]:
            hyp = chosen[f"reader_hyp_{hd}"]
            lab = int(chosen[f"role_label_{hd}"])
            rr = base.mk(
                row_id=f"{eval_name}|compose|oracle={use_oracle}|{g}|{hd}|tag={chosen['candidate_tag']}",
                context=chosen["context"], hypothesis=hyp, label=lab,
                split="held", eval_set=eval_name, arm="composition",
                query_family=chosen["query_family"], focal_world=chosen["focal_world"],
                secondary_world=chosen["secondary_world"],
                changed_focal=chosen["changed_focal"], stable_secondary=chosen["stable_secondary"],
                template_group=chosen["template_group"],
                train_kind="composition_selected" if not use_oracle else "composition_oracle",
                hyp_dir=hd, pair_key=f"{eval_name}|{g}|qf={chosen['query_family']}",
                hyp_wording="held", ctx_wording=chosen["ctx_wording"],
                label_mode="role_swap" if chosen.get("role_swap") else "normal")
            rr.update({"selected_tag": chosen["candidate_tag"], "expected_tag": chosen["expected_tag"],
                       "selector_correct": ok_sel, "selector_qmode": chosen.get("selector_qmode"),
                       "role_swap": bool(chosen.get("role_swap", False))})
            state_rows.append(rr)
    top1 = float(np.mean([x["ok"] for x in selected])) if selected else float("nan")
    byq = collections.defaultdict(list)
    for x in selected:
        byq[x["qf"]].append(x["ok"])
    sel_summary = {"top1": top1, "groups": len(selected),
                   "by_query_family": {k: float(np.mean(v)) for k, v in sorted(byq.items())}}
    return state_rows, sel_summary


def reader_eval(reader_model, tok, rows: list[dict[str, Any]], args, dev) -> dict[str, Any]:
    ev = base.full_eval(reader_model, tok, {"compose": rows}, args.max_len, dev, args.eval_batch_size)["compose"]
    c = ev["contrastive"]
    return {"con_acc": c.get("contrastive_acc"),
            "by_q": c.get("by_query", {}),
            "by_m": c.get("by_query_margin", {}),
            "std_acc": ev["standard"].get("acc")}


def composition_eval(selector_model, reader_model, tok, suites: dict[str, list[dict[str, Any]]], args, dev) -> dict[str, Any]:
    out = {}
    for name, selrows in suites.items():
        chosen_rows, chosen_sel = selected_reader_rows(selector_model, tok, selrows, args, dev,
                                                       use_oracle=False, eval_name=name)
        oracle_rows, oracle_sel = selected_reader_rows(selector_model, tok, selrows, args, dev,
                                                       use_oracle=True, eval_name=name)
        out[name] = {"selector_top1_used": chosen_sel,
                     "composed": reader_eval(reader_model, tok, chosen_rows, args, dev),
                     "oracle_selector": oracle_sel,
                     "oracle_reader": reader_eval(reader_model, tok, oracle_rows, args, dev)}
    return out


def write_md(summary: dict[str, Any], out: Path) -> None:
    lines = ["# research selector-reader bridge\n\n"]
    lines.append("Purpose: train a direct address reader R, train a separate role-to-tag selector M with explicit candidate tags, then compose M and frozen R without oracle routing.\n\n")
    lines.append("## Construction\n\n")
    lines.append(f"Reader train rows: {summary['construction']['reader_train_rows']}; selector train rows: {summary['construction']['selector_train_rows']}.\n\n")
    lines.append(f"Selector group audit: `{json.dumps(summary['construction']['selector_group_audit'], sort_keys=True)}`\n\n")
    lines.append(f"Token audit: `{json.dumps(summary['construction']['token_audit'], sort_keys=True)}`\n\n")
    lines.append("## Reader R fit\n\n")
    for phase in ["reader_tag_pretrain", "reader_inline_direct"]:
        r = summary["results"].get(phase)
        if r is None:
            continue
        lines.append(f"### {phase}\n\nBest train acc: {r.get('best_train_acc')}\n\n")
        lines.append(f"Fit: `{json.dumps(r.get('fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")
    lines.append("## Selector M fit\n\n")
    sr = summary["results"].get("selector_train")
    if sr:
        lines.append(f"Best train top1: {sr.get('best_train_top1')}\n\n")
        lines.append(f"Final top1: `{json.dumps(sr.get('final_top1', {}), sort_keys=True)}`\n\n")
        lines.append(f"Final binary: `{json.dumps(sr.get('final_binary', {}), sort_keys=True)}`\n\n")
    lines.append("## Selector eval\n\n")
    lines.append("| eval | top1 | focal_before | focal_after | secondary_before | secondary_after |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, ev in sorted(summary.get("selector_evals", {}).items()):
        t = ev.get("top1", {})
        bq = t.get("by_query_family", {})
        gv = lambda q: float(bq.get(q, float('nan')))
        lines.append(f"| {name} | {float(t.get('top1', float('nan'))):.3f} | {gv('focal_before'):.3f} | {gv('focal_after'):.3f} | {gv('secondary_before'):.3f} | {gv('secondary_after'):.3f} |\n")
    lines.append("\n## Composition eval\n\n")
    lines.append("| eval | selector_top1 | comp_con | comp_fb | comp_fa | comp_sb | comp_sa | oracle_con | oracle_fb | oracle_fa | oracle_sb | oracle_sa |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, ev in sorted(summary.get("composition", {}).items()):
        sel = ev.get("selector_top1_used", {})
        comp = ev.get("composed", {})
        orc = ev.get("oracle_reader", {})
        cbq, obq = comp.get("by_q", {}), orc.get("by_q", {})
        gv = lambda d, q: float(d.get(q, float('nan')))
        lines.append(f"| {name} | {float(sel.get('top1', float('nan'))):.3f} | {float(comp.get('con_acc', float('nan'))):.3f} | {gv(cbq,'focal_before'):.3f} | {gv(cbq,'focal_after'):.3f} | {gv(cbq,'secondary_before'):.3f} | {gv(cbq,'secondary_after'):.3f} | {float(orc.get('con_acc', float('nan'))):.3f} | {gv(obq,'focal_before'):.3f} | {gv(obq,'focal_after'):.3f} | {gv(obq,'secondary_before'):.3f} | {gv(obq,'secondary_after'):.3f} |\n")
    lines.append("\n## Boundary\n\n")
    lines.append(summary.get("boundary", "") + "\n")
    (out / "selector_reader_summary.md").write_text("".join(lines), "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=27000)
    ap.add_argument("--mode", default="inline_role", choices=["inline_role"])
    ap.add_argument("--shared_address_ns", default="s269_inline_direct")
    ap.add_argument("--shared_address_suffix", default="_direct_tag")
    ap.add_argument("--base_stable", type=int, default=80)
    ap.add_argument("--base_train_wording", action="store_true")
    ap.add_argument("--sparse_changed", type=int, default=16)
    ap.add_argument("--sparse_stable", type=int, default=16)
    ap.add_argument("--eval_held_changed", type=int, default=40)
    ap.add_argument("--eval_held_stable", type=int, default=40)
    ap.add_argument("--eval_train_changed", type=int, default=40)
    ap.add_argument("--eval_train_stable", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--selector_batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--reader_tag_epochs", type=int, default=5)
    ap.add_argument("--reader_direct_epochs", type=int, default=8)
    ap.add_argument("--selector_epochs", type=int, default=8)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--selector_pos_weight", type=float, default=3.0)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # Reader uses the proven head-calibrated path: tag-only direct then inline-role direct.
    reader_tag_tr, reader_tag_ev, reader_tag_con = st270.build_train_eval(args, "tag_only", "direct_tag")
    reader_direct_tr, reader_direct_ev, reader_direct_con = st270.build_train_eval(args, args.mode, "direct_tag")
    sel_tr, sel_con = selector_train_rows(args)
    sel_evals = selector_eval_suite(args)

    all_rows = reader_tag_tr + reader_direct_tr + sel_tr
    for d in [reader_tag_ev, reader_direct_ev, sel_evals]:
        for rows in d.values():
            all_rows += rows
    ta = token_audit(tok, all_rows, args.max_len)
    construction = {
        "reader_tag_train_rows": len(reader_tag_tr),
        "reader_train_rows": len(reader_direct_tr),
        "selector_train_rows": len(sel_tr),
        "reader_tag_train": reader_tag_con,
        "reader_direct_train": reader_direct_con,
        "selector_train": sel_con,
        "selector_eval_counts": {k: len(v) for k, v in sel_evals.items()},
        "selector_group_audit": {"train": count_groups(sel_tr),
                                  **{k: count_groups(v) for k, v in sel_evals.items()}},
        "token_audit": ta,
    }
    print(json.dumps({"status": "BUILT", "reader_tag_rows": len(reader_tag_tr),
                      "reader_direct_rows": len(reader_direct_tr),
                      "selector_rows": len(sel_tr),
                      "selector_evals": {k: len(v) for k, v in sel_evals.items()},
                      "token_audit": ta,
                      "selector_group_audit": construction["selector_group_audit"]}), flush=True)

    if args.dry_build:
        summary = {"status": "DRY", "args": vars(args), "construction": construction,
                   "boundary": "Dry construction only; no model training."}
        (out / "selector_reader_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "selector_reader_summary.json")}), flush=True)
        return

    dev = args.device
    results: dict[str, Any] = {}

    # Train and freeze R.
    reader = st270.make_model(Path(args.model_path), dev, seed=args.seed)
    results["reader_tag_pretrain"] = st270.train_loop(
        reader, tok, reader_tag_tr, args, dev, seed=args.seed,
        epochs=args.reader_tag_epochs, label="reader_tag_pretrain",
        init_best_from_current=False, track_eval_rows=reader_tag_ev)
    results["reader_inline_direct"] = st270.train_loop(
        reader, tok, reader_direct_tr, args, dev, seed=args.seed + 1,
        epochs=args.reader_direct_epochs, label="reader_inline_direct",
        init_best_from_current=True, track_eval_rows=reader_direct_ev)
    reader_direct_eval = st270.summarize_eval(st270.eval_sets(reader, tok, reader_direct_ev, args, dev))

    # Train M as a separate model.
    selector = st270.make_model(Path(args.model_path), dev, seed=args.seed + 100)
    results["selector_train"] = train_selector(
        selector, tok, sel_tr, args, dev, seed=args.seed + 100,
        epochs=args.selector_epochs, label="selector_train", track_evals=sel_evals)
    selector_evals = selector_eval(selector, tok, sel_evals, args, dev)
    comp = composition_eval(selector, reader, tok, sel_evals, args, dev)

    summary = {"status": "DONE", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "results": results,
               "reader_direct_eval": reader_direct_eval,
               "selector_evals": selector_evals, "composition": comp,
               "boundary": "Small pretrained selector-reader bridge. R is trained then held fixed; M is separate; composition uses M-selected tags, with oracle-reader ceiling reported only as a reference."}
    (out / "selector_reader_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "json": str(out / "selector_reader_summary.json"),
                      "md": str(out / "selector_reader_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
