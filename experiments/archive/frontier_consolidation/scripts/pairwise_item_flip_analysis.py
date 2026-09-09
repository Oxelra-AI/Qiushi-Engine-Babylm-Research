#!/usr/bin/env python3
"""research: item-level flip analysis for two saved official-compatible eval payloads.

Default comparison: research legal chck_50M vs adapter128 scale1.75 chck_50M.
Only saved prediction JSON and official evaluation data are read; no model inference.
The parser is adapted from the validated research item reconstruction.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(".")
DEFAULT_BASE = Path("experiments/archive/frontier_consolidation/data/50M_eval/eval/per_target/legal_chck50M.json")
DEFAULT_CAND = Path("experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/eval/per_target/adapter128_scale1p75_h100M50M_seed43022.json")
DEFAULT_OUT = Path("experiments/archive/frontier_consolidation/data/scale1p75_50m_item_flips")
DISCRETE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pred_nested(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = load_json(path)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for uid, rec in raw.items():
        if isinstance(rec, dict):
            out[str(uid)] = rec.get("predictions", [])
        elif isinstance(rec, list):
            out[str(uid)] = rec
        else:
            out[str(uid)] = []
    return out


def get_pred(preds: Dict[str, List[Dict[str, Any]]], uid: str, idx: int) -> str | None:
    arr = preds.get(str(uid))
    if arr is None or idx >= len(arr):
        return None
    return norm_text(arr[idx].get("pred"))


def parse_report_average(path: Path) -> float | None:
    if not path or not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("### AVERAGE") and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except Exception:
                return None
    return None


@dataclass
class ItemRow:
    item_id: str
    uid: str
    correct: bool
    pred: str | None
    gold: str
    column: str
    sub: str | None = None
    meta: Dict[str, Any] | None = None


def official_score(rows: List[ItemRow], column: str) -> float:
    if not rows:
        return float("nan")
    if column == "GlobalPIQA":
        sub_to_rows: Dict[str, List[ItemRow]] = defaultdict(list)
        for r in rows:
            sub_to_rows[str(r.sub)].append(r)
        sub_scores=[]
        for sub_rows in sub_to_rows.values():
            uid_stats: Dict[str, List[int]] = defaultdict(lambda:[0,0])
            for r in sub_rows:
                uid_stats[r.uid][0] += int(r.correct)
                uid_stats[r.uid][1] += 1
            sub_scores.append(100.0 * float(np.mean([c/n for c,n in uid_stats.values()])))
        return float(np.mean(sub_scores)) if sub_scores else float("nan")
    uid_stats: Dict[str, List[int]] = defaultdict(lambda:[0,0])
    for r in rows:
        uid_stats[r.uid][0] += int(r.correct)
        uid_stats[r.uid][1] += 1
    uid_acc={uid:100.0*c/n for uid,(c,n) in uid_stats.items()}
    if column == "Entity":
        vals=[]
        for split in ["regular", "ambiref", "move_contents"]:
            svals=[acc for uid, acc in uid_acc.items() if uid.startswith(split)]
            if svals:
                vals.append(float(np.mean(svals)))
        return float(np.mean(vals)) if vals else float("nan")
    return float(np.mean(list(uid_acc.values())))


def report_score_from_payload(payload: Dict[str, Any], column: str) -> float | None:
    scores = payload.get("official_overall", {}).get("scores", {})
    if column == "GlobalPIQA":
        return scores.get("GlobalPIQA")
    return scores.get(column)


class PayloadLoader:
    def __init__(self, payload_path: Path):
        self.path = payload_path
        self.payload = load_json(payload_path)
        self.tasks = self.payload["tasks"]

    def rec(self, column_or_sub: str) -> Dict[str, Any]:
        return self.tasks[column_or_sub]

    def load_blimp_like(self, column: str) -> Tuple[List[ItemRow], Dict[str, Any]]:
        rec = self.rec(column)
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        counters: Counter[str] = Counter()
        rows=[]; missing=0
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                if "field" in raw:
                    uid = str(raw["UID"])
                else:
                    uid = file_path.stem
                idx = counters[uid]; counters[uid]+=1
                pred = get_pred(preds, uid, idx)
                if pred is None: missing += 1
                gold = norm_text(raw["sentence_good"])
                rows.append(ItemRow(f"{column}:{uid}:{idx}", uid, pred == gold, pred, gold, column, meta={"file":file_path.stem}))
        return rows, {"missing_predictions": missing, "uids": len(counters), "report_average": parse_report_average(ROOT / rec.get("report", ""))}

    def load_ewok(self) -> Tuple[List[ItemRow], Dict[str, Any]]:
        rec = self.rec("EWoK")
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        counters: Counter[str] = Counter()
        rows=[]; missing=0; match_counts=Counter()
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                uid = str(raw["Domain"])
                idx = counters[uid]; counters[uid]+=1
                pred = get_pred(preds, uid, idx)
                if pred is None: missing += 1
                good = norm_text(" ".join([raw["Context1"], raw["Target1"]]))
                bad = norm_text(" ".join([raw["Context2"], raw["Target1"]]))
                if pred == good: match_counts["good"] += 1
                elif pred == bad: match_counts["bad"] += 1
                else: match_counts["other"] += 1
                rows.append(ItemRow(f"EWoK:{uid}:{idx}", uid, pred == good, pred, good, "EWoK", meta={"Domain":uid, "file":file_path.stem}))
        return rows, {"missing_predictions": missing, "uids": len(counters), "match_counts": dict(match_counts), "report_average": parse_report_average(ROOT / rec.get("report", ""))}

    def load_entity(self) -> Tuple[List[ItemRow], Dict[str, Any]]:
        rec = self.rec("Entity")
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        counters: Counter[str] = Counter()
        rows=[]; missing=0; skipped=0
        for file_path in sorted(data_dir.glob("*.jsonl")):
            family=file_path.stem
            for raw in read_jsonl(file_path):
                if any("nothing" in str(opt) for opt in raw["options"]):
                    skipped += 1; continue
                uid = f"{family}_{int(raw['numops'])}_ops"
                idx = counters[uid]; counters[uid]+=1
                pred = get_pred(preds, uid, idx)
                if pred is None: missing += 1
                gold = norm_text(raw["options"][0])
                rows.append(ItemRow(f"Entity:{uid}:{idx}", uid, pred == gold, pred, gold, "Entity", meta={"family":family, "numops":int(raw['numops'])}))
        return rows, {"missing_predictions": missing, "uids": len(counters), "skipped_nothing_rows": skipped, "report_average": parse_report_average(ROOT / rec.get("report", ""))}

    def load_comps(self) -> Tuple[List[ItemRow], Dict[str, Any]]:
        rec = self.rec("COMPS")
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        file_to_uid={"comps_base.jsonl":"base","comps_wugs.jsonl":"wugs","comps_wugs_dist-before.jsonl":"wugs_dist_before","comps_wugs_dist-in-between.jsonl":"wugs_dist_in_between"}
        counters: Counter[str]=Counter()
        rows=[]; missing=0
        for filename, uid in file_to_uid.items():
            for raw in read_jsonl(data_dir/filename):
                idx=counters[uid]; counters[uid]+=1
                pred=get_pred(preds, uid, idx)
                if pred is None: missing += 1
                gold=norm_text(" ".join([raw["prefix_acceptable"], raw["property_phrase"]]))
                rows.append(ItemRow(f"COMPS:{uid}:{idx}", uid, pred == gold, pred, gold, "COMPS", meta={"file":filename, "uid":uid}))
        return rows, {"missing_predictions": missing, "uids": len(counters), "report_average": parse_report_average(ROOT / rec.get("report", ""))}

    def load_global_sub(self, sub: str) -> Tuple[List[ItemRow], Dict[str, Any]]:
        rec = self.rec(sub)
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        rows=[]; missing=0
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                uid = str(raw["example_id"])
                pred = get_pred(preds, uid, 0)
                if pred is None:
                    raw_rec = preds.get(uid)
                    if raw_rec and raw_rec:
                        pred = norm_text(raw_rec[0].get("pred"))
                    else:
                        missing += 1
                label = int(raw["label"])
                gold = norm_text(" " + raw[f"solution{label}"])
                rows.append(ItemRow(f"GlobalPIQA:{sub}:{uid}", uid, pred == gold, pred, gold, "GlobalPIQA", sub=sub, meta={"sub":sub, "file":file_path.stem}))
        return rows, {"missing_predictions": missing, "n_items": len(rows), "report_average": parse_report_average(ROOT / rec.get("report", "")), "task_score_record": rec.get("score")}

    def load_globalpiqa(self) -> Tuple[List[ItemRow], Dict[str, Any]]:
        all_rows=[]; meta={}
        for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rows, m = self.load_global_sub(sub)
            all_rows.extend(rows); meta[sub]=m
        return all_rows, meta

    def load_column(self, column: str) -> Tuple[List[ItemRow], Dict[str, Any]]:
        if column in ("BLiMP", "Supplement"):
            return self.load_blimp_like(column)
        if column == "EWoK":
            return self.load_ewok()
        if column == "Entity":
            return self.load_entity()
        if column == "COMPS":
            return self.load_comps()
        if column == "GlobalPIQA":
            return self.load_globalpiqa()
        raise KeyError(column)


def uid_group(column: str, row: ItemRow) -> str:
    if column == "BLiMP":
        return row.uid
    if column == "Supplement":
        return row.uid
    if column == "EWoK":
        return row.uid
    if column == "Entity":
        return row.uid
    if column == "COMPS":
        return row.uid
    if column == "GlobalPIQA":
        return str(row.sub)
    return row.uid


def compare_column(base_loader: PayloadLoader, cand_loader: PayloadLoader, column: str) -> Dict[str, Any]:
    base_rows, base_meta = base_loader.load_column(column)
    cand_rows, cand_meta = cand_loader.load_column(column)
    bmap={r.item_id:r for r in base_rows}
    cmap={r.item_id:r for r in cand_rows}
    common=sorted(set(bmap)&set(cmap))
    only_base=sorted(set(bmap)-set(cmap))
    only_cand=sorted(set(cmap)-set(bmap))
    base_common=[bmap[i] for i in common]
    cand_common=[cmap[i] for i in common]
    base_score=official_score(base_common, column)
    cand_score=official_score(cand_common, column)
    payload_base_score=report_score_from_payload(base_loader.payload, column)
    payload_cand_score=report_score_from_payload(cand_loader.payload, column)
    flips=Counter(); group=defaultdict(lambda: Counter())
    examples={"gain":[], "loss":[], "both_correct":[], "both_wrong":[]}
    for item_id in common:
        b=bmap[item_id]; c=cmap[item_id]
        if (not b.correct) and c.correct:
            typ="gain"
        elif b.correct and (not c.correct):
            typ="loss"
        elif b.correct and c.correct:
            typ="both_correct"
        else:
            typ="both_wrong"
        flips[typ]+=1
        group[uid_group(column, b)][typ]+=1
        group[uid_group(column, b)]["n"]+=1
        if len(examples[typ]) < 8:
            examples[typ].append({"item_id": item_id, "uid": b.uid, "sub": b.sub, "base_pred": b.pred, "cand_pred": c.pred, "gold": b.gold, "base_correct": b.correct, "cand_correct": c.correct, "meta": b.meta})
    group_rows=[]
    for g,cnt in group.items():
        n=cnt["n"]
        # Baseline group item accuracy = both_correct + loss (correct under base = both_correct + base-only-correct=loss).
        base_correct=cnt["both_correct"]+cnt["loss"]
        cand_correct=cnt["both_correct"]+cnt["gain"]
        b_acc=base_correct/n if n else 0
        c_acc=cand_correct/n if n else 0
        group_rows.append({"group":g,"n":n,"gain":cnt["gain"],"loss":cnt["loss"],"both_correct":cnt["both_correct"],"both_wrong":cnt["both_wrong"],"net_gain_minus_loss":cnt["gain"]-cnt["loss"],"net_pct":100.0*(cnt["gain"]-cnt["loss"])/n,"base_item_pct":100*b_acc,"cand_item_pct":100*c_acc})
    group_rows=sorted(group_rows, key=lambda x:(x["net_pct"], x["net_gain_minus_loss"]), reverse=True)
    worst_groups=sorted(group_rows, key=lambda x:(x["net_pct"], x["net_gain_minus_loss"]))[:12]
    best_groups=group_rows[:12]
    groups_by_name={r["group"]: r for r in group_rows}
    total=len(common)
    return {
        "column":column,
        "n_base":len(base_rows),"n_cand":len(cand_rows),"n_common":total,"only_base":len(only_base),"only_cand":len(only_cand),
        "base_score_reconstructed_common":base_score,
        "cand_score_reconstructed_common":cand_score,
        "delta_score_reconstructed_common":cand_score-base_score,
        "base_score_payload":payload_base_score,
        "cand_score_payload":payload_cand_score,
        "delta_score_payload":None if payload_base_score is None or payload_cand_score is None else payload_cand_score-payload_base_score,
        "payload_reconstruction_errors":{"base":None if payload_base_score is None else base_score-payload_base_score,"candidate":None if payload_cand_score is None else cand_score-payload_cand_score},
        "flip_counts":dict(flips),
        "flip_pct":{k:100.0*v/total for k,v in flips.items()},
        "gain_minus_loss_items":flips["gain"]-flips["loss"],
        "gain_minus_loss_pct":100.0*(flips["gain"]-flips["loss"])/total if total else 0,
        "groups_all_by_item_net":group_rows,
        "groups_by_name":groups_by_name,
        "best_groups_by_item_net":best_groups,
        "worst_groups_by_item_net":worst_groups,
        "base_meta":base_meta,
        "cand_meta":cand_meta,
        "examples":examples,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base", default=str(DEFAULT_BASE))
    p.add_argument("--candidate", default=str(DEFAULT_CAND))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--label", default="scale1p75_vs_step35_50M")
    args=p.parse_args()
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_loader=PayloadLoader(Path(args.base))
    cand_loader=PayloadLoader(Path(args.candidate))
    cols={}
    for col in DISCRETE_COLUMNS:
        cols[col]=compare_column(base_loader, cand_loader, col)
    base_scores=base_loader.payload.get("official_overall",{}).get("scores",{})
    cand_scores=cand_loader.payload.get("official_overall",{}).get("scores",{})
    cheap_payload={
        "base":{c:base_scores.get(c) for c in CHEAP_COLS},
        "candidate":{c:cand_scores.get(c) for c in CHEAP_COLS},
    }
    cheap_payload["deltas"]={c:(None if cheap_payload["base"].get(c) is None or cheap_payload["candidate"].get(c) is None else cheap_payload["candidate"][c]-cheap_payload["base"][c]) for c in CHEAP_COLS}
    cheap_payload["base_cheap7"] = sum(float(cheap_payload["base"][c]) for c in CHEAP_COLS)/7
    cheap_payload["candidate_cheap7"] = sum(float(cheap_payload["candidate"][c]) for c in CHEAP_COLS)/7
    cheap_payload["cheap7_delta"] = cheap_payload["candidate_cheap7"] - cheap_payload["base_cheap7"]
    aggregate={
        "discrete_reconstructed_mean_delta": float(np.mean([cols[c]["delta_score_reconstructed_common"] for c in DISCRETE_COLUMNS])),
        "discrete_payload_mean_delta": float(np.mean([cols[c]["delta_score_payload"] for c in DISCRETE_COLUMNS if cols[c]["delta_score_payload"] is not None])),
        "total_gain_items": int(sum(cols[c]["flip_counts"].get("gain",0) for c in DISCRETE_COLUMNS)),
        "total_loss_items": int(sum(cols[c]["flip_counts"].get("loss",0) for c in DISCRETE_COLUMNS)),
        "total_common_items": int(sum(cols[c]["n_common"] for c in DISCRETE_COLUMNS)),
    }
    aggregate["total_gain_minus_loss"] = aggregate["total_gain_items"] - aggregate["total_loss_items"]
    aggregate["total_gain_minus_loss_pct"] = 100.0 * aggregate["total_gain_minus_loss"] / aggregate["total_common_items"]
    interpretation=[]
    interpretation.append(f"Payload cheap7 delta is {cheap_payload['cheap7_delta']:+.4f}; discrete reconstructed mean delta is {aggregate['discrete_reconstructed_mean_delta']:+.4f} over {aggregate['total_common_items']} item rows.")
    for col in DISCRETE_COLUMNS:
        c=cols[col]
        interpretation.append(f"{col}: reconstructed delta {c['delta_score_reconstructed_common']:+.3f}, item gains {c['flip_counts'].get('gain',0)}, losses {c['flip_counts'].get('loss',0)}, net {c['gain_minus_loss_items']:+d}; strongest groups +{c['best_groups_by_item_net'][:3]} / -{c['worst_groups_by_item_net'][:3]}.")
    # Specific scientific readouts for current route.
    if "Entity" in cols:
        ent=cols["Entity"]
        pos=[g for g in ent["best_groups_by_item_net"] if g["net_gain_minus_loss"]>0][:5]
        neg=[g for g in ent["worst_groups_by_item_net"] if g["net_gain_minus_loss"]<0][:5]
        interpretation.append(f"Entity net positives are {pos}; net negatives are {neg}. This tests whether the 50M adapter gain is concentrated in high-operation state tracking.")
    if "EWoK" in cols:
        ew=cols["EWoK"]
        interpretation.append(f"EWoK best/worst domain groups are {ew['best_groups_by_item_net'][:5]} / {ew['worst_groups_by_item_net'][:5]}; compare these to research subtask report losses in active-passive, material, number, and quantitative properties.")
    out={
        "status":"PAIRWISE_ITEM_FLIP_ANALYSIS",
        "label":args.label,
        "base_payload":args.base,
        "candidate_payload":args.candidate,
        "columns":cols,
        "cheap_payload":cheap_payload,
        "aggregate":aggregate,
        "interpretation":interpretation,
    }
    out_json=out_dir/f"{args.label}.json"
    out_md=out_dir/f"{args.label}.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines=[]
    lines.append(f"# research pairwise item-flip analysis: {args.label}")
    lines.append("")
    lines.append(f"Base: `{args.base}`")
    lines.append(f"Candidate: `{args.candidate}`")
    lines.append("")
    lines.append("## Column summary")
    lines.append("")
    lines.append("| column | payload Δ | reconstructed Δ | common rows | gains | losses | net gain-loss | best groups | worst groups |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for col in DISCRETE_COLUMNS:
        c=cols[col]
        best=", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["best_groups_by_item_net"][:4])
        worst=", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["worst_groups_by_item_net"][:4])
        pd=c["delta_score_payload"]
        lines.append(f"| {col} | {pd:+.3f} | {c['delta_score_reconstructed_common']:+.3f} | {c['n_common']} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} | {best} | {worst} |")
    lines.append("")
    lines.append("## Cheap-column payload scores")
    lines.append("")
    lines.append(f"- base cheap7: {cheap_payload['base_cheap7']:.4f}")
    lines.append(f"- candidate cheap7: {cheap_payload['candidate_cheap7']:.4f}")
    lines.append(f"- Δcheap7: {cheap_payload['cheap7_delta']:+.4f}")
    lines.append(f"- deltas: {cheap_payload['deltas']}")
    lines.append("")
    lines.append("## Interpretation")
    for x in interpretation:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Example flips")
    for col in DISCRETE_COLUMNS:
        lines.append(f"### {col}")
        for typ in ["gain","loss"]:
            lines.append(f"#### {typ}")
            for ex in cols[col]["examples"][typ][:5]:
                gold=str(ex['gold']).replace('|','/')[:160]
                bp=str(ex['base_pred']).replace('|','/')[:100]
                cp=str(ex['cand_pred']).replace('|','/')[:100]
                lines.append(f"- `{ex['item_id']}` uid={ex['uid']} gold={gold!r} base={bp!r} cand={cp!r}")
    out_md.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":out["status"],"label":args.label,"out_json":str(out_json),"out_md":str(out_md),"cheap7_delta":cheap_payload["cheap7_delta"],"discrete_mean_delta":aggregate["discrete_reconstructed_mean_delta"],"total_net_items":aggregate["total_gain_minus_loss"]}, indent=2))

if __name__ == "__main__":
    main()
