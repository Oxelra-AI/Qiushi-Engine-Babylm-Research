#!/usr/bin/env python3
"""Collate research common-screen evaluations and compare real-pretraining candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

EQ7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]


def load_eval(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    scores = obj.get("scores", {})
    return {"path": str(path), "tag": obj.get("tag", path.stem), "scores": scores, "source": obj.get("source_model_path") or obj.get("model_path")}


def eq(scores: Dict[str, Any]) -> tuple[Optional[float], int]:
    vals = [scores.get(k) for k in EQ7]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/archive/functional_learning/data/common_eval")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--out", default="experiments/archive/functional_learning/data/common_eval/collated_common_eval.json")
    args = ap.parse_args()

    root = Path(args.root)
    paths = []
    out_target = Path(args.out)
    for p in root.rglob("*_eval.json"):
        if p.resolve() == out_target.resolve():
            continue
        paths.append(p)
    for e in args.extra:
        p = Path(e)
        if p.exists():
            paths.append(p)
    uniq = []
    seen = set()
    for p in sorted(paths):
        s = str(p)
        if s not in seen:
            uniq.append(p); seen.add(s)

    rows = []
    for p in uniq:
        try:
            r = load_eval(p)
        except Exception as exc:
            rows.append({"path": str(p), "error": str(exc)})
            continue
        scores = r["scores"]
        e, n = eq(scores)
        r["equal7_or_valid_mean"] = e
        r["n_equal_columns"] = n
        rows.append(r)

    by_tag = {r.get("tag", Path(r.get("path", "")).stem): r for r in rows if "scores" in r}
    refs = [t for t in by_tag if "chck82" in t.lower() or "coherent86" in t.lower()]
    comparisons = []
    for tag, r in by_tag.items():
        scores = r["scores"]
        for ref_tag in refs:
            if ref_tag == tag:
                continue
            ref = by_tag[ref_tag]["scores"]
            deltas = {k: (None if scores.get(k) is None or ref.get(k) is None else float(scores[k]) - float(ref[k])) for k in EQ7}
            e, n = eq(scores); er, nr = eq(ref)
            comparisons.append({
                "tag": tag,
                "reference": ref_tag,
                "delta_equal_valid_mean": None if e is None or er is None or n != nr else e - er,
                "n_common_equal_columns": min(n, nr),
                "deltas": deltas,
            })

    out = {"status": "COMMON_EVAL_COLLATION", "eval_files": [str(p) for p in uniq], "rows": rows, "comparisons": comparisons}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("tag\tn\teq_mean\t" + "\t".join(EQ7))
    for r in rows:
        if "scores" not in r:
            print(f"{r.get('path')}\tERROR\t{r.get('error')}")
            continue
        s = r["scores"]
        print("\t".join([str(r["tag"]), str(r["n_equal_columns"]), fmt(r["equal7_or_valid_mean"])] + [fmt(s.get(k)) for k in EQ7]))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
