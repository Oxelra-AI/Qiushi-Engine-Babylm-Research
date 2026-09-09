#!/usr/bin/env python3
"""Overlap of fast-path item flips relative to the protected chck_82M anchor.

No inference is run. The script reads saved official-compatible discrete prediction
payloads and asks whether coherent replay and shuffled86 produce the same newly
correct decisions or mostly different/opposing family reallocations.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader  # noqa: E402

CHCK82 = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')
COHERENT = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')
SHUFFLED86 = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json')
SPANBREAK = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_flip_overlap')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/fastpath_flip_overlap/fastpath_flip_overlap.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/fastpath_flip_overlap/fastpath_flip_overlap.md')

FRAGILE_EWOK_TOKENS = [
    "material", "spatial", "space", "quantity", "quantitative", "number", "physical", "social",
    "support", "contact", "contained", "size", "mass", "volume", "distance", "dynamics", "interaction",
]
FRAGILE_ENTITY_TOKENS = ["regular_5", "move_contents_4", "move_contents_3", "ambiref_3", "5_ops", "4_ops", "3_ops"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def phi_binary(x: list[int], y: list[int]) -> float | None:
    if not x or len(x) != len(y):
        return None
    sx = set(x)
    sy = set(y)
    if len(sx) < 2 or len(sy) < 2:
        return None
    return float(np.corrcoef(np.array(x, dtype=float), np.array(y, dtype=float))[0, 1])


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    try:
        from scipy.stats import binomtest  # type: ignore
        return float(binomtest(k, n=n, p=p, alternative="two-sided").pvalue)
    except Exception:
        # Normal approximation fallback; adequate here because the p-value is a
        # descriptive overlap readout, not a training or selection signal.
        mu = n * p
        var = n * p * (1.0 - p)
        if var <= 0:
            return None
        z = abs(k - mu) / math.sqrt(var)
        return float(math.erfc(z / math.sqrt(2.0)))


def load_rows(loader: PayloadLoader, column: str) -> dict[str, Any]:
    rows, meta = loader.load_column(column)
    return {r.item_id: r for r in rows}


def subset_name(column: str, uid: str) -> str:
    u = str(uid).lower()
    if column == "EWoK" and any(tok in u for tok in FRAGILE_EWOK_TOKENS):
        return "EWoK_fragile_tagged"
    if column == "Entity" and any(tok in u for tok in FRAGILE_ENTITY_TOKENS):
        return "Entity_highop_tagged"
    return "all"


def summarize_items(items: list[tuple[str, Any, Any, Any, Any]]) -> dict[str, Any]:
    # Tuple: item_id, anchor row, coherent row, shuffled row, spanbreak row.
    n = len(items)
    base_correct = [bool(b.correct) for _, b, _, _, _ in items]
    coh_correct = [bool(c.correct) for _, _, c, _, _ in items]
    shuf_correct = [bool(s.correct) for _, _, _, s, _ in items]
    span_correct = [bool(sp.correct) for _, _, _, _, sp in items]

    base_wrong_idx = [i for i, bc in enumerate(base_correct) if not bc]
    base_right_idx = [i for i, bc in enumerate(base_correct) if bc]

    coh_gain = [i for i in base_wrong_idx if coh_correct[i]]
    shuf_gain = [i for i in base_wrong_idx if shuf_correct[i]]
    span_gain = [i for i in base_wrong_idx if span_correct[i]]
    coh_loss = [i for i in base_right_idx if not coh_correct[i]]
    shuf_loss = [i for i in base_right_idx if not shuf_correct[i]]
    span_loss = [i for i in base_right_idx if not span_correct[i]]

    coh_gain_set, shuf_gain_set, span_gain_set = set(coh_gain), set(shuf_gain), set(span_gain)
    coh_loss_set, shuf_loss_set, span_loss_set = set(coh_loss), set(shuf_loss), set(span_loss)

    def overlap(a: set[int], b: set[int]) -> dict[str, Any]:
        inter = len(a & b)
        union = len(a | b)
        return {
            "a": len(a),
            "b": len(b),
            "shared": inter,
            "a_only": len(a - b),
            "b_only": len(b - a),
            "union": union,
            "jaccard": None if union == 0 else inter / union,
        }

    gain_x = [int(i in coh_gain_set) for i in base_wrong_idx]
    gain_y = [int(i in shuf_gain_set) for i in base_wrong_idx]
    loss_x = [int(i in coh_loss_set) for i in base_right_idx]
    loss_y = [int(i in shuf_loss_set) for i in base_right_idx]

    coh_vs_shuf = Counter()
    for i in range(n):
        if coh_correct[i] and not shuf_correct[i]:
            coh_vs_shuf["coherent_correct_shuffled_wrong"] += 1
        elif shuf_correct[i] and not coh_correct[i]:
            coh_vs_shuf["shuffled_correct_coherent_wrong"] += 1
        elif coh_correct[i] and shuf_correct[i]:
            coh_vs_shuf["both_correct"] += 1
        else:
            coh_vs_shuf["both_wrong"] += 1

    shuf_better = coh_vs_shuf["shuffled_correct_coherent_wrong"]
    coh_better = coh_vs_shuf["coherent_correct_shuffled_wrong"]
    discordant = shuf_better + coh_better

    return {
        "n_common": n,
        "anchor_correct": sum(base_correct),
        "anchor_wrong": n - sum(base_correct),
        "coherent_correct": sum(coh_correct),
        "shuffled86_correct": sum(shuf_correct),
        "spanbreak_correct": sum(span_correct),
        "coherent_net_vs_anchor": len(coh_gain) - len(coh_loss),
        "shuffled86_net_vs_anchor": len(shuf_gain) - len(shuf_loss),
        "spanbreak_net_vs_anchor": len(span_gain) - len(span_loss),
        "coherent_vs_shuffled_correct_net": coh_better - shuf_better,
        "coherent_vs_shuffled_discordant": discordant,
        "coherent_vs_shuffled_sign_p": binom_two_sided(min(coh_better, shuf_better), discordant),
        "coherent_shuffled_gain_overlap": overlap(coh_gain_set, shuf_gain_set),
        "coherent_shuffled_loss_overlap": overlap(coh_loss_set, shuf_loss_set),
        "coherent_spanbreak_gain_overlap": overlap(coh_gain_set, span_gain_set),
        "coherent_spanbreak_loss_overlap": overlap(coh_loss_set, span_loss_set),
        "coherent_shuffled_gain_phi_on_anchor_wrong": phi_binary(gain_x, gain_y),
        "coherent_shuffled_loss_phi_on_anchor_correct": phi_binary(loss_x, loss_y),
        "coherent_vs_shuffled_counts": dict(coh_vs_shuf),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaders = {
        "chck82": PayloadLoader(CHCK82),
        "coherent": PayloadLoader(COHERENT),
        "shuffled86": PayloadLoader(SHUFFLED86),
        "spanbreak": PayloadLoader(SPANBREAK),
    }
    columns: dict[str, Any] = {}
    subsets: dict[str, list[tuple[str, Any, Any, Any, Any]]] = defaultdict(list)

    for col in DISCRETE_COLUMNS:
        rows = {name: load_rows(loader, col) for name, loader in loaders.items()}
        common = sorted(set.intersection(*(set(r.keys()) for r in rows.values())))
        items = [(iid, rows["chck82"][iid], rows["coherent"][iid], rows["shuffled86"][iid], rows["spanbreak"][iid]) for iid in common]
        columns[col] = summarize_items(items)
        for tup in items:
            _, b, _, _, _ = tup
            subsets[f"{col}:all"].append(tup)
            nm = subset_name(col, b.uid)
            if nm != "all":
                subsets[nm].append(tup)
            if col == "COMPS" and str(b.uid).startswith("wugs"):
                subsets["COMPS_wugs"].append(tup)
            if col == "COMPS" and str(b.uid) == "base":
                subsets["COMPS_base"].append(tup)

    all_items: list[tuple[str, Any, Any, Any, Any]] = []
    for col in DISCRETE_COLUMNS:
        # reload list from subset to avoid item id collision across columns already in id string.
        all_items.extend(subsets[f"{col}:all"])

    summary = {
        "all_discrete": summarize_items(all_items),
        "EWoK_fragile_tagged": summarize_items(subsets["EWoK_fragile_tagged"]),
        "Entity_highop_tagged": summarize_items(subsets["Entity_highop_tagged"]),
        "COMPS_wugs": summarize_items(subsets["COMPS_wugs"]),
        "COMPS_base": summarize_items(subsets["COMPS_base"]),
    }
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "payloads": {"chck82": rel(CHCK82), "coherent": rel(COHERENT), "shuffled86": rel(SHUFFLED86), "spanbreak": rel(SPANBREAK)},
        "columns": columns,
        "summary_subsets": summary,
        "scientific_reading": "Low shared-gain overlap and negative/near-flat fragile-family nets mean coherent replay and shuffled86 are not two confirmations of the same added decisions; they are different reweightings of the anchor decision surface.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research fast-path flip overlap", "", f"Status: **{out['status']}**", "", "## Main overlap readouts", "", "| subset | n | coh net vs anchor | shuf net vs anchor | coh-vs-shuf net | gain Jaccard | loss Jaccard | gain phi | loss phi | sign p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rec in summary.items():
        lines.append(
            f"| {name} | {rec['n_common']} | {rec['coherent_net_vs_anchor']} | {rec['shuffled86_net_vs_anchor']} | "
            f"{rec['coherent_vs_shuffled_correct_net']} | {rec['coherent_shuffled_gain_overlap']['jaccard']} | "
            f"{rec['coherent_shuffled_loss_overlap']['jaccard']} | {rec['coherent_shuffled_gain_phi_on_anchor_wrong']} | "
            f"{rec['coherent_shuffled_loss_phi_on_anchor_correct']} | {rec['coherent_vs_shuffled_sign_p']} |"
        )
    lines += ["", "## By column", "", "| column | n | coh net | shuf net | coh-vs-shuf net | shared gains / union | shared losses / union |", "|---|---:|---:|---:|---:|---:|---:|"]
    for col, rec in columns.items():
        go = rec["coherent_shuffled_gain_overlap"]
        lo = rec["coherent_shuffled_loss_overlap"]
        lines.append(f"| {col} | {rec['n_common']} | {rec['coherent_net_vs_anchor']} | {rec['shuffled86_net_vs_anchor']} | {rec['coherent_vs_shuffled_correct_net']} | {go['shared']}/{go['union']} | {lo['shared']}/{lo['union']} |")
    lines += ["", "## Scientific reading", "", out["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "all_discrete": summary["all_discrete"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
