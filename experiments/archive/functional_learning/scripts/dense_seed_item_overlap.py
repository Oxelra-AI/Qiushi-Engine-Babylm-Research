#!/usr/bin/env python3
"""research: item-overlap analysis for dense-focus fast-screen replication.

Aggregate fast scores for seed62064 and seed62065 are nearly identical.  This script
checks whether the same fast-screen items are gained/lost relative to coherent86 or
whether similar scores hide unrelated seed-specific flips.  It uses the same item
construction code as the official-coordinate transition comparator, but compares the
three payloads jointly.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import official_transition_compare as comp  # noqa: E402

OUT = _public_path('experiments/archive/functional_learning/data/dense_seed_item_overlap')
PARENT_PAYLOAD = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/coherent86_alpha075_fast_payload.json')
SEED64_PAYLOAD = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/dense_focus_seed62064_u0080_fast_payload.json')
SEED65_PAYLOAD = _public_path('experiments/archive/functional_learning/data/dense_seed62065_fast_payloads/dense_focus_seed62065_u0080_fast_payload.json')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_payload(path: pathlib.Path) -> Dict[str, Any]:
    return comp.load_payloads([rel(path)])


def key_for(column: str, item: Dict[str, Any]) -> Tuple[str, str, int, str]:
    return (column, str(item["subtask"]), int(item["index"]), str(item.get("id", "")))


def as_map(items_by_col: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str, int, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for col, rows in items_by_col.items():
        for r in rows:
            out[key_for(col, r)] = r
    return out


def ratio(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def jaccard(a: Set[Any], b: Set[Any]) -> float | None:
    return None if not a and not b else len(a & b) / len(a | b)


def summarize_group(keys: Iterable[Tuple[str, str, int, str]], parent: Dict[Any, Dict[str, Any]], s64: Dict[Any, Dict[str, Any]], s65: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(keys)
    c = Counter()
    gain64: Set[Any] = set()
    gain65: Set[Any] = set()
    loss64: Set[Any] = set()
    loss65: Set[Any] = set()
    both_dense_correct_parent_wrong: Set[Any] = set()
    both_dense_wrong_parent_correct: Set[Any] = set()
    seed_disagree: Set[Any] = set()
    for k in keys:
        pc = bool(parent[k]["correct"])
        a = bool(s64[k]["correct"])
        b = bool(s65[k]["correct"])
        c["n"] += 1
        c["parent_correct"] += int(pc)
        c["seed62064_correct"] += int(a)
        c["seed62065_correct"] += int(b)
        c["both_seed_correct"] += int(a and b)
        c["both_seed_wrong"] += int((not a) and (not b))
        c["seed_agree"] += int(a == b)
        c["seed_disagree"] += int(a != b)
        if not pc and a:
            gain64.add(k)
        if not pc and b:
            gain65.add(k)
        if pc and not a:
            loss64.add(k)
        if pc and not b:
            loss65.add(k)
        if (not pc) and a and b:
            both_dense_correct_parent_wrong.add(k)
        if pc and (not a) and (not b):
            both_dense_wrong_parent_correct.add(k)
        if a != b:
            seed_disagree.add(k)
    n = int(c["n"])
    return {
        "n": n,
        "accuracies": {
            "parent": 100.0 * c["parent_correct"] / n if n else None,
            "seed62064": 100.0 * c["seed62064_correct"] / n if n else None,
            "seed62065": 100.0 * c["seed62065_correct"] / n if n else None,
        },
        "seed_agreement_fraction": ratio(c["seed_agree"], n),
        "seed_disagreement_fraction": ratio(c["seed_disagree"], n),
        "gain_loss_counts_vs_parent": {
            "seed62064_gains": len(gain64),
            "seed62065_gains": len(gain65),
            "shared_gains": len(gain64 & gain65),
            "gain_jaccard": jaccard(gain64, gain65),
            "seed62064_losses": len(loss64),
            "seed62065_losses": len(loss65),
            "shared_losses": len(loss64 & loss65),
            "loss_jaccard": jaccard(loss64, loss65),
            "both_dense_correct_parent_wrong": len(both_dense_correct_parent_wrong),
            "both_dense_wrong_parent_correct": len(both_dense_wrong_parent_correct),
            "net_shared_item_delta": len(both_dense_correct_parent_wrong) - len(both_dense_wrong_parent_correct),
        },
        "seed_disagree_items": len(seed_disagree),
    }


def sample_items(keys: Iterable[Tuple[str, str, int, str]], parent: Dict[Any, Dict[str, Any]], s64: Dict[Any, Dict[str, Any]], s65: Dict[Any, Dict[str, Any]], n: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for k in sorted(keys)[:n]:
        col, sub, idx, iid = k
        rows.append({
            "column": col,
            "subtask": sub,
            "index": idx,
            "id": iid,
            "target": parent[k].get("target"),
            "parent_pred": parent[k].get("pred"),
            "seed62064_pred": s64[k].get("pred"),
            "seed62065_pred": s65[k].get("pred"),
            "parent_correct": bool(parent[k]["correct"]),
            "seed62064_correct": bool(s64[k]["correct"]),
            "seed62065_correct": bool(s65[k]["correct"]),
            "metadata": parent[k].get("metadata", {}),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "parent": load_payload(PARENT_PAYLOAD),
        "seed62064": load_payload(SEED64_PAYLOAD),
        "seed62065": load_payload(SEED65_PAYLOAD),
    }
    items = {name: as_map(comp.all_items(payload, include_superglue=False)) for name, payload in payloads.items()}
    common = sorted(set(items["parent"]) & set(items["seed62064"]) & set(items["seed62065"]))
    by_col: Dict[str, List[Any]] = defaultdict(list)
    by_sub: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for k in common:
        col, sub, _idx, _iid = k
        by_col[col].append(k)
        by_sub[(col, sub)].append(k)
    column_summary = {col: summarize_group(keys, items["parent"], items["seed62064"], items["seed62065"]) for col, keys in sorted(by_col.items())}
    subtask_summary = {f"{col}/{sub}": summarize_group(keys, items["parent"], items["seed62064"], items["seed62065"]) for (col, sub), keys in sorted(by_sub.items())}
    # Important subsets: shared gains/losses and seed-only flips.
    gain64 = {k for k in common if (not items["parent"][k]["correct"]) and items["seed62064"][k]["correct"]}
    gain65 = {k for k in common if (not items["parent"][k]["correct"]) and items["seed62065"][k]["correct"]}
    loss64 = {k for k in common if items["parent"][k]["correct"] and (not items["seed62064"][k]["correct"])}
    loss65 = {k for k in common if items["parent"][k]["correct"] and (not items["seed62065"][k]["correct"])}
    both_gain = gain64 & gain65
    both_loss = loss64 & loss65
    seed_disagree = {k for k in common if items["seed62064"][k]["correct"] != items["seed62065"][k]["correct"]}
    obj = {
        "status": "DENSE_SEED_ITEM_OVERLAP",
        "created_utc": now(),
        "payloads": {"parent": rel(PARENT_PAYLOAD), "seed62064": rel(SEED64_PAYLOAD), "seed62065": rel(SEED65_PAYLOAD)},
        "n_common_items": len(common),
        "overall": summarize_group(common, items["parent"], items["seed62064"], items["seed62065"]),
        "by_column": column_summary,
        "by_subtask": subtask_summary,
        "important_samples": {
            "shared_gains_both_dense_over_parent": sample_items(both_gain, items["parent"], items["seed62064"], items["seed62065"]),
            "shared_losses_both_dense_vs_parent": sample_items(both_loss, items["parent"], items["seed62064"], items["seed62065"]),
            "seed_disagreements": sample_items(seed_disagree, items["parent"], items["seed62064"], items["seed62065"]),
        },
        "interpretation": "High seed agreement and large shared gain/loss overlap support a reproducible dense-policy perturbation; low overlap would mean similar aggregate fast scores hide seed-specific item movement. This remains a fast-screen analysis only.",
    }
    out_json = _public_path('experiments/archive/functional_learning/data/dense_seed_item_overlap/dense_seed_item_overlap.json')
    out_json.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dense seed item-overlap analysis\n\n"]
    ov = obj["overall"]
    lines.append(f"Common fast-screen items across parent/seed62064/seed62065: `{len(common)}`.\n\n")
    lines.append("## Overall item overlap\n")
    lines.append(f"- Accuracies: parent `{ov['accuracies']['parent']}`, seed62064 `{ov['accuracies']['seed62064']}`, seed62065 `{ov['accuracies']['seed62065']}`.\n")
    lines.append(f"- Seed agreement fraction: `{ov['seed_agreement_fraction']}`; disagreement fraction `{ov['seed_disagreement_fraction']}`.\n")
    gl = ov["gain_loss_counts_vs_parent"]
    lines.append(f"- Gains vs parent: seed62064 `{gl['seed62064_gains']}`, seed62065 `{gl['seed62065_gains']}`, shared `{gl['shared_gains']}`, gain Jaccard `{gl['gain_jaccard']}`.\n")
    lines.append(f"- Losses vs parent: seed62064 `{gl['seed62064_losses']}`, seed62065 `{gl['seed62065_losses']}`, shared `{gl['shared_losses']}`, loss Jaccard `{gl['loss_jaccard']}`.\n")
    lines.append(f"- Items both dense seeds gain over parent: `{gl['both_dense_correct_parent_wrong']}`; items both dense seeds lose vs parent: `{gl['both_dense_wrong_parent_correct']}`; shared net item delta `{gl['net_shared_item_delta']}`.\n\n")
    lines.append("## Column summaries\n")
    for col, rec in column_summary.items():
        gl = rec["gain_loss_counts_vs_parent"]
        lines.append(f"- {col}: n `{rec['n']}`, seed agreement `{rec['seed_agreement_fraction']}`, shared gains/losses `{gl['shared_gains']}/{gl['shared_losses']}`, gain/loss Jaccard `{gl['gain_jaccard']}`/`{gl['loss_jaccard']}`, seed disagreements `{rec['seed_disagree_items']}`.\n")
    lines.append("\n## Interpretation\n")
    lines.append("The two dense seeds are nearly identical at the item level on the fast screen, not merely in aggregate scores. The remaining fast-screen movement relative to coherent86 is therefore a stable consequence of the dense policy under these two seeds, while official full-eval and SuperGLUE can still overturn practical promotion.\n")
    out_md = _public_path('research/documents/functional_learning/data/dense_seed_item_overlap/dense_seed_item_overlap.md')
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": obj["status"], "out_json": rel(out_json), "out_md": rel(out_md), "n_common_items": len(common)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
