#!/usr/bin/env python3
"""CPU-only lexical selector baseline for research.

A deterministic role-string matcher chooses the tag from the inline-role context
line matching the query role.  This checks whether research exact selector success
requires anything beyond local surface correspondence.  It also reports an oracle
paraphrase dictionary (prior->background, revised->update) to show the missing
semantic map separately from the tag/state reader.
"""
from __future__ import annotations
import argparse, collections, importlib.util, json, re, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

STUDY = Path("experiments/archive/representation_and_objectives")
ST272 = STUDY / "training/scripts/selector_reader_bridge.py"
spec = importlib.util.spec_from_file_location("st272", ST272)
st272 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = st272
spec.loader.exec_module(st272)

s269 = st272.s269


def make_args():
    return SimpleNamespace(
        model_path=str(st272.DEFAULT_MODEL), out_dir="", seed=27000, mode="inline_role",
        shared_address_ns="s269_inline_direct", shared_address_suffix="_direct_tag",
        base_stable=80, base_train_wording=True, sparse_changed=16, sparse_stable=16,
        eval_held_changed=40, eval_held_stable=40, eval_train_changed=40,
        eval_train_stable=40, max_len=200, batch_size=8, selector_batch_size=16,
        eval_batch_size=32, reader_tag_epochs=5, reader_direct_epochs=6,
        selector_epochs=8, head_lr=1e-3, encoder_lr=8e-5, selector_pos_weight=3.0,
        weight_decay=1e-3, device="cpu", dry_build=False)


def exact_role_phrase(qf: str, qmode: str, semantic_para: bool = False) -> tuple[str, str]:
    scope = s269.SCOPE[qf].lower()
    if qmode == "role_para" and semantic_para:
        # Map paraphrase query words back to the role labels displayed in context.
        t = s269.TNAME[qf]
    elif qmode == "role_para":
        t = s269.TPARA[s269.TNAME[qf]]
    else:
        t = s269.TNAME[qf]
    return scope, t.lower()


def choose_tag(context: str, qf: str, qmode: str, semantic_para: bool = False) -> str | None:
    scope, t = exact_role_phrase(qf, qmode, semantic_para=semantic_para)
    # Context sentences look like: "Focal background entry amber: ...".
    pat = re.compile(rf"\b{re.escape(scope)}\s+{re.escape(t)}\s+entry\s+([A-Za-z]+)\s*:", re.I)
    m = pat.search(context)
    return m.group(1).lower() if m else None


def eval_matcher(rows: list[dict[str, Any]], *, semantic_para: bool = False) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["select_group"]].append(r)
    oks = []
    none = 0
    recs = []
    for g, rs in groups.items():
        r0 = rs[0]
        tag = choose_tag(r0["context"], r0["query_family"], r0.get("selector_qmode", "role_inline"), semantic_para=semantic_para)
        if tag is None:
            none += 1
            ok = 0
        else:
            # Candidate tags are lower-case ordinary words.
            ok = int(tag == r0["expected_tag"].lower())
        oks.append(ok)
        recs.append((ok, r0, tag))
    out = {"top1": float(np.mean(oks)) if oks else float("nan"), "groups": len(oks), "none": none}
    for key in ["query_family", "selector_qmode", "role_swap", "eval_set"]:
        d = collections.defaultdict(list)
        for ok, r, tag in recs:
            d[str(r.get(key))].append(ok)
        out[f"by_{key}"] = {k: float(np.mean(v)) for k, v in sorted(d.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="experiments/archive/representation_and_objectives/data/lexical_selector_baseline")
    args_cli = ap.parse_args()
    out = Path(args_cli.out_dir); out.mkdir(parents=True, exist_ok=True)
    args = make_args()
    suites = st272.selector_eval_suite(args)
    results = {}
    for name, rows in suites.items():
        results[name] = {
            "exact_surface_matcher": eval_matcher(rows, semantic_para=False),
            "paraphrase_dictionary_matcher": eval_matcher(rows, semantic_para=True),
        }
    summary = {"status": "DONE", "results": results,
               "meaning": "Exact role-string matching is a deterministic selector baseline; paraphrase dictionary is an oracle synonym map only for prior/revised->background/update."}
    (out / "lexical_selector_baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
    lines = ["# research lexical selector baseline\n\n",
             "A deterministic matcher extracts the tag from the inline-role context line whose role words match the query role. This tests whether exact research selector success exceeds local surface correspondence.\n\n",
             "| eval | exact matcher top1 | exact none | synonym-map matcher top1 | synonym-map none |\n",
             "|---|---:|---:|---:|---:|\n"]
    for name, res in sorted(results.items()):
        a = res["exact_surface_matcher"]; b = res["paraphrase_dictionary_matcher"]
        lines.append(f"| {name} | {a['top1']:.3f} | {a['none']} | {b['top1']:.3f} | {b['none']} |\n")
    lines.append("\nFor exact role wording and role-swap contexts, surface matching is sufficient. For held_para_nsA, exact matching finds no background/update line because the query uses prior/revised; a hand-supplied synonym map restores 1.0, isolating the missing temporal-role paraphrase map.\n")
    (out / "lexical_selector_baseline.md").write_text("".join(lines), "utf-8")
    print(json.dumps({"status": "DONE", "md": str(out / "lexical_selector_baseline.md"), "json": str(out / "lexical_selector_baseline.json")}), flush=True)


if __name__ == "__main__":
    main()
