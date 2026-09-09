#!/usr/bin/env python3
"""research: Comparison-channel causal intervention on learned-equality raw-text gauge.

Scientific purpose
------------------
Test whether comparison graph edges are causally necessary for gauge transport
in the dual-position learned-equality setting (research).

Three interventions:
  full_graph    : all comparison edges retained (research reproduction)
  no_cmp        : training comparisons removed entirely; only state loss
  shuffled_cmp  : training comparison labels randomly permuted

Key prediction:
  - full_graph: graph_same=1.0 for +1, 0.0 for -1 (reproduces research)
  - no_cmp: graph_same ≈ constant for both signs (h1/h3 never supervised in training)
  - shuffled_cmp: graph_same ≈ constant (noisy comparisons don't orient)

If no_cmp/shuffled remove sign reversal while full_graph preserves it,
comparison edges are causally necessary for gauge transport.

Data structure justification:
  Training state queries contain only s_give/s_receive (common) and h0_dax/h2_norp
  (direct anchors). h1_mep and h3_ziv state labels exist ONLY in evaluation.
  The comparison graph connects h0↔h1, h0↔h2, h1↔h2, h2↔h3, providing the only
  training-time pathway from h0/h2 anchors to h1/h3 events.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
import dualpos_learned_equality_gauge_probe as s299
import posalign_gauge_probe as p298
import raw_span_discovery_probe as base

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/comparison_channel"


def shuffle_comparison_labels(tc, seed):
    """Return comparison copies with labels randomly permuted."""
    rng = random.Random(seed + 777)
    labels = [c.label for c in tc]
    rng.shuffle(labels)
    shuffled = []
    for c, new_label in zip(tc, labels):
        c_new = copy.copy(c)
        c_new.label = new_label
        shuffled.append(c_new)
    return shuffled


def graph_manifest(ts, tc, es, ec):
    """Document graph structure for provenance."""
    train_rels = Counter(q.relation for q in ts)
    train_changed = sum(1 for q in ts if q.is_changed)
    train_da = sum(1 for q in ts if q.is_direct_anchor)
    comp_edges = Counter()
    comp_labels = Counter()
    for c in tc:
        comp_edges[(c.relation1, c.relation2)] += 1
        comp_labels[str(c.label)] += 1

    eval_rels = Counter(q.relation for q in es)
    eval_comp_edges = Counter()
    for c in ec:
        eval_comp_edges[(c.relation1, c.relation2)] += 1

    return {
        "train_state": {
            "total": len(ts), "by_relation": dict(train_rels),
            "n_changed": train_changed, "n_direct_anchor": train_da,
            "h1h3_count": sum(v for k, v in train_rels.items() if k in ("h1_mep", "h3_ziv")),
        },
        "train_comparison": {
            "total": len(tc),
            "edge_types": {f"{k[0]}->{k[1]}": v for k, v in sorted(comp_edges.items())},
            "label_balance": dict(comp_labels),
        },
        "eval_state": {"total": len(es), "by_relation": dict(eval_rels)},
        "eval_comparison": {
            "total": len(ec),
            "edge_types": {f"{k[0]}->{k[1]}": v for k, v in sorted(eval_comp_edges.items())},
        },
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def run_cell(args, cond, bs, paired, vocab, ts, tc_train, es, ec, char_ids):
    """Run one cell using s299.run_one with modified training comparisons.
    
    tc_train may be original, empty (no_cmp), or shuffled (shuffled_cmp).
    Evaluation always uses the original ec.
    """
    # Create a namespace for s299.run_one
    s299_args = argparse.Namespace(
        seed=args.seed, gpu=args.gpu,
        char_pair_mode="full_alphabet",
        char_pair_epochs=args.char_pair_epochs,
        char_pair_lr=args.char_pair_lr,
        epochs=args.epochs, lr=args.lr, wd=args.wd,
        cmp_weight=args.cmp_weight,
        state_weight=args.state_weight,
        static_weight=args.static_weight,
        print_every=args.print_every,
        arm=args.arm,
        out=args.out / args.intervention,
        no_name_vocab=True,
    )
    rec = s299.run_one(s299_args, cond, bs, paired, vocab, ts, tc_train, es, ec, char_ids)
    rec["intervention"] = args.intervention
    rec["tc_train_size"] = len(tc_train)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervention", choices=["full_graph", "no_cmp", "shuffled_cmp"],
                    required=True)
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=base.DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk"])
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1, -1],
                    default=[1])
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--char-dim", type=int, default=16)
    ap.add_argument("--char-pair-epochs", type=int, default=60)
    ap.add_argument("--char-pair-lr", type=float, default=5e-2)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=45)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # Patch dual-position matcher
    s299.patch_dual_matcher()

    # Load all data (original)
    ts, tc, es, ec, _, counts = base.load_dataset(args.data_root, args.arm)

    # Apply intervention to training comparisons
    if args.intervention == "no_cmp":
        tc_train = []
        print(f"[research] no_cmp: removed all {len(tc)} training comparisons", flush=True)
    elif args.intervention == "shuffled_cmp":
        tc_train = shuffle_comparison_labels(tc, args.seed)
        n_flipped = sum(1 for a, b in zip(tc, tc_train) if a.label != b.label)
        print(f"[research] shuffled_cmp: permuted labels of {len(tc)} comparisons "
              f"({n_flipped} labels changed)", flush=True)
    else:
        tc_train = tc
        print(f"[research] full_graph: retaining all {len(tc)} comparisons", flush=True)

    # Build vocab (same as research; use original tc for vocab coverage)
    vocab = base.RawVocab()
    train_name_set = set()
    for q in ts:
        for n in q.names: train_name_set.add(n)
    for c in tc:
        for n in c.names: train_name_set.add(n)
    base.collect_raw_vocab(vocab, ts, tc, exclude_names=train_name_set)

    char_ids = list(range(1, 27))  # full alphabet

    # Save graph manifest once
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = graph_manifest(ts, tc, es, ec)
    manifest["intervention"] = args.intervention
    manifest["tc_train_size"] = len(tc_train)
    write_json(args.out / "graph_manifest.json", manifest)

    print(f"research comparison-channel probe: intervention={args.intervention} "
          f"conditions={args.conditions} bridge_signs={args.bridge_signs} "
          f"seed={args.seed} tc_train={len(tc_train)} vocab={len(vocab.itos)}",
          flush=True)

    results = []
    for bs in args.bridge_signs:
        # Fresh untouched initialization per sign (same seed → same init for sign pair)
        paired = p298.create_paired_posalign(
            len(vocab.itos), args.conditions, args.seed,
            args.emb_dim, args.hidden, args.char_dim)
        for cond in args.conditions:
            rec = run_cell(args, cond, bs, paired, vocab, ts, tc_train, es, ec, char_ids)
            results.append(rec)
            ce = rec.get("central_eval", {})
            print(json.dumps({
                "done": f"step300_{args.intervention}_{cond}_bs{'+' if bs == 1 else '-'}1",
                "graph_same": ce.get("graph_same"),
                "unchanged": ce.get("unchanged"),
                "hh_closure": ce.get("hh_closure"),
                "mixed_acc": ce.get("mixed_acc"),
                "direct_same": ce.get("direct_same"),
                "pair_both_graph_same": ce.get("pair_both_graph_same"),
                "matching_final": rec.get("matching_final"),
            }, indent=2), flush=True)

    # Summary table
    lines = [f"# research comparison-channel: {args.intervention}", "",
             f"Training comparisons: {len(tc_train)} (original: {len(tc)})", "",
             "| cond | bs | graph_same | unchanged | hh_closure | mixed | direct_same | pair_both |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ce = r.get("central_eval", {})
        lines.append(
            f"| {r['condition']} | {r['bridge_sign']} | "
            f"{ce.get('graph_same', 0):.3f} | {ce.get('unchanged', 0):.3f} | "
            f"{ce.get('hh_closure', 0):.3f} | {ce.get('mixed_acc', 0):.3f} | "
            f"{ce.get('direct_same', 0):.3f} | {ce.get('pair_both_graph_same', 0):.3f} |")

    tag = f"step300_{args.intervention}"
    write_json(args.out / f"{tag}_results.json", {
        "intervention": args.intervention, "seed": args.seed,
        "results": results, "manifest": manifest, "counts": counts})
    (args.out / f"{tag}_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": f"STEP300_{args.intervention.upper()}_COMPLETE",
        "summary": str(args.out / f"{tag}_summary.md"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
