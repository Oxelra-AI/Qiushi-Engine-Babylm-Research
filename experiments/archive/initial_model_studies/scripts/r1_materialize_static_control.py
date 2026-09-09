#!/usr/bin/env python3
"""research R1 matched corpora with an indirect-but-static control.

Control correction: direct object-naming control is confounded by surface form,
local information, and prediction difficulty. This materializer instead creates:

  ordered_dynamic: indirect location references whose resolution can depend on
                   previous state updates.
  static_indirect_control: the same indirect-reference style, similar query and
                   answer statistics, but every queried object is moved exactly
                   once from its initial unique location; source references are
                   never changed before use. Thus the answer can be recovered by
                   initial-world lookup + the local indirect sentence, not by
                   composing cross-step state updates.

Both streams mix the same official examples and use disjoint train/heldout vocab
consistent with the research evaluator.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, random, sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAINER = ROOT / "training/scripts/babylm_masked_train_leadershape.py"
GEN = ROOT / "scripts/r1_generator_v3.py"
spec = importlib.util.spec_from_file_location("trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = trainer; spec.loader.exec_module(trainer)
spec2 = importlib.util.spec_from_file_location("r1gen", GEN)
r1gen = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = r1gen; spec2.loader.exec_module(r1gen)

TRAIN_LOCS = r1gen.LOCATIONS[:7]
TRAIN_ITEMS = r1gen.ITEMS[:14]
HELDOUT_LOCS = r1gen.LOCATIONS[7:]
HELDOUT_ITEMS = r1gen.ITEMS[14:]

def wc(text: str) -> int:
    return len(text.split())

def set_train_vocab():
    r1gen.LOCATIONS = list(TRAIN_LOCS)
    r1gen.ITEMS = list(TRAIN_ITEMS)

def unique_initial_assignment(rng: random.Random, items: list[str], locs: list[str]) -> dict[str, str]:
    chosen = rng.sample(locs, len(items)) if len(locs) >= len(items) else [rng.choice(locs) for _ in items]
    return {obj: loc for obj, loc in zip(items, chosen)}

def build_static_indirect_scenario(rng: random.Random, n_locs: int, n_items: int, n_ops: int):
    """Indirect sentences but no dynamic dependency.

    Each source location is an initial unique location and is referenced before any
    operation can alter that source. Each queried/moved object is touched once.
    Distractor moves occur after the query-relevant move but affect other objects.
    """
    locs = rng.sample(r1gen.LOCATIONS, min(n_locs, len(r1gen.LOCATIONS)))
    items = rng.sample(r1gen.ITEMS, min(n_items, len(r1gen.ITEMS), len(locs)))
    init = unique_initial_assignment(rng, items, locs)
    state = dict(init)
    ops = []
    order = list(items); rng.shuffle(order)
    touched = set()
    for obj in order[:min(n_ops, len(order))]:
        src = init[obj]  # deliberately initial source, never previously changed
        dst_candidates = [l for l in locs if l != src]
        dst = rng.choice(dst_candidates)
        text = r1gen.make_indirect_text(rng, src, dst)
        ops.append(r1gen.Operation(kind="indirect_static", obj=obj, src=src, dst=dst, text=text, ref_by_loc=True))
        state[obj] = dst
        touched.add(obj)
    # Query one touched object. Add 2 distractor direct operations afterward to
    # decorrelate most-recent-location while keeping answer unchanged.
    query_obj = rng.choice(list(touched)) if touched else rng.choice(items)
    answer = state[query_obj]
    other = [x for x in items if x != query_obj]
    for _ in range(min(2, len(other))):
        d = rng.choice(other)
        d_src = state[d]
        cands = [l for l in locs if l != d_src and l != answer]
        if not cands:
            cands = [l for l in locs if l != d_src]
        if not cands:
            continue
        d_dst = rng.choice(cands)
        ops.append(r1gen.Operation(kind="direct_distractor", obj=d, src=d_src, dst=d_dst, text=r1gen.make_direct_text(rng, d, d_src, d_dst), ref_by_loc=False))
        state[d] = d_dst
    passage = r1gen.build_passage(init, ops, query_obj, state)
    return {
        "text": passage,
        "words": wc(passage),
        "source": "r1_static_indirect_control",
        "query_obj": query_obj,
        "answer": state[query_obj],
        "n_indirect": sum(1 for op in ops if op.ref_by_loc),
        "n_dynamic_dependency": 0,
    }

def build_generated_unique(target_words: int, seed: int):
    set_train_vocab()
    rng = random.Random(seed)
    ordered = []; static = []
    ow = sw = 0; sid = 0
    dyn_indirect = static_indirect = 0
    while ow < target_words or sw < target_words:
        n_locs = rng.randint(5, 7); n_items = rng.randint(4, 5); n_ops = rng.randint(5, 8)
        sc = r1gen.generate_scenario(rng, n_locs=n_locs, n_items=n_items, n_ops=n_ops, indirect_fraction=0.70)
        if ow < target_words:
            ordered.append({
                "text": sc.passage, "words": wc(sc.passage), "source": "r1_ordered_dynamic_indirect",
                "example_id": sid, "query_obj": sc.query_obj, "answer": sc.answer,
                "n_indirect": sc.n_indirect,
                "n_dynamic_dependency": 1,
            })
            ow += ordered[-1]["words"]
            dyn_indirect += int(sc.n_indirect > 0)
        if sw < target_words:
            st = build_static_indirect_scenario(rng, n_locs=n_locs, n_items=n_items, n_ops=n_ops)
            st["example_id"] = sid
            static.append(st); sw += st["words"]; static_indirect += int(st["n_indirect"] > 0)
        sid += 1
    return ordered, static, {
        "seed": seed, "target_generated_words_per_arm": target_words,
        "ordered_unique_words": ow, "static_unique_words": sw,
        "ordered_examples": len(ordered), "static_examples": len(static),
        "ordered_indirect_example_fraction": dyn_indirect / max(1, len(ordered)),
        "static_indirect_example_fraction": static_indirect / max(1, len(static)),
        "train_locations": TRAIN_LOCS, "train_items": TRAIN_ITEMS,
        "heldout_locations": HELDOUT_LOCS, "heldout_items": HELDOUT_ITEMS,
    }

def build_official_unique(outdir: pathlib.Path, target_words: int, words_per_example: int):
    raw_dir, manifest_files = trainer.download_dataset(
        argparse.Namespace(dataset_id="BabyLM-community/BabyLM-2026-Strict-Small", dataset_revision="c92ab16b4f08858304b0815706065b3354d8fc0a"),
        outdir / "official_raw",
    )
    files = [raw_dir / n for n in trainer.TRAIN_FILES]
    examples=[]; used=0
    for i, ex in enumerate(trainer.iter_examples(files, target_words, words_per_example)):
        examples.append({"text": ex.text, "words": ex.words, "source": f"official::{ex.source}", "example_id": i})
        used += ex.words
    if used != target_words:
        raise RuntimeError(f"official words {used} != {target_words}")
    return examples, {"official_unique_words": used, "official_examples": len(examples), "official_manifest": manifest_files}

def exposure_stream(unique_examples: list[dict], exposure_words: int, seed: int, label: str):
    stream=[]; total=0; epoch=0; meta=[]
    while total < exposure_words:
        epoch += 1
        exs = list(unique_examples)
        random.Random(seed + 1000003 * epoch).shuffle(exs)
        before=total; added=0
        for ex in exs:
            if total >= exposure_words: break
            rem = exposure_words - total
            if ex["words"] <= rem:
                row = dict(ex)
            else:
                row = dict(ex); row["text"] = " ".join(ex["text"].split()[:rem]); row["words"] = rem; row["trimmed_final_row"] = True
            row["exposure_epoch"] = epoch; row["exposure_index"] = len(stream)
            stream.append(row); total += row["words"]; added += row["words"]
        meta.append({"epoch": epoch, "before": before, "after": total, "words_added": added})
    return stream, {"label": label, "exposure_words": total, "epochs_materialized": epoch, "epoch_meta": meta}

def write_jsonl(path: pathlib.Path, rows: list[dict]):
    total=0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            actual = wc(r["text"])
            if actual != r["words"]: raise RuntimeError(f"word mismatch {actual}!={r['words']}")
            total += actual; f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return total

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--generated_unique_words", type=int, default=100_000)
    p.add_argument("--official_unique_words", type=int, default=400_000)
    p.add_argument("--exposure_words", type=int, default=5_000_000)
    p.add_argument("--words_per_example", type=int, default=80)
    p.add_argument("--seed", type=int, default=210)
    args=p.parse_args(); out=pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ordered, static, gen_meta = build_generated_unique(args.generated_unique_words, args.seed)
    official, off_meta = build_official_unique(out, args.official_unique_words, args.words_per_example)
    ordered_stream, ometa = exposure_stream(list(official)+ordered, args.exposure_words, args.seed+11, "ordered_dynamic")
    static_stream, smeta = exposure_stream(list(official)+static, args.exposure_words, args.seed+11, "static_indirect_control")
    op = out / "r1_ordered_dynamic_5M.jsonl"; sp = out / "r1_static_indirect_control_5M.jsonl"
    ow = write_jsonl(op, ordered_stream); sw = write_jsonl(sp, static_stream)
    meta={"status":"R1_STATIC_CONTROL_CORPORA", "ordered_jsonl":str(op), "static_control_jsonl":str(sp), "ordered_exposure_words":ow, "static_control_exposure_words":sw, "generated_meta":gen_meta, "official_meta":off_meta, "ordered_stream_meta":ometa, "static_stream_meta":smeta, "interpretation":"Both generated arms use indirect-reference style; static control removes dynamic cross-step dependency by ensuring indirect source references resolve from the initial unique world and are not changed before use."}
    mp=out/"r1_static_control_meta.json"; mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"ordered":ow,"static":sw,"meta":str(mp)}, indent=2))
if __name__=="__main__": main()
