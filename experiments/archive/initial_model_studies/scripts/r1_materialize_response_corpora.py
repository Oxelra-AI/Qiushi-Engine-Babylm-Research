#!/usr/bin/env python3
"""research R1 matched learning-response corpora.

Produces two JSONL exposure streams for the pre-10M learning-response test:
  - ordered: generated passages with indirect location references; resolving the
    final answer requires state tracking.
  - direct_control: matched scenarios whose operations explicitly name the moved
    object, removing the indirect state-resolution dependency while preserving
    style, vocabulary, and many surface statistics.

Each stream mixes the same official BabyLM examples with generated examples and
then repeats the unique pool for multiple shuffled passes, so a short unique pool
can yield thousands of optimizer updates without pretending to be a full 10M run.
This is a learning-response experiment, not a SOTA candidate.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, random, sys
from dataclasses import asdict

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


def words(text: str) -> int:
    return len(text.split())


def direct_control_text(sc, rng: random.Random) -> str:
    """Convert all operations in a scenario to direct object-naming operations."""
    lines = []
    for obj, loc in sc.init_loc.items():
        lines.append(f"Initially, {obj} is in {loc}.")
    for op in sc.operations:
        # Use the actual resolved object/src/dst, but make it locally visible.
        lines.append(r1gen.make_direct_text(rng, op.obj, op.src, op.dst))
    lines.append(f"After all these changes, {sc.query_obj} is in {sc.answer}.")
    return " ".join(lines)


def set_train_vocab():
    r1gen.LOCATIONS = list(TRAIN_LOCS)
    r1gen.ITEMS = list(TRAIN_ITEMS)


def build_generated_unique(target_words: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    set_train_vocab()
    rng = random.Random(seed)
    ordered = []
    control = []
    ow = cw = 0
    scenario_id = 0
    n_indirect = 0
    while ow < target_words or cw < target_words:
        sc = r1gen.generate_scenario(
            rng, n_locs=rng.randint(5, 7), n_items=rng.randint(4, 5),
            n_ops=rng.randint(5, 8), indirect_fraction=0.70,
        )
        ctrl = direct_control_text(sc, rng)
        if ow < target_words:
            ordered.append({
                "text": sc.passage,
                "words": words(sc.passage),
                "source": "r1_ordered_indirect",
                "example_id": scenario_id,
                "query_obj": sc.query_obj,
                "answer": sc.answer,
                "n_indirect": sc.n_indirect,
            })
            ow += ordered[-1]["words"]
        if cw < target_words:
            control.append({
                "text": ctrl,
                "words": words(ctrl),
                "source": "r1_direct_control",
                "example_id": scenario_id,
                "query_obj": sc.query_obj,
                "answer": sc.answer,
                "n_indirect_original": sc.n_indirect,
            })
            cw += control[-1]["words"]
        n_indirect += int(sc.n_indirect > 0)
        scenario_id += 1
    meta = {
        "generated_seed": seed,
        "target_generated_words_per_arm": target_words,
        "ordered_unique_words": ow,
        "control_unique_words": cw,
        "ordered_examples": len(ordered),
        "control_examples": len(control),
        "train_locations": TRAIN_LOCS,
        "train_items": TRAIN_ITEMS,
        "heldout_locations": HELDOUT_LOCS,
        "heldout_items": HELDOUT_ITEMS,
        "scenario_count": scenario_id,
    }
    return ordered, control, meta


def build_official_unique(outdir: pathlib.Path, target_words: int, words_per_example: int, seed: int) -> tuple[list[dict], dict]:
    raw_dir, manifest_files = trainer.download_dataset(
        argparse.Namespace(
            dataset_id="BabyLM-community/BabyLM-2026-Strict-Small",
            dataset_revision="c92ab16b4f08858304b0815706065b3354d8fc0a",
        ),
        outdir / "official_raw",
    )
    files = [raw_dir / n for n in trainer.TRAIN_FILES]
    examples = []
    used = 0
    for i, ex in enumerate(trainer.iter_examples(files, target_words, words_per_example)):
        examples.append({"text": ex.text, "words": ex.words, "source": f"official::{ex.source}", "example_id": i})
        used += ex.words
    if used != target_words:
        raise RuntimeError(f"official unique words mismatch {used} != {target_words}")
    return examples, {"official_unique_words": used, "official_examples": len(examples), "official_manifest": manifest_files}


def exposure_stream(unique_examples: list[dict], exposure_words: int, seed: int, label: str) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    stream = []
    total = 0
    epoch = 0
    epoch_meta = []
    while total < exposure_words:
        epoch += 1
        exs = list(unique_examples)
        rng_epoch = random.Random(seed + 1000003 * epoch)
        rng_epoch.shuffle(exs)
        before = total
        added = 0
        for ex in exs:
            if total >= exposure_words:
                break
            remaining = exposure_words - total
            if ex["words"] <= remaining:
                row = dict(ex)
            else:
                # Trim final row only; this preserves exact exposure accounting.
                toks = ex["text"].split()[:remaining]
                row = dict(ex); row["text"] = " ".join(toks); row["words"] = remaining; row["trimmed_final_row"] = True
            row["exposure_epoch"] = epoch
            row["exposure_index"] = len(stream)
            stream.append(row)
            total += row["words"]
            added += row["words"]
        epoch_meta.append({"epoch": epoch, "words_added": added, "before": before, "after": total})
    return stream, {"label": label, "exposure_words": total, "epochs_materialized": epoch, "epoch_meta": epoch_meta}


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> int:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            actual = words(row["text"])
            if actual != row["words"]:
                raise RuntimeError(f"word mismatch row {row.get('exposure_index')} {actual} != {row['words']}")
            total += actual
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--generated_unique_words", type=int, default=100_000)
    p.add_argument("--official_unique_words", type=int, default=400_000)
    p.add_argument("--exposure_words", type=int, default=5_000_000)
    p.add_argument("--words_per_example", type=int, default=80)
    p.add_argument("--seed", type=int, default=209)
    args = p.parse_args()
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ordered_gen, control_gen, gen_meta = build_generated_unique(args.generated_unique_words, args.seed)
    official, off_meta = build_official_unique(out, args.official_unique_words, args.words_per_example, args.seed)
    rng = random.Random(args.seed)
    ordered_unique = list(official) + list(ordered_gen)
    control_unique = list(official) + list(control_gen)
    # Match source proportions while letting order differ by condition-specific shuffles.
    ordered_stream, ordered_meta = exposure_stream(ordered_unique, args.exposure_words, args.seed + 11, "ordered_indirect")
    control_stream, control_meta = exposure_stream(control_unique, args.exposure_words, args.seed + 11, "direct_control")

    op = out / "r1_ordered_indirect_5M.jsonl"
    cp = out / "r1_direct_control_5M.jsonl"
    ow = write_jsonl(op, ordered_stream)
    cw = write_jsonl(cp, control_stream)
    meta = {
        "status": "R1_RESPONSE_CORPORA",
        "seed": args.seed,
        "ordered_jsonl": str(op),
        "direct_control_jsonl": str(cp),
        "ordered_exposure_words": ow,
        "direct_control_exposure_words": cw,
        "generated_meta": gen_meta,
        "official_meta": off_meta,
        "ordered_stream_meta": ordered_meta,
        "control_stream_meta": control_meta,
        "interpretation": "Matched learning-response streams: same official examples, matched generated scenarios, ordered arm uses indirect state-resolving operations; control arm explicitly names objects, removing the state-resolution dependency.",
    }
    (out / "r1_response_corpora_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ordered": ow, "control": cw, "meta": str(out / "r1_response_corpora_meta.json")}, indent=2))

if __name__ == "__main__":
    main()
