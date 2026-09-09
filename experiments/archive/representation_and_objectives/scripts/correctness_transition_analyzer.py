#!/usr/bin/env python3
"""research: Item-level correctness transition analysis across the compact-view triangle.

For each task with item-level binary correctness, computes a 2×2 transition matrix
between pairs of arms: retained-correct (CC), newly-correct (WC), lost-correct (CW),
retained-wrong (WW). This distinguishes stable competence gains from answer churn.

Uses official BabyLM scoring logic exactly as implemented in calculate_results_from_pred.py.
"""
import json, pathlib, sys, collections

# ---- arm prediction roots ----
VIEW_ROOT = pathlib.Path("experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/eval_outputs/density_reinvest_noaoa/compact_view_reinvest")
REPEAT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/eval_outputs/compact_triangle_noaoa_guarded/compact_repeat_reinvest")
ADJBREAK_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/eval_outputs/compact_triangle_noaoa_guarded/adjbreak_reinvest")

# ---- gold data root ----
EVAL_DATA = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data")
FAST_EVAL = EVAL_DATA / "fast_eval"
FULL_EVAL = EVAL_DATA / "full_eval"

ARMS = {"compact_view": VIEW_ROOT, "compact_repeat": REPEAT_ROOT, "adjbreak": ADJBREAK_ROOT}

# ---- Prediction file locators ----
def pred_path(arm_root, task):
    """Return predictions.json path for a given arm and task."""
    mapping = {
        "BLiMP": "BLiMP/chck_100M/*/zero_shot/mlm/blimp/blimp_fast/predictions.json",
        "Supplement": "Supplement/chck_100M/*/zero_shot/mlm/blimp/supplement_fast/predictions.json",
        "EWoK": "EWoK/chck_100M/*/zero_shot/mlm/ewok/ewok_fast/predictions.json",
        "Entity": "Entity/chck_100M/*/zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
        "COMPS": "COMPS/chck_100M/*/zero_shot/mlm/comps/comps/predictions.json",
        "GlobalPIQA_parallel": "GlobalPIQA_parallel/chck_100M/*/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "GlobalPIQA_nonparallel": "GlobalPIQA_nonparallel/chck_100M/*/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    }
    candidates = list(arm_root.glob(mapping[task]))
    assert len(candidates) == 1, f"Expected 1 predictions file for {task} in {arm_root}, found {len(candidates)}: {candidates}"
    return candidates[0]


def load_predictions(fpath):
    """Load predictions.json, returning dict[subtask -> list[dict(id, pred)]]."""
    with open(fpath) as f:
        return json.load(f)


# ---- Correctness extractors for each task ----

def score_blimp_supplement(pred_dict, gold_dir):
    """BLiMP or Supplement: correct if pred == sentence_good."""
    items = []
    for subtask, data in pred_dict.items():
        gold_file = gold_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            print(f"  WARNING: gold file missing for {subtask}, skipping", file=sys.stderr)
            continue
        with open(gold_file) as gf:
            gold_rows = [json.loads(line) for line in gf]
        preds = data["predictions"]
        for pred_item, gold_row in zip(preds, gold_rows):
            correct = pred_item["pred"].strip() == gold_row["sentence_good"].strip()
            items.append({"id": pred_item["id"], "subtask": subtask, "correct": correct})
    return items


def score_ewok(pred_dict, gold_dir):
    """EWoK: correct if pred == Context1 + ' ' + Target1."""
    items = []
    for subtask, data in pred_dict.items():
        gold_file = gold_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            print(f"  WARNING: gold file missing for ewok/{subtask}, skipping", file=sys.stderr)
            continue
        with open(gold_file) as gf:
            gold_rows = [json.loads(line) for line in gf]
        preds = data["predictions"]
        for pred_item, gold_row in zip(preds, gold_rows):
            target = " ".join([gold_row["Context1"], gold_row["Target1"]]).strip()
            correct = pred_item["pred"].strip() == target
            items.append({"id": pred_item["id"], "subtask": subtask, "correct": correct,
                          "domain": gold_row.get("Domain", subtask)})
    return items


def score_entity(pred_dict, gold_dir):
    """Entity Tracking: correct if pred == options[0], positionally matched by numops groups."""
    items = []
    # Gold data is in regular.jsonl (fast eval has just one file)
    gold_file = gold_dir / "regular.jsonl"
    if not gold_file.exists():
        print(f"  WARNING: entity gold file missing", file=sys.stderr)
        return items
    with open(gold_file) as gf:
        gold_rows = [json.loads(line) for line in gf]

    # Group gold rows by numops to match prediction subtask keys
    # Predictions are keyed as "regular_0_ops", "regular_1_ops", etc.
    numops_groups = collections.defaultdict(list)
    for row in gold_rows:
        numops_groups[row["numops"]].append(row)

    for numops in sorted(numops_groups.keys()):
        subtask_key = f"regular_{numops}_ops"
        if subtask_key not in pred_dict:
            continue
        preds = pred_dict[subtask_key]["predictions"]
        golds = numops_groups[numops]
        for pred_item, gold_row in zip(preds, golds):
            correct = pred_item["pred"].strip() == gold_row["options"][0].strip()
            items.append({"id": pred_item["id"], "subtask": subtask_key, "correct": correct,
                          "numops": numops})
    return items


def score_comps(pred_dict, gold_dir):
    """COMPS: correct if pred == prefix_acceptable + ' ' + property_phrase."""
    subtask_to_file = {
        "base": "comps_base",
        "wugs_dist_before": "comps_wugs_dist-before",
        "wugs_dist_in_between": "comps_wugs_dist-in-between",
        "wugs": "comps_wugs",
    }
    items = []
    for subtask, data in pred_dict.items():
        file_stem = subtask_to_file.get(subtask)
        if file_stem is None:
            continue
        gold_file = gold_dir / f"{file_stem}.jsonl"
        if not gold_file.exists():
            print(f"  WARNING: comps gold file missing for {subtask}", file=sys.stderr)
            continue
        with open(gold_file) as gf:
            gold_rows = [json.loads(line) for line in gf]
        preds = data["predictions"]
        for pred_item, gold_row in zip(preds, gold_rows):
            target = " ".join([gold_row["prefix_acceptable"], gold_row["property_phrase"]]).strip()
            correct = pred_item["pred"].strip() == target
            items.append({"id": pred_item["id"], "subtask": subtask, "correct": correct})
    return items


def score_global_piqa(pred_dict, gold_file):
    """GlobalPIQA: correct if pred == solution{label}."""
    items = []
    with open(gold_file) as gf:
        gold_rows = [json.loads(line) for line in gf]

    # Build gold lookup by example_id
    gold_by_id = {}
    for row in gold_rows:
        eid = row.get("example_id", "")
        gold_by_id[eid] = row

    for example_id, data in pred_dict.items():
        # Match prediction example_id to gold example_id
        gold_row = gold_by_id.get(example_id)
        if gold_row is None:
            # Try without the _eng_latn suffix
            print(f"  WARNING: no gold row for GlobalPIQA {example_id}", file=sys.stderr)
            continue
        label = gold_row["label"]
        correct_answer = gold_row[f"solution{label}"].strip()
        preds = data["predictions"]
        for pred_item in preds:
            correct = pred_item["pred"].strip() == correct_answer
            items.append({"id": pred_item["id"], "correct": correct,
                          "example_id": example_id})
    return items


# ---- Transition matrix computation ----

def compute_transitions(items_a, items_b, label_a, label_b):
    """Compute 2x2 correctness transition matrix between two arms.
    items_a and items_b are lists of dicts with 'id' and 'correct' keys.
    Returns dict with CC, CW, WC, WW counts and derived metrics.
    """
    # Build lookup by id
    a_by_id = {item["id"]: item["correct"] for item in items_a}
    b_by_id = {item["id"]: item["correct"] for item in items_b}
    
    # Find common ids
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    
    cc = cw = wc = ww = 0
    for item_id in common_ids:
        ca = a_by_id[item_id]
        cb = b_by_id[item_id]
        if ca and cb:
            cc += 1
        elif ca and not cb:
            cw += 1
        elif not ca and cb:
            wc += 1
        else:
            ww += 1
    
    total = cc + cw + wc + ww
    a_correct = cc + cw
    b_correct = cc + wc
    
    # McNemar-style asymmetry: newly correct vs newly lost
    net_gain = wc - cw  # positive means B gained more than lost vs A
    
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_common": total,
        "CC": cc,
        "CW": cw,
        "WC": wc,
        "WW": ww,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "a_accuracy": a_correct / max(total, 1) * 100,
        "b_accuracy": b_correct / max(total, 1) * 100,
        "accuracy_delta": (b_correct - a_correct) / max(total, 1) * 100,
        "retained_rate": cc / max(a_correct, 1) * 100,  # fraction of A's correct items kept by B
        "discovery_rate": wc / max(ww + cw, 1) * 100,   # fraction of A's wrong items fixed by B (note: denom is A-wrong pool = WC+WW)
        "loss_rate": cw / max(a_correct, 1) * 100,       # fraction of A's correct items lost by B
        "net_gain": net_gain,
        "churn": cw + wc,
        "churn_fraction": (cw + wc) / max(total, 1) * 100,
    }


# ---- Subfamily analysis ----

def transitions_by_group(items_a, items_b, label_a, label_b, group_key):
    """Compute transitions grouped by a given key in the items dicts."""
    groups_a = collections.defaultdict(list)
    groups_b = collections.defaultdict(list)
    for item in items_a:
        groups_a[item.get(group_key, "unknown")].append(item)
    for item in items_b:
        groups_b[item.get(group_key, "unknown")].append(item)
    
    results = {}
    for group in sorted(set(groups_a.keys()) | set(groups_b.keys())):
        if group in groups_a and group in groups_b:
            results[str(group)] = compute_transitions(groups_a[group], groups_b[group], label_a, label_b)
    return results


# ---- Main ----

def main():
    out_dir = pathlib.Path("experiments/archive/representation_and_objectives/data/correctness_transitions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task definitions: (task_name, scorer_fn, gold_path, group_key_for_subfamilies)
    tasks = [
        ("BLiMP", score_blimp_supplement, FAST_EVAL / "blimp_fast", "subtask"),
        ("Supplement", score_blimp_supplement, FAST_EVAL / "supplement_fast", "subtask"),
        ("EWoK", score_ewok, FAST_EVAL / "evaluation_data" / "fast_eval" / "ewok_fast", "subtask"),
        ("Entity", score_entity, FAST_EVAL / "entity_tracking_fast", "numops"),
        ("COMPS", score_comps, FULL_EVAL / "comps", "subtask"),
        ("GlobalPIQA_parallel", score_global_piqa, FAST_EVAL / "global_piqa_parallel" / "eng_latn.jsonl", "example_id"),
        ("GlobalPIQA_nonparallel", score_global_piqa, FAST_EVAL / "global_piqa_nonparallel" / "eng_latn.jsonl", None),
    ]

    all_results = {}
    arm_items = {}  # {task -> {arm -> items}}

    for task_name, scorer, gold_path, group_key in tasks:
        print(f"\n=== {task_name} ===")
        task_items = {}
        for arm_name, arm_root in ARMS.items():
            try:
                pf = pred_path(arm_root, task_name)
                pred_dict = load_predictions(pf)
                items = scorer(pred_dict, gold_path)
                task_items[arm_name] = items
                n_correct = sum(1 for x in items if x["correct"])
                print(f"  {arm_name}: {n_correct}/{len(items)} correct ({n_correct/max(len(items),1)*100:.2f}%)")
            except Exception as e:
                print(f"  {arm_name}: ERROR - {e}", file=sys.stderr)
                task_items[arm_name] = []

        arm_items[task_name] = task_items

        # Compute transitions for all three pairs
        pairs = [
            ("compact_repeat", "compact_view", "view_vs_repeat"),
            ("adjbreak", "compact_view", "view_vs_adjbreak"),
            ("compact_repeat", "adjbreak", "adjbreak_vs_repeat"),
        ]

        task_transitions = {}
        for base_arm, improved_arm, pair_name in pairs:
            if task_items.get(base_arm) and task_items.get(improved_arm):
                trans = compute_transitions(
                    task_items[base_arm], task_items[improved_arm],
                    base_arm, improved_arm
                )
                task_transitions[pair_name] = trans
                print(f"\n  {pair_name}: CC={trans['CC']} CW={trans['CW']} WC={trans['WC']} WW={trans['WW']}")
                print(f"    net_gain={trans['net_gain']:+d} churn_frac={trans['churn_fraction']:.1f}%")
                print(f"    retained_rate={trans['retained_rate']:.1f}% discovery_rate={trans['discovery_rate']:.1f}% loss_rate={trans['loss_rate']:.1f}%")

                # Subfamily breakdown for interesting tasks
                if group_key and task_name in ("EWoK", "Entity", "COMPS", "GlobalPIQA_parallel"):
                    subfam = transitions_by_group(
                        task_items[base_arm], task_items[improved_arm],
                        base_arm, improved_arm, group_key
                    )
                    task_transitions[f"{pair_name}_by_{group_key}"] = subfam

        all_results[task_name] = task_transitions

    # ---- Summary table ----
    print("\n\n" + "=" * 80)
    print("SUMMARY: Net item gains per task for each arm pair")
    print("=" * 80)
    print(f"{'Task':<25} {'view-repeat':>15} {'view-adjbreak':>15} {'adjbrk-repeat':>15}")
    print("-" * 70)
    for task_name in [t[0] for t in tasks]:
        tr = all_results.get(task_name, {})
        def fmt(pair_key):
            t = tr.get(pair_key, {})
            if not t:
                return "N/A"
            return f"{t['net_gain']:+d} ({t['churn_fraction']:.0f}%)"
        print(f"{task_name:<25} {fmt('view_vs_repeat'):>15} {fmt('view_vs_adjbreak'):>15} {fmt('adjbreak_vs_repeat'):>15}")

    # ---- Mechanism-relevant summary ----
    print("\n\nMECHANISM QUALITY CHECK:")
    print("If view→repeat gains are real competence (not churn), we expect:")
    print("  - High retained_rate (most of repeat's successes kept)")
    print("  - Positive net_gain (view solves items repeat cannot)")
    print("  - Low loss_rate (view rarely loses what repeat had)")
    print()
    for task_name in [t[0] for t in tasks]:
        tr = all_results.get(task_name, {})
        vr = tr.get("view_vs_repeat")
        if vr:
            quality = "STRONG" if vr["retained_rate"] > 90 and vr["net_gain"] > 0 and vr["loss_rate"] < 15 else \
                      "MODERATE" if vr["retained_rate"] > 80 and vr["net_gain"] > 0 else "WEAK/CHURN"
            print(f"  {task_name}: retained={vr['retained_rate']:.1f}% discovery={vr['discovery_rate']:.1f}% loss={vr['loss_rate']:.1f}% net={vr['net_gain']:+d} => {quality}")

    # Save
    out_file = out_dir / "correctness_transitions.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_file}")

    # Also save arm-level item correctness for deeper analysis
    summary_file = out_dir / "arm_correctness_summary.json"
    summary = {}
    for task_name, items_by_arm in arm_items.items():
        summary[task_name] = {}
        for arm_name, items in items_by_arm.items():
            n = len(items)
            nc = sum(1 for x in items if x["correct"])
            summary[task_name][arm_name] = {"n_items": n, "n_correct": nc, "accuracy": nc / max(n, 1) * 100}
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved arm summary to {summary_file}")


if __name__ == "__main__":
    main()
