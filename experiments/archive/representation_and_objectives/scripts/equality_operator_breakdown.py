#!/usr/bin/env python3
"""research equality-operator breakdown by held-name character coverage."""
from __future__ import annotations
import copy, json, random, sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT = Path("experiments/archive/representation_and_objectives")
SCRIPT_DIR = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import equality_operator_emergence_pilot as pilot  # noqa: E402

OUT = PROJECT / "data/equality_operator_breakdown"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def eval_queries(matcher, examples, train_letters, device):
    matcher.eval()
    rows = []
    with torch.no_grad():
        for text, names in examples:
            _, forms = pilot.base.raw_tokenize(text)
            forms_l = [str(f).lower() for f in forms]
            tcp = pilot.forms_to_tensor(forms_l, device)
            smask = pilot.special_mask(forms_l, device)
            positions = {n.lower(): [i for i, f in enumerate(forms_l) if f == n.lower()] for n in names}
            for cand in names:
                cand_l = cand.lower()
                if not positions.get(cand_l):
                    continue
                q = pilot.name_chars(cand_l).to(device)
                s = matcher.scores(tcp, q).clone()
                s[smask] = -1.0
                arg = int(torch.argmax(s).item())
                true_pos = positions[cand_l][0]
                other = names[1] if cand == names[0] else names[0]
                other_l = other.lower()
                other_pos = positions[other_l][0]
                rows.append({
                    "candidate": cand_l,
                    "other": other_l,
                    "correct": int(arg == true_pos),
                    "arg_is_other": int(arg == other_pos),
                    "arg_form": forms_l[arg] if 0 <= arg < len(forms_l) else "",
                    "candidate_has_unseen_letter": int(any(ch not in train_letters for ch in cand_l)),
                    "pair_has_unseen_letter": int(any(ch not in train_letters for ch in cand_l + other_l)),
                    "true_score": float(s[true_pos].cpu()),
                    "other_score": float(s[other_pos].cpu()),
                    "max_score": float(torch.max(s).cpu()),
                    "true_minus_other": float((s[true_pos] - s[other_pos]).cpu()),
                    "true_minus_argmax": float((s[true_pos] - torch.max(s)).cpu()),
                })
    return rows


def summarize_rows(rows):
    def frac(xs, key):
        return sum(r[key] for r in xs) / len(xs) if xs else float("nan")
    def mean(xs, key):
        return sum(r[key] for r in xs) / len(xs) if xs else float("nan")
    out = {"n": len(rows), "acc": frac(rows, "correct"),
           "arg_other_frac": frac(rows, "arg_is_other"),
           "true_minus_other_mean": mean(rows, "true_minus_other"),
           "true_minus_argmax_mean": mean(rows, "true_minus_argmax")}
    for flag in ["candidate_has_unseen_letter", "pair_has_unseen_letter"]:
        for val in [0, 1]:
            xs = [r for r in rows if r[flag] == val]
            out[f"{flag}_{val}_n"] = len(xs)
            out[f"{flag}_{val}_acc"] = frac(xs, "correct")
            out[f"{flag}_{val}_true_minus_other"] = mean(xs, "true_minus_other")
            out[f"{flag}_{val}_true_minus_argmax"] = mean(xs, "true_minus_argmax")
    by_name = {}
    for name in sorted(set(r["candidate"] for r in rows)):
        xs = [r for r in rows if r["candidate"] == name]
        by_name[name] = {"n": len(xs), "acc": frac(xs, "correct"),
                         "has_unseen_letter": int(any(r["candidate_has_unseen_letter"] for r in xs)),
                         "true_minus_other": mean(xs, "true_minus_other"),
                         "true_minus_argmax": mean(xs, "true_minus_argmax")}
    out["by_candidate_name"] = by_name
    return out


def main():
    random.seed(29900); torch.manual_seed(29900)
    device = torch.device("cpu")
    ts, tc, es, ec, _, _ = pilot.base.load_dataset(pilot.base.DEFAULT_DATA_ROOT, pilot.base.DEFAULT_ARM)
    train_events = pilot.unique_events(ts, tc)
    eval_events = pilot.unique_events(es, ec)
    train_letters = set(pilot.letters_in_names(pilot.lower_names(train_events)))

    torch.manual_seed(29900)
    shared = pilot.SharedPosMatcher(16).to(device)
    torch.manual_seed(29900)
    dual0 = pilot.DualPosMatcher(16).to(device)

    models = {"shared_pos_initial": shared, "dual_pos_initial": dual0}
    dual_name = copy.deepcopy(dual0)
    pilot.train_name_level(dual_name, train_events, device, 60, 5e-2, 0)
    models["dual_pos_name_level_train"] = dual_name
    dual_train_alpha = copy.deepcopy(dual0)
    pilot.train_char_pairs(dual_train_alpha, pilot.chars_to_ids(train_letters), device, 60, 5e-2, 0)
    models["dual_pos_train_alphabet_char_pairs"] = dual_train_alpha
    dual_full_alpha = copy.deepcopy(dual0)
    pilot.train_char_pairs(dual_full_alpha, list(range(1, 27)), device, 60, 5e-2, 0)
    models["dual_pos_full_alphabet_char_pairs"] = dual_full_alpha

    all_rows = {}
    summaries = {}
    for name, model in models.items():
        rows = eval_queries(model, eval_events, train_letters, device)
        all_rows[name] = rows
        summaries[name] = summarize_rows(rows)

    write_json(OUT / "equality_operator_breakdown.json", {"summaries": summaries, "rows": all_rows})

    lines = ["# research equality-operator held-name breakdown", "",
             "Train-name letter support: `" + ''.join(sorted(train_letters)) + "`", "",
             "| model | all | cand seen-letter only | cand has unseen letter | pair seen-letter only | pair has unseen letter | mean true-other | mean true-argmax |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['acc']:.3f} | {s['candidate_has_unseen_letter_0_acc']:.3f} | {s['candidate_has_unseen_letter_1_acc']:.3f} | {s['pair_has_unseen_letter_0_acc']:.3f} | {s['pair_has_unseen_letter_1_acc']:.3f} | {s['true_minus_other_mean']:.3f} | {s['true_minus_argmax_mean']:.3f} |")
    lines += ["", "## Official held candidates", ""]
    names = sorted(next(iter(summaries.values()))["by_candidate_name"].keys())
    lines.append("| name | unseen letter? | " + " | ".join(summaries.keys()) + " |")
    lines.append("|---|---:|" + "---:|" * len(summaries))
    for nm in names:
        unseen = summaries["shared_pos_initial"]["by_candidate_name"][nm]["has_unseen_letter"]
        vals = [summaries[m]["by_candidate_name"][nm]["acc"] for m in summaries]
        lines.append(f"| {nm} | {unseen} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/equality_operator_breakdown/equality_operator_breakdown.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "EQUALITY_OPERATOR_BREAKDOWN_COMPLETE",
                      "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/equality_operator_breakdown/equality_operator_breakdown.md')),
                      "json": str(OUT / "equality_operator_breakdown.json")}, indent=2))


if __name__ == "__main__":
    main()
