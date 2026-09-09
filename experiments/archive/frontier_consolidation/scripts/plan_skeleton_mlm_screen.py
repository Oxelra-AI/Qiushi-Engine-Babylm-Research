#!/usr/bin/env python3
"""research dry plan for a possible skeleton-vs-compact MLM screen.

This writes exact commands only; it launches nothing.  The plan is deliberately
limited to a short 20M/40M screen and must be reviewed against pending DeBERTa
common-grid and triangle evidence before any H100 use.
"""
from __future__ import annotations
import argparse, json, pathlib

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
SCAFFOLD = ROOT / "data/skeleton_reinvest_pool_scaffold"
OUT = ROOT / "data/skeleton_mlm_screen_plan"
TRAINER = ROOT / "scripts/train_density_arm.py"
EVAL = ROOT / "scripts/selected_mlm_checkpoint_eval.py"
TOKENIZER = ROOT / "data/compliant_tokenizer"

VARIANTS = {
    "compact": SCAFFOLD / "cleanqwen_fineweb_compact_skeleton_reinvest_10M.jsonl",
    "prefix_repeat": SCAFFOLD / "cleanqwen_fineweb_prefix_repeat_skeleton_reinvest_10M.jsonl",
    "content_spread": SCAFFOLD / "cleanqwen_fineweb_content_spread_skeleton_reinvest_10M.jsonl",
    "scored_source_skeleton": SCAFFOLD / "cleanqwen_fineweb_scored_source_skeleton_skeleton_reinvest_10M.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT)
    ap.add_argument("--total-words", type=int, default=40_000_000)
    ap.add_argument("--checkpoints", default="chck_20M chck_40M")
    ap.add_argument("--gpu0", default="0")
    ap.add_argument("--gpu1", default="1")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for idx, (name, pool) in enumerate(VARIANTS.items()):
        gpu = args.gpu0 if idx % 2 == 0 else args.gpu1
        run_dir = ROOT / f"training/runs/skeleton_screen_{name}_seed43022_40M"
        out_eval = ROOT / f"data/skeleton_screen_eval_{name}_40M"
        commands.append({
            "variant": name,
            "purpose": "short MLM mechanism screen only; compare broad cheap7 families, not endpoint chasing",
            "train_command": (
                f"CUDA_VISIBLE_DEVICES={gpu} PYTHONDONTWRITEBYTECODE=1 python -B {TRAINER} "
                f"--jsonl {pool} --tokenizer {TOKENIZER} --run-dir {run_dir} "
                f"--total-words {args.total_words} --checkpoint-interval 20000000 "
                f"--seed 43022 --gpu 0"
            ),
            "eval_command": (
                f"CUDA_VISIBLE_DEVICES={gpu} PYTHONDONTWRITEBYTECODE=1 python -B {EVAL} "
                f"--run-dir {run_dir} --out-dir {out_eval} --gpu 0 "
                f"--endpoints {args.checkpoints}"
            ),
            "pool": str(pool),
            "run_dir": str(run_dir),
            "eval_dir": str(out_eval),
        })
    plan = {
        "status": "SKELETON_MLM_SCREEN_PLAN_ONLY",
        "not_launched": True,
        "why_not_launched": [
            "The two DeBERTa common-grid comparisons remain pending",
            "A01 triangle official-compatible scores are not yet available",
            "source-only skeleton variants are telegraphic, so Lead/Reviewer should judge value before H100 use",
        ],
        "expensive_work_question": "Does source-wide exact-length skeleton recurrence reproduce the compact-view advantage over first-N repetition under the bidirectional MLM coordinate?",
        "decision_logic": [
            "continue skeleton route only if content_spread or scored_source_skeleton beats prefix_repeat across cheap6/cheap5 and relation/state without volatile-column-only movement",
            "if compact beats source-only skeleton broadly, natural compressed sentence/paraphrase style remains load-bearing",
            "if source-only skeleton harms broad families despite coverage, close extractive skeleton as too distribution-shifted",
            "do not use this as endpoint work; it is a mechanism dissection before any mature exposure",
        ],
        "commands": commands,
    }
    out_json = args.out_dir / "skeleton_mlm_screen_plan.json"
    out_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    md = args.out_dir / "skeleton_mlm_screen_plan.md"
    lines = ["# research skeleton-vs-compact MLM screen dry plan", "", "No command in this file was launched.", ""]
    lines.append(f"Question: {plan['expensive_work_question']}")
    lines.append("")
    lines.append("## Why this is not launched now")
    for x in plan["why_not_launched"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Planned train/eval commands")
    for c in commands:
        lines.append("")
        lines.append(f"### {c['variant']}")
        lines.append("```bash")
        lines.append(c["train_command"])
        lines.append(c["eval_command"])
        lines.append("```")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    md.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(args.out_dir), "n_commands": len(commands)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
