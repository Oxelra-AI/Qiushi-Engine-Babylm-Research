#!/usr/bin/env python3
"""research: train bridge_compactgap or bridge_mid DeBERTa 100M.

Uses the exact research stock DeBERTa recipe on research bridge 100M streams.
Adapted from research extractive trainer.

No official selected eval, SuperGLUE, AoA, upload, or leaderboard submission.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, hashlib, json, os, pathlib, subprocess, sys, time

def find_root():
    return _PUBLIC_ROOT

ROOT = find_root()
WS = ROOT/"experiments/archive/frontier_consolidation"
RUNS = WS/"training/runs"
TRAINER = ROOT/"experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WS/"data/compliant_tokenizer"
STREAM_DIR = WS/"data/bridge_pools_and_streams"
MANIFEST = STREAM_DIR/"bridge_pools_and_streams_manifest.json"
TOK_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"

ARMS = {
    "bridge_compactgap": {"stream": "bridge_compactgap_100M.jsonl",
        "run": "bridge_compactgap_deberta100M_seed43022",
        "role": "compact-gap-matched adjacency, zero novel content, source-only"},
    "bridge_mid": {"stream": "bridge_mid_100M.jsonl",
        "run": "bridge_mid_deberta100M_seed43022",
        "role": "intermediate adjacency dose, zero novel content, source-only"},
}

RECIPE = dict(hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
    seed=43, extra_init_seed=43022, train_rng_seed=43023,
    batch_size=256, seq_length=256, max_seq_length=256,
    learning_rate=0.001, weight_decay=0.01, warmup_fraction=0.06,
    lr_total_steps=2529, masking_curriculum="wwm_fixed",
    mask_prob_start=0.15, mask_prob_end=0.15,
    checkpoint_words=10_000_000, max_word_exposure=100_000_000,
    num_workers=0, log_every=50, dynamics_trace_every=200)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    args=ap.parse_args()
    cfg=ARMS[args.arm]
    stream_path=STREAM_DIR/cfg["stream"]
    run_dir=RUNS/cfg["run"]

    # Preflight
    for p in [TRAINER, TOKENIZER/"tokenizer.json", stream_path, MANIFEST]:
        assert p.exists(), f"missing {p}"
    tok_sha=sha(TOKENIZER/"tokenizer.json")
    assert tok_sha==TOK_SHA, f"tokenizer sha mismatch: {tok_sha}"
    manifest=json.loads(MANIFEST.read_text())
    expected_sha=manifest["stream_results"][args.arm]["stream_sha256"]
    actual_sha=sha(stream_path)
    assert actual_sha==expected_sha, f"stream sha mismatch: {actual_sha} != {expected_sha}"
    sr=manifest["stream_results"][args.arm]
    assert sr["total_rows"]==647400 and sr["total_words"]==100000000 and sr["replaced_rows"]==30050

    if args.count_words:
        rows=words=0
        with open(stream_path) as f:
            for line in f:
                o=json.loads(line); rows+=1; words+=o["words"]
        assert rows==647400 and words==100000000, f"count: {rows} rows {words} words"
        print(f"Word count verified: {rows} rows, {words} words")

    preflight={"arm":args.arm,"stream":str(stream_path),"stream_sha":actual_sha,
               "tokenizer_sha":tok_sha,"recipe":RECIPE,"run_dir":str(run_dir),
               "role":cfg["role"],"no_eval_upload_aoa_leaderboard":True}

    cmd=[sys.executable,"-B",str(TRAINER),
         "--example_jsonl",str(stream_path),
         "--example_jsonl_label",f"step233_{args.arm}_source_attested_reinvest",
         "--example_jsonl_meta",str(MANIFEST),
         "--output_dir",str(run_dir),
         "--tokenizer_path",str(TOKENIZER),
         "--tokenizer_label","compliant16k_reinvest10M"]
    for k,v in RECIPE.items(): cmd+=[f"--{k}",str(v)]

    if args.dry_run:
        preflight["command"]=cmd; print(json.dumps(preflight,indent=2)); return

    run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/"train_command.json").write_text(json.dumps(preflight,indent=2)+"\n")
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    env["TOKENIZERS_PARALLELISM"]="false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    env["TMPDIR"]=f"/tmp/q_frontier_consolidation_step233_{args.arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True,exist_ok=True)
    env["PYTHONDONTWRITEBYTECODE"]="1"

    started=now()
    with open(run_dir/"train_stdout.log","w") as out, open(run_dir/"train_stderr.log","w") as err:
        out.write(json.dumps({"event":"start","utc":started,"arm":args.arm,"gpu":args.gpu})+"\n"); out.flush()
        proc=subprocess.run(cmd,stdout=out,stderr=err,text=True,env=env,cwd=str(ROOT))
    finished=now()

    mp=run_dir/"scientific_metrics.json"
    result={"status":f"BRIDGE_TRAIN_{'FINISHED' if proc.returncode==0 else 'FAILED'}",
            "arm":args.arm,"gpu":args.gpu,"returncode":proc.returncode,
            "started_utc":started,"finished_utc":finished,
            "run_dir":str(run_dir),"stdout_log":str(run_dir/"train_stdout.log"),
            "stderr_log":str(run_dir/"train_stderr.log"),
            "metrics_path":str(mp),"metrics_exists":mp.exists(),
            "tokenizer_dir":str(TOKENIZER),"no_official_eval_upload_aoa_or_leaderboard":True}
    if mp.exists():
        m=json.loads(mp.read_text())
        result["metrics"]={k:m.get(k) for k in ["word_exposure","actual_training_steps","loss_first","loss_last","parameter_count","vocab_size","tokenizer_label"]}
        result["metrics"]["saved_checkpoints"]=len(m.get("saved_checkpoints",[]))
        sc=m.get("saved_checkpoints",[])
        result["metrics"]["first_checkpoint"]=sc[0]["name"] if sc else None
        result["metrics"]["last_checkpoint"]=sc[-1]["name"] if sc else None
    if proc.returncode!=0:
        result["stderr_tail"]=open(run_dir/"train_stderr.log").read()[-4000:] if (run_dir/"train_stderr.log").exists() else ""
    (run_dir/"launcher_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); raise SystemExit(proc.returncode)

if __name__=="__main__": main()
