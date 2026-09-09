#!/usr/bin/env python3
"""research: evaluate packed target-selective 100M endpoints and run signature readout.

Run this after both training tasks finish and their `hf_model/chck_100M` directories
and `scientific_metrics.json` files are present.
"""
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys, time

EVAL = pathlib.Path("experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py")
SIGNATURE = pathlib.Path("experiments/archive/representation_and_objectives/scripts/packed_targetselect_endpoint_signature.py")
FULL_JSON = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/per_target/compact_view_reinvest.json")
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

DEFAULT_ARMS = {
    "drop_abs": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_abs_content_100M_fast"),
    "drop_copied_word": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_copied_content_wholeword_100M_fast"),
}


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    print("RUN", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        p=subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return {"cmd":cmd,"returncode":p.returncode,"elapsed_sec":round(time.time()-t0,1),"log":str(log_path)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drop_abs_run", default=str(DEFAULT_ARMS["drop_abs"]))
    ap.add_argument("--drop_copied_run", default=str(DEFAULT_ARMS["drop_copied_word"]))
    ap.add_argument("--out_root", default="experiments/archive/representation_and_objectives/data/packed_targetselect_100M_eval")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-signature", action="store_true", help="Run only endpoint evaluations; the research/231 v2 signature supersedes the quick research signature reader.")
    args=ap.parse_args()
    out=pathlib.Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    gpus=[int(x) for x in args.gpus.split(',') if x.strip()]
    if len(gpus)<2: gpus=gpus*2
    arms={"drop_abs":pathlib.Path(args.drop_abs_run), "drop_copied_word":pathlib.Path(args.drop_copied_run)}
    preflight={}
    for name,run_dir in arms.items():
        model=run_dir/"hf_model/chck_100M"
        metrics=run_dir/"scientific_metrics.json"
        preflight[name]={"run_dir":str(run_dir),"model_exists":model.exists(),"metrics_exists":metrics.exists()}
        if not model.exists() or not metrics.exists():
            summary={"status":"PACKED_TARGETSELECT_EVAL_WAITING_FOR_TRAINING","preflight":preflight}
            (out/"eval_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(summary,indent=2),flush=True)
            return
    results={"preflight":preflight}
    specs=[("drop_abs", arms["drop_abs"], gpus[0], "dropabs100"), ("drop_copied_word", arms["drop_copied_word"], gpus[1], "dropcopied_word100")]
    # Run both evaluations concurrently (one per GPU).
    procs={}
    for name,run_dir,gpu,target in specs:
        arm_out=out/name
        cmd=[sys.executable,"-B",str(EVAL),"--arm","reinvest","--target",target,"--run-dir",str(run_dir),"--endpoint","chck_100M","--out-root",str(arm_out),"--collate-root",str(out/"collated"),"--gpu",str(gpu),"--columns",*COLUMNS]
        if args.force: cmd.append("--force")
        log=out/"logs"/f"eval_{name}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        f=log.open("w",encoding="utf-8")
        print("LAUNCH",name,"gpu",gpu," ".join(cmd),flush=True)
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
        procs[name]={"process":p,"file":f,"cmd":cmd,"log":str(log),"start":time.time()}
    for name,rec in procs.items():
        rc=rec["process"].wait(); rec["file"].close()
        results[name]={"returncode":rc,"elapsed_sec":round(time.time()-rec["start"],1),"cmd":rec["cmd"],"log":rec["log"]}
        if rc!=0:
            results[name]["log_tail"]=pathlib.Path(rec["log"]).read_text(encoding="utf-8",errors="replace")[-6000:]
    if any(results[n]["returncode"]!=0 for n in ["drop_abs","drop_copied_word"]):
        summary={"status":"PACKED_TARGETSELECT_EVAL_FAILED","results":results}
        (out/"eval_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
        raise SystemExit(1)
    if args.skip_signature:
        summary={"status":"PACKED_TARGETSELECT_ENDPOINT_EVAL_DONE","out_root":str(out),"results":results,"signature_json":None,"meaning":"Endpoint prediction generation only; strengthened signature v2 should be run by the postrun reader."}
        (out/"eval_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
        return
    arm_jsons=[f"drop_abs={out/'drop_abs/per_target/dropabs100.json'}", f"drop_copied_word={out/'drop_copied_word/per_target/dropcopied_word100.json'}"]
    sig_cmd=[sys.executable,"-B",str(SIGNATURE),"--full-json",str(FULL_JSON),"--output_dir",str(out/"signature")]
    for s in arm_jsons:
        sig_cmd.extend(["--arm-json",s])
    sig=run(sig_cmd,out/"logs/signature.log")
    results["signature"]=sig
    status="PACKED_TARGETSELECT_EVAL_AND_SIGNATURE_DONE" if sig["returncode"]==0 else "PACKED_TARGETSELECT_SIGNATURE_FAILED"
    summary={"status":status,"out_root":str(out),"results":results,"signature_json":str(out/"signature/packed_targetselect_endpoint_signature.json")}
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    if status.endswith("FAILED"):
        raise SystemExit(1)

if __name__=='__main__':
    main()
