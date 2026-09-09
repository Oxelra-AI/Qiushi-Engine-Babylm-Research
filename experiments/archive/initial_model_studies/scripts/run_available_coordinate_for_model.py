#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
FULL_EVAL = STRICT / "evaluation_data" / "full_eval"
BACKEND = "mlm"
TASKS = [
    ("blimp", "blimp", FULL_EVAL / "blimp_filtered", "blimp_filtered"),
    ("supplement", "blimp", FULL_EVAL / "supplement_filtered", "supplement_filtered"),
    ("entity_tracking", "entity_tracking", FULL_EVAL / "entity_tracking", "entity_tracking"),
    ("comps", "comps", FULL_EVAL / "comps", "comps"),
    ("global_piqa_parallel", "global_piqa_parallel", FULL_EVAL / "global_piqa_parallel", "global_piqa_parallel"),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", FULL_EVAL / "global_piqa_nonparallel", "global_piqa_nonparallel"),
]

def setup_env():
    env=os.environ.copy(); hf_home=ROOT/"training/hf_home"
    env["HF_HOME"]=str(hf_home.resolve()); env["HF_HUB_CACHE"]=str((hf_home/"hub").resolve())
    env["TRANSFORMERS_CACHE"]=str((hf_home/"transformers").resolve())
    env["HF_MODULES_CACHE"]=str((ROOT/"training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM","false")
    for k in ["HF_HOME","HF_HUB_CACHE","TRANSFORMERS_CACHE","HF_MODULES_CACHE"]: pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, logf):
    line="$ "+" ".join(cmd); print(line, flush=True); logf.write("\n"+line+"\n"); logf.flush()
    p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True); logf.write(p.stdout+f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode!=0: raise RuntimeError(f"failed {p.returncode}: {' '.join(cmd)}\n{p.stdout[-8000:]}")

def read_avg(report:pathlib.Path):
    txt=report.read_text(encoding="utf-8", errors="replace")
    m=re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m: raise RuntimeError(f"cannot parse average from {report}\n{txt[:1000]}")
    return float(m.group(1))

def read_reading(report:pathlib.Path):
    txt=report.read_text(encoding="utf-8", errors="replace")
    out={}
    for label,key in [("EYE TRACKING SCORE","reading_eye_tracking"),("SELF-PACED READING SCORE","reading_self_paced")]:
        m=re.search(re.escape(label)+r":\s*([0-9.\-]+)", txt)
        if not m: raise RuntimeError(f"cannot parse {label} from {report}\n{txt}")
        out[key]=float(m.group(1))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_note", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--revision", default="chck_100M")
    args=ap.parse_args()
    model=pathlib.Path(args.model_path).resolve(); outdir=pathlib.Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    log_path=pathlib.Path(args.log); log_path.parent.mkdir(parents=True, exist_ok=True)
    scores={}; reports={}; env=setup_env()
    with log_path.open("a", encoding="utf-8") as logf:
        for col,task,data_path,dataset_name in TASKS:
            if not data_path.exists() or not any(data_path.rglob('*')): raise RuntimeError(f"missing data {data_path}")
            run([sys.executable,"-m","evaluation_pipeline.sentence_zero_shot.run","--model_path_or_name",str(model),"--backend",BACKEND,"--task",task,"--data_path",str(data_path.resolve()),"--save_predictions","--revision_name",args.revision,"--batch_size","64","--output_dir",str(outdir)], env, logf)
            report=outdir/"hf_model"/args.revision/"zero_shot"/BACKEND/task/dataset_name/"best_temperature_report.txt"
            scores[col]=read_avg(report); reports[col]=str(report)
        reading_csv=FULL_EVAL/"reading"/"reading_data.csv"
        run([sys.executable,"-m","evaluation_pipeline.reading.run","--model_path_or_name",str(model),"--backend",BACKEND,"--data_path",str(reading_csv.resolve()),"--revision_name",args.revision,"--output_dir",str(outdir)], env, logf)
        reading_report=outdir/"hf_model"/args.revision/"zero_shot"/BACKEND/"reading"/"report.txt"
        reports["reading"]=str(reading_report); scores.update(read_reading(reading_report))
    derived={"GlobalPIQA_mean_parallel_nonparallel":(scores["global_piqa_parallel"]+scores["global_piqa_nonparallel"])/2.0,"Reading_mean_eye_self_paced":(scores["reading_eye_tracking"]+scores["reading_self_paced"])/2.0}
    payload={"run_name":args.run_name,"model_path":str(model),"revision":args.revision,"backend":BACKEND,"scores":scores,"derived_columns":derived,"reports":reports,"missing_columns":["EWoK","AoA","SuperGLUE"],"note":"Available zero-shot/Reading coordinate only; EWoK full gated, AoA/SuperGLUE not run by this script."}
    out_json=pathlib.Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True); out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=[f"# Available official coordinate — {args.run_name}","",f"Evidence JSON: `{out_json}`","","| column/task | score |","|---|---:|"]
    for k,label in [("blimp","BLiMP"),("supplement","Supplement"),("entity_tracking","Entity"),("comps","COMPS"),("global_piqa_parallel","GlobalPIQA parallel"),("global_piqa_nonparallel","GlobalPIQA nonparallel")]: lines.append(f"| {label} | {scores[k]:.2f} |")
    lines += [f"| GlobalPIQA mean | {derived['GlobalPIQA_mean_parallel_nonparallel']:.2f} |", f"| Reading eye | {scores['reading_eye_tracking']:.2f} |", f"| Reading self-paced | {scores['reading_self_paced']:.2f} |", f"| Reading simple mean | {derived['Reading_mean_eye_self_paced']:.2f} |"]
    pathlib.Path(args.out_note).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":"AVAILABLE_COORDINATE_DONE","run_name":args.run_name,"out":str(out_json),"scores":scores,"derived":derived}, indent=2))
if __name__=="__main__": main()
