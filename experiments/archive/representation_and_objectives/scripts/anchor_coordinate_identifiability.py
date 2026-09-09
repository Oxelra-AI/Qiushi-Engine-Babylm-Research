#!/usr/bin/env python3
"""research: role-coordinate identifiability simulation.

This is a non-neural companion to the factorized role-anchor run. It asks what a
learner can identify from different kinds of supervision when each predicate has
an unobserved absolute role coordinate. The point is to separate absolute role
anchoring from relative held-held consistency, because research's factorized run
showed that a shuffled held-anchor can score perfectly on held-held composition
while being completely wrong on mixed held-seen composition.

The simulation uses R role categories. Seen predicates have known coordinates.
Held predicates are unknown. Supervision types:
  - noheld/exposure: no coordinate information for held predicates;
  - true_anchor: direct role-name anchor gives the correct coordinate;
  - shuffled_anchor: all held coordinates are transformed by a wrong permutation;
  - held_seen_coverage_m: m random equality/difference comparisons to known seen
    coordinates per held predicate;
  - held_held_only_m: m relative comparisons among held predicates only.

Reported surfaces:
  - mixed_accuracy: correct absolute role equality between a held predicate and a
    seen predicate; this is the meaningful transfer surface.
  - heldheld_consistency: correct equality/difference among held predicates; this
    can be high even when all held coordinates are globally permuted.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path
from typing import Any

import numpy as np

OUT_DIR = Path("experiments/archive/representation_and_objectives/data/anchor_coordinate_identifiability")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def derangement(R: int, rng: random.Random) -> list[int]:
    while True:
        p=list(range(R)); rng.shuffle(p)
        if all(p[i] != i for i in range(R)):
            return p


def infer_from_seen_comparisons(R: int, obs: list[tuple[int, int]]) -> int:
    """obs are (seen_role, eq_label), eq_label=1 means held role equals seen_role."""
    possible=set(range(R))
    for r,y in obs:
        if y:
            possible &= {r}
        else:
            possible -= {r}
    if len(possible) == 1:
        return next(iter(possible))
    # deterministic unresolved default; this makes residual ambiguity visible.
    return min(possible) if possible else 0


def infer_clusters_from_heldheld(R: int, true_roles: list[int], comps: list[tuple[int,int,int]]) -> list[int]:
    """Infer relative clusters from equality constraints, then assign cluster IDs by order.

    This deliberately lacks any absolute seen anchor, so cluster labels are only
    arbitrary. Correct held-held equality may be high, but mixed absolute accuracy
    should remain near chance.
    """
    n=len(true_roles)
    parent=list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    # Use only equality positives to cluster. Negative constraints are not enough
    # to assign an absolute color without anchors.
    for i,j,eq in comps:
        if eq: union(i,j)
    cluster_to_id={}; out=[]
    for i in range(n):
        r=find(i)
        if r not in cluster_to_id:
            cluster_to_id[r]=len(cluster_to_id) % R
        out.append(cluster_to_id[r])
    return out


def mixed_accuracy(est: list[int], true: list[int], seen_roles: list[int]) -> float:
    ok=0; n=0
    for e,t in zip(est,true):
        for sr in seen_roles:
            ok += int((e == sr) == (t == sr)); n += 1
    return ok/n


def heldheld_consistency(est: list[int], true: list[int]) -> float:
    ok=0; n=0
    for i in range(len(true)):
        for j in range(i+1,len(true)):
            ok += int((est[i] == est[j]) == (true[i] == true[j])); n += 1
    return ok/max(1,n)


def trial(R: int, n_seen: int, n_held: int, m: int, rng: random.Random) -> dict[str,dict[str,float]]:
    seen_roles=[i % R for i in range(n_seen)]
    true=[i % R for i in range(n_held)]
    rng.shuffle(true)
    # no information: deterministic default, intentionally ambiguous.
    default=[0 for _ in true]
    out={
        "noheld": {"mixed": mixed_accuracy(default,true,seen_roles), "heldheld": heldheld_consistency(default,true)},
        "true_anchor": {"mixed": mixed_accuracy(true,true,seen_roles), "heldheld": heldheld_consistency(true,true)},
    }
    perm=derangement(R,rng)
    shuf=[perm[t] for t in true]
    out["shuffled_anchor"]={"mixed": mixed_accuracy(shuf,true,seen_roles), "heldheld": heldheld_consistency(shuf,true)}
    # m random comparisons to seen predicates per held predicate.
    est=[]
    for t in true:
        obs=[]
        for _ in range(m):
            sr=rng.choice(seen_roles)
            obs.append((sr, int(t == sr)))
        est.append(infer_from_seen_comparisons(R, obs))
    out["held_seen_coverage"]={"mixed": mixed_accuracy(est,true,seen_roles), "heldheld": heldheld_consistency(est,true)}
    # m random comparisons among held predicates per held predicate; no absolute anchor.
    comps=[]
    for _ in range(max(0,m*n_held)):
        i,j=rng.sample(range(n_held),2)
        comps.append((i,j,int(true[i] == true[j])))
    hh=infer_clusters_from_heldheld(R,true,comps)
    out["held_held_only"]={"mixed": mixed_accuracy(hh,true,seen_roles), "heldheld": heldheld_consistency(hh,true)}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--roles", type=str, default="2,3,4,6")
    ap.add_argument("--n-seen", type=int, default=12)
    ap.add_argument("--n-held", type=int, default=24)
    ap.add_argument("--max-m", type=int, default=10)
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng=random.Random(25502)
    results={}
    for R in [int(x) for x in args.roles.split(',') if x.strip()]:
        rres={}
        for m in range(args.max_m+1):
            vals={}
            for _ in range(args.trials):
                tr=trial(R,args.n_seen,args.n_held,m,rng)
                for arm,d in tr.items():
                    vals.setdefault(arm,{"mixed":[],"heldheld":[]})
                    vals[arm]["mixed"].append(d["mixed"]); vals[arm]["heldheld"].append(d["heldheld"])
            rres[str(m)]={arm:{k:{"mean":float(np.mean(v)),"std":float(np.std(v))} for k,v in surf.items()} for arm,surf in vals.items()}
        results[str(R)]=rres
    summary={"status":"ANCHOR_COORDINATE_IDENTIFIABILITY","created_utc":now(),"config":vars(args),"results":results,
             "interpretation":"Absolute role-coordinate anchors remove a permutation/inversion ambiguity. Held-held relative consistency can be perfect under a wrong global coordinate transform, so mixed held-seen surfaces are the decisive evidence. Random composition coverage with anchored seen predicates can also identify coordinates, but only insofar as it supplies absolute constraints; pure exposure and held-held-only constraints do not."}
    write_json(OUT_DIR/"anchor_coordinate_identifiability_summary.json", summary)
    md=["# research role-coordinate identifiability simulation","",summary["interpretation"],"", "## Selected means", "", "| R | m | arm | mixed | held-held |", "|---:|---:|---|---:|---:|"]
    for R in [2,3,4,6]:
        if str(R) not in results: continue
        for m in [0,1,2,4,8,10]:
            if str(m) not in results[str(R)]: continue
            for arm in ["noheld","true_anchor","shuffled_anchor","held_seen_coverage","held_held_only"]:
                d=results[str(R)][str(m)][arm]
                md.append(f"| {R} | {m} | {arm} | {d['mixed']['mean']:.3f} | {d['heldheld']['mean']:.3f} |")
    md += ["", f"Summary JSON: `{OUT_DIR/'anchor_coordinate_identifiability_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/anchor_coordinate_identifiability/anchor_coordinate_identifiability_summary.md')).write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status":summary["status"],"summary_json":str(OUT_DIR/"anchor_coordinate_identifiability_summary.json"),"R2_m0":results["2"]["0"],"R4_m4":results["4"]["4"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
