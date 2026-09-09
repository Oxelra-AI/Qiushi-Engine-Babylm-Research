# coherent86 endpoint robustness synthesis coherent86 endpoint robustness synthesis

## Result

Three independent official-compatible SuperGLUE finetunings of the identical bit-reproducible coherent86 model (`model.safetensors` SHA `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`):

| run | SuperGLUE | projected Overall(AoA0) | delta vs chck_82M |
|---|---:|---:|---:|
| frozen anchor coherent replay item reading first | 69.77796826428681 | 42.058107584920755 | +0.115626 |
| coherent86 endpoint robustness synthesis repeat | 69.78975515726182 | 42.059417239695755 | +0.116936 |
| chck84 late item dynamics synthesis | 69.84279617564938 | 42.065310686 | +0.122830 |

SuperGLUE mean 69.803507, population std 0.028196, range 0.064828. Minimum projected Overall(AoA0) 42.058108, +0.115626 over the protected submitted `chck_82M` (Overall 41.942481167385985). All three runs exceed the chck_82M SuperGLUE 69.7661813713118.

The endpoint margin is therefore robust to the most variable expensive component (SuperGLUE finetuning) across two independent evaluations and three seeds/runs. The rest of the +0.116 margin is the cheap7 redistribution: cheap7 44.10642857142857 vs chck_82M 43.95944987645173 (+0.146979), carried by Supplement +0.712 and GlobalPIQA +0.487 against EWoK -0.145 and COMPS -0.201.

## Endpoint-carrier readiness

- Public HF revision `leslie721007/babylm-strict-small-coherent86` @ `ad128352c154702704e8b29c24cf94d986fe5c7f`, trusted-code logits exactly match the source checkpoint, native fallback correctly flagged as the wrong 34.47M-parameter function, current Strict-Small validator PASS, standalone current-Space helper PASS (`data/coherent86_hf_public_bundle/public_helper_validation_standalone.json`).
- Truthful carrier SHA `4a0278a689ea48bdb88215090e34a94caa3fc8ba10df78d19a9195d3698b533e`, AoA scalar 0.0, no borrowed 82M fast history.
- No leaderboard submission was performed in this work; the public revision and validated carrier were available.

## Mechanism status — unchanged and unresolved

Robust endpoint arithmetic does NOT establish a general slow-fast learning principle. Item and overlap evidence remain redistributive:

- coherent vs chck_82M: net -115 discrete items (3116 gain / 3231 loss over 170722 rows); EWoK net -2, high-op Entity net -1.
- coherent vs shuffled86: net -1113 items; low shared-flip overlap (all-discrete gain Jaccard 0.0875).

## Pending Decisive Control

chck84 late item dynamics synthesis launched an exposure-matched no-training ordinary scale1.75 `chck_86M` (86,006,729 words, matched to coherent total 86,005,295) cheap7 evaluation. This is the correct control: if ordinary continued scale1.75 training to 86M reaches a similar score with a similar redistribution profile, then coherent86's private fast path is not adding a distinct capability beyond simply training the backbone 4M words further. Coherent86 as an endpoint stands regardless, but the mechanism claim requires coherent86 to beat exposure-matched chck_86M with seed-stable added decisions and without renewed anchor erosion. Await chck_86M item-transition comparison before any Arm 4 retention machinery or new training.
