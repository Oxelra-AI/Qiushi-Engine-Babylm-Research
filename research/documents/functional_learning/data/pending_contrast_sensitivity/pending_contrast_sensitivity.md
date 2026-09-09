# paired seed measurement closeout status pending paired-contrast sensitivity

Created: `2026-09-08T09:31:57Z`

Admission source: `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_boolq/strict_split_eval_admission.json`

## MS62065

Known non-SuperGLUE sum: `310.675`; known SG six-task sum: `414.3097285031356`; missing: `['multirc']`.

If MultiRC=m, Overall=( `310.675` + (`414.3097285031356` + m)/7 )/9.

MultiRC threshold for MS65 to equal MS64 Overall `42.20253795433653`: `69.7251626200657`.

MultiRC threshold for clean65-MS65 to equal the seed64 residual `0.04387437787291759`: `68.80024704832545`.

## O62065

Known non-BLiMP/COMPS/SuperGLUE sum: `189.25`; known SG six-task sum: `416.4723546961086`; missing: `['BLiMP', 'COMPS', 'SuperGLUE']`.

If BLiMP=b, COMPS=c, MultiRC=m, Overall=( `189.25` + b + c + (`416.4723546961086` + m)/7 )/9.

For equality with O64 Overall `42.09260586321611`, b+c+m/7 must equal `130.08740209807235`.
