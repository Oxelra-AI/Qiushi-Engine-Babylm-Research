# frozen anchor coherent replay item reading fast-path flip overlap

Status: **COMPLETE**

## Main overlap readouts

| subset | n | coh net vs anchor | shuf net vs anchor | coh-vs-shuf net | gain Jaccard | loss Jaccard | gain phi | loss phi | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_discrete | 170722 | -115 | 998 | -1113 | 0.08747643732327992 | 0.08835366231193115 | 0.13712507650191108 | 0.1545160025094337 | 1.5746941302591727e-10 |
| EWoK_fragile_tagged | 5408 | 6 | 58 | -52 | 0.06581059390048154 | 0.10091743119266056 | 0.0829461108815809 | 0.1739962171589189 | 0.11927280890260172 |
| Entity_highop_tagged | 2748 | 1 | -45 | 46 | 0.09375 | 0.1037037037037037 | 0.1718857058613308 | 0.1879680683997553 | 0.0017403727905944194 |
| COMPS_wugs | 41688 | -99 | 97 | -196 | 0.0954143532798684 | 0.1017159396320033 | 0.1416619224705893 | 0.14979469844107657 | 0.037031251919346475 |
| COMPS_base | 49340 | -38 | 565 | -603 | 0.0764832022873481 | 0.08070987654320988 | 0.10106080267339448 | 0.11876416213415168 | 6.537468515246673e-08 |

## By column

| column | n | coh net | shuf net | coh-vs-shuf net | shared gains / union | shared losses / union |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 30 | 451 | -421 | 386/3863 | 264/3504 |
| Supplement | 5218 | -14 | -78 | 64 | 14/177 | 28/255 |
| EWoK | 7618 | -2 | 32 | -34 | 57/799 | 75/751 |
| Entity | 6780 | 7 | -73 | 80 | 28/263 | 44/313 |
| COMPS | 91028 | -137 | 662 | -799 | 999/11858 | 1015/11317 |
| GlobalPIQA | 203 | 1 | 4 | -3 | 1/16 | 1/11 |

## Scientific reading

Low shared-gain overlap and negative/near-flat fragile-family nets mean coherent replay and shuffled86 are not two confirmations of the same added decisions; they are different reweightings of the anchor decision surface.

JSON: `experiments/archive/frontier_consolidation/data/fastpath_flip_overlap/fastpath_flip_overlap.json`
