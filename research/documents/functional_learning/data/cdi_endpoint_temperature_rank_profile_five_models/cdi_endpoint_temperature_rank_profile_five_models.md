# earlier analysis CDI endpoint temperature/rank profile including dense-mask/sparse-label

Dense-mask/sparse-label is compared to coherent86, sparse, and dense endpoints on CDI masked-token NLL and ranks. This tests preservation cost at the endpoint only; measured AoA remains the official trajectory score.

- `coherent86`: mean NLL T1=8.337592463947226, Tfit=8.132485504525064, mean rank=1686.4427434170238, ΔNLL T1=0.0, ΔNLL Tfit=0.0, Δrank=0.0, rank improved fraction=0.0
- `sparse_focus_seed62064`: mean NLL T1=8.376263109648878, Tfit=8.176610347202626, mean rank=1758.855480710349, ΔNLL T1=0.03867064570165162, ΔNLL Tfit=0.04412484267756197, Δrank=72.41273729332516, rank improved fraction=0.1879975505205144
- `dense_focus_seed62064`: mean NLL T1=8.48921271096242, Tfit=8.261161781227742, mean rank=1796.36252296387, ΔNLL T1=0.15162024701519458, ΔNLL Tfit=0.12867627670267778, Δrank=109.9197795468463, rank improved fraction=0.2927127985303123
- `dense_focus_seed62065`: mean NLL T1=8.493278867939791, Tfit=8.26458585750858, mean rank=1798.5137783221066, ΔNLL T1=0.15568640399256586, ΔNLL Tfit=0.1321003529835157, Δrank=112.07103490508267, rank improved fraction=0.2951622780159216
- `densemask_sparselabel_seed62064`: mean NLL T1=8.495450051397595, Tfit=8.265398987781643, mean rank=1797.4617268830373, ΔNLL T1=0.15785758745037043, ΔNLL Tfit=0.13291348325657792, Δrank=111.01898346601347, rank improved fraction=0.29332516840171463
