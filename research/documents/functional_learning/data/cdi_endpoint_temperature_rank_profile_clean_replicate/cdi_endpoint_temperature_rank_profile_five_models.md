# earlier analysis CDI endpoint temperature/rank profile including dense-mask/sparse-label

Dense-mask/sparse-label is compared to coherent86, sparse, and dense endpoints on CDI masked-token NLL and ranks. This tests preservation cost at the endpoint only; measured AoA remains the official trajectory score.

- `coherent86`: mean NLL T1=8.337592463947226, Tfit=8.132485504525064, mean rank=1686.4427434170238, ΔNLL T1=0.0, ΔNLL Tfit=0.0, Δrank=0.0, rank improved fraction=0.0
- `sparse_focus_seed62064`: mean NLL T1=8.376263109648878, Tfit=8.176610347202626, mean rank=1758.855480710349, ΔNLL T1=0.03867064570165162, ΔNLL Tfit=0.04412484267756197, Δrank=72.41273729332516, rank improved fraction=0.1879975505205144
- `densemask_sparselabel_seed62064`: mean NLL T1=8.495450051397595, Tfit=8.265398987781643, mean rank=1797.4617268830373, ΔNLL T1=0.15785758745037043, ΔNLL Tfit=0.13291348325657792, Δrank=111.01898346601347, rank improved fraction=0.29332516840171463
- `clean_pres_lambda1_eval_seed62064`: mean NLL T1=8.41498271680214, Tfit=8.19471985929212, mean rank=1732.4439681567667, ΔNLL T1=0.07739025285491459, ΔNLL Tfit=0.062234354767055064, Δrank=46.0012247397428, rank improved fraction=0.37048377219840783
- `clean_pres_lambda1_eval_seed62065`: mean NLL T1=8.419674155258432, Tfit=8.199311857615909, mean rank=1737.6080832823025, ΔNLL T1=0.08208169131120581, ΔNLL Tfit=0.06682635309084548, Δrank=51.165339865278625, rank improved fraction=0.352112676056338
