# earlier analysis CDI endpoint temperature/rank profile including dense-mask/sparse-label

Dense-mask/sparse-label is compared to coherent86, sparse, and dense endpoints on CDI masked-token NLL and ranks. This tests preservation cost at the endpoint only; measured AoA remains the official trajectory score.

- `coherent86`: mean NLL T1=8.337592471051055, Tfit=8.132485550507761, mean rank=1686.444580526638, ΔNLL T1=0.0, ΔNLL Tfit=0.0, Δrank=0.0, rank improved fraction=0.0
- `sparse_focus_seed62064`: mean NLL T1=8.376263018753962, Tfit=8.17661027637392, mean rank=1758.8560930802205, ΔNLL T1=0.03867054770290578, ΔNLL Tfit=0.04412472586615715, Δrank=72.41151255358237, rank improved fraction=0.1879975505205144
- `densemask_sparselabel_seed62064`: mean NLL T1=8.49544990330647, Tfit=8.265398853936388, mean rank=1797.4605021432947, ΔNLL T1=0.15785743225541535, ΔNLL Tfit=0.1329133034286255, Δrank=111.01592161665646, rank improved fraction=0.29332516840171463
- `dense_focus_seed62064`: mean NLL T1=8.489212738476358, Tfit=8.261161827585367, mean rank=1796.3619105939988, ΔNLL T1=0.15162026742530277, ΔNLL Tfit=0.12867627707760568, Δrank=109.91733006736068, rank improved fraction=0.2927127985303123
- `dense_focus_seed62065`: mean NLL T1=8.49327895176424, Tfit=8.264585949796203, mean rank=1798.514390691978, ΔNLL T1=0.15568648071318583, ΔNLL Tfit=0.13210039928844067, Δrank=112.06981016533986, rank improved fraction=0.2951622780159216
- `clean_pres_lambda1_eval_full80`: mean NLL T1=8.414982714303738, Tfit=8.194719857381727, mean rank=1732.4439681567667, ΔNLL T1=0.07739024325268228, ΔNLL Tfit=0.062234306873966326, Δrank=45.999387630128595, rank improved fraction=0.37048377219840783
- `clean_pres_lambda1_train_full80`: mean NLL T1=8.449793855402453, Tfit=8.222131841356621, mean rank=1752.0018371096141, ΔNLL T1=0.1122013843513966, ΔNLL Tfit=0.0896462908488586, Δrank=65.55725658297612, rank improved fraction=0.352112676056338
- `pres_lambda1_trainmode_confounded`: mean NLL T1=8.446841788255773, Tfit=8.21947902312305, mean rank=1749.1102265768525, ΔNLL T1=0.10924931720471773, ΔNLL Tfit=0.0869934726152877, Δrank=62.66564605021433, rank improved fraction=0.35027556644213104
