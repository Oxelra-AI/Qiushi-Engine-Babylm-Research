# Parent-82M Score and Reproduction Boundaries

Status: repeated endpoint evaluation and standalone-load validation completed; from-corpus reproduction was still pending in this particular record.

The scale-1.75 checkpoint had actual cumulative exposure 82,012,495 words and measured Overall 41.942481167385985, cheap7 43.95944987645173, SuperGLUE 69.7661813713118 and AoA 0.0. The earlier repeated score was 41.94225421386936: the reported difference was +0.000227, with maximum column difference 0.0059.

A standalone model check found the expected adapter-augmented DeBERTa-v2 class, fast tokenizer, vocabulary 16,384, 35,463,008 parameters, adapter scale 1.75 and bottleneck 128, with finite masked-LM outputs. This establishes loadability of that preserved model, not training reproduction.

Against the contemporaneous 41.80 reference, reported column deltas were Supplement +6.9278, Reading +2.7287, BLiMP +1.2913, EWoK -6.0145, GlobalPIQA -2.0923, COMPS -1.3788, Entity -0.1360 and SuperGLUE -0.0238. The score advantage was therefore not a uniform competence improvement.

The reproduction criterion was exact checkpoint identity; if the from-corpus checkpoint differed, it would require direct evaluation rather than inheriting the old score. Selecting an exposure point from an existing trajectory also did not by itself establish a general data-efficient learning principle. Context-conditioned alternative-binding deficits remained scientifically distinct from endpoint score and loadability.
