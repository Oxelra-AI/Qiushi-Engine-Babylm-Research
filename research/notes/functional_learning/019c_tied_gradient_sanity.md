# Step019c tied-gradient sanity check

Sampled query-first bound batch: seed=100, epoch=401, n=128.
Held entity token rows: [13, 14]; held input count=0, held target count=0.

## Main implementation facts

- Tied held-row gradient L2: [1.5e-07, 1e-08]
- Manual non-target softmax gradient matches tied autograd: max |diff|=1.066e-14.
- Untied held input gradient L2: [0.0, 0.0]
- Untied held output gradient L2: [1.5e-07, 1e-08]; matches tied held-row gradient with max |diff|=0.000e+00.

Thus, in this implementation, a held entity absent from the continuation input and target stream is still trained as a negative output class when embeddings are tied. Untying routes that same gradient to `out.weight` and leaves the held input embedding with zero data gradient.

## One AdamW step

- wd=0 tied held input displacement: [0.00130919, 0.00023161]; untied held input displacement: [0.0, 0.0].
- wd=0.01 tied held input displacement: [0.00131519, 0.00023724]; untied held input displacement: [2.409e-05, 2.321e-05].
- Expected untied decay-only displacement lr*wd*||row||: [2.429e-05, 2.335e-05].

This does not by itself show that row drift causes held-transfer loss; Step019b must establish that by row restoration/freezing and hybrid evaluations. It does verify the concrete gradient path that makes the hypothesis possible.

Full JSON: `experiments/archive/functional_learning/data/revision_019c_tied_gradient_sanity/results.json`
