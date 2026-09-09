# axis noise floor and clean training state centered-axis correction

This supplement corrects the first axis noise floor and clean training state summary by separating raw dot products from the actual PCA-centered coordinates. scale1p75 item mechanism and cross route tradeoff fitted PCA after subtracting the scale1p75 item mechanism and cross route tradeoff archive mean, so a new vector's comparable coordinate is `(delta - mean) dot PC`, not the raw `delta dot PC`.

## Common10_80 1x seed directions

- Stable6 raw family-vector cosine remains -0.0918; ex-Entity cosine remains -0.4232.
- The stable6 centered scale1p75 item mechanism and cross route tradeoff coordinates have a shared PC2-negative / PC3-positive component, but the ex-Entity centered coordinates are small and sign-unstable. The shared stable6 axis component is therefore mainly the already-known positive Entity coordinate, not a reproduced broad ex-Entity direction.

| label | centered L2 | PC1 centered proj/cos | PC2 centered proj/cos | PC3 centered proj/cos | PC2 top contributions | PC3 top contributions |
|---|---:|---:|---:|---:|---|---|
| seed43022_common10_80_VminusR | 1.5392 | +0.0542/+0.039 | -0.6717/-0.668 | +0.9316/+0.685 | `[["Entity", -0.6345536706002427], ["Reading", -0.05888819580013388], ["BLiMP", 0.03508045152495116], ["EWoK", -0.004829594685120333]]` | `[["Entity", 0.9215786594781871], ["Reading", -0.04314135429911349], ["COMPS", 0.026715352990724323], ["BLiMP", 0.017000195789529257]]` |
| seed43122_common10_80_VminusR | 2.3050 | +0.2378/+0.115 | -0.6266/-0.416 | +1.0706/+0.526 | `[["Entity", -0.8086727685032957], ["BLiMP", 0.20707531028730725], ["Supplement", -0.04864615304367261], ["EWoK", 0.03257904661871262]]` | `[["Entity", 1.1744563154899454], ["EWoK", -0.10619605008439151], ["BLiMP", 0.10034992894997083], ["Supplement", -0.07943131717357925]]` |
| seed43022_common10_80_VminusR_exEntity | 1.0354 | -0.0638/-0.069 | -0.0371/-0.105 | +0.0101/+0.028 | `[["Reading", -0.05888819580013388], ["BLiMP", 0.03508045152495116], ["EWoK", -0.004829594685120333], ["COMPS", -0.004674121881869815]]` | `[["Reading", -0.04314135429911349], ["COMPS", 0.026715352990724323], ["BLiMP", 0.017000195789529257], ["EWoK", 0.015742752851882466]]` |
| seed43122_common10_80_VminusR_exEntity | 1.7906 | +0.0875/+0.055 | +0.1820/+0.298 | -0.1039/-0.164 | `[["BLiMP", 0.20707531028730725], ["Supplement", -0.04864615304367261], ["EWoK", 0.03257904661871262], ["Reading", -0.010844672562754173]]` | `[["EWoK", -0.10619605008439151], ["BLiMP", 0.10034992894997083], ["Supplement", -0.07943131717357925], ["COMPS", -0.010636836306417603]]` |
| dose2p64:V_minus_R:stable6 | 3.1743 | +1.3554/+0.476 | -1.7849/-0.861 | +1.6238/+0.579 | `[["Entity", -1.472414769709732], ["Supplement", -0.3189774449306324], ["BLiMP", 0.0354251306006274], ["Reading", -0.02763428773840942]]` | `[["Entity", 2.138425940206766], ["Supplement", -0.5208386894798565], ["Reading", -0.020244814464559602], ["BLiMP", 0.017167229322917]]` |
| D1_VminusB:chck_80M:stable6 | 4.6668 | +1.6112/+0.385 | -2.2974/-0.754 | +2.1996/+0.534 | `[["Entity", -2.0504901747478685], ["Supplement", -0.45414309087411214], ["BLiMP", 0.2187943988602945], ["Reading", -0.03626556877941526]]` | `[["Entity", 2.9779797581658043], ["Supplement", -0.7415423756329947], ["BLiMP", 0.10602906908515175], ["COMPS", -0.08310008354287304]]` |
| D1_BminusCold:chck_80M:stable6 | 3.4304 | +2.3051/+0.749 | -1.2626/-0.563 | -0.0016/-0.001 | `[["BLiMP", -0.7352772826110532], ["Entity", -0.4068058905430489], ["Supplement", -0.10124988610718988], ["Reading", -0.0406797399054092]]` | `[["Entity", 0.5908146854148064], ["BLiMP", -0.35631975133188776], ["Supplement", -0.16532472382654415], ["EWoK", -0.10803348410123981]]` |

## Scientific correction

- The earlier analysis statement that the intervention norm grows while broad net stays flat is too broad. For V-R dose, all-family radius growth is overwhelmingly the Entity component; ex-Entity radius is nearly dose-invariant.
- Ex-Entity V-R is not absent: across dose it can reorient tangentially, and from 1x to MAX its endpoint-to-endpoint ex-Entity displacement is nontrivial. But with only seed43022 at higher doses and with the two 1x seed directions anti-aligned over common10_80, that movement is currently indistinguishable from same-coordinate basin variation rather than a reproducible trade-off direction.
- scale1p75 item mechanism and cross route tradeoff axes remain useful descriptors. They do not establish a shared intervention manifold here because the stable6 PC2/PC3 similarity is mainly Entity, ex-Entity projections are small/unstable, and rows omitting GlobalPIQA use restricted axes with large missing PC loadings.
- reference and margin state adds an independent check on the Entity V-B surface: official macro V-B is real and positive across zero/nonzero operations, but paired binding flow does not move examples into both-correct retained-state form; it shifts affected-only upward while eroding unaffected-only. That further weakens an Entity-driven argument for launching the permuted companion before broad ex-Entity V-B or content-admission evidence changes.
- The incoming matched-clean scores should therefore be read primarily as a content-placement test, not as a rescue of the broad V-R/conservation-vector idea. The full breadth table should be read family-by-family and ex-Entity before any mechanism-bearing V-B claim.

## Files
- centered_axis_key_rows_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_key_rows.csv`
- centered_axis_family_contributions_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_family_contributions.csv`
- common10_80_seed_sign_agreement_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/common10_80_seed_sign_agreement.csv`
- summary_md: `research/documents/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_correction.md`

No model loading, training, official evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA work, upload, or leaderboard action occurred.
