# earlier analysis dense-focus seed replication synthesis
## Frontier coordinate
- Coherent86/v4 official projected Overall(AoA0): `42.1210247099666`.
- At the time of this note, seed62064 official-compatible evaluation was pending; seed62065 official evaluation had started after replication. Neither pending evaluation supplied the fast-screen results below.
## Fast-screen replication
- seed62064: cheap7 `44.80857142857143` (delta `0.24428571428572354`), Entity `28.44` (delta `0.6600000000000001`), GlobalPIQA `40.05` (delta `1.4849999999999994`), BLiMP delta `-0.5499999999999972`, Supplement delta `-0.4000000000000057`.
- seed62065: cheap7 `44.777142857142856` (delta `0.2128571428571462`), Entity `28.43` (delta `0.6499999999999986`), GlobalPIQA `40.05` (delta `1.4849999999999994`), BLiMP delta `-0.5700000000000074`, Supplement delta `-0.4000000000000057`.
- Seed62065 minus seed62064 on identical fast items: cheap7 `-0.03203861948693998`, BLiMP `-0.022388059701498264`, EWoK `-0.18181818181817988`, Entity `-0.001803063085997536`, COMPS `-0.008261031802874186`, GlobalPIQA `0.0`, Reading `-0.00999999999999801`.
## Trained-Qwen view and source-help
- seed62064: view-only ΔNLL `0.07783330366192745`, with-source ΔNLL `-0.02879843047071244`, source-help Δ `0.10663173413263989` over `4783` targets / `1200` pairs.
- seed62065: view-only ΔNLL `0.0806818830034779`, with-source ΔNLL `-0.029691693394520064`, source-help Δ `0.11037357639799796` over `4783` targets / `1200` pairs.
## Controlled original/altered-source common-target movement
- seed62064: mean delta `0.20787532545847906`, no-source `0.020240915786652325`, source-original `0.26365687021763906`, source-altered `0.33972819037114577`, held-source `0.3662232840563067`, trained-content `0.13335863905950132`, both source conditions correct `21/25`.
- seed62065: mean delta `0.20564517308957875`, no-source `0.02088397023223698`, source-original `0.2572671786110316`, source-altered `0.33878437042546766`, held-source `0.36692036312888376`, trained-content `0.12975096601225874`, both source conditions correct `21/25`.
## Mechanism controls prepared, not substituted for official validation
- Alignment×clue script now records copied/noncopied target classes; dry run sampled `160` base targets with copy counts `{'copied': 86, 'not_copied': 74}` and length-matched target-absent shuffled sources.
- Dense-mask/sparse-label training wrapper dry run: first macro labelled `184` groups / `247` target tokens while masking `1217` groups / `1598` tokens; label-to-mask token ratio `0.15456821026282855`. This separates dense input masking from dense supervised coverage for a future causal arm.
## Interpretation
Seed62065 reproduces seed62064 closely enough that dense unchanged-Qwen focus remains the leading practical candidate and merits official-compatible seed62065 evaluation. The common-target and Qwen view probes now look seed-stable, but they do not by themselves prove a general learning principle or a lawful v5. The official-compatible endpoint can still fail if full BLiMP/Supplement/EWoK/SuperGLUE costs erase the fast gains. Mechanism work should preserve the stronger original/altered-source evidence and use shuffled-source alignment only as a complementary trained-material readout; if the official signal survives, the dense-mask/sparse-label arm is the focused training contrast for input-side masking versus added targets.
