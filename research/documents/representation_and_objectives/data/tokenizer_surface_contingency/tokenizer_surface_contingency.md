# corrected endpoint interpretation — exact tokenizer surface contingency

CPU-only tokenizer comparison; no model inference. Official evaluation text is used only to interpret later corrected-tokenizer score movement, not to choose data or vocabulary.

compliant tokenizer SHA: `4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738`. Shared token strings with inherited tokenizer: 13795/16384 = 0.841980.

## Allowed reinvest 10M pool

| group | texts | words | new/old tokens | old-only % | new-only % | mean token Δ | top old-only | top new-only |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ALL | 64740 | 10000000 | 0.9919 | 0.80 | 1.47 | -1.857 | `Ġsubsequ`:373, `atter`:363, `/B`:349, `inner`:317, `ĠEur`:287 | `.]`:5696, `Ġ?`:1231, `]?`:1131, `]!`:681, `'?`:580 |
| reinvest_changed_block | 3005 | 423511 | 0.9816 | 1.20 | 2.85 | -3.793 | `iency`:75, `Ġeffic`:71, `Ġtherm`:43, `rient`:43, `ands`:41 | `Ġresearchers`:206, `Ġprotein`:118, `ĠResearchers`:103, `Ġdiabetes`:93, `Ġantib`:79 |
| inherited_qwen_pairs | 12236 | 1656800 | 0.9851 | 0.93 | 2.18 | -2.759 | `Ġsubsequ`:298, `owned`:124, `ĠEur`:78, `ounted`:73, `Ġawait`:69 | `ĠConsequently`:236, `Ġsubsequently`:226, `Ġfeaturing`:176, `.]`:164, `Ġencountered`:156 |
| shared_filler::childes | 16093 | 2574880 | 0.9937 | 0.36 | 0.91 | -1.685 | `/B`:339, `/T`:157, `aron`:142, `alks`:141, `agnet`:133 | `.]`:5369, `]?`:1109, `]!`:676, `'?`:478, `?]`:356 |
| shared_filler::gutenberg | 11621 | 1859360 | 0.9921 | 0.90 | 1.49 | -1.698 | `atter`:132, `BER`:128, `ĠGlad`:127, `inner`:123, `acon`:115 | `ĠAldred`:356, `ĠZara`:319, `ĠJeffreys`:307, `ĠScrooge`:263, `ĠWentworth`:229 |
| shared_filler::open_subtitles | 11434 | 1829440 | 0.9963 | 0.94 | 1.27 | -0.857 | `GER`:149, `UM`:146, `TING`:120, `EF`:120, `EE`:119 | `Ġ?`:795, `"?`:246, `AI`:236, `ÃŃt`:199, `EP`:195 |
| shared_filler::simple_wiki | 6619 | 1059040 | 0.9919 | 1.35 | 1.93 | -2.091 | `ĠEur`:136, `ĠDoub`:116, `ĠEc`:101, `iy`:97, `osl`:90 | `ĠGers`:290, `ĠRodger`:144, `ĠVaud`:140, `ĠSyrian`:119, `ĠHemings`:117 |
| shared_filler::bnc_spoken | 3599 | 575840 | 0.9937 | 0.63 | 1.15 | -1.266 | `cillor`:81, `ĠCoun`:49, `ĠBuck`:46, `Ġquid`:34, `Ġinnit`:33 | `Ġ?`:387, `ĠOxfordshire`:120, `Ġgreenbelt`:110, `ĠBanbury`:93, `Ġcriteria`:85 |
| shared_filler::switchboard | 132 | 21120 | 0.9993 | 0.36 | 0.36 | -0.159 | `-uh`:19, `Ġcans`:7, `TS`:5, `Ġsavings`:3, `Ġcorrupt`:3 | `ĠSalvador`:5, `Ġhumid`:4, `Ġkarate`:4, `eez`:3, `Ġtransportation`:3 |

## Current official evaluation text surface

| group | texts | words | new/old tokens | old-only % | new-only % | mean token Δ | top old-only | top new-only |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ALL | 512074 | 0 | 1.0000 | 1.07 | 1.06 | 0.001 | `Ġdisk`:14122, `ĠJesse`:10864, `ĠYan`:4389, `NN`:4024, `Ġatoms`:2020 | `Ġterrorist`:2204, `Ġinteract`:2200, `ĠAbu`:1922, `ĠAlban`:1667, `ĠSaudi`:1581 |
| family::BLiMP | 119750 | 0 | 1.0043 | 2.29 | 1.93 | 0.044 | `ĠWhose`:841, `Ġdistur`:695, `Ġannoying`:629, `Ġskate`:628, `Ġdisl`:602 | `gra`:1490, `Ġconceal`:1198, `Ġsenator`:1016, `Ġlibraries`:872, `Ġteenager`:826 |
| family::Supplement | 10436 | 0 | 1.0025 | 1.56 | 1.28 | 0.042 | `?Ċ`:946, `Ġannoying`:72, `Ġrepaired`:72, `adu`:70, `Ġbracelet`:66 | `Ġsib`:120, `uate`:70, `Ġmanagers`:68, `Ġbrace`:66, `Ġengineers`:60 |
| family::EWoK | 30472 | 0 | 1.0322 | 3.92 | 0.92 | 0.293 | `ĠJesse`:5398, `ĠYan`:2172, `Ġhates`:690, `ordin`:420, `Ġscrewdriver`:404 | `Ġinteract`:960, `dri`:404, `Ġri`:150, `Ġwrink`:120, `Ġsyrup`:104 |
| family::EWoK_concat | 15236 | 0 | 1.0322 | 3.92 | 0.92 | 0.586 | `ĠJesse`:5398, `ĠYan`:2172, `Ġhates`:690, `ordin`:420, `Ġscrewdriver`:404 | `Ġinteract`:960, `dri`:404, `Ġri`:150, `Ġwrink`:120, `Ġsyrup`:104 |
| family::Entity | 56898 | 0 | 1.0015 | 0.15 | 0.00 | 0.247 | `Ġdisk`:14076 |  |
| family::COMPS | 182056 | 0 | 1.0107 | 1.69 | 0.80 | 0.155 | `Ġhippo`:1516, `zard`:1516, `bil`:1414, `Ġcrocodile`:1339, `Ġdolphin`:1177 | `Ġliz`:1516, `Ġbeetle`:1284, `Ġrhin`:1215, `phin`:1177, `roach`:1062 |
| family::GlobalPIQA_parallel | 515 | 0 | 1.0071 | 1.03 | 0.45 | 0.212 | `Ġcupboard`:19, `pack`:15, `Ġheavier`:10, `Ġbaking`:10, `Ġmeasuring`:10 | `Ġmaint`:15, `enance`:15, `Ġsib`:11, `Ġspinach`:7, `Ġshave`:5 |
| family::GlobalPIQA_nonparallel | 300 | 0 | 1.0103 | 2.22 | 1.02 | 0.253 | `ĠPlace`:18, `Ġbaking`:14, `inkle`:8, `Ġvanilla`:6, `Ġbake`:6 | `Ġsprink`:9, `Ġrazor`:8, `Ġyogurt`:7, `Ġcherries`:7, `verage`:5 |
| family::Reading_sentence | 1726 | 0 | 1.0059 | 0.75 | 0.20 | 0.068 | `Ġcans`:13, `Ġfrag`:11, `Ġparked`:11, `Ġrubbed`:10, `ĠMick`:10 | `Ġfragr`:11, `Ġgrub`:9, `ucked`:7, `Ġdismay`:7, `Ġshave`:6 |
| family::Reading_word | 1726 | 0 | 1.0035 | 0.55 | 0.25 | 0.004 | `Ġcans`:1, `Ġtuck`:1, `Ġbowls`:1, `ubby`:1, `Ġjeans`:1 | `ucked`:1, `Ġgrub`:1, `Ġdismay`:1, `Ġfragr`:1, `Ġshave`:1 |
| family::SuperGLUE | 92959 | 0 | 0.9959 | 1.30 | 1.66 | -0.705 | `NN`:4024, `Ġatoms`:2020, `instein`:1847, `Ġcontroll`:1786, `pit`:1312 | `Ġterrorist`:2204, `ĠAbu`:1922, `ĠAlban`:1667, `ĠSaudi`:1581, `Ġagencies`:1428 |

## EWoK domain surface

| domain | texts | new/old tokens | old-only % | new-only % | mean token Δ |
|---|---:|---:|---:|---:|---:|
| agent-properties | 13260 | 1.0287 | 3.98 | 1.02 | 0.443 |
| material-dynamics | 4620 | 1.0163 | 3.46 | 2.64 | 0.124 |
| material-properties | 1020 | 1.0285 | 2.52 | 2.20 | 0.306 |
| physical-dynamics | 720 | 1.0257 | 2.57 | 0.00 | 0.250 |
| physical-interactions | 3336 | 1.0347 | 2.82 | 1.63 | 0.441 |
| physical-relations | 4908 | 1.0036 | 0.36 | 0.00 | 0.044 |
| quantitative-properties | 1884 | 1.0273 | 3.45 | 1.56 | 0.340 |
| social-interactions | 1764 | 1.0426 | 5.45 | 0.94 | 0.444 |
| social-properties | 1968 | 1.0394 | 4.58 | 0.83 | 0.372 |
| social-relations | 9288 | 1.0621 | 7.40 | 0.18 | 0.607 |
| spatial-relations | 2940 | 1.0338 | 2.65 | 0.70 | 0.495 |

## Word-level fragmentation examples

### ALL

Changed word types 6621/83742; weighted extra pieces when tokenizer is longer: 219880; weighted saved pieces when shorter: 270530.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| disk | 14120 | 41 | 2 | 3 | `['Ġ', 'Ġdisk']` | `['Ġ', 'Ġdis', 'k']` |
| octopus | 993 | 36 | 2 | 5 | `['Ġ', 'Ġoctopus']` | `['Ġ', 'Ġo', 'ct', 'op', 'us']` |
| crocodile | 1341 | 30 | 2 | 4 | `['Ġ', 'Ġcrocodile']` | `['Ġ', 'Ġcro', 'cod', 'ile']` |
| david | 2106 | 1258 | 3 | 4 | `['Ġ', 'Ġd', 'avid']` | `['Ġ', 'Ġd', 'av', 'id']` |
| screwdriver | 1034 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |
| atoms | 2034 | 56 | 2 | 3 | `['Ġ', 'Ġatoms']` | `['Ġ', 'Ġat', 'oms']` |
| muslim | 1880 | 212 | 3 | 4 | `['Ġ', 'Ġmus', 'lim']` | `['Ġ', 'Ġmus', 'l', 'im']` |
| einstein | 1719 | 11 | 3 | 4 | `['Ġ', 'Ġe', 'instein']` | `['Ġ', 'Ġe', 'in', 'stein']` |
| aliens | 848 | 31 | 2 | 4 | `['Ġ', 'Ġaliens']` | `['Ġ', 'Ġal', 'i', 'ens']` |
| hates | 1545 | 59 | 2 | 3 | `['Ġ', 'Ġhates']` | `['Ġ', 'Ġhat', 'es']` |

### family::EWoK

Changed word types 54/866; weighted extra pieces when tokenizer is longer: 3316; weighted saved pieces when shorter: 2020.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| screwdriver | 404 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |
| hates | 690 | 59 | 2 | 3 | `['Ġ', 'Ġhates']` | `['Ġ', 'Ġhat', 'es']` |
| chess | 328 | 58 | 2 | 3 | `['Ġ', 'Ġchess']` | `['Ġ', 'Ġc', 'hess']` |
| landlord | 290 | 44 | 2 | 3 | `['Ġ', 'Ġlandlord']` | `['Ġ', 'Ġlandl', 'ord']` |
| toothpaste | 108 | 59 | 2 | 4 | `['Ġ', 'Ġtoothpaste']` | `['Ġ', 'Ġtooth', 'p', 'aste']` |
| antennas | 152 | 7 | 4 | 5 | `['Ġ', 'Ġanten', 'n', 'as']` | `['Ġ', 'Ġan', 'ten', 'n', 'as']` |
| heating | 120 | 53 | 2 | 3 | `['Ġ', 'Ġheating']` | `['Ġ', 'Ġhe', 'ating']` |
| bouncy | 56 | 15 | 2 | 4 | `['Ġ', 'Ġbouncy']` | `['Ġ', 'Ġb', 'ounc', 'y']` |
| decides | 100 | 51 | 2 | 3 | `['Ġ', 'Ġdecides']` | `['Ġ', 'Ġdec', 'ides']` |
| employee | 90 | 64 | 2 | 3 | `['Ġ', 'Ġemployee']` | `['Ġ', 'Ġemploy', 'ee']` |

### family::Supplement

Changed word types 129/1802; weighted extra pieces when tokenizer is longer: 2048; weighted saved pieces when shorter: 2513.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| david | 706 | 1258 | 3 | 4 | `['Ġ', 'Ġd', 'avid']` | `['Ġ', 'Ġd', 'av', 'id']` |
| repaired | 72 | 55 | 2 | 3 | `['Ġ', 'Ġrepaired']` | `['Ġ', 'Ġrep', 'aired']` |
| annoying | 72 | 65 | 2 | 3 | `['Ġ', 'Ġannoying']` | `['Ġ', 'Ġannoy', 'ing']` |
| bracelet | 66 | 62 | 2 | 3 | `['Ġ', 'Ġbracelet']` | `['Ġ', 'Ġbrace', 'let']` |
| cheat | 50 | 47 | 2 | 3 | `['Ġ', 'Ġcheat']` | `['Ġ', 'Ġche', 'at']` |
| employee | 50 | 64 | 2 | 3 | `['Ġ', 'Ġemployee']` | `['Ġ', 'Ġemploy', 'ee']` |
| coke | 50 | 85 | 2 | 3 | `['Ġ', 'Ġcoke']` | `['Ġ', 'Ġco', 'ke']` |
| gardener | 48 | 53 | 2 | 3 | `['Ġ', 'Ġgardener']` | `['Ġ', 'Ġgard', 'ener']` |
| aeroplanes | 22 | 5 | 3 | 5 | `['Ġ', 'Ġaeropl', 'anes']` | `['Ġ', 'Ġaer', 'op', 'l', 'anes']` |
| restaurants | 42 | 70 | 2 | 3 | `['Ġ', 'Ġrestaurants']` | `['Ġ', 'Ġrestaur', 'ants']` |

### family::GlobalPIQA_parallel

Changed word types 22/780; weighted extra pieces when tokenizer is longer: 129; weighted saved pieces when shorter: 48.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| cupboard | 19 | 62 | 2 | 3 | `['Ġ', 'Ġcupboard']` | `['Ġ', 'Ġcup', 'board']` |
| backpack | 15 | 39 | 3 | 4 | `['Ġ', 'Ġback', 'pack']` | `['Ġ', 'Ġback', 'p', 'ack']` |
| baking | 10 | 64 | 2 | 3 | `['Ġ', 'Ġbaking']` | `['Ġ', 'Ġb', 'aking']` |
| heavier | 10 | 68 | 2 | 3 | `['Ġ', 'Ġheavier']` | `['Ġ', 'Ġheav', 'ier']` |
| measuring | 10 | 71 | 2 | 3 | `['Ġ', 'Ġmeasuring']` | `['Ġ', 'Ġmeas', 'uring']` |
| fluffy | 5 | 29 | 2 | 4 | `['Ġ', 'Ġfluffy']` | `['Ġ', 'Ġfl', 'uff', 'y']` |
| freeze | 9 | 83 | 2 | 3 | `['Ġ', 'Ġfreeze']` | `['Ġ', 'Ġfree', 'ze']` |
| sweeping | 7 | 66 | 2 | 3 | `['Ġ', 'Ġsweeping']` | `['Ġ', 'Ġswe', 'eping']` |
| crispy | 5 | 2 | 3 | 4 | `['Ġ', 'Ġcris', 'py']` | `['Ġ', 'Ġcris', 'p', 'y']` |
| windy | 5 | 31 | 2 | 3 | `['Ġ', 'Ġwindy']` | `['Ġ', 'Ġwind', 'y']` |

### family::GlobalPIQA_nonparallel

Changed word types 53/840; weighted extra pieces when tokenizer is longer: 129; weighted saved pieces when shorter: 82.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| baking | 14 | 64 | 2 | 3 | `['Ġ', 'Ġbaking']` | `['Ġ', 'Ġb', 'aking']` |
| waffle | 4 | 32 | 2 | 4 | `['Ġ', 'Ġwaffle']` | `['Ġ', 'Ġwa', 'ff', 'le']` |
| vanilla | 6 | 53 | 2 | 3 | `['Ġ', 'Ġvanilla']` | `['Ġ', 'Ġvan', 'illa']` |
| bake | 6 | 64 | 2 | 3 | `['Ġ', 'Ġbake']` | `['Ġ', 'Ġb', 'ake']` |
| shampoo | 3 | 52 | 2 | 4 | `['Ġ', 'Ġshampoo']` | `['Ġ', 'Ġsha', 'mp', 'oo']` |
| skateboard | 5 | 5 | 3 | 4 | `['Ġ', 'Ġskate', 'board']` | `['Ġ', 'Ġsk', 'ate', 'board']` |
| icing | 5 | 12 | 2 | 3 | `['Ġ', 'Ġicing']` | `['Ġ', 'Ġ', 'icing']` |
| zipper | 5 | 30 | 2 | 3 | `['Ġ', 'Ġzipper']` | `['Ġ', 'Ġz', 'ipper']` |
| jeans | 5 | 41 | 2 | 3 | `['Ġ', 'Ġjeans']` | `['Ġ', 'Ġje', 'ans']` |
| crispy | 4 | 2 | 3 | 4 | `['Ġ', 'Ġcris', 'py']` | `['Ġ', 'Ġcris', 'p', 'y']` |

### family::COMPS

Changed word types 365/3312; weighted extra pieces when tokenizer is longer: 44605; weighted saved pieces when shorter: 16398.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| crocodile | 1339 | 30 | 2 | 4 | `['Ġ', 'Ġcrocodile']` | `['Ġ', 'Ġcro', 'cod', 'ile']` |
| octopus | 825 | 36 | 2 | 5 | `['Ġ', 'Ġoctopus']` | `['Ġ', 'Ġo', 'ct', 'op', 'us']` |
| hippo | 1516 | 57 | 2 | 3 | `['Ġ', 'Ġhippo']` | `['Ġ', 'Ġhipp', 'o']` |
| hedgehog | 720 | 26 | 3 | 5 | `['Ġ', 'Ġhedge', 'hog']` | `['Ġ', 'Ġhe', 'dge', 'h', 'og']` |
| gerbil | 1414 | 0 | 3 | 4 | `['Ġ', 'Ġger', 'bil']` | `['Ġ', 'Ġger', 'b', 'il']` |
| dolphin | 1177 | 42 | 2 | 3 | `['Ġ', 'Ġdolphin']` | `['Ġ', 'Ġdol', 'phin']` |
| tortoise | 1168 | 18 | 3 | 4 | `['Ġ', 'Ġtorto', 'ise']` | `['Ġ', 'Ġtor', 'to', 'ise']` |
| guinea | 1099 | 51 | 3 | 4 | `['Ġ', 'Ġgu', 'inea']` | `['Ġ', 'Ġgu', 'in', 'ea']` |
| moose | 1090 | 22 | 2 | 3 | `['Ġ', 'Ġmoose']` | `['Ġ', 'Ġm', 'oose']` |
| calf | 1002 | 64 | 2 | 3 | `['Ġ', 'Ġcalf']` | `['Ġ', 'Ġcal', 'f']` |

### family::SuperGLUE

Changed word types 6478/83035; weighted extra pieces when tokenizer is longer: 134169; weighted saved pieces when shorter: 231008.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| atoms | 2034 | 56 | 2 | 3 | `['Ġ', 'Ġatoms']` | `['Ġ', 'Ġat', 'oms']` |
| muslim | 1880 | 212 | 3 | 4 | `['Ġ', 'Ġmus', 'lim']` | `['Ġ', 'Ġmus', 'l', 'im']` |
| einstein | 1719 | 11 | 3 | 4 | `['Ġ', 'Ġe', 'instein']` | `['Ġ', 'Ġe', 'in', 'stein']` |
| aliens | 848 | 31 | 2 | 4 | `['Ġ', 'Ġaliens']` | `['Ġ', 'Ġal', 'i', 'ens']` |
| louis | 1408 | 406 | 3 | 4 | `['Ġ', 'Ġlou', 'is']` | `['Ġ', 'Ġl', 'ou', 'is']` |
| cockpit | 1301 | 7 | 3 | 4 | `['Ġ', 'Ġcock', 'pit']` | `['Ġ', 'Ġcock', 'p', 'it']` |
| oatmeal | 410 | 22 | 2 | 5 | `['Ġ', 'Ġoatmeal']` | `['Ġ', 'Ġo', 'at', 'me', 'al']` |
| sunday | 1190 | 962 | 3 | 4 | `['Ġ', 'Ġsund', 'ay']` | `['Ġ', 'Ġsu', 'nd', 'ay']` |
| sultan | 1147 | 46 | 3 | 4 | `['Ġ', 'Ġs', 'ultan']` | `['Ġ', 'Ġs', 'ult', 'an']` |
| controller | 1102 | 34 | 3 | 4 | `['Ġ', 'Ġcontroll', 'er']` | `['Ġ', 'Ġcontro', 'll', 'er']` |

### ewok_domain::spatial-relations

Changed word types 1/87; weighted extra pieces when tokenizer is longer: 624; weighted saved pieces when shorter: 0.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| screwdriver | 312 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |

### ewok_domain::material-dynamics

Changed word types 8/60; weighted extra pieces when tokenizer is longer: 136; weighted saved pieces when shorter: 644.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| coke | 76 | 85 | 2 | 3 | `['Ġ', 'Ġcoke']` | `['Ġ', 'Ġco', 'ke']` |
| taps | 60 | 99 | 2 | 3 | `['Ġ', 'Ġtaps']` | `['Ġ', 'Ġtap', 's']` |

### ewok_domain::physical-dynamics

Changed word types 4/69; weighted extra pieces when tokenizer is longer: 180; weighted saved pieces when shorter: 0.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| sinking | 80 | 70 | 2 | 3 | `['Ġ', 'Ġsinking']` | `['Ġ', 'Ġsink', 'ing']` |
| sliding | 60 | 64 | 2 | 3 | `['Ġ', 'Ġsliding']` | `['Ġ', 'Ġsl', 'iding']` |
| underwater | 20 | 41 | 2 | 3 | `['Ġ', 'Ġunderwater']` | `['Ġ', 'Ġunder', 'water']` |
| slippery | 20 | 49 | 2 | 3 | `['Ġ', 'Ġslippery']` | `['Ġ', 'Ġslipp', 'ery']` |

### ewok_domain::physical-interactions

Changed word types 8/155; weighted extra pieces when tokenizer is longer: 1272; weighted saved pieces when shorter: 220.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| screwdriver | 496 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |
| heating | 240 | 53 | 2 | 3 | `['Ġ', 'Ġheating']` | `['Ġ', 'Ġhe', 'ating']` |
| repaired | 20 | 55 | 2 | 3 | `['Ġ', 'Ġrepaired']` | `['Ġ', 'Ġrep', 'aired']` |
| heavier | 20 | 68 | 2 | 3 | `['Ġ', 'Ġheavier']` | `['Ġ', 'Ġheav', 'ier']` |

### ewok_domain::physical-relations

Changed word types 0/90; weighted extra pieces when tokenizer is longer: 0; weighted saved pieces when shorter: 0.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|

### ewok_domain::material-properties

Changed word types 6/109; weighted extra pieces when tokenizer is longer: 320; weighted saved pieces when shorter: 72.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| bouncy | 112 | 15 | 2 | 4 | `['Ġ', 'Ġbouncy']` | `['Ġ', 'Ġb', 'ounc', 'y']` |
| fragile | 60 | 34 | 3 | 4 | `['Ġ', 'Ġfrag', 'ile']` | `['Ġ', 'Ġfra', 'g', 'ile']` |
| transparent | 36 | 28 | 3 | 4 | `['Ġ', 'Ġtrans', 'parent']` | `['Ġ', 'Ġtrans', 'p', 'arent']` |

### ewok_domain::social-relations

Changed word types 6/153; weighted extra pieces when tokenizer is longer: 888; weighted saved pieces when shorter: 196.

| word | eval count | train count | old len | len | old tokens | tokens |
|---|---:|---:|---:|---:|---|---|
| landlord | 580 | 44 | 2 | 3 | `['Ġ', 'Ġlandlord']` | `['Ġ', 'Ġlandl', 'ord']` |
| employee | 180 | 64 | 2 | 3 | `['Ġ', 'Ġemployee']` | `['Ġ', 'Ġemploy', 'ee']` |
| suite | 88 | 68 | 2 | 3 | `['Ġ', 'Ġsuite']` | `['Ġ', 'Ġsu', 'ite']` |
| hates | 40 | 59 | 2 | 3 | `['Ġ', 'Ġhates']` | `['Ġ', 'Ġhat', 'es']` |

## Use after corrected full official scores arrive

Compare actual score movement to this exact tokenizer surface. Broad score losses with small tokenization pressure point to changed MLM target geometry or training dynamics under the legal vocabulary rather than crude truncation. Localized losses in families/domains with large old-only occurrence or concrete-word fragmentation make legal tokenizer-learning research plausible, but any such research must use only allowed 10M text and generic tokenizer objectives.
