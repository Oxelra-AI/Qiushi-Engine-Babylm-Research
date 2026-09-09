# tokenizer learning contingency and gpu priority — tokenizer-learning contingency evidence

CPU-only analysis while the compliant retrain is pending. It does not train or evaluate a model. Official evaluation text is used only to interpret possible score movement, not to choose pretraining or tokenizer vocabulary.

## Vocabulary and occurrence mass

Old and compliant vocabularies each contain 16384 tokens; shared token strings: 13809 (0.843).

The occurrence mass is more informative than raw vocabulary overlap: if old-only token occurrences are tiny on the allowed pool and evaluation surface, a below-leader score should not be attributed to simple missing old-tokenizer surface pieces.

### Allowed reinvest 10M pretraining pool

| group | texts/rows | words field | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token Δ | top old-only tokens | top new-only tokens |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ALL | 64740 | 10000000 | 0.9915 | 0.79 | 1.49 | -1.930 | `Ġsubsequ`:373, `atter`:363, `/B`:349, `inner`:317, `ĠEur`:287 | `.]`:5696, `Ġ?`:1231, `]?`:1131, `]!`:681, `'?`:580 |
| reinvest_changed_block | 3005 | 423511 | 0.9811 | 1.18 | 2.89 | -3.907 | `iency`:75, `Ġeffic`:71, `Ġtherm`:43, `rient`:43, `ands`:41 | `Ġresearchers`:206, `Ġprotein`:118, `ĠResearchers`:103, `Ġdiabetes`:93, `Ġantib`:79 |
| inherited_qwen_pairs | 12236 | 1656800 | 0.9845 | 0.91 | 2.22 | -2.858 | `Ġsubsequ`:298, `owned`:124, `ĠEur`:78, `ounted`:73, `Ġawait`:69 | `ĠConsequently`:236, `Ġsubsequently`:226, `Ġfeaturing`:176, `.]`:164, `Ġencountered`:156 |
| shared_filler::childes | 16093 | 2574880 | 0.9936 | 0.35 | 0.92 | -1.709 | `/B`:339, `/T`:157, `aron`:142, `alks`:141, `agnet`:133 | `.]`:5369, `]?`:1109, `]!`:676, `'?`:478, `?]`:356 |
| shared_filler::gutenberg | 11621 | 1859360 | 0.9917 | 0.89 | 1.52 | -1.788 | `atter`:132, `BER`:128, `ĠGlad`:127, `inner`:123, `acon`:115 | `ĠAldred`:356, `ĠZara`:319, `ĠJeffreys`:307, `ĠScrooge`:263, `ĠWentworth`:229 |
| shared_filler::open_subtitles | 11434 | 1829440 | 0.9960 | 0.93 | 1.28 | -0.930 | `GER`:149, `UM`:146, `TING`:120, `EF`:120, `EE`:119 | `Ġ?`:795, `"?`:246, `AI`:236, `ÃŃt`:199, `EP`:195 |
| shared_filler::simple_wiki | 6619 | 1059040 | 0.9915 | 1.32 | 1.95 | -2.206 | `ĠEur`:136, `ĠDoub`:116, `ĠEc`:101, `iy`:97, `osl`:90 | `ĠGers`:290, `ĠRodger`:144, `ĠVaud`:140, `ĠSyrian`:119, `ĠHemings`:117 |
| shared_filler::bnc_spoken | 3599 | 575840 | 0.9935 | 0.62 | 1.17 | -1.313 | `cillor`:81, `ĠCoun`:49, `ĠBuck`:46, `Ġquid`:34, `Ġinnit`:33 | `Ġ?`:387, `ĠOxfordshire`:120, `Ġgreenbelt`:110, `ĠBanbury`:93, `Ġcriteria`:85 |
| shared_filler::switchboard | 132 | 21120 | 0.9991 | 0.34 | 0.36 | -0.197 | `-uh`:19, `Ġcans`:7, `TS`:5, `Ġsavings`:3, `Ġcorrupt`:3 | `ĠSalvador`:5, `Ġhumid`:4, `Ġkarate`:4, `eez`:3, `Ġtransportation`:3 |

### Official evaluation text surface

| group | texts/rows | words field | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token Δ | top old-only tokens | top new-only tokens |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ALL | 512074 | 0 | 0.9993 | 1.02 | 1.08 | -0.039 | `Ġdisk`:14122, `ĠYan`:4389, `NN`:4024, `Ġatoms`:2020, `ordin`:1932 | `Ġterrorist`:2204, `Ġinteract`:2200, `ĠAbu`:1922, `ĠAlban`:1667, `ĠSaudi`:1581 |
| family::BLiMP | 119750 | 0 | 1.0042 | 2.29 | 1.94 | 0.043 | `ĠWhose`:841, `Ġdistur`:695, `Ġannoying`:629, `Ġskate`:628, `Ġdisl`:602 | `gra`:1490, `Ġconceal`:1198, `Ġsenator`:1016, `Ġlibraries`:872, `Ġteenager`:826 |
| family::Supplement | 10436 | 0 | 1.0016 | 1.56 | 1.34 | 0.026 | `?Ċ`:946, `Ġannoying`:72, `Ġrepaired`:72, `adu`:70, `Ġbracelet`:66 | `Ġsib`:120, `grad`:70, `uate`:70, `Ġmanagers`:68, `Ġbrace`:66 |
| family::EWoK | 30472 | 0 | 1.0128 | 1.98 | 0.94 | 0.116 | `ĠYan`:2172, `Ġhates`:690, `ordin`:420, `Ġscrewdriver`:404, `Ġchess`:328 | `Ġinteract`:960, `dri`:404, `Ġri`:150, `Ġwrink`:120, `Ġsyrup`:104 |
| family::EWoK_concat | 15236 | 0 | 1.0128 | 1.98 | 0.94 | 0.232 | `ĠYan`:2172, `Ġhates`:690, `ordin`:420, `Ġscrewdriver`:404, `Ġchess`:328 | `Ġinteract`:960, `dri`:404, `Ġri`:150, `Ġwrink`:120, `Ġsyrup`:104 |
| family::Entity | 56898 | 0 | 1.0015 | 0.15 | 0.00 | 0.247 | `Ġdisk`:14076 |  |
| family::COMPS | 182056 | 0 | 1.0099 | 1.65 | 0.83 | 0.145 | `Ġhippo`:1516, `zard`:1516, `bil`:1414, `Ġcrocodile`:1339, `Ġdolphin`:1177 | `Ġliz`:1516, `Ġbeetle`:1284, `Ġrhin`:1215, `phin`:1177, `roach`:1062 |
| family::GlobalPIQA_parallel | 515 | 0 | 1.0061 | 1.03 | 0.36 | 0.183 | `Ġcupboard`:19, `pack`:15, `Ġheavier`:10, `Ġbaking`:10, `Ġmeasuring`:10 | `Ġmaintenance`:15, `Ġsib`:11, `Ġspinach`:7, `Ġshave`:5, `Ġscen`:5 |
| family::GlobalPIQA_nonparallel | 300 | 0 | 1.0091 | 2.11 | 1.03 | 0.223 | `ĠPlace`:18, `Ġbaking`:14, `inkle`:8, `Ġvanilla`:6, `Ġbake`:6 | `Ġsprink`:9, `Ġrazor`:8, `Ġyogurt`:7, `Ġcherries`:7, `verage`:5 |
| family::Reading_sentence | 1726 | 0 | 1.0059 | 0.75 | 0.20 | 0.068 | `Ġcans`:13, `Ġfrag`:11, `Ġparked`:11, `Ġrubbed`:10, `ĠMick`:10 | `Ġfragr`:11, `Ġgrub`:9, `ucked`:7, `Ġdismay`:7, `Ġshave`:6 |
| family::Reading_word | 1726 | 0 | 1.0035 | 0.55 | 0.25 | 0.004 | `Ġcans`:1, `Ġtuck`:1, `Ġbowls`:1, `ubby`:1, `Ġjeans`:1 | `ucked`:1, `Ġgrub`:1, `Ġdismay`:1, `Ġfragr`:1, `Ġshave`:1 |
| family::SuperGLUE | 92959 | 0 | 0.9954 | 1.28 | 1.69 | -0.786 | `NN`:4024, `Ġatoms`:2020, `instein`:1847, `Ġcontroll`:1786, `pit`:1312 | `Ġterrorist`:2204, `ĠAbu`:1922, `ĠAlban`:1667, `ĠSaudi`:1581, `Ġagencies`:1428 |

## EWoK domain tokenization pressure

| domain | texts | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token Δ |
|---|---:|---:|---:|---:|---:|
| agent-properties | 13260 | 1.0058 | 1.70 | 1.04 | 0.090 |
| material-dynamics | 4620 | 0.9856 | 0.39 | 2.72 | -0.110 |
| material-properties | 1020 | 1.0226 | 1.93 | 2.21 | 0.243 |
| physical-dynamics | 720 | 1.0257 | 2.57 | 0.00 | 0.250 |
| physical-interactions | 3336 | 1.0248 | 1.83 | 1.65 | 0.315 |
| physical-relations | 4908 | 1.0000 | 0.00 | 0.00 | 0.000 |
| quantitative-properties | 1884 | 1.0147 | 2.19 | 1.58 | 0.183 |
| social-interactions | 1764 | 1.0217 | 3.37 | 0.96 | 0.227 |
| social-properties | 1968 | 1.0110 | 1.74 | 0.85 | 0.104 |
| social-relations | 9288 | 1.0350 | 4.68 | 0.19 | 0.342 |
| spatial-relations | 2940 | 1.0184 | 1.11 | 0.71 | 0.269 |

## Word-level fragmentation readout

### ALL

Changed word types: 6646 / 83742; weighted extra pieces when compliant tokenizer is longer: 216144; weighted saved pieces when compliant tokenizer is shorter: 275704.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| octopus | 993 | 36 | 2 | 5 | `['Ġ', 'Ġoctopus']` | `['Ġ', 'Ġo', 'ct', 'op', 'us']` |
| oatmeal | 410 | 22 | 2 | 5 | `['Ġ', 'Ġoatmeal']` | `['Ġ', 'Ġo', 'at', 'me', 'al']` |
| auspicious | 78 | 21 | 2 | 5 | `['Ġ', 'Ġauspicious']` | `['Ġ', 'Ġa', 'us', 'p', 'icious']` |
| auspiciousness | 2 | 1 | 3 | 6 | `['Ġ', 'Ġauspicious', 'ness']` | `['Ġ', 'Ġa', 'us', 'p', 'icious', 'ness']` |
| crocodile | 1341 | 30 | 2 | 4 | `['Ġ', 'Ġcrocodile']` | `['Ġ', 'Ġcro', 'cod', 'ile']` |
| screwdriver | 1034 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |
| aliens | 848 | 31 | 2 | 4 | `['Ġ', 'Ġaliens']` | `['Ġ', 'Ġal', 'i', 'ens']` |
| hedgehog | 720 | 26 | 3 | 5 | `['Ġ', 'Ġhedge', 'hog']` | `['Ġ', 'Ġhe', 'dge', 'h', 'og']` |
| francisco | 383 | 241 | 3 | 5 | `['Ġ', 'Ġfranc', 'isco']` | `['Ġ', 'Ġfr', 'anc', 'is', 'co']` |
| skateboards | 366 | 0 | 3 | 5 | `['Ġ', 'Ġskate', 'boards']` | `['Ġ', 'Ġsk', 'ate', 'bo', 'ards']` |
| trolley | 296 | 17 | 2 | 4 | `['Ġ', 'Ġtrolley']` | `['Ġ', 'Ġt', 'roll', 'ey']` |
| ladders | 262 | 60 | 2 | 4 | `['Ġ', 'Ġladders']` | `['Ġ', 'Ġl', 'add', 'ers']` |

### family::EWoK

Changed word types: 54 / 866; weighted extra pieces when compliant tokenizer is longer: 3316; weighted saved pieces when compliant tokenizer is shorter: 2020.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| screwdriver | 404 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |
| toothpaste | 108 | 59 | 2 | 4 | `['Ġ', 'Ġtoothpaste']` | `['Ġ', 'Ġtooth', 'p', 'aste']` |
| bouncy | 56 | 15 | 2 | 4 | `['Ġ', 'Ġbouncy']` | `['Ġ', 'Ġb', 'ounc', 'y']` |
| hates | 690 | 59 | 2 | 3 | `['Ġ', 'Ġhates']` | `['Ġ', 'Ġhat', 'es']` |
| chess | 328 | 58 | 2 | 3 | `['Ġ', 'Ġchess']` | `['Ġ', 'Ġc', 'hess']` |
| landlord | 290 | 44 | 2 | 3 | `['Ġ', 'Ġlandlord']` | `['Ġ', 'Ġlandl', 'ord']` |
| antennas | 152 | 7 | 4 | 5 | `['Ġ', 'Ġanten', 'n', 'as']` | `['Ġ', 'Ġan', 'ten', 'n', 'as']` |
| heating | 120 | 53 | 2 | 3 | `['Ġ', 'Ġheating']` | `['Ġ', 'Ġhe', 'ating']` |
| decides | 100 | 51 | 2 | 3 | `['Ġ', 'Ġdecides']` | `['Ġ', 'Ġdec', 'ides']` |
| employee | 90 | 64 | 2 | 3 | `['Ġ', 'Ġemployee']` | `['Ġ', 'Ġemploy', 'ee']` |
| untrustworthy | 50 | 1 | 4 | 5 | `['Ġ', 'Ġunt', 'rust', 'worthy']` | `['Ġ', 'Ġunt', 'r', 'ust', 'worthy']` |
| suite | 44 | 68 | 2 | 3 | `['Ġ', 'Ġsuite']` | `['Ġ', 'Ġsu', 'ite']` |

### family::Supplement

Changed word types: 132 / 1802; weighted extra pieces when compliant tokenizer is longer: 2038; weighted saved pieces when compliant tokenizer is shorter: 2671.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| aeroplanes | 22 | 5 | 3 | 5 | `['Ġ', 'Ġaeropl', 'anes']` | `['Ġ', 'Ġaer', 'op', 'l', 'anes']` |
| david | 706 | 1258 | 3 | 4 | `['Ġ', 'Ġd', 'avid']` | `['Ġ', 'Ġd', 'av', 'id']` |
| annoying | 72 | 65 | 2 | 3 | `['Ġ', 'Ġannoying']` | `['Ġ', 'Ġannoy', 'ing']` |
| repaired | 72 | 55 | 2 | 3 | `['Ġ', 'Ġrepaired']` | `['Ġ', 'Ġrep', 'aired']` |
| bracelet | 66 | 62 | 2 | 3 | `['Ġ', 'Ġbracelet']` | `['Ġ', 'Ġbrace', 'let']` |
| coke | 50 | 85 | 2 | 3 | `['Ġ', 'Ġcoke']` | `['Ġ', 'Ġco', 'ke']` |
| employee | 50 | 64 | 2 | 3 | `['Ġ', 'Ġemployee']` | `['Ġ', 'Ġemploy', 'ee']` |
| cheat | 50 | 47 | 2 | 3 | `['Ġ', 'Ġcheat']` | `['Ġ', 'Ġche', 'at']` |
| gardener | 48 | 53 | 2 | 3 | `['Ġ', 'Ġgardener']` | `['Ġ', 'Ġgard', 'ener']` |
| hunter | 42 | 125 | 2 | 3 | `['Ġ', 'Ġhunter']` | `['Ġ', 'Ġhun', 'ter']` |
| restaurants | 42 | 70 | 2 | 3 | `['Ġ', 'Ġrestaurants']` | `['Ġ', 'Ġrestaur', 'ants']` |
| mansion | 40 | 57 | 2 | 3 | `['Ġ', 'Ġmansion']` | `['Ġ', 'Ġmans', 'ion']` |

### family::GlobalPIQA_parallel

Changed word types: 22 / 780; weighted extra pieces when compliant tokenizer is longer: 129; weighted saved pieces when compliant tokenizer is shorter: 63.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| fluffy | 5 | 29 | 2 | 4 | `['Ġ', 'Ġfluffy']` | `['Ġ', 'Ġfl', 'uff', 'y']` |
| cupboard | 19 | 62 | 2 | 3 | `['Ġ', 'Ġcupboard']` | `['Ġ', 'Ġcup', 'board']` |
| backpack | 15 | 39 | 3 | 4 | `['Ġ', 'Ġback', 'pack']` | `['Ġ', 'Ġback', 'p', 'ack']` |
| measuring | 10 | 71 | 2 | 3 | `['Ġ', 'Ġmeasuring']` | `['Ġ', 'Ġmeas', 'uring']` |
| heavier | 10 | 68 | 2 | 3 | `['Ġ', 'Ġheavier']` | `['Ġ', 'Ġheav', 'ier']` |
| baking | 10 | 64 | 2 | 3 | `['Ġ', 'Ġbaking']` | `['Ġ', 'Ġb', 'aking']` |
| freeze | 9 | 83 | 2 | 3 | `['Ġ', 'Ġfreeze']` | `['Ġ', 'Ġfree', 'ze']` |
| sweeping | 7 | 66 | 2 | 3 | `['Ġ', 'Ġsweeping']` | `['Ġ', 'Ġswe', 'eping']` |
| cans | 5 | 65 | 2 | 3 | `['Ġ', 'Ġcans']` | `['Ġ', 'Ġcan', 's']` |
| floors | 5 | 64 | 2 | 3 | `['Ġ', 'Ġfloors']` | `['Ġ', 'Ġflo', 'ors']` |
| loaf | 5 | 53 | 2 | 3 | `['Ġ', 'Ġloaf']` | `['Ġ', 'Ġlo', 'af']` |
| parked | 5 | 52 | 2 | 3 | `['Ġ', 'Ġparked']` | `['Ġ', 'Ġpark', 'ed']` |

### family::SuperGLUE

Changed word types: 6502 / 83035; weighted extra pieces when compliant tokenizer is longer: 131705; weighted saved pieces when compliant tokenizer is shorter: 235225.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| oatmeal | 410 | 22 | 2 | 5 | `['Ġ', 'Ġoatmeal']` | `['Ġ', 'Ġo', 'at', 'me', 'al']` |
| octopus | 108 | 36 | 2 | 5 | `['Ġ', 'Ġoctopus']` | `['Ġ', 'Ġo', 'ct', 'op', 'us']` |
| auspicious | 78 | 21 | 2 | 5 | `['Ġ', 'Ġauspicious']` | `['Ġ', 'Ġa', 'us', 'p', 'icious']` |
| auspiciousness | 2 | 1 | 3 | 6 | `['Ġ', 'Ġauspicious', 'ness']` | `['Ġ', 'Ġa', 'us', 'p', 'icious', 'ness']` |
| aliens | 848 | 31 | 2 | 4 | `['Ġ', 'Ġaliens']` | `['Ġ', 'Ġal', 'i', 'ens']` |
| francisco | 383 | 241 | 3 | 5 | `['Ġ', 'Ġfranc', 'isco']` | `['Ġ', 'Ġfr', 'anc', 'is', 'co']` |
| envoys | 200 | 4 | 3 | 5 | `['Ġ', 'Ġenv', 'oys']` | `['Ġ', 'Ġen', 'v', 'oy', 's']` |
| subtropical | 179 | 22 | 3 | 5 | `['Ġ', 'Ġsubt', 'ropical']` | `['Ġ', 'Ġsub', 't', 'rop', 'ical']` |
| skateboards | 156 | 0 | 3 | 5 | `['Ġ', 'Ġskate', 'boards']` | `['Ġ', 'Ġsk', 'ate', 'bo', 'ards']` |
| merchants | 147 | 39 | 2 | 4 | `['Ġ', 'Ġmerchants']` | `['Ġ', 'Ġmer', 'chan', 'ts']` |
| trampoline | 107 | 0 | 3 | 5 | `['Ġ', 'Ġtramp', 'oline']` | `['Ġ', 'Ġtr', 'amp', 'ol', 'ine']` |
| francisco's | 103 | 0 | 4 | 6 | `['Ġ', 'Ġfranc', 'isco', "'s"]` | `['Ġ', 'Ġfr', 'anc', 'is', 'co', "'s"]` |

### ewok_domain::spatial-relations

Changed word types: 1 / 87; weighted extra pieces when compliant tokenizer is longer: 624; weighted saved pieces when compliant tokenizer is shorter: 0.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| screwdriver | 312 | 43 | 2 | 4 | `['Ġ', 'Ġscrewdriver']` | `['Ġ', 'Ġscrew', 'dri', 'ver']` |

### ewok_domain::material-dynamics

Changed word types: 8 / 60; weighted extra pieces when compliant tokenizer is longer: 136; weighted saved pieces when compliant tokenizer is shorter: 644.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| coke | 76 | 85 | 2 | 3 | `['Ġ', 'Ġcoke']` | `['Ġ', 'Ġco', 'ke']` |
| taps | 60 | 99 | 2 | 3 | `['Ġ', 'Ġtaps']` | `['Ġ', 'Ġtap', 's']` |

### ewok_domain::physical-dynamics

Changed word types: 4 / 69; weighted extra pieces when compliant tokenizer is longer: 180; weighted saved pieces when compliant tokenizer is shorter: 0.

Top words longer under the compliant tokenizer:

| word | eval count | train count | old len | new len | old tokens | new tokens |
|---|---:|---:|---:|---:|---|---|
| sinking | 80 | 70 | 2 | 3 | `['Ġ', 'Ġsinking']` | `['Ġ', 'Ġsink', 'ing']` |
| sliding | 60 | 64 | 2 | 3 | `['Ġ', 'Ġsliding']` | `['Ġ', 'Ġsl', 'iding']` |
| slippery | 20 | 49 | 2 | 3 | `['Ġ', 'Ġslippery']` | `['Ġ', 'Ġslipp', 'ery']` |
| underwater | 20 | 41 | 2 | 3 | `['Ġ', 'Ġunderwater']` | `['Ġ', 'Ġunder', 'water']` |

## Interpretation for a possible below-leader compliant result

If the compliant endpoint loses only modestly, the tokenization evidence suggests continuing to official collation rather than attributing the movement to crude length/truncation. If it falls well below the visible leader, the first fork should be: compare whether losses concentrate in families/domains where this analysis shows unusually high old-only occurrence mass or word fragmentation. A concentration there would make legal tokenizer learning worth studying using only the 10M pool and generic tokenizer priors. A broad loss or a loss in domains with small tokenization pressure would point instead to changed MLM target geometry, optimizer/initialization dynamics, or the data mechanism itself under the new vocabulary.
