# orbit pair audit and slot reuse update globally injective alias feasibility

Unique candidate keys: 9723; candidate key-events: 24970

Profile deficit exists: True

| profile | keys | key-events | aliases | surplus | examples |
|---|---:|---:|---:|---:|---|
| (3, 2, 4) | 3737 | 7103 | 9305 | 5568 | [{'key': 'bmi', 'original': 'BMI', 'row': 10}, {'key': 'burroughs', 'original': 'Burroughs', 'row': 12}, {'key': 'tavel', 'original': 'Tavel', 'row': 21}] |
| (2, 1, 3) | 2964 | 7812 | 7611 | 4647 | [{'key': 'calvi', 'original': 'Calvi', 'row': 30}, {'key': 'teano', 'original': 'Teano', 'row': 30}, {'key': 'romulus', 'original': 'Romulus', 'row': 33}] |
| (4, 3, 5) | 1665 | 2530 | 3977 | 2312 | [{'key': 'raspbian', 'original': 'Raspbian', 'row': 14}, {'key': 'ciphet', 'original': 'CIPHET', 'row': 17}, {'key': 'icwa', 'original': 'ICWA', 'row': 26}] |
| (5, 4, 6) | 577 | 817 | 1074 | 497 | [{'key': 'brodersdorf', 'original': 'Brodersdorf', 'row': 36}, {'key': 'gethashcode', 'original': 'GetHashCode', 'row': 97}, {'key': 'mcclellan', 'original': 'McClellan', 'row': 126}] |
| (1, 0, 2) | 400 | 6194 | 489 | 89 | [{'key': 'americans', 'original': 'Americans', 'row': 24}, {'key': 'oklahoma', 'original': 'Oklahoma', 'row': 31}, {'key': 'germany', 'original': 'Germany', 'row': 33}] |
| (6, 5, 7) | 201 | 272 | 242 | 41 | [{'key': 'styrofoam', 'original': 'Styrofoam', 'row': 200}, {'key': 'sleeptunertm', 'original': 'SleepTunerTM', 'row': 214}, {'key': 'anglo-nederlandish', 'original': 'Anglo-Nederlandish', 'row': 356}] |
| (7, 6, 8) | 106 | 128 | 38 | -68 | [{'key': 'max-planck-institute', 'original': 'Max-Planck-Institute', 'row': 202}, {'key': 'fda-approved', 'original': 'FDA-approved', 'row': 219}, {'key': 'fipronil-treated', 'original': 'Fipronil-treated', 'row': 934}] |
| (8, 7, 9) | 50 | 66 | 8 | -42 | [{'key': 'g-protein-coupled', 'original': 'G-protein-coupled', 'row': 142}, {'key': 'ancistrorhynchus', 'original': 'Ancistrorhynchus', 'row': 632}, {'key': 'siebengebirge', 'original': 'Siebengebirge', 'row': 845}] |
| (9, 8, 10) | 15 | 40 | 0 | -15 | [{'key': 'bourgogne-franche-comt', 'original': 'Bourgogne-Franche-Comt', 'row': 3965}, {'key': 'flaumont-waudrechies', 'original': 'Flaumont-Waudrechies', 'row': 4138}, {'key': 'corcelles-sur-chavornay', 'original': 'Corcelles-sur-Chavornay', 'row': 10094}] |
| (10, 9, 11) | 5 | 5 | 0 | -5 | [{'key': 'roquelaure-saint-aubin', 'original': 'Roquelaure-Saint-Aubin', 'row': 39056}, {'key': "monlezun-d'armagnac", 'original': "Monlezun-d'Armagnac", 'row': 43050}, {'key': "saint-martin-d'armagnac", 'original': "Saint-Martin-d'Armagnac", 'row': 44297}] |
| (12, 11, 13) | 3 | 3 | 0 | -3 | [{'key': 'aulnoy-lez-valenciennes', 'original': 'Aulnoy-lez-Valenciennes', 'row': 35689}, {'key': 'llanrhaeadr-ym-mochnant', 'original': 'Llanrhaeadr-ym-Mochnant', 'row': 44103}, {'key': 'lah-dle-ah-dle-ah-dle', 'original': 'Lah-dle-ah-dle-ah-dle', 'row': 57435}] |

Greedy unique-alias trial: assigned 9568, failed 155 with max_compat_check=200.

JSON: `experiments/archive/representation_and_objectives/data/global_injective_feasibility/global_injective_feasibility.json`
