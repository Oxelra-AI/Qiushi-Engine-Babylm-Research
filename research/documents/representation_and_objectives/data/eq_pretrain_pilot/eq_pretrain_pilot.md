# raw span identity routing result equality pretraining pilot

Train examples: 352; eval examples: 752; vocab size: 60

| epoch | train both gate | eval both gate | train cand>other | eval cand>other | train neither(cand) | eval neither(cand) | elapsed s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.599 | 0.585 | 0.738 | 0.731 | 3.2 |
| 1 | 0.000 | 0.000 | 0.659 | 0.725 | 0.704 | 0.703 | 8.7 |
| 5 | 0.000 | 0.000 | 0.670 | 0.601 | 0.472 | 0.509 | 23.8 |
| 10 | 0.227 | 0.000 | 0.761 | 0.636 | 0.395 | 0.459 | 40.5 |
| 25 | 0.625 | 0.189 | 0.844 | 0.842 | 0.192 | 0.362 | 84.0 |
| 50 | 1.000 | 0.407 | 1.000 | 1.000 | 0.078 | 0.339 | 158.5 |
| 100 | 1.000 | 0.654 | 1.000 | 0.971 | 0.010 | 0.227 | 449.4 |
