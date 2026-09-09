# BabyLM Evaluation 2026
!BabyLM Challenge

This repository contains the setup for evaluation for BabyLM 2026. We provide separate evaluation for the Strict (+Strict-Small) track, and the Multilingual track. See the two track directories for more specific information on evaluation for these two tracks.

The Multilingual Chinese zero-shot suite includes the Hanzi structure and
pinyin minimal-pair evaluations.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation` channel, which is dedicated to support for use of this repository.

We also welcome pull requests!

## Leaderboard
The leaderboard is live now: 

[![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow?style=for-the-badge)](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)

### Submitting to the leaderboard

The two tracks use different submission formats:

#### Strict and Strict-small tracks

Upload a **predictions file** produced by the BabyLM evaluation pipeline. After running both zero-shot and fine-tuning evaluation on your final model, create the submission file with:

```bash
cd strict
bash scripts/collate_preds.sh NAME_OF_YOUR_MODEL BACKEND SUBMISSION_TRACK
```

Scores are computed server-side against the held-out targets, so you only upload predictions. Add the `--fast` flag to also include checkpoint evaluation results, which is required for a full BabyLM Challenge submission.

#### Multilingual track

Upload the **scores and predictions files** produced by the collator. Public
tasks use their pre-computed scores; hidden Hanzi and MECO scores are computed
server-side from raw predictions/surprisals:

```bash
cd multilingual
bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "eng nld zho"
bash scripts/finetune_model.sh --model_name YOUR_MODEL --langs "eng nld zho"

# Produce <model>_submission.json and <model>_predictions.json
python scripts/collate_results.py --model_name YOUR_MODEL
```

> [!NOTE]
> Incomplete evaluation is allowed for both tracks: you can submit results covering only the languages or tasks your model was trained on. Missing tasks are set to 0 and are taken into account when computing the average score.

> [!IMPORTANT]
> Make sure your model is **publicly available** on HuggingFace before submitting.

## Baselines
We train GPT-2 baseline models on the datasets. We provide monolingual models, as well as bi- and trilingual models, trained on equal language splits. The models are available on HuggingFace [here](https://huggingface.co/BabyLM-community/models).

Following [BabyBabelLM](https://arxiv.org/pdf/2510.10159), we divide evaluation for the multilingual models up in zero-shot and fine-tuning evaluation. Zero-shot evaluation for the multilingual track is done through `lm-eval`. Fine-tuning is adapted from a script of previous BabyLM editions.

### Strict / Strict-Small
#### Zero-shot
| task (metric) | Baseline-GPT2-Strict | Baseline-Strict-Interaction | Baseline-GPT2-Strict-Small | Baseline-Strict-Small-Interaction |
| --- | ---: | ---: | ---: | ---: |
| BLiMP (acc) | **74.73** | 72.89 | **65.23** | 63.07 |
| BLiMP Supplement (acc) | 65.00 | **65.09** | 57.25 | **58.13** |
| EWoK (acc) | **54.37** | 54.23 | 50.63 | **51.45** |
| Entity Tracking (acc) | **16.91** | 15.89 | 19.10 | **19.62** |
| COMPS (acc) | **55.85** | 55.37 | **51.81** | 51.29 |
| GlobalPIQA (acc) | **36.62** | 36.18 | 35.09 | **36.14** |

#### Human-likeness
| task (metric) | Baseline-GPT2-Strict | Baseline-Strict-Interaction | Baseline-GPT2-Strict-Small | Baseline-Strict-Small-Interaction |
| --- | ---: | ---: | ---: | ---: |
| Reading (delta % R2) | **6.93** | 3.43 | **5.63** | 5.10 |
| AoA (MSE) | -11.58 | **-11.36** | **-12.15** |  |

#### Finetune ((Super)GLUE)
| task (metric) | Baseline-GPT2-Strict | Baseline-Strict-Interaction | Baseline-GPT2-Strict-Small | Baseline-Strict-Small-Interaction |
| --- | ---: | ---: | ---: | ---: |
| boolq (accuracy) | **69.66** | 67.46 | 67.71 | **68.01** |
| mnli (accuracy) | **60.76** | 60.68 | 49.84 | **51.20** |
| mrpc (f1) | 85.34 | **86.27** | **81.37** | 81.00 |
| multirc (accuracy) | 65.92 | **66.21** | **65.76** | 65.31 |
| qqp (f1) | **71.56** | 70.51 | 61.67 | **64.31** |
| rte (accuracy) | 57.55 | **58.27** | **56.83** | 53.96 |
| wsc (accuracy) | **63.46** | 61.54 | **63.46** | 61.54 |
| *avg* | **67.75** | 67.28 | **63.80** | 63.62 |

### Multilingual Track

#### Zero-shot Tasks
| task | GPT2-Strict | GPT2-en_nld_equal | GPT2-en_zho_equal | GPT2-nld_zho_equal | GPT2-en_nld_zho_equal |
| --- | ---: | ---: | ---: | ---: | ---: |
| **zeroshot_eng** |  |  |  |  |  |
| blimp | **73.81** | 72.18 | 72.28 |  | 70.49 |
| global_piqa | 36.62 | 34.65 | 38.58 |  | **39.12** |
| hellaswag_en_mubench | 26.45 | **26.51** | 26.35 |  | **26.51** |
| multiblimp_eng | **88.57** | 87.92 | 88.18 |  | 85.97 |
| winogrande_en_mubench | **51.44** | 50.54 | 50.95 |  | 50.12 |
| xstorycloze_en_mubench | **50.70** | 50.62 | 49.46 |  | 49.54 |
| *avg* | 54.60 | 53.74 | 54.30 |  | 53.63 |
| **zeroshot_nld** |  |  |  |  |  |
| blimp_nl |  | **81.70** |  | 80.54 | 77.04 |
| global_piqa |  | **41.61** |  | 40.65 | 38.11 |
| hellaswag_nl_mubench |  | 26.45 |  | **26.96** | 26.65 |
| multiblimp_nld |  | 92.62 |  | **93.69** | 92.11 |
| winogrande_nl_mubench |  | 48.80 |  | **51.03** | 49.38 |
| xcomps_nl |  | **53.87** |  | 52.68 | 53.04 |
| xstorycloze_nl_mubench |  | 48.92 |  | **49.38** | 48.99 |
| *avg* |  | 56.28 |  | 56.42 | 55.05 |
| **zeroshot_zho** |  |  |  |  |  |
| hellaswag_zh_mubench |  |  | **27.06** | 26.67 | 26.74 |
| global_piqa |  |  | **39.61** | 37.18 | 31.74 |
| winogrande_zh_mubench |  |  | 49.55 | **50.87** | 49.63 |
| xcomps_zh |  |  | 53.61 | **54.06** | 53.47 |
| xstorycloze_zh_mubench |  |  | **48.84** | 47.99 | 47.91 |
| zhoblimp |  |  | **77.03** | 76.09 | 74.03 |
| *avg* |  |  | 49.28 | 48.81 | 47.25 |

#### Chinese Hanzi Zero-shot
| task | GPT2-Strict | GPT2-en_nld_equal | GPT2-en_zho_equal | GPT2-nld_zho_equal | GPT2-en_nld_zho_equal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hanzi Structure |  |  | 55.05 | **56.15** | 55.05 |
| Hanzi Pinyin |  |  | 48.75 | 48.20 | **50.05** |

#### MECO Reading Time
| task | GPT2-Strict | GPT2-en_nld_equal | GPT2-en_zho_equal | GPT2-nld_zho_equal | GPT2-en_nld_zho_equal |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 English | 5086.50 | 5203.98 | **5209.61** |  | 5157.10 |
| L1 Dutch |  | **425.05** |  | 276.33 | 388.27 |
| L1 Chinese |  |  | **638.52** | 628.91 | 632.02 |
| L2 Dutch |  | 13.80 |  |  | **55.92** |
| L2 Chinese |  |  | **107.10** |  | 75.47 |

#### Finetuning tasks
| task | GPT2-Strict | GPT2-en_nld_equal | GPT2-en_zho_equal | GPT2-nld_zho_equal | GPT2-en_nld_zho_equal |
| --- | ---: | ---: | ---: | ---: | ---: |
| **en** |  |  |  |  |  |
| arc | 24.17 | 24.38 | 23.54 |  | **24.79** |
| belebele | **26.14** | 22.73 | **26.14** |  | **26.14** |
| bmlama | 11.01 | 10.93 | 8.94 |  | **12.00** |
| mnli | **51.91** | 45.61 | 49.38 |  | 50.00 |
| pos |  | **93.83** | 93.48 |  | 93.40 |
| sib200 | **82.50** | 77.00 | 21.50 |  | 69.50 |
| truthfulqa | **23.21** | **23.21** | **23.21** |  | **23.21** |
| xnli | 46.55 | **47.30** | 46.50 |  | 45.65 |
| *avg* | 33.19 | 43.12 | 36.59 |  | 43.09 |
| **nl** |  |  |  |  |  |
| arc |  | **24.38** |  | **24.38** | **24.38** |
| belebele |  | 22.73 |  | 22.73 | **26.14** |
| bmlama |  | **12.00** |  | 9.69 | 9.93 |
| include |  | **31.25** |  | 19.64 | 19.64 |
| mnli |  | 45.38 |  | 48.87 | **49.27** |
| pos |  | 95.54 |  | **95.56** | 94.97 |
| sib200 |  | **76.00** |  | 21.50 | 71.00 |
| truthfulqa |  | **23.21** |  | **23.21** | **23.21** |
| *avg* |  | 41.31 |  | 33.20 | 39.82 |
| **zh** |  |  |  |  |  |
| arc |  |  | 23.54 | 24.79 | **25.21** |
| belebele |  |  | **26.14** | 22.73 | 22.73 |
| bmlama |  |  | 8.86 | **12.00** | 10.93 |
| include |  |  | 26.79 | 26.79 | **28.57** |
| mnli |  |  | 46.06 | 44.99 | **47.58** |
| pos |  |  | 90.72 | **91.00** | 90.40 |
| sib200 |  |  | **78.00** | 75.50 | 72.50 |
| truthfulqa |  |  | **23.21** | **23.21** | **23.21** |
| xnli |  |  | **46.30** | 43.65 | 45.05 |
| *avg* |  |  | 41.07 | 40.52 | 40.69 |

## Citation
```
@misc{choshen2026babylmturns4goes,
      title={BabyLM Turns 4 and Goes Multilingual: Call for Papers for the 2026 BabyLM Workshop}, 
      author={Leshem Choshen and Ryan Cotterell and Mustafa Omer Gul and Jaap Jumelet and Tal Linzen and Aaron Mueller and Suchir Salhan and Raj Sanjay Shah and Alex Warstadt and Ethan Gotlieb Wilcox},
      year={2026},
      eprint={2602.20092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.20092}, 
}
```
