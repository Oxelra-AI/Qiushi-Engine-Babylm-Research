---
license: mit
---
# Detoxified 10M Strict-Small BabyLM Training Dataset (BabyLM Turns 4, 2026 BabyLM)

BabyLM 2026 Strict-Small training set. Total: **10M tokens**.


Please cite the following: 

```
@misc{choshen2026babylmturns4papers,
      title={BabyLM Turns 4: Call for Papers for the 2026 BabyLM Workshop}, 
      author={Leshem Choshen and Ryan Cotterell and Mustafa Omer Gul and Jaap Jumelet and Tal Linzen and Aaron Mueller and Suchir Salhan and Raj Sanjay Shah and Alex Warstadt and Ethan Gotlieb Wilcox},
      year={2026},
      eprint={2602.20092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.20092}, 
}
```

## Token Counts Breakdown


| File | Tokens |
|------|-------:|
| bnc_spoken.train.txt | 762,073 |
| childes.train.txt | 2,841,101 |
| gutenberg.train.txt | 2,557,721 |
| open_subtitles.train.txt | 2,282,877 |
| simple_wiki.train.txt | 1,531,437 |
| switchboard.train.txt | 24,791 |
| **Total** | **10,000,000** |

## Data Decontamination

All training data was subjected to precorpus debiasing using pipeline for detecting and removing toxic content from naturalistic language corpora. The pipeline applies hate speech detection, sentiment scoring, and emotion analysis to flag problematic sentences, which are then filtered using gender and race word lists alongside an explicit slur lexicon. This ensures that demographic mentions and identity-related language in the corpus do not carry harmful associations that could be learned by models trained on this data. Training Data Decontamination was motivated by efforts on the Interaction Track (e.g., Salhan et al 2025; Trhlik, Caines & Buttery 2026).



```
@misc{salhan2025teacherdemonstrationsbabylmszone,
      title={Teacher Demonstrations in a BabyLM's Zone of Proximal Development for Contingent Multi-Turn Interaction}, 
      author={Suchir Salhan and Hongyi Gu and Donya Rooein and Diana Galvan-Sosa and Gabrielle Gaudeau and Andrew Caines and Zheng Yuan and Paula Buttery},
      year={2025},
      eprint={2510.20411},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.20411}, 
}
```

```
@misc{trhlik2026biasdynamicsbabylmscomputeefficient,
      title={Bias Dynamics in BabyLMs: Towards a Compute-Efficient Sandbox for Democratising Pre-Training Debiasing}, 
      author={Filip Trhlik and Andrew Caines and Paula Buttery},
      year={2026},
      eprint={2601.09421},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09421}, 
}
