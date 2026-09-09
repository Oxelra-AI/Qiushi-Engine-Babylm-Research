# changed block neardup scan direct read of score-bearing changed-block exact-overlap hits

## Purpose

The refined exact-7gram overlap audit was broadly reassuring, but the small score-bearing `glue_valid_scored` changed-block hits still required direct inspection before treating the reinvest endpoint as submission-grade. This note performs that reading using the overlap audit and exact evaluation/training lines. It is CPU/text work only.

Source audit: `experiments/archive/representation_and_objectives/data/reinvest_overlap_refined/compact_reinvest_refined_eval_overlap_audit.json`.

## Audit counts to preserve

For `candidate_changed_block_only` (the FineWeb compact changed block inside the 42.0868 reinvest endpoint):

- Scanned 3005 rows / 423,511 words.
- Any exact-7gram overlap rows: 32.
- Score-bearing overlap rows: 8.
- GLUE-valid score-bearing overlap rows: 5.
- GLUE-train overlap rows: 25.
- Unique matching 7-grams: 56.
- Eval-record hits by file role: 71 GLUE-train (fine-tuning train, not directly scored), 7 GLUE-valid scored, 5 zero-shot/human-scored.

The changed block is not worse than the clean-Qwen official rows it replaced; the refined audit gives changed-minus-heldout: score-bearing rows −12, GLUE-valid rows −14, unique matching 7-grams −103.

## Directly inspected GLUE-valid score-bearing hits

### 1. BoolQ valid line 130

Eval line:
`experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl:130`

- Eval exact phrase: `in many other parts of the world`.
- Eval item asks whether England is the only country to change clocks; passage is about European Summer Time and says DST is found in many other parts of the world.
- Training row line 222 / example_id 950221 says: `As in many other parts of the world, surnames were a rarity until the late Joseon dynasty (1392-1910).`

Reading: generic comparative phrase; completely different topic (DST vs surnames/Joseon). Not a target-example copy.

### 2. QQP valid line 6011

Eval line:
`.../glue_filtered/qqp.valid.jsonl:6011`

- Eval question pair: `How can I help a friend who is experiencing serious depression?` / `How do I help a friend who is in depression?`
- Exact overlap examples in the audit: `how can i help a friend who`, `can i help a friend who is`, `i help a friend who is in`.
- Training row line 663 / example_id 950662 says: `How can I help a friend who is in a violent relationship? - Tell your friend that you are worried about them.`

Reading: shared generic advice-question stem, different condition and no target pair or label. It is a public phrase-template overlap, not a QQP validation item memorization.

### 3. BoolQ valid line 269

Eval line:
`.../glue_filtered/boolq.valid.jsonl:269`

- Eval exact phrase: `effect of the 2004 indian ocean earthquake`.
- Eval asks about tsunami in Maldives; passage is `Effect of the 2004 Indian Ocean earthquake on the Maldives ...`.
- Training row line 723 / example_id 950722 says: `GAM are the former rebel movement who fought the government for 29 years until the cataclysmic effect of the 2004 Indian Ocean Earthquake.`

Reading: shared public historical event phrase, different proposition and target question. Not a held-out BoolQ example copy.

### 4. BoolQ valid line 365

Eval line:
`.../glue_filtered/boolq.valid.jsonl:365`

- Eval exact phrase: `compared to other parts of the world`.
- Eval asks whether the Balkans were part of the Ottoman Empire; passage compares Ottoman religious practices to other parts of the world.
- Training row line 1753 / example_id 951752 says: `Everyone is aware that overall U.S. math achievement isn't very good in relative terms or compared to other parts of the world.`

Reading: generic comparative phrase; different topic and proposition. Not target leakage.

### 5. MNLI valid line 4316

Eval line:
`.../glue_filtered/mnli.valid.jsonl:4316`

- Eval exact phrase: `it would be a good idea to` in hypothesis: `The layman thought it would be a good idea to question every parishioner.`
- Training row line 2607 / example_id 952606 says: `There are some foods which carry more of a risk than others, and it would be a good idea to avoid or limit these foods if you want to maintain proper dental health.`

Reading: generic idiom, not shared premise/hypothesis content. Not target leakage.

## Other score-bearing hits

- AoA hits found by grep in `aoa/cdi_childes.json` are ordinary context phrases such as `At the height of the Cold War...` and `it would be a good idea to...`; they are not AoA target word identities used for corpus design. The changed block was not selected using CDI/AoA words or ages.
- The VQA exact phrase reported by the audit (`are on the left side of the`) is a generic spatial phrase; `vqa_distractors_info.json` contains many generic questions/distractors involving `left side`, `side of`, `alive`, etc. The training phrase is about Christ scenes on the left side of a mirror, not a visual-question target item.

## Judgment

The direct-read evidence supports the reinvest endpoint review/consolidated margin and preservation evidence conclusion: exact 7-word overlaps in the changed block are sparse, lower than in the heldout official rows that were replaced, and the inspected GLUE-valid score-bearing hits are generic public phrases embedded in different topics rather than whole target examples, labels, or task-specific answer patterns. This does **not** prove absence of all near-duplicates or paraphrase-like overlap, but it removes the main concern that the 42.0868 endpoint is explained by obvious exact validation leakage.

Remaining contamination-safety work, if the seed43122 AoA evidence remains viable: run a stronger near-duplicate scan (e.g. MinHash/SimHash or high-recall token-window Jaccard) over changed-block rows versus score-bearing eval strings, prioritizing GLUE-valid, zero-shot, and human-scored text. This should be done before final submission packaging, but it does not justify changing or retraining the frozen endpoint by itself.
