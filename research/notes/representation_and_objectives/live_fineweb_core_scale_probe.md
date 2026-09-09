# live fineweb core scale probe live FineWeb core-fact scale probe

CPU/network-only source-yield measurement for a possible corrected FineWeb source/source+view family; it does not train or evaluate a model.

Scanned 4,836 docs / 3,756,512 document words / 234,640 raw sentences.

Core accepted before near-dedup: 16,818 rows / 435,229 words.

After local near-dedup (threshold 0.82): 16,817 rows / 435,210 words; rejected 1 rows / 19 words.

Doc caps after near-dedup: cap8 11,554 rows / 300,030 words; cap4 8,615 rows / 223,513 words; cap2 5,649 rows / 146,633 words.

Yield per scanned document word: strict_anchor_like=58.613%, core_all=11.586%, core_neardedup=11.585%, core_doc_cap8=7.987%, core_doc_cap4=5.950%, core_doc_cap2=3.903%.

Projected scanned document words needed for target cap8/cap4 core-source budgets:

| target source words | cap8 projection | cap4 projection | core-neardedup projection |
|---:|---:|---:|---:|
| 1,000,000 | 12,520,455 | 16,806,682 | 8,631,493 |
| 1,750,000 | 21,910,796 | 29,411,694 | 15,105,112 |
| 2,500,000 | 31,301,137 | 42,016,706 | 21,578,732 |
| 3,500,000 | 43,821,591 | 58,823,388 | 30,210,225 |

Use: if the repaired seqsafe96 result supports source breadth, this probe estimates the live FineWeb scanning scale and provides a larger candidate source pool for manual/independent_review quality reading and possible faithful-view prompt construction.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_core_scale_probe/live_fineweb_core_scale_probe_summary.json`

Sample accepted rows: `experiments/archive/representation_and_objectives/data/live_fineweb_core_scale_probe/live_fineweb_core_fact_scale_sample.json`

Sample rejected rows: `experiments/archive/representation_and_objectives/data/live_fineweb_core_scale_probe/live_fineweb_core_fact_scale_reject_sample.json`
