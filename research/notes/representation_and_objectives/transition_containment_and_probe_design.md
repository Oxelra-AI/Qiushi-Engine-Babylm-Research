# transition containment and probe design — transition asset containment and future probe design

## What was measured

The anchor matched controls ultra-clean transition/control asset was checked against three legal 10M FW corpora: compact, row-block breadth, and interleaved breadth. Exact containment means the sentence text appears as a substring of a row in that 10M corpus. This matters because a later warm-start probe can remain inside a legal 10M word set only if it reweights text already present in the checkpoint's training pool.

## Main counts

Stored word fields in the anchor matched controls JSONL are not all identical to Python whitespace counts, so this transition containment and probe design output uses actual `len(text.split())` for any future training-compatible accounting and records the mismatch explicitly.

| asset | actual words | stored field sum | field mismatches | in compact | in row-block breadth | in interleaved breadth |
|---|---:|---:|---:|---:|---:|---:|
| transition | 29903 | 30000 | 125 (-97) | 25055 | 26304 | 26304 |
| anchor_control | 29943 | 29958 | 132 (-15) | 24976 | 26422 | 26422 |

Compact-contained matched word cap: **24976** words.
Contained in all three FW corpora matched word cap: **24616** words.

## Scientific reading

Most of the useful transition/control material is inherited from the current compact-compatible pool rather than being novel external text. A future probe should therefore use a legal warm-start contrast only by changing relative exposure inside the original 10M pool. It should not continue a completed 100M endpoint, and it should not add non-contained sentences to an existing checkpoint.

## Future minimum-cost probe, if FW allocation fails

- Use a compact-family intermediate checkpoint, preferably chck_70M or chck_80M, not chck_100M, so the total exposure can still end at exactly 100M.
- Construct two remaining-exposure streams with identical total words and identical common removals: one repeats the compact-contained transition subset, the other repeats the compact-contained anchor-control subset.
- Use only sentences exact-contained in the checkpoint's 10M pool; otherwise the warm-start branch would add new unique text beyond the pool used before the checkpoint.
- Interpret the result as a relative-exposure probe, not as a final data recipe: it tests whether explicit physical/spatial/quantity transitions can move EWoK four-cell and GlobalPIQA hard-row ranks without the broad damage seen in row-block breadth.
- Read cheap7, GlobalPIQA all-option margins, and EWoK four-cell before any full official endpoint work. A useful signal is compact-level broad columns plus reduced hard-row margins; an EWoK-only rise with GlobalPIQA or Reading loss repeats the earlier relation-data failure pattern.

## Files

- json: `experiments/archive/representation_and_objectives/data/transition_containment/transition_containment_and_probe_design.json`
- sentence_csv: `experiments/archive/representation_and_objectives/data/transition_containment/transition_control_sentence_containment.csv`
- transition_exact_in_compact: `experiments/archive/representation_and_objectives/data/transition_containment/transition_exact_in_compact.jsonl`
- anchor_control_exact_in_compact: `experiments/archive/representation_and_objectives/data/transition_containment/anchor_control_exact_in_compact.jsonl`
- note: `research/notes/representation_and_objectives/transition_containment_and_probe_design.md`

## Important correction to the anchor matched controls asset (do not skip)

The anchor matched controls v3 30k treatment/control pair was matched on **stored word fields** and on the **full 882-sentence pool**, but a legal warm-start probe can only use the compact-contained, whitespace-verified subset. Those are not the same set, so the anchor matched controls "tight match" does not survive intact:

- Actual whitespace words differ from stored fields: transition 29,903 actual vs 30,000 stored (125 sentences off, net -97 words); anchor_control 29,943 actual vs 29,958 stored (132 sentences off, net -15 words). The BabyLM trainers verify `len(text.split())`, so the actual counts govern.
- Compact-contained subset is unbalanced at the sentence level: transition 571 sentences / 25,055 words vs anchor_control 574 sentences / 24,976 words. The anchor matched controls match on source/length/required-capability was computed on the full pool, not on this compact-contained subset.
- Row-block and interleaved breadth contain slightly more (602/602 sentences, 26,304 words treatment; 611/611, 26,422 control), because those arms carry the whole-sentence breadth companions that the compact arm shortens.

**Consequence for any future probe:** a matched treatment/control pair must be **re-derived on the compact-contained, whitespace-exact subset** (`transition_exact_in_compact.jsonl`, `anchor_control_exact_in_compact.jsonl`), then re-matched on source label, length bin, and required capability at that subset size, with identical actual word totals. Do not reuse the anchor matched controls 30k files as a ready matched pair.

## What this establishes and what it does not

- It establishes that a legal in-pool relative-exposure probe is **feasible** without adding new unique training words: ~24,616 words of transition material and a comparable control set are exact-contained in all three FW 10M corpora, so a warm-start from an intermediate FW checkpoint can reweight text already inside that checkpoint's pool.
- It does **not** establish that the probe is worth running. That decision is still gated by the pending FW endpoints: `s112_t16_tool1` (EWoK four-cell for compact and row-block breadth) and `s106_t10_tool1` (interleaved breadth training). If those show that a single FW arm can combine compact-level broad columns with breadth-level hard-relation rank movement, the FW allocation route is the live SOTA-facing line and this transition probe stays a fallback. If neither does, this compact-contained matched probe is the cheapest next mechanism test — but it must be re-derived as above, not launched from the anchor matched controls files.
- The transition material is dominated by `compact_experience_aligned_10m_rows` (25,058 of 29,903 treatment words), i.e. Gutenberg/OpenSubtitles narrative rather than FineWeb world-knowledge text. This is a caution: a physical/spatial/quantity-transition signal drawn mostly from literary narrative may not transfer to GlobalPIQA/EWoK world relations, and could repeat the INITIAL_MODEL_STUDIES pattern of moving EWoK while failing binding/GlobalPIQA. Any future probe should track that source composition explicitly.
