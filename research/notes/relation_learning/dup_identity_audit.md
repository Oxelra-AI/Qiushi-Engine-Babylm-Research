# dup identity audit DUP identity audit
This audit was run before reading any seed43122 DUP scores. It compares the stream launched in earlier analysis against the COMPACT_EXPERIENCE duplication arm used in the current report.
## Conclusion
The streams are **not the same construction**. earlier analysis launched `official_original_dup_100M` from COMPACT_EXPERIENCE earlier analysis, but the report's DUP values were scored from `selected_original_dup_all_100M` from COMPACT_EXPERIENCE frequency concentration probe. Therefore the earlier analysis DUP task is obsolete for the intended replication and must not be reported as a DUP replication.
## Metadata comparison
- earlier analysis selected duplicate pairs: 36687
- frequency concentration probe selected pairs: 37594
- earlier analysis duplicate-pair words: 1656800
- frequency concentration probe duplicate-pair words: 1698026
- Pair boundary preserved: earlier analysis=True, frequency concentration probe=True
- Pair truncation: earlier analysis=False, frequency concentration probe=False
- frequency concentration probe keeps all earlier analysis selected pair ids: True
- Pool SHA256: earlier analysis `c6ac969b46103cd342606ef8915f6e7b6610a6d3ec0e1b9db7ab97c3833ba634`, frequency concentration probe `7ab08ee9db11b618b80f73e98d581d4f024f2b0f82f15405572f04f0f0e6dad7`
- Training SHA256: earlier analysis `77838bcbffe0e467ce339a6ffd89abe2d81ab124ed4d5c2b75835fad95d5cb2e`, frequency concentration probe `b33c57e960ed2b0519f782d44c88d9ec8cdf7344710c0f525256d1050c3f582a`
## Stream scan of 10M pools
- earlier analysis pool rows=64392, packed pair rows=12247, unique pair_ids=36687, pair-id occurrences=36687, total words=10000000
- frequency concentration probe pool rows=64438, packed pair rows=12550, unique pair_ids=37594, pair-id occurrences=37594 (from sidecar row_meta), total words=10000000
- Pair-id intersection=36687; missing from earlier analysis=907; extra in earlier analysis=0; identical sets=False
## Scientific implication
The correct two-seed duplication replication must train on `experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_100M.jsonl` with metadata `selected_original_dup_all_metadata.json`. The earlier analysis `official_original_dup` run, if retained at all, is a separate one-seed exact-recurrence intervention and not evidence for the pre-stated DUP replication.

Full JSON: `experiments/archive/relation_learning/data/dup_identity_audit/dup_identity_audit.json`
