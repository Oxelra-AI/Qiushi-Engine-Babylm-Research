# fw 100m cheap7 synthesis — FW three-arm GlobalPIQA row/margin synthesis

## parallel

Common rows: 103; oracle-union accuracy across three arms: 40.78; all-wrong rows: 61; all-correct rows: 17.

- compact: acc=24.27, correct=25, rank_counts={'3': 30, '2': 24, '4': 24, '1': 25}, mean top-minus-correct=1.059.
- rowblock: acc=29.13, correct=30, rank_counts={'4': 17, '3': 30, '1': 30, '2': 26}, mean top-minus-correct=0.862.
- interleaved: acc=26.21, correct=27, rank_counts={'3': 28, '2': 25, '4': 23, '1': 27}, mean top-minus-correct=0.984.
- correctness patterns compact/rowblock/interleaved: {'000': 61, '111': 17, '010': 8, '001': 6, '110': 2, '011': 3, '100': 5, '101': 1}
- rowblock vs compact: better/same/worse rank=34/50/19; lower/higher margin=49/35; mean margin delta=-0.198.
- interleaved vs compact: better/same/worse rank=24/59/20; lower/higher margin=50/35; mean margin delta=-0.076.
- layout disagreement: {'rowblock_better_rank_interleaved_not': 19, 'interleaved_better_rank_rowblock_not': 9, 'both_better_rank_than_compact': 15}

## nonparallel

Common rows: 100; oracle-union accuracy across three arms: 68.00; all-wrong rows: 32; all-correct rows: 36.

- compact: acc=53.00, correct=53, rank_counts={'1': 53, '2': 47}, mean top-minus-correct=0.240.
- rowblock: acc=45.00, correct=45, rank_counts={'1': 45, '2': 55}, mean top-minus-correct=0.223.
- interleaved: acc=56.00, correct=56, rank_counts={'1': 56, '2': 44}, mean top-minus-correct=0.209.
- correctness patterns compact/rowblock/interleaved: {'111': 36, '011': 5, '101': 7, '100': 8, '000': 32, '110': 2, '001': 8, '010': 2}
- rowblock vs compact: better/same/worse rank=7/78/15; lower/higher margin=32/30; mean margin delta=-0.017.
- interleaved vs compact: better/same/worse rank=13/77/10; lower/higher margin=31/26; mean margin delta=-0.030.
- layout disagreement: {'rowblock_better_rank_interleaved_not': 2, 'interleaved_better_rank_rowblock_not': 8, 'both_better_rank_than_compact': 5}

Files: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_threearm_synthesis/fw_globalpiqa_threearm_synthesis.json`
