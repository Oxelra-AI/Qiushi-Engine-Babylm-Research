# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 2560 with labels {'0': 1280, '1': 1280}

- exposure: rows=512 queries={'neutral_mention': 512} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- stable_only: rows=512 queries={'focal_after': 128, 'focal_before': 128, 'neutral_mention': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- changed_only: rows=512 queries={'focal_after': 128, 'focal_before': 128, 'neutral_mention': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}

