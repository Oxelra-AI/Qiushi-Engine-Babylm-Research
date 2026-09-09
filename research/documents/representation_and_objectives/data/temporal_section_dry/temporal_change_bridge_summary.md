# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 5120 with labels {'0': 2560, '1': 2560}

- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- oracle_secondary: rows=1024 queries={'focal_after': 256, 'focal_before': 256, 'secondary_after': 256, 'secondary_before': 256} labels={'0': 512, '1': 512} changed={'False': 512, 'True': 512}

