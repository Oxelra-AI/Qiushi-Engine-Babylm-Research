# earlier analysis symmetric margin row family

This file family repairs the binding-row design for local pilots: operation-count and entity-position features are balanced for op_count 1--4, while retention rows with 0--4 distractor operations make source retention explicit under operations.

## train_frame_seen
Rows 4860; pairs 2160; row words 346734; span errors 0; bad pairs 0.
Operation-count source fractions:
- k=0: n=540, P(source)=1.000, roles={'source_state': 540}
- k=1: n=1080, P(source)=0.500, roles={'source_state': 540, 'new_state': 540}
- k=2: n=1080, P(source)=0.500, roles={'source_state': 540, 'new_state': 540}
- k=3: n=1080, P(source)=0.500, roles={'source_state': 540, 'new_state': 540}
- k=4: n=1080, P(source)=0.500, roles={'source_state': 540, 'new_state': 540}
Query-first source fractions for operation-present rows:
- query_is_first=True: n=2160, P(source)=0.500, roles={'source_state': 1080, 'new_state': 1080}
- query_is_first=False: n=2160, P(source)=0.500, roles={'source_state': 1080, 'new_state': 1080}

## heldout_frame_all
Rows 2160; pairs 960; row words 152968; span errors 0; bad pairs 0.
Operation-count source fractions:
- k=0: n=240, P(source)=1.000, roles={'source_state': 240}
- k=1: n=480, P(source)=0.500, roles={'source_state': 240, 'new_state': 240}
- k=2: n=480, P(source)=0.500, roles={'source_state': 240, 'new_state': 240}
- k=3: n=480, P(source)=0.500, roles={'source_state': 240, 'new_state': 240}
- k=4: n=480, P(source)=0.500, roles={'source_state': 240, 'new_state': 240}
Query-first source fractions for operation-present rows:
- query_is_first=True: n=960, P(source)=0.500, roles={'source_state': 480, 'new_state': 480}
- query_is_first=False: n=960, P(source)=0.500, roles={'source_state': 480, 'new_state': 480}

