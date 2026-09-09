# selector reader bridge synthesis and macro challenge selector-reader bridge

Purpose: train a direct address reader R, train a separate role-to-tag selector M with explicit candidate tags, then compose M and frozen R without oracle routing.

## Construction

Reader train rows: 5632; selector train rows: 11264.

Selector group audit: `{"held_exact_nsA": {"group_size_max": 4, "group_size_min": 4, "groups": 640, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 2560}, "held_exact_nsB": {"group_size_max": 4, "group_size_min": 4, "groups": 640, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 2560}, "held_para_nsA": {"group_size_max": 4, "group_size_min": 4, "groups": 640, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 2560}, "held_role_swap_nsA": {"group_size_max": 4, "group_size_min": 4, "groups": 640, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 2560}, "train": {"group_size_max": 4, "group_size_min": 4, "groups": 2816, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 11264}, "trainChanged_exact_nsA": {"group_size_max": 4, "group_size_min": 4, "groups": 640, "positive_per_group_max": 1, "positive_per_group_min": 1, "rows": 2560}}`

Token audit: `{"frac_gt_max_len": 0.0, "max": 110, "min": 78, "n": 40448, "p50": 96.0, "p90": 101.0}`

## Reader R fit

### reader_tag_pretrain

Best train acc: 0.9996448863636364

Fit: `{"base_stable_anchor|ctx=tag_only|q=direct_tag": 0.999609375, "base_stable_train|ctx=tag_only|q=direct_tag": 1.0, "sparse_changed_focal|ctx=tag_only|q=direct_tag": 0.99609375, "sparse_stable_focal|ctx=tag_only|q=direct_tag": 1.0}`

### reader_inline_direct

Best train acc: 1.0

Fit: `{"base_stable_anchor|ctx=inline_role|q=direct_tag": 1.0, "base_stable_train|ctx=inline_role|q=direct_tag": 1.0, "sparse_changed_focal|ctx=inline_role|q=direct_tag": 1.0, "sparse_stable_focal|ctx=inline_role|q=direct_tag": 1.0}`

## Selector M fit

Best train top1: 0.2531960227272727

Final top1: `{"by_eval_set": {"base_stable_anchor": 0.2515625, "base_stable_train": 0.25546875, "sparse_changed_focal": 0.25, "sparse_stable_focal": 0.25}, "by_query_family": {"focal_after": 0.24088541666666666, "focal_before": 0.2526041666666667, "secondary_after": 0.2640625, "secondary_before": 0.2578125}, "by_role_swap": {"False": 0.2531960227272727}, "by_selector_qmode": {"role_inline": 0.2531960227272727}, "groups": 2816, "top1": 0.2531960227272727}`

Final binary: `{"by_query_family": {"focal_after": 0.75, "focal_before": 0.75, "secondary_after": 0.75, "secondary_before": 0.75}, "by_role_swap": {"False": 0.75}, "by_selector_qmode": {"role_inline": 0.75}, "by_train_kind": {"base_stable_anchor|ctx=inline_role|q=role_inline": 0.75, "base_stable_train|ctx=inline_role|q=role_inline": 0.75, "sparse_changed_focal|ctx=inline_role|q=role_inline": 0.75, "sparse_stable_focal|ctx=inline_role|q=role_inline": 0.75}, "pos_rate_pred": 0.0, "pos_rate_true": 0.25, "row_acc": 0.75}`

## Selector eval

| eval | top1 | focal_before | focal_after | secondary_before | secondary_after |
|---|---:|---:|---:|---:|---:|
| held_exact_nsA | 0.252 | 0.225 | 0.212 | 0.306 | 0.263 |
| held_exact_nsB | 0.253 | 0.250 | 0.312 | 0.250 | 0.200 |
| held_para_nsA | 0.253 | 0.219 | 0.219 | 0.319 | 0.256 |
| held_role_swap_nsA | 0.253 | 0.237 | 0.225 | 0.250 | 0.300 |
| trainChanged_exact_nsA | 0.245 | 0.256 | 0.231 | 0.281 | 0.212 |

## Composition eval

| eval | selector_top1 | comp_con | comp_fb | comp_fa | comp_sb | comp_sa | oracle_con | oracle_fb | oracle_fa | oracle_sb | oracle_sa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held_exact_nsA | 0.252 | 0.750 | 0.487 | 0.512 | 1.000 | 1.000 | 0.994 | 0.981 | 0.994 | 1.000 | 1.000 |
| held_exact_nsB | 0.253 | 0.753 | 0.463 | 0.550 | 1.000 | 1.000 | 0.995 | 0.994 | 0.988 | 1.000 | 1.000 |
| held_para_nsA | 0.253 | 0.747 | 0.481 | 0.506 | 1.000 | 1.000 | 0.994 | 0.981 | 0.994 | 1.000 | 1.000 |
| held_role_swap_nsA | 0.253 | 0.752 | 0.537 | 0.469 | 1.000 | 1.000 | 0.994 | 1.000 | 0.975 | 1.000 | 1.000 |
| trainChanged_exact_nsA | 0.245 | 0.747 | 0.506 | 0.481 | 1.000 | 1.000 | 0.992 | 0.988 | 0.981 | 1.000 | 1.000 |

## Boundary

Small pretrained selector-reader bridge. R is trained then held fixed; M is separate; composition uses M-selected tags, with oracle-reader ceiling reported only as a reference.
