# selector reader bridge synthesis and macro challenge selector-reader bridge — group-softmax selector repair

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

Best train top1: 1.0

Final top1: `{"by_eval_set": {"base_stable_anchor": 1.0, "base_stable_train": 1.0, "sparse_changed_focal": 1.0, "sparse_stable_focal": 1.0}, "by_query_family": {"focal_after": 1.0, "focal_before": 1.0, "secondary_after": 1.0, "secondary_before": 1.0}, "by_role_swap": {"False": 1.0}, "by_selector_qmode": {"role_inline": 1.0}, "groups": 2816, "top1": 1.0}`

Final binary: `{"by_query_family": {"focal_after": 0.9993489583333334, "focal_before": 0.9993489583333334, "secondary_after": 0.99921875, "secondary_before": 0.9984375}, "by_role_swap": {"False": 0.9991122159090909}, "by_selector_qmode": {"role_inline": 0.9991122159090909}, "by_train_kind": {"base_stable_anchor|ctx=inline_role|q=role_inline": 0.9990234375, "base_stable_train|ctx=inline_role|q=role_inline": 0.9990234375, "sparse_changed_focal|ctx=inline_role|q=role_inline": 1.0, "sparse_stable_focal|ctx=inline_role|q=role_inline": 1.0}, "pos_rate_pred": 0.2491122159090909, "pos_rate_true": 0.25, "row_acc": 0.9991122159090909}`

## Selector eval

| eval | top1 | focal_before | focal_after | secondary_before | secondary_after |
|---|---:|---:|---:|---:|---:|
| held_exact_nsA | 0.997 | 1.000 | 1.000 | 1.000 | 0.988 |
| held_exact_nsB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| held_para_nsA | 0.511 | 0.000 | 1.000 | 0.044 | 1.000 |
| held_role_swap_nsA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| trainChanged_exact_nsA | 0.998 | 1.000 | 1.000 | 1.000 | 0.994 |

## Composition eval

| eval | selector_top1 | comp_con | comp_fb | comp_fa | comp_sb | comp_sa | oracle_con | oracle_fb | oracle_fa | oracle_sb | oracle_sa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held_exact_nsA | 0.997 | 0.994 | 0.981 | 0.994 | 1.000 | 1.000 | 0.994 | 0.981 | 0.994 | 1.000 | 1.000 |
| held_exact_nsB | 1.000 | 0.995 | 0.994 | 0.988 | 1.000 | 1.000 | 0.995 | 0.994 | 0.988 | 1.000 | 1.000 |
| held_para_nsA | 0.511 | 0.750 | 0.006 | 0.994 | 1.000 | 1.000 | 0.994 | 0.981 | 0.994 | 1.000 | 1.000 |
| held_role_swap_nsA | 1.000 | 0.994 | 1.000 | 0.975 | 1.000 | 1.000 | 0.994 | 1.000 | 0.975 | 1.000 | 1.000 |
| trainChanged_exact_nsA | 0.998 | 0.992 | 0.988 | 0.981 | 1.000 | 1.000 | 0.992 | 0.988 | 0.981 | 1.000 | 1.000 |

## Boundary

Small pretrained selector-reader bridge with group-softmax selector. R is trained then held fixed; M is separate and sees no state labels; composition uses M-selected tags, with oracle-reader ceiling reported only as a reference.

## Selector objective repair

This run replaces the 3:1 binary candidate objective with a four-way group softmax over candidate tags. It directly optimizes selector top-1 and removes the all-negative row-accuracy shortcut observed in the first selector reader bridge synthesis and macro challenge run.
