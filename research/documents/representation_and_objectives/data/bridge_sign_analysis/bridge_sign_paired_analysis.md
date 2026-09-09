# causal gauge experimental design bridge-sign paired analysis

## primary

### shared_trunk|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 0 | 192 | -0.9990
| direct_anchor_changed_same_init | 96 | 0 | 96 | -0.9990
| graph_transfer_changed | 192 | 0 | 192 | -0.9945
| graph_transfer_changed_same_init | 96 | 0 | 96 | -0.9945
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=1.0, de1 same/opp=128/0, acc bs+=1.0, acc bs-=1.0
**mixed_held_seen_orientation**: n=512, product_invariant=0.0, de1 same/opp=256/256, acc bs+=1.0, acc bs-=0.0

### tied|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 0 | 192 | -0.9812
| direct_anchor_changed_same_init | 96 | 0 | 96 | -0.9812
| graph_transfer_changed | 192 | 0 | 192 | -0.9703
| graph_transfer_changed_same_init | 96 | 0 | 96 | -0.9703
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=1.0, de1 same/opp=0/128, acc bs+=1.0, acc bs-=1.0
**mixed_held_seen_orientation**: n=512, product_invariant=0.0, de1 same/opp=256/256, acc bs+=1.0, acc bs-=0.0

### untied|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 0 | 192 | -0.9936
| direct_anchor_changed_same_init | 96 | 0 | 96 | -0.9936
| graph_transfer_changed | 192 | 144 | 48 | 0.4678
| graph_transfer_changed_same_init | 96 | 72 | 24 | 0.4678
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=1.0, de1 same/opp=128/0, acc bs+=1.0, acc bs-=1.0
**mixed_held_seen_orientation**: n=512, product_invariant=1.0, de1 same/opp=512/0, acc bs+=0.5, acc bs-=0.5

## bridge_only

### tied|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 0 | 192 | -0.9938
| direct_anchor_changed_same_init | 96 | 0 | 96 | -0.9938
| graph_transfer_changed | 192 | 192 | 0 | 1.0000
| graph_transfer_changed_same_init | 96 | 96 | 0 | 1.0000
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=0.25, de1 same/opp=64/64, acc bs+=0.5, acc bs-=0.25
**mixed_held_seen_orientation**: n=512, product_invariant=0.5, de1 same/opp=384/128, acc bs+=0.75, acc bs-=0.25

### untied|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 0 | 192 | -0.9938
| direct_anchor_changed_same_init | 96 | 0 | 96 | -0.9938
| graph_transfer_changed | 192 | 192 | 0 | 1.0000
| graph_transfer_changed_same_init | 96 | 96 | 0 | 1.0000
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=1.0, de1 same/opp=128/0, acc bs+=0.5, acc bs-=0.5
**mixed_held_seen_orientation**: n=512, product_invariant=1.0, de1 same/opp=512/0, acc bs+=0.5, acc bs-=0.5

## comparison_only

### tied|seed29000

#### State d_e sign reversal by category

| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_anchor_changed | 192 | 192 | 0 | 1.0000
| direct_anchor_changed_same_init | 96 | 96 | 0 | 1.0000
| graph_transfer_changed | 192 | 192 | 0 | 1.0000
| graph_transfer_changed_same_init | 96 | 96 | 0 | 1.0000
| unchanged | 384 | 384 | 0 | 1.0000

#### Comparison product analysis

**heldheld_unseen_edge_closure**: n=128, product_invariant=1.0, de1 same/opp=128/0, acc bs+=0.625, acc bs-=0.625
**mixed_held_seen_orientation**: n=512, product_invariant=1.0, de1 same/opp=512/0, acc bs+=0.125, acc bs-=0.125


## Scientific reading

Gauge transport predicts: (1) tied direct anchor d_e reverses (post-connector), (2) tied graph transfer d_e reverses, (3) held-held comparison products invariant (both flip → product unchanged), (4) mixed held-seen products reverse (held flips, seen stable), (5) untied and shared_trunk show no coherent d_e reversal in graph transfer relations, (6) bridge-only (no comparisons) shows direct anchor fit but no graph transfer reversal, (7) comparison-only shows no bridge_sign effect because bridge anchors are absent.
