# earlier analysis structural contrast-margin probe

Created: 2026-09-01T03:46:30Z
Seed: 42
Pool SHA256: `215944978157394d...`
Content SHA256: `6c677d6f6c161d8e...`
Total pairs: 1148 (948 structural + 200 surface controls)

## Family counts

| Family | Count | Plan weight |
|---|---:|---:|
| belief_report | 110 | 0.15 |
| directed_relation | 208 | 0.20 |
| entity_state | 260 | 0.30 |
| polarity_relation | 120 | 0.10 |
| surface_control | 200 | 0.00 |
| temporal_order | 250 | 0.25 |
| surface_control | 200 | -0.25 (L_t) |

## Sub-type counts

- belief_report/role_swap: 110
- directed_relation/comp_arg_swap: 58
- directed_relation/dir_arg_swap: 150
- entity_state/depth2_stale: 100
- entity_state/depth3_stale: 100
- entity_state/depth4_stale: 60
- polarity_relation/negation_removal: 120
- surface_control/word_replace_belief_report: 40
- surface_control/word_replace_directed_relation: 40
- surface_control/word_replace_entity_state: 40
- surface_control/word_replace_polarity_relation: 40
- surface_control/word_replace_temporal_order: 40
- temporal_order/clause_swap_after: 83
- temporal_order/clause_swap_and_then: 18
- temporal_order/clause_swap_before: 72
- temporal_order/clause_swap_first_then: 1
- temporal_order/clause_swap_then: 76

## Source balance

### belief_report
  - childes: 45
  - simple_wiki: 22
  - open_subtitles: 13
  - gutenberg: 13
  - qwen_pair_packed: 9
  - cleanqwen_fineweb_compact_view_reinvest: 6
  - bnc_spoken: 2
### directed_relation
  - simple_wiki: 80
  - qwen_pair_packed: 55
  - gutenberg: 30
  - cleanqwen_fineweb_compact_view_reinvest: 25
  - open_subtitles: 8
  - bnc_spoken: 5
  - childes: 5
### entity_state
  - controlled_template: 260
### polarity_relation
  - gutenberg: 34
  - qwen_pair_packed: 31
  - open_subtitles: 20
  - simple_wiki: 18
  - childes: 12
  - bnc_spoken: 3
  - cleanqwen_fineweb_compact_view_reinvest: 2
### surface_control
  - simple_wiki: 48
  - controlled_template: 40
  - qwen_pair_packed: 30
  - gutenberg: 23
  - open_subtitles: 22
  - childes: 18
  - cleanqwen_fineweb_compact_view_reinvest: 11
  - bnc_spoken: 8
### temporal_order
  - open_subtitles: 72
  - simple_wiki: 48
  - gutenberg: 40
  - qwen_pair_packed: 39
  - cleanqwen_fineweb_compact_view_reinvest: 30
  - bnc_spoken: 20
  - childes: 1

## Sample pairs

### entity_state
  - **es_d2_0001** (depth2_stale)
    Coherent: The hat is in the basket. Sam moves the hat to the bed. The hat is now on the bed.
    Perturbed: The hat is in the basket. Sam moves the hat to the bed. The hat is now in the basket.
  - **es_d2_0002** (depth2_stale)
    Coherent: The bag is in the closet. Bob puts the bag to the bed. The bag is now on the bed.
    Perturbed: The bag is in the closet. Bob puts the bag to the bed. The bag is now in the closet.
  - **es_d2_0003** (depth2_stale)
    Coherent: The card is in the basket. Bob moves the card to the counter. The card is now on the counter.
    Perturbed: The card is in the basket. Bob moves the card to the counter. The card is now in the basket.

### temporal_order
  - **temp_0001** (clause_swap_after)
    Coherent: She was not allowed to live, after her husband was dead..
    Perturbed: her husband was dead., after She was not allowed to live.
  - **temp_0002** (clause_swap_then)
    Coherent: But you can see if this lot gets converted to carbonate and, then that water then gets mixed down to the deep.
    Perturbed: that water then gets mixed down to the deep, then But you can see if this lot gets converted to carbonate and.
  - **temp_0003** (clause_swap_after)
    Coherent: The club secured promotion to the Promotion League, after defeating FC Paradiso in 2022..
    Perturbed: defeating FC Paradiso in 2022., after The club secured promotion to the Promotion League.

### directed_relation
  - **dr_comp_0001** (comp_arg_swap)
    Coherent: our doom is less piteous than thine.
    Perturbed: thine is less piteous than our doom.
  - **dr_comp_0002** (comp_arg_swap)
    Coherent: discovered that the boy is more resolute than he had expected.
    Perturbed: he had expected is more resolute than discovered that the boy.
  - **dr_comp_0003** (comp_arg_swap)
    Coherent: observed that the King is more important than the crown.
    Perturbed: the crown is more important than observed that the King.

### belief_report
  - **br_0001** (role_swap)
    Coherent: BRO handed Chi the wrong glasses] *CHI: no mine are pink.
    Perturbed: Chi handed BRO the wrong glasses] *CHI: no mine are pink.
  - **br_0002** (role_swap)
    Coherent: Stephanie gave Seth the first spot in.
    Perturbed: Seth gave Stephanie the first spot in.
  - **br_0003** (role_swap)
    Coherent: Holst wrote to Santhagens -Waller..
    Perturbed: Santhagens wrote to Holst -Waller..

### polarity_relation
  - **pol_0001** (negation_removal)
    Coherent: But they were not free as they had been before.
    Perturbed: But they were free as they had been before.
  - **pol_0002** (negation_removal)
    Coherent: for Governor again in the 2014 election but he did not make the ballot.
    Perturbed: for Governor again in the 2014 election but he did make the ballot.
  - **pol_0003** (negation_removal)
    Coherent: They thought that some groups of people did not even deserve to live.
    Perturbed: They thought that some groups of people did even deserve to live.

### surface_control
  - **surf_enti_0001** (word_replace_entity_state)
    Coherent: The cup is in the cupboard. Amy places the cup to the floor. The cup is now on the floor.
    Perturbed: The cup is in the third. Amy places the cup to the floor. The cup is now on the floor.
  - **surf_enti_0002** (word_replace_entity_state)
    Coherent: The coin is on the desk. Bob puts the coin to the box. The coin is now in the box.
    Perturbed: The dinner is on the desk. Bob puts the coin to the box. The coin is now in the box.
  - **surf_enti_0003** (word_replace_entity_state)
    Coherent: The sock is in the cupboard. The bottle is on the bench. Sam takes the sock to the shelf. Alice puts the bottle to the t
    Perturbed: The sock is in the cupboard. The bottle is on the bench. Sam takes the sock to the shelf. Alice puts the bottle to the t
