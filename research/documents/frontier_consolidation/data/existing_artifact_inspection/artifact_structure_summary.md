# earlier analysis existing artifact structure inspection

## eval_ewok
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered/physical-dynamics.jsonl`
- exists: True size: 34220
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 10,
      "keys": [
        "Domain",
        "ConceptA",
        "ConceptB",
        "ContextType",
        "ContextDiff",
        "TargetDiff",
        "Context1",
        "Context2",
        "Target1",
        "Target2"
      ],
      "samples": [
        {
          "key": "Domain",
          "value": {
            "type": "str",
            "repr": "'physical-dynamics'"
          }
        },
        {
          "key": "ConceptA",
          "value": {
            "type": "str",
            "repr": "'accelerate'"
          }
        },
        {
          "key": "ConceptB",
          "value": {
            "type": "str",
            "repr": "'slow down'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 10,
      "keys": [
        "Domain",
        "ConceptA",
        "ConceptB",
        "ContextType",
        "ContextDiff",
        "TargetDiff",
        "Context1",
        "Context2",
        "Target1",
        "Target2"
      ],
      "samples": [
        {
          "key": "Domain",
          "value": {
            "type": "str",
            "repr": "'physical-dynamics'"
          }
        },
        {
          "key": "ConceptA",
          "value": {
            "type": "str",
            "repr": "'accelerate'"
          }
        },
        {
          "key": "ConceptB",
          "value": {
            "type": "str",
            "repr": "'slow down'"
          }
        }
      ]
    }
  ]
}
```

## eval_blimp
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered/wh_island.jsonl`
- exists: True size: 321273
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 10,
      "keys": [
        "sentence_good",
        "sentence_bad",
        "field",
        "linguistics_term",
        "UID",
        "simple_LM_method",
        "one_prefix_method",
        "two_prefix_method",
        "lexically_identical",
        "pair_id"
      ],
      "samples": [
        {
          "key": "sentence_good",
          "value": {
            "type": "str",
            "repr": "'Who have those men revealed they helped?'"
          }
        },
        {
          "key": "sentence_bad",
          "value": {
            "type": "str",
            "repr": "'Who have those men revealed who helped?'"
          }
        },
        {
          "key": "field",
          "value": {
            "type": "str",
            "repr": "'syntax'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 10,
      "keys": [
        "sentence_good",
        "sentence_bad",
        "field",
        "linguistics_term",
        "UID",
        "simple_LM_method",
        "one_prefix_method",
        "two_prefix_method",
        "lexically_identical",
        "pair_id"
      ],
      "samples": [
        {
          "key": "sentence_good",
          "value": {
            "type": "str",
            "repr": "\"Who isn't Craig realizing he kisses?\""
          }
        },
        {
          "key": "sentence_bad",
          "value": {
            "type": "str",
            "repr": "\"Who isn't Craig realizing who kisses?\""
          }
        },
        {
          "key": "field",
          "value": {
            "type": "str",
            "repr": "'syntax'"
          }
        }
      ]
    }
  ]
}
```

## eval_supplement
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_tricky.jsonl`
- exists: True size: 31876
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 5,
      "keys": [
        "row",
        "fragment",
        "contrast",
        "sentence_good",
        "sentence_bad"
      ],
      "samples": [
        {
          "key": "row",
          "value": {
            "type": "int",
            "repr": "0"
          }
        },
        {
          "key": "fragment",
          "value": {
            "type": "str",
            "repr": "'sentence'"
          }
        },
        {
          "key": "contrast",
          "value": {
            "type": "str",
            "repr": "'animate vs. inanimate (tricky)'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 5,
      "keys": [
        "row",
        "fragment",
        "contrast",
        "sentence_good",
        "sentence_bad"
      ],
      "samples": [
        {
          "key": "row",
          "value": {
            "type": "int",
            "repr": "0"
          }
        },
        {
          "key": "fragment",
          "value": {
            "type": "str",
            "repr": "'fragment'"
          }
        },
        {
          "key": "contrast",
          "value": {
            "type": "str",
            "repr": "'animate vs. inanimate (tricky)'"
          }
        }
      ]
    }
  ]
}
```

## eval_entity
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/entity_tracking/regular.jsonl`
- exists: True size: 2369451
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 5,
      "keys": [
        "sample_id",
        "numops",
        "input_prefix",
        "options",
        "example_id"
      ],
      "samples": [
        {
          "key": "sample_id",
          "value": {
            "type": "int",
            "repr": "1521"
          }
        },
        {
          "key": "numops",
          "value": {
            "type": "int",
            "repr": "0"
          }
        },
        {
          "key": "input_prefix",
          "value": {
            "type": "str",
            "repr": "'Box 0 contains the beer and the map, Box 1 contains the hat and the painting and the shell, Box 2 contains nothing, Box 3 contains the bone and the drink and the mirror, Box 4 contains the clock and the coat and the key, Box 5 contains the dress, Box 6 contains the newspaper and the picture and the"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 5,
      "keys": [
        "sample_id",
        "numops",
        "input_prefix",
        "options",
        "example_id"
      ],
      "samples": [
        {
          "key": "sample_id",
          "value": {
            "type": "int",
            "repr": "1451"
          }
        },
        {
          "key": "numops",
          "value": {
            "type": "int",
            "repr": "0"
          }
        },
        {
          "key": "input_prefix",
          "value": {
            "type": "str",
            "repr": "'Box 0 contains the ball and the tape and the television, Box 1 contains the clock and the cross, Box 2 contains the paper and the rose, Box 3 contains the key, Box 4 contains the drug and the tea, Box 5 contains the branch and the card, Box 6 contains the gift. Box 3 contains '"
          }
        }
      ]
    }
  ]
}
```

## eval_comps
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/comps/comps_base.jsonl`
- exists: True size: 14215247
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 9,
      "keys": [
        "id",
        "property",
        "acceptable_concept",
        "unacceptable_concept",
        "prefix_acceptable",
        "property_phrase",
        "prefix_unacceptable",
        "negative_sample_type",
        "similarity"
      ],
      "samples": [
        {
          "key": "id",
          "value": {
            "type": "int",
            "repr": "1"
          }
        },
        {
          "key": "property",
          "value": {
            "type": "str",
            "repr": "'absorbs sweat'"
          }
        },
        {
          "key": "acceptable_concept",
          "value": {
            "type": "str",
            "repr": "'sock'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 9,
      "keys": [
        "id",
        "property",
        "acceptable_concept",
        "unacceptable_concept",
        "prefix_acceptable",
        "property_phrase",
        "prefix_unacceptable",
        "negative_sample_type",
        "similarity"
      ],
      "samples": [
        {
          "key": "id",
          "value": {
            "type": "int",
            "repr": "2"
          }
        },
        {
          "key": "property",
          "value": {
            "type": "str",
            "repr": "'absorbs sweat'"
          }
        },
        {
          "key": "acceptable_concept",
          "value": {
            "type": "str",
            "repr": "'sock'"
          }
        }
      ]
    }
  ]
}
```

## eval_globalpiqa_parallel
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl`
- exists: True size: 59276
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 15,
      "keys": [
        "prompt",
        "solution0",
        "solution1",
        "solution2",
        "solution3",
        "label",
        "language",
        "eng_prompt",
        "eng_solution0",
        "eng_solution1",
        "eng_solution2",
        "eng_solution3",
        "categories",
        "example_id",
        "supplement"
      ],
      "samples": [
        {
          "key": "prompt",
          "value": {
            "type": "str",
            "repr": "'A plastic bag is filled with air and then sealed. When an object is placed on the bag, what happens?'"
          }
        },
        {
          "key": "solution0",
          "value": {
            "type": "str",
            "repr": "'The amount of air in the bag decreases'"
          }
        },
        {
          "key": "solution1",
          "value": {
            "type": "str",
            "repr": "'The amount of air in the bag increases'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 15,
      "keys": [
        "prompt",
        "solution0",
        "solution1",
        "solution2",
        "solution3",
        "label",
        "language",
        "eng_prompt",
        "eng_solution0",
        "eng_solution1",
        "eng_solution2",
        "eng_solution3",
        "categories",
        "example_id",
        "supplement"
      ],
      "samples": [
        {
          "key": "prompt",
          "value": {
            "type": "str",
            "repr": "'A cup is placed with the opening on the table. When I pour water over the cup, what happens to the water?'"
          }
        },
        {
          "key": "solution0",
          "value": {
            "type": "str",
            "repr": "'The water does not touch the cup'"
          }
        },
        {
          "key": "solution1",
          "value": {
            "type": "str",
            "repr": "'The water goes into the cup'"
          }
        }
      ]
    }
  ]
}
```

## eval_globalpiqa_nonparallel
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl`
- exists: True size: 45480
```json
{
  "row_count_first_counted": 3,
  "samples": [
    {
      "type": "dict",
      "len": 11,
      "keys": [
        "prompt",
        "solution0",
        "solution1",
        "label",
        "language",
        "eng_translated0",
        "eng_translated1",
        "approx_cultural_score",
        "llm_used",
        "example_id",
        "supplement"
      ],
      "samples": [
        {
          "key": "prompt",
          "value": {
            "type": "str",
            "repr": "'To serve condiments at a BBQ,'"
          }
        },
        {
          "key": "solution0",
          "value": {
            "type": "str",
            "repr": "'use a waffle iron for the various condiments.'"
          }
        },
        {
          "key": "solution1",
          "value": {
            "type": "str",
            "repr": "'use a muffin tin for the various condiments.'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 11,
      "keys": [
        "prompt",
        "solution0",
        "solution1",
        "label",
        "language",
        "eng_translated0",
        "eng_translated1",
        "approx_cultural_score",
        "llm_used",
        "example_id",
        "supplement"
      ],
      "samples": [
        {
          "key": "prompt",
          "value": {
            "type": "str",
            "repr": "'To celebrate Christmas without room for a tree,'"
          }
        },
        {
          "key": "solution0",
          "value": {
            "type": "str",
            "repr": "'tape holiday lights to the wall in pattern of a tree.'"
          }
        },
        {
          "key": "solution1",
          "value": {
            "type": "str",
            "repr": "'tape holiday treats to the wall in pattern of a tree.'"
          }
        }
      ]
    }
  ]
}
```

## eval_reading
- path: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv`
- exists: True size: 615013
```json
{
  "fieldnames": [
    "",
    "item",
    "word",
    "word2",
    "sentence",
    "context_length",
    "sent_id",
    "item_id",
    "list",
    "Subtlex_log10",
    "length",
    "RTfirstfix",
    "RTfirstpass",
    "RTgopast",
    "RTrightbound",
    "self_paced_reading_time",
    "ELAN",
    "LAN",
    "N400",
    "P600",
    "EPNP",
    "PNP",
    "prev_word",
    "prev_item",
    "prev_length"
  ],
  "samples": [
    {
      "type": "dict",
      "len": 25,
      "keys": [
        "",
        "item",
        "word",
        "word2",
        "sentence",
        "context_length",
        "sent_id",
        "item_id",
        "list",
        "Subtlex_log10",
        "length",
        "RTfirstfix",
        "RTfirstpass",
        "RTgopast",
        "RTrightbound",
        "self_paced_reading_time",
        "ELAN",
        "LAN",
        "N400",
        "P600"
      ],
      "samples": [
        {
          "key": "",
          "value": {
            "type": "str",
            "repr": "'0'"
          }
        },
        {
          "key": "item",
          "value": {
            "type": "str",
            "repr": "'Arthur placed'"
          }
        },
        {
          "key": "word",
          "value": {
            "type": "str",
            "repr": "'the'"
          }
        }
      ]
    },
    {
      "type": "dict",
      "len": 25,
      "keys": [
        "",
        "item",
        "word",
        "word2",
        "sentence",
        "context_length",
        "sent_id",
        "item_id",
        "list",
        "Subtlex_log10",
        "length",
        "RTfirstfix",
        "RTfirstpass",
        "RTgopast",
        "RTrightbound",
        "self_paced_reading_time",
        "ELAN",
        "LAN",
        "N400",
        "P600"
      ],
      "samples": [
        {
          "key": "",
          "value": {
            "type": "str",
            "repr": "'1'"
          }
        },
        {
          "key": "item",
          "value": {
            "type": "str",
            "repr": "'Finally Maria sat down with a'"
          }
        },
        {
          "key": "word",
          "value": {
            "type": "str",
            "repr": "'cup'"
          }
        }
      ]
    }
  ]
}
```

## pred_blimp70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/BLiMP/chck_70M/full_complianttok_reinvest_seed43022_70M_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json`
- exists: True size: 5991001
```json
{
  "type": "dict",
  "len": 67,
  "keys": [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "coordinate_structure_constraint_complex_left_branch",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "distractor_agreement_relational_noun",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adjective_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "drop_argument",
    "distractor_agreement_relative_clause"
  ],
  "samples": [
    {
      "key": "adjunct_island",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'adjunct_island_0', 'pred': 'Who should Derek hug Richard after shocking?'}, {'id': 'adjunct_island_1', 'pred': 'What had Theresa walked through while talking about that high school?'}, {'id': 'adjunct_island_2', 'pred': 'Who will Katherine discover without hiring Erin?'}, {'id': 'adjunct_is"
          }
        ]
      }
    },
    {
      "key": "anaphor_gender_agreement",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'anaphor_gender_agreement_0', 'pred': \"Katherine can't help herself.\"}, {'id': 'anaphor_gender_agreement_1', 'pred': \"Marie won't think about herself.\"}, {'id': 'anaphor_gender_agreement_2', 'pred': \"Mark hasn't discussed itself.\"}, {'id': 'anaphor_gender_agreement_3', 'pred': 'Stephen impre"
          }
        ]
      }
    },
    {
      "key": "anaphor_number_agreement",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'anaphor_number_agreement_0', 'pred': 'Susan revealed herself.'}, {'id': 'anaphor_number_agreement_1', 'pred': \"Renee hasn't hurt herself.\"}, {'id': 'anaphor_number_agreement_2', 'pred': 'These patients do respect themselves.'}, {'id
```

## pred_ewok70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/EWoK/chck_70M/full_complianttok_reinvest_seed43022_70M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json`
- exists: True size: 930633
```json
{
  "type": "dict",
  "len": 11,
  "keys": [
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations"
  ],
  "samples": [
    {
      "key": "agent-properties",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'agent-properties_0', 'pred': 'Ali is in the bakery. Ali sees the candle inside. Ali believes that the candle is in the bakery.'}, {'id': 'agent-properties_1', 'pred': 'Ali is in the bakery. Ali sees the candle inside. Ali doubts that the candle is in the bakery.'}, {'id': 'agent-properties_"
          }
        ]
      }
    },
    {
      "key": "material-dynamics",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'material-dynamics_0', 'pred': 'Ali sees something that is liquid. It breaks.'}, {'id': 'material-dynamics_1', 'pred': 'Ali sees something that is liquid. It drips.'}, {'id': 'material-dynamics_2', 'pred': 'Chao sees something that is liquid. It breaks.'}, {'id': 'material-dynamics_3', 'pred"
          }
        ]
      }
    },
    {
      "key": "material-properties",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'material-properties_0', 'pred': 'Ali dropped the volleyball, and saw it lying on the floor. The volleyball is bouncy.'}, {'id': 'material-properties_1', 'pred': 'Ali dropped the volleyball, and saw it jumping off the floor. The volleyball is not bouncy.'}, {'id': 'material-properties_2', 'p"
          }
        ]
      }
    }
  ]
}
```

## pred_entity70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/Entity/chck_70M/full_complianttok_reinvest_seed43022_70M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json`
- exists: True size: 468683
```json
{
  "type": "dict",
  "len": 18,
  "keys": [
    "ambiref_0_ops",
    "ambiref_1_ops",
    "ambiref_2_ops",
    "ambiref_3_ops",
    "ambiref_4_ops",
    "ambiref_5_ops",
    "regular_0_ops",
    "regular_1_ops",
    "regular_2_ops",
    "regular_3_ops",
    "regular_4_ops",
    "regular_5_ops",
    "move_contents_0_ops",
    "move_contents_1_ops",
    "move_contents_2_ops",
    "move_contents_3_ops",
    "move_contents_4_ops",
    "move_contents_5_ops"
  ],
  "samples": [
    {
      "key": "ambiref_0_ops",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'ambiref_0_ops_0', 'pred': 'the red shirt and the yellow coat and the blue jacket.'}, {'id': 'ambiref_0_ops_1', 'pred': 'the yellow meat.'}, {'id': 'ambiref_0_ops_2', 'pred': 'the green letter and the yellow book and the blue machine.'}, {'id': 'ambiref_0_ops_3', 'pred': 'the red engine.'}, "
          }
        ]
      }
    },
    {
      "key": "ambiref_1_ops",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'ambiref_1_ops_0', 'pred': 'the blue book and the red drug.'}, {'id': 'ambiref_1_ops_1', 'pred': 'the big note.'}, {'id': 'ambiref_1_ops_2', 'pred': 'the blue cup.'}, {'id': 'ambiref_1_ops_3', 'pred': 'the green cup and the yellow knife and the yellow bell.'}, {'id': 'ambiref_1_ops_4', 'pred"
          }
        ]
      }
    },
    {
      "key": "ambiref_2_ops",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'ambiref_2_ops_0', 'pred': 'the blue clock.'}, {'id': 'ambiref_2_ops_1', 'pred': 'the big drink.'}, {'id': 'ambiref_2_ops_2', 'pred': 'the small glass and the blue fan.'}, {'id': 'ambiref_2_ops_3', 'pred': 'the big bottle.'}, {'id': 'ambiref_2_ops_4', 'pred': 'the big television and the blue"
          }
        ]
      }
    }
  ]
}
```

## pred_comps70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/COMPS/chck_70M/full_complianttok_reinvest_seed43022_70M_COMPS/zero_shot/mlm/comps/comps/predictions.json`
- exists: True size: 7597962
```json
{
  "type": "dict",
  "len": 4,
  "keys": [
    "wugs",
    "wugs_dist_in_between",
    "wugs_dist_before",
    "base"
  ],
  "samples": [
    {
      "key": "wugs",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'wugs_0', 'pred': 'A wug is a mussel. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_1', 'pred': 'A wug is a mussel. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_2', 'pred': 'A blicket is a mussel. Therefore, a blicket attaches to rocks.'}, {'id': 'wugs_3', 'pred': 'A dax is a pi"
          }
        ]
      }
    },
    {
      "key": "wugs_dist_in_between",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'wugs_dist_in_between_0', 'pred': 'A wug is a mussel. A fep is a clam. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_dist_in_between_1', 'pred': 'A wug is a mussel. A fep is a scallop. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_dist_in_between_2', 'pred': 'A blicket is a musse"
          }
        ]
      }
    },
    {
      "key": "wugs_dist_before",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'wugs_dist_before_0', 'pred': 'A fep is a clam. A wug is a mussel. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_dist_before_1', 'pred': 'A fep is a scallop. A wug is a mussel. Therefore, a wug attaches to rocks.'}, {'id': 'wugs_dist_before_2', 'pred': 'A dax is a crayfish. A blicket i"
          }
        ]
      }
    }
  ]
}
```

## pred_globalpiqa70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/GlobalPIQA_parallel/chck_70M/full_complianttok_reinvest_seed43022_70M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json`
- exists: True size: 12910
```json
{
  "type": "dict",
  "len": 103,
  "keys": [
    "parallel_ex000000_eng_latn",
    "parallel_ex000001_eng_latn",
    "parallel_ex000002_eng_latn",
    "parallel_ex000003_eng_latn",
    "parallel_ex000004_eng_latn",
    "parallel_ex000005_eng_latn",
    "parallel_ex000008_eng_latn",
    "parallel_ex000009_eng_latn",
    "parallel_ex000010_eng_latn",
    "parallel_ex000011_eng_latn",
    "parallel_ex000013_eng_latn",
    "parallel_ex000014_eng_latn",
    "parallel_ex000015_eng_latn",
    "parallel_ex000016_eng_latn",
    "parallel_ex000017_eng_latn",
    "parallel_ex000018_eng_latn",
    "parallel_ex000020_eng_latn",
    "parallel_ex000021_eng_latn",
    "parallel_ex000022_eng_latn",
    "parallel_ex000023_eng_latn"
  ],
  "samples": [
    {
      "key": "parallel_ex000000_eng_latn",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'parallel_ex000000_eng_latn_0', 'pred': ' The amount of air in the bag increases'}]"
          }
        ]
      }
    },
    {
      "key": "parallel_ex000001_eng_latn",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'parallel_ex000001_eng_latn_0', 'pred': ' The water goes into the cup'}]"
          }
        ]
      }
    },
    {
      "key": "parallel_ex000002_eng_latn",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'parallel_ex000002_eng_latn_0', 'pred': ' The ball sticks to the window'}]"
          }
        ]
      }
    }
  ]
}
```

## pred_reading70
- path: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/Reading/chck_70M/full_complianttok_reinvest_seed43022_70M_Reading/zero_shot/mlm/reading/predictions.json`
- exists: True size: 123022
```json
{
  "type": "dict",
  "len": 1,
  "keys": [
    "reading"
  ],
  "samples": [
    {
      "key": "reading",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 0, 'pred': 2.2088919853663382, 'prev_pred': 12.252676822559504}, {'id': 1, 'pred': 5.959836534177256, 'prev_pred': 3.9908172312686947}, {'id': 2, 'pred': 5.8525593005094665, 'prev_pred': 3.4896508133516466}, {'id': 3, 'pred': 11.768309328149291, 'prev_pred': 2.9782301558113766}, {'id': 4, 'p"
          }
        ]
      }
    }
  ]
}
```

## pred_blimp100
- path: `experiments/archive/frontier_consolidation/data/compliant_full_eval/official_outputs/complianttok_reinvest_seed43022/BLiMP/chck_100M/full_complianttok_reinvest_seed43022_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json`
- exists: True size: 5991760
```json
{
  "type": "dict",
  "len": 67,
  "keys": [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "coordinate_structure_constraint_complex_left_branch",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "distractor_agreement_relational_noun",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adjective_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "drop_argument",
    "distractor_agreement_relative_clause"
  ],
  "samples": [
    {
      "key": "adjunct_island",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'adjunct_island_0', 'pred': 'Who should Derek hug after shocking Richard?'}, {'id': 'adjunct_island_1', 'pred': 'What had Theresa walked through while talking about that high school?'}, {'id': 'adjunct_island_2', 'pred': 'Who will Katherine discover Erin without hiring?'}, {'id': 'adjunct_is"
          }
        ]
      }
    },
    {
      "key": "anaphor_gender_agreement",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'anaphor_gender_agreement_0', 'pred': \"Katherine can't help herself.\"}, {'id': 'anaphor_gender_agreement_1', 'pred': \"Marie won't think about herself.\"}, {'id': 'anaphor_gender_agreement_2', 'pred': \"Mark hasn't discussed itself.\"}, {'id': 'anaphor_gender_agreement_3', 'pred': 'Stephen impre"
          }
        ]
      }
    },
    {
      "key": "anaphor_number_agreement",
      "value": {
        "type": "dict",
        "len": 1,
        "keys": [
          "predictions"
        ],
        "samples": [
          {
            "key": "predictions",
            "value": "[{'id': 'anaphor_number_agreement_0', 'pred': 'Susan revealed herself.'}, {'id': 'anaphor_number_agreement_1', 'pred': \"Renee hasn't hurt herself.\"}, {'id': 'anaphor_number_agreement_2', 'pred': 'These patients do respect themselves.'}, {'id
```

## aoa_score100
- path: `experiments/archive/frontier_consolidation/data/compliant_full_eval/aoa_outputs/complianttok_reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`
- exists: True size: 94
```json
{
  "type": "dict",
  "len": 2,
  "keys": [
    "aoa",
    "curve_fitness_record"
  ],
  "samples": [
    {
      "key": "aoa",
      "value": {
        "type": "float",
        "repr": "0.0"
      }
    },
    {
      "key": "curve_fitness_record",
      "value": {
        "type": "dict",
        "len": 2,
        "keys": [
          "curve_fitness",
          "n_words"
        ],
        "samples": [
          {
            "key": "curve_fitness",
            "value": "0.0"
          },
          {
            "key": "n_words",
            "value": "236"
          }
        ]
      }
    }
  ]
}
```

## aoa_surprisal
- path: `experiments/archive/frontier_consolidation/data/compliant_full_eval/aoa_outputs/complianttok_reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
- exists: True size: 39626025
```json
{
  "type": "dict",
  "len": 2,
  "keys": [
    "metadata",
    "results"
  ],
  "samples": [
    {
      "key": "metadata",
      "value": {
        "type": "dict",
        "len": 4,
        "keys": [
          "model_name",
          "use_bos_only",
          "total_steps",
          "completed_steps"
        ],
        "samples": [
          {
            "key": "model_name",
            "value": "'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model'"
          },
          {
            "key": "use_bos_only",
            "value": "False"
          },
          {
            "key": "total_steps",
            "value": "19"
          }
        ]
      }
    },
    {
      "key": "results",
      "value": {
        "type": "list",
        "len": 152095,
        "first": "{'step': 'chck_1M', 'word_count': 1000000, 'target_word': 'airplane', 'context_id': 0, 'context': 'Not only does this freshen up the look and feel of a brand, but it is much cheaper to paint an', 'surprisal': 10.34854507446289}"
      }
    }
  ]
}
```
