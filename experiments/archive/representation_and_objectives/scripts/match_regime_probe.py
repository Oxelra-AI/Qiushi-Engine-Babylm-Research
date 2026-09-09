#!/usr/bin/env python3
"""Probe additional matching regimes for research transition/control subset."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
p=Path('experiments/archive/representation_and_objectives/scripts/transition_structure_matcher.py')
spec=importlib.util.spec_from_file_location('matcher', p)
m=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)
t=m.load_rows(m.IN_T,'transition')
c=m.load_rows(m.IN_C,'anchor_control')
regimes={
 'source_origin_len_cap_quote': ['source_label','origin_source','length_bin','required_capability','quote_bin'],
 'source_origin_len_cap_proper': ['source_label','origin_source','length_bin','required_capability','proper_bin'],
 'source_origin_len_cap_pronoun': ['source_label','origin_source','length_bin','required_capability','pronoun_bin'],
 'source_origin_len_cap_digit': ['source_label','origin_source','length_bin','required_capability','digit_bin'],
 'source_origin_len_cap_comma': ['source_label','origin_source','length_bin','required_capability','comma_bin'],
 'surface_no_comma': ['source_label','origin_source','length_bin','required_capability','quote_bin','proper_bin','pronoun_bin','digit_bin'],
 'surface_no_pronoun': ['source_label','origin_source','length_bin','required_capability','quote_bin','proper_bin','digit_bin','comma_bin'],
 'surface_no_digit': ['source_label','origin_source','length_bin','required_capability','quote_bin','proper_bin','pronoun_bin','comma_bin'],
 'surface_no_proper': ['source_label','origin_source','length_bin','required_capability','quote_bin','pronoun_bin','digit_bin','comma_bin'],
 'surface_no_quote': ['source_label','origin_source','length_bin','required_capability','proper_bin','pronoun_bin','digit_bin','comma_bin'],
 'source_label_origin_source_only': ['source_label','origin_source'],
 'source_origin_len': ['source_label','origin_source','length_bin'],
 'source_origin_len_cap': ['source_label','origin_source','length_bin','required_capability'],
}
out=[]
for name, fields in regimes.items():
    r=m.matched_by_regime(t,c,name,fields)
    s=m.summarize_pair(name,r['selected_transition'],r['selected_control'],fields)
    out.append({
      'name':name,'fields':fields,'words':s['transition_words'],'t_sent':s['transition_sentences'],'c_sent':s['control_sentences'],
      'style_l1':s['style_l1_by_words'],
      'markers_t':s['mean_transition_markers_transition'],'markers_c':s['mean_transition_markers_control'],
      'physical_t':s['mean_physical_anchor_transition'],'physical_c':s['mean_physical_anchor_control'],
      'spatial_t':s['mean_spatial_anchor_transition'],'spatial_c':s['mean_spatial_anchor_control'],
      'quantity_t':s['mean_quantity_anchor_transition'],'quantity_c':s['mean_quantity_anchor_control'],
    })
for x in sorted(out,key=lambda z:(-z['words'],z['name'])):
    print(json.dumps(x,ensure_ascii=False))
