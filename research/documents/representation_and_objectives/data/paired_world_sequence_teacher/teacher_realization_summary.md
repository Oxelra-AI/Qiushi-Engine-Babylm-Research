# paired world stress teacher and route teacher realization verification

## Purpose

Approved teacher models are used only to verify that the paired-world sentences and hypotheses express the intended role assignment. This is not a student-learning result.

## Aggregate

- prompts per teacher: 3200
- cross-teacher agreement: 0.879
- both-teachers correct: 0.879
- score-ablated both-teachers correct: 0.900
- held-template both-teachers correct: 0.734

## By variant

| variant | agreement | both correct | n |
|---|---:|---:|---:|
| hyp_lost_to | 0.830 | 0.830 | 800 |
| name_swapped_context | 0.907 | 0.907 | 800 |
| original_ablated | 0.900 | 0.900 | 800 |
| original_score_visible | 0.879 | 0.879 | 800 |

## Failure sample

### f0010_c1_original_ablated_defeated_AB
- variant=original_ablated sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Martin Vilarrubi overcame Miguel Angel Lopez Jaen in the round of 16 at Spain 2 3 on March 3, 2003.
- hypothesis: Miguel Angel Lopez Jaen defeated Martin Vilarrubi.

### f0010_c2_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Martin Vilarrubi fell to Miguel Angel Lopez Jaen at Spain 3 2 in the round of 16 on May 5, 2003. The score was 6-4 6-2.
- hypothesis: Martin Vilarrubi defeated Miguel Angel Lopez Jaen.

### f0015_c1_original_ablated_defeated_AB
- variant=original_ablated sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Fernando Gonzalez won against Tommy Robredo in the round of 64 of Australian Open on January 14, 2002.
- hypothesis: Tommy Robredo defeated Fernando Gonzalez.

### f0015_c1_original_score_visible_defeated_AB
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Fernando Gonzalez won against Tommy Robredo in the round of 64 of Australian Open on January 14, 2002. The score was 6-2 6-4 6-4.
- hypothesis: Tommy Robredo defeated Fernando Gonzalez.

### f0015_c2_name_swapped_context_defeated_AB
- variant=name_swapped_context sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Fernando Gonzalez won against Tommy Robredo in the semifinal of Stuttgart on July 14, 2003.
- hypothesis: Tommy Robredo defeated Fernando Gonzalez.

### f0015_c2_original_ablated_defeated_BA
- variant=original_ablated sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Tommy Robredo won against Fernando Gonzalez in the semifinal of Stuttgart on July 14, 2003.
- hypothesis: Fernando Gonzalez defeated Tommy Robredo.

### f0016_c2_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Desislava Topalova fell to Sandra Kleinova at Bratislava in the qualifying round on October 18, 1999. The score was 7-6 6-3.
- hypothesis: Desislava Topalova defeated Sandra Kleinova.

### f0020_c2_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 32 at Florianopolis CH on April 23, 2007, Daniel Koellerer was unable to overcome Juan Pablo Brzezicki.
- hypothesis: Juan Pablo Brzezicki lost to Daniel Koellerer.

### f0020_c2_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 32 at Florianopolis CH on April 23, 2007, Daniel Koellerer was unable to overcome Juan Pablo Brzezicki.
- hypothesis: Daniel Koellerer lost to Juan Pablo Brzezicki.

### f0020_c2_name_swapped_context_defeated_BA
- variant=name_swapped_context sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 32 at Florianopolis CH on April 23, 2007, Juan Pablo Brzezicki was unable to overcome Daniel Koellerer.
- hypothesis: Daniel Koellerer defeated Juan Pablo Brzezicki.

### f0020_c2_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 32 at Florianopolis CH on April 23, 2007, Daniel Koellerer was unable to overcome Juan Pablo Brzezicki. The score was 6-2 6-0.
- hypothesis: Daniel Koellerer defeated Juan Pablo Brzezicki.

### f0031_c1_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the match at La Liga 05/06 on April 2, 2006, Sevilla was unable to overcome Betis.
- hypothesis: Sevilla lost to Betis.

### f0031_c1_name_swapped_context_defeated_BA
- variant=name_swapped_context sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the match at La Liga 05/06 on April 2, 2006, Betis was unable to overcome Sevilla.
- hypothesis: Sevilla defeated Betis.

### f0031_c1_original_ablated_defeated_AB
- variant=original_ablated sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the match at La Liga 05/06 on April 2, 2006, Sevilla was unable to overcome Betis.
- hypothesis: Betis defeated Sevilla.

### f0031_c1_original_score_visible_defeated_AB
- variant=original_score_visible sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the match at La Liga 05/06 on April 2, 2006, Sevilla was unable to overcome Betis. The score was 2-1.
- hypothesis: Betis defeated Sevilla.

### f0031_c2_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Sevilla emerged victorious over Betis in the match at La Liga 05/06 on November 19, 2005.
- hypothesis: Betis lost to Sevilla.

### f0032_c1_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: During the match at La Liga 14/15 on March 7, 2015, Granada prevailed against Malaga.
- hypothesis: Malaga lost to Granada.

### f0037_c1_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the quarterfinal at Turkey F12 on March 26, 2018, Genaro Alberto Olivieri was unable to overcome Cem Ilkel.
- hypothesis: Cem Ilkel lost to Genaro Alberto Olivieri.

### f0037_c1_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the quarterfinal at Turkey F12 on March 26, 2018, Genaro Alberto Olivieri was unable to overcome Cem Ilkel.
- hypothesis: Genaro Alberto Olivieri lost to Cem Ilkel.

### f0037_c1_name_swapped_context_defeated_BA
- variant=name_swapped_context sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the quarterfinal at Turkey F12 on March 26, 2018, Cem Ilkel was unable to overcome Genaro Alberto Olivieri.
- hypothesis: Genaro Alberto Olivieri defeated Cem Ilkel.

### f0037_c1_original_ablated_defeated_AB
- variant=original_ablated sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the quarterfinal at Turkey F12 on March 26, 2018, Genaro Alberto Olivieri was unable to overcome Cem Ilkel.
- hypothesis: Cem Ilkel defeated Genaro Alberto Olivieri.

### f0037_c1_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the quarterfinal at Turkey F12 on March 26, 2018, Genaro Alberto Olivieri was unable to overcome Cem Ilkel. The score was 6-4 6-2.
- hypothesis: Genaro Alberto Olivieri defeated Cem Ilkel.

### f0057_c1_original_ablated_defeated_AB
- variant=original_ablated sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Benjamin Balleret won against Niels Desein in the round of 32 of Geneva CH on August 17, 2009.
- hypothesis: Niels Desein defeated Benjamin Balleret.

### f0057_c1_original_score_visible_defeated_AB
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Benjamin Balleret won against Niels Desein in the round of 32 of Geneva CH on August 17, 2009. The score was 6-2 7-6(5).
- hypothesis: Niels Desein defeated Benjamin Balleret.

### f0057_c2_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Benjamin Balleret fell to Niels Desein at Senegal F2 in the semifinal on August 4, 2008.
- hypothesis: Niels Desein lost to Benjamin Balleret.

### f0077_c2_original_score_visible_defeated_AB
- variant=original_score_visible sport=badminton split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: During the round of 32 at Yonex French Open 2013 on October 23, 2013, Karin SCHNAASE-BEERMANN prevailed against Ella DIEHL. The score was 14-21 19-21.
- hypothesis: Ella DIEHL defeated Karin SCHNAASE-BEERMANN.

### f0090_c1_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Wojciech Marek fell to Eric Vanshelboim at M15 Monastir in the semifinal on January 4, 2021.
- hypothesis: Eric Vanshelboim lost to Wojciech Marek.

### f0090_c1_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Wojciech Marek fell to Eric Vanshelboim at M15 Monastir in the semifinal on January 4, 2021. The score was 6-4 5-7 7-6(4).
- hypothesis: Wojciech Marek defeated Eric Vanshelboim.

### f0101_c2_name_swapped_context_defeated_BA
- variant=name_swapped_context sport=badminton split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: TAI Tzu Ying overcame TEE Jing Yi in the quarterfinal at PROTON MALAYSIA OPEN SUPER SERIES 2010 on January 19, 2010.
- hypothesis: TEE Jing Yi defeated TAI Tzu Ying.

### f0101_c2_original_ablated_defeated_AB
- variant=original_ablated sport=badminton split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: TEE Jing Yi overcame TAI Tzu Ying in the quarterfinal at PROTON MALAYSIA OPEN SUPER SERIES 2010 on January 19, 2010.
- hypothesis: TAI Tzu Ying defeated TEE Jing Yi.

### f0101_c2_original_score_visible_defeated_AB
- variant=original_score_visible sport=badminton split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: TEE Jing Yi overcame TAI Tzu Ying in the quarterfinal at PROTON MALAYSIA OPEN SUPER SERIES 2010 on January 19, 2010. The score was 21-19 20-22 24-22.
- hypothesis: TAI Tzu Ying defeated TEE Jing Yi.

### f0107_c1_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 16 at Peru F2 on September 6, 1999, Rodolfo Rake was unable to overcome Patricio Arquez.
- hypothesis: Rodolfo Rake lost to Patricio Arquez.

### f0107_c1_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 16 at Peru F2 on September 6, 1999, Rodolfo Rake was unable to overcome Patricio Arquez.
- hypothesis: Patricio Arquez lost to Rodolfo Rake.

### f0107_c1_name_swapped_context_defeated_AB
- variant=name_swapped_context sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 16 at Peru F2 on September 6, 1999, Patricio Arquez was unable to overcome Rodolfo Rake.
- hypothesis: Rodolfo Rake defeated Patricio Arquez.

### f0107_c1_original_ablated_defeated_BA
- variant=original_ablated sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 16 at Peru F2 on September 6, 1999, Rodolfo Rake was unable to overcome Patricio Arquez.
- hypothesis: Patricio Arquez defeated Rodolfo Rake.

### f0107_c1_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: In the round of 16 at Peru F2 on September 6, 1999, Rodolfo Rake was unable to overcome Patricio Arquez. The score was 0-6 7-5 6-4.
- hypothesis: Patricio Arquez defeated Rodolfo Rake.

### f0130_c2_original_score_visible_defeated_AB
- variant=original_score_visible sport=badminton split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: TAN Wee Kiong / GOH V Shem proved too strong for SHIN Baek Cheol / KO Sung Hyun in the round of 16 at YONEX-SUNRISE Hong Kong Open on November 19, 2015. The score was 20-22 18-21.
- hypothesis: SHIN Baek Cheol / KO Sung Hyun defeated TAN Wee Kiong / GOH V Shem.

### f0131_c1_hyp_lost_to_lost_to_BA
- variant=hyp_lost_to sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Ath Bilbao emerged victorious over Barcelona in the match at La Liga 05/06 on May 20, 2006.
- hypothesis: Barcelona lost to Ath Bilbao.

### f0131_c2_hyp_lost_to_lost_to_AB
- variant=hyp_lost_to sport=football split=train template_split=train gold=ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: It was Barcelona who came out on top against Ath Bilbao in the match of La Liga 14/15 on September 13, 2014.
- hypothesis: Ath Bilbao lost to Barcelona.

### f0141_c2_original_score_visible_defeated_BA
- variant=original_score_visible sport=tennis split=train template_split=train gold=NOT_ENTAILED qwen=NOT_ENTAILED llama=ENTAILED
- context: Iztok Bozic fell to Sander Groen at Italy 1 Masters 3 in the round of 32 on December 31, 1990. The score was 6-1 1-6 6-1.
- hypothesis: Iztok Bozic defeated Sander Groen.

## Scientific meaning

If these values are high, the current realizations are linguistically faithful enough to serve as probes and as inputs to harder source construction. The sequence stress results still decide whether student training is premature: current sports outcomes alone remain too narrow unless cross-template, cross-predicate, and event-to-state transfer are established.

## Files

- rows: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/teacher_labeled_rows.jsonl`
- paired model rows: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/teacher_cross_model_pairs.jsonl`
- summary JSON: `experiments/archive/representation_and_objectives/data/paired_world_sequence_teacher/teacher_realization_summary.json`
