# role switch exposure geometry and route implication — role-switch legal screen exposure geometry and interpretation

## Why this note was needed

The shared-tokenizer from-scratch role-switch/role-fixed 80M screen launched in earlier analysis is cleanly controlled: both arms use the same legal tokenizer trained on the 9,977,920-word common intersection, identical initial weights, identical first effective batch, and identical masks/gradients before packet rows appear. But the scientific meaning of a flat readout depends on how much role-switch credit actually enters the legal stream. role switch exposure geometry and route implication therefore audited the training geometry without polling or touching the running H100 tasks.

Primary artifact: `data/role_packet_exposure_geometry/role_packet_exposure_geometry_summary.json`.

## Main quantitative facts

The legal screen embeds 22,080 packet words per 10M corpus pass and trains to 80M, so it gives exactly 176,640 packet word-exposures, the same packet word exposure used by the role switch packet screen synthesis disposable packet-only screen. The difference is dilution and placement:

- Packet word fraction of the 80M legal stream: **0.002208** (0.2208%).
- Target word events: **1,560 per pass**, **12,480 across 80M**.
- Expected WWM-selected role-target word events at p=0.15: **1,872**.
  - Expected actual `<mask>` replacements: **1,497.6**.
  - Expected random replacements: **187.2**.
  - Expected keep-selected targets: **187.2**.
- Packet rows after packing: **138 per pass**, all 160 words.
- Optimizer updates touching packets: **29 / 2024** (1.4328%).
- Touched updates contain a median of **48 packet rows / 7,680 packet words** and **483 target events**, but these are mixed with ordinary corpus rows inside 256-row effective batches.
- Packet placements are identical across treatment and control, at source row positions 3012..3710 in each pass. They appear around optimizer steps:
  - pass 0: 12–15
  - pass 1: 265–268
  - pass 2: 518–521
  - pass 3: 771–774
  - pass 4: 1024–1027
  - pass 5: 1277–1279
  - pass 6: 1530–1532
  - pass 7: 1783–1785

The first selected OpenSubtitles rows were used for the in-place replacement, so the packet block is clustered early in each corpus pass, immediately after the initial compact/FineWeb block, rather than uniformly distributed across the whole pass.

## Relation to the role switch packet screen synthesis disposable packet-only result

The role switch packet screen synthesis disposable continuation used the same 176,640 packet word-exposure but as packet-only updates from an already trained reheat checkpoint. It produced strong synthetic learning:

- treatment both_correct 0.6987 vs role-fixed 0.3750;
- held-out-style treatment minus role-fixed +0.1815;
- unseen container treatment 0.5611 vs role-fixed 0.2167.

That screen also showed weak natural official-surface transfer. role switch exposure geometry and route implication clarifies one reason a legal from-scratch screen may be weaker: the same packet words are only 0.2208% of the full stream and affect only 29 optimizer updates, instead of 104 packet-only disposable updates. A flat natural readout would not contradict the synthetic learnability finding; it would show that this sparse primary-text insertion does not by itself create the desired general natural binding movement.

## Control semantics

The role-fixed control is not a no-information baseline. For each pair, it uses one fixed alternative in both AB/BA directions, alternating by pair_id to preserve target counts. Therefore, for every family and direction, exactly half of the role-fixed rows agree with the treatment target and half train the opposite target:

- comparative: AB same/opposite 90/90, BA same/opposite 90/90;
- spatial: AB same/opposite 120/120, BA same/opposite 120/120;
- state_change: AB same/opposite 90/90, BA same/opposite 90/90;
- transfer: AB same/opposite 90/90, BA same/opposite 90/90.

Thus role_switch-minus-role_fixed remains the exchange-specific signal. Broad treatment/control equality by itself is expected; the decisive question is whether role_switch moves the natural hard surfaces above role_fixed.

## Interpretation rule for the running screen

The predeclared route-relevant question remains natural transfer, not synthetic packet memorization. Continue Route B only if role_switch improves GlobalPIQA_parallel hard-52 ranks/margins and EWoK stable conditional reversals above role_fixed while preserving broad cheap7.

If the legal screen is flat on natural hard surfaces, close the specific mechanism **sparse role-switch primary-text replacement under ordinary WWM/AdamW** as a SOTA-relevant route. Do not claim that context-conditioned binding is solved or impossible. Also do not simply repeat the route by adding more packet text, changing names, or moving rows. Any further work must change the mechanism of credit assignment or representation in a way that follows from the evidence, e.g. a protected context-difference side path or another architecture/objective that makes competition between alternatives explicit without leaking official evaluation content.

## Prepared readout assets

- Natural cheap+hard readout wrapper: `scripts/screen_readout.py`.
- Natural GlobalPIQA hard-rank reader: `scripts/globalpiqa_margin_reader.py`.
- Natural EWoK stable-failure reader: `scripts/ewok_interaction_reader.py`.
- Synthetic packet readout prepared in role switch exposure geometry and route implication: `scripts/packet_synthetic_reader.py`; ready-only verified in `data/packet_synthetic_reader/readiness.json`.
- Exposure geometry script: `scripts/role_packet_exposure_geometry.py`.

## Research consequence

This audit sharpened the active uncertainty. The running screen is scientifically worth reading because it is the first confound-free legal from-scratch role-switch/control pair. But the dose/placement geometry makes the current intervention a stringent low-dose primary-text test, not a full test of all possible context-conditioned binding mechanisms. A negative result should redirect toward mechanisms that amplify conditional competition at the architecture/objective level rather than more sparse packet insertions.
