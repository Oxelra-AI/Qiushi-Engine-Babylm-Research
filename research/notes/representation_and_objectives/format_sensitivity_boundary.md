# format sensitivity boundary — Format sensitivity as a data-efficient learning boundary

## Phase A results

| Arm | Format | Base fit | Changed focal | Overall train | Result |
|-----|--------|----------|---------------|--------------|--------|
| tag_only | `Entry TAG:` content | 1.0 | 1.0 | 1.0 | **all fit** ✓ |
| inline_direct | `Focal Time entry TAG:` content | 0.5 | 0.5 | 0.5 | **total collapse** ✗ |

## Format sensitivity results

| Arm | Format | Base fit | Changed focal | Overall | Result |
|-----|--------|----------|---------------|---------|--------|
| prefix1_direct | `The entry TAG:` content | 0.502 | 0.5 | 0.501 | **total collapse** ✗ |
| postfix_direct | `Entry TAG (role):` content | 1.0 | 0.508 | 0.978 | **base fits, changed fails** |

## Interpretation

### Finding 1: Extreme format sensitivity in acquisition

Adding even a SINGLE word ("The") before "entry TAG:" prevents training entirely — not just changed-focal rows but ALL categories including the easy 5,120-row base stable task. This is not about tag collision, role words, or task complexity. It is about the model's optimization requiring the exact token "Entry" at a specific position relative to the segment start.

Comparison with role coordinate collision and route:
- role coordinate collision and route filler (4 tag-free extra sentences): training collapsed
- role coordinate collision and route labeldup (4 "label TAG" sentences): base fit, changed focal failed
- format sensitivity boundary prefix1 (just "The" before "entry TAG"): training collapsed
- format sensitivity boundary inline ("Scope Time entry TAG:"): training collapsed
- format sensitivity boundary postfix ("Entry TAG (role):"): base fit, changed focal failed

The collapse pattern correlates with whether "Entry" remains the first token of the content segment. When it does (postfix, labeldup), base facts can be learned. When it doesn't (prefix1, inline, filler), everything fails.

### Finding 2: Postfix annotations reproduce the collision pattern

`Entry TAG (focal background):` preserves "Entry TAG" at the start, so base and stable rows fit (1.0). But `sparse_changed_focal` was only 0.508 — the same collision pattern as role coordinate collision and route's labeldup/entrydup/slotdecl. The parenthetical role annotation acts as an extra tag-adjacent mention that selectively disrupts changed-focal acquisition.

Postfix held eval (not fully interpretable since changed focal didn't fit):
- direct_hH: focal_before=0.0, focal_after=1.0, secondary=1.0/1.0
- roleSwap_direct_hH: focal_before=1.0, focal_after=0.0 — INVERTS with swap
- role_hH: focal_before=0.25, focal_after=0.75, secondary=1.0/1.0

The role-swap inversion of DIRECT-TAG answers shows that even when querying by tag, the model's output is influenced by the parenthetical role annotation.

### Finding 3: Dissociation between acquisition and inference

role coordinate collision and route insertion matrix showed: a fitted tag-only model reads through tag-free filler at inference perfectly. But training WITH that same filler prevents acquisition. format sensitivity boundary confirms: the inline format prevents training but the fitted model processes tag-free extra tokens without difficulty.

**The acquisition bottleneck is narrower than the inference capability.**

## Scientific significance for data-efficient learning

This is a genuine data-efficient learning phenomenon:

> Under limited data, the learner's optimization landscape is extremely sensitive to the exact context format. A specific token pattern ("Entry TAG:") creates a gradient anchor that enables convergence; displacing that pattern by even one token position eliminates the learning signal for the entire task.

This connects to the broader research:
1. **Supplied coordinates enable state selection** — but only in the exact format that the optimizer can converge on
2. **Coordinate collision disrupts changed-state records** — when tag-adjacent material adds competing cues
3. **Format sensitivity constrains role-to-coordinate learning** — role annotations can't be injected alongside tags during fine-tuning

The section bridge (Background:/Update:) worked because it used the role word AS the primary address, matching the "first-token anchor" requirement. The tag bridge worked because it used tags without role information. COMBINING both cannot be made to work through any tested format.

## Implication for the semantic-role-to-coordinate map

The tested approach of injecting role annotations into the fine-tuning context is blocked by format sensitivity. The remaining routes to role-to-coordinate learning:

1. **Section-bridge composition**: use role words (Background/Update) as addresses in contexts that also contain unique tag identifiers, but query via role rather than tag
2. **Pretraining-level format exposure**: the format sensitivity may be specific to fine-tuning a frozen-format pretrained model; a model pretrained on diverse entry formats might not have this constraint
3. **Multi-phase learning**: first train tag-only R(k,v), freeze it, then train a separate role→tag map M(r,k) with the tags held constant
4. **Compositional evaluation only**: test whether the pretrained model already encodes role-to-record mapping from pretraining, without fine-tuning

## Files
- `data/tag_only1/` — tag_only baseline: all fit, all readouts 1.0
- `data/inline_direct1/` — inline format: total collapse (0.5 everywhere)
- `data/postfix_direct1/` — postfix format: base fits, changed focal 0.508
- `data/prefix1_direct1/` — prefix1 format: total collapse (0.5 everywhere)
- `training/scripts/inline_role_bridge.py` — construction script with all format modes
