# format sensitivity boundary — Inline-role bridge: interpretation plan

## Scientific question

role coordinate collision and route established that:
1. Unique entry tags enable changed-focal state selection (R(k,v) ceiling)
2. ANY extra tag-bearing sentence disrupts changed-focal selection (collision)
3. Tag-free filler is harmless at both readout and acquisition

**New hypothesis**: if role annotations are placed INSIDE the entry sentence (inline), 
each tag appears exactly once → no cross-sentence collision → R(k,v) should survive.

If R survives, the natural follow-up is whether role-based queries (C(r,v)) can 
compose: "focal background entry" → select entry with that annotation → extract content.

## Arms and decision table

| Arm | Context format | Query type | Tests |
|-----|----------------|------------|-------|
| tag_only | `Entry TAG: content` | direct-tag | R baseline |
| inline_direct | `Scope Time entry TAG: content` | direct-tag | R with inline roles |
| inline_role | `Scope Time entry TAG: content` | role phrase | C composition |
| compact_direct | `Preamble: bg=TAG1, up=TAG2. Entry TAG: content` | direct-tag | compact collision |

### Phase A decision (tag_only + inline_direct)

| tag_only | inline_direct | Reading |
|----------|---------------|---------|
| fits     | fits          | Inline annotations avoid collision → proceed to Phase B |
| fits     | fails         | Even co-located role words disrupt acquisition → problem is deeper than cross-sentence collision |
| fails    | *             | Construction error → investigate |

### Phase B decision (inline_role + compact_direct)

| inline_role | compact_direct | Reading |
|-------------|----------------|---------|
| fits + swap inverts | fails | Role composition works via co-location; compact preamble creates expected collision |
| fits + swap inverts | fits  | Both formats avoid collision; inline is not special |
| fits + no swap | * | Model uses position/order, not role annotation |
| fails | fails | Role composition and compact collision both fail |
| fails | fits | Direct retrieval survives compact format; composition is the hard part |

### Role-swap interpretation
If inline_role fits:
- `roleSwap_role`: swap focal annotations, use role queries
  - If focal inverts: model follows annotations ✓
  - If focal unchanged: model uses shortcut, not annotations ✗
- `roleSwap_direct`: swap annotations, use direct-tag queries
  - Should stay unchanged (tag→content doesn't change with annotation swap)
  
### Held paraphrase interpretation
- `rolePara`: background→prior, update→revised
  - If works: semantic generalization beyond exact training tokens
  - If fails: only exact role-word lookup, not semantic role understanding

## Files produced
- `data/tag_only1/` — Phase A control
- `data/inline_direct1/` — Phase A test
- `data/inline_role1/` — Phase B composition (if Phase A passes)
- `data/compact_direct1/` — Phase B collision check (if Phase A passes)
- `notes/269_inline_role_bridge_results.md` — final synthesis
