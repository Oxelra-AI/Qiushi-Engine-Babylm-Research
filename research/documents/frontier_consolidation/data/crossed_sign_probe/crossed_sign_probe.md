# crossed sign probe design and smoke crossed-sign interaction probe

Created: 2026-09-01T04:17:28Z
Seed: 42
Content SHA256: `376e0e1f11c5f71fc3ac819d1b83510c755bb205e3acbf45a2ca0f0f1e584762`
Total items: 250

## Families

- **agent_action**: 45
- **entity_state**: 70
- **property_bind**: 45
- **spatial_put**: 45
- **temporal_order**: 45

### entity_state depth breakdown
- depth 2: 35
- depth 3: 25
- depth 4: 10

## Design

2×2 factorial: C1 makes alt_A correct, C2 makes alt_B correct.
Interaction Δ = [PLL(A|C1) - PLL(B|C1)] - [PLL(A|C2) - PLL(B|C2)]
Expected Δ > 0. Crossed-sign = margin(C1) > 0 AND margin(C2) < 0.

Alternatives differ in exactly one content word/phrase.
Entity_state alternatives vary in location (same preposition).
Property_bind alternatives vary in entity name ("the adj one is...").
Agent_action alternatives vary in person name ("the role is...").
Temporal_order alternatives vary in person name ("the first to... was...").
Spatial_put alternatives vary in location (same preposition).

## Sample items

### agent_action (id: agent_action_007)
- C1: Sam cooks and Alice cleans.
- C2: Alice cooks and Sam cleans.
- alt_A: The cook is Sam.
- alt_B: The cook is Alice.

### entity_state (id: entity_state_d2_019)
- C1: The stone is in the bag. Sam carries the stone to the drawer.
- C2: The stone is in the drawer. Sam carries the stone to the bag.
- alt_A: The stone is now in the drawer.
- alt_B: The stone is now in the bag.

### property_bind (id: property_bind_019)
- C1: The bear is heavy and the lion is light.
- C2: The bear is light and the lion is heavy.
- alt_A: The heavy one is the bear.
- alt_B: The heavy one is the lion.

### spatial_put (id: spatial_put_005)
- C1: Tom places the coin on the table.
- C2: Tom places the coin on the bed.
- alt_A: The coin is on the table.
- alt_B: The coin is on the bed.

### temporal_order (id: temporal_order_016)
- C1: Alice left before Anna.
- C2: Anna left before Alice.
- alt_A: The first to leave was Alice.
- alt_B: The first to leave was Anna.
