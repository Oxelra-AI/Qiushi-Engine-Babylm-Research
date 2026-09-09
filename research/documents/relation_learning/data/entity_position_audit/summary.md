# research synthesis correction: Entity position/order shortcut audit

## Purpose
Check whether the updated entity systematically appears before/after the
unchanged entity in context. If so, position could substitute for entity identity.

## Train (1666 pairs)

### Source sentence
- Updated entity first: 308
- Unchanged entity first: 650
- Same position: 0
- Missing: 708
- Fraction updated first: 0.3215
- Mean position diff (upd-unch): 24.39

### Shared context prefix (source + update)
- Updated entity first: 310
- Unchanged entity first: 1257
- Same: 0
- Missing: 99
- Fraction updated first: 0.1978
- Mean position diff (upd-unch): 45.04

## Heldout (200 pairs)

### Source sentence
- Updated entity first: 29
- Unchanged entity first: 86
- Same position: 0
- Missing: 85
- Fraction updated first: 0.2522
- Mean position diff (upd-unch): 27.44

### Shared context prefix (source + update)
- Updated entity first: 29
- Unchanged entity first: 160
- Same: 0
- Missing: 11
- Fraction updated first: 0.1534
- Mean position diff (upd-unch): 50.46

## Risk: HIGH
