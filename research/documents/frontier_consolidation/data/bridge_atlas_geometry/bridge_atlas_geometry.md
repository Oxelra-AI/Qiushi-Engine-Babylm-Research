# bridge atlas geometry result bridge atlas-compatible geometry

## Atlas pooled benchmarks
| variant | pooled gap1 | pooled skip | mean source span | absent frac |
|---|---:|---:|---:|---:|
| compact | 0.7775 | 0.2225 | 0.8072 | 0.1729 |
| extractive_balanced | 0.5278 | 0.4722 | 0.9715 | 0.0000 |
| extractive_wide | 0.6465 | 0.3535 | 0.9281 | 0.0000 |

## Bridge groups — pooled gap1/skip (atlas-comparable)
| group | n | bridge pooled gap1 | bridge pooled skip | natcomp pooled gap1 | natcomp pooled skip |
|---|---:|---:|---:|---:|---:|
| all_bridge | 103 | 0.8753 | 0.1247 | 0.8158 | 0.1842 |
| extraction_like | 48 | 0.9357 | 0.0643 | 0.8602 | 0.1398 |
| transformation_like | 55 | 0.7949 | 0.2051 | 0.7644 | 0.2356 |
| substantive_only | 8 | 0.7027 | 0.2973 | 0.7660 | 0.2340 |

## Bridge groups — per-pair mean (for distribution reference)
| group | n | mean gap1 | mean skip | mean span | mean absent | mean compress |
|---|---:|---:|---:|---:|---:|---:|
| all_bridge | 103 | 0.8356 | 0.1644 | 0.8990 | 0.0223 | 0.7500 |
| extraction_like | 48 | 0.9180 | 0.0820 | 0.9297 | 0.0029 | 0.8280 |
| transformation_like | 55 | 0.7637 | 0.2363 | 0.8722 | 0.0393 | 0.6819 |
| substantive_only | 8 | 0.6458 | 0.3542 | 0.6897 | 0.1069 | 0.6049 |

## Matched natural compact — per-pair mean (same 103 pairs)
| group | n | mean gap1 | mean skip | mean span | mean absent | mean compress |
|---|---:|---:|---:|---:|---:|---:|
| all_bridge | 103 | 0.8063 | 0.1937 | 0.8027 | 0.1528 | 0.6575 |
| extraction_like | 48 | 0.8582 | 0.1418 | 0.7881 | 0.1358 | 0.7054 |
| transformation_like | 55 | 0.7621 | 0.2379 | 0.8155 | 0.1677 | 0.6157 |
| substantive_only | 8 | 0.7693 | 0.2307 | 0.5746 | 0.1077 | 0.6031 |

## Proximity to atlas benchmarks (bridge pooled gap1)
- **all_bridge**: bridge_pooled_gap1=0.8753, closest=compact (dist 0.0978)
- **extraction_like**: bridge_pooled_gap1=0.9357, closest=compact (dist 0.1582)
- **transformation_like**: bridge_pooled_gap1=0.7949, closest=compact (dist 0.0174)
- **substantive_only**: bridge_pooled_gap1=0.7027, closest=extractive_wide (dist 0.0562)
