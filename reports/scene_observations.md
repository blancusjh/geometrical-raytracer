# Scene Observations

## Ellipse

**Zoom luminance (mean ± std)**
| Region | GL | AGG | Δ (AGG-GL) |
| --- | --- | --- | --- |
| focus | 0.1091 ± 0.2988 | 0.1100 ± 0.2469 | 0.0009 |
| axis_left | 0.1926 ± 0.3763 | 0.1873 ± 0.3224 | -0.0053 |
| axis_right | 0.2502 ± 0.4114 | 0.1657 ± 0.2581 | -0.0845 |

**Focus miss**: rays=25, mean miss=0.0000, max miss=0.0000

**Notes:**
- See renders in `renders/` for side-by-side GL/AGG comparisons.
- AGG yields smoother edges but reduces luminance where many segments overlap.

## Cartesian Dioptrique

**Zoom luminance (mean ± std)**
| Region | GL | AGG | Δ (AGG-GL) |
| --- | --- | --- | --- |
| surface | 0.0267 ± 0.1552 | 0.0234 ± 0.1013 | -0.0033 |
| post_surface | 0.0467 ± 0.2028 | 0.0107 ± 0.0403 | -0.0359 |
| pre_surface | 0.0613 ± 0.2305 | 0.0563 ± 0.1523 | -0.0050 |

**Focus miss**: rays=7, mean miss=0.0003, max miss=0.0005

**Notes:**
- See renders in `renders/` for side-by-side GL/AGG comparisons.
- AGG yields smoother edges but reduces luminance where many segments overlap.

## Cartesian Singlet

**Zoom luminance (mean ± std)**
| Region | GL | AGG | Δ (AGG-GL) |
| --- | --- | --- | --- |
| front_surface | 0.0283 ± 0.1595 | 0.0224 ± 0.0907 | -0.0059 |
| back_surface | 0.0283 ± 0.1595 | 0.0144 ± 0.0592 | -0.0139 |
| final_focus | 0.0955 ± 0.2820 | 0.0806 ± 0.1738 | -0.0149 |

**Focus miss**: rays=11, mean miss=0.0371, max miss=0.0668

**Notes:**
- See renders in `renders/` for side-by-side GL/AGG comparisons.
- AGG yields smoother edges but reduces luminance where many segments overlap.

## High Aperture Singlet

**Zoom luminance (mean ± std)**
| Region | GL | AGG | Δ (AGG-GL) |
| --- | --- | --- | --- |
| front_surface | 0.0509 ± 0.2114 | 0.0430 ± 0.1305 | -0.0079 |
| back_surface | 0.0509 ± 0.2114 | 0.0430 ± 0.1303 | -0.0080 |
| intermediate_axis | 0.2736 ± 0.4186 | 0.1556 ± 0.2002 | -0.1180 |

**Focus miss**: rays=33, mean miss=0.0914, max miss=0.1957

**Notes:**
- See renders in `renders/` for side-by-side GL/AGG comparisons.
- AGG yields smoother edges but reduces luminance where many segments overlap.
