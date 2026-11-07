# Resolved refinement check (elastic step load)

Production: `results/consolidation/full_elastic_step_20261026_201037`; study: `results/consolidation/refinement_20261026_201037`.
Compared up to t = 3933 s (the end common to all runs).

## Change of the resolved reference (refined minus production)

| Run | Status | Settlement | Drained volume | Cells 1-9 pressure | Top-cell pressure | Settlement at end |
|---|---|---|---|---|---|---|
| mesh | complete at t = 3933 s, 114520 elements, maxh 0.075, time grid / 1 | 0.500 % | 1.273 % | 0.0062 | 0.0182 | 0.500 % |
| time | complete at t = 3933 s, 48537 elements, maxh 0.09, time grid / 2 | 0.134 % | 0.271 % | 0.0041 | 0.0029 | 0.007 % |

Settlement and drained volume: max |change| over the production final value. Pressures: max |change| over the traction from t = 0.002 s.

## Table 4 statistics of FE² against each reference

| Reference | Settlement max | Drained volume max | Cells 1-9 max | Top cell max (at t) |
|---|---|---|---|---|
| production | 0.862 % | 1.630 % | 0.0273 | 0.0991 (2.68 s) |
| mesh | 0.685 % | 1.552 % | 0.0332 | 0.0931 (2.68 s) |
| time | 0.853 % | 1.620 % | 0.0291 | 0.0984 (2.68 s) |
