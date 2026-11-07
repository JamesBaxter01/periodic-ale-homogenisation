# History-free baselines

## consolidation: fitted parameters

`{"K": 26.4573, "G": 28.0233, "kappa": 4.87515e-06, "alpha": 0.470892, "biot_modulus": 231.366, "porosity": 0.254826}`

Cell rest tangent (cubic): `{"C1111": 63.8216, "C3333": 63.8127, "C1122": 7.77508, "C1212": 19.946, "kappa_33": 4.87447e-06, "alpha_33": 0.470984}`

## Consolidation (step load): Table 4 statistics against the resolved runs

| Resolved skeleton | Model | Settlement max | Drained volume max | Cells 1-9 max | Top cell max (at t) | Final settlement (mm) |
|---|---|---|---|---|---|---|
| elastic | finite_strain | 57.378 % | 1.516 % | 0.3425 | 0.2570 (0.00204 s) | 74.565 |
| elastic | FE² | 0.862 % | 1.630 % | 0.0273 | 0.0991 (2.68 s) | 74.994 |
| viscoelastic | finite_strain | 57.378 % | 1.528 % | 0.0393 | 0.1220 (2.96 s) | 74.565 |
| viscoelastic | FE² | 0.858 % | 1.622 % | 0.0220 | 0.0980 (2.82 s) | 75.014 |
| elastic | linear | 58.286 % | 4.604 % | 0.3281 | 0.2424 (0.00204 s) | 78.322 |
| viscoelastic | linear | 58.286 % | 4.628 % | 0.0235 | 0.1088 (3.11 s) | 78.322 |

Each row compares the named model with the resolved run of that skeleton (Table 4 conventions).
