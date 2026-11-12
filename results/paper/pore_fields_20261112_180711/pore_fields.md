# Section 5.2 mechanism check: the pulse of Figure 4 split by mechanism

Stresses in Pa, positive in tension; departures at the same compression as the stationary cell (figure4_stationary.npz). Written by examples/microscale/pore_field_summary.py.

## Elastic skeleton, T = 0.2 s

| t/T | P33 (Pa) | P33 − stationary | solid elastic − stationary | Kelvin-Voigt | fluid, pore flow | fluid, dilatation | m_f − stationary | Q/Q0 | stationary Q/Q0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | -0.0000 | 1.000 | 1.000 |
| 0.10 | -2.62 | -1.39 | -0.53 | +0.00 | -0.56 | -0.30 | +0.0047 | 0.761 | 0.907 |
| 0.20 | -8.09 | -3.50 | -1.68 | +0.00 | -1.15 | -0.67 | +0.0137 | 0.394 | 0.678 |
| 0.30 | -14.26 | -5.22 | -2.93 | +0.00 | -1.43 | -0.87 | +0.0214 | 0.166 | 0.426 |
| 0.40 | -18.57 | -5.66 | -3.71 | +0.00 | -1.19 | -0.76 | +0.0242 | 0.153 | 0.258 |
| 0.50 | -18.78 | -4.31 | -3.43 | +0.00 | -0.52 | -0.36 | +0.0200 | 0.293 | 0.204 |
| 0.60 | -14.43 | -1.52 | -1.93 | +0.00 | +0.29 | +0.12 | +0.0085 | 0.514 | 0.258 |
| 0.70 | -7.52 | +1.51 | +0.10 | +0.00 | +0.89 | +0.51 | -0.0065 | 0.771 | 0.426 |
| 0.80 | -1.06 | +3.53 | +1.71 | +0.00 | +1.13 | +0.70 | -0.0186 | 1.014 | 0.678 |
| 0.90 | +2.65 | +3.89 | +2.30 | +0.00 | +0.96 | +0.63 | -0.0225 | 1.142 | 0.907 |
| 1.00 | +2.59 | +2.59 | +1.84 | +0.00 | +0.43 | +0.33 | -0.0171 | 1.042 | 1.000 |
| 1.10 | +1.57 | +1.57 | +1.18 | +0.00 | +0.23 | +0.16 | -0.0104 | 0.944 | 1.000 |
| 1.20 | +1.04 | +1.04 | +0.81 | +0.00 | +0.14 | +0.09 | -0.0068 | 0.923 | 1.000 |
| 1.30 | +0.74 | +0.74 | +0.59 | +0.00 | +0.09 | +0.06 | -0.0046 | 0.924 | 1.000 |
| 1.40 | +0.55 | +0.55 | +0.45 | +0.00 | +0.06 | +0.04 | -0.0033 | 0.931 | 1.000 |
| 1.50 | +0.42 | +0.42 | +0.36 | +0.00 | +0.04 | +0.02 | -0.0024 | 0.939 | 1.000 |

Dissipation above the steady background flow, pulse and hold: flow within the pores 67.4 %, uniform dilatation 32.6 %, skeleton 0.0 %.
Peak compressive stress 19.28 Pa at t/T = 0.45 (stationary peak 14.47 Pa).
Largest tension in the pulse 3.11 Pa at t/T = 0.95: solid elastic +1.87, Kelvin-Voigt +0.00, fluid viscous +1.24 Pa; fluid content -0.0209 against the stationary cell.
Pore-pressure fluctuation at t/T = 0.25: -0.0769 to 0.0456 (solver units; background -0.00525 to 0.00496).

## Elastic skeleton, T = 2 s

| t/T | P33 (Pa) | P33 − stationary | solid elastic − stationary | Kelvin-Voigt | fluid, pore flow | fluid, dilatation | m_f − stationary | Q/Q0 | stationary Q/Q0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | -0.0000 | 1.000 | 1.000 |
| 0.10 | -1.59 | -0.36 | -0.22 | +0.00 | -0.08 | -0.05 | +0.0015 | 0.898 | 0.907 |
| 0.20 | -5.46 | -0.88 | -0.63 | +0.00 | -0.16 | -0.09 | +0.0034 | 0.692 | 0.678 |
| 0.30 | -10.39 | -1.35 | -1.07 | +0.00 | -0.18 | -0.11 | +0.0049 | 0.478 | 0.426 |
| 0.40 | -14.46 | -1.55 | -1.33 | +0.00 | -0.14 | -0.08 | +0.0053 | 0.332 | 0.258 |
| 0.50 | -15.64 | -1.17 | -1.12 | +0.00 | -0.04 | -0.02 | +0.0034 | 0.273 | 0.204 |
| 0.60 | -13.17 | -0.26 | -0.38 | +0.00 | +0.07 | +0.05 | -0.0004 | 0.296 | 0.258 |
| 0.70 | -8.37 | +0.66 | +0.42 | +0.00 | +0.15 | +0.10 | -0.0038 | 0.414 | 0.426 |
| 0.80 | -3.49 | +1.10 | +0.83 | +0.00 | +0.17 | +0.10 | -0.0048 | 0.628 | 0.678 |
| 0.90 | -0.23 | +1.00 | +0.81 | +0.00 | +0.12 | +0.07 | -0.0036 | 0.849 | 0.907 |
| 1.00 | +0.57 | +0.57 | +0.53 | +0.00 | +0.03 | +0.01 | -0.0013 | 0.947 | 1.000 |
| 1.10 | +0.34 | +0.34 | +0.33 | +0.00 | +0.01 | +0.00 | -0.0004 | 0.972 | 1.000 |
| 1.20 | +0.24 | +0.24 | +0.23 | +0.00 | +0.01 | +0.00 | -0.0002 | 0.985 | 1.000 |
| 1.30 | +0.17 | +0.17 | +0.16 | +0.00 | +0.01 | +0.00 | -0.0001 | 0.992 | 1.000 |
| 1.40 | +0.12 | +0.12 | +0.12 | +0.00 | +0.00 | +0.00 | -0.0001 | 0.996 | 1.000 |
| 1.50 | +0.09 | +0.09 | +0.09 | +0.00 | +0.00 | +0.00 | -0.0000 | 0.998 | 1.000 |

Dissipation above the steady background flow, pulse and hold: flow within the pores 79.5 %, uniform dilatation 20.5 %, skeleton 0.0 %.
Peak compressive stress 15.70 Pa at t/T = 0.48 (stationary peak 14.47 Pa).
Largest tension in the pulse 0.62 Pa at t/T = 0.98: solid elastic +0.55, Kelvin-Voigt +0.00, fluid viscous +0.07 Pa; fluid content -0.0018 against the stationary cell.
Pore-pressure fluctuation at t/T = 0.25: -0.152 to 0.0945 (solver units; background -0.00525 to 0.00496).

## Viscoelastic skeleton, T = 0.2 s

| t/T | P33 (Pa) | P33 − stationary | solid elastic − stationary | Kelvin-Voigt | fluid, pore flow | fluid, dilatation | m_f − stationary | Q/Q0 | stationary Q/Q0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | -0.0000 | 1.000 | 1.000 |
| 0.10 | -7.26 | -6.03 | -0.35 | -4.58 | -0.74 | -0.37 | +0.0033 | 0.902 | 0.907 |
| 0.20 | -15.78 | -11.19 | -1.24 | -7.92 | -1.33 | -0.70 | +0.0109 | 0.658 | 0.678 |
| 0.30 | -22.22 | -13.19 | -2.38 | -8.47 | -1.50 | -0.83 | +0.0188 | 0.403 | 0.426 |
| 0.40 | -23.37 | -10.46 | -3.28 | -5.41 | -1.10 | -0.67 | +0.0232 | 0.247 | 0.258 |
| 0.50 | -17.84 | -3.38 | -3.32 | +0.46 | -0.27 | -0.25 | +0.0215 | 0.222 | 0.204 |
| 0.60 | -8.01 | +4.90 | -2.27 | +6.32 | +0.61 | +0.24 | +0.0130 | 0.312 | 0.258 |
| 0.70 | +1.33 | +10.36 | -0.65 | +9.26 | +1.17 | +0.58 | +0.0005 | 0.496 | 0.426 |
| 0.80 | +6.49 | +11.08 | +0.77 | +8.40 | +1.24 | +0.67 | -0.0111 | 0.737 | 0.678 |
| 0.90 | +6.37 | +7.60 | +1.51 | +4.72 | +0.86 | +0.51 | -0.0174 | 0.956 | 0.907 |
| 1.00 | +1.58 | +1.58 | +1.48 | -0.25 | +0.17 | +0.18 | -0.0164 | 1.041 | 1.000 |
| 1.10 | +0.98 | +0.98 | +1.15 | -0.38 | +0.10 | +0.12 | -0.0123 | 1.024 | 1.000 |
| 1.20 | +0.78 | +0.78 | +0.90 | -0.29 | +0.08 | +0.09 | -0.0094 | 1.011 | 1.000 |
| 1.30 | +0.62 | +0.62 | +0.71 | -0.22 | +0.06 | +0.06 | -0.0071 | 1.000 | 1.000 |
| 1.40 | +0.50 | +0.50 | +0.57 | -0.17 | +0.05 | +0.05 | -0.0055 | 0.991 | 1.000 |
| 1.50 | +0.42 | +0.42 | +0.47 | -0.13 | +0.04 | +0.04 | -0.0042 | 0.985 | 1.000 |

Dissipation above the steady background flow, pulse and hold: flow within the pores 19.1 %, uniform dilatation 9.0 %, skeleton 71.9 %.
Peak compressive stress 23.73 Pa at t/T = 0.37 (stationary peak 14.47 Pa).
Largest tension in the pulse 7.08 Pa at t/T = 0.85: solid elastic -1.45, Kelvin-Voigt +6.82, fluid viscous +1.71 Pa; fluid content -0.0151 against the stationary cell.
Pore-pressure fluctuation at t/T = 0.25: -0.298 to 0.119 (solver units; background -0.00525 to 0.00495).

## Viscoelastic skeleton, T = 2 s

| t/T | P33 (Pa) | P33 − stationary | solid elastic − stationary | Kelvin-Voigt | fluid, pore flow | fluid, dilatation | m_f − stationary | Q/Q0 | stationary Q/Q0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | -0.0000 | 1.000 | 1.000 |
| 0.10 | -1.99 | -0.76 | -0.21 | -0.42 | -0.09 | -0.05 | +0.0016 | 0.908 | 0.907 |
| 0.20 | -6.15 | -1.56 | -0.60 | -0.71 | -0.16 | -0.09 | +0.0038 | 0.699 | 0.678 |
| 0.30 | -11.13 | -2.09 | -1.05 | -0.76 | -0.18 | -0.11 | +0.0054 | 0.478 | 0.426 |
| 0.40 | -14.93 | -2.02 | -1.33 | -0.48 | -0.13 | -0.08 | +0.0057 | 0.327 | 0.258 |
| 0.50 | -15.61 | -1.14 | -1.14 | +0.05 | -0.04 | -0.02 | +0.0038 | 0.266 | 0.204 |
| 0.60 | -12.63 | +0.28 | -0.41 | +0.57 | +0.07 | +0.05 | -0.0002 | 0.291 | 0.258 |
| 0.70 | -7.59 | +1.44 | +0.39 | +0.80 | +0.15 | +0.10 | -0.0041 | 0.413 | 0.426 |
| 0.80 | -2.79 | +1.80 | +0.82 | +0.71 | +0.17 | +0.10 | -0.0056 | 0.629 | 0.678 |
| 0.90 | +0.16 | +1.39 | +0.82 | +0.39 | +0.12 | +0.07 | -0.0044 | 0.852 | 0.907 |
| 1.00 | +0.55 | +0.55 | +0.55 | -0.03 | +0.02 | +0.01 | -0.0018 | 0.954 | 1.000 |
| 1.10 | +0.32 | +0.32 | +0.34 | -0.02 | +0.01 | +0.00 | -0.0005 | 0.972 | 1.000 |
| 1.20 | +0.23 | +0.23 | +0.24 | -0.01 | +0.01 | +0.00 | -0.0002 | 0.983 | 1.000 |
| 1.30 | +0.16 | +0.16 | +0.17 | -0.01 | +0.01 | +0.00 | -0.0001 | 0.990 | 1.000 |
| 1.40 | +0.12 | +0.12 | +0.12 | -0.01 | +0.00 | +0.00 | -0.0001 | 0.994 | 1.000 |
| 1.50 | +0.09 | +0.09 | +0.09 | -0.01 | +0.00 | +0.00 | -0.0000 | 0.996 | 1.000 |

Dissipation above the steady background flow, pulse and hold: flow within the pores 44.9 %, uniform dilatation 11.8 %, skeleton 43.3 %.
Peak compressive stress 15.81 Pa at t/T = 0.47 (stationary peak 14.47 Pa).
Largest tension in the pulse 0.72 Pa at t/T = 0.96: solid elastic +0.48, Kelvin-Voigt +0.14, fluid viscous +0.10 Pa; fluid content -0.0029 against the stationary cell.
Pore-pressure fluctuation at t/T = 0.25: -0.163 to 0.1 (solver units; background -0.00525 to 0.00495).

## What these numbers support (Section 5.2)

- The pores lag behind the imposed deformation: under load the cell holds more fluid than the stationary cell at the
  same compression, and less on unloading.
- The fluid must flow within the pores to follow: most of the extra fluid dissipation is deviatoric flow within the
  pores; the rest is the uniform dilatation lambda_p, by which fluid leaves or enters the cell.
- The lag adds to the stress: the fluid the cell still holds makes the solid carry more of the compression (the solid
  elastic part), and the fluid's viscous stress adds the rest. The stress peaks above the stationary one, just before
  mid-pulse, and the flux decreases less while the pores still hold more fluid (T = 2 s).
- On unloading the cell has not yet drawn its fluid back in, and the stress becomes tensile.
- Skeleton viscosity (T = 0.2 s) adds its stress directly, carrying the higher peak and most of the tension, and
  stiffens the pore walls against the pressure that drives the pore flow: the pore pressure is larger, yet the fluid
  content and flux stay closer to the stationary cell.
- The pore-pressure fluctuation has zero fluid mean, so it reaches the homogenised stress only through the solid walls
  (its own part, fluid pressure, is zero to round-off).
