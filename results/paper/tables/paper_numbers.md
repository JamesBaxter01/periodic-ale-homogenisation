# Numbers in the text of the paper, recomputed from the saved results

Written by `examples/paper_numbers.py`.


## Section 4: set-up

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Cell side ℓ (m) | 1 | 1 | yes |
| Channel width, of ℓ | 0.33 | 0.33 | yes |
| Initial porosity | 0.255 | 0.25483 | yes |
| Bulk modulus K (Pa) | 50 | 50 | yes |
| Shear modulus G (Pa) | 50 | 50 | yes |
| Fluid viscosity μ_f (Pa s) | 1 | 1 | yes |
| Skeleton viscosity η_s (Pa s) | 2 | 2 | yes |
| Skeleton viscosity ζ_s (Pa s) | 1 | 1 | yes |
| Discretisation order | 2 | 2 | yes |
| Durations T (s) | 0.2, 2 | 0.2, 2 | yes |
| Shear in the loading | 0.02 | 0.02 | yes |
| Compression in the loading | 0.2 | 0.2 | yes |
| Pressure amplitude (Pa) | 20 | 20 | yes |
| Gradient amplitude (Pa/m) | 10 | 10 | yes |
| Gradient direction | (1, 0.5, 0.25) | [1.0, 0.5, 0.25] | yes |
| Outputs sampled at t/T | 1/4, 1/2, 3/4, 1 | [0.25, 0.5, 0.75, 1.0] | yes |
| Tangents sampled at t/T | 1/2, 1 | [0.5, 1.0] | yes |
| Baseline h_max/ℓ | 0.09 | 0.09 | yes |
| Baseline elements | 5912 | 5912 | yes |
| Baseline unknowns: active_dofs, the size of K_h (the saved total_dofs, 43382, also counts periodic copies and fixed values) | 29226 | not from a saved run | – |
| Baseline steps N | 100 | 100 | yes |
| Baseline Newton tolerance | 1e-8 | 1e-08 | yes |
| Mesh-law ratio λ_m/μ_m | 5 | 5 | yes |
| Quadrature degree (the solver's default rule) | 5 | 5 | yes |

## Section 4.1: discretisation and algebraic errors

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Mesh levels | 6 | 6 | yes |
| Coarsest mesh h_max/ℓ | 0.15 | 0.15 | yes |
| Finest mesh h_max/ℓ | 0.03 | 0.03 | yes |
| Steps in the mesh study | 100 | 100 | yes |
| Time-step levels | 6 | 6 | yes |
| Fewest steps | 25 | 25 | yes |
| Most steps | 800 | 800 | yes |
| Mesh of the time-step study h_max/ℓ | 0.09 | 0.09 | yes |
| Newton tolerance of both studies | 1e-10 | 1e-10, 1e-10 | yes |
| Loosest tolerance compared | 1e-6 | 1e-06 | yes |
| Tightest tolerance compared | 1e-12 | 1e-12 | yes |
| Algebraic error at tolerance 1e-8, largest | 2.6e-8 | 2.6289e-08 | yes |
| Temporal order, smallest | 0.84 | 0.83945 | yes |
| Temporal order, largest | 1.07 | 1.0674 | yes |
| Stress and content error, largest total (%) | 1.01 | 1.0111 | yes |
| Flux error, largest total (%) | 3.72 | 3.7183 | yes |
| That flux error is mainly from the mesh (mesh, time step, %) | mainly the mesh | 3.47, 0.25 | yes |
| Flux total on the 0.06ℓ mesh (%, T = 0.2 s) | 1.3 | 1.3139 | yes |
| Flux total on the 0.06ℓ mesh (%, T = 2 s) | 1.9 | 1.9203 | yes |
| Cost per step, 0.06ℓ against 0.09ℓ (T = 0.2 s) | 5 | 5.3959 | yes |
| Cost per step, 0.06ℓ against 0.09ℓ (T = 2 s) | 6 | 6.2291 | yes |
| N = 100 to 200: change on the 0.045ℓ and 0.09ℓ meshes, for changes above 1e-4 | within 10 % | 8.7 % | yes |
| Smallest ALE Jacobian, baseline (T = 0.2 s) | 0.59 | 0.59268 | yes |
| Smallest ALE Jacobian, baseline (T = 2 s) | 0.37 | 0.36934 | yes |
| Finest mesh at T = 2 s inverts in the final increment | final increment | failed after 99 of 100 steps | yes |
| Final-time reference levels at T = 2 s | 0.09, 0.06, 0.045 | 0.09 0.06 0.045 | yes |
| Its mesh errors against the tabulated maxima (%) | below | 0.47 < 0.91, 0.32 < 0.39, 2.16 < 3.47 | yes |
| Newton tolerance of the mesh-law histories | 1e-10 | 1e-10 | yes |
| Mesh law: stress, content and full tangent, largest change | 6e-5 | 5.509e-05 | yes |
| Mesh law: flux, largest change (%) | 0.6 | 0.56452 | yes |
| Mesh law: change over the discretisation error, largest | below a quarter | 0.217 | yes |

## Section 4.2: algorithmic tangent

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Inputs perturbed | 13 | 13 | yes |
| Perturbations ρ | 10 | 10 | yes |
| Largest ρ | 1e-1 | 0.1 | yes |
| Smallest ρ | 1e-6 | 1e-06 | yes |
| Slope of e(ρ) at the largest perturbations | ρ² | 2.005 to 2.006 | yes |
| Minimum of e(ρ), smallest | 2.2e-10 | 2.2294e-10 | yes |
| Minimum of e(ρ), largest | 1.2e-9 | 1.16e-09 | yes |
| Scaled tangent blocks | 9 | 9 | yes |
| Largest block error (%, T = 0.2 s) | 0.30 | 0.29736 | yes |
| That block (T = 0.2 s) | stress/p | stress/p | yes |
| Largest block error (%, T = 2 s) | 1.22 | 1.2225 | yes |
| That block (T = 2 s) | content/p | content/p | yes |

## Section 4.3: power balance and constraints

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Power defect e_W, largest at N = 25 | 1.4e-3 | 0.0013834 | yes |
| Power defect e_W, largest at N = 800 | 4.6e-5 | 4.6071e-05 | yes |
| Power defect e_W, largest at N = 100 | 3.6e-4 | 0.00036358 | yes |
| Order of e_W in the time step (T = 0.2 s) | 1 | 0.98393 | yes |
| Order of e_W in the time step (T = 2 s) | 1 | 0.98321 | yes |
| Corrected residual, worst | 6.1e-8 | 6.1479e-08 | yes |
| It is set by the histories at Newton tolerance | 1e-6 | 1e-06 | yes |
| Corrected residual, worst at Newton tolerance 1e-8 | 5.9e-11 | 5.9492e-11 | yes |
| Peak storage power, of the stress power (%, T = 0.2 s) | 29 | 28.516 | yes |
| Peak flux power, of the stress power (%, T = 0.2 s) | 0.2 | 0.20025 | yes |
| Peak storage power, of the stress power (%, T = 2 s) | 70 | 69.909 | yes |
| Peak flux power, of the stress power (%, T = 2 s) | 1 | 1.0224 | yes |

## Section 5.1: Biot coefficient

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Pore pressures (Pa) | 1, 10, 20, 30 | 1, 10, 20, 30 | yes |
| Fixed cell against the published values, largest difference | within 0.015 | 0.0143 | yes |
| Uncertainty in reading the published figure | about 0.01 | not from a saved run | – |
| Periodic α above fixed, share of the computed cases | much of the range | 100 % | yes |
| The difference shrinks as K grows (p = 1 Pa: largest, at K, and at the largest K) | shrinks | 0.092 at K = 183 Pa, 0.008 at K = 10000 Pa | yes |
| Softest bulk modulus K (Pa) | 10 | 10 | yes |
| Its Poisson ratio | -0.44 | -0.4375 | yes |
| J_s where the law's hydrostatic stress stops increasing | 0.5 | 0.5002 | yes |
| Fixed cell at K = 10 Pa, p = 30 Pa | does not equilibrate | α = nan | yes |
| Threshold of the fraction below, J_s | 0.5 | 0.5 | yes |
| The compression run's periodic cell against Figure 3 (α) | the same | 0.4301, 0.4301 | yes |
| Solid below J_s = 0.5, periodic cell, settled | about 1–2 % | 1.24 % | yes |
| Solid below J_s = 0.5, fixed cell, end of the ramp | about 1–2 % | 1.19 % | yes |
| Solid below J_s = 0.5, fixed cell, last step before it fails | about 1–2 % | 2.08 % | yes |

## Section 5.2: compression pulse

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Peak compression | 0.20 | 0.2 | yes |
| Durations T (s) | 2, 0.2 | 2, 0.2 | yes |
| Steps per pulse | 100 | 100 | yes |
| Steps in the hold | 50 | 50 | yes |
| Pressure gradient (Pa/m) | -1 | -1 | yes |
| Pressure difference across the cell (Pa) | 1 | 1 | yes |
| Short pulse over the fluid viscous time | 10 | 10 | yes |
| Short pulse over the skeleton viscous time | 5 | 5 | yes |
| Increments of the stationary cell | 3 | 3 | yes |
| Length of each increment (s) | 1000 | 1000 | yes |
| Its outputs' change over the last increment, largest | below 1e-3 | 3.3e-06 | yes |
| T = 2 s: flux decreases less than the stationary cell (minima) | less | 0.27, 0.26 against 0.20 | yes |
| T = 2 s: stress peaks higher than the stationary cell (Pa) | higher | 15.7, 15.8 against 14.5 | yes |
| T = 2 s: the peak comes slightly before mid-pulse (t/T) | slightly before | 0.48, 0.46 | yes |
| T = 2 s: tensile near the end of unloading (Pa) | tensile | 0.62, 0.72 | yes |
| T = 0.2 s, elastic: the flux falls further than the stationary cell (minima) | further | 0.14 against 0.20 | yes |
| T = 0.2 s, elastic: the flux overshoots its initial value | overshoots | 1.13 | yes |
| T = 0.2 s: skeleton viscosity raises the peak stress (Pa) | raises | 23.7 against 19.3 | yes |
| T = 0.2 s: skeleton viscosity brings the peak earlier (t/T) | earlier | 0.36 against 0.46 | yes |
| T = 0.2 s: skeleton viscosity more than doubles the tension | more than doubles | × 2.28 | yes |
| T = 0.2 s: peak stress, elastic (Pa) | 19.3 | 19.278 | yes |
| T = 0.2 s: peak stress, viscoelastic (Pa) | 23.7 | 23.705 | yes |
| T = 0.2 s: tension during unloading, elastic (Pa) | 3.1 | 3.0939 | yes |
| T = 0.2 s: tension during unloading, viscoelastic (Pa) | 7.1 | 7.0681 | yes |
| T = 0.2 s: flux far closer to the stationary cell with skeleton viscosity (largest departure) | far closer | 0.064 against 0.353 | yes |
| T = 2 s: departure nearly the same for both skeletons | nearly the same | 0.072, 0.067 | yes |
| T = 2 s, elastic: flux departure at the end of the hold | -0.012 | -0.011755 | yes |
| T = 2 s, viscoelastic: flux departure at the end of the hold | -0.012 | -0.012484 | yes |
| Table 3 pulse durations (s), in the sweep | 0.2, 2, 20, 200 | 0.2, 0.355656, 0.632456, 1.12468, 2, 3.55656, 6.32456, 11.2468, 20, 35.5656, 63.2456, 112.468, 200 | yes |
| Flux departure at T = 20 s, elastic (%) | 1.3 | 1.3012 | yes |
| Flux departure at T = 20 s, viscoelastic (%) | 1.3 | 1.3279 | yes |
| Energy dissipated per pulse at T = 20 s, elastic (%) | 5.8 | 5.8075 | yes |
| Energy dissipated per pulse at T = 20 s, viscoelastic (%) | 7.5 | 7.4508 | yes |
| Energy per pulse below 1 % only at T = 200 s (at 200 s; at 112 s, the next shorter pulse) | below 1% only at 200 s | 0.62, 0.79; 1.11, 1.40 | yes |
| Elastic, T = 0.2 s, hold: fluid content at its start | 0.238 | 0.23771 | yes |
| Elastic, T = 0.2 s, hold: fluid content at its end | 0.252 | 0.25243 | yes |
| Elastic, T = 0.2 s, hold: axial tension at its start (Pa) | 2.6 | 2.5923 | yes |
| Elastic, T = 0.2 s, hold: axial tension at its end (Pa) | 0.42 | 0.42423 | yes |

## Section 5.3.1: set-up

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Column height H (m) | 1 | 1 | yes |
| Cell length ℓ (m) | 0.1 | 0.1 | yes |
| Cells in the column | 10 | 10 | yes |
| Step traction (Pa) | 5 | 5 | yes |
| Run stops when the pore pressure is below this fraction of its peak | 1e-3 | 0.001 | yes |
| Consolidation coefficient at rest c_v (m²/s), from Table 5 | 6.3e-4 | 0.00062509 | yes |
| Drainage time H²/c_v (s) | 1600 | 1599.8 | yes |
| Pressure time across a cell ℓ²/c_v (s) | 16 | 15.998 | yes |
| Final compression of the cells (%) | 7.5 | 7.5032 | yes |
| Fluid viscous time μ_f/G (s) | 0.02 | 0.02 | yes |
| Skeleton viscous time η_s/G (s) | 0.04 | 0.04 | yes |
| Resolved mesh, elements (elastic run) | 48537 | 48537 | yes |
| Resolved mesh, unknowns (elastic run) | 236206 | 236206 | yes |
| Resolved mesh, elements (viscoelastic run) | 48537 | 48537 | yes |
| Resolved mesh, unknowns (viscoelastic run) | 236206 | 236206 | yes |
| FE² macroscale elements | 10 | 10 | yes |
| FE² integration points | 40 | 40 | yes |
| FE² displacement order | 3 | 3 | yes |
| FE² pressure order | 2 | 2 | yes |
| Linear Biot law, 10 elements against 160: settlement | within about 1.5 % | 0.27 % | yes |
| Linear Biot law, 10 elements against 160: drained volume | within about 1.5 % | 0.60 % | yes |
| Linear Biot law, 10 elements against 160: pressure, cells 1 to 9 | within about 1.5 % | 0.07 % | yes |
| Linear Biot law, 10 elements against 160: pressure, top cell | within about 1.5 % | 1.42 % | yes |
| First time step (s) | 1e-6 | 1e-06 | yes |
| Growth per step (%) | 5 | 5 | yes |
| Steps (elastic) | 415 | 415 | yes |
| End of the run (s, elastic) | 3933 | 3933 | yes |
| Steps halved late in the drainage (elastic) | at most 4 | 3 | yes |
| Steps (viscoelastic) | 424 | 424 | yes |
| End of the run (s, viscoelastic) | 4361 | 4361.4 | yes |
| Steps halved late in the drainage (viscoelastic) | at most 4 | 4 | yes |

## Section 5.3.1: step-load response

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Elastic settlement at 10 ms, resolved (mm) | 32.5 | 32.531 | yes |
| Viscoelastic time to settle as far (ms) | 60 | 62.128 | yes |
| Skeletons' settlements at t = 0.3 s, relative difference (below 0.5 %) | coincide | 0.13 % | yes |
| Settlement at t = 1 s (mm, elastic) | 41 | 41.062 | yes |
| Drained at t = 1 s, of the final volume (%, elastic) | 1.5 | 1.5111 | yes |
| Settlement at t = 1 s (mm, viscoelastic) | 41 | 41.042 | yes |
| Drained at t = 1 s, of the final volume (%, viscoelastic) | 1.5 | 1.4986 | yes |
| Viscoelastic pore pressure does not overshoot the traction (resolved, largest cell over traction) | no overshoot | 0.9601 | yes |
| Settlement difference, largest (%) | 0.86 | 0.86221 | yes |
| Drained volume difference, largest (%) | 1.63 | 1.6298 | yes |
| Pressure difference, cells 1 to 9, largest | 0.027 | 0.027314 | yes |
| Pressure statistics start (ms) | 2 | 2 | yes |
| Cell thickness over the drainage layer at 2 ms | 100 | 89.436 | yes |
| Undrained value 1/α, from Table 5 | 2.1 | 2.1232 | yes |
| Elastic pressure peak over traction, FE² | 1.97 | 1.974 | yes |
| Elastic pressure peak over traction, resolved | 1.78 | 1.7847 | yes |

## Section 5.3.1: load ramped over 10 s (elastic)

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Ramp time (s) | 10 | 10 | yes |
| Settlement difference (%) | 0.72 | 0.71773 | yes |
| Drained volume difference (%) | 1.37 | 1.3673 | yes |
| Pressure difference, cells 1 to 9 | 0.021 | 0.021278 | yes |
| Pressure difference, top cell | 0.075 | 0.075231 | yes |

## Section 5.3.1: Figure 7 (cell at the base, viscoelastic)

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Early, time (s) | 0.0098 | 0.0097991 | yes |
| Early, F_33 | 0.990 | 0.99022 | yes |
| Undrained, time (s) | 10 | 10.022 | yes |
| Undrained, F_33 | 0.959 | 0.95907 | yes |
| Undrained, pore pressure over traction (%) | 96 | 95.916 | yes |
| Drained, time (s) | 4361 | 4361.4 | yes |
| Drained, F_33 | 0.925 | 0.925 | yes |

## Section 5.3.1: top cell, storage moment and refined reference

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Top-cell pressure difference, largest (elastic) | 0.10 | 0.099052 | yes |
| Time of that difference (s, elastic) | 2.7 | 2.6843 | yes |
| Settlement difference peaks at (s, elastic) | 2 | 1.9077 | yes |
| Drained-volume difference peaks at (s, elastic) | 2 | 2.0031 | yes |
| Top-cell difference below 1e-3 after (s, elastic) | 60 | 55.281 | yes |
| Top-cell pressure difference, largest (viscoelastic) | 0.10 | 0.098008 | yes |
| Time of that difference (s, viscoelastic) | 2.8 | 2.8186 | yes |
| Settlement difference peaks at (s, viscoelastic) | 2 | 1.9077 | yes |
| Drained-volume difference peaks at (s, viscoelastic) | 2 | 2.0031 | yes |
| Top-cell difference below 1e-3 after (s, viscoelastic) | 60 | 55.281 | yes |
| Storage moment rate over flux at 10 ms | 0.56 | 0.56276 | yes |
| Storage moment rate over flux, largest from 0.1 to 3 s | 0.10 | 0.10174 | yes |
| Finer resolved mesh, h_max/ℓ | 0.075 | 0.075 | yes |
| Finer resolved mesh, elements | 114520 | 114520 | yes |
| Change of the reference, settlement (%) | 0.50 | 0.50025 | yes |
| Change of the reference, drained volume (%) | 1.27 | 1.2733 | yes |
| FE² against the finer reference, settlement (%) | 0.68 | 0.68499 | yes |
| FE² against the finer reference, drained volume (%) | 1.55 | 1.552 | yes |
| FE² against the finer reference, top-cell pressure | 0.093 | 0.093063 | yes |

## Section 5.3.1: history-free Biot law and drained state

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Fitted cross term C_1122 (Pa) | 7.78 | 7.7751 | yes |
| Fitted shear modulus (Pa) | 28.0 | 28.023 | yes |
| Cell shear stiffness C_1313, Table 5 (Pa) | 19.94 | 19.943 | yes |
| Biot law, drained volume difference (%) | 1.52 | 1.5161 | yes |
| FE², drained volume difference (%) | 1.63 | 1.6298 | yes |
| Biot law, largest settlement difference (%) | 57 | 57.378 | yes |
| Drained lateral permeability below rest (%) | 35 | 34.797 | yes |
| Drained axial permeability against rest | within 1 % | 0.65 % | yes |

## Section 5.3.2: set-up

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Cycles | 5 | 5 | yes |
| Increments per cycle | 40 | 40 | yes |
| Growth of the mean settlement after the first cycle, largest (mm) | less than 0.7 | 0.666 | yes |
| T = 20 and 200 s: still consolidating (growth of the mean settlement after the first cycle, smallest, mm) | still consolidating | 1.99 | yes |
| T = 20 and 200 s: work of the mean traction on the settlement over the last cycle, the same in FE² with either kind of cell (largest relative spread) | the same | 0.11 % | yes |
| Macroscopic residual, stationary run at T = 0.2 s | 1e-6 | 1e-06 | yes |
| Macroscopic residual, the other runs | 1e-8 | 1e-08 | yes |
| Stationary run at T = 0.2 s: tolerance of its resumed segments | 1e-6 | 1e-06 | yes |
| Stationary run at T = 0.2 s: largest residual of an accepted step | at most 1e-6 | 6.3e-07 | yes |
| That residual against the departure from the transient cells (smallest settlement difference) | far below | 6.3e-07 against 0.147 | yes |

## Section 5.3.2: transient cells against the resolved column

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Settlement difference over all increments, of the resolved amplitude (%, all four periods) | 1.6 | 1.5658 | yes |
| Pressure difference, cells 1 to 9, largest (all four periods) | 0.021 | 0.021032 | yes |
| Pressure difference, top cell, largest (all four periods) | 0.081 | 0.080656 | yes |
| Last-cycle amplitude difference, largest (%, all four periods) | 1.8 | 1.7864 | yes |
| Last-cycle phase lag difference, largest (degrees, all four periods) | within 0.26 | 0.259 | yes |
| Energy excess, smallest (mJ/m²) | 0.6 | 0.58219 | yes |
| Energy excess, largest (mJ/m²) | 0.9 | 0.87296 | yes |
| Relative energy excess, largest (%) | 10 | 9.5937 | yes |
| The largest relative excess is at T = 2 s | T = 2 s | T = 2 s | yes |
| Layer √(c_v T) at T = 0.2 s (mm) | 11 | 11.181 | yes |
| Layer √(c_v T) at T = 2 s (mm) | 35 | 35.358 | yes |
| Share of the excess above the top cell's centre, smallest (%) | 65 | 65.309 | yes |
| Share of the excess above the top cell's centre, largest (%) | 79 | 79.312 | yes |
| Dissipation of λ_p, smallest (mJ/m²) | 0.07 | 0.068202 | yes |
| Dissipation of λ_p, largest (mJ/m²) | 0.18 | 0.18305 | yes |

## Section 5.3.2: effect of skeleton viscosity (resolved)

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Skeleton viscosity matters most at the shortest period (energy, viscoelastic over elastic, per period) | shortest period | 0.2 s: × 2.89, 2 s: × 2.57, 20 s: × 1.18, 200 s: × 1.01 | yes |
| T = 0.2 s, amplitude reduction (%) | 29 | 28.585 | yes |
| T = 0.2 s, lag, elastic (degrees) | 10 | 10.096 | yes |
| T = 0.2 s, lag, viscoelastic (degrees) | 45 | 45.073 | yes |
| T = 0.2 s, energy nearly triples | nearly triples | × 2.89 | yes |
| T = 2 s, amplitude difference (%) | 1 | 0.91311 | yes |
| T = 2 s, lag and energy more than double | more than double | × 2.61, × 2.57 | yes |

## Section 5.3.2: stationary cells against the resolved column

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| T = 0.2 s, elastic: amplitude above resolved (%) | 17 | 16.735 | yes |
| T = 0.2 s, elastic: energy, of the resolved (%) | 1.3 | 1.3311 | yes |
| T = 0.2 s, viscoelastic: amplitude above resolved (%) | 63 | 63.461 | yes |
| T = 0.2 s, viscoelastic: energy, of the resolved (%) | 0.5 | 0.46122 | yes |
| T = 2 s, elastic: amplitude above resolved (%) | 4.8 | 4.7558 | yes |
| T = 2 s, elastic: energy, of the resolved (%) | 20 | 20.097 | yes |
| T = 2 s, viscoelastic: amplitude above resolved (%) | 5.7 | 5.7211 | yes |
| T = 2 s, viscoelastic: energy, of the resolved (%) | 8 | 7.8109 | yes |
| T = 2 s, lag a fifth or less of the resolved | a fifth or less | 0.192, 0.074 | yes |
| Elastic, energy missed at T = 2 s (%) | 80 | 79.903 | yes |
| Elastic, energy missed at T = 0.2 s (%) | 99 | 98.669 | yes |
| T = 20 s, elastic: energy missed by stationary cells, of the resolved (%) | 25 | 24.896 | yes |
| T = 20 s, viscoelastic: energy missed by stationary cells, of the resolved (%) | 36 | 36.464 | yes |
| T = 200 s, elastic: energy missed by stationary cells, of the resolved (%) | 5 | 5.3158 | yes |
| T = 200 s, viscoelastic: energy missed by stationary cells, of the resolved (%) | 6 | 6.1339 | yes |
| T = 20 and 200 s: energy missed by transient cells, smallest (%) | 2 | 1.7824 | yes |
| T = 20 and 200 s: energy missed by transient cells, largest (%) | 4 | 4.2119 | yes |
| T = 20 and 200 s: stationary settlement difference over all increments, largest, of the resolved amplitude (%) | 1.9 | 1.9129 | yes |

## Section 5.3.3: computational cost

| Quantity | Paper | Recomputed | Agrees |
|---|---|---|---|
| Replayed steps (elastic) | 415 | 415 | yes |
| Replay time on one core, on this machine (elastic) | less than 1 s | 0.71 s | – |
| Replay against the saved solution, largest relative difference (elastic) | to the macroscopic Newton tolerance | 2.9e-06 | yes |
| Replayed steps (viscoelastic) | 424 | 424 | yes |
| Replay time on one core, on this machine (viscoelastic) | less than 1 s | 0.63 s | – |
| Replay against the saved solution, largest relative difference (viscoelastic) | to the macroscopic Newton tolerance | 2.6e-08 | yes |
| The Aire node: two 84-core AMD EPYC 9634 processors | as stated | not from a saved run | – |
| FE² cell increment on one core (s, elastic) | 46 | 45.59 | yes |
| FE² cell calls (elastic) | 49280 | 49280 | yes |
| FE² cores (elastic) | 40 | 40 | yes |
| FE² wall time (h, elastic) | 17.1 | 17.07 | yes |
| FE² core-hours (elastic) | 680 | 682.82 | yes |
| FE² cell increment on one core (s, viscoelastic) | 51 | 50.97 | yes |
| FE² cell calls (viscoelastic) | 50000 | 50000 | yes |
| FE² cores (viscoelastic) | 40 | 40 | yes |
| FE² wall time (h, viscoelastic) | 19.8 | 19.77 | yes |
| FE² core-hours (viscoelastic) | 790 | 790.78 | yes |
| Resolved cores (elastic) | 8 | 8 | yes |
| Resolved wall time (h, elastic) | 9.5 | 9.4967 | yes |
| Resolved core-hours (elastic) | 76 | 75.973 | yes |
| Resolved cores (viscoelastic) | 8 | 8 | yes |
| Resolved wall time (h, viscoelastic) | 11.5 | 11.546 | yes |
| Resolved core-hours (viscoelastic) | 92 | 92.367 | yes |
| Increments of a cyclic run | 200 | 200 | yes |
| Cyclic FE², shortest (h) | 8.1 | 8.0852 | yes |
| Cyclic FE², longest (h) | 9.6 | 9.6177 | yes |
| Cyclic resolved, shortest (h) | 5.4 | 5.4483 | yes |
| Cyclic resolved, longest (h) | 8.1 | 8.1206 | yes |
| Long increments of a stationary cell evaluation | 3 | 3 | yes |
| Stationary against transient FE², wall time (T = 2 s) | 4 | 4.0097 | yes |
| Stationary against transient FE², wall time (T = 0.2 s) | 6 | 5.8378 | yes |
