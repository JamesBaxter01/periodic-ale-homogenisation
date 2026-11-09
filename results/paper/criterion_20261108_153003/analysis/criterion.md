# History criterion, step 1 (cell sweep), 8 Nov 2026

Jobs on Calder: 847830 (elastic), 847831 (viscoelastic), 847832 (40 steps). Commit edff39c plus the
uncommitted branch `study/history-criterion`; solver sha 60df5ec4, the same as the paper data.
Pulse: raised cosine to 20 % compression, 100 steps, then a hold of half the pulse; T = 0.2–200 s (13 values).
Scripts in this folder are run with the `fem` env, and each takes the criterion folder as its argument.

## Checks
- The viscoelastic T = 0.2–20 s pulses reproduce `figure4_rates.npz` to 2e-9 (relative).
- 40 vs 100 steps per pulse (T = 2 and 20 s, both skeletons): departures within 1.1 %, peak stress
  within 0.4 %, loop energy within 2.5 %, end fluid content the same (`compare_steps.py`).
- Flux floor: the even part of the flux departure (mean of loading and unloading) tends to
  1.2e-3 of the rest flux at peak compression, the same for both skeletons and independent of T. The
  stationary cell reaches each compression in a single 1000 s step, and the transient cell in 100. This
  floor is a difference between the two cells' discretisations, not history. The rate-odd part follows
  1/T exactly (`floor.py`).

## Result (`asymptote.py`, `departure_vs_T.png`)
Each departure X of the stationary cell satisfies X <= C/T over the whole sweep. C is the 1/T coefficient,
reached from T of about 60τ. Units: τ = 0.7 s, μ_f/G = 0.02 s.

| measure | elastic C | viscoelastic C | X <= 1 % for T >= |
|---|---|---|---|
| flux, max over pulse (rate-odd) | 0.25 s = 0.36τ = 13 μ_f/G | 0.26 s = 0.37τ = 13 μ_f/G | 25–26 s (36τ) |
| stress, max over pulse / stationary peak | 0.44 s = 0.63τ = 22 μ_f/G | 0.53 s = 0.75τ = 26 μ_f/G | 44–53 s (63–75τ) |
| energy lost per pulse / stored work | 1.25 s = 1.8τ = 62 μ_f/G | 1.58 s = 2.25τ = 79 μ_f/G | 125–158 s (180–225τ) |

- Skeleton viscosity leaves the flux coefficient unchanged and raises the stress and energy
  coefficients by 20–27 %. That is about π²η_s/G = 0.39 s of extra loss, against 0.33 s observed.
- Hold relaxation fits give τ = 0.5–0.75 s for T = 0.6–11 s, both skeletons. This is consistent with the
  paper's "about 0.7 s".
- Caveats: one cell, one amplitude, one pulse shape. The 1/T form is general (first-order lag of a
  relaxing cell), but the coefficients are specific to this cell.
