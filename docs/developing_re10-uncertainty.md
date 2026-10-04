# Developing flow, Re_h = 10: how wide the band is, and what moves a result out of it

All errors are relative L2 against the finite-element reference, integrated
on its own mesh (`reference.relative_errors`); entrance length is a relative
error. Every polish run is 100 Levenberg-Marquardt steps warm-started from a
finished network, selecting on 32,768 held-out points with the mean of the
last 20 iterates as a candidate, unless stated otherwise.

## 1. The band: the same polish on three seeds

G4 + averaged polish, seeds 1-3 (other training draws, other held-out set):

| | u | v | p | entrance length | total-pressure drop | dissipation | energy balance |
|---|---|---|---|---|---|---|---|
| mean | 7.75e-6 | 1.48e-4 | 2.01e-5 | 3.5e-5 | 5.3e-6 | 1.06e-5 | 3.0e-6 |
| range | 7.4-8.2e-6 | 1.36-1.67e-4 | 1.96-2.03e-5 | 1.3-5.9e-5 | 4.8-6.2e-6 | 1.01-1.14e-5 | 1.4-5.6e-6 |
| spread | +-5% | +-11% | +-2% | 4.6x | +-13% | +-6% | 4x |

The entrance length is set where the centreline slope is ~0.02, so it
amplifies a local centreline error ~28x; it and the energy balance cannot rank
runs at the 1e-5 level without several seeds.

## 2. What moves a result out of the band

| change (against the band above) | u | v | total-pressure drop | verdict |
|---|---|---|---|---|
| no averaging, 4,096 held-out points (control) | 1.18e-5 | 2.12e-4 | 1.43e-5 | far worse: selection was a lottery |
| 16,384 points per step instead of 4,096 | 6.08e-6 | 1.32e-4 | 5.97e-6 | better on u, v: outside the band |
| start from G3 instead of G4 | 4.62e-6 | 1.36e-4 | 1.97e-6 | much better: G3's network is better, not luckier |
| G3 and 16,384 points per step (P5) | **3.59e-6** | **1.00e-4** | **1.12e-6** | best: the two levers add up |
| focus fraction 0.75 instead of 0.5 | 7.72e-6 | 1.98e-4 | 1.11e-5 | u same, v and objective worse |
| focus fraction 0.25 | - | - | - | unstable: no step beat the start |
| focus fraction 0 (uniform) | - | - | - | unstable: held-out loss x1,000 in one step |

## 3. Why sampling near the inlet corners is not optional

Held-out loss along the polish, against its starting value:

| focus fraction | steps > 10x the start | median iterate / start |
|---|---|---|
| 0 (uniform) | 100% | 2.2e4 |
| 0.25 | 94% | 85 |
| 0.5 | 4% | 1.6 |
| 0.75 | 1% | 0.93 |

With uniform sampling the inlet corners (0.025% of the area) get about one
point per 4,096-point draw, so each Gauss-Newton step is free to wreck the
corner flow. Yet at focus 0.5, 64% of the training loss lies in that 0.025%,
and a 4,096-point held-out estimate of it has a 14.7% relative standard
deviation (5% at 32,768) - larger than the differences between plateau
iterates, which is why selection needs many held-out points and averaging.

## 4. The reference's own limits

u 8.7e-7, v 9.8e-6, p 1.5e-5, total-pressure drop 6.7e-8 (GCI). Every
top run's pressure error (1.9-2.3e-5) is within 1.5x of the reference's, so
pressure cannot rank them; its error lies 44% in the inlet strip x* < 0.05
and 45% in the bulk, so only more elements, not grading, would lower it.

## 5. Best so far (P5: G3 + 100 averaged LM steps at 16,384 points)

u 3.59e-6 (entrance 4.36e-6, developed 3.06e-6), v 1.00e-4, p 2.06e-5 (at the
reference's own limit), total-pressure drop 1.12e-6, dissipation 2.2e-6,
energy balance 3.3e-6. Against G3 alone: u -34%, v -36%, objective 10x.
It selected iterate 94, not a mean: its held-out loss was still falling at
step 100, so `re10_polish_g3_more` continues it for 150 steps (seed 2).

## 6. The input map (G5 = G3 with the exponential input map, l = 2)

| | u | v | p | total-pressure drop | dissipation | energy balance |
|---|---|---|---|---|---|---|
| G3, linear map | 5.42e-6 | 1.57e-4 | 2.22e-5 | 1.10e-5 | 5.3e-6 | 1.23e-5 |
| G5, exponential map | 3.63e-6 | 1.08e-4 | 1.71e-5 | 1.07e-6 | 3.2e-6 | 1.6e-6 |

Unpolished, G5 matches P5 (the best polished network) and has the lowest p of
any run, near the reference's 1.5e-5. It was selected on 4,096 held-out points
(fresh/held-out loss 1.25), so its averaged polish (`re10_polish_g5`, same
protocol as `re10_polish_g3`) is the fair comparison; `re10_polish_g5_points`
then combines the input map with the 16,384-point polish.

## 7. Lower loss, higher error: the polish of G5

| | u | u, entrance | u, developed | v | p | total-pressure drop | fresh-point loss |
|---|---|---|---|---|---|---|---|
| G5 | 3.63e-6 | 4.45e-6 | 3.05e-6 | 1.08e-4 | 1.71e-5 | 1.07e-6 | 7.3e-9 |
| G5 + averaged polish (P6) | 4.59e-6 | 3.96e-6 | 4.92e-6 | 9.2e-5 | 1.67e-5 | 1.42e-6 | 4.4e-9 |
| G3 + the same polish (P3) | 4.62e-6 | 6.58e-6 | 2.91e-6 | 1.36e-4 | 2.01e-5 | 1.97e-6 | 1.1e-8 |

The 4,096-point polish took G5 to a lower loss and raised its u error, all of
it in the developed region. With the polish protocol held equal (P3 vs P6) the
exponential map improves v 32%, p 17%, the total-pressure drop 28% and leaves
u unchanged.

**Correction (after P7).** I first put the developed-region loss down to the
loss weighting the inlet corners 64%. P7 (below) lowered the loss further and
kept the developed region at G5's level, so a lower loss does not inherently
cost developed-flow accuracy. The better explanation is sampling noise: at
4,096 points a draw holds ~1,200 points in the developed region, too few to pin
it down in each step, and the polish's scatter lands there.

## 8. G5 + averaged polish at 16,384 points (P7): best field errors

| | u | u, entrance | u, developed | v | p | total-pressure drop | dissipation | fresh-point loss |
|---|---|---|---|---|---|---|---|---|
| G5 | 3.63e-6 | 4.45e-6 | 3.05e-6 | 1.08e-4 | 1.71e-5 | 1.07e-6 | 3.2e-6 | 7.3e-9 |
| P7 | **3.08e-6** | 2.98e-6 | 3.14e-6 | **6.85e-5** | 1.72e-5 | 3.85e-6 | 4.2e-6 | 3.5e-9 |

Best u and v of any run, p at the reference's limit. But the total-pressure
drop error rose 3.6x, beyond the +-13% seed band: the design objective is an
integral with cancellations, and its error does not follow the field errors.
For the bifurcation study, whose objective it is, it must be ranked on its own
and with seeds.

## 9. The objective's error is mostly a flux error, and normalising removes it

A PINN carries its flux only to ~1e-6 (soft inlet, soft continuity); 7 of 8
LM networks carry 1-6e-6 too much. Dissipation scales as Q^2, and its
relative error is close to twice each network's mean flux error (G3 6.5e-6
predicted / 5.3e-6 measured, G4 8.9 / 10.0, P5 2.2 / 2.2, P6 3.0 / 3.0, P7
4.4 / 4.2, in 1e-6). Normalising by the network's own mean flux cancels it:

| over G3, G4, G5, P3, P5-P8 | mean (bias) | rms | worst |
|---|---|---|---|
| dissipation | +3.7e-6 | 5.1e-6 | 1.0e-5 |
| **resistance = dissipation / Q^2** | **-4.6e-7** | **1.2e-6** | 2.5e-6 |
| total-pressure drop (boundary fluxes) | +3.1e-6 | 4.5e-6 | 1.1e-5 |
| total-pressure drop / Q | +9.5e-7 | 3.1e-6 | 7.7e-6 |

The total-pressure drop improves less because it is computed from boundary
energy fluxes, and equals dissipation / q_in only up to the network's energy
imbalance (1e-6 to 1e-5). For the bifurcation study: evaluate the objective
from the volume dissipation, normalised by the measured flux and rescaled to
the prescribed one, not from boundary pressures. P5's resistance is within
4e-8 of the reference's, P6's 1.2e-7, P7's 3.4e-7. `energy.budget` now
reports it and the summary ranks it (reference floor: the dissipation GCI,
1.8e-7, since the reference's flux is exact to 5e-8).

## 10. Continuing P5 for 150 steps (P8)

The loss kept falling (held-out medians by 25-step block 9.0, 8.2, 7.5, 7.2,
7.0, 6.6 in 1e-9) while u moved 2% (3.59e-6 to 3.52e-6, inside the seed band)
and v 3%: a plateau in the field errors, not in the loss.

## 11. P7 on three seeds: the band of the best network

The same polish of G5 (100 averaged LM steps at 16,384 points, 32k
held-out) on seeds 1, 2 and 3. Relative errors against the reference,
integrated on its mesh:

| P7 | seed 1 | seed 2 | seed 3 | mean | max / min |
|---|---|---|---|---|---|
| u | 3.08e-6 | 4.00e-6 | 2.71e-6 | 3.26e-6 | 1.48 |
| v | 6.84e-5 | 7.19e-5 | 7.22e-5 | 7.08e-5 | 1.05 |
| p | 1.72e-5 | 1.78e-5 | 1.68e-5 | 1.73e-5 | 1.06 |
| entrance length | 4.24e-5 | 1.94e-5 | 4.18e-5 | 3.45e-5 | 2.18 |
| total-pressure drop | 3.85e-6 | 7.21e-6 | 5.5e-7 | 3.87e-6 | 13.1 |
| resistance | 3.39e-7 | 8.06e-7 | 1.87e-6 | 1.01e-6 | 5.5 |
| training (min) | 72 | 75 | 75 | 74 | |

The fields are tight: v and p vary by 5-6% between seeds, and p sits on the
reference's own floor (1.5e-5), so p cannot be ranked further. u varies by
1.5x, like the baseline band (section 1). The objectives are not tight:
the total-pressure drop spans 13x and the resistance 5.5x, so a single
seed's objective error is a draw from a range, not a property of the setup.
Seed 1's resistance (3.4e-7, quoted in section 9) was the lucky end; the
honest figure for P7 is ~1e-6, and P5's single-seed 4.4e-8 is most likely
selection luck too (one seed; section 1 showed the same of the baseline).

For the bifurcation: a single trained network's pressure drop should be
quoted with a seed band, and comparisons between designs closer than about
1e-5 relative are not resolved by one network per design.
