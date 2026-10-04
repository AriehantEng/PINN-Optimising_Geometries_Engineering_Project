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
| start from G3 instead of G4 | **4.62e-6** | 1.36e-4 | **1.97e-6** | much better: G3's network is better, not luckier |
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
