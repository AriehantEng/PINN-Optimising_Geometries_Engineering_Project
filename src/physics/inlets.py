"""Case B inlet profiles, and why the obvious one cannot be solved.

The natural developing-flow inlet is a top hat: u*(0, y) = U for every y. It is
also incompatible with the ansatz. No-slip is imposed exactly by writing
u = phi(y) N_u with phi = 1 - y^2, so matching a top hat at x* = 0 requires

    N_u(0, y) = U / (1 - y^2)

which is UNBOUNDED at y = +-1. The network is being asked for infinity at two
points, and it cannot comply: measured on runs/caseB_re10, the momentum-x
residual reaches 525 at the inlet corner against 3.8e-06 for Case A, and the
loss floors at 2.45e-03 against 8.35e-13.

The regularised family
----------------------

    u*(0, y) = U0 (1 - y^(2m)),    U0 = (2/3)(2m+1)/(2m)

with U0 chosen so the flux is exactly 4/3 for every m -- the same flux the top
hat carries, so nothing else in the problem changes.

What makes this family the right one is the exact factorisation

    1 - y^(2m) = (1 - y^2) (1 + y^2 + ... + y^(2m-2))

so that

    N_u(0, y) = U0 (1 + y^2 + ... + y^(2m-2))

is a BOUNDED polynomial for every finite m, with maximum U0*m at the wall. The
singularity is precisely the m -> infinity limit, and it is approached linearly
in m. The wall layer has thickness ~1/(2m).

It is also the physically honest choice. A real inlet has a finite boundary
layer; the top hat is the idealisation, and recovering it as a limit is more
defensible than imposing it and then fighting the consequences.

DEGENERACY -- m = 1
-------------------

At m = 1, U0 = 1 and the profile is 1 - y^2: the fully developed Case A
solution. Case B at m = 1 is ALREADY DEVELOPED at its inlet, so there is no
entrance region, the entrance length is zero, and the case tests nothing. It is
registered in `DEGENERATE_SHARPNESS` and reported by the sentinel matrix.
"""

from __future__ import annotations

import jax.numpy as jnp

#: Sharpness values at which the inlet IS the developed profile, so the case is
#: vacuous. m = 1 gives U0 = 1 and u = 1 - y^2 exactly.
DEGENERATE_SHARPNESS = frozenset({1})

#: The flux every profile in this family carries, by construction.
FLUX = 4.0 / 3.0


def peak(sharpness: int) -> float:
    """U0 such that the profile carries flux 4/3.

    int_{-1}^{1} (1 - y^(2m)) dy = 2 - 2/(2m+1) = 2 (2m)/(2m+1), so
    U0 = (4/3) / (2 (2m)/(2m+1)) = (2/3)(2m+1)/(2m).

    m = 1 -> 1 (the parabola); m -> infinity -> 2/3 (the top hat).
    """
    _validate(sharpness)
    return (2.0 / 3.0) * (2 * sharpness + 1) / (2 * sharpness)


def profile(y, sharpness: int):
    """u*(0, y) = U0 (1 - y^(2m))."""
    _validate(sharpness)
    return peak(sharpness) * (1.0 - y ** (2 * sharpness))


def network_value_at_wall(sharpness: int) -> float:
    """max |N_u(0, y)| = U0 * m, the demand the ansatz places on the network.

    Diverges linearly in m, which is the quantitative statement of how the top
    hat becomes unsolvable.
    """
    _validate(sharpness)
    return peak(sharpness) * sharpness


def wall_layer_thickness(sharpness: int) -> float:
    """Approximate thickness of the near-wall transition, ~1/(2m)."""
    _validate(sharpness)
    return 1.0 / (2.0 * sharpness)


def _validate(sharpness) -> None:
    if not isinstance(sharpness, (int,)) or isinstance(sharpness, bool):
        raise TypeError(
            f"inlet sharpness must be an integer, got {sharpness!r}. The "
            "profile is 1 - y^(2m) and a non-integer m is not a polynomial, "
            "so the bounded-N_u argument would not hold."
        )
    if sharpness < 1:
        raise ValueError(
            f"inlet sharpness must be at least 1, got {sharpness}. m = 1 is "
            "the developed parabola; larger m sharpens toward the top hat."
        )
