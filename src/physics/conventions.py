"""Reynolds-number and length-scale conventions (SPEC v2, S1).

Two Reynolds numbers live in this project and they are NOT equal.

Solver convention -- half-height `h`, centreline velocity `U_max`:

    Re_h = rho * U_max * h / mu

Correlation convention -- hydraulic diameter `D_h`, mean velocity `U_mean`:

    Re_Dh = rho * U_mean * D_h / mu

For a wide plane channel, D_h = 4A/P = 4(2hw)/(2w) = 4h, and for the parabolic
profile U_mean = (2/3) U_max, so

    Re_Dh = (8/3) Re_h

and the reference LENGTHS differ by a factor of four (D_h = 4h). Chained
through a correlation that is linear in Re, that is roughly an order of
magnitude -- which is the signature that showed up as a 289% entrance-length
miss while every other quantity from the same solution was within a few
percent.

Rule enforced by naming throughout this module: every function that consumes a
Reynolds number says WHICH ONE in its name. A parameter called plain `re` is a
bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Re_Dh / Re_h for a wide plane channel. Derivation in the module docstring.
RE_DH_OVER_RE_H = 8.0 / 3.0

#: D_h / h for a wide plane channel.
D_H_OVER_H = 4.0

#: U_mean / U_max for the parabolic profile.
U_MEAN_OVER_U_MAX = 2.0 / 3.0


class MissingCitationError(ValueError):
    """Raised when a correlation is declared without a source citation."""


@dataclass(frozen=True)
class EntranceLengthCorrelation:
    """Blended Durst-type correlation, in HYDRAULIC-DIAMETER units.

        L_e / D_h = [ c0^n + (c1 * Re_Dh)^n ]^(1/n)

    The constants differ between pipe and plane-channel geometry and must be
    read from the source, never assumed. `citation` is mandatory: a wrong
    constant here is invisible in the output, whereas a missing citation is
    not.
    """

    c0: float
    c1: float
    n: float
    citation: str

    def __post_init__(self):
        if not isinstance(self.citation, str) or not self.citation.strip():
            raise MissingCitationError(
                "an entrance-length correlation must carry a non-empty "
                "'citation'. The constants differ between pipe and plane-"
                "channel geometry and a wrong one is invisible in the output."
            )
        for name in ("c0", "c1", "n"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"correlation.{name} must be finite and positive, got {value!r}"
                )

    def length_over_dh(self, reynolds_dh):
        """L_e / D_h."""
        return (self.c0**self.n + (self.c1 * reynolds_dh) ** self.n) ** (1.0 / self.n)


# --------------------------------------------------------------------------- #
# Reynolds numbers
# --------------------------------------------------------------------------- #


def reynolds_h(rho, u_max, h, mu):
    """Solver convention: Re_h = rho U_max h / mu."""
    return rho * u_max * h / mu


def reynolds_dh(rho, u_mean, d_h, mu):
    """Correlation convention: Re_Dh = rho U_mean D_h / mu."""
    return rho * u_mean * d_h / mu


def re_h_to_re_dh(re_h):
    """Re_Dh = (8/3) Re_h. Exact for a wide plane channel."""
    return RE_DH_OVER_RE_H * re_h


def re_dh_to_re_h(re_dh):
    """Inverse of `re_h_to_re_dh`."""
    return re_dh / RE_DH_OVER_RE_H


# --------------------------------------------------------------------------- #
# Entrance length
# --------------------------------------------------------------------------- #


def entrance_length_over_h(re_h, correlation: EntranceLengthCorrelation):
    """L_e / h, converting conventions internally.

    Takes the SOLVER Reynolds number, converts to the correlation's convention,
    evaluates, and converts the length back:  L_e/h = 4 * (L_e/D_h).

    Returning in units of `h` means call sites never touch D_h and cannot get
    the factor of four wrong.
    """
    return D_H_OVER_H * correlation.length_over_dh(re_h_to_re_dh(re_h))
