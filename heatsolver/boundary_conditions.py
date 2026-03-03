"""
Factory functions for common boundary / initial condition profiles.

Each function returns a callable f(s) that works for both scalar and
array inputs, so the same factory can be used for:
  - initial conditions  u(x, 0) = f(x)   where s = x (array)
  - boundary conditions u(0, t) = f(t)   where s = t (scalar)
"""

import numpy as np
from typing import Callable


def constant(value: float) -> Callable:
    """Return f(s) = value."""
    return lambda s: value


def linear(slope: float, intercept: float = 0.0) -> Callable:
    """Return f(s) = slope * s + intercept."""
    return lambda s: slope * s + intercept


def polynomial(coeffs) -> Callable:
    """Return f(s) = c0 + c1*s + c2*s^2 + ...

    Parameters
    ----------
    coeffs : array-like
        Coefficients ordered from lowest to highest degree:
        coeffs[0] is the constant term, coeffs[1] the linear term, etc.
    """
    c = np.asarray(coeffs, dtype=float)[::-1]  # np.polyval expects high→low
    return lambda s: np.polyval(c, s)
