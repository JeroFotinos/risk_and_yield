from __future__ import annotations
import numpy as np
Array = np.ndarray

def effective_precipitation(pp: Array) -> Array:
    """
    Piecewise-linear effective precipitation (mm/day) following the MATLAB rule.

    Given daily precipitation `pp` (mm), returns ppef. The breakpoints are
    inspired by the original aux/pp_cotas mapping. For the original MATLAB,
    refer to `agromodel_model_plantgrowth_v27.m`, lines 202-208.

    Parameters
    ----------
    pp : ndarray or scalar
        Daily precipitation [mm].

    Returns
    -------
    ndarray
        Effective precipitation [mm/day], per the breakpoint mapping.
    """
    pp = np.asarray(pp, dtype=float)
    xp = np.array([0, 25, 50, 75, 100, 125, 150], dtype=float)
    yp = np.array([0, 23.75, 46.25, 66.75, 83.0, 94.25, 100.25], dtype=float)
    return np.interp(pp, xp, yp, left=0.95 * pp, right=100.25 + 0.05 * (pp - 150))
