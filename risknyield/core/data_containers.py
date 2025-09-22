"""
Core containers for weather, soil, and simulation outputs.

This module defines immutable data structures used by the crop model and a
small helper to broadcast inputs to a common grid shape.

Classes
-------
Weather
    Frozen dataclass storing forcing time series (temperature, PAR,
    precipitation, ET0) with validation and normalization.
Results
    Container for simulation output tensors (time × space).
Soil
    Static spatial context and soil capacities (no evolution logic).

Functions
---------
_to_array
    Coerce scalars/arrays to a 2D array of target shape by broadcasting.

Notes
-----
- ``Weather`` coerces input series to 1-D float arrays and checks length
  consistency.
- ``Soil`` harmonizes per-pixel arrays to shape ``(H, W)`` and derives the
  per-layer available-water capacity as
  ``aut = soil_layer_depth_mm * (cc - pmp)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

Array = np.ndarray


def _to_array(x: Array | float | int, shape: Tuple[int, int]) -> Array:
    """
    Coerce scalars/arrays to a 2D array of given shape by broadcasting.

    Parameters
    ----------
    x : array-like or scalar
        Input to coerce.
    shape : tuple of int
        Target (H, W) shape.

    Returns
    -------
    ndarray
        Broadcasted array with dtype=float64.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=np.float64)
    elif arr.shape != shape:
        arr = np.broadcast_to(arr, shape).astype(np.float64, copy=False)
    return arr


# -------------------------
# Data containers
# -------------------------


@dataclass(frozen=True)
class Weather:
    """
    Forcing time series used by the crop model.

    Dataclass for storing temperature, radiation, precipitation, and
    evapotranspiration time series. The class is frozen to ensure that the
    input weather data cannot be accidentally modified during simulation.

    Attributes
    ----------
    temp : ndarray, shape (T,)
        Mean air temperature [°C].
    par : ndarray, shape (T,)
        Photosynthetically active radiation [MJ/m²/day].
    precip : ndarray, shape (T,)
        Daily precipitation [mm/day].
    et0 : ndarray, shape (T,)
        Reference evapotranspiration [mm/day].

    Raises
    ------
    ValueError
        If any of the input arrays are not 1-D or if their lengths do not
        match
    """

    temp: Array
    par: Array
    precip: Array
    et0: Array

    def __post_init__(self):
        """
        Normalize inputs to 1-D float arrays and validate lengths.

        This method converts ``temp``, ``par``, ``precip``, and ``et0`` to
        ``float`` NumPy arrays with one dimension and checks that all series
        share the same length ``T``.

        Parameters
        ----------
        self : Weather
            The instance being initialized.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any input series is not 1-D or if the lengths are inconsistent.
        """
        # coerce to 1-D float arrays
        temp = np.asarray(self.temp, dtype=float)
        par = np.asarray(self.par, dtype=float)
        precip = np.asarray(self.precip, dtype=float)
        et0 = np.asarray(self.et0, dtype=float)

        # shape checks
        if (
            temp.ndim != 1
            or par.ndim != 1
            or precip.ndim != 1
            or et0.ndim != 1
        ):
            raise ValueError("All weather series must be 1-D.")
        T = temp.shape[0]
        if not (par.shape[0] == precip.shape[0] == et0.shape[0] == T):
            raise ValueError("All weather series must have the same length T.")

        # We assign using object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "temp", temp)
        object.__setattr__(self, "par", par)
        object.__setattr__(self, "precip", precip)
        object.__setattr__(self, "et0", et0)


@dataclass
class Results:
    """Simulation outputs (time × space) for key variables."""

    dates: Array  # (T,) or integer day indices
    temp: Array  # (T,)
    par: Array  # (T,)
    precip: Array  # (T,)
    et0: Array  # (T,)

    root_depth: Array  # (H, W, T)
    transpiration: Array  # (H, W, T)
    eff_precip: Array  # (H, W, T)
    soil_evap: Array  # (H, W, T)

    au_layers: Array  # (H, W, L, T)
    p_au: Array  # (H, W, T)

    ceh: Array  # (H, W, T)
    ceh_r: Array  # (H, W, T)
    ceh_pc: Array  # (H, W, T)

    cover: Array  # (H, W, T)

    t_eur: Array  # (H, W, T)
    eur_act: Array  # (H, W, T)

    biomass_daily: Array  # (H, W, T)
    biomass_cum: Array  # (H, W, T)
    yield_tensor: Array  # (H, W, T)


@dataclass
class Soil:
    """
    Spatial soil/crop context and static capacities (no evolution logic).

    Parameters
    ----------
    lat, lon : array-like, shape (H, W) or 1D broadcastable
        Coordinates.
    water0 : ndarray, shape (H, W)
        Initial available water in the top layer [mm].
    dds0 : ndarray, shape (H, W)
        Days after sowing at t=0 (can be <0 before emergence).
    crop_mask : ndarray, shape (H, W), bool
        Crop masks (non-overlapping).
    n_layers : int, default=4
        Number of soil layers. They were working with 4 soil layers in the
        last version of the MATLAB code (thus the default).
    cc : float, default=0.27
        Field capacity fraction [0-1]. From spanish “capacidad campo” (para
        agua disponible).
    pmp : float, default=0.12
        Wilting point fraction [0-1].
    soil_layer_depth_mm : float or ndarray, shape (H, W), default=500.0
        Effective per-layer depth [mm]. Per-layer capacity is
        ``aut = soil_layer_depth_mm * (cc - pmp)``.

    Notes
    -----
    - Values for `cc` and `pmp` are spatially constant here, but could be
      extended to arrays if needed by passing arrays of shape (H, W), or by
      providing aut directly.
    - cc and pmp values are taken from `parametros_maiz.m`, having checked
      that they match the ones in `parametros_soja.m`.
    """

    lat: Array
    lon: Array
    water0: Array
    dds0: Array
    crop_mask: Array
    n_layers: int = 4
    cc: float = 0.27
    pmp: float = 0.12
    soil_layer_depth_mm: float | Array = 500.0

    H: int = field(init=False)
    W: int = field(init=False)
    aut: Array = field(init=False)

    def __post_init__(self):
        """
        Harmonize shapes to ``(H, W)`` and derive per-layer capacity.

        This method infers ``H`` and ``W`` from ``water0``, broadcasts
        ``dds0``, ``lat``, ``lon``, ``crop_mask``, and
        ``soil_layer_depth_mm`` to ``(H, W)``, and computes the per-layer
        available-water capacity ``aut = soil_layer_depth_mm * (cc - pmp)``.

        Parameters
        ----------
        self : Soil
            The instance being initialized.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``water0`` is not 2-D or if shapes cannot be reconciled to
            ``(H, W)``.
        """
        if np.asarray(self.water0).ndim != 2:
            raise ValueError("water0 must be 2D (H, W).")
        self.H, self.W = np.asarray(self.water0).shape
        shape = (self.H, self.W)

        self.dds0 = _to_array(self.dds0, shape)
        self.lat = _to_array(self.lat, shape)
        self.lon = _to_array(self.lon, shape)

        self.crop_mask = _to_array(self.crop_mask, shape).astype(
            bool, copy=False
        )

        self.soil_layer_depth_mm = _to_array(self.soil_layer_depth_mm, shape)
        self.aut = self.soil_layer_depth_mm * (
            float(self.cc) - float(self.pmp)
        )
