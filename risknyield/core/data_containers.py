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
    """
    temp: Array
    par: Array
    precip: Array
    et0: Array

    def __post_init__(self):
        # coerce to 1-D float arrays
        temp   = np.asarray(self.temp,   dtype=float)
        par    = np.asarray(self.par,    dtype=float)
        precip = np.asarray(self.precip, dtype=float)
        et0    = np.asarray(self.et0,    dtype=float)

        # shape checks
        if temp.ndim != 1 or par.ndim != 1 or precip.ndim != 1 or et0.ndim != 1:
            raise ValueError("All weather series must be 1-D.")
        T = temp.shape[0]
        if not (par.shape[0] == precip.shape[0] == et0.shape[0] == T):
            raise ValueError("All weather series must have the same length T.")

        # assign using object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "temp",   temp)
        object.__setattr__(self, "par",    par)
        object.__setattr__(self, "precip", precip)
        object.__setattr__(self, "et0",    et0)



@dataclass
class Results:
    """
    Simulation outputs (time x space) for key variables.

    Notes
    -----
    Shapes are (H, W, T) unless otherwise stated.
    """
    dates: Array                 # (T,) or integer day indices
    temp: Array                  # (T,)
    par: Array                   # (T,)
    precip: Array                # (T,)
    et0: Array                   # (T,)

    root_depth: Array            # (H, W, T)
    transpiration: Array         # (H, W, T)
    eff_precip: Array            # (H, W, T)
    soil_evap: Array             # (H, W, T)

    au_layers: Array             # (H, W, L, T)
    p_au: Array                  # (H, W, T)

    ceh: Array                   # (H, W, T)
    ceh_r: Array                 # (H, W, T)
    ceh_pc: Array                # (H, W, T)

    cover: Array                 # (H, W, T)

    t_eur: Array                 # (H, W, T)
    eur_act: Array               # (H, W, T)

    biomass_daily: Array         # (H, W, T)
    biomass_cum: Array           # (H, W, T)
    yield_tensor: Array          # (H, W, T)


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
        Number of soil layers.
    cc : float, default=0.27
        Field capacity fraction [0-1].
    pmp : float, default=0.12
        Wilting point fraction [0-1].
    soil_layer_depth_mm : float or ndarray, shape (H, W), default=500.0
        Effective per-layer depth [mm]. Per-layer capacity is
        ``aut = soil_layer_depth_mm * (cc - pmp)``.
    """
    lat: Array
    lon: Array
    water0: Array
    dds0: Array
    crop_mask: Array
    n_layers: int = 4  # They were working with 4 soil layers in the last version of the code
    cc: float = 0.27  # capacidad campo para agua disponible 0.32 - 0.35; de parametros_maiz.m
    pmp: float = 0.12  # es un % funcion espacial del tipo de suelo (laboratorio/mapas); de parametros_maiz.m
    soil_layer_depth_mm: float | Array = 500.0  # should be the per-layer depth used for access rules

    H: int = field(init=False)
    W: int = field(init=False)
    aut: Array = field(init=False)

    def __post_init__(self):
        if np.asarray(self.water0).ndim != 2:
            raise ValueError("water0 must be 2D (H, W).")
        self.H, self.W = np.asarray(self.water0).shape
        shape = (self.H, self.W)

        self.dds0 = _to_array(self.dds0, shape)
        self.lat = _to_array(self.lat, shape)
        self.lon = _to_array(self.lon, shape)

        self.crop_mask = _to_array(self.crop_mask, shape).astype(bool, copy=False)

        self.soil_layer_depth_mm = _to_array(self.soil_layer_depth_mm, shape)
        self.aut = self.soil_layer_depth_mm * (float(self.cc) - float(self.pmp))
