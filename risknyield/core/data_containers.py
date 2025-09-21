from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Tuple, ClassVar

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

def effective_precipitation(pp: Array) -> Array:
    """Piecewise-linear effective precipitation (mm/day) following the MATLAB rule.

    Given daily precipitation `pp` (mm), returns ppef.
    The breakpoints are inspired by the original aux/pp_cotas mapping.
    """
    pp = np.asarray(pp)
    # Breakpoints and slopes derived from the MATLAB snippet
    # (0, 0), (25, 23.75), (50, 46.25), (75, 66.75), (100, 83), (125, 94.25), (150, 100.25)
    xp = np.array([0, 25, 50, 75, 100, 125, 150], dtype=float)
    yp = np.array([0, 23.75, 46.25, 66.75, 83.0, 94.25, 100.25], dtype=float)
    ppef = np.interp(pp, xp, yp, left=0.95 * pp, right=100.25 + 0.05 * (pp - 150))
    return ppef

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


# -------------------------
# Crop parameter interface
# -------------------------

@dataclass(frozen=True)
class CropParams(ABC):
    """
    Abstract parameter set for a crop.

    Notes
    -----
    Python does not support truly abstract dataclass *fields*, but combining
    ``@dataclass`` with ``ABC`` is fine. To keep the base abstract at runtime,
    we define one abstract property (``crop_name``). Concrete subclasses must
    define all numeric fields used by the model.
    """
    crop_name: ClassVar[str]

    # Soil water stress thresholds (fractions of available water)
    au_up: float
    au_down: float
    au_up_r: float
    au_down_r: float
    au_up_pc: float
    au_down_pc: float

    # Shape parameters for the stress response (dimensionless)
    c_forma: float
    c_forma_r: float
    c_forma_pc: float

    # Canopy cover development (days after sowing, DAS)
    dds_in: int
    dds_max: int
    dds_sen: int
    dds_fin: int
    # This last one was “no_days_cultivo” in MATLAB --- ver linea 29 de
    # agromodel_model_plantgrowth_v27.m

    # Initial, final, and maximum canopy cover (fractions)
    c_in: float  # initial cover at emergence
    c_fin: float  # residual cover in senescence
    c_max: float  # maximum attainable cover (maize population default)
    alpha1: float
    #     = (c_max - c_in) / (
    #     dds_max - dds_in
    # )  # daily growth rate to max cover

    # Root dynamics
    root_growth_rate: float  # mm/day (maize)
    root_max_mm: float  # This comes from
    # agromodel_model_plantgrowth_v27.m > line 197, where they take the roots
    # to be the minimum between 2000 mm and the new updated value. I think
    # they set 2000 mm as the max because they considered 4 layers of 500 mm.
    # If that's the case, we should generalize this to
    # root_max_mm = n_layers * layer_threshold_mm
    # layer_threshold_mm: float  # 500.0  # per-layer depth used for access rules
    # This comes from lines 220 to 223 in agromodel_model_plantgrowth_v27.m
    # THE LAYER THRESHOLD MUST BE PROVIDED BY THE Soil CONTEXT
    # SEE Soil.soil_layer_depth_mm

    # Radiation use efficiency (biomass per MJ PAR)
    # Units: g DM per MJ PAR (g/MJ)
    eur_pot: float  # 3.65

    # Thermal stress (simple trapezoid response)
    tbr: float  # 8.0
    tor1: float  # 29.0
    tor2: float  # 39.0
    tcr: float  # 45.0

    # Harvest index / ICI (logistic ramp around flowering)
    # (taken from parametros_maiz.m, lines 39 to 42)
    df: int  # 58
    ic_in: float  # 0.001
    ic_pot_t: float  # 0.48
    Y: float  # 0.19

    # Transpiration coefficient
    KC: float  # 0.94

    # WARNING: Optional crude ET0 proxy (only used if ET0 not provided) -- SHOULD NOT BE (?)
    # k_et0_T: float  # 0.1
    # k_et0_PAR: float  # 0.02
    # WARNING: Hargreaves' Equation should have been previously used to estimate ET0
    # ET0(idx) = 0.0023*(TMPMED(idx)+17.78).*(RAD(idx)/2.45).*((TMPMAX(idx) - TMPMIN(idx)).^0.5);
    # as taken from unificar_climas_vs.m > line 61

    # # Harvest index to translate biomass to yield
    # harvest_index: float = 0.5
    # Harvest index multiplier (keep 1.0 to avoid double-counting with ICI) -- CHECK (?)
    harvest_index: float  # 1.0
