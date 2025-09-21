from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from risknyield.core.data_containers import CropParams

@dataclass(frozen=True)
class MaizeParams(CropParams):
    """
    Parameterization for maize.

    Crop- and model-parameters controlling growth/stress dynamics.
    Many names mirror the original MATLAB variables for traceability.
    Defaults here are **maize** (from `parametros_maiz.m`).

    Notes
    -----
    Extend with all maize-specific parameters used in your growth,
    stress, partitioning, and senescence equations as needed.
    """
    crop_name: ClassVar[str] = "maize"

    # Soil water stress thresholds (fractions of available water)
    au_up: float = 0.72
    au_down: float = 0.20
    au_up_r: float = 0.69
    au_down_r: float = 0.0
    au_up_pc: float = 0.60
    au_down_pc: float = 0.15

    # Shape parameters for the stress response (dimensionless)
    c_forma: float = 4.9
    c_forma_r: float = 6.0
    c_forma_pc: float = 1.3

    # Canopy cover development (days after sowing, DAS)
    dds_in: int = 7
    dds_max: int = 47
    dds_sen: int = 87
    dds_fin: int = (
        120  # This was “no_days_cultivo” in MATLAB --- ver linea 29 de agromodel_model_plantgrowth_v27.m
    )
    c_in: float = 0.039  # initial cover at emergence
    c_fin: float = 0.01  # residual cover in senescence
    c_max: float = 0.89  # maximum attainable cover (maize population default)
    alpha1: float = (c_max - c_in) / (
        dds_max - dds_in
    )  # daily growth rate to max cover

    # Root dynamics
    root_growth_rate: float = 30.0  # mm/day (maize)
    root_max_mm: float = 2000.0  # This comes from
    # agromodel_model_plantgrowth_v27.m > line 197, where they take the roots
    # to be the minimum between 2000 mm and the new updated value. I think
    # they set 2000 mm as the max because they considered 4 layers of 500 mm.
    # If that's the case, we should generalize this to
    # root_max_mm = n_layers * layer_threshold_mm
    # layer_threshold_mm: float = 500.0  # 500.0  # per-layer depth used for access rules
    # This comes from lines 220 to 223 in agromodel_model_plantgrowth_v27.m

    # Radiation use efficiency (biomass per MJ PAR)
    # Units: g DM per MJ PAR (g/MJ)
    eur_pot: float = 3.65

    # Thermal stress (simple trapezoid response)
    tbr: float = 8.0
    tor1: float = 29.0
    tor2: float = 39.0
    tcr: float = 45.0

    # Harvest index / ICI (logistic ramp around flowering)
    # (taken from parametros_maiz.m, lines 39 to 42)
    df: int = 58
    ic_in: float = 0.001
    ic_pot_t: float = 0.48
    Y: float = 0.19

    # Transpiration coefficient
    KC: float = 0.94

    # # Harvest index to translate biomass to yield
    # harvest_index: float = 0.5
    # Harvest index multiplier (keep 1.0 to avoid double-counting with ICI) -- CHECK (?)
    harvest_index: float = 1.0
