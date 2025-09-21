# tests/test_regression_full.py
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import numpy.testing as npt
import pytest

from risknyield.core.data_containers import Soil, Weather
from risknyield.core.main import CropModel
from risknyield.core.crops import CropParams
from risknyield.library.io_hdf5 import load_results_vars_hdf5

# Tolerances reasonably robust to BLAS / platform differences
ATOL = 1e-6
RTOL = 1e-5

FIXDIR   = Path(__file__).parent / "fixtures"
INPUTS   = FIXDIR / "maize_inputs.h5"
BASELINE = FIXDIR / "maize_baseline.h5"

# These match what you persisted via save_results_hdf5(...)
FIELDS_1D = ["dates", "temp", "par", "precip", "et0"]
FIELDS_3D = [
    "root_depth", "transpiration", "eff_precip", "soil_evap",
    "p_au", "ceh", "ceh_r", "ceh_pc", "cover",
    "t_eur", "eur_act",
    "biomass_daily", "biomass_cum", "yield_tensor",
]
FIELDS_4D = ["au_layers"]

def _load_inputs(path: Path):
    with h5py.File(path, "r") as f:
        s = f["soil"]; w = f["weather"]
        mask_maize = s["mask_maize"][...].astype(bool)
        mask_soy   = s["mask_soy"][...].astype(bool)
        lat        = s["lat"][...]
        lon        = s["lon"][...]
        dds0       = s["dds0"][...]
        water0     = s["water0"][...]

        temp   = w["temp"][...]
        par    = w["par"][...]
        precip = w["precip"][...]
        et0    = w["et0"][...]

    soil = Soil(
        lat=lat, lon=lon, water0=water0, dds0=dds0,
        crop_mask=mask_maize,
    )
    weather = Weather(temp=temp, par=par, precip=precip, et0=et0)
    return soil, weather

@pytest.mark.slow
def test_full_results_match_baseline():
    assert INPUTS.exists(),   f"Missing inputs fixture: {INPUTS}"
    assert BASELINE.exists(), f"Missing baseline fixture: {BASELINE}"

    soil, weather = _load_inputs(INPUTS)
    cur = CropModel(soil=soil, weather=weather, params=CropParams.maize()).evolve()

    # Load all baseline variables in one pass (still selective I/O per variable)
    names = FIELDS_1D + FIELDS_3D + FIELDS_4D
    base = load_results_vars_hdf5(BASELINE, names=names)

    # Quick grid/time sanity check using a representative 3D variable
    H, W, T = cur.biomass_cum.shape
    bH, bW, bT = base["biomass_cum"].shape
    assert (H, W, T) == (bH, bW, bT), f"Grid/time mismatch: {(H, W, T)} vs {(bH, bW, bT)}"

    # 1D variables (time series)
    for name in FIELDS_1D:
        a = getattr(cur, name)
        b = base[name]
        assert a.shape == b.shape, f"Shape mismatch in {name}: {a.shape} vs {b.shape}"
        npt.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f"Mismatch in {name}")

    # 3D variables (H, W, T)
    for name in FIELDS_3D:
        a = getattr(cur, name)
        b = base[name]
        assert a.shape == b.shape, f"Shape mismatch in {name}: {a.shape} vs {b.shape}"
        npt.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f"Mismatch in {name}")

    # 4D variables (H, W, L, T)
    for name in FIELDS_4D:
        a = getattr(cur, name)
        b = base[name]
        assert a.shape == b.shape, f"Shape mismatch in {name}: {a.shape} vs {b.shape}"
        npt.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f"Mismatch in {name}")

    # Extra aggregate guardrail: total yield series over time
    cur_total_y  = np.nansum(np.nansum(cur.yield_tensor,  axis=0), axis=0)
    base_total_y = np.nansum(np.nansum(base["yield_tensor"], axis=0), axis=0)
    npt.assert_allclose(cur_total_y, base_total_y, rtol=RTOL, atol=ATOL, equal_nan=True,
                        err_msg="Mismatch in total yield series")
