from pathlib import Path
import numpy as np
import numpy.testing as npt
import h5py
from risknyield.core.data_containers import Soil, Weather
from risknyield.core.main import CropModel
from risknyield.core.crops import MaizeParams
from risknyield.library.io_hdf5 import load_results_vars_hdf5  # already in your repo

ATOL = 1e-6
RTOL = 1e-5
FIXDIR = Path(__file__).parent / "fixtures"
INPUTS = FIXDIR / "maize_inputs.h5"
BASELINE = FIXDIR / "maize_baseline.h5"

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

def test_biomass_cum_matches_baseline():
    assert INPUTS.exists(), f"Missing {INPUTS}"
    assert BASELINE.exists(), f"Missing {BASELINE}"

    soil, weather = _load_inputs(INPUTS)
    cur = CropModel(soil=soil, weather=weather, params=MaizeParams()).evolve()

    base = load_results_vars_hdf5(BASELINE, names=["biomass_cum"])
    npt.assert_allclose(cur.biomass_cum, base["biomass_cum"], rtol=RTOL, atol=ATOL)

def test_yield_matches_baseline():
    soil, weather = _load_inputs(INPUTS)
    cur = CropModel(soil=soil, weather=weather, params=MaizeParams()).evolve()

    base = load_results_vars_hdf5(BASELINE, names=["yield_tensor"])
    npt.assert_allclose(cur.yield_tensor, base["yield_tensor"], rtol=RTOL, atol=ATOL)
