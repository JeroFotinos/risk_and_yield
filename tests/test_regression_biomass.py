from pathlib import Path
import numpy as np
import numpy.testing as npt
import h5py
from risknyield.core.main import Soil, Weather
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
        mask_maize=mask_maize, mask_soy=mask_soy,
        lat=lat, lon=lon, water0=water0, dds0=dds0,
    )
    weather = Weather(temp=temp, par=par, precip=precip, et0=et0)
    return soil, weather

def test_biomass_cum_matches_baseline():
    assert INPUTS.exists(), f"Missing {INPUTS}"
    assert BASELINE.exists(), f"Missing {BASELINE}"

    soil, weather = _load_inputs(INPUTS)
    cur = soil.evolve("maize", weather)

    base = load_results_vars_hdf5(BASELINE, names=["biomass_cum"])
    npt.assert_allclose(cur.biomass_cum, base["biomass_cum"], rtol=RTOL, atol=ATOL)

def test_yield_matches_baseline():
    soil, weather = _load_inputs(INPUTS)
    cur = soil.evolve("maize", weather)

    base = load_results_vars_hdf5(BASELINE, names=["yield_"])
    npt.assert_allclose(cur.yield_, base["yield_"], rtol=RTOL, atol=ATOL)
