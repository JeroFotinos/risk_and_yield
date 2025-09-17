from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

from risknyield.core.main import Weather
from risknyield.core.main import Soil

from risknyield.library.io_hdf5 import save_results_hdf5

Array = np.ndarray

DATA_PATH = Path(Path(__file__).parent.parent, "data")
WEATHER_CSV_PATH = Path(DATA_PATH, "Weather", "weather.csv")
SOIL_MAT_PATH = Path(DATA_PATH, "Soil")


# ================================================
# Helper functions for HDF5 persistence for inputs
# ================================================

def _suggest_chunks(shape: tuple[int, ...]) -> Optional[tuple[int, ...]]:
    """Chunk 2D arrays for spatial access; 1D leave contiguous."""
    if len(shape) == 2:
        H, W = shape
        return (min(H, 128), min(W, 128))
    return None

def _write_ds(g: h5py.Group, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    dset = g.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=_suggest_chunks(arr.shape),
    )
    dset.attrs["shape"] = arr.shape
    dset.attrs["dtype"] = str(arr.dtype)

def save_inputs_hdf5(
    path: Path,
    *,
    mask_maize: np.ndarray,
    mask_soy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dds0: np.ndarray,
    water0: np.ndarray,
    temp: np.ndarray,
    par: np.ndarray,
    precip: np.ndarray,
    et0: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist the *inputs* required to re-run the scenario, decoupled from examples/.
    Produces tests/fixtures/maize_inputs.h5 with groups:
      /soil/{mask_maize,mask_soy,lat,lon,dds0,water0}
      /weather/{temp,par,precip,et0}
    File attributes record start/end dates as ISO strings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "risknyield.inputs/1"
        f.attrs["start_date"] = start_date.strftime("%Y-%m-%d")
        f.attrs["end_date"] = end_date.strftime("%Y-%m-%d")
        if extra_meta:
            for k, v in extra_meta.items():
                f.attrs[k] = str(v)

        g_soil = f.create_group("soil")
        _write_ds(g_soil, "mask_maize", mask_maize.astype(bool))
        _write_ds(g_soil, "mask_soy",   mask_soy.astype(bool))
        _write_ds(g_soil, "lat",        lat)
        _write_ds(g_soil, "lon",        lon)
        _write_ds(g_soil, "dds0",       dds0)
        _write_ds(g_soil, "water0",     water0)

        g_weather = f.create_group("weather")
        _write_ds(g_weather, "temp",   temp)
        _write_ds(g_weather, "par",    par)
        _write_ds(g_weather, "precip", precip)
        _write_ds(g_weather, "et0",    et0)

    print(f"[ok] Wrote inputs snapshot: {path.resolve()}")


# -----------------------------
# Utility helpers
# -----------------------------
def load_matlab_file_as_dict(filename: str, verbose=False) -> Dict[str, Any]:
    loaded_dict: Dict[str, Any] = loadmat(Path(SOIL_MAT_PATH, filename))
    if verbose:
        pprint(loaded_dict.keys())
    return loaded_dict


def load_weather(path: Path, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y-%m-%d")
    df.rename(
        columns={
            "FECHA": "scene_date",
            "TMPMED": "temp",
            "RAD": "par",
            "PREC": "precip",
            "ET0": "et0",
        },
        inplace=True,
    )
    cols = ["scene_date", "temp", "par", "precip", "et0"]
    df = df.loc[
        (df["scene_date"] >= start_date) & (df["scene_date"] <= end_date),
        cols,
    ]
    return df


# -----------------------------
# Loaders
# -----------------------------
def load_soil_from_data() -> Tuple[Array, Array, Array, Array, Array, Array]:
    mask_maize: Array = load_matlab_file_as_dict("mat_maiz_2021_lowres.mat")[
        "clase_maiz_2021_lowres"
    ]
    lat: Array = load_matlab_file_as_dict("mat_maiz_2021_lat_lowres.mat")["lat_lowres"]
    lon: Array = load_matlab_file_as_dict("mat_maiz_2021_lon_lowres.mat")["lon_lowres"]
    dds0: Array = load_matlab_file_as_dict("mat_dds_maiz_est_lowres.mat")["dds_est"]
    water0: Array = (
        load_matlab_file_as_dict("mat_aguadisp_saocom_maiz_2021-2022_2.mat")[
            "a_disp_campo"
        ]
        # / 100.0
    )  # Si water0 es entre (0,1) hay que dividir. Pregungar a jero. Porcentaje de agua util.
    # Rta: no, es el agua total por unidad de área del pixel en mm.
    mask_soy = np.zeros_like(mask_maize)
    return (mask_maize, mask_soy, lat, lon, dds0, water0)


def load_weather_from_data(
    data_path: Path, start_time: datetime, end_time: datetime
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
]:
    df_weather: pd.DataFrame = load_weather(data_path, start_time, end_time)
    temp, par, precip, et0 = (
        df_weather["temp"].to_numpy(),
        df_weather["par"].to_numpy(),
        df_weather["precip"].to_numpy(),
        df_weather["et0"].to_numpy(),
    )
    return temp, par, precip, et0


# -----------------------------
# Set parameters from data files
# -----------------------------
start_date = datetime(2021, 12, 4)
end_date = datetime(2022, 6, 2)

mask_maize, mask_soy, lat, lon, dds0, water0 = load_soil_from_data()
temp, par, precip, et0 = load_weather_from_data(WEATHER_CSV_PATH, start_date, end_date)

# -----------------------------
# Initialize Soil model
# -----------------------------
weather = Weather(
    temp=temp,
    par=par,
    precip=precip,
    et0=et0,
)

target_crop = "maize"
soil = Soil(
    mask_maize=mask_maize,
    mask_soy=mask_soy,
    lat=lat,
    lon=lon,
    water0=water0,
    dds0=dds0,
)

# -----------------------------
# Run simulation
# -----------------------------

results = soil.evolve(target_crop, weather)

# # -----------------------------
# # Save results
# # -----------------------------
# BASELINE = Path("tests/fixtures/maize_baseline.h5")
# save_results_hdf5(results, BASELINE, extra_meta={
#     "scenario": "maize_2021_12_04__2022_06_02",
# })
# print(f"Saved results to {BASELINE}")

# # -----------------------------
# # Save inputs (for tests)
# # -----------------------------
# INPUTS = Path("tests/fixtures/maize_inputs.h5")
# save_inputs_hdf5(
#     INPUTS,
#     mask_maize=mask_maize,
#     mask_soy=mask_soy,
#     lat=lat,
#     lon=lon,
#     dds0=dds0,
#     water0=water0,
#     temp=temp,
#     par=par,
#     precip=precip,
#     et0=et0,
#     start_date=start_date,
#     end_date=end_date,
#     extra_meta={"scenario": "maize_2021_12_04__2022_06_02"},
# )
# print(f"Saved inputs to {INPUTS}")
