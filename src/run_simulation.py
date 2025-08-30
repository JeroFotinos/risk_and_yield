from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from main import Weather
from main import Soil

Array = np.ndarray

DATA_PATH = Path(Path(__file__).parent, "data")


# -----------------------------
# Utility helpers
# -----------------------------
def load_matlab_file_as_dict(filename: str, verbose=False) -> Dict[str, Any]:
    loaded_dict: Dict[str, Any] = loadmat(DATA_PATH / "Soil" / filename)
    if verbose:
        pprint(loaded_dict.keys())
    return loaded_dict


def load_weather(path):
    df = pd.read_csv(path, sep=";")
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y%m%d")
    df.rename(
        columns={
            "TMED(C)": "temp",
            "RAD(MJ/M2)": "par",
            "LLUVIA(mm)": "precip",
            "EVAP_TRANS(mm)": "et0",
        },
        inplace=True,
    )
    return df


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
        / 100.0
    )  # Si water0 es entre (0,1) hay que dividir. Pregungar a jero. Porcentaje de agua util.
    mask_soy = np.zeros_like(mask_maize)
    return (mask_maize, mask_soy, lat, lon, dds0, water0)


def load_weather_from_data() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    df_weather: pd.DataFrame = load_weather(DATA_PATH / "Weather" / "weather.csv")
    temp, par, precip, et0 = (
        df_weather["temp"],
        df_weather["par"],
        df_weather["precip"],
        df_weather["et0"],
    )
    return temp, par, precip, et0


# -----------------------------
# Set parameters from data files
# -----------------------------
mask_maize, mask_soy, lat, lon, dds0, water0 = load_soil_from_data()
temp, par, precip, et0 = load_weather_from_data()

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
print(results)
