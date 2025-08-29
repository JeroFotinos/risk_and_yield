import os
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

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

def load_wheather(path):
    df = pd.read_csv(path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y%m%d')
    df.rename(columns={'TMED(C)': 'temp', 'RAD(MJ/M2)': 'par', 'LLUVIA(mm)': 'precip', 'EVAP_TRANS(mm)': 'et0'}, inplace=True)
    return df


# -----------------------------
# Set parameters from data files
# -----------------------------

mask_maize: Array = load_matlab_file_as_dict("mat_maiz_2021_lowres.mat")[
    "clase_maiz_2021_lowres"
]
lat: Array = load_matlab_file_as_dict("mat_maiz_2021_lat_lowres.mat")["lat_lowres"]
lon: Array = load_matlab_file_as_dict("mat_maiz_2021_lon_lowres.mat")["lon_lowres"]
dds0: Array = load_matlab_file_as_dict("mat_dds_maiz_est_lowres.mat")["dds_est"]
water0: Array = (
    load_matlab_file_as_dict("mat_aguadisp_saocom_maiz_2021-2022_2.mat")["a_disp_campo"]
    / 100.0
)  # Si water0 es entre (0,1) hay que dividir. Pregungar a jero. Porcentaje de agua util.

df_weather: pd.DataFrame = load_wheather(DATA_PATH / "Weather" / "weather.csv")

# -----------------------------
# Initialize Soil model
# -----------------------------
weather = Weather(temp=df_weather['temp'], par=df_weather['par'], precip=df_weather['precip'], et0=df_weather['et0'])

target_crop = "maize"
soil = Soil(
    mask_maize=mask_maize,
    mask_soy=np.zeros_like(mask_maize),
    lat=lat,
    lon=lon,
    water0=water0,
    dds0=dds0,
)

# -----------------------------
# Run simulation
# -----------------------------

soil.evolve(target_crop, weather)

