from main import Soil
from pprint import pprint
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pprint import pprint
from scipy.io import loadmat
from typing import Dict
from typing import Any

DATA_PATH = Path(Path(__file__).parent, "data")

# -----------------------------
# Utility helpers
# -----------------------------
def load_matlab_file_as_dict(filename: str) -> Dict[str, Any]:
    loaded_dict: Dict[str, Any] = loadmat(os.path.join(DATA_PATH, filename))
    pprint(loaded_dict.keys())
    return loaded_dict


# -----------------------------
# Load data
# -----------------------------

mask_maize = load_matlab_file_as_dict("mat_maiz_2021_lowres.mat")[
    "clase_maiz_2021_lowres"
]  
lat = load_matlab_file_as_dict("mat_maiz_2021_lat_lowres.mat")["lat_lowres"]
lon = load_matlab_file_as_dict("mat_maiz_2021_lon_lowres.mat")["lon_lowres"]
dds0 = load_matlab_file_as_dict("mat_dds_maiz_est_lowres.mat")["dds_est"]
water0 = (
    load_matlab_file_as_dict("mat_aguadisp_saocom_maiz_2021-2022_2.mat")["a_disp_campo"]
    / 100
)

soil = Soil(
    mask_maize=mask_maize,
    mask_soy=np.logical_not(mask_maize),
    lat=lat,
    lon=lon,
    water0=water0,
    dds0=dds0,
)




