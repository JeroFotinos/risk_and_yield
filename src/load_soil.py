import os
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat

from main import Soil

Array = np.ndarray

DATA_PATH = Path(Path(__file__).parent, "data", "Soil")


# -----------------------------
# Utility helpers
# -----------------------------
def load_matlab_file_as_dict(filename: str, verbose=False) -> Dict[str, Any]:
    loaded_dict: Dict[str, Any] = loadmat(os.path.join(DATA_PATH, filename))
    if verbose:
        pprint(loaded_dict.keys())
    return loaded_dict


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
    / 100
)  # Si water0 es entre (0,1) hay que dividir. Pregungar a jero

# -----------------------------
# Initialize Soil model
# -----------------------------

soil = Soil(
    mask_maize=mask_maize,
    mask_soy=np.zeros_like(mask_maize),
    lat=lat,
    lon=lon,
    water0=water0,
    dds0=dds0,
)
