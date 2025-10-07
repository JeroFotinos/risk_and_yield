from pathlib import Path
import h5py
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


def load_all_datasets(path: str | Path) -> Dict[str, np.ndarray]:
    """Carga todos los datasets de un archivo HDF5 plano (sin grupos)."""
    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset):
                arrays[name] = ds[:]  # pasar a numpy
    return arrays


def rename_h5_keys(file_path: Path, mapping: dict[str, str]) -> None:
    """Renames datasets or groups in an HDF5 file based on a mapping.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file to modify.
    mapping : dict[str, str]
        Dictionary mapping old keys to new keys.
    """
    with h5py.File(file_path, "r+") as f:
        for old_key, new_key in mapping.items():
            if old_key in f:
                print(f"Renaming '{old_key}' → '{new_key}'")
                f.copy(f[old_key], new_key)
                del f[old_key]
            else:
                print(f"Key '{old_key}' not found — skipped.")


def plot_pixel_key(
    x: int,
    y: int,
    data_dict: Dict[str, np.ndarray],
    key: str = "biomass_cum",
):
    t = np.arange(181)
    plt.figure(figsize=(10, 6))
    data_pixel = data_dict[key][:, x, y]
    plt.plot(
        t,
        data_pixel,
    )
    plt.xlabel("Time")
    plt.ylabel(f"Property value-{key}")
    plt.title(f"{key}")
    plt.show()
