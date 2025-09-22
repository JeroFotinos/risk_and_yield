"""Module to save/load Results objects to/from HDF5 files."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py

import numpy as np

# List fields you want to persist (explicit is safer than introspecting
# properties)
RESULTS_ARRAY_FIELDS = [
    "dates",
    "temp",
    "par",
    "precip",
    "et0",
    "root_depth",
    "transpiration",
    "eff_precip",
    "soil_evap",
    "au_layers",
    "p_au",
    "ceh",
    "ceh_r",
    "ceh_pc",
    "cover",
    "t_eur",
    "eur_act",
    "biomass_daily",
    "biomass_cum",
    "yield_",
]


def _git_commit_or_none() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _suggest_chunks(shape: tuple[int, ...]) -> Optional[tuple[int, ...]]:
    """
    Choose chunk sizes to enable efficient time slicing.

    For 3D (H, W, T): ``(min(H, 64), min(W, 64), min(T, 1))``.
    For 4D (H, W, L, T): ``(min(H, 64), min(W, 64), min(L, 4), min(T, 1))``.
    For 1D/2D, return ``None`` (let HDF5 pick or store contiguous).

    Parameters
    ----------
    shape : tuple of int
        Dataset shape.

    Returns
    -------
    tuple of int or None
        Suggested chunk shape, or ``None`` if unchunked/automatic.
    """
    ndim = len(shape)
    if ndim == 3:
        H, W, T = shape
        return (min(H, 64), min(W, 64), min(T, 1))
    if ndim == 4:
        H, W, L, T = shape
        return (min(H, 64), min(W, 64), min(L, 4), min(T, 1))
    return None


def _write_dataset(g: h5py.Group, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    # Handle datetime64
    # (if we ever store true datetimes) — store epoch seconds int64
    if np.issubdtype(arr.dtype, np.datetime64):
        arr = arr.astype("datetime64[s]").astype(np.int64)
        dset = g.create_dataset(
            name,
            data=arr,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=_suggest_chunks(arr.shape),
        )
        dset.attrs["logical_dtype"] = "datetime64[s]"
        return

    # Everything else stays as-is (float, int, bool)
    dset = g.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=_suggest_chunks(arr.shape),
    )
    # Helpful shape metadata (humans/tools)
    dset.attrs["shape"] = arr.shape
    dset.attrs["dtype"] = str(arr.dtype)


def save_results_hdf5(
    results: Any, path: Path, extra_meta: Optional[Dict[str, Any]] = None
) -> None:
    """Persist arrays from a Results-like object to HDF5 with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata
    meta = {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": _git_commit_or_none(),
        "notes": "Regression baseline for risknyield",
    }
    if extra_meta:
        meta.update(extra_meta)

    with h5py.File(path, "w") as f:
        # File-level metadata
        for k, v in meta.items():
            f.attrs[k] = (
                json.dumps(v)
                if isinstance(v, (dict, list))
                else ("" if v is None else v)
            )

        # Store arrays in a group
        g = f.create_group("results")

        for name in RESULTS_ARRAY_FIELDS:
            if not hasattr(results, name):
                continue
            arr = getattr(results, name)
            _write_dataset(g, name, arr)

    print(f"[ok] Wrote HDF5 snapshot: {path.resolve()}")


def load_results_vars_hdf5(
    path: Path, names: Iterable[str]
) -> Dict[str, np.ndarray]:
    """
    Load only selected variables from the HDF5 snapshot.

    Returns
    -------
    dict
        Mapping of variable name to array.
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        g = f["results"]
        for name in names:
            if name not in g:
                raise KeyError(f"Variable '{name}' not found in HDF5 file.")
            out[name] = g[name][...]  # load only this variable
            # Reconstruct datetime if we detect it
            if g[name].attrs.get("logical_dtype", "").startswith("datetime64"):
                out[name] = out[name].astype("datetime64[s]")
    return out


def load_results_hdf5(path: Path, ResultsClass) -> Any:
    """
    Load a full Results object from HDF5.

    Constructs ``ResultsClass`` by passing arrays as keyword arguments; field
    names are those in ``RESULTS_ARRAY_FIELDS``.

    Parameters
    ----------
    path : pathlib.Path
        HDF5 file path.
    ResultsClass : type
        Class to instantiate (must accept the fields listed in
        ``RESULTS_ARRAY_FIELDS`` as keyword arguments).

    Returns
    -------
    Any
        An instance of ``ResultsClass`` populated from the file.
    """
    with h5py.File(path, "r") as f:
        g = f["results"]
        kwargs = {}
        for name in RESULTS_ARRAY_FIELDS:
            if name in g:
                arr = g[name][...]
                if (
                    g[name]
                    .attrs.get("logical_dtype", "")
                    .startswith("datetime64")
                ):
                    arr = arr.astype("datetime64[s]")
                kwargs[name] = arr
        return ResultsClass(**kwargs)


# # Note on the choice of HDF5:
# # - There's no fragility across refactors or environment coupling as with
# # pickle.
# # - It's efficient for large arrays, with compression and chunking.
# # - You can slice a single variable without loading the whole file, e.g.:
# with h5py.File("tests/fixtures/maize_baseline.h5") as f:
#     last_frame = f["results/biomass_cum"][:, :, -1]  # reads only that slice
