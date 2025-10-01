.. risknyield documentation master file

risknyield
==========

Vectorized crop growth & water-balance modeling in Python.

**risknyield** simulates canopy cover, water stresses, RUE-driven biomass,
and a yield proxy on a spatial grid, using simple daily forcing (temperature,
PAR, precipitation, ET0). The core model is deterministic and designed for
reproducible regression testing and HDF5-based snapshots.

Features
--------

- Deterministic daily time-stepping over a weather horizon
- Layered soil water balance with top–down percolation
- Water & thermal stress factors (CEH, T\ :sup:`EUR`)
- RUE-based biomass and logistic harvest index
- Clean separation of model, data containers, hydrology helpers
- HDF5 persistence for results and input snapshots

Quickstart
----------

.. code-block:: python

   from risknyield.core.crops import CropParams
   from risknyield.core.data_containers import Soil, Weather
   from risknyield.core.model import CropModel
   import numpy as np

   # Minimal toy data (replace with real arrays)
   H, W, T, L = 5, 5, 10, 4
   soil = Soil(
       lat=np.zeros((H, W)),
       lon=np.zeros((H, W)),
       water0=np.full((H, W), 80.0),     # mm
       dds0=np.zeros((H, W)),            # days after sowing at t=0
       crop_mask=np.ones((H, W), bool),
       n_layers=L,
   )
   weather = Weather(
       temp=np.full(T, 25.0),
       par=np.full(T, 15.0),
       precip=np.zeros(T),
       et0=np.full(T, 4.0),
   )
   params = CropParams.maize()

   results = CropModel(soil=soil, weather=weather, params=params).evolve()
   print(results.biomass_cum.shape)  # -> (H, W, T)

User Guide & API
----------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
