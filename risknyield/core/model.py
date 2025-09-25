"""
Vectorized crop growth and water-balance model (time-stepping engine).

This module implements the deterministic, grid-based evolution of the crop–
soil–atmosphere system over a weather time horizon. The public entry point is
:class:`CropModel`, which composes the static containers
:class:`~.data_containers.Soil`, :class:`~.data_containers.Weather`, and
:class:`~.crops.CropParams` and produces a :class:`~.data_containers.Results`
object.

The implementation follows a modular design:
state allocation and initialization, water availability and stress factors,
canopy cover dynamics, flux computation (transpiration, effective
precipitation, soil evaporation), per-layer water balance with top–down
percolation, root growth, and biomass/yield updates. All heavy computation is
vectorized over the spatial grid and layers.

Design Principles
-----------------
- **Deterministic & reproducible**: given the same inputs, `evolve()` returns
  the same `Results` (no random shifts, no ex-post rescaling).
- **Pure numerics**: no I/O, plotting, or file access; inputs/outputs are
  NumPy arrays.
- **Separation of concerns**: data containers in ``data_containers.py``,
  crop parameters in ``crops.py``, hydrologic helpers in ``library/``.
- **Numerical robustness**: extensive clipping, safe divisions, and masking.

See Also
--------
risknyield.core.data_containers : ``Soil``, ``Weather``, ``Results``.
risknyield.core.crops : ``CropParams`` presets (e.g., ``maize()``, ``soy()``).
risknyield.library.hydrology : ``effective_precipitation``.
risknyield.library.io_hdf5 : HDF5 persistence helpers for regression tests.

Examples
--------
>>> from risknyield.core.model import CropModel
>>> from risknyield.core.crops import CropParams
>>> # soil, weather loaded elsewhere
>>> m = CropModel(soil=soil, weather=weather, params=CropParams.maize())
>>> res = m.evolve()  # returns Results with (time × space) arrays
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risknyield.core.crops import CropParams
from risknyield.core.data_containers import Results, Soil, Weather
from risknyield.library.hydrology import effective_precipitation

Array = np.ndarray


@dataclass(slots=True)
class CropModel:
    r"""Deterministic, vectorized crop water–biomass model.

    The model integrates daily over the weather horizon to update per-pixel
    state: root-accessible water (by layer), canopy cover, stress factors,
    biomass, and yield proxy. It uses:
      - :class:`~risknyield.core.data_containers.Soil` for grid geometry,
        masks, initial water, per-layer capacities, and layer thickness.
      - :class:`~risknyield.core.data_containers.Weather` for forcing
        time series (temperature, PAR, precipitation, ET0).
      - :class:`~risknyield.core.crops.CropParams` for phenology, stress
        thresholds/shapes, RUE, and partition/harvest parameters.

    Parameters
    ----------
    soil : Soil
        Spatial context and static capacities (grid shape, masks, ``aut``,
        layer depth).
    weather : Weather
        Daily forcings used throughout the simulation.
    params : CropParams
        Crop-specific parameters (phenology, RUE, stress thresholds, etc.).

    Notes
    -----
    - **Complexity.** Time complexity is :math:`O(H\\,W\\,L\\,T)`; memory
      scales with the number of state/output arrays retained in
      :class:`Results`.
    - **Units.** Water [mm], temperature [°C], PAR [MJ m⁻² day⁻¹], RUE
       [g MJ⁻¹], biomass/yield [g m⁻²], depths [mm], cover/stress in [0, 1].
    - **Masking.** All fluxes and covers are zeroed outside the crop mask.
    - **Extensibility.** The private methods factor distinct physical/logical
      substeps (cover growth, stress computation, layer balance, root growth),
      making it straightforward to swap formulations or add crops.

    See Also
    --------
    risknyield.core.data_containers.Results
        Output container returned by :meth:`evolve`.
    risknyield.core.crops.CropParams
        Parameter presets (``maize()``, ``soy()``) and validation logic.
    risknyield.library.hydrology.effective_precipitation
        Piecewise mapping from precipitation to effective precipitation.

    Examples
    --------
    >>> model = CropModel(
    ...     soil=soil,
    ...     weather=weather,
    ...     params=CropParams.maize(),
    ... )
    >>> results = model.evolve()
    >>> results.biomass_cum.shape
    (H, W, T)
    """

    soil: Soil
    weather: Weather
    params: CropParams

    # ---------------------------
    # Public API
    # ---------------------------
    def evolve(self) -> Results:
        """Simulates over the weather horizon returning a Results object."""
        # Alias for object attributes
        s, w, cp = self.soil, self.weather, self.params

        # Dimensions
        H, W, L, T = s.H, s.W, s.n_layers, len(w.temp)

        # Allocate state arrays
        state = self._alloc_state(H, W, L, T)

        # ----- t = 0 initialization
        dds, ct_max_prev = self._init_state_t0(s.dds0, s, cp, state)
        layer_threshold = s.soil_layer_depth_mm  # per-layer depth (mm)

        # ====================== main loop ======================
        for t in range(T):
            if t > 0:
                dds = self._advance_dds(dds, s.crop_mask)
                # roll yesterday's layers into today’s slot (in-place updates)
                state["au"][:, :, :, t] = state["au"][:, :, :, t - 1]

            root_prev = (
                state["root_depth"][:, :, t - 1]
                if t > 0
                else state["root_depth"][:, :, 0]
            )

            # --- Water availability BEFORE evap (used for DD90 & evap)
            # Calculate current available water fraction
            p_au_now = self._compute_p_au(
                au_t=state["au"][:, :, :, t],
                root_depth=root_prev,
                aut=s.aut,
                layer_threshold=s.soil_layer_depth_mm,
                L=s.n_layers,
            )

            # Store in state
            state["p_au"][
                :, :, t
            ] = p_au_now  # as in original code (overwritten later)

            # --- Water stress coefficients (CEH for CT, RUE, HI/IC)
            # Calculate coefficients
            ceh_t, ceh_r_t, ceh_pc_t = self._water_stress_coeffs(
                p_au_now, s.crop_mask, cp
            )
            # Store in state
            state["ceh"][:, :, t] = ceh_t
            state["ceh_r"][:, :, t] = ceh_r_t
            state["ceh_pc"][:, :, t] = ceh_pc_t

            # --- Thermal stress factor for RUE
            # Calculate coefficient
            t_eur_t = self._thermal_stress_scalar(
                float(w.temp[t]), H, W, cp, s.crop_mask
            )
            # Store in state
            state["t_eur"][:, :, t] = t_eur_t

            # --- Actual RUE
            # Calculate coefficient
            eur_act_t = self._rue_actual(cp.eur_pot, t_eur_t, ceh_r_t)
            # Store in state
            state["eur_act"][:, :, t] = eur_act_t

            # --- Canopy cover next
            # Save previous cover
            ct_prev = (
                state["cover"][:, :, t - 1]
                if t > 0
                else state["cover"][:, :, 0]
            )
            # Calculate next cover and update running max
            ct_t, ct_max_prev = self._cover_next(
                ct_prev, dds, ceh_t, cp, ct_max_prev, s.crop_mask
            )
            # Store in state
            state["cover"][:, :, t] = ct_t

            # --- Transpiration, effective precipitation, and evaporation
            # Calculate fluxes (and update dd90 in-place)
            transp_t, ppef_t, eva_t, state["dd90"] = self._fluxes_step(
                et0_t=float(w.et0[t]),
                precip_t=float(w.precip[t]),
                cover_t=ct_t,
                ceh_r_t=ceh_r_t,
                p_au_now=p_au_now,
                dd90=state["dd90"],
                cp=cp,
                crop_mask=s.crop_mask,
            )
            # Store in state
            state["transpiration"][:, :, t] = transp_t
            state["eff_precip"][:, :, t] = ppef_t
            state["soil_evap"][:, :, t] = eva_t

            # --- Per-layer water balance with percolation
            # With previously calculated fluxes, compute new layer water
            # taking into account precipitation and evaporation on the top
            # layer, and transpiration and percolation on all layers.
            au_t = self._water_balance_layers(
                au_prev=state["au"][:, :, :, t],
                transp_t=transp_t,
                ppef_t=ppef_t,
                eva_t=eva_t,
                aut=s.aut,
                root_prev=root_prev,
                L=L,
                layer_threshold=layer_threshold,
            )
            state["au"][:, :, :, t] = au_t

            # --- Root length growth
            state["root_depth"][:, :, t] = self._root_next(
                root_prev, ceh_r_t, cp
            )

            # --- Water availability AFTER evap
            # recompute p_au AFTER balance (final output for day t)
            state["p_au"][:, :, t] = self._compute_p_au(
                au_t=state["au"][:, :, :, t],
                root_depth=state["root_depth"][:, :, t],
                aut=s.aut,
                layer_threshold=s.soil_layer_depth_mm,
                L=s.n_layers,
            )

            # --- harvest index (logistic around flowering)
            ici_t = self._harvest_index(dds, cp)

            # --- biomass and yield
            bi_t, bt_t, y_t = self._biomass_yield_step(
                cover_t=ct_t,
                par_t=float(w.par[t]),
                eur_act_t=eur_act_t,
                bt_prev=(
                    state["biomass_cum"][:, :, t - 1]
                    if t > 0
                    else np.zeros((H, W))
                ),
                ici_t=ici_t,
                ceh_r_t=ceh_r_t,
                harvest_index=cp.harvest_index,
            )
            state["biomass_daily"][:, :, t] = bi_t
            state["biomass_cum"][:, :, t] = bt_t
            state["yield_tensor"][:, :, t] = y_t

        # ---------- package ----------
        state["dates"][:] = np.arange(T)  # (T,)
        state["temp"][:] = w.temp
        state["par"][:] = w.par
        state["precip"][:] = w.precip
        state["et0"][:] = w.et0

        return self._package_results(state)

    # --------------------------- End of public API --------------------------

    # ---------------------------
    # Allocation & init
    # ---------------------------
    @staticmethod
    def _alloc_state(H: int, W: int, L: int, T: int) -> dict[str, np.ndarray]:
        """
        Allocate zero-initialized arrays for all model state variables.

        Parameters
        ----------
        H : int
            Grid height (number of rows).
        W : int
            Grid width (number of columns).
        L : int
            Number of soil layers.
        T : int
            Number of time steps in the weather horizon.

        Returns
        -------
        dict of {str: ndarray}
            Mapping from variable name to NumPy array, with shapes and dtypes:
            - 1D time series (dtype=float unless noted):
            * ``dates``: (T,), **int**
            * ``temp``, ``par``, ``precip``, ``et0``: (T,), float
            - 2D diagnostics (H, W), **int**:
            * ``dd90``: drought-days counter (consecutive days with
            ``p_au < 0.9``)
            - 3D fields (H, W, T), float:
            ``root_depth``, ``transpiration``, ``eff_precip``, ``soil_evap``,
            ``p_au``, ``ceh``, ``ceh_r``, ``ceh_pc``, ``cover``,
            ``t_eur``, ``eur_act``, ``biomass_daily``, ``biomass_cum``,
            ``yield_tensor``
            - 4D layered field (H, W, L, T), float:
            ``au``

        Notes
        -----
        - Arrays are allocated in C-order and zero-initialized. Callers fill
        them in-place during the simulation loop.
        - DD90 is the drought-days counter used for evaporation decay. It
        could probably be generalized to tune the percentage, and we could
        also make it store the history of drought days if needed by reshaping
        to (H, W, T).
        """

        def zT_f() -> np.ndarray:
            return np.zeros((T,), dtype=float)

        def zT_i() -> np.ndarray:
            return np.zeros((T,), dtype=int)

        def zHW() -> np.ndarray:
            return np.zeros((H, W), dtype=int)

        def zHWT() -> np.ndarray:
            return np.zeros((H, W, T), dtype=float)

        def zHWLT() -> np.ndarray:
            return np.zeros((H, W, L, T), dtype=float)

        state: dict[str, np.ndarray] = {
            # 1D time series
            "dates": zT_i(),
            "temp": zT_f(),
            "par": zT_f(),
            "precip": zT_f(),
            "et0": zT_f(),
            # 2D diagnostics
            "dd90": zHW(),
            # 3D fields
            "root_depth": zHWT(),
            "transpiration": zHWT(),
            "eff_precip": zHWT(),
            "soil_evap": zHWT(),
            "p_au": zHWT(),
            "ceh": zHWT(),
            "ceh_r": zHWT(),
            "ceh_pc": zHWT(),
            "cover": zHWT(),
            "t_eur": zHWT(),
            "eur_act": zHWT(),
            "biomass_daily": zHWT(),
            "biomass_cum": zHWT(),
            "yield_tensor": zHWT(),
            # 4D layered field
            "au": zHWLT(),
        }
        return state

    def _init_state_t0(
        self, dds0: Array, s: Soil, cp: CropParams, state: dict
    ) -> tuple[Array, Array]:
        """
        Initialize model state at t=0 (in-place).

        This sets ``cover[:, :, 0]``, layer water ``au[:, :, :, 0]``,
        continuous drought-days counter ``dd90[:, :]`` (where drought is
        defined as ``p_au < 0.9``), and ``root_depth[:, :, 0]``, and also
        primes ``p_au[:, :, 0]`` from these initial fields. Returns the
        working days-after-sowing array and the initial running-maximum cover.

        Parameters
        ----------
        dds0 : ndarray of shape (H, W)
            Days after sowing at t=0 (can be negative before emergence).
        s : Soil
            Soil container (uses ``crop_mask``, ``water0``, ``aut``,
            and ``soil_layer_depth_mm``).
        cp : CropParams
            Crop parameters (used by root/cover initializers).
        state : dict of {str, ndarray}
            Preallocated arrays from ``_alloc_state``.

        Returns
        -------
        dds : ndarray of shape (H, W)
            Working copy of days-after-sowing.
        ct_max_prev : ndarray of shape (H, W)
            Running maximum canopy cover up to (and including) t=0.
        """
        # Working copy of DAS
        dds = np.array(dds0, copy=True)

        # Cover at t=0
        cover0 = self._get_init_cover_t0(dds, s.crop_mask, scheme="fill_ones")
        state["cover"][:, :, 0] = cover0

        # Per-layer available water at t=0
        au0 = self._get_init_au_layers_t0(s.water0, s.aut, s.crop_mask)
        state["au"][:, :, :, 0] = au0

        # Initial root depth at t=0
        root0 = self._initial_root_lengths(dds, s.crop_mask, cp)
        state["root_depth"][:, :, 0] = root0

        # Prime p_au at t=0 (same logic used pre-evap in the loop)
        state["p_au"][:, :, 0] = self._compute_p_au(
            au_t=state["au"][:, :, :, 0],
            root_depth=state["root_depth"][:, :, 0],
            aut=s.aut,
            layer_threshold=s.soil_layer_depth_mm,
            L=s.n_layers,
        )
        # Note that this wasn't being calculated in the original MATLAB code
        # (but should have been, in my opinion).

        # Drought-day counter
        state["dd90"][:, :] = 0

        # Running maximum for senescence slope
        ct_max_prev = cover0.copy()

        return dds, ct_max_prev

    # ---------------------------
    # Days-after-sowing update
    # ---------------------------
    @staticmethod
    def _advance_dds(dds: Array, crop_mask: Array) -> Array:
        """Increment days-after-sowing on crop pixels by 1 day."""
        return dds + 1.0 * crop_mask

    # ---------------------------
    # Initial conditions for layers and cover
    # ---------------------------
    def _get_init_au_layers_t0(
        self,
        water0: np.ndarray,
        aut: np.ndarray,
        crop_mask: np.ndarray,
        *,
        n_layers: int | None = None,
        deep_fill_fraction: float = 0.5,
    ) -> np.ndarray:
        """
        Initialize available water per layer at time t=0 (per pixel).

        Parameters
        ----------
        water0 : ndarray of shape (H, W)
            Initial available water measured for the top soil layer (mm) at
            t=0. Values will be clipped to the corresponding layer capacity.
        aut : ndarray of shape (H, W) or (H, W, L)
            Available water capacity (mm). If 2D, the same per-pixel capacity
            is used for all layers. If 3D, it must provide the per-layer
            capacity for each pixel. Values are combined with ``crop_mask`` to
            zero capacities outside the crop.
        crop_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask selecting pixels belonging to the target crop.
            Capacities and initialized water outside the mask are set to zero.
        n_layers : int, optional
            Number of soil layers ``L``. If ``aut`` is 3D, it is inferred from
            ``aut.shape[2]``. If ``aut`` is 2D and ``n_layers`` is not
            provided, the method attempts to use ``self.soil.n_layers``;
            otherwise a ``ValueError`` is raised.
        deep_fill_fraction : float, default=0.5
            Fraction of the (masked) layer capacity used to initialize layers
            below the top layer (layers 1..L-1). Must satisfy
            ``0.0 <= deep_fill_fraction <= 1.0``.

        Returns
        -------
        au0 : ndarray of shape (H, W, L)
            Initial available water per layer at ``t=0`` (mm). Values are
            clipped to the (masked) layer capacities and are zero outside
            ``crop_mask``.

        Notes
        -----
        This routine mirrors the behavior used in the original MATLAB
        reference (``agromodel_model_plantgrowth_v27.m``, lines 90-93): the
        measured top-layer water is assigned directly to layer 0 (capped by
        capacity), while deeper layers are initialized as a fixed fraction of
        their capacity (default 0.5), also capped. This provides a reasonable
        starting profile when only surface measurements are available.

        Raises
        ------
        ValueError
            If ``aut.ndim`` is not 2 or 3, or if ``n_layers`` cannot be
            inferred when ``aut`` is 2D, or if ``deep_fill_fraction`` is
            outside ``[0, 1]``.

        Examples
        --------
        >>> au0 = self._init_au_layers_t0(
        ...     water0, self.aut, crop_mask, n_layers=self.soil.n_layers)
        >>> au0.shape
        (H, W, L)
        """
        if not (0.0 <= deep_fill_fraction <= 1.0):
            raise ValueError("deep_fill_fraction must be in [0, 1].")

        H, W = water0.shape
        mask3 = crop_mask.astype(float)[..., None]

        # Determine L and build per-layer capacities
        if aut.ndim == 3:
            H2, W2, L = aut.shape
            if (H2, W2) != (H, W):
                raise ValueError(
                    f"'aut' spatial shape mismatch: {(H2, W2)} vs {(H, W)}"
                )
            aut_layers = aut.astype(float, copy=False)
        elif aut.ndim == 2:
            if n_layers is None:
                n_layers = getattr(self.soil, "n_layers", None)
                if n_layers is None:
                    raise ValueError(
                        "Provide n_layers when 'aut' is 2D and "
                        "self.soil.n_layers is unavailable."
                    )
            L = int(n_layers)
            aut_layers = np.broadcast_to(
                aut.astype(float, copy=False)[..., None], (H, W, L)
            )
        else:
            raise ValueError(f"'aut' must be 2D or 3D, got ndim={aut.ndim}.")

        # Apply crop mask to capacities so non-crop pixels are zeroed
        aut_layers_masked = aut_layers * mask3

        # Allocate output (H, W, L)
        au0 = np.zeros((H, W, L), dtype=float)

        # Top layer: measured water0, clipped by capacity of layer 0
        au0[:, :, 0] = np.clip(
            water0.astype(float, copy=False), 0.0, aut_layers_masked[:, :, 0]
        )

        # Deeper layers: fraction of capacity, clipped
        if L > 1:
            au0[:, :, 1:] = np.clip(
                deep_fill_fraction * aut_layers_masked[:, :, 1:],
                0.0,
                aut_layers_masked[:, :, 1:],
            )

        return au0

    def _get_init_cover_t0(
        self,
        dds: np.ndarray,
        crop_mask: np.ndarray,
        *,
        scheme: str = "fill_ones",
    ) -> np.ndarray:
        """
        Initialize canopy cover at time t=0 on a per-pixel basis.

        Parameters
        ----------
        dds : ndarray of shape (H, W)
            Days after sowing at t=0 for each pixel. Values less than
            ``cp.dds_in`` indicate pre-emergence; values equal to
            ``cp.dds_in`` indicate emergence; values greater than
            ``cp.dds_in`` indicate post-emergence. Typically an
            integer grid, but floats are accepted.
        crop_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask that selects pixels belonging to the target crop.
            Pixels outside the mask are forced to zero cover.
        scheme : {"fill_ones", "linear"}, default="fill_ones"
            Initialization scheme:
            - "fill_ones": Emulates the original MATLAB behavior. Sets cover
            to 0.0 for ``dds < cp.dds_in``, to ``cp.c_in`` for
            ``dds == cp.dds_in``, and to 1.0 for ``dds > cp.dds_in``; then
            masks and clips to ``[0.0, cp.c_max]``.
            - "linear": Uses a linear backfill from ``cp.c_in`` at
            ``cp.dds_in`` with slope ``cp.alpha1`` for ``dds >= cp.dds_in``,
            clipped to ``[0.0, cp.c_max]``, and 0.0 for ``dds < cp.dds_in``.

        Returns
        -------
        cover0 : ndarray of shape (H, W), dtype=float
            Initial canopy cover fraction at ``t=0`` for the grid. Values are
            in ``[0.0, cp.c_max]`` and zeroed outside ``crop_mask``.

        Notes
        -----
        - The "fill_ones" scheme corresponds to the simplified initialization
        in the MATLAB reference, which is optimistic for pixels already past
        emergence at ``t=0``. See agromodel_model_plantgrowth_v27.m > lines
        130~134 and 340~357.
        - The "linear" scheme provides a more consistent backfill based on
        the crop parameters, but may still be optimistic since it does not
        account for prior stress history (we don't know ceh(t) for t<0).
        - If ``dds`` is floating-point and may not be exactly integral, you
        may prefer using a tolerance around ``cp.dds_in`` before equality
        tests, e.g. ``np.isclose(dds, cp.dds_in, atol=0.5)`` for day-level
        semantics.

        Raises
        ------
        AssertionError
            If ``cp.c_in > cp.c_max`` or ``cp.c_max > 1.0``.

        Examples
        --------
        >>> cover0 = self._get_init_cover_t0(
        ...     dds, crop_mask, scheme="fill_ones")
        >>> cover0.shape
        (H, W)
        """
        assert (
            0.0 <= self.params.c_in <= self.params.c_max <= 1.0
        ), "Require 0 ≤ c_in ≤ c_max ≤ 1"

        # We initialize to zero
        cover_t = np.zeros_like(dds, dtype=float)

        # Emergence day
        just_emerged = dds == self.params.dds_in
        cover_t[just_emerged] = self.params.c_in

        # We select the ones already growing at t=0
        already_growing = dds > self.params.dds_in

        # If already growing, we need to backfill the cover
        # Constant backfill to 1.0 after emergence (original MATLAB behavior)
        if scheme == "fill_ones":
            cover_t[already_growing] = 1.0
        # Linear backfill from c_in with slope alpha1 for dds ≥ dds_in
        # (still optimistic since we don't know the stress history; senescence
        # phase is not handled here)
        elif scheme == "linear":
            cover_t[already_growing] = np.clip(
                self.params.c_in
                + (dds[already_growing] - self.params.dds_in)
                * self.params.alpha1,
                0.0,
                self.params.c_max,
            )
        else:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Use 'fill_ones' or 'linear'."
            )

        # Mask and clip to [0, c_max]
        cover0 = np.clip(
            cover_t * crop_mask.astype(float), 0.0, float(self.params.c_max)
        )
        return cover0

    # ---------------------------
    # Water availability & stresses
    # ---------------------------
    @staticmethod
    def _compute_p_au(
        au_t: Array,
        root_depth: Array,
        aut: Array,
        layer_threshold: Array,
        L: int,
    ) -> Array:
        r"""
        Compute fraction of available water for plants in each pixel.

        Compute fraction of available water in each pixel, taking into account
        the root-accessible layers. For each pixel (i, j), build a per-layer
        accessibility mask using the **strict** rule used in the original
        MATLAB code:
            - layer 0 is always accessible
            - layer k (>=1) is accessible iff root_depth > k * layer_threshold

        Let ``mask_layers`` be that boolean mask (H, W, L), ``A_k`` the
        available water in layer k (mm), and ``aut`` the per-layer capacity
        (mm). Then

        .. math::

            n_{acc} = \\sum_{k=0}^{L-1} \\mathbf{1}\\{\\text{accessible}\\},
            \\quad
            W_{acc} = \\sum_{k=0}^{L-1} A_k \\cdot
            \\mathbf{1}\\{\\text{accessible}\\},\\\\
            \\text{cap}_{acc} = n_{acc} \\cdot aut,\\quad
            p_{AU} = \\mathrm{clip}\\bigg(
            \\frac{W_{acc}}{\\max(\\text{cap}_{acc},\\,10^{-9})},\\ 0,\\ 1
            \\bigg).

        Parameters
        ----------
        au_t : ndarray, shape (H, W, L)
            Available water per layer at the *current* day (mm).
        root_depth : ndarray, shape (H, W)
            Root depth at the end of the previous day (mm) for pre-evaporation
            use, or at the end of today for the post-balance output.
        aut : ndarray, shape (H, W)
            Per-layer available water capacity (mm).
        layer_threshold : float or ndarray, shape (H, W)
            Thickness (depth) of each soil layer (mm).
        L : int
            Number of soil layers.

        Returns
        -------
        ndarray, shape (H, W)
            Fraction of available water (in [0, 1]) within the root-accessible
            portion of the soil profile.
        """
        H, W = root_depth.shape

        # Number of accessible layers (1..L)
        n_accessible = (
            np.floor_divide(
                np.maximum(root_depth, 0.0), layer_threshold
            ).astype(int)
            + 1
        )
        n_accessible = np.clip(n_accessible, 1, L)

        # Per-layer accessibility mask: layer k is usable iff n_accessible > k
        mask_layers = np.stack([(n_accessible > k) for k in range(L)], axis=2)

        # Total accessible water
        sum_layers = np.sum(au_t * mask_layers, axis=2)  # (H, W)

        # maximum (potential) accessible capacity
        cap_accessible = aut * n_accessible.astype(float)  # (H, W)

        # Fraction of available water, capped to [0, 1]:
        # It's the actual water total in the accessible layers (sum_layers)
        # over by the potential maximum accessible capacity (cap_accessible)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(
                sum_layers / np.maximum(cap_accessible, 1e-9), 0.0, 1.0
            )

    @staticmethod
    def stress_sigmoid(pau, up, down, c):
        """
        Generic water stress function based on a sigmoid curve.

        This function is designed to model the water stress response of plants
        based on the available water (pau) relative to defined upper and lower
        thresholds (up and down). The shape of the response curve is
        controlled by the parameter c.

        It implements the basic formula:
        1 - (exp(hrs*c)-1)/(exp(c)-1)

        Parameters
        ----------
        pau : array_like
            Available water (porcentaje de agua útil).
        up : float
            Upper threshold for water stress.
        down : float
            Lower threshold for water stress.
        c : float
            Shape parameter for the sigmoid function.

        Returns
        -------
        s : array_like
            Water stress coefficient (0 to 1) for each pixel (i,j).

        Notes
        -----
        `np.expm1` is used to compute the exponential function minus
        one, for all elements in the array. See docs:
        https://numpy.org/doc/stable/reference/generated/numpy.expm1.html
        """
        hrs = (up - pau) / max(up - down, 1e-6)
        # 1 - (exp(hrs*c)-1)/(exp(c)-1), clamped to [0,1]
        num = np.expm1(np.clip(hrs, 0, 1) * c)
        den = np.expm1(c)
        s = 1.0 - num / max(den, 1e-9)
        # we replace values under the minimum by 0
        s = np.where(pau < down, 0.0, s)
        # and values over the maximum by 1
        s = np.where(pau > up, 1.0, s)
        return s

    @staticmethod
    def _water_stress_coeffs(
        p_au_now: Array, crop_mask: Array, cp: CropParams
    ) -> tuple[Array, Array, Array]:
        """Get stress coeff. for CT (ceh), RUE (ceh_r), and HI/IC (ceh_pc)."""
        ceh = (
            CropModel.stress_sigmoid(
                p_au_now, cp.au_up, cp.au_down, cp.c_forma
            )
            * crop_mask
        )
        ceh_r = (
            CropModel.stress_sigmoid(
                p_au_now, cp.au_up_r, cp.au_down_r, cp.c_forma_r
            )
            * crop_mask
        )
        ceh_pc = (
            CropModel.stress_sigmoid(
                p_au_now, cp.au_up_pc, cp.au_down_pc, cp.c_forma_pc
            )
            * crop_mask
        )
        return ceh, ceh_r, ceh_pc

    @staticmethod
    def _thermal_stress_scalar(
        Ti: float, H: int, W: int, cp: CropParams, crop_mask: Array
    ) -> Array:
        """
        Thermal stress factor T°EUR on the (H,W) grid for daily temperature.

        Thermal stress factor T°EUR on the (H,W) grid for scalar daily mean
        temperature. Currently, we model the temperature response as a
        trapezoid:
        - Crushing cold (factor=0) below tbr
        - Linear increase from 0 to 1 between tbr and tor1
        - No stress (factor=1) between tor1 and tor2
        - Linear decrease from 1 to 0 between tor2 and tcr
        - Crushing heat (factor=0) above tcr

        Parameters
        ----------
        Ti : float
            Daily mean temperature (°C).
        H : int
            Grid height (number of rows).
        W : int
            Grid width (number of columns).
        cp : CropParams
            Crop parameters (uses ``tbr``, ``tor1``, ``tor2``, ``tcr``).
        crop_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask selecting pixels belonging to the target crop.
            Values outside the mask are set to zero.

        Returns
        -------
        t_eur_t : ndarray of shape (H, W)
            Thermal stress factor for RUE at temperature Ti, in [0.0, 1.0].
        """
        # Constant grid with today's temperature for each pixel
        ti = np.full((H, W), Ti, dtype=float)

        # Initialize to zero (full stress)
        th = np.zeros_like(ti)
        # Notice that this won't be overwritten neither in the cold nor in the
        # heat crushing zones, so it remains 0 there.

        # Region with no thermic stress
        th[(ti > cp.tor1) & (ti < cp.tor2)] = 1.0

        # Select regions with stress due to LOW temperatures, but above
        # the minimum viable temperature (tbr)
        sel = (ti > cp.tbr) & (ti < cp.tor1)
        # for those, we use a linear interpolation between factor 0 at tbr
        # (crushing cold) and a factor 1 at tor1
        th[sel] = (ti[sel] - cp.tbr) / (cp.tor1 - cp.tbr + 1e-9)

        # Select regions with stress due to HIGH temperatures, but below
        # the maximum viable temperature (tcr)
        sel = (ti > cp.tor2) & (ti < cp.tcr)
        # for those, we use a linear interpolation between factor 1 at tor2
        # and a factor 0 at tcr (crushing heat)
        th[sel] = (cp.tcr - ti[sel]) / (cp.tcr - cp.tor2 + 1e-9)
        return th * crop_mask

    # ---------------------------
    # Radiation-use efficiency
    # ---------------------------
    @staticmethod
    def _rue_actual(eur_pot: float, t_eur_t: Array, ceh_r_t: Array) -> Array:
        """
        Compute actual radiation-use efficiency (RUE) on the (H,W) grid.

        Actual RUE = potential RUE × thermal-stress factor × water-stress
        factor.

        Parameters
        ----------
        eur_pot : float
            Potential radiation-use efficiency (g/MJ), a crop-specific
            constant.
        t_eur_t : ndarray of shape (H, W)
            Thermal stress factor for RUE at time t, in [0.0, 1.0].
        ceh_r_t : ndarray of shape (H, W)
            Water-stress coefficient for RUE at time t, in [0.0, 1.0].
        """
        return eur_pot * t_eur_t * ceh_r_t

    # ---------------------------
    # Cover dynamics
    # ---------------------------
    @staticmethod
    def _cover_next(
        ct_prev: Array,
        dds: Array,
        ceh_t: Array,
        cp: CropParams,
        ct_max_prev: Array,
        crop_mask: Array,
    ) -> tuple[Array, Array]:
        """
        Get canopy cover at day *t* and the running maximum for senescence.

        The update is piecewise in days after sowing (DAS) with water-stress
        modulation via ``ceh_t ∈ [0, 1]``:

        - **Pre-leaf:** if ``dds ≤ dds_in``, then ``ct_t = 0``.
        - **Emergence:** if ``dds = dds_in``, then ``ct_t = c_in``.
        - **Growth:** if ``dds_in < dds < dds_max``, then
          ``ct_t = ct_{t-1} + α₁ · ceh_t``, where
          ``α₁ = (c_max - c_in)/(dds_max - dds_in)`` unless overridden.
        - **Plateau:** if ``dds_max ≤ dds < dds_sen``, then
          ``ct_t = ct_{t-1}``.
        - **Senescence:** define the running maximum
          ``ct_max(t) = max(ct_max(t-1), ct_{t-1})`` and the slope
          ``β₁ = (ct_max(t) - c_fin) / (dds_fin - dds_sen)``;
          if ``dds ≥ dds_sen``, then
          ``ct_t = max(ct_{t-1} - β₁ · (2 − ceh_t), c_fin)``.

        After the piecewise update, the crop mask is applied and the result is
        clipped to ``[0, c_max]``.

        Parameters
        ----------
        ct_prev : ndarray of shape (H, W)
            Canopy cover at day ``t-1``.
        dds : ndarray of shape (H, W)
            Days after sowing evaluated for day ``t``.
        ceh_t : ndarray of shape (H, W)
            Water-stress factor for cover (CEH) in ``[0, 1]`` at day ``t``.
        cp : CropParams
            Crop parameter set (phenology breakpoints and cover constants).
        ct_max_prev : ndarray of shape (H, W)
            Running maximum canopy cover up to (and excluding) day ``t``.
        crop_mask : ndarray of shape (H, W), bool
            Mask of crop pixels (non-crop pixels are zeroed).

        Returns
        -------
        ct_t : ndarray of shape (H, W)
            Canopy cover at day ``t`` (masked and clipped).
        ct_max : ndarray of shape (H, W)
            Updated running maximum canopy cover
            ``max(ct_max_prev, ct_prev)``.

        Notes
        -----
        - All operations are pointwise per pixel; no spatial coupling is
        introduced.
        - Note that here they have the correction I suggested from
        looking to their manuscript: in ideal water conditions,
        the water stress coefficient should be 1, and then we would
        reduce canopy cover just by beta1; if we have maximum stress,
        the water stress coefficient should be 0, and we would reduce
        canopy cover by beta1 times 2.

        """
        # We initialize the array for the current ct
        ct_i = np.ones_like(ct_prev)

        # ---- Pre-leaves stage
        ct_i[dds <= cp.dds_in] = 0.0

        # ---- Emergence
        ct_i[dds == cp.dds_in] = cp.c_in

        # ---- Growth Phase
        grow = (dds > cp.dds_in) & (dds < cp.dds_max)
        ct_i[grow] = ct_prev[grow] + (cp.alpha1 * ceh_t[grow]) * ct_i[grow]

        # ---- Constant maximum cover phase
        stay = (dds >= cp.dds_max) & (dds < cp.dds_sen)
        ct_i[stay] = ct_prev[stay]

        # ---- Senescence
        # We use the running maximum to compute the senescence slope
        # (ct_max_prev is the max up to t-1, ct_prev is today's value;
        # note that the maximum is not the last one if we're already in
        # senescence for some time now)
        ct_max = np.maximum(ct_max_prev, ct_prev)
        # We use the running maximum to compute the senescence slope
        beta1 = (ct_max - cp.c_fin) / max(cp.dds_fin - cp.dds_sen, 1e-6)
        # we select the pixels in senescence
        sen = dds >= cp.dds_sen
        # and apply the senescence equation
        ct_i[sen] = np.maximum(
            ct_prev[sen] - beta1[sen] * (2.0 - ceh_t[sen]) * ct_i[sen],
            cp.c_fin,
        )

        # We apply the crop mask and clip to [0, c_max]
        ct_t = np.clip(ct_i * crop_mask, 0.0, cp.c_max)
        return ct_t, ct_max  # update ct_max_prev externally

    # ---------------------------
    # Fluxes: transpiration / effective precip / evaporation
    # ---------------------------
    @staticmethod
    def _fluxes_step(
        et0_t: float,
        precip_t: float,
        cover_t: Array,
        ceh_r_t: Array,
        p_au_now: Array,
        dd90: Array,
        cp: CropParams,
        crop_mask: Array,
    ) -> tuple[Array, Array, Array]:
        r"""
        Compute fluxes of water into and out of the soil for day *t*.

        This routine computes the daily water fluxes: water arriving to the
        soil as effective precipitation, water leaving the soil as
        transpiration through the crop, and water leaving the soil as direct
        evaporation. Soil evaporation depends on the amount of days with
        drought (i.e., when the available water is below 90%), counted by
        the ``dd90`` variable in the state, returned to be externally updated.
        Formulas (per-pixel):
        - **Transpiration:** :math:`T_t = \\mathrm{ceh\\_r}_t \\cdot
        (\\mathrm{cover}_t \\cdot K_C) \\cdot \\mathrm{ET0}_t`, where
        :math:`\\mathrm{ceh\\_r}_t` is the RUE correction for water stress
        (0 to 1, where 1 is no stress), :math:`\\mathrm{cover}_t` is the
        canopy cover fraction at day ``t``, :math:`\\mathrm{ET0}_t` is the
        reference evapotranspiration at day ``t`` [mm day⁻¹], and
        :math:`K_C` is the crop transpiration coefficient from ``cp.KC``.
        - **Effective precip.:** :math:`P^{\\mathrm{eff}}_t =
        f(\\mathrm{precip}_t)`, broadcast and masked, where
        :math:`f` is the effective-precipitation function defined in
        ``effective_precipitation()`` in the `hydrology` module and
        :math:`\\mathrm{precip}_t` is the daily precipitation at day
        ``t`` [mm].
        - **Soil evaporation:** :math:`E_t = 1.1\\,
        \\mathrm{ET0}_t\\,(1-\\mathrm{cover}_t)` and,
        where :math:`p_{\\mathrm{AU}}<0.9`, multiply by
        :math:`(\\max(\\mathrm{dd90}'_t,1))^{-1/2}` with
        :math:`\\mathrm{dd90}'_t=\\mathrm{dd90}_{t-1}+1`,
        resetting :math:`\\mathrm{dd90}'_t=0` where
        :math:`p_{\\mathrm{AU}}>0.9`.

        Parameters
        ----------
        et0_t : float
            Reference evapotranspiration at day ``t`` [mm day⁻¹].
        precip_t : float
            Precipitation at day ``t`` [mm].
        cover_t : ndarray, shape (H, W)
            Canopy cover fraction at day ``t``.
        ceh_r_t : ndarray, shape (H, W)
            Water-stress factor for RUE/Transpiration in ``[0, 1]`` at day
            ``t`` (precomputed).
        p_au_now : ndarray, shape (H, W)
            Fraction of available water before today’s balance (0–1).
        dd90 : ndarray, shape (H, W), int
            Drought-days counter (consecutive days with ``p_au_now < 0.9``).
            **Updated in-place** by this routine.
        cp : CropParams
            Crop parameters (uses ``KC``).
        crop_mask : ndarray, shape (H, W), bool
            Crop mask; fluxes are zero outside crop pixels.

        Returns
        -------
        transp_t : ndarray, shape (H, W)
            Transpiration [mm day⁻¹].
        ppef_t : ndarray, shape (H, W)
            Effective precipitation [mm day⁻¹].
        eva_t : ndarray, shape (H, W)
            Soil evaporation [mm day⁻¹].
        dd90_new : ndarray, shape (H, W), int
            Updated drought-days counter.
        """
        # Transpiration: ceh_r * (cover * KC) * ET0
        transp_t = ceh_r_t * (cover_t * cp.KC) * float(et0_t)

        # Effective precipitation (scalar -> grid) and mask
        ppef_t = effective_precipitation(float(precip_t)) * crop_mask

        # Update drought-days counter in place
        dd90_new = dd90 + 1
        dd90_new[p_au_now > 0.9] = 0

        # Evaporation base term
        eva_t = 1.1 * float(et0_t) * (1.0 - cover_t)
        # Soil evaporation with DD90 decay when p_au < 0.9
        low = p_au_now < 0.9
        if np.any(low):
            decay = np.power(np.maximum(dd90_new.astype(float), 1.0), -0.5)
            eva_t[low] = 1.1 * float(et0_t) * (1.0 - cover_t[low]) * decay[low]

        # Mask fluxes
        transp_t *= crop_mask
        eva_t *= crop_mask

        return transp_t, ppef_t, eva_t, dd90_new

    # ---------------------------
    # Layer water balance
    # ---------------------------
    @staticmethod
    def _water_balance_layers(
        au_prev: Array,
        transp_t: Array,
        ppef_t: Array,
        eva_t: Array,
        aut: Array,
        root_prev: Array,
        L: int,
        layer_threshold: Array,
    ) -> Array:
        """
        Update per-layer water: equipartitioned transpiration and percolation.

        Accessibility for deeper layers follows the original strict rule:
        layer k (k>=1) is usable **only if**
        ``root_prev > k * layer_threshold``. Transpiration is split
        **equally** among the accessible layers for each pixel.
        New water balance per layer (k=0..L-1):
        - Gain: percolation from above + effective precip. (only in top layer)
        - Loss: transpiration (equipartitioned over accessible layers) +
          soil evaporation (only in top layer)
        - Overflow percolates to the next layer down

        Parameters
        ----------
        au_prev : ndarray, shape (H, W, L)
            Layered available water at previous step [mm].
        transp_t : ndarray, shape (H, W)
            Transpiration at day ``t`` [mm day⁻¹].
        ppef_t : ndarray, shape (H, W)
            Effective precipitation at day ``t`` [mm day⁻¹], already masked.
        eva_t : ndarray, shape (H, W)
            Soil evaporation at day ``t`` [mm day⁻¹], already masked.
        aut : float or ndarray, shape (H, W)
            Per-layer available water capacity [mm]. Broadcast across layers.
        root_prev : ndarray, shape (H, W)
            Root depth before today’s growth [mm]; determines accessibility.
        L : int
            Number of soil layers.
        layer_threshold : float or ndarray, shape (H, W)
            Per-layer depth threshold [mm]; layer ``k`` is accessible where
            ``root_prev > k * layer_threshold``. Broadcast across the grid.

        Returns
        -------
        au_t : ndarray, shape (H, W, L)
            Updated layered available water after fluxes and percolation [mm].

        Notes
        -----
        - Inputs are not modified in place; the returned array is a copy.
        - All 2D fields are broadcast as needed over the layer dimension.
        - Mass is conserved locally except for overflow passed as percolation
        to deeper layers and clipping at [0, aut].
        """
        H, W = root_prev.shape
        au_t = au_prev.copy()

        # Number of accessible layers and mask
        n_accessible = (
            np.floor_divide(
                np.maximum(root_prev, 0.0), layer_threshold
            ).astype(int)
            + 1
        )
        n_accessible = np.clip(n_accessible, 1, L)
        mask_layers = np.stack([(n_accessible > k) for k in range(L)], axis=2)

        # Per-layer share of transpiration on accessible layers
        # Layer-wise loss = (transp / n_accessible) on accessible layers
        per_pixel_layer_share = transp_t / np.maximum(
            n_accessible.astype(float), 1.0
        )
        # shape (H, W)
        loss_layers = per_pixel_layer_share[..., None] * mask_layers
        # shape (H, W, L)

        # Percolate from top to bottom
        perc_prev = np.zeros((H, W), dtype=float)
        for k in range(L):
            gain = perc_prev.copy()
            if k == 0:
                gain += ppef_t - eva_t
            new_k = au_t[:, :, k] + gain - loss_layers[:, :, k]

            # Over-capacity percolates
            perc = np.maximum(new_k - aut, 0.0)
            au_t[:, :, k] = np.clip(new_k - perc, 0.0, aut)

            # Pass percolation to next layer
            perc_prev = perc

        return au_t

    # ---------------------------
    # Harvest index & biomass
    # ---------------------------
    @staticmethod
    def _harvest_index(dds: Array, cp: CropParams) -> Array:
        r"""
        Logistic harvest-index/ICI ramp around flowering.

        For each pixel, the index is zero before flowering and rises following
        a logistic curve after the flowering day ``df``:

        .. math::

            \\mathrm{ICI}(\\Delta d) =
            \\begin{cases}
            0, & \\Delta d \\le 0, \\\\
            \\dfrac{\\mathrm{ic\\_in}\\,\\mathrm{ic\\_pot\\_t}}
                {\\mathrm{ic\\_in}+(\\mathrm{ic\\_pot\\_t}-\\mathrm{ic\\_in})
                e^{-Y\\,\\Delta d}}, & \\Delta d > 0,
            \\end{cases}

        where :math:`\\Delta d = \\mathrm{dds} - \\mathrm{df}`.

        Parameters
        ----------
        dds : ndarray of shape (H, W)
            Days-after-sowing at the current day ``t`` (per pixel).
        cp : CropParams
            Crop parameters; uses ``df``, ``ic_in``, ``ic_pot_t``, and ``Y``.

        Returns
        -------
        ici : ndarray of shape (H, W)
            Instantaneous harvest index (unitless), in ``[0, ic_pot_t]``.
        """
        ddf = dds - cp.df
        ic_pot = cp.ic_pot_t  # scalar
        ici = np.zeros_like(ddf, dtype=float)
        mask = ddf > 0
        ici[mask] = (cp.ic_in * ic_pot) / (
            cp.ic_in + (ic_pot - cp.ic_in) * np.exp(-cp.Y * ddf[mask])
        )
        return ici

    @staticmethod
    def _biomass_yield_step(
        cover_t: Array,
        par_t: float,
        eur_act_t: Array,
        bt_prev: Array,
        ici_t: Array,
        ceh_r_t: Array,
        harvest_index: float,
    ) -> tuple[Array, Array, Array]:
        r"""
        Compute daily biomass, cumulative biomass, and yield for a single day.

        Formulas (per pixel):

        .. math::

            \\begin{aligned}
            B_t      &= \\max\\big(0,\\; \\mathrm{cover}_t\\;
                    \\mathrm{PAR}_t\\; \\mathrm{EUR}^{act}_t\\big),\\\\
            BT_t     &= BT_{t-1} + B_t,\\\\
            Y_t      &= BT_t\\; \\mathrm{ICI}_t\\; \\mathrm{CEH}_R{}_t\\; h,
            \\end{aligned}

        where ``h`` is ``harvest_index`` (keep 1.0 if ICI already encapsulates
        partitioning), ``PAR_t`` is scalar for day ``t``, and all other fields
        are grids ``(H, W)``.

        Parameters
        ----------
        cover_t : ndarray of shape (H, W)
            Canopy cover fraction at day ``t``.
        par_t : float
            Photosynthetically active radiation at day ``t`` [MJ m⁻² day⁻¹].
        eur_act_t : ndarray of shape (H, W)
            Actual radiation-use efficiency at day ``t`` [g MJ⁻¹].
        bt_prev : ndarray of shape (H, W)
            Cumulative biomass up to day ``t-1`` [g m⁻²].
        ici_t : ndarray of shape (H, W)
            Harvest index / partition coefficient at day ``t`` (unitless).
        ceh_r_t : ndarray of shape (H, W)
            Water-stress factor for RUE/Transpiration at day ``t`` in
            ``[0, 1]``.
        harvest_index : float
            Multiplicative factor applied when converting biomass to yield
            (unitless).

        Returns
        -------
        bi_t : ndarray of shape (H, W)
            Daily biomass increment at day ``t`` [g m⁻² day⁻¹].
        bt_t : ndarray of shape (H, W)
            Cumulative biomass up to day ``t`` [g m⁻²].
        y_t : ndarray of shape (H, W)
            Yield proxy at day ``t`` [g m⁻²].
        """
        bi_t = np.maximum(0.0, cover_t * par_t * eur_act_t)
        bt_t = bt_prev + bi_t
        y_t = bt_t * ici_t * ceh_r_t * harvest_index
        return bi_t, bt_t, y_t

    # ---------------------------
    # Root growth logic
    # ---------------------------
    def _root_increment(self, ceh_r_t: Array, cp: CropParams) -> Array:
        """Daily root increment (mm/day) under current water stress.

        Stress is taken into account via ceh_r_t as in
        agromodel_model_plantgrowth_v27.m, line 197.
        """
        return cp.root_growth_rate * ceh_r_t

    def _root_next(
        self, root_prev: Array, ceh_r_t: Array, cp: CropParams
    ) -> Array:
        """Next-day root length with optional cap."""
        return np.minimum(
            cp.root_max_mm, root_prev + self._root_increment(ceh_r_t, cp)
        )

    def _initial_root_lengths(
        self,
        dds0: Array,
        crop_mask: Array,
        cp: CropParams,
        # ceh_guess: float = 1.0,
    ) -> Array:
        """Initializes root lengths at t=0.

        Current Implementation:
        -----------------------
        Currently, roots are initialized to zero, following the original
        MATLAB code (see agromodel_model_plantgrowth_v27.m > line 86 & 197).
        I guess this is because root length is calculated as
        root[:, :, t] = np.minimum(cp.root_max_mm, root_prev + rg)
        where
        rg = cp.root_growth_rate * ceh_r[:, :, t]
        and `ceh_r` is the RUE correction for water stress (0 to 1,
        where 1 indicates no stress), which is based on the available water
        at time t. Since we don't know the stress history before t=0,
        they just set root_prev = 0 at t=0, which means that roots start
        growing from zero at the beginning of the simulation.

        Alternative Implementation (commented out):
        -------------------------------------------
        In the future, we could approximate initial root length from
        days-after-sowing at model start. Since we don't know the stress
        history before t=0, we could assume a constant stress multiplier
        `ceh_guess` (defaulting to some reasonable average value). I'll leave
        hints to this implementation here, commented out.
        """
        # # Alternative implementation (commented out):
        # # We calculate the amount of growth days (dds0 can be negative)
        # dpos = np.maximum(dds0, 0.0)  # only positive days after sowing
        # # contribute; dds<0 indicates that the crop will be sown in |dds|
        # # days

        # # This would be the implementation with a guess for ceh:
        # L0 = dpos * cp.root_growth_rate * float(ceh_guess)
        # L0 = np.minimum(L0, cp.root_max_mm)
        # return L0 * crop_mask

        # Current implementation: roots start at 0
        return np.zeros_like(dds0) * crop_mask

    # ---------------------------
    # Packaging results
    # ---------------------------
    @staticmethod
    def _package_results(state: dict[str, Array]) -> Results:
        """Construct Results object from the internal state dict."""
        return Results(
            dates=state["dates"],
            temp=state["temp"],
            par=state["par"],
            precip=state["precip"],
            et0=state["et0"],
            root_depth=state["root_depth"],
            transpiration=state["transpiration"],
            eff_precip=state["eff_precip"],
            soil_evap=state["soil_evap"],
            au_layers=state["au"],
            p_au=state["p_au"],
            ceh=state["ceh"],
            ceh_r=state["ceh_r"],
            ceh_pc=state["ceh_pc"],
            cover=state["cover"],
            t_eur=state["t_eur"],
            eur_act=state["eur_act"],
            biomass_daily=state["biomass_daily"],
            biomass_cum=state["biomass_cum"],
            yield_tensor=state["yield_tensor"],
        )
