from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from risknyield.core.data_containers import Results, Weather, Soil
from risknyield.core.crops import CropParams
from risknyield.library.hydrology import effective_precipitation

Array = np.ndarray


@dataclass(slots=True)
class CropModel:
    soil: Soil
    weather: Weather
    params: CropParams

    # Water stress function (generic)
    def stress_sigmoid(self, pau, up, down, c):
        """
        Water stress function based on a sigmoid curve.

        This function is designed to model the water stress response of plants
        based on the available water (pau) relative to defined upper and lower
        thresholds (up and down). The shape of the response curve is controlled
        by the parameter c.

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

    # -------------------------- Root growth logic --------------------------
    def _root_increment(self, ceh_r_t: Array, cp: CropParams) -> Array:
        """Daily root increment (mm/day) under current water stress.
        
        Stress is taken into account via ceh_r_t as in
        agromodel_model_plantgrowth_v27.m, line 197.
        """
        return cp.root_growth_rate * ceh_r_t

    def _root_next(self, root_prev: Array, ceh_r_t: Array, cp: CropParams) -> Array:
        """Next-day root length with optional cap."""
        return np.minimum(cp.root_max_mm, root_prev + self._root_increment(ceh_r_t, cp))

    # ----------------------- Initialization Methods ------------------------
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
        MATLAB code. I guess this is because root length is calculated as
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
        # # contribute; dds<0 indicates that the crop will be sown in |dds| days

        # # This would be the implementation with a guess for ceh:
        # L0 = dpos * cp.root_growth_rate * float(ceh_guess)
        # L0 = np.minimum(L0, cp.root_max_mm)
        # return L0 * crop_mask

        # Current implementation: roots start at 0
        return np.zeros_like(dds0) * crop_mask

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
            Days after sowing at t=0 for each pixel. Values less than ``cp.dds_in``
            indicate pre-emergence; values equal to ``cp.dds_in`` indicate emergence;
            values greater than ``cp.dds_in`` indicate post-emergence. Typically an
            integer grid, but floats are accepted.
        crop_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask that selects pixels belonging to the target crop. Pixels
            outside the mask are forced to zero cover.
        cp : object
            Crop-parameter object providing at least the attributes:
            ``c_in`` (float), ``dds_in`` (int or float), ``alpha1`` (float),
            and ``c_max`` (float). Must satisfy ``0.0 <= c_in <= c_max <= 1.0``.
        scheme : {"fill_ones", "linear"}, default="fill_ones"
            Initialization scheme:
            - "fill_ones": Emulates the original MATLAB behavior. Sets cover to
            0.0 for ``dds < cp.dds_in``, to ``cp.c_in`` for ``dds == cp.dds_in``,
            and to 1.0 for ``dds > cp.dds_in``; then masks and clips to
            ``[0.0, cp.c_max]``.
            - "linear": Uses a linear backfill from ``cp.c_in`` at ``cp.dds_in``
            with slope ``cp.alpha1`` for ``dds >= cp.dds_in``, clipped to
            ``[0.0, cp.c_max]``, and 0.0 for ``dds < cp.dds_in``.

        Returns
        -------
        cover0 : ndarray of shape (H, W), dtype=float
            Initial canopy cover fraction at ``t=0`` for the grid. Values are in
            ``[0.0, cp.c_max]`` and zeroed outside ``crop_mask``.

        Notes
        -----
        - The "fill_ones" scheme corresponds to the simplified initialization in
        the MATLAB reference, which is optimistic for pixels already past
        emergence at ``t=0``. See agromodel_model_plantgrowth_v27.m > lines
        130~134 and 340~357.
        - The "linear" scheme provides a more consistent backfill based on
        the crop parameters, but may still be optimistic since it does not
        account for prior stress history (we don't know ceh(t) for t<0).
        - If ``dds`` is floating-point and may not be exactly integral, you may
        prefer using a tolerance around ``cp.dds_in`` before equality tests, e.g.
        ``np.isclose(dds, cp.dds_in, atol=0.5)`` for day-level semantics.

        Raises
        ------
        AssertionError
            If ``cp.c_in > cp.c_max`` or ``cp.c_max > 1.0``.

        Examples
        --------
        >>> cover0 = self._get_init_cover_t0(dds, crop_mask, scheme="fill_ones")
        >>> cover0.shape
        (H, W)
        """
        assert 0.0 <= self.params.c_in <= self.params.c_max <= 1.0, "Require 0 ≤ c_in ≤ c_max ≤ 1"

        # We initialize to zero
        cover_t = np.zeros_like(dds, dtype=float)

        # Emergence day
        just_emerged = (dds == self.params.dds_in)
        cover_t[just_emerged] = self.params.c_in

        # We select the ones already growing at t=0
        already_growing = (dds > self.params.dds_in)

        # If already growing, we need to backfill the cover
        # Constant backfill to 1.0 after emergence (original MATLAB behavior)
        if scheme == "fill_ones":
            cover_t[already_growing] = 1.0
        # Linear backfill from c_in with slope alpha1 for dds ≥ dds_in
        # (still optimistic since we don't know the stress history; senescence
        # phase is not handled here)
        elif scheme == "linear":
            cover_t[already_growing] = np.clip(
                self.params.c_in + (dds[already_growing] - self.params.dds_in) * self.params.alpha1,
                0.0, self.params.c_max
            )
        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'fill_ones' or 'linear'.")

        # Mask and clip to [0, c_max]
        cover0 = np.clip(cover_t * crop_mask.astype(float), 0.0, float(self.params.c_max))
        return cover0

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
            Initial available water measured for the top soil layer (mm) at t=0.
            Values will be clipped to the corresponding layer capacity.
        aut : ndarray of shape (H, W) or (H, W, L)
            Available water capacity (mm). If 2D, the same per-pixel capacity is
            used for all layers. If 3D, it must provide the per-layer capacity for
            each pixel. Values are combined with ``crop_mask`` to zero capacities
            outside the crop.
        crop_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask selecting pixels belonging to the target crop. Capacities
            and initialized water outside the mask are set to zero.
        n_layers : int, optional
            Number of soil layers ``L``. If ``aut`` is 3D, it is inferred from
            ``aut.shape[2]``. If ``aut`` is 2D and ``n_layers`` is not provided,
            the method attempts to use ``self.soil.n_layers``; otherwise a ``ValueError``
            is raised.
        deep_fill_fraction : float, default=0.5
            Fraction of the (masked) layer capacity used to initialize layers
            below the top layer (layers 1..L-1). Must satisfy ``0.0 <= deep_fill_fraction <= 1.0``.

        Returns
        -------
        au0 : ndarray of shape (H, W, L)
            Initial available water per layer at ``t=0`` (mm). Values are clipped
            to the (masked) layer capacities and are zero outside ``crop_mask``.

        Notes
        -----
        This routine mirrors the behavior used in the original MATLAB reference
        (``agromodel_model_plantgrowth_v27.m``, lines 90-93): the measured top-layer
        water is assigned directly to layer 0 (capped by capacity), while deeper
        layers are initialized as a fixed fraction of their capacity (default 0.5),
        also capped. This provides a reasonable starting profile when only surface
        measurements are available.

        Raises
        ------
        ValueError
            If ``aut.ndim`` is not 2 or 3, or if ``n_layers`` cannot be inferred
            when ``aut`` is 2D, or if ``deep_fill_fraction`` is outside ``[0, 1]``.

        Examples
        --------
        >>> au0 = self._init_au_layers_t0(water0, self.aut, crop_mask, n_layers=self.soil.n_layers)
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
                raise ValueError(f"'aut' spatial shape mismatch: {(H2, W2)} vs {(H, W)}")
            aut_layers = aut.astype(float, copy=False)
        elif aut.ndim == 2:
            if n_layers is None:
                n_layers = getattr(self.soil, "n_layers", None)
                if n_layers is None:
                    raise ValueError("Provide n_layers when 'aut' is 2D and self.soil.n_layers is unavailable.")
            L = int(n_layers)
            aut_layers = np.broadcast_to(aut.astype(float, copy=False)[..., None], (H, W, L))
        else:
            raise ValueError(f"'aut' must be 2D or 3D, got ndim={aut.ndim}.")

        # Apply crop mask to capacities so non-crop pixels are zeroed
        aut_layers_masked = aut_layers * mask3

        # Allocate output (H, W, L)
        au0 = np.zeros((H, W, L), dtype=float)

        # Top layer: measured water0, clipped by capacity of layer 0
        au0[:, :, 0] = np.clip(water0.astype(float, copy=False), 0.0, aut_layers_masked[:, :, 0])

        # Deeper layers: fraction of capacity, clipped
        if L > 1:
            au0[:, :, 1:] = np.clip(
                deep_fill_fraction * aut_layers_masked[:, :, 1:],
                0.0,
                aut_layers_masked[:, :, 1:],
            )

        return au0


    # ------------------------------------------------------------------------
    #                          Main evolution routine
    # ------------------------------------------------------------------------
    def evolve(self) -> Results:
        """Run the simulation for the crop over the weather time horizon.

        Notes
        -----
        * This version **removes** legacy features from the MATLAB script:
          - no random shifts of sowing dates
          - no hail/frost damage (FCCH)
          - no ex-post yield rescaling (rend_esc)
          - no GIF/export; the method returns arrays in a `Results` object.
        * It keeps the core water balance in n layers, canopy cover dynamics,
          water/thermal stresses, RUE-based biomass, and yield via HI.
        """
        # ----------- Aliases and shapes
        s, w, cp = self.soil, self.weather, self.params
        H, W, L, T = s.H, s.W, s.n_layers, len(w.temp)

        # ----------- Allocate arrays
        root = np.zeros((H, W, T), float)
        cover = np.zeros((H, W, T), float)

        transp = np.zeros((H, W, T), float)
        ppef = np.zeros((H, W, T), float)
        eva = np.zeros((H, W, T), float)
        
        au = np.zeros((H, W, L, T), float)
        p_au = np.zeros((H, W, T), float)

        ceh = np.zeros((H, W, T), float)
        ceh_r = np.zeros((H, W, T), float)
        ceh_pc = np.zeros((H, W, T), float)

        t_eur = np.zeros((H, W, T), float)
        eur_act = np.zeros((H, W, T), float)

        bi = np.zeros((H, W, T), float)
        bt = np.zeros((H, W, T), float)
        rend = np.zeros((H, W, T), float)

        # DD90 es el contador de días con sequía - son los días seguidos con
        # agua útil menor a 90% (a generalizar)
        DD90 = np.zeros((H, W), int)

        # -------------------- Initial conditions at t=0 --------------------
        # Copy initial days after sowing
        dds = np.copy(s.dds0)
        # Canopy cover at t=0
        cover[:, :, 0] = self._get_init_cover_t0(dds, s.crop_mask, scheme="fill_ones")
        # Initial layer water distribution
        au[:, :, :, 0] = self._get_init_au_layers_t0(s.water0, s.aut, s.crop_mask)
        # Initialize roots based on dds0
        root[:, :, 0] = self._initial_root_lengths(dds, s.crop_mask, cp)
        # Roots are being initialized to 0 inspite of having dds > 0 at t=0 (!)
        # This is following the original MATLAB code, but seems odd. See
        # agromodel_model_plantgrowth_v27.m > line 86 & 197.
        # Maybe we should estimate an initial_root_lengths (?)

        # FROM SOIL, NOT FROM CROP (legacy: cp.layer_threshold_mm)
        layer_threshold = s.soil_layer_depth_mm

        # ============================= Time loop ============================
        for t in range(T):
            # Update (current) Days After Sowing
            if t > 0:
                dds = dds + 1.0 * s.crop_mask

            # We save the previous root length for water access and root
            # growth calculations
            root_prev = root[:, :, 0] if t == 0 else root[:, :, t - 1]

            # ------------------------ Water Dynamics ------------------------
            # carry yesterday's state into today's slot so we can update in-place
            if t > 0:
                au[:, :, :, t] = au[:, :, :, t-1]

            # water fraction before updating for the day (used for DD90 & evaporation)
            # Accessible capacity multiplier (e.g., 1 + I(root>500) + I(root>1000) + ...)
            accessible_mult = 1.0
            # sum_layers tiene el agua total del pixel ij, sumando sólo las capas accesibles
            # por la planta, de acuerdo al largo de su raíz
            sum_layers = au[
                :, :, 0, t
            ]  # will be overwritten later, but ok for pre-evap
            for k in range(1, L):
                mask_k = root_prev > k * layer_threshold
                sum_layers = sum_layers + au[:, :, k, t] * mask_k  # agua
                # total a ese tiempo accesible por las plantas del pixel ij
                accessible_mult = accessible_mult + mask_k  # ahora
                # accessible_mult pasa a ser un array 2D donde el
                # valor (i,j) indica el número de capas de suelo accesibles
                # por las plantas en ese pixel, de acuerdo al largo de sus raíces
            cap_accessible = (
                s.aut * accessible_mult
            )  # cant máxima (capacity) de agua accesible por el pixel ij,
            # sumando todas las capas (we're doing an element-wise product);
            # aut es el agua accesible máxima *por capa*, por el cual multiplicamos a accessible_mult
            # que me dice cuantas capas son accessibles por las plantas en cada pixel
            with np.errstate(invalid="ignore", divide="ignore"):
                # el porcentaje de agua útil para las plantas en el pixel ij es el agua total
                # accessible por las plantas `sum_layers`, dividido el máximo posible `cap_accessible`
                p_au_now = np.clip(
                    sum_layers / np.maximum(cap_accessible, 1e-9), 0.0, 1.0
                )
            p_au[:, :, t] = p_au_now  # porcentaje de agua útil

            # ----------- Water stresses coefficients calculation -----------
            #                    (for CT, EUR, and HI/IC)

            # Correction for the Canopy Cover CT due to water stress.
            # Remember that CT_i = CT_{i-1} \pm \alpha/\beta * CEH(t),
            # depending on the days after sowing. Cf. lines 490 and 507.
            ceh[:, :, t] = (
                self.stress_sigmoid(p_au_now, cp.au_up, cp.au_down, cp.c_forma)
                * s.crop_mask
            )

            # This is the correction for RUE due to water stress (RUE_{Act} = RUE_{Pot} * T°EUR * CEHR)
            ceh_r[:, :, t] = (
                self.stress_sigmoid(p_au_now, cp.au_up_r, cp.au_down_r, cp.c_forma_r)
                * s.crop_mask
            )

            # Stress correction for HI/IC - TO BE CHECKED (not in doc) (?)
            ceh_pc[:, :, t] = (
                self.stress_sigmoid(p_au_now, cp.au_up_pc, cp.au_down_pc, cp.c_forma_pc)
                * s.crop_mask
            )

            # ---------------- Thermal stress trapezoid for RUE --------------
            Ti = w.temp[t]
            ti = np.full((H, W), Ti)
            th = np.zeros_like(ti)
            # región sin estrés térmico
            th[(ti > cp.tor1) & (ti < cp.tor2)] = 1.0
            # mask for stressed, yet not null efficiency, due to LOW temperatures -- we “select” those values
            sel = (ti > cp.tbr) & (ti < cp.tor1)
            # for those, the efficiency is the fraction of departure from the
            # minimal viable temperature, with respect to the maximum departure.
            # (Below tbr the efficiency is 0, and above tor1 it is 1, so this is a
            # linear interpolation between those.)
            th[sel] = (ti[sel] - cp.tbr) / (cp.tor1 - cp.tbr + 1e-9)
            # Now we select stress, yet not null efficiency points, due to HIGH temperatures
            sel = (ti > cp.tor2) & (ti < cp.tcr)
            # and we interpolate again
            th[sel] = (cp.tcr - ti[sel]) / (cp.tcr - cp.tor2 + 1e-9)
            # And finally, T°EUR, the factor that gives us the correction to
            # the RUE for temperature stress
            # (remember that RUE_{Act} = RUE_{Pot} * T°EUR * CEHR)
            # depending on the daily mean temperature, is for the current
            # time t just these factors applied to the crop mask (so that
            # non-crop pixels are zeroed out).
            t_eur[:, :, t] = th * s.crop_mask

            # Actual RUE:
            # the Potential RUE (a crop-dependent constant eur_pot),
            # times the correction for water stress (ceh_r),
            # times the correction for temperature stress (t_eur).
            eur_act[:, :, t] = cp.eur_pot * ceh_r[:, :, t] * t_eur[:, :, t]

            # -------------------- Canopy cover dynamics --------------------

            # Canopy cover at previous time step
            ct_prev = cover[:, :, 0] if t == 0 else cover[:, :, t - 1]

            # We initialize the array for the current ct
            ct_i = np.ones_like(ct_prev)

            # ----------- Pre-leaves stage
            # Plants that haven't grown leaves yet have 0 cover
            ct_i[dds <= cp.dds_in] = 0.0

            # ----------- Initial Cover
            # Plants that are in the day where they start to have leaves have
            # the initial cover c_in
            ct_i[dds == cp.dds_in] = cp.c_in

            # ----------- Growth Phase
            # Now we select plants that are in the growth phase
            grow_phase = (dds > cp.dds_in) & (dds < cp.dds_max)
            # and for those, the new canopy cover will be the old one, plus
            # the potential growth rate alpha, corrected for water stress via
            # ceh.
            ct_i[grow_phase] = (
                ct_prev[grow_phase]
                + (cp.alpha1 * ceh[:, :, t][grow_phase]) * ct_i[grow_phase]
            )

            # ----------- Stay Phase (Max CT Phase)
            stay_phase = (dds >= cp.dds_max) & (dds < cp.dds_sen)
            ct_i[stay_phase] = ct_prev[stay_phase]

            # ----------- Senescence
            # Max canopy cover up to this point. (It's not the last one if
            # we're in the senescence phase for some time now.)
            ct_max = (
                np.maximum.accumulate(cover[:, :, : t + 1], axis=2).max(axis=2)
                if t > 0
                else ct_prev
            )

            # We calculate the slope for the decrease in cover
            beta1 = (ct_max - cp.c_fin) / max(cp.dds_fin - cp.dds_sen, 1e-6)

            # We select pixels in the senescence phase via the dds mask
            sen_phase = dds >= cp.dds_sen
            # We calculate the new canopy cover, and we take either that, or the stationary canopy cover c_fin
            ct_i[sen_phase] = np.maximum(
                ct_prev[sen_phase]
                - beta1[sen_phase] * (2.0 - ceh[:, :, t][sen_phase]) * ct_i[sen_phase],
                cp.c_fin,
            )
            # Note that here they have the correction I suggested from
            # looking to their manuscript: in ideal water conditions,
            # the water stress coefficient should be 1, and then we would
            # reduce canopy cover just by beta1; if we have maximum stress,
            # the water stress coefficient should be 0, and we would reduce
            # canopy cover by beta1 times 2.

            # Sanity Enforcement (should be replaced by a sanity check):
            # no cover should exceed the maximum or be negative.
            cover[:, :, t] = np.clip(ct_i * s.crop_mask, 0.0, cp.c_max)

            # -------------- Transpiration and soil evaporation --------------

            # ----------- Potential evapotranspiration (ET0)
            et0_t = float(w.et0[t])

            # ----------- Transpiration
            transp_t = ceh_r[:, :, t] * (cover[:, :, t] * cp.KC) * et0_t
            # where: `ceh_r` is the RUE correction for water stress (0 to 1,
            # where 1 indicates no stress), `cover` is the canopy cover
            # fraction, and `cp.KC` is the crop transpiration coefficient.
            transp[:, :, t] = transp_t

            # ----------- Effective precipitation
            # (spatially uniform forcing, applied to all pixels in crop)
            ppef_t = effective_precipitation(w.precip[t])
            ppef[:, :, t] = ppef_t * s.crop_mask

            # ----------- Soil evaporation
            # (with DD90 decay when p_au<0.9, as in script)
            # We add a day to the list of days with available water percentage below 0.9
            DD90 = DD90 + 1
            # We set that counter to 0 for the days where p_au > 0.9
            DD90[p_au_now > 0.9] = 0
            # Evaporation will be proportional to the potential
            # evapotranspiration, and the fraction of the area not covered by
            # the plant, i.e., 1-CT (with CT -> canopy cover).
            eva_t = 1.1 * et0_t * (1.0 - cover[:, :, t])
            # We select pixels with low available water
            low = p_au_now < 0.9
            # For those, transpiration is actually reduced by a factor
            # 1/\sqrt{DD90}, provided that DD90 > 1.
            eva_t[low] = (
                1.1  # why 1.1? (?)
                * et0_t  # potential evapotranspiration
                * (1.0 - cover[:, :, t][low])  # fraction of area not covered by plant
                * np.power(
                    np.maximum(DD90[low], 1.0), -0.5
                )  # decay in transpiration by reduced soil water
            )
            # Note: np.power raises the elements of the first array to powers
            # from second array, element-wise. See docs:
            # https://numpy.org/doc/stable/reference/generated/numpy.power.html

            # Finally, we save the evaporation for the current time step
            # (zeroing out the masked areas)
            eva[:, :, t] = eva_t * s.crop_mask

            # ----------------------------------------------------------------
            #                     Water balance by layers
            # ----------------------------------------------------------------

            # ----------- Transpiration sharing
            # Split transpiration among accessible layers based on root depth
            # We'll assume equipartitioning of transpired water among layers.

            # For each pixel, compute the number of accessible layers today
            n_accessible = np.floor_divide(np.maximum(root_prev, 0.0), layer_threshold).astype(int) + 1
            n_accessible = np.clip(n_accessible, 1, L)  # ensure in [1, L]
            
            # Per-layer mask: for layer k, it's 1 where k < n_accessible, else 0
            # Shape: (H, W, L)
            mask_layers = np.stack([ (n_accessible > k) for k in range(L) ], axis=2)

            # Per-pixel share = transp / n_accessible; broadcast to layers, then mask
            # Shape transp_t: (H, W)  -> expand to (H, W, 1)
            per_pixel_share = transp_t / np.maximum(n_accessible, 1)
            loss_layers = mask_layers * per_pixel_share[..., None]  # (H, W, L)

            # Update layers sequentially with correct, layer-specific loss
            perc_prev = np.zeros((H, W))
            for k in range(L):
                gain = perc_prev.copy()
                if k == 0:
                    gain = gain + ppef[:, :, t] - eva[:, :, t]
                loss_k = loss_layers[:, :, k]  # Note the layer-specific loss
                # au[:, :, k, t] = au[:, :, k, t] + gain - loss

                # perc = np.maximum(au[:, :, k, t] - aut, 0.0)
                # au[:, :, k, t] = np.clip(au[:, :, k, t], 0.0, aut)
                # perc_prev = perc

                # unconstrained update
                new_k = au[:, :, k, t] + gain - loss_k

                # percolation based on over-capacity
                perc = np.maximum(new_k - s.aut, 0.0)

                # store (remove percolation) and clip [0, aut]
                au[:, :, k, t] = np.clip(new_k - perc, 0.0, s.aut)

                # pass percolation downward
                perc_prev = perc


            # ----------- Root growth (mm)
            root[:, :, t] = self._root_next(root_prev, ceh_r[:, :, t], cp)

            # -------------- Update Fraction of available water --------------
            # - Recompute p_au after water balance for outputs (at end of day)

            # For this, we'll calculate the sum of the available water across layers
            sum_layers = au[:, :, 0, t]  # the initial layer always adds to the total

            # accessible_mult will indicate the number of layers accessible to
            # plants in each pixel
            accessible_mult = 1.0

            # For the other layers, we only sum useful water if they're
            # accessible by the plants
            for k in range(1, L):
                # for pixels where roots access the layer k
                mask_k = root[:, :, t] > k * layer_threshold
                # we add the available/useful water on layer k to the total,
                # but only for pixels where roots access the layer
                sum_layers = sum_layers + au[:, :, k, t] * mask_k
                # we add layer k to the number of accessible layers for the
                # appropriate pixels
                accessible_mult = accessible_mult + mask_k

            # ------ Maximum Capacity
            # For getting the fraction of available water, we need to know the
            # water that we actually have sum_layers, but also the maximum
            # capacity that the accessible layers can hold.
            # As the per-layer capacity aut is soil_layer_depth_mm * (cc - pmp),
            # and all layers have the same thickness/depth soil_layer_depth_mm,
            # we can simply multiply by the number of accessible layers
            # to get the total depth soil_layer_depth_mm * accessible_mult, thus
            # obtaining the capacity accessible to plants as
            # cap_accessible = accessible_mult * soil_layer_depth_mm * (cc - pmp),
            # i.e., cap_accessible = aut * accessible_mult
            cap_accessible = s.aut * accessible_mult

            # ------ Compute Fraction of Available Water
            # Finally, we compute the fraction of available water by dividing
            # the water that we actually have on the accessible layers by the
            # maximum capacity they can hold.
            with np.errstate(invalid="ignore", divide="ignore"):
                p_au[:, :, t] = np.clip(
                    sum_layers / np.maximum(cap_accessible, 1e-9), 0.0, 1.0
                )

            # -------------------- Harvest index dynamics --------------------
            # (simplified logistic with cp.df and cp.Y)
            ddf = dds - cp.df
            ic_pot = np.full((H, W), cp.ic_pot_t)
            # (Optional carry-over stress on ic_pot could be added here)
            ici = np.zeros((H, W))
            mask_flowering = ddf > 0
            # This formula is as taken from agromodel_model_plantgrowth_v27.m,
            # line 403
            ici[mask_flowering] = (cp.ic_in * ic_pot[mask_flowering]) / (
                cp.ic_in
                + (ic_pot[mask_flowering] - cp.ic_in)
                * np.exp(-cp.Y * ddf[mask_flowering])
            )

            # ---------------- Biomass and Yield Calculation ----------------
            # Daily biomass from PAR capture and RUE
            bi[:, :, t] = np.maximum(0.0, cover[:, :, t] * w.par[t] * eur_act[:, :, t])
            # Cumulative biomass
            bt[:, :, t] = bi[:, :, t] if t == 0 else bt[:, :, t - 1] + bi[:, :, t]
            # Yield
            rend[:, :, t] = bt[:, :, t] * ici * ceh_r[:, :, t] * cp.harvest_index

        # Package Results
        days = np.arange(T)
        return Results(
            dates=days,
            temp=w.temp,
            par=w.par,
            precip=w.precip,
            et0=w.et0,
            root_depth=root,
            transpiration=transp,
            eff_precip=ppef,
            soil_evap=eva,
            au_layers=au,
            p_au=p_au,
            ceh=ceh,
            ceh_r=ceh_r,
            ceh_pc=ceh_pc,
            cover=cover,
            t_eur=t_eur,
            eur_act=eur_act,
            biomass_daily=bi,
            biomass_cum=bt,
            yield_tensor=rend,
        )
