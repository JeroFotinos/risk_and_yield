from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple
import numpy as np

Array = np.ndarray
CropName = Literal["soy", "maize"]


# ----------------------------------------------------------------------------
#                               Utility helpers
# ----------------------------------------------------------------------------


def _to_array(x, shape: Tuple[int, ...]) -> Array:
    """Broadcast scalars/1D arrays to a target 2D shape.
    If x is None, returns None. If x is a 1D vector with length matching one
    dimension of shape, it is broadcast accordingly.
    """
    if x is None:
        return None
    x = np.asarray(x)
    if x.shape == ():
        return np.full(shape, float(x))
    if x.shape == shape:
        return x
    if x.ndim == 1 and x.shape[0] in shape:
        if x.shape[0] == shape[0]:
            return np.tile(x[:, None], (1, shape[1]))
        if x.shape[0] == shape[1]:
            return np.tile(x[None, :], (shape[0], 1))
    raise ValueError(f"Cannot broadcast array of shape {x.shape} to {shape}")


def effective_precipitation(pp: Array) -> Array:
    """Piecewise-linear effective precipitation (mm/day) following the MATLAB rule.

    Given daily precipitation `pp` (mm), returns ppef.
    The breakpoints are inspired by the original aux/pp_cotas mapping.
    """
    pp = np.asarray(pp)
    # Breakpoints and slopes derived from the MATLAB snippet
    # (0, 0), (25, 23.75), (50, 46.25), (75, 66.75), (100, 83), (125, 94.25), (150, 100.25)
    xp = np.array([0, 25, 50, 75, 100, 125, 150], dtype=float)
    yp = np.array([0, 23.75, 46.25, 66.75, 83.0, 94.25, 100.25], dtype=float)
    ppef = np.interp(pp, xp, yp, left=0.95 * pp, right=100.25 + 0.05 * (pp - 150))
    return ppef


# ----------------------------------------------------------------------------
#                               Data containers
# ----------------------------------------------------------------------------


@dataclass
class Weather:
    """Time series forcing for the model.

    Attributes
    ----------
    temp : (T,) mean air temperature (°C)
    par  : (T,) photosynthetically active radiation (MJ/m^2/day)
    precip : (T,) daily precipitation (mm)
    et0 : Optional[(T,)] reference ET (mm/day). If None, a crude proxy is used.
    """

    temp: Array
    par: Array
    precip: Array
    et0: Optional[Array] = None

    def __post_init__(self):
        self.temp = np.asarray(self.temp, dtype=float)
        self.par = np.asarray(self.par, dtype=float)
        self.precip = np.asarray(self.precip, dtype=float)
        if self.et0 is not None:
            self.et0 = np.asarray(self.et0, dtype=float)
        n = len(self.temp)
        if not (
            len(self.par) == len(self.precip) == n
            and (self.et0 is None or len(self.et0) == n)
        ):
            raise ValueError("All weather series must share the same length")

    @property
    def T(self) -> int:
        return self.temp.shape[0]


@dataclass
class CropParams:
    """Crop- and model-parameters controlling growth/stress dynamics.
    Many names mirror the original MATLAB variables for traceability.
    Defaults here are **maize** (from `parametros_maiz.m`).
    """

    # Soil water stress thresholds (fractions of available water)
    au_up: float = 0.72
    au_down: float = 0.20
    au_up_r: float = 0.69
    au_down_r: float = 0.0
    au_up_pc: float = 0.60
    au_down_pc: float = 0.15

    # Shape parameters for the stress response (dimensionless)
    c_forma: float = 4.9
    c_forma_r: float = 6.0
    c_forma_pc: float = 1.3

    # Canopy cover development (days after sowing, DAS)
    dds_in: int = 7
    dds_max: int = 47
    dds_sen: int = 87
    dds_fin: int = (
        120  # This was “no_days_cultivo” in MATLAB --- ver linea 29 de agromodel_model_plantgrowth_v27.m
    )
    c_in: float = 0.039  # initial cover at emergence
    c_fin: float = 0.01  # residual cover in senescence
    c_max: float = 0.89  # maximum attainable cover (maize population default)
    alpha1: float = (c_max - c_in) / (
        dds_max - dds_in
    )  # daily growth rate to max cover

    # Root dynamics
    root_growth_rate: float = 30.0  # mm/day (maize)
    root_max_mm: float = 2000.0  # This comes from
    # agromodel_model_plantgrowth_v27.m > line 197, where they take the roots
    # to be the minimum between 2000 mm and the new updated value. I think
    # they set 2000 mm as the max because they considered 4 layers of 500 mm.
    # If that's the case, we should generalize this to
    # root_max_mm = n_layers * layer_threshold_mm
    layer_threshold_mm: float = 500.0  # 500.0  # per-layer depth used for access rules
    # This comes from lines 220 to 223 in agromodel_model_plantgrowth_v27.m

    # Radiation use efficiency (biomass per MJ PAR)
    # Units: g DM per MJ PAR (g/MJ)
    eur_pot: float = 3.65

    # Thermal stress (simple trapezoid response)
    tbr: float = 8.0
    tor1: float = 29.0
    tor2: float = 39.0
    tcr: float = 45.0

    # Harvest index / ICI (logistic ramp around flowering)
    # (taken from parametros_maiz.m, lines 39 to 42)
    df: int = 58
    ic_in: float = 0.001
    ic_pot_t: float = 0.48
    Y: float = 0.19

    # Transpiration coefficient
    KC: float = 0.94

    # WARNING: Optional crude ET0 proxy (only used if ET0 not provided) -- SHOULD NOT BE (?)
    k_et0_T: float = 0.1
    k_et0_PAR: float = 0.02
    # WARNING: Hargreaves' Equation should have been previously used to estimate ET0
    # ET0(idx) = 0.0023*(TMPMED(idx)+17.78).*(RAD(idx)/2.45).*((TMPMAX(idx) - TMPMIN(idx)).^0.5);
    # as taken from unificar_climas_vs.m > line 61

    # # Harvest index to translate biomass to yield
    # harvest_index: float = 0.5
    # Harvest index multiplier (keep 1.0 to avoid double-counting with ICI) -- CHECK (?)
    harvest_index: float = 1.0


@dataclass
class Results:
    """Simulation outputs (time × space) for key variables."""

    dates: Array  # day indices 0..T-1
    temp: Array
    par: Array
    precip: Array
    et0: Array

    root_depth: Array  # (H,W,T)
    transpiration: Array  # (H,W,T)
    eff_precip: Array  # (H,W,T)
    soil_evap: Array  # (H,W,T)

    au_layers: Array  # (H,W,n_layers,T) available water per layer (mm)
    p_au: Array  # (H,W,T) fraction of available water

    ceh: Array  # (H,W,T) canopy stress from water
    ceh_r: Array  # (H,W,T) radiation stress from water
    ceh_pc: Array  # (H,W,T) HI stress from water

    cover: Array  # (H,W,T) canopy cover fraction

    t_eur: Array  # (H,W,T) thermal stress for RUE
    eur_act: Array  # (H,W,T) actual RUE (g/MJ scaled by stresses)

    biomass_daily: Array  # (H,W,T)  [g/m^2/day]
    biomass_cum: Array  # (H,W,T)  [g/m^2]
    yield_: Array  # (H,W,T)  [g/m^2] (equivalent units)

    @property
    def yield_tensor(self) -> Array:
        """Alias to the 3D yield array for convenience (H×W×T)."""
        return self.yield_

    @property
    def yield_tensor(self) -> Array:
        return self.yield_

    # SIMPLIFY (?)


# ----------------------------------------------------------------------------
#                                  Core model
# ----------------------------------------------------------------------------


@dataclass
class Soil:
    """Grid-based soil–crop water/biomass model (vectorized, n layers).

    Parameters
    ----------
    lat : (H,) or (H,W) latitudes
    lon : (W,) or (H,W) longitudes
    water0 : (H,W) initial available water per pixel & layer 1 (mm)
    dds0 : (H,W) days after sowing at t=0 (can be <0 before emergence)
    mask_maize, mask_soy : (H,W) boolean masks for pixels planted with each crop
    n_layers : number of soil layers (default 4)
    cc, pmp : field capacity & wilting point (fraction, 0-1)
    soil_depth_mm : scalar or (H,W) effective rootable soil depth per layer (mm)
                    The per-layer capacity aut = soil_depth_mm * (cc - pmp)
    """

    lat: Array
    lon: Array
    water0: Array
    dds0: Array
    mask_maize: Array
    mask_soy: Array
    n_layers: int = (
        4  # They were working with 4 soil layers in the last version of the code
    )
    cc: float = (
        0.27  # capacidad campo para agua disponible 0.32 - 0.35; de parametros_maiz.m
    )
    pmp: float = (
        0.12  # es un % funcion espacial del tipo de suelo (laboratorio/mapas); de parametros_maiz.m
    )
    soil_depth_mm: float | Array = (
        500.0  # should be the per-layer depth used for access rules
    )
    # This comes from lines 220 to 223 in agromodel_model_plantgrowth_v27.m
    # Note that it coincides with `CropParams.layer_threshold_mm` -- CHECK (?)

    H: int = field(init=False)
    W: int = field(init=False)

    def __post_init__(self):
        # Harmonize shapes
        if self.water0.ndim != 2:
            raise ValueError("water0 must be 2D (H,W)")
        self.H, self.W = self.water0.shape
        shape = (self.H, self.W)
        self.dds0 = _to_array(self.dds0, shape)
        self.mask_maize = _to_array(self.mask_maize.astype(bool), shape)
        self.mask_soy = _to_array(self.mask_soy.astype(bool), shape)
        if np.any(self.mask_maize & self.mask_soy):
            raise ValueError("A pixel cannot be both maize and soy")
        self.lat = _to_array(self.lat, shape)
        self.lon = _to_array(self.lon, shape)
        self.soil_depth_mm = _to_array(self.soil_depth_mm, shape)

        # Per-layer available water capacity (aut) in mm
        self.aut = self.soil_depth_mm * (float(self.cc) - float(self.pmp))

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
        cp,
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
        >>> cover0 = self._get_init_cover_t0(dds, crop_mask, cp, scheme="fill_ones")
        >>> cover0.shape
        (H, W)
        """
        assert 0.0 <= cp.c_in <= cp.c_max <= 1.0, "Require 0 ≤ c_in ≤ c_max ≤ 1"

        # We initialize to zero
        cover_t = np.zeros_like(dds, dtype=float)

        # Emergence day
        just_emerged = (dds == cp.dds_in)
        cover_t[just_emerged] = cp.c_in

        # We select the ones already growing at t=0
        already_growing = (dds > cp.dds_in)

        # If already growing, we need to backfill the cover
        # Constant backfill to 1.0 after emergence (original MATLAB behavior)
        if scheme == "fill_ones":
            cover_t[already_growing] = 1.0
        # Linear backfill from c_in with slope alpha1 for dds ≥ dds_in
        # (still optimistic since we don't know the stress history; senescence
        # phase is not handled here)
        elif scheme == "linear":
            cover_t[already_growing] = np.clip(
                cp.c_in + (dds[already_growing] - cp.dds_in) * cp.alpha1,
                0.0, cp.c_max
            )
        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'fill_ones' or 'linear'.")

        # Mask and clip to [0, c_max]
        cover0 = np.clip(cover_t * crop_mask.astype(float), 0.0, float(cp.c_max))
        return cover0


    # ------------------------------------------------------------------------
    #                          Main evolution routine
    # ------------------------------------------------------------------------
    def evolve(
        self,
        crop: CropName,
        weather: Weather,
        params_overrides: Optional[Dict[str, float]] = None,
    ) -> Results:
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
        cp = CropParams()
        if params_overrides:
            for k, v in params_overrides.items():
                if not hasattr(cp, k):
                    raise AttributeError(f"Unknown parameter '{k}'")
                setattr(cp, k, v)

        T = weather.T  # timesteps
        H, W = self.H, self.W  # width and height of the Area of Interest
        L = int(self.n_layers)  # number of soil layers

        # ----------- Choose crop mask (bool 2D)
        crop_mask = self.mask_maize if crop == "maize" else self.mask_soy
        crop_mask = crop_mask.astype(float)

        # ----------- Allocate arrays
        root = np.zeros((H, W, T))
        transp = np.zeros((H, W, T))
        ppef = np.zeros((H, W, T))
        eva = np.zeros((H, W, T))
        au = np.zeros((H, W, L, T))
        p_au = np.zeros((H, W, T))
        ceh = np.zeros((H, W, T))
        ceh_r = np.zeros((H, W, T))
        ceh_pc = np.zeros((H, W, T))

        cover = np.zeros((H, W, T))
        t_eur = np.zeros((H, W, T))
        eur_act = np.zeros((H, W, T))

        bi = np.zeros((H, W, T))
        bt = np.zeros((H, W, T))
        rend = np.zeros((H, W, T))

        # -------------------- Initial conditions at t=0 --------------------
        # Copy initial days after sowing
        dds = np.copy(self.dds0)

        # ----------- Canopy cover at t=0
        # Estimation of initial cover at t=0, based on dds0
        cover_t = self._get_init_cover_t0(dds, crop_mask, cp, scheme="fill_ones")
        cover[:, :, 0] = cover_t

        # Distribute initial water among layers:
        # layer1 = provided water0 (capped at aut), others start at 0.5*aut (capped)
        aut = self.aut * crop_mask
        # ponemos la densidad de agua inicial en la capa más superficial tal cual
        # porque es donde se mide el agua inicial, y estimamos la mitad para las capas más profundas
        # ver agromodel_model_plantgrowth_v27.m > lineas 90 a 93
        au[:, :, 0, 0] = np.clip(self.water0, 0, aut)
        for ell in range(1, L):
            au[:, :, ell, 0] = np.clip(0.5 * aut, 0, aut)

        # Helpers for ET0 and soil evaporation decay
        et0 = weather.et0
        # if ET0 not provided, compute (not here) Hargreaves (much closer to the MATLAB intent)
        if et0 is None:
            # need Tmax/Tmin; if you only have Tmean, keep ET0 small to avoid blow-up
            raise ValueError("Provide ET0 (Hargreaves/Penman-Monteith) instead of the crude proxy.")
        
        et0 = np.maximum(et0, 0.0)

        # DD90 es el contador de días con sequía - son los días seguidos con agua útil menor a 90% (a generalizar)
        DD90 = np.zeros((H, W))

        # Initialize roots based on dds0
        root[:, :, 0] = self._initial_root_lengths(dds, crop_mask, cp)
        # Roots are being initialized to 0 inspite of having dds > 0 at t=0 (!)
        # This is following the original MATLAB code, but seems odd. See
        # agromodel_model_plantgrowth_v27.m > line 86 & 197.
        # Maybe we should estimate an initial_root_lengths (?)

        par = weather.par.astype(float)
        temp = weather.temp.astype(float)
        prec = weather.precip.astype(float)

        layer_threshold = cp.layer_threshold_mm

        # ============================= Time loop ============================
        for t in range(T):
            # Update (current) Days After Sowing
            if t > 0:
                dds = dds + 1.0 * crop_mask

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
                aut * accessible_mult
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
                * crop_mask
            )

            # This is the correction for RUE due to water stress (RUE_{Act} = RUE_{Pot} * T°EUR * CEHR)
            ceh_r[:, :, t] = (
                self.stress_sigmoid(p_au_now, cp.au_up_r, cp.au_down_r, cp.c_forma_r)
                * crop_mask
            )

            # Stress correction for HI/IC - TO BE CHECKED (not in doc) (?)
            ceh_pc[:, :, t] = (
                self.stress_sigmoid(p_au_now, cp.au_up_pc, cp.au_down_pc, cp.c_forma_pc)
                * crop_mask
            )

            # ---------------- Thermal stress trapezoid for RUE --------------
            Ti = temp[t]
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
            t_eur[:, :, t] = th * crop_mask

            # Actual RUE:
            # the Potential RUE (a crop-dependent constant eur_pot),
            # times the correction for water stress (ceh_r),
            # times the correction for temperature stress (t_eur).
            eur_act[:, :, t] = cp.eur_pot * ceh_r[:, :, t] * t_eur[:, :, t]

            # -------------------- Canopy cover dynamics --------------------

            # Canopy cover at previous time step
            if t > 0:
                ct_old = cover[:, :, t - 1]
            else:
                ct_old = cover_t

            # We initialize the array for the current ct
            ct_i = np.ones_like(ct_old)

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
                ct_old[grow_phase]
                + (cp.alpha1 * ceh[:, :, t][grow_phase]) * ct_i[grow_phase]
            )

            # ----------- Stay Phase (Max CT Phase)
            stay_phase = (dds >= cp.dds_max) & (dds < cp.dds_sen)
            ct_i[stay_phase] = ct_old[stay_phase]

            # ----------- Senescence
            # Max canopy cover up to this point. (It's not the last one if
            # we're in the senescence phase for some time now.)
            ct_max = (
                np.maximum.accumulate(cover[:, :, : t + 1], axis=2).max(axis=2)
                if t > 0
                else ct_old
            )

            # We calculate the slope for the decrease in cover
            beta1 = (ct_max - cp.c_fin) / max(cp.dds_fin - cp.dds_sen, 1e-6)

            # We select pixels in the senescence phase via the dds mask
            sen_phase = dds >= cp.dds_sen
            # We calculate the new canopy cover, and we take either that, or the stationary canopy cover c_fin
            ct_i[sen_phase] = np.maximum(
                ct_old[sen_phase]
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
            cover[:, :, t] = np.clip(ct_i * crop_mask, 0.0, cp.c_max)

            # -------------- Transpiration and soil evaporation --------------

            # ----------- Potential evapotranspiration (ET0)
            et0_t = float(et0[t])

            # ----------- Transpiration
            transp_t = ceh_r[:, :, t] * (cover[:, :, t] * cp.KC) * et0_t
            # where: `ceh_r` is the RUE correction for water stress (0 to 1,
            # where 1 indicates no stress), `cover` is the canopy cover
            # fraction, and `cp.KC` is the crop transpiration coefficient.
            transp[:, :, t] = transp_t

            # ----------- Effective precipitation
            # (spatially uniform forcing, applied to all pixels in crop)
            ppef_t = effective_precipitation(prec[t])
            ppef[:, :, t] = ppef_t * crop_mask

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
            eva[:, :, t] = eva_t * crop_mask

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
                perc = np.maximum(new_k - aut, 0.0)

                # store (remove percolation) and clip [0, aut]
                au[:, :, k, t] = np.clip(new_k - perc, 0.0, aut)

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
            # As the per-layer capacity aut is soil_depth_mm * (cc - pmp),
            # and all layers have the same thickness/depth soil_depth_mm,
            # we can simply multiply by the number of accessible layers
            # to get the total depth soil_depth_mm * accessible_mult, thus
            # obtaining the capacity accessible to plants as
            # cap_accessible = accessible_mult * soil_depth_mm * (cc - pmp),
            # i.e., cap_accessible = aut * accessible_mult
            cap_accessible = aut * accessible_mult

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
            bi[:, :, t] = np.maximum(0.0, cover[:, :, t] * par[t] * eur_act[:, :, t])
            # Cumulative biomass
            bt[:, :, t] = bi[:, :, t] if t == 0 else bt[:, :, t - 1] + bi[:, :, t]
            # Yield
            rend[:, :, t] = bt[:, :, t] * ici * ceh_r[:, :, t] * cp.harvest_index

        # Package Results
        days = np.arange(T)
        return Results(
            dates=days,
            temp=temp,
            par=par,
            precip=prec,
            et0=et0,
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
            yield_=rend,
        )
