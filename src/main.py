from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple
import numpy as np

Array = np.ndarray
CropName = Literal["soy", "maize"]


# -----------------------------
# Utility helpers
# -----------------------------


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


# -----------------------------
# Data containers
# -----------------------------


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
    root_max_mm: float = 2000.0  # WHERE DID THIS COME FROM??? (?)
    layer_threshold_mm: float = 500.0  # per-layer depth used for access rules
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


# -----------------------------
# Core model
# -----------------------------


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

    # -------------------------
    # Main evolution routine
    # -------------------------
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

        # Choose crop mask (bool 2D)
        crop_mask = self.mask_maize if crop == "maize" else self.mask_soy
        crop_mask = crop_mask.astype(float)

        # Allocate arrays
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

        # Initial conditions at t=0
        dds = np.copy(self.dds0)
        cover_t = np.where(
            dds <= cp.dds_in, 0.0, np.where(dds == cp.dds_in, cp.c_in, 1.0)
        )

        # Distribute initial water among layers:
        # layer1 = provided water0 (capped at aut), others start at 0.5*aut (capped)
        cover[:, :, 0] = np.clip(cover_t * crop_mask, 0.0, cp.c_max)
        # en lugar del clip podría poner
        assert cp.c_in <= cp.c_max <= 1.0
        aut = self.aut * crop_mask
        # ponemos la densidad de agua inicial en la capa más superficial tal cual
        # porque es donde se mide el agua inicial, y estimamos la mitad para las capas más profundas
        # ver agromodel_model_plantgrowth_v27.m > lineas 90 a 93
        au[:, :, 0, 0] = np.clip(self.water0, 0, aut)
        for ell in range(1, L):
            au[:, :, ell, 0] = np.clip(0.5 * aut, 0, aut)

        # Helpers for ET0 and soil evaporation decay
        et0 = weather.et0
        # If evotranspiration is not given, we estimate a refference value
        if et0 is None:
            et0 = cp.k_et0_T * weather.temp + cp.k_et0_PAR * weather.par
        et0 = np.maximum(et0, 0.0)

        # DD90 es el contador de días con sequía - son los días seguidos con agua útil menor a 90% (a generalizar)
        DD90 = np.zeros((H, W))
        root[:, :, 0] = 0.0
        par = weather.par.astype(float)
        temp = weather.temp.astype(float)
        prec = weather.precip.astype(float)

        layer_threshold = cp.layer_threshold_mm

        # ============================= Time loop ============================
        for t in range(T):
            # Update (current) Days After Sowing
            if t > 0:
                dds = dds + 1.0 * crop_mask

            if t == 0:
                root_prev = root[:, :, 0]
            else:
                root_prev = root[:, :, t - 1]

            # ------------------------ Water Dynamics ------------------------

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
            # (for CT, EUR, and HI/IC)

            # Water stress function (generic)
            def stress_sigmoid(pau, up, down, c):
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

            # Correction for the Canopy Cover CT due to water stress.
            # Remember that CT_i = CT_{i-1} \pm \alpha/\beta * CEH(t),
            # depending on the days after sowing. Cf. lines 490 and 507.
            ceh[:, :, t] = (
                stress_sigmoid(p_au_now, cp.au_up, cp.au_down, cp.c_forma) * crop_mask
            )

            # This is the correction for RUE due to water stress (RUE_{Act} = RUE_{Pot} * T°EUR * CEHR)
            ceh_r[:, :, t] = (
                stress_sigmoid(p_au_now, cp.au_up_r, cp.au_down_r, cp.c_forma_r)
                * crop_mask
            )

            # Stress for HI/IC - TO BE CHECKED (?)
            ceh_pc[:, :, t] = (
                stress_sigmoid(p_au_now, cp.au_up_pc, cp.au_down_pc, cp.c_forma_pc)
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
            # with DD90 decay when p_au<0.9 (as in script)
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
                * (
                    1.0 - cover[:, :, t][low]
                )  # fraction of area not covered by plant
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


            # ------------------
            # Water balance by layers
            # ------------------

            # ----------- Transpiration sharing
            # Split transpiration among accessible layers based on root depth
            tr_share = np.zeros_like(transp_t)  # H x W matrix storing the
            # transpiration share for each layer in pixel (i,j). We'll assume
            # equipartitioning of transpired water among layers.

            # layer categories as in MATLAB: 0-500, 500-1000, 1000-1500, >1500
            # ... generalized
            for k in range(L):
                # First, we define the layer boundaries
                lower = k * layer_threshold
                upper = (k + 1) * layer_threshold
                # Now we create a boolean mask that indicates if roots end at
                # layer k
                in_layer = (root_prev > lower) & (root_prev <= upper)
                # and we calculate the transpiration share for every layer in
                # pixel (i,j)
                if k == 0:
                    tr_share[in_layer] = transp_t[in_layer]
                else:
                    tr_share[in_layer] = transp_t[in_layer] / (k + 1)
            # if deeper than last threshold, distribute equally among L layers
            deeper = root_prev > (L - 1) * layer_threshold
            if np.any(deeper):
                tr_share[deeper] = transp_t[deeper] / L

            # Update layers sequentially (perc from k flows into k+1)
            perc_prev = np.zeros((H, W))

            for k in range(L):
                gain = perc_prev.copy()  # note that this is 0 for k=0
                # Layer 0: gains effective precipitation minus soil
                # evaporation and transp layer share
                if k == 0:
                    gain = gain + ppef[:, :, t] - eva[:, :, t]
                # Other layers: gain percolation from above, lose transp share
                loss = tr_share
                # 1. We update without constrains, temporarily allowing over
                # capacity
                au[:, :, k, t] = au[:, :, k, t] + gain - loss
                # 2. Percolation to next layer is the excess when in over
                # capacity
                perc = np.maximum(au[:, :, k, t] - aut, 0.0)
                # 3. And *NOW* we clip to correct overcapacity (no need to
                # substract perc, since it's the excess clipped out)
                au[:, :, k, t] = np.clip(au[:, :, k, t], 0.0, aut)
                # Update percolation for next layer
                perc_prev = perc

            # Root growth (mm)
            rg = cp.root_growth_rate * ceh_r[:, :, t]
            if t == 0:
                root[:, :, t] = np.minimum(cp.root_max_mm, root_prev + rg)
            else:
                root[:, :, t] = np.minimum(cp.root_max_mm, root[:, :, t - 1] + rg)

            # Recompute p_au after water balance for outputs (at end of day)
            sum_layers = au[:, :, 0, t]
            accessible_mult = 1.0
            for k in range(1, L):
                mask_k = root[:, :, t] > k * layer_threshold
                sum_layers = sum_layers + au[:, :, k, t] * mask_k
                accessible_mult = accessible_mult + mask_k
            cap_accessible = aut * accessible_mult
            with np.errstate(invalid="ignore", divide="ignore"):
                p_au[:, :, t] = np.clip(
                    sum_layers / np.maximum(cap_accessible, 1e-9), 0.0, 1.0
                )

            # Harvest index dynamics (simplified logistic with cp.df and cp.Y)
            ddf = dds - cp.df
            ic_pot = np.full((H, W), cp.ic_pot_t)
            # (Optional carry-over stress on ic_pot could be added here)
            ici = np.zeros((H, W))
            mask_flowering = ddf > 0
            ici[mask_flowering] = (cp.ic_in * ic_pot[mask_flowering]) / (
                cp.ic_in
                + (ic_pot[mask_flowering] - cp.ic_in)
                * np.exp(-cp.Y * ddf[mask_flowering])
            )

            # Daily biomass from PAR capture and RUE
            bi[:, :, t] = np.maximum(0.0, cover[:, :, t] * par[t] * eur_act[:, :, t])
            bt[:, :, t] = bi[:, :, t] if t == 0 else bt[:, :, t - 1] + bi[:, :, t]
            rend[:, :, t] = bt[:, :, t] * ici * ceh_r[:, :, t] * cp.harvest_index

        # Package results
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


# -----------------------------
# Minimal usage example (pseudo)
# -----------------------------
if __name__ == "__main__":
    H, W, T = 50, 60, 120
    lat = np.linspace(-32.9, -33.1, H)
    lon = np.linspace(-64.5, -64.2, W)
    water0 = np.full((H, W), 60.0)
    dds0 = np.zeros((H, W)) - 5  # start 5 days before emergence
    mask_maize = np.zeros((H, W), bool)
    mask_soy = np.zeros((H, W), bool)
    mask_maize[10:30, 10:40] = True
    mask_soy[30:48, 20:50] = True

    weather = Weather(
        temp=20 + 8 * np.sin(np.linspace(0, 2 * np.pi, T)),
        par=25 + 5 * np.sin(np.linspace(0, 2 * np.pi, T) + 0.5),
        precip=np.maximum(0, 10 * np.random.gamma(0.8, 1, size=T) - 4),
        et0=None,  # will use crude proxy
    )

    soil = Soil(
        lat, lon, water0, dds0, mask_maize, mask_soy, n_layers=4, soil_depth_mm=500
    )
    res_maize = soil.evolve("maize", weather)
    res_soy = soil.evolve("soy", weather)
    # Access final yield fields (t = T-1): res_maize.yield_[:, :, -1]
