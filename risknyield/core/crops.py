"""
Crop parameter presets and dataclass container.

This module provides a single, concrete dataclass :class:`CropParams` that
encapsulates crop-specific parameters used by the crop growth model. The class
is **frozen** (immutable) and uses **slots** for memory efficiency. It defaults
to a maize parameterization and includes convenience constructors to load other
presets (e.g., soybean).

The parameters cover phenology and canopy cover trajectory, water-stress
response thresholds and shapes, root growth, radiation-use efficiency (RUE),
thermal stress (trapezoid), and harvest-index/partitioning controls.

Classes
-------
CropParams
    Immutable container for crop parameters. Defaults to maize. Provides
    :meth:`CropParams.maize`, :meth:`CropParams.soy`, and
    :meth:`CropParams.from_preset` for preset selection.

Functions
---------
None. All factory helpers are classmethods on :class:`CropParams`.

Notes
-----
- **Scope**: This module defines *crop* parameters only. Soil and hydrology
  parameters (e.g., field capacity ``cc``, wilting point ``pmp``,
  soil depth, Curve Number ``CN``, runoff/soil-evaporation coefficients) belong
  to soil/hydrology components and are intentionally excluded here.
- **Defaults**: Instantiating :class:`CropParams` without arguments yields a
  maize configuration. Use :meth:`CropParams.soy` or
  :meth:`CropParams.from_preset("soy")` for soybean.
- **Derived parameters**: If ``alpha1`` (early linear cover-growth slope) is
  not provided, it is computed as ::

      alpha1 = (c_max - c_in) / max(dds_max - dds_in, 1e-12)

- **Validation**: The constructor checks basic consistency:
  ``0 ≤ c_in ≤ c_max ≤ 1``, ``0 ≤ c_fin ≤ 1``,
  ``dds_in ≤ dds_max ≤ dds_sen ≤ dds_fin``,
  all water-stress thresholds in ``[0, 1]``,
  and ordered thermal breakpoints ``tbr ≤ tor1 ≤ tor2 ≤ tcr``.

Examples
--------
Create maize parameters (defaults):

>>> from risknyield.core.crops import CropParams
>>> cp = CropParams.maize()      # equivalent to CropParams()

Create soybean parameters from the built-in preset:

>>> cp_soy = CropParams.soy()

Override selected fields (immutability enforced after construction):

>>> cp_custom = CropParams.maize()
>>> cp_custom = CropParams(
...     crop_name="maize",
...     dds_in=7, dds_max=50, dds_sen=90, dds_fin=125,
...     c_in=cp_custom.c_in, c_fin=cp_custom.c_fin, c_max=0.92,
...     alpha1=None,  # will be recomputed from the new dds/c values
...     au_up=cp_custom.au_up, au_down=cp_custom.au_down,
...     c_forma=cp_custom.c_forma, au_up_r=cp_custom.au_up_r,
...     au_down_r=cp_custom.au_down_r, c_forma_r=cp_custom.c_forma_r,
...     au_up_pc=cp_custom.au_up_pc, au_down_pc=cp_custom.au_down_pc,
...     c_forma_pc=cp_custom.c_forma_pc,
...     root_growth_rate=cp_custom.root_growth_rate,
...     root_max_mm=cp_custom.root_max_mm, eur_pot=cp_custom.eur_pot,
...     tbr=cp_custom.tbr, tor1=cp_custom.tor1, tor2=cp_custom.tor2,
...     tcr=cp_custom.tcr, df=cp_custom.df, ic_in=cp_custom.ic_in,
...     ic_pot_t=cp_custom.ic_pot_t, Y=cp_custom.Y, KC=cp_custom.KC,
...     harvest_index=cp_custom.harvest_index,
... )

Use with the model:

>>> from risknyield.core.data_containers import Soil, Weather
>>> from risknyield.core.main import CropModel
>>> soil, weather = ...  # constructed elsewhere
>>> params = CropParams.soy()
>>> model = CropModel(soil=soil, weather=weather, params=params)
>>> results = model.evolve()

See Also
--------
risknyield.core.data_containers : Soil/Weather/Results containers.
risknyield.core.main.CropModel : Execution engine consuming CropParams.
risknyield.library.hydrology : Hydrology kernels (effective precipitation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CropParams:
    """
    Concrete crop parameter set (defaults to maize).

    Parameters
    ----------
    crop_name : str, default="maize"
        Human-readable crop identifier.
    dds_in, dds_max, dds_sen, dds_fin : int
        Phenology breakpoints in days after sowing (emergence, max cover,
        senescence onset, end of season).
    c_in : float
        Cover fractions in [0, 1] at emergence.
    c_max : float
        Cover fractions in [0, 1] at maximum.
    c_fin : float
        Cover fractions in [0, 1] at end of season.
    alpha1 : float or None, default=None
        Early linear growth slope for cover between dds_in and dds_max.
        If None, computed as (c_max - c_in)/(dds_max - dds_in).
    au_up : float
        Upper water-stress threshold for canopy (CEH) in [0, 1].
    au_down : float
        Lower water-stress threshold for canopy (CEH) in [0, 1].
    au_up_r : float
        Upper water-stress threshold for RUE in [0, 1].
    au_down_r : float
        Lower water-stress threshold for RUE in [0, 1].
    au_up_pc : float
        Upper water-stress threshold for harvest index / partition
        coefficient in [0, 1].
    au_down_pc : float
        Lower water-stress threshold for harvest index / partition
        coefficient in [0, 1].
    c_forma : float
        Shape parameter (dimensionless) for the canopy cover stress response.
    c_forma_r : float
        Shape parameter (dimensionless) for the RUE stress response.
    c_forma_pc : float
        Shape parameter (dimensionless) for the harvest index / partition
        coefficient stress response.
    root_growth_rate : float
        Root growth rate [mm/day].
    root_max_mm : float
        Maximum rooting depth [mm].
    eur_pot : float
        Potential RUE [g DM / MJ PAR] (DM refers to dry matter and PAR to
        photosynthetically active radiation).
    tbr : float
        Temperature below which we have crushing cold stress [°C].
        Must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr.
    tor1 : float
        Lower optimal temperature for growth [°C].
        Must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr.
    tor2 : float
        Upper optimal temperature for growth [°C].
        Must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr.
    tcr : float
        Temperature above which we have crushing heat stress [°C].
        Must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr.
    df, ic_in, ic_pot_t, Y : (int, float, float, float)
        Harvest index / ICI logistic parameters.
    KC : float, default=0.94
        Transpiration coefficient.
    harvest_index : float, default=1.0
        Multiplier applied to convert biomass to yield (keep 1.0 if ICI
        already used).

    Notes
    -----
    - This class is frozen for immutability.
    - It uses slots for memory efficiency.
    - Defaults reproduce the **maize** configuration; use `from_preset("soy")`
      or `soy()` to instantiate soybean parameters.
    - Soil/hydrology parameters like `cc`, `pmp`, `prof_suelo`, `CN`, `S`,
      `c_esuelo`, `esuelo_pot` belong in `Soil`/hydrology modules, not here.

    Examples
    --------
    >>> cp = CropParams.maize()          # or simply CropParams()
    >>> cp = CropParams.soy()            # or CropParams.from_preset("soy")
    """

    # --- Species ---
    crop_name: str = "maize"

    # --- Phenology / cover ---
    dds_in: int = 7
    dds_max: int = 47
    dds_sen: int = 87
    dds_fin: int = 120

    c_in: float = 0.039
    c_fin: float = 0.01
    c_max: float = 0.89

    alpha1: float | None = None  # computed if None

    # --- Water stress thresholds (fractions of available water) ---
    # canopy cover (CEH)
    au_up: float = 0.72
    au_down: float = 0.20
    c_forma: float = 4.9

    # RUE
    au_up_r: float = 0.69
    au_down_r: float = 0.0
    c_forma_r: float = 6.0

    # harvest index / partition coefficient
    au_up_pc: float = 0.60
    au_down_pc: float = 0.15
    c_forma_pc: float = 1.3

    # --- Root dynamics ---
    root_growth_rate: float = 30.0  # mm/day
    root_max_mm: float = 2000.0  # mm

    # --- Radiation use efficiency ---
    eur_pot: float = 3.65  # g/MJ

    # --- Thermal stress trapezoid (°C) ---
    tbr: float = 8.0
    tor1: float = 29.0
    tor2: float = 39.0
    tcr: float = 45.0

    # --- Harvest index / ICI (logistic) ---
    df: int = 58
    ic_in: float = 0.001
    ic_pot_t: float = 0.48
    Y: float = 0.19

    # --- Flux scaling ---
    KC: float = 0.94
    harvest_index: float = 1.0

    # -------------------------
    # Post-init: compute & validate
    # -------------------------
    def __post_init__(self):
        r"""
        Compute derived parameters and run validations.

        This method computes ``alpha1`` when it is not provided and performs a
        set of sanity checks on cover fractions, phenology ordering,
        water-stress thresholds, thermal breakpoints, root parameters, and
        scaling coefficients.

        Parameters
        ----------
        self : CropParams
            The instance being initialized.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any validation fails. Possible messages include:
            - "Cover fractions must satisfy 0 ≤ c_in ≤ c_max ≤ 1 and
              0 ≤ c_fin ≤ 1."
            - "Phenology must satisfy dds_in ≤ dds_max ≤ dds_sen ≤ dds_fin."
            - "All water-stress thresholds must be in [0, 1]."
            - "Thermal trapezoid must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr."
            - "Root parameters must have root_growth_rate ≥ 0 and
              root_max_mm > 0."
            - "eur_pot must be positive."
            - "KC must be positive."
            - "harvest_index must be in [0, 1]."

        Notes
        -----
        If ``alpha1`` is ``None``, it is computed as

        .. math::

        \\alpha_1 = \\frac{c_{\\max} - c_{\\in}}{\\max(\\mathrm{dds}_{\\max}
        - \\mathrm{dds}_{\\in}, 10^{-12})}.

        A denominator guard is applied to avoid division by zero when
        ``dds_max == dds_in``. Because :class:`CropParams` is frozen,
        assignment of the derived ``alpha1`` uses ``object.__setattr__``.
        """
        # Compute alpha1 if not provided
        if self.alpha1 is None:
            denom = max(self.dds_max - self.dds_in, 1e-12)
            object.__setattr__(
                self, "alpha1", (self.c_max - self.c_in) / denom
            )

        # Basic validations
        if not (
            0.0 <= self.c_in <= self.c_max <= 1.0 and 0.0 <= self.c_fin <= 1.0
        ):
            raise ValueError(
                "Cover fractions must satisfy "
                "0 ≤ c_in ≤ c_max ≤ 1 and 0 ≤ c_fin ≤ 1."
            )
        if not (self.dds_in <= self.dds_max <= self.dds_sen <= self.dds_fin):
            raise ValueError(
                "Phenology must satisfy dds_in ≤ dds_max ≤ dds_sen ≤ dds_fin."
            )
        for v in (
            self.au_up,
            self.au_down,
            self.au_up_r,
            self.au_down_r,
            self.au_up_pc,
            self.au_down_pc,
        ):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    "All water-stress thresholds must be in [0, 1]."
                )
        if not (self.tbr <= self.tor1 <= self.tor2 <= self.tcr):
            raise ValueError(
                "Thermal trapezoid must satisfy tbr ≤ tor1 ≤ tor2 ≤ tcr."
            )
        if self.root_growth_rate < 0.0 or self.root_max_mm <= 0.0:
            raise ValueError(
                "Root parameters must have root_growth_rate ≥ 0 and "
                "root_max_mm > 0."
            )
        if self.eur_pot <= 0.0:
            raise ValueError("eur_pot must be positive.")
        if self.KC <= 0.0:
            raise ValueError("KC must be positive.")
        if not (0.0 <= self.harvest_index <= 1.0):
            raise ValueError("harvest_index must be in [0, 1].")

    # -------------------------
    # Convenience constructors / presets
    # -------------------------
    @classmethod
    def maize(cls) -> "CropParams":
        """Return a `CropParams` instance with maize defaults."""
        return cls(crop_name="maize")

    @classmethod
    def soy(cls) -> "CropParams":
        """Return a `CropParams` instance with soybean parameters."""
        return cls.from_preset("soy")

    @classmethod
    def from_preset(cls, name: str) -> "CropParams":
        """
        Instantiate from a named preset.

        Parameters
        ----------
        name : {'maize', 'soy'}
            Preset identifier.

        Returns
        -------
        CropParams
            Parameter set for the given preset.

        Raises
        ------
        KeyError
            If `name` is not a known preset.
        """
        presets: Mapping[str, dict] = {
            # maize is the class default; listed for clarity
            "maize": dict(crop_name="maize"),
            # --- soybean (mapped from the MATLAB soy parameter file) ---
            "soy": dict(
                crop_name="soy",
                # phenology & cover
                dds_in=7,
                dds_max=60,
                dds_sen=120,
                dds_fin=137,  # agromodel_model_plantgrowth_v27.m > line 53
                c_in=0.039,
                c_fin=0.01,
                c_max=0.95,
                alpha1=None,  # computed from c_in/c_max/dds_in/dds_max
                # water stress thresholds & shapes
                au_up=0.65,
                au_down=0.15,
                c_forma=1.2,
                au_up_r=0.5,
                au_down_r=0.0,
                c_forma_r=3.0,
                au_up_pc=0.60,
                au_down_pc=0.15,
                c_forma_pc=1.3,
                # roots
                root_growth_rate=30.0,
                root_max_mm=2000.0,
                # RUE
                eur_pot=1.36,
                # thermal trapezoid (°C)
                tbr=10.0,
                tor1=20.0,
                tor2=30.0,
                tcr=40.0,
                # ICI / harvest index
                df=37,
                ic_in=0.001,
                ic_pot_t=0.48,
                Y=0.19,
                # transpiration coefficient
                KC=1.11,
                harvest_index=1.0,
            ),
        }
        try:
            return cls(**presets[name])
        except KeyError as e:
            raise KeyError(
                f"Unknown preset '{name}'. Known: {sorted(presets)}"
            ) from e
