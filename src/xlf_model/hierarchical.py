"""Phase 3 (b) -- hierarchical stack: a JOINT likelihood over a galaxy survey.

The single-galaxy inference (``inference.py``) fits one galaxy's detected HMXB
luminosities for ``theta = (log10 xi_eff, gamma)``, where ``xi_eff = xi * SFR``
folds the (unknown, per-galaxy) star-formation rate into an effective
normalisation.  That is the right move when SFR is a nuisance you do not want to
assume.

This module asks the complementary survey question: given **N_gal galaxies that
share ONE global XLF** -- the same per-SFR normalisation ``xi`` and the same
slope ``gamma`` -- but each with its **own (SFR, distance, detector exposure)**,
what can the *stack* tell us that the single best galaxy cannot?  Most galaxies
in a shallow all-sky survey (eROSITA) land in the few-detections regime where one
galaxy barely constrains ``gamma`` at all; the joint fit pools their faint
information.

The hierarchy (deliberately simple)
-----------------------------------
A genuine hierarchical model would give each galaxy its own latent parameters
drawn from a population prior.  Here the hierarchy is **degenerate by design**:
there is exactly ONE shared global ``theta = (log10 xi, gamma)`` and **no
per-galaxy scatter** (no per-galaxy normalisation jitter, no slope spread).  Each
galaxy differs only through KNOWN, fixed quantities -- its SFR, distance and
detector -- which are *not* fitted.  Per-galaxy scatter (a log-normal spread on
``xi`` around the global value) is the documented optional extension, not the
core experiment, and is intentionally omitted.

The joint likelihood
--------------------
Because the galaxies are independent given the shared ``theta``, the joint
log-likelihood is just the **sum** of the per-galaxy unbinned Poisson-process
log-likelihoods (``inference.PoissonProcessLikelihood``):

    ln L_joint(theta) = sum_g  ln L_g(theta)
                      = sum_g [ sum_i ln lambda_g(L_{g,i} | theta) - Lambda_g(theta) ].

The shared parameter is the **per-SFR** normalisation ``log10 xi`` (NOT the
effective ``log10 xi_eff``): each galaxy ``g`` folds in its OWN ``SFR_g``, so its
effective normalisation is ``xi * SFR_g`` and the per-galaxy XLF used inside its
``PoissonProcessLikelihood`` is built with that galaxy's SFR.  This is the only
thing that differs from the single-galaxy builder, and it is what makes the SFRs
*known covariates* rather than nuisances: with the SFRs supplied, every galaxy
constrains the SAME ``xi``, so they stack coherently.

This module imports and reuses the single-galaxy machinery wholesale
(``PoissonProcessLikelihood``, ``ObservationModel``, ``LogLGrid``,
``Parameter``, ``HMXBXLF``) -- it adds only the per-galaxy SFR-folding builder
and the summation wrapper, duplicating none of the likelihood maths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .forward import DetectorPreset
from .inference import (
    LogLGrid,
    ObservationModel,
    Parameter,
    PoissonProcessLikelihood,
)
from .xlf import HMXBXLF

__all__ = [
    "GalaxySpec",
    "make_hmxb_xlf_builder_with_sfr",
    "single_galaxy_likelihood",
    "JointHMXBLikelihood",
    "build_joint_likelihood",
]


# ---------------------------------------------------------------------------
# Per-galaxy fixed covariates (SFR, distance, detector, absorption) + its data
# ---------------------------------------------------------------------------
@dataclass
class GalaxySpec:
    """One galaxy in the survey: its detected luminosities + KNOWN covariates.

    Attributes
    ----------
    L_obs : ndarray
        Detected (observed) luminosities to fit for this galaxy, erg/s.
    SFR : float
        Star-formation rate (Msun/yr) -- a KNOWN covariate, not fitted.  Folds
        into this galaxy's effective normalisation ``xi * SFR``.
    distance_Mpc : float
        Galaxy distance (Mpc).
    detector : DetectorPreset
        The detector preset this galaxy was observed with.
    absorption_flux_factor : float
        Multiplicative absorption factor (default 1.0).
    name : str
        Optional label for accounting/plots.
    """

    L_obs: np.ndarray
    SFR: float
    distance_Mpc: float
    detector: DetectorPreset
    absorption_flux_factor: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        self.L_obs = np.asarray(self.L_obs, dtype=float)

    @property
    def n_det(self) -> int:
        return int(self.L_obs.size)


# ---------------------------------------------------------------------------
# XLF builder with the galaxy's SFR folded in (the ONLY new likelihood piece)
# ---------------------------------------------------------------------------
def make_hmxb_xlf_builder_with_sfr(
    *,
    SFR: float,
    L_cut: float,
    L_min: float,
    L_ref: float,
    band: str = "0.5-8 keV",
):
    """Return ``make_xlf(theta)`` with ``theta = (log10 xi, gamma)`` for one SFR.

    The shared global parameter is the **per-SFR** log-normalisation ``log10 xi``
    (so ``xi`` is in the same units as the Mineo+12 ``xi``, NOT the effective
    ``xi*SFR``).  This galaxy's XLF carries its OWN ``SFR``, so its effective
    normalisation is ``xi * SFR`` -- exactly how the forward model built it.

    This mirrors :func:`inference.make_hmxb_xlf_builder` with ``param_kind =
    "log_xi"`` EXCEPT that ``SFR`` is the galaxy's real SFR (folded in) rather
    than 1.0, and ``theta[0]`` is therefore the per-SFR ``log10 xi`` shared
    across the whole survey.
    """

    def make_xlf(theta: Sequence[float]) -> HMXBXLF:
        log_xi, gamma = float(theta[0]), float(theta[1])
        xi = 10.0**log_xi
        return HMXBXLF(
            xi=xi,
            gamma=gamma,
            L_cut=L_cut,
            L_min=L_min,
            L_ref=L_ref,
            SFR=float(SFR),  # this galaxy's known SFR (the covariate)
            band=band,
        )

    return make_xlf


def single_galaxy_likelihood(
    gal: GalaxySpec,
    parameters: list[Parameter],
    *,
    L_cut: float,
    L_min: float,
    L_ref: float,
    band: str = "0.5-8 keV",
    selection_aware: bool = True,
    grid: LogLGrid | None = None,
) -> PoissonProcessLikelihood:
    """Build the per-galaxy :class:`PoissonProcessLikelihood` for a survey member.

    Uses the shared global parametrisation ``theta = (log10 xi, gamma)`` with the
    galaxy's own SFR folded into the XLF normalisation.  ``parameters`` (the
    priors) are shared across the survey, so the same list object is handed to
    every galaxy.
    """
    obs = ObservationModel(
        distance_Mpc=float(gal.distance_Mpc),
        detector=gal.detector,
        absorption_flux_factor=float(gal.absorption_flux_factor),
        selection_aware=selection_aware,
    )
    make_xlf = make_hmxb_xlf_builder_with_sfr(
        SFR=float(gal.SFR), L_cut=L_cut, L_min=L_min, L_ref=L_ref, band=band
    )
    return PoissonProcessLikelihood(
        L_obs=gal.L_obs,
        make_xlf=make_xlf,
        obs=obs,
        grid=grid or LogLGrid(),
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# The JOINT likelihood: sum of the per-galaxy log-likelihoods
# ---------------------------------------------------------------------------
@dataclass
class JointHMXBLikelihood:
    """Joint unbinned Poisson-process likelihood over a survey of galaxies.

    Holds a list of per-galaxy :class:`PoissonProcessLikelihood` objects that all
    share the SAME ``parameters`` (priors) and the SAME ``theta = (log10 xi,
    gamma)``.  The joint log-likelihood is the sum of the members'
    log-likelihoods (independence given the shared global parameters).

    The class exposes the SAME interface a single ``PoissonProcessLikelihood``
    does -- ``log_likelihood``, ``prior_transform``, ``log_prior``,
    ``log_posterior``, ``param_names``, ``ndim`` -- so it drops straight into
    :func:`inference.fit_xlf` with no sampler changes.

    Notes
    -----
    If ANY galaxy assigns ``-inf`` to a parameter vector (e.g. a detected source
    the model cannot produce, or a non-positive ``Lambda``), the joint
    log-likelihood is ``-inf``: a global ``theta`` that is impossible for one
    galaxy is impossible for the survey.  This is the correct Poisson-process
    behaviour and matches the single-galaxy convention.
    """

    members: list[PoissonProcessLikelihood]
    parameters: list[Parameter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("JointHMXBLikelihood needs at least one galaxy")
        if not self.parameters:
            self.parameters = self.members[0].parameters
        # every member must share the identical parameter set (same priors/dim)
        names = self.param_names
        for m in self.members:
            if [p.name for p in m.parameters] != names:
                raise ValueError(
                    "all survey galaxies must share the same parameters/priors"
                )

    # ----- the joint log-likelihood -----
    def log_likelihood(self, theta: Sequence[float]) -> float:
        """``ln L_joint(theta) = sum_g ln L_g(theta)`` (independent galaxies)."""
        total = 0.0
        for m in self.members:
            ll = m.log_likelihood(theta)
            if not np.isfinite(ll):
                return -np.inf
            total += ll
        return float(total)

    def per_galaxy_log_likelihood(self, theta: Sequence[float]) -> np.ndarray:
        """Vector of each galaxy's ``ln L_g(theta)`` (diagnostic / hand-check)."""
        return np.array(
            [m.log_likelihood(theta) for m in self.members], dtype=float
        )

    # ----- prior / posterior (shared parameters; delegate to the first member)
    @property
    def ndim(self) -> int:
        return len(self.parameters)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.array(
            [p.from_unit_cube(ui) for p, ui in zip(self.parameters, u)],
            dtype=float,
        )

    def log_prior(self, theta: Sequence[float]) -> float:
        for p, x in zip(self.parameters, theta):
            if not p.in_bounds(float(x)):
                return -np.inf
        return 0.0

    def log_posterior(self, theta: Sequence[float]) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # ----- survey accounting -----
    @property
    def n_galaxies(self) -> int:
        return len(self.members)

    @property
    def total_n_det(self) -> int:
        return int(sum(m.L_obs.size for m in self.members))


# ---------------------------------------------------------------------------
# One-call constructor from a list of GalaxySpecs + shared priors
# ---------------------------------------------------------------------------
def build_joint_likelihood(
    galaxies: Sequence[GalaxySpec],
    parameters: list[Parameter],
    *,
    L_cut: float,
    L_min: float,
    L_ref: float,
    band: str = "0.5-8 keV",
    selection_aware: bool = True,
    grid: LogLGrid | None = None,
) -> JointHMXBLikelihood:
    """Assemble a :class:`JointHMXBLikelihood` from a survey of galaxies.

    Each galaxy gets its own :class:`PoissonProcessLikelihood` (its SFR folded
    into the XLF normalisation, its distance/detector into ``P_det``), all
    sharing the same ``parameters`` (priors) and the global ``theta = (log10 xi,
    gamma)``.

    Parameters
    ----------
    galaxies : sequence of GalaxySpec
        The survey members (each with its detected luminosities + covariates).
    parameters : list[Parameter]
        Shared priors for ``(log10 xi, gamma)`` (same object reused per galaxy).
    L_cut, L_min, L_ref : float
        XLF support and reference luminosity (shared across the survey).
    band : str
        Energy-band bookkeeping label.
    selection_aware : bool
        Selection-aware (completeness ramp) vs naive (hard cut) -- applied to
        every galaxy identically.
    grid : LogLGrid, optional
        Integration grid for each galaxy's ``Lambda``.
    """
    if not galaxies:
        raise ValueError("need at least one galaxy")
    grid = grid or LogLGrid()
    members = [
        single_galaxy_likelihood(
            g, parameters, L_cut=L_cut, L_min=L_min, L_ref=L_ref, band=band,
            selection_aware=selection_aware, grid=grid,
        )
        for g in galaxies
    ]
    return JointHMXBLikelihood(members=members, parameters=parameters)
