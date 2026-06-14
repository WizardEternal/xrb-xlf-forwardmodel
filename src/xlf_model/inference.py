"""Phase 2 -- Bayesian recovery of XLF parameters (the inverse problem).

This module fits an X-ray luminosity function (XLF) to a list of *detected*
source luminosities using an **unbinned Poisson-process likelihood** in the
spirit of Marshall, Avni, Tananbaum & Zamorani 1983 (ApJ 269, 35; bibcode
``1983ApJ...269...35M``).

The likelihood
--------------
A detected catalogue is a realisation of an inhomogeneous Poisson point process
on luminosity, with intensity (the expected differential detected count)

    lambda(L | theta) = dN/dL(L | theta) * P_det(L)

where ``dN/dL`` is the XLF (parametrised by theta) and ``P_det(L)`` is the
probability that a source of intrinsic luminosity ``L`` is detected -- here the
forward model's smooth completeness ramp ``C(F(L))`` evaluated at the flux that
``L`` produces at the galaxy's distance (with the same absorption factor).  The
log-likelihood of a catalogue ``{L_i}`` is then the standard unbinned
Poisson-process form

    ln L(theta) = sum_i ln lambda(L_i | theta)  -  Lambda(theta),
    Lambda(theta) = integral lambda(L | theta) dL                 (expected
                                                                   detected count)

``Lambda(theta)`` is the *expected total number of detected sources* -- exactly
the quantity the forward model's funnel reports as ``n_detected`` in
expectation.  We evaluate it numerically on a fixed log-L grid (see
``LogLGrid``); the grid choice is documented there and its convergence is
checked in the test suite (``test_inference.py``).

The "naive" (deliberately-wrong) variant
-----------------------------------------
The headline experiment of the repo contrasts the selection-aware likelihood
above with a *naive* fitter that ignores selection physics: it imposes a HARD
flux cut at the nominal limit (``P_det = 1`` above the limit, ``0`` below) with
NO completeness ramp and NO awareness of Eddington bias.  Both variants share
one code path and one API; a flag (``selection_aware=False``) switches between
them.  The naive fit is expected to bias the recovered slope in the
faint/marginal regime -- that bias is the thing we measure.

Eddington-bias caveat
---------------------
The data handed to the fitter are the **observed** (Poisson-noise-scattered)
luminosities ``L_obs`` produced by ``forward.observe`` -- not the intrinsic
``L_true``.  The selection-aware likelihood here corrects for **completeness**
(the ``P_det(L)`` term) but it treats each observed luminosity as if it were the
intrinsic one: it does **not** deconvolve the measurement-noise scattering that
produces Eddington bias.  Near the detection threshold, where up-scattered
sources are preferentially detected, this leaves a residual bias even in the
selection-aware fit.  Full noise deconvolution (a per-source convolution of
``lambda`` with the count-noise kernel) is out of scope for this stage; the
recovery suite (Phase 3) quantifies the residual effect across the N_det grid.

API / extensibility
-------------------
The likelihood is built around two small protocols:

  * an XLF *builder* ``make_xlf(theta) -> xlf`` that turns a parameter vector
    into any object exposing ``dN_dL(L)`` and ``L_min``/``L_cut`` (so the
    HMXB single power law today, the LMXB broken power law tomorrow, drop in
    without touching the likelihood);
  * an :class:`ObservationModel` that owns the distance, detector preset and
    absorption, and therefore the ``P_det(L)`` map (selection-aware or naive).

The concrete HMXB case is wired up by :func:`hmxb_likelihood_from_config` /
:func:`make_hmxb_problem`; an LMXB equivalent can be added the same way.

Samplers
--------
:func:`fit_xlf` runs either UltraNest (``sampler="ultranest"``, primary;
ReactiveNestedSampler, returns posterior samples + ``logZ``) or emcee
(``sampler="emcee"``, fallback; affine-invariant ensemble, autocorr-based
thinning).  Both read identical priors from the config and return a uniform
:class:`FitResult`.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .forward import DetectorPreset, detector_from_config, luminosity_to_flux
from .xlf import HMXBXLF

__all__ = [
    "LogLGrid",
    "ObservationModel",
    "Parameter",
    "PoissonProcessLikelihood",
    "FitResult",
    "make_hmxb_xlf_builder",
    "hmxb_parameters_from_config",
    "make_hmxb_problem",
    "fit_xlf",
]


# ---------------------------------------------------------------------------
# Integration grid for Lambda(theta)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogLGrid:
    """Fixed logarithmic luminosity grid used to evaluate ``Lambda(theta)``.

    ``Lambda(theta) = integral dN/dL * P_det dL`` is computed by the trapezoid
    rule on a grid uniform in ``log10 L`` (the XLF and the completeness ramp are
    both smooth in log-L, so a log grid is the natural, efficient choice).

    Grid choice (documented; convergence checked in the tests)
    ----------------------------------------------------------
    * span: ``[L_min, L_cut]`` of the XLF being integrated, taken from the XLF
      object at evaluation time so the grid always covers the full support;
    * density: ``points_per_dex`` points per decade (default 200).  Over the
      ~5 decades of XLF support that is ~1000 points -- the test suite confirms
      that doubling the density changes ``ln L`` by < 1e-3, i.e. the trapezoid
      error on ``Lambda`` (and hence on ``ln L``) is already negligible at the
      default.

    A log grid + trapezoid on a power-law integrand is exact in the limit and
    converges quickly; we keep the integrand in *linear* ``dN/dL dL`` form
    (mapping ``dL = L ln10 d(log10 L)``) so the same ``dN_dL`` used in the
    forward model is reused verbatim.
    """

    points_per_dex: int = 200
    min_points: int = 64

    def grid(self, L_lo: float, L_hi: float) -> np.ndarray:
        """Return the log-spaced L grid spanning ``[L_lo, L_hi]`` (inclusive)."""
        L_lo = float(L_lo)
        L_hi = float(L_hi)
        if not (L_hi > L_lo > 0.0):
            raise ValueError("require 0 < L_lo < L_hi")
        ndex = np.log10(L_hi / L_lo)
        n = max(int(np.ceil(self.points_per_dex * ndex)) + 1, self.min_points)
        return np.logspace(np.log10(L_lo), np.log10(L_hi), n)

    def integrate(self, integrand: np.ndarray, L: np.ndarray) -> float:
        """Trapezoid integral of ``integrand`` (a function of L) over ``L``."""
        return float(np.trapezoid(integrand, L))


# ---------------------------------------------------------------------------
# Observation model: owns distance + detector + absorption -> P_det(L)
# ---------------------------------------------------------------------------
@dataclass
class ObservationModel:
    """Maps intrinsic luminosity to a detection probability ``P_det(L)``.

    This is the single point that re-uses the forward model's selection
    machinery: it converts ``L`` to the same observed flux the forward model
    uses (distance + multiplicative absorption) and applies the detector's
    completeness.  Two selection modes:

    * ``selection_aware=True`` (default): ``P_det(L) = C(F(L))`` -- the smooth
      error-function completeness ramp from ``forward.completeness_erf`` (exactly
      what the forward model draws detections from).
    * ``selection_aware=False`` (the naive fitter): a HARD step at the nominal
      flux limit -- ``P_det = 1`` for ``F(L) >= flux_limit_50`` else ``0``.
      No ramp, no Eddington-bias awareness.

    Both modes use the SAME detector preset, distance and absorption, so the
    selection-aware vs naive comparison differs *only* in the selection model.
    """

    distance_Mpc: float
    detector: DetectorPreset
    absorption_flux_factor: float = 1.0
    selection_aware: bool = True

    def flux(self, L) -> np.ndarray:
        """Observed flux a source of luminosity ``L`` produces (incl. absorption)."""
        f = luminosity_to_flux(L, self.distance_Mpc)
        return np.asarray(f, dtype=float) * float(self.absorption_flux_factor)

    def p_det(self, L) -> np.ndarray:
        """Detection probability ``P_det(L)`` (selection-aware or naive)."""
        flux = self.flux(L)
        if self.selection_aware:
            return self.detector.completeness(flux)
        # naive: hard flux cut at the nominal 50% limit, no ramp
        return np.where(flux >= self.detector.flux_limit_50, 1.0, 0.0).astype(float)


# ---------------------------------------------------------------------------
# Parameters & priors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    """One fitted parameter with a prior, used by both samplers.

    ``prior`` is ``"log-uniform"`` (sampled uniformly in log10 between ``lo`` and
    ``hi``, which are then interpreted as log10 bounds) or ``"uniform"`` (sampled
    uniformly in the value between ``lo`` and ``hi``).  The stored ``lo``/``hi``
    are always in the *sampling* space (i.e. log10 for log-uniform): this keeps
    the unit-cube transform and the emcee prior trivially consistent.

    ``name`` is the label shown in corner plots and summaries.
    """

    name: str
    lo: float
    hi: float
    prior: str = "uniform"

    def __post_init__(self) -> None:
        if self.prior not in ("uniform", "log-uniform"):
            raise ValueError(f"unknown prior {self.prior!r}")
        if not (self.hi > self.lo):
            raise ValueError(f"parameter {self.name}: require hi > lo")

    def from_unit_cube(self, u: float) -> float:
        """Map a unit-cube coordinate ``u in [0,1]`` to the sampling value."""
        return self.lo + u * (self.hi - self.lo)

    def in_bounds(self, x: float) -> bool:
        return self.lo <= x <= self.hi


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
# An XLF builder turns a parameter vector (in *sampling* space) into an XLF
# object exposing ``dN_dL(L)`` plus ``L_min``/``L_cut``.
XLFBuilder = Callable[[Sequence[float]], object]


@dataclass
class PoissonProcessLikelihood:
    """Unbinned Poisson-process (Marshall-style) likelihood for detected L.

    Parameters
    ----------
    L_obs : ndarray
        The observed (detected) luminosities to fit, erg/s.  These are the
        ``L_obs`` column from ``forward.observe`` (noise-scattered); see the
        module docstring's Eddington-bias caveat.
    make_xlf : callable
        ``make_xlf(theta) -> xlf`` mapping a parameter vector to an XLF object.
    obs : ObservationModel
        Supplies ``P_det(L)`` (selection-aware or naive).
    grid : LogLGrid
        Integration grid for ``Lambda(theta)``.
    parameters : list[Parameter]
        The fitted parameters (defines dimensionality, priors, labels).

    Notes
    -----
    A source with ``L_i`` outside the XLF support (``dN/dL = 0``) or with
    ``P_det(L_i) = 0`` contributes ``ln lambda = -inf``; this correctly assigns
    zero probability to a model that cannot produce a detected source there.
    For the naive (hard-cut) model that is exactly the intended behaviour --
    sources below the cut are deemed impossible -- so in practice the naive
    fitter is only ever handed sources above its own cut (the script filters the
    data to the cut for the naive run; see ``run_fit_single.py``).
    """

    L_obs: np.ndarray
    make_xlf: XLFBuilder
    obs: ObservationModel
    grid: LogLGrid
    parameters: list[Parameter]

    def __post_init__(self) -> None:
        self.L_obs = np.asarray(self.L_obs, dtype=float)

    # ----- pieces -----
    def expected_detected(self, theta: Sequence[float]) -> float:
        """``Lambda(theta)`` = expected detected count = int dN/dL * P_det dL."""
        xlf = self.make_xlf(theta)
        L = self.grid.grid(xlf.L_min, xlf.L_cut)
        lam = np.asarray(xlf.dN_dL(L), dtype=float) * self.obs.p_det(L)
        return self.grid.integrate(lam, L)

    def log_intensity(self, theta: Sequence[float]) -> np.ndarray:
        """``ln lambda(L_i | theta)`` for each detected source (vectorized)."""
        xlf = self.make_xlf(theta)
        dN = np.asarray(xlf.dN_dL(self.L_obs), dtype=float)
        pdet = self.obs.p_det(self.L_obs)
        lam = dN * pdet
        out = np.full(lam.shape, -np.inf)
        good = lam > 0.0
        out[good] = np.log(lam[good])
        return out

    # ----- the log-likelihood -----
    def log_likelihood(self, theta: Sequence[float]) -> float:
        """``ln L(theta) = sum_i ln lambda(L_i) - Lambda(theta)``."""
        Lambda = self.expected_detected(theta)
        if not np.isfinite(Lambda) or Lambda <= 0.0:
            return -np.inf
        ln_lam = self.log_intensity(theta)
        if not np.all(np.isfinite(ln_lam)):
            # a detected source the model assigns zero rate to -> impossible model
            return -np.inf
        return float(np.sum(ln_lam) - Lambda)

    # ----- prior helpers (shared by both samplers) -----
    @property
    def ndim(self) -> int:
        return len(self.parameters)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Unit-cube -> sampling-space (UltraNest prior_transform)."""
        u = np.asarray(u, dtype=float)
        return np.array(
            [p.from_unit_cube(ui) for p, ui in zip(self.parameters, u)], dtype=float
        )

    def log_prior(self, theta: Sequence[float]) -> float:
        """Flat (in sampling space) log-prior with hard bounds (for emcee)."""
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


# ---------------------------------------------------------------------------
# HMXB single-power-law problem (the primary use case)
# ---------------------------------------------------------------------------
def make_hmxb_xlf_builder(
    *,
    L_cut: float,
    L_min: float,
    L_ref: float,
    SFR: float = 1.0,
    band: str = "0.5-8 keV",
    param_kind: str = "log_xi",
) -> XLFBuilder:
    """Return a ``make_xlf(theta)`` for the HMXB single power law.

    The cutoff ``L_cut`` is **fixed** (single power laws do not constrain the
    bright-end cutoff from a handful of faint-galaxy detections; it sits well
    above the data, so fixing it is both standard and harmless -- the
    forward-model RESULTS already note N(>1e38) is insensitive to it).  The two
    free parameters are the normalisation and the slope.

    ``param_kind`` selects the normalisation parametrisation of ``theta``:

    * ``"log_xi"`` (default): ``theta = (log10 xi_eff, gamma)`` where
      ``xi_eff = xi * SFR`` is the effective normalisation actually multiplying
      the shape (so a fit at fixed/unknown SFR recovers the product).  ``SFR`` is
      folded in as 1.0 and the recovered ``log10 xi_eff`` equals
      ``log10(xi*SFR)``.
    * ``"log_norm"``: identical maths, alias kept for readability when the user
      thinks of it as "log normalisation" rather than "log xi".

    Both map ``theta[0]`` to the HMXB ``xi`` (with ``SFR`` carried separately as
    1.0 so ``xi`` IS the effective normalisation).
    """
    if param_kind not in ("log_xi", "log_norm"):
        raise ValueError(f"unknown param_kind {param_kind!r}")

    def make_xlf(theta: Sequence[float]) -> HMXBXLF:
        log_norm, gamma = float(theta[0]), float(theta[1])
        xi_eff = 10.0**log_norm
        return HMXBXLF(
            xi=xi_eff,
            gamma=gamma,
            L_cut=L_cut,
            L_min=L_min,
            L_ref=L_ref,
            SFR=1.0,  # xi already folds in SFR (effective normalisation)
            band=band,
        )

    return make_xlf


def hmxb_parameters_from_config(cfg: dict) -> list[Parameter]:
    """Build the HMXB fit parameters (priors) from a parsed config.

    Reads ``cfg['inference']['priors']`` if present; otherwise falls back to
    sensible defaults (log10 xi_eff in [-2, 3], gamma in [0.5, 3.5]).
    """
    inf = cfg.get("inference", {})
    pri = inf.get("priors", {})
    lx = pri.get("log_xi_eff", {"lo": -2.0, "hi": 3.0, "prior": "uniform"})
    ga = pri.get("gamma", {"lo": 0.5, "hi": 3.5, "prior": "uniform"})
    return [
        Parameter("log10_xi_eff", float(lx["lo"]), float(lx["hi"]),
                  lx.get("prior", "uniform")),
        Parameter("gamma", float(ga["lo"]), float(ga["hi"]),
                  ga.get("prior", "uniform")),
    ]


def make_hmxb_problem(
    L_obs,
    cfg: dict,
    *,
    distance_Mpc: float,
    detector_name: str | None = None,
    detector: DetectorPreset | None = None,
    absorption_flux_factor: float | None = None,
    selection_aware: bool = True,
    grid: LogLGrid | None = None,
) -> PoissonProcessLikelihood:
    """Assemble the full HMXB :class:`PoissonProcessLikelihood`.

    This is the one-call constructor used by the fit script and the tests: it
    wires the config's HMXB preset (for the fixed cutoff and support) into the
    XLF builder, builds the observation model from the chosen detector preset,
    and reads the priors from the config.

    Parameters
    ----------
    L_obs : ndarray
        Detected luminosities to fit (erg/s).
    cfg : dict
        Parsed YAML config (``configs/xlf_defaults.yaml`` style).
    distance_Mpc : float
        Galaxy distance.
    detector_name : str, optional
        Detector preset key in ``cfg['forward']['detectors']``.  Ignored if
        ``detector`` is given.
    detector : DetectorPreset, optional
        A ready-made detector preset (overrides ``detector_name``).
    absorption_flux_factor : float, optional
        Multiplicative absorption; defaults to the config value.
    selection_aware : bool
        ``True`` = completeness-ramp likelihood; ``False`` = naive hard cut.
    grid : LogLGrid, optional
        Integration grid; defaults to ``LogLGrid()``.
    """
    h = cfg["hmxb"]
    preset = h[h["preset"]]
    L_min = float(cfg["forward"]["L_min"])
    L_ref = float(cfg["L_unit"])
    L_cut = float(preset["L_cut"])
    band = str(preset.get("band", "0.5-8 keV"))

    if detector is None:
        detector = detector_from_config(cfg, name=detector_name)
    if absorption_flux_factor is None:
        absorption_flux_factor = float(cfg["forward"].get("absorption_flux_factor", 1.0))

    obs = ObservationModel(
        distance_Mpc=float(distance_Mpc),
        detector=detector,
        absorption_flux_factor=float(absorption_flux_factor),
        selection_aware=selection_aware,
    )
    make_xlf = make_hmxb_xlf_builder(
        L_cut=L_cut, L_min=L_min, L_ref=L_ref, band=band, param_kind="log_xi"
    )
    params = hmxb_parameters_from_config(cfg)
    return PoissonProcessLikelihood(
        L_obs=np.asarray(L_obs, dtype=float),
        make_xlf=make_xlf,
        obs=obs,
        grid=grid or LogLGrid(),
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """Uniform result of a fit, regardless of sampler.

    Attributes
    ----------
    samples : ndarray, shape (n_samples, ndim)
        Posterior samples in sampling space (log10 xi_eff, gamma).
    param_names : list[str]
    sampler : str
        ``"ultranest"`` or ``"emcee"``.
    logZ : float or None
        Log-evidence (UltraNest only; ``None`` for emcee).
    logZ_err : float or None
    wall_time_s : float
        Wall-clock seconds for the sampling call.
    extra : dict
        Sampler-specific diagnostics (e.g. autocorr time, n_eff).
    """

    samples: np.ndarray
    param_names: list[str]
    sampler: str
    logZ: float | None = None
    logZ_err: float | None = None
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)

    # ----- summaries -----
    def median(self) -> np.ndarray:
        return np.median(self.samples, axis=0)

    def quantiles(self, q=(0.16, 0.5, 0.84)) -> np.ndarray:
        """Per-parameter quantiles, shape (len(q), ndim)."""
        return np.quantile(self.samples, q, axis=0)

    def summary(self) -> dict:
        """Per-parameter dict of ``(median, minus, plus)`` 68% credible bounds."""
        lo, med, hi = self.quantiles((0.16, 0.5, 0.84))
        return {
            name: {
                "median": float(med[i]),
                "minus": float(med[i] - lo[i]),
                "plus": float(hi[i] - med[i]),
                "q16": float(lo[i]),
                "q84": float(hi[i]),
            }
            for i, name in enumerate(self.param_names)
        }

    def format_summary(self, truth: dict | None = None) -> str:
        """Pretty table of posterior medians +/- 68%, optionally vs truth."""
        s = self.summary()
        lines = []
        head = f"{'param':>14}  {'median':>10}  {'-68%':>9}  {'+68%':>9}"
        if truth is not None:
            head += f"  {'truth':>10}"
        lines.append(head)
        lines.append("-" * len(head))
        for name in self.param_names:
            d = s[name]
            row = (f"{name:>14}  {d['median']:>10.4f}  "
                   f"{d['minus']:>9.4f}  {d['plus']:>9.4f}")
            if truth is not None and name in truth:
                row += f"  {truth[name]:>10.4f}"
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Samplers behind one interface
# ---------------------------------------------------------------------------
def fit_xlf(
    like: PoissonProcessLikelihood,
    cfg: dict | None = None,
    sampler: str = "ultranest",
    *,
    seed: int = 0,
    **kwargs,
) -> FitResult:
    """Fit an XLF likelihood with the chosen sampler.

    Parameters
    ----------
    like : PoissonProcessLikelihood
        The assembled likelihood (e.g. from :func:`make_hmxb_problem`).
    cfg : dict, optional
        Parsed config; ``cfg['inference']['sampler']`` supplies defaults
        (n_live, n_walkers, n_steps, ...).
    sampler : {"ultranest", "emcee"}
    seed : int
        Seed for reproducibility (UltraNest's internal RNG / emcee init + RNG).
    **kwargs
        Forwarded to the sampler runner (override config defaults).
    """
    cfg = cfg or {}
    sset = cfg.get("inference", {}).get("sampler", {})
    if sampler == "ultranest":
        return _fit_ultranest(like, sset, seed=seed, **kwargs)
    if sampler == "emcee":
        return _fit_emcee(like, sset, seed=seed, **kwargs)
    raise ValueError(f"unknown sampler {sampler!r} (use 'ultranest' or 'emcee')")


def _fit_ultranest(
    like: PoissonProcessLikelihood,
    sset: dict,
    *,
    seed: int = 0,
    min_num_live_points: int | None = None,
    max_ncalls: int | None = None,
    quiet: bool = True,
    **kwargs,
) -> FitResult:
    """UltraNest ReactiveNestedSampler (primary)."""
    import logging
    import time

    from ultranest import ReactiveNestedSampler

    nlive = int(min_num_live_points or sset.get("n_live", 400))
    maxc = max_ncalls if max_ncalls is not None else sset.get("max_ncalls", None)

    if quiet:
        # UltraNest reports progress via the logging module, not stdout, so the
        # stdout/stderr redirect below is not enough on its own.
        logging.getLogger("ultranest").setLevel(logging.WARNING)

    # UltraNest seeds via numpy's global RNG; set it for determinism.
    np.random.seed(int(seed))

    names = like.param_names

    def loglike(theta):
        return like.log_likelihood(theta)

    def transform(u):
        return like.prior_transform(u)

    # log_dir=None keeps everything in memory: the HDF5 point store (which needs
    # h5py and has Windows file-locking quirks, UltraNest issue #61) is never
    # created.  Single-galaxy fits are short, so resumability is moot.
    sampler = ReactiveNestedSampler(
        names,
        loglike,
        transform,
        vectorized=False,
        log_dir=None,
    )
    run_kwargs = dict(min_num_live_points=nlive, show_status=not quiet, viz_callback=False)
    if maxc is not None:
        run_kwargs["max_ncalls"] = int(maxc)
    run_kwargs.update(kwargs)

    t0 = time.perf_counter()
    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = sampler.run(**run_kwargs)
    else:
        result = sampler.run(**run_kwargs)
    wall = time.perf_counter() - t0

    samples = np.asarray(result["samples"], dtype=float)

    return FitResult(
        samples=samples,
        param_names=names,
        sampler="ultranest",
        logZ=float(result["logz"]),
        logZ_err=float(result.get("logzerr", np.nan)),
        wall_time_s=wall,
        extra={"ncall": int(result.get("ncall", 0)),
               "niter": int(result.get("niter", 0))},
    )


def _fit_emcee(
    like: PoissonProcessLikelihood,
    sset: dict,
    *,
    seed: int = 0,
    n_walkers: int | None = None,
    n_steps: int | None = None,
    n_burn: int | None = None,
    progress: bool = False,
    **kwargs,
) -> FitResult:
    """emcee EnsembleSampler (fallback) with autocorr-based thinning."""
    import time

    import emcee

    ndim = like.ndim
    nwalk = int(n_walkers or sset.get("n_walkers", max(2 * ndim + 2, 16)))
    nsteps = int(n_steps or sset.get("n_steps", 4000))
    nburn = int(n_burn if n_burn is not None else sset.get("n_burn", nsteps // 4))

    rng = np.random.default_rng(int(seed))

    # Walker init: jitter around the prior-box centre (small ball), clipped to
    # bounds.  A small ball + many steps is robust for a smooth 2-D posterior.
    centre = np.array(
        [0.5 * (p.lo + p.hi) for p in like.parameters], dtype=float
    )
    widths = np.array([0.05 * (p.hi - p.lo) for p in like.parameters], dtype=float)
    p0 = centre[None, :] + widths[None, :] * rng.standard_normal((nwalk, ndim))
    for j, p in enumerate(like.parameters):
        p0[:, j] = np.clip(p0[:, j], p.lo + 1e-6, p.hi - 1e-6)

    sampler = emcee.EnsembleSampler(
        nwalk, ndim, like.log_posterior, **kwargs
    )

    t0 = time.perf_counter()
    sampler.run_mcmc(p0, nsteps, progress=progress)
    wall = time.perf_counter() - t0

    # autocorr-based thinning (fall back gracefully if the chain is too short)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.nanmax(tau))
    except Exception:
        tau_max = float(nsteps) / 50.0
    if not np.isfinite(tau_max) or tau_max <= 0:
        tau_max = float(nsteps) / 50.0
    thin = max(int(tau_max / 2.0), 1)
    burn = max(int(2.0 * tau_max), nburn)
    burn = min(burn, nsteps - 1)

    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)

    return FitResult(
        samples=np.asarray(flat, dtype=float),
        param_names=like.param_names,
        sampler="emcee",
        logZ=None,
        logZ_err=None,
        wall_time_s=wall,
        extra={
            "tau_max": tau_max,
            "thin": thin,
            "burn": burn,
            "n_walkers": nwalk,
            "n_steps": nsteps,
            "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        },
    )
