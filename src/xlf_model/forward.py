"""Forward model: from an XLF to a detected-source table, stage by stage.

The pipeline (each stage a pure function) is:

  1. population draw   -- N ~ Poisson(expected number above L_min); luminosities
                          via the XLF's analytic inverse-CDF sampler.
  2. observation layer -- luminosity -> flux (distance); multiplicative
                          absorption factor; smooth error-function completeness
                          ramp around the detector flux limit (NOT a hard cut).
  3. measurement noise -- expected counts from flux*exposure/cts_to_flux,
                          Poisson-fluctuated; observed flux/luminosity
                          back-converted with the SAME factor -> Eddington bias
                          emerges naturally near the threshold.

The detection decision is stochastic: a source is "detected" with probability
equal to its completeness, drawn with a Bernoulli.  The funnel accounting dict
records n_drawn -> n_above_limit -> n_detected.

Everything is reproducible from a numpy Generator (seed lives in the config).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

__all__ = [
    "MPC_TO_CM",
    "luminosity_to_flux",
    "flux_to_luminosity",
    "completeness_erf",
    "DetectorPreset",
    "detector_from_config",
    "draw_population",
    "observe",
    "ForwardResult",
    "run_forward",
    "format_funnel",
]

# 1 Mpc in cm (IAU 2015 parsec: 3.0856775814913673e18 cm/pc)
MPC_TO_CM = 3.0856775814913673e24


# ---------------------------------------------------------------------------
# Distance / flux conversions
# ---------------------------------------------------------------------------
def luminosity_to_flux(L, distance_Mpc: float):
    """Convert luminosity (erg/s) to flux (erg/s/cm^2) at a distance.

    F = L / (4 pi d^2).  Vectorized over L.
    """
    d_cm = float(distance_Mpc) * MPC_TO_CM
    return np.asarray(L, dtype=float) / (4.0 * np.pi * d_cm * d_cm)


def flux_to_luminosity(F, distance_Mpc: float):
    """Inverse of :func:`luminosity_to_flux`: flux -> luminosity (erg/s)."""
    d_cm = float(distance_Mpc) * MPC_TO_CM
    return np.asarray(F, dtype=float) * (4.0 * np.pi * d_cm * d_cm)


# ---------------------------------------------------------------------------
# Completeness ramp (smooth, NOT a hard cut)
# ---------------------------------------------------------------------------
def completeness_erf(flux, flux_limit_50: float, width_dex: float):
    """Smooth error-function completeness as a function of flux.

    Completeness is 0.5 at ``flux_limit_50``, rises to 1 well above and falls
    to 0 well below, with a 1-sigma width of ``width_dex`` in log10(flux):

        C(F) = 0.5 * (1 + erf( log10(F/F_lim) / (sqrt(2) * width_dex) ))

    Vectorized over ``flux``.  Non-positive fluxes map to completeness 0.
    """
    flux = np.asarray(flux, dtype=float)
    out = np.zeros_like(flux)
    pos = flux > 0.0
    x = np.log10(flux[pos] / float(flux_limit_50))
    out[pos] = 0.5 * (1.0 + erf(x / (np.sqrt(2.0) * float(width_dex))))
    return out


# ---------------------------------------------------------------------------
# Detector preset
# ---------------------------------------------------------------------------
@dataclass
class DetectorPreset:
    """A detector configuration.

    Attributes
    ----------
    name : str
        Preset key (e.g. ``"erosita_erass1"``).
    flux_limit_50 : float
        Flux (erg/s/cm^2) at 50% completeness.
    completeness_width_dex : float
        1-sigma width of the erf completeness ramp, in dex of flux.
    exposure_s : float
        Exposure time (s), used for the counts conversion.
    cts_to_flux : float
        Counts-to-flux factor (erg/cm^2 per count): expected counts =
        flux * exposure_s / cts_to_flux.
    band : str
        Energy-band bookkeeping label.
    """

    name: str
    flux_limit_50: float
    completeness_width_dex: float
    exposure_s: float
    cts_to_flux: float
    band: str = ""

    def completeness(self, flux):
        return completeness_erf(flux, self.flux_limit_50, self.completeness_width_dex)

    def expected_counts(self, flux):
        """Expected detected counts for a given flux: F * t_exp / cts_to_flux."""
        flux = np.asarray(flux, dtype=float)
        return flux * self.exposure_s / self.cts_to_flux


def detector_from_config(cfg: dict, name: str | None = None) -> DetectorPreset:
    """Build a :class:`DetectorPreset` from the parsed config."""
    fwd = cfg["forward"]
    name = name or fwd["demo_galaxy"]["detector"]
    d = fwd["detectors"][name]
    return DetectorPreset(
        name=name,
        flux_limit_50=float(d["flux_limit_50"]),
        completeness_width_dex=float(d["completeness_width_dex"]),
        exposure_s=float(d["exposure_s"]),
        cts_to_flux=float(d["cts_to_flux"]),
        band=str(d.get("band", "")),
    )


# ---------------------------------------------------------------------------
# Stage 1: population draw
# ---------------------------------------------------------------------------
def draw_population(xlf, rng: np.random.Generator, L_min: float | None = None):
    """Draw a Poisson population of source luminosities from an XLF.

    Parameters
    ----------
    xlf : HMXBXLF | LMXBXLF
        Any object exposing ``expected_number(L_lo=...)`` and ``sample(n, rng)``.
    rng : numpy.random.Generator
        Random generator (seeded upstream for reproducibility).
    L_min : float, optional
        Lower luminosity bound; defaults to ``xlf.L_min``.

    Returns
    -------
    luminosities : ndarray
        Drawn luminosities (erg/s), length N (a Poisson draw).
    expected_N : float
        The Poisson mean (expected number above L_min).
    """
    L_min = float(getattr(xlf, "L_min")) if L_min is None else float(L_min)
    expected_N = xlf.expected_number(L_lo=L_min)
    n = int(rng.poisson(expected_N))
    lum = xlf.sample(n, rng)
    return lum, expected_N


# ---------------------------------------------------------------------------
# Stage 2 + 3: observation + measurement noise
# ---------------------------------------------------------------------------
def observe(
    luminosities,
    distance_Mpc: float,
    detector: DetectorPreset,
    rng: np.random.Generator,
    absorption_flux_factor: float = 1.0,
):
    """Apply the observation layer and measurement noise to a luminosity list.

    Stages
    ------
    * luminosity -> intrinsic flux (distance) -> absorbed (observed) flux.
    * completeness C(F_obs); detection ~ Bernoulli(C).  This is the SMOOTH
      selection -- not a hard flux cut.
    * for detected sources: expected counts = F_obs * t_exp / cts_to_flux,
      Poisson-fluctuated to integer counts; observed flux back-converted with
      the SAME factor; observed luminosity from observed flux (un-absorbed using
      the same absorption factor, i.e. corrected back to intrinsic frame).

    The Poisson count fluctuation combined with the steeply-falling XLF produces
    Eddington bias: near the threshold, up-scattered sources are preferentially
    detected, so the mean observed L exceeds the mean true L.

    Returns
    -------
    table : dict of ndarray
        Per-detected-source arrays: ``L_true``, ``flux_true``, ``flux_obs``,
        ``counts``, ``L_obs``, ``completeness``.
    funnel : dict
        Accounting: ``n_drawn``, ``n_above_limit``, ``n_detected`` (+ the
        flux-limit value used for the "above limit" bookkeeping line).
    """
    luminosities = np.asarray(luminosities, dtype=float)
    n_drawn = int(luminosities.size)

    # intrinsic and absorbed (observed) flux
    flux_true = luminosity_to_flux(luminosities, distance_Mpc)
    flux_obs_intrinsic = flux_true * float(absorption_flux_factor)

    # bookkeeping "above limit": sources whose absorbed flux exceeds the 50%
    # flux limit (a reference count; detection itself is the smooth erf draw)
    above_limit_mask = flux_obs_intrinsic > detector.flux_limit_50
    n_above_limit = int(np.count_nonzero(above_limit_mask))

    # smooth completeness and stochastic detection
    comp = detector.completeness(flux_obs_intrinsic)
    detected = rng.random(n_drawn) < comp if n_drawn > 0 else np.zeros(0, dtype=bool)

    idx = np.nonzero(detected)[0]
    L_det = luminosities[idx]
    flux_det_true = flux_obs_intrinsic[idx]

    # measurement noise: Poisson-fluctuated counts, then back-convert
    expected_cts = detector.expected_counts(flux_det_true)
    counts = rng.poisson(expected_cts) if idx.size > 0 else np.zeros(0, dtype=int)
    # observed (noisy) flux from counts, using the SAME factor
    flux_obs = counts * detector.cts_to_flux / detector.exposure_s
    # observed luminosity: de-redden with the same absorption factor, then de-distance
    flux_obs_intrinsic_frame = flux_obs / float(absorption_flux_factor)
    L_obs = flux_to_luminosity(flux_obs_intrinsic_frame, distance_Mpc)

    table = {
        "L_true": L_det,
        "flux_true": flux_det_true,
        "flux_obs": flux_obs,
        "counts": counts.astype(float),
        "L_obs": L_obs,
        "completeness": comp[idx],
    }
    funnel = {
        "n_drawn": n_drawn,
        "n_above_limit": n_above_limit,
        "n_detected": int(idx.size),
        "flux_limit_50": float(detector.flux_limit_50),
    }
    return table, funnel


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
@dataclass
class ForwardResult:
    """Container for one forward-model run of a single galaxy / single XLF."""

    table: dict
    funnel: dict
    expected_N: float
    detector: DetectorPreset
    distance_Mpc: float
    component: str = ""
    L_drawn: np.ndarray = None  # full drawn population (before selection)


def run_forward(
    xlf,
    distance_Mpc: float,
    detector: DetectorPreset,
    rng: np.random.Generator,
    absorption_flux_factor: float = 1.0,
    L_min: float | None = None,
    component: str = "",
) -> ForwardResult:
    """Run the full forward model for one XLF component and one galaxy.

    Combines :func:`draw_population` and :func:`observe`.  The funnel dict is
    augmented with ``expected_N`` (the Poisson mean) for completeness.
    """
    lum, expected_N = draw_population(xlf, rng, L_min=L_min)
    table, funnel = observe(
        lum,
        distance_Mpc=distance_Mpc,
        detector=detector,
        rng=rng,
        absorption_flux_factor=absorption_flux_factor,
    )
    funnel["expected_N"] = float(expected_N)
    return ForwardResult(
        table=table,
        funnel=funnel,
        expected_N=float(expected_N),
        detector=detector,
        distance_Mpc=float(distance_Mpc),
        component=component,
        L_drawn=lum,
    )


# ---------------------------------------------------------------------------
# Funnel pretty-printer
# ---------------------------------------------------------------------------
def format_funnel(funnel: dict, title: str = "forward-model funnel") -> str:
    """Pretty-print a funnel accounting dict as a stage-by-stage table.

    Accepts either a single funnel dict (from :func:`observe`/:func:`run_forward`)
    or one augmented with ``expected_N``.  Shows the drop-off at each stage with
    the surviving fraction relative to the number drawn.
    """
    n_drawn = funnel["n_drawn"]
    n_above = funnel["n_above_limit"]
    n_det = funnel["n_detected"]
    exp_N = funnel.get("expected_N", None)
    flim = funnel.get("flux_limit_50", None)

    def frac(n):
        return (n / n_drawn) if n_drawn > 0 else 0.0

    width = 58
    lines = []
    lines.append("+" + "-" * width + "+")
    lines.append("| {:<{w}}|".format(title, w=width - 1))
    lines.append("+" + "-" * width + "+")
    if exp_N is not None:
        lines.append("| {:<34}{:>22} |".format("expected N (Poisson mean)", f"{exp_N:11.2f}"))
    lines.append("| {:<34}{:>22} |".format("n_drawn  (>L_min)", f"{n_drawn:11d}"))
    above_lbl = "n_above_limit"
    if flim is not None:
        above_lbl = f"n_above_limit (F>{flim:.1e})"
    lines.append(
        "| {:<34}{:>22} |".format(above_lbl, f"{n_above:8d} ({frac(n_above):5.1%})")
    )
    lines.append(
        "| {:<34}{:>22} |".format("n_detected", f"{n_det:8d} ({frac(n_det):5.1%})")
    )
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)
