r"""CXB-contamination bias on the recovered HMXB slope.

WHY THIS EXISTS
---------------
The optional real-data demonstration (M101, the Antennae) fits each galaxy's
HMXB XLF ABOVE a conservative threshold of ``1.5 x L_lim``.  For those two
galaxies the resulting fit thresholds are ``logL = 36.54`` (M101) and
``logL = 37.10`` (Antennae), and the median fit-source luminosity is
``logL ~ 36.97 / 37.55``.  M101 has ZERO fit sources above ``1e39 erg/s``.  This
sits squarely inside the luminosity range (Mineo+12 Sec 7.3:
``logL ~ 36.5-38.5``) where the cosmic X-ray background (CXB), i.e. unrelated
background AGN seen through the galaxy, contaminates the point-source list.  M12
model the CXB explicitly (their Eq. 17, an additive statistical component, ~30%
of sources in their regions by design); the repo's demonstration does NOT.  The
fit is dominated by this CXB-affected regime.

This script quantifies how much a CXB contaminant biases the recovered HMXB
slope, using the repo's OWN machinery (no new physics): build a mock detected
catalogue that is a mixture of

    (1-f) HMXBs  with the true slope gamma=1.6   (Mineo+12 default), plus
       f  CXB sources with a FLATTER power-law luminosity distribution
          (logN-logS slope ~0.8-1.2, i.e. a differential slope < 1.6, so the CXB
           contributes relatively MORE bright sources than a steep HMXB XLF),

draw it over the demonstration's fit range ``logL in [36.5, 39.5]``, hand the
COMBINED list to the repo's selection-aware :class:`PoissonProcessLikelihood`
(which fits a single HMXB power law and has no CXB term, exactly the
demonstration's situation), and record the recovered ``gamma`` vs the true 1.6.
The induced ``Delta gamma = gamma_recovered - 1.6`` is the bias the missing CXB
term imprints.

The result: 20-30% CXB contamination with a flatter contaminant slope induces
``Delta gamma ~ -0.10 to -0.31``, the same sign and magnitude as (a) the
demonstration's aware-vs-naive gap (~0.2) and (b) the aware slopes (1.48 / 1.43)
sitting below 1.60.  Conclusion: the demonstration's absolute slopes and its
aware-naive gap are NOT interpretable as a pure selection effect without adding
M12's CXB model to the likelihood.

This is a self-contained Monte-Carlo, fully reproducible from the constants
below; it is NOT a heavy rerun and touches no fit-result tables.  ``--quick``
(lighter N_det / fewer realizations + live points) finishes the whole 9-cell
table in a few minutes; the full run is heavier.

Usage
-----
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe scripts\cxb_bias_estimate.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from xlf_model.forward import DetectorPreset  # noqa: E402
from xlf_model.inference import (  # noqa: E402
    LogLGrid,
    ObservationModel,
    Parameter,
    PoissonProcessLikelihood,
    fit_xlf,
    make_hmxb_xlf_builder,
)
from xlf_model.xlf import _inverse_cdf_powerlaw  # noqa: E402

# ---------------------------------------------------------------------------
# Constants of the experiment
# ---------------------------------------------------------------------------
TRUE_GAMMA = 1.60               # Mineo+12 HMXB single-PL slope (the truth)
LOG_LFIT_LO = 36.5              # fit range lower edge (logL), matches the demo
LOG_LFIT_HI = 39.5              # fit range upper edge (logL)
L_REF = 1.0e38                  # L38 definition (matches xlf_defaults / the demo)
L_CUT = 1.0e41                  # cutoff fixed at M12's per-galaxy value (the demo)

# CXB contaminant differential power-law slopes to scan.  A CXB "power-law slope
# ~0.8-1.2" for the logN-logS, i.e. a FLATTER-than-HMXB contaminant, is what
# biases gamma low (it adds relatively more bright sources).  We scan the
# differential slope directly over the 0.8-1.2 band so the contaminant is
# unambiguously flatter than the gamma=1.6 HMXB population.
CXB_SLOPES = (0.8, 1.0, 1.2)
CXB_FRACTIONS = (0.10, 0.20, 0.30)   # fraction of the detected list that is CXB

N_DET = 200                     # detected-list size per mock (rich, like the demo)
N_REALIZATIONS = 200            # Monte-Carlo realizations per (slope, fraction) cell
N_LIVE = 400                    # UltraNest live points (full run; apples-to-apples)
SEED_BASE = 20260612

# --quick smoke/sanity settings: smaller mock catalogues, fewer realizations and
# live points so the whole 9-cell table finishes in a few minutes.  The CXB bias
# is a large, systematic effect (Delta gamma of order -0.1 to -0.3), so it is
# already resolved well above the per-cell scatter at these settings.
QUICK_N_DET = 120
QUICK_N_REALIZATIONS = 12
QUICK_N_LIVE = 150


def _sample_powerlaw_L(slope: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` luminosities (erg/s) from a single power law ``dN/dL ~ L^-slope``
    truncated to the fit range ``[10**LOG_LFIT_LO, 10**LOG_LFIT_HI]``.

    Reuses the repo's analytic inverse-CDF sampler (``_inverse_cdf_powerlaw``);
    no new sampling code.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    a = 10.0 ** LOG_LFIT_LO
    b = 10.0 ** LOG_LFIT_HI
    u = rng.random(int(n))
    return _inverse_cdf_powerlaw(u, a, b, slope)


def _build_likelihood(L_obs: np.ndarray) -> PoissonProcessLikelihood:
    """The repo's selection-aware HMXB single-PL likelihood, fit range only.

    There is NO CXB term, exactly the real-data demonstration's situation: the
    likelihood can only describe the catalogue with one HMXB power law.  We set
    the completeness to ~1 across the whole fit range (a flat, fully-complete
    detector well below the fit-range floor) so the recovered bias is purely the
    CXB-mixture effect and not a selection artefact, the controlled comparison we
    want.
    """
    # detector with F50 far below the fit-range floor => completeness ~ 1 over
    # the whole fit range (the contamination, not selection, is under test).
    L_floor = 10.0 ** LOG_LFIT_LO
    detector = DetectorPreset(
        name="cxb_test_complete",
        flux_limit_50=L_floor / (4.0 * np.pi * (10.0 * 3.0856775814913673e24) ** 2) / 1e3,
        completeness_width_dex=0.2,
        exposure_s=1.0,
        cts_to_flux=1.0,
        band="0.5-8 keV",
    )
    obs = ObservationModel(
        distance_Mpc=10.0, detector=detector,
        absorption_flux_factor=1.0, selection_aware=True,
    )
    make_xlf = make_hmxb_xlf_builder(
        L_cut=L_CUT, L_min=L_floor, L_ref=L_REF, band="0.5-8 keV", param_kind="log_xi",
    )
    params = [
        Parameter("log10_xi_eff", -2.0, 6.0, "uniform"),
        Parameter("gamma", 0.5, 4.0, "uniform"),
    ]
    return PoissonProcessLikelihood(
        L_obs=np.asarray(L_obs, dtype=float),
        make_xlf=make_xlf,
        obs=obs,
        grid=LogLGrid(points_per_dex=200),
        parameters=params,
    )


def _fit_gamma(L_obs: np.ndarray, seed: int, *, n_live: int = N_LIVE) -> float:
    """Fit the mixed catalogue with the HMXB-only likelihood; return median gamma.

    Uses the repo's own single-galaxy UltraNest settings (``n_live`` live points,
    no ncall cap; default 400 = the full-run value), so the recovered gamma is
    apples-to-apples with the demonstration's fits.  ``--quick`` lowers ``n_live``.
    """
    like = _build_likelihood(L_obs)
    fr = fit_xlf(like, cfg={"inference": {"sampler": {"n_live": int(n_live)}}},
                 sampler="ultranest", seed=seed)
    return float(fr.summary()["gamma"]["median"])


def run_cell(cxb_slope: float, cxb_frac: float, *, n_real: int, seed0: int,
             n_det: int = N_DET, n_live: int = N_LIVE):
    """Return the array of recovered gamma over ``n_real`` mixture realizations."""
    gammas = np.empty(n_real)
    for i in range(n_real):
        rng = np.random.default_rng(seed0 + i)
        n_cxb = int(round(n_det * cxb_frac))
        n_hmxb = n_det - n_cxb
        # HMXB component: the true gamma=1.60 power law over the fit range
        L_hmxb = _sample_powerlaw_L(TRUE_GAMMA, n_hmxb, rng)
        # CXB component: a flatter power law (logN-logS slope ~0.8-1.2)
        L_cxb = _sample_powerlaw_L(cxb_slope, n_cxb, rng)
        L_obs = np.concatenate([L_hmxb, L_cxb])
        gammas[i] = _fit_gamma(L_obs, seed=seed0 + i, n_live=n_live)
    return gammas


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-real", type=int, default=None,
                   help="realizations per cell (default: full=%d, quick=%d)"
                        % (N_REALIZATIONS, QUICK_N_REALIZATIONS))
    p.add_argument("--quick", action="store_true",
                   help="fast smoke/sanity run (lighter N_det, fewer realizations "
                        "+ live points) to populate the result table in minutes")
    args = p.parse_args(argv)
    if args.quick:
        n_real = args.n_real if args.n_real is not None else QUICK_N_REALIZATIONS
        n_det, n_live = QUICK_N_DET, QUICK_N_LIVE
    else:
        n_real = args.n_real if args.n_real is not None else N_REALIZATIONS
        n_det, n_live = N_DET, N_LIVE

    print("=" * 74)
    print("CXB contamination bias on the recovered HMXB slope"
          + ("  [--quick]" if args.quick else ""))
    print(f"  true HMXB gamma         = {TRUE_GAMMA:.2f}  (Mineo+12)")
    print(f"  fit range               = logL [{LOG_LFIT_LO}, {LOG_LFIT_HI}]"
          f"  (the demo's regime)")
    print(f"  CXB contaminant slopes  = {CXB_SLOPES}  (flatter logN-logS)")
    print(f"  CXB fractions           = {CXB_FRACTIONS}")
    print(f"  N_det per mock          = {n_det}, realizations/cell = {n_real}, "
          f"n_live = {n_live}")
    print("=" * 74)

    # header
    hdr = "  f_CXB  |" + "".join(f"  gamma_CXB={s:<4}  " for s in CXB_SLOPES)
    print("\nRecovered gamma (median over realizations; truth = 1.60):")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    table = {}  # (frac, slope) -> (mean_gamma, dgamma, std)
    for frac in CXB_FRACTIONS:
        cells = []
        for slope in CXB_SLOPES:
            seed0 = SEED_BASE + int(round(frac * 100)) * 1000 + int(round(slope * 10))
            gammas = run_cell(slope, frac, n_real=n_real, seed0=seed0,
                              n_det=n_det, n_live=n_live)
            g_mean = float(np.mean(gammas))
            g_std = float(np.std(gammas))
            dg = g_mean - TRUE_GAMMA
            table[(frac, slope)] = (g_mean, dg, g_std)
            cells.append(f"{g_mean:5.3f} (d={dg:+.3f})")
        row = f"  {frac:4.0%}  |" + "".join(f"  {c:>16}" for c in cells)
        print(row)

    print("\nInduced bias Delta gamma = gamma_recovered - 1.60:")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for frac in CXB_FRACTIONS:
        cells = [f"{table[(frac, s)][1]:+.3f}" for s in CXB_SLOPES]
        row = f"  {frac:4.0%}  |" + "".join(f"  {c:>16}" for c in cells)
        print(row)

    # the bias range across the 20-30% rows
    dvals = [table[(f, s)][1] for f in (0.20, 0.30) for s in CXB_SLOPES]
    print("\n" + "-" * 74)
    print(f"  20-30% CXB contamination induces Delta gamma in "
          f"[{min(dvals):+.3f}, {max(dvals):+.3f}]")
    print(f"  (compare: demo aware-naive gap ~ -0.2; aware slopes 1.48/1.43 "
          f"vs truth 1.60)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
