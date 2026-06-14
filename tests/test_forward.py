"""Tests for the forward model (src/xlf_model/forward.py).

Coverage:
  * Poisson population-draw mean matches the XLF integral;
  * erf-ramp completeness -> 0 far below, 1 far above, 0.5 at the limit;
  * Eddington bias: mean observed L > true L near threshold for a steep XLF;
  * same seed -> identical output (reproducibility).
"""

import numpy as np
import pytest

from xlf_model.forward import (
    DetectorPreset,
    completeness_erf,
    detector_from_config,
    draw_population,
    flux_to_luminosity,
    luminosity_to_flux,
    observe,
    run_forward,
)
from xlf_model.xlf import HMXBXLF, hmxb_from_config, lmxb_from_config


# ---------------------------------------------------------------------------
# Poisson population-draw mean matches the integral
# ---------------------------------------------------------------------------
def test_poisson_draw_mean_matches_integral(config):
    xlf = hmxb_from_config(config, SFR=10.0)  # large SFR -> many sources
    expected = xlf.expected_number()
    rng = np.random.default_rng(0)
    counts = []
    for _ in range(400):
        lum, exp_N = draw_population(xlf, rng)
        counts.append(lum.size)
        assert exp_N == pytest.approx(expected)
    mean = np.mean(counts)
    # sterr of the mean over 400 Poisson draws ~ sqrt(expected/400)
    sterr = np.sqrt(expected / len(counts))
    assert abs(mean - expected) < 4.0 * sterr, (mean, expected, sterr)


def test_draw_luminosities_within_support(config):
    xlf = lmxb_from_config(config, Mstar=1e11)
    rng = np.random.default_rng(1)
    lum, _ = draw_population(xlf, rng)
    assert lum.min() >= xlf.L_min
    assert lum.max() <= xlf.L_cut


# ---------------------------------------------------------------------------
# erf-ramp completeness limits
# ---------------------------------------------------------------------------
def test_completeness_erf_limits():
    flim = 5e-14
    width = 0.2
    # far below -> 0
    assert completeness_erf(flim * 1e-3, flim, width) == pytest.approx(0.0, abs=1e-8)
    # far above -> 1
    assert completeness_erf(flim * 1e3, flim, width) == pytest.approx(1.0, abs=1e-8)
    # exactly at the limit -> 0.5
    assert float(completeness_erf(flim, flim, width)) == pytest.approx(0.5, abs=1e-12)


def test_completeness_monotonic():
    flim, width = 5e-14, 0.3
    f = np.logspace(-16, -11, 200)
    c = completeness_erf(f, flim, width)
    assert np.all(np.diff(c) >= -1e-12)
    assert c[0] < 0.01 and c[-1] > 0.99


def test_completeness_nonpositive_flux_is_zero():
    assert float(completeness_erf(0.0, 5e-14, 0.2)) == 0.0
    assert float(completeness_erf(-1.0, 5e-14, 0.2)) == 0.0


# ---------------------------------------------------------------------------
# flux <-> luminosity round trip
# ---------------------------------------------------------------------------
def test_flux_luminosity_roundtrip():
    L = np.array([1e37, 1e38, 1e39])
    F = luminosity_to_flux(L, 12.3)
    L2 = flux_to_luminosity(F, 12.3)
    assert np.allclose(L, L2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Eddington bias: mean observed L > true L near threshold for a steep XLF
# ---------------------------------------------------------------------------
def test_eddington_bias_near_threshold():
    """A steep XLF observed near the flux threshold should show mean L_obs > L_true.

    Construct a detector whose 50% limit sits in the middle of a steep
    population, with few counts at threshold (large Poisson scatter), and many
    sources, so the bias is statistically clear.
    """
    # steep HMXB-like XLF, narrow luminosity range around the threshold
    xlf = HMXBXLF(xi=5e4, gamma=2.5, L_cut=1e40, L_min=1e37, SFR=1.0)
    distance = 5.0  # Mpc
    # choose flux limit so it lands within the populated range; few counts there
    # at threshold so Poisson scatter (and thus Eddington bias) is strong.
    L_thresh = 3e38
    flim = float(luminosity_to_flux(L_thresh, distance))
    # cts_to_flux & exposure: ~8 expected counts at the threshold flux
    exposure = 1000.0
    cts_to_flux = flim * exposure / 8.0
    det = DetectorPreset(
        name="test", flux_limit_50=flim, completeness_width_dex=0.15,
        exposure_s=exposure, cts_to_flux=cts_to_flux, band="test",
    )
    rng = np.random.default_rng(42)

    L_true_all = []
    L_obs_all = []
    for _ in range(60):
        lum, _ = draw_population(xlf, rng)
        table, _ = observe(lum, distance, det, rng)
        # restrict to a window around the threshold where bias is maximal
        Lt = table["L_true"]
        Lo = table["L_obs"]
        m = (Lt > L_thresh * 0.3) & (Lt < L_thresh * 3.0) & (Lo > 0)
        L_true_all.append(Lt[m])
        L_obs_all.append(Lo[m])
    Lt = np.concatenate(L_true_all)
    Lo = np.concatenate(L_obs_all)
    assert Lt.size > 500, f"too few sources for a stable test: {Lt.size}"
    # near threshold, detected sources are preferentially up-scattered:
    # mean observed luminosity exceeds mean true luminosity
    assert Lo.mean() > Lt.mean(), (Lo.mean(), Lt.mean())


# ---------------------------------------------------------------------------
# reproducibility: same seed -> identical output
# ---------------------------------------------------------------------------
def test_same_seed_identical_output(config):
    xlf = hmxb_from_config(config, SFR=2.0)
    det = detector_from_config(config, "erosita_erass1")
    distance = 8.0

    def run(seed):
        rng = np.random.default_rng(seed)
        return run_forward(xlf, distance, det, rng)

    r1 = run(999)
    r2 = run(999)
    assert r1.funnel == r2.funnel
    for k in r1.table:
        assert np.array_equal(r1.table[k], r2.table[k]), k
    assert np.array_equal(r1.L_drawn, r2.L_drawn)


def test_different_seed_different_output(config):
    xlf = hmxb_from_config(config, SFR=2.0)
    det = detector_from_config(config, "erosita_erass1")
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    ra = run_forward(xlf, 8.0, det, rng_a)
    rb = run_forward(xlf, 8.0, det, rng_b)
    # extremely unlikely to be identical
    assert ra.L_drawn.size != rb.L_drawn.size or not np.array_equal(
        ra.L_drawn, rb.L_drawn
    )


# ---------------------------------------------------------------------------
# expected detected count matches the smooth-ramp integral
# ---------------------------------------------------------------------------
def test_expected_detected_matches_ramp_integral(config):
    """Mean n_detected over many sims = int dN/dL * C(L) dL."""
    xlf = hmxb_from_config(config, SFR=20.0)  # boost stats
    det = detector_from_config(config, "chandra_like")
    distance = 10.0

    L = np.logspace(np.log10(xlf.L_min), np.log10(xlf.L_cut), 20000)
    flux = luminosity_to_flux(L, distance)
    C = completeness_erf(flux, det.flux_limit_50, det.completeness_width_dex)
    expected_det = np.trapezoid(xlf.dN_dL(L) * C, L)

    rng = np.random.default_rng(5)
    ndet = []
    for _ in range(200):
        r = run_forward(xlf, distance, det, rng)
        ndet.append(r.funnel["n_detected"])
    mean = np.mean(ndet)
    sterr = np.sqrt(expected_det / len(ndet))
    assert abs(mean - expected_det) < 4.0 * sterr, (mean, expected_det, sterr)
