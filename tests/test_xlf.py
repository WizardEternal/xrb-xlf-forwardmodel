"""Tests for the luminosity functions (src/xlf_model/xlf.py).

Coverage:
  * analytic N(>L) vs scipy.integrate.quad on the differential form
    (both XLFs, several parameter sets);
  * analytic expected_number vs quad;
  * inverse-CDF sampling vs the analytic CDF (KS test, 1e5 samples);
  * inverse-CDF sampling vs brute-force rejection sampling (KS test);
  * broken-power-law continuity at the two breaks;
  * normalization scales linearly with SFR (HMXB) and M* (LMXB).
"""

import numpy as np
import pytest
from scipy import integrate, stats

from xlf_model.xlf import HMXBXLF, LMXBXLF, hmxb_from_config, lmxb_from_config


# ---------------------------------------------------------------------------
# parameter sets to exercise
# ---------------------------------------------------------------------------
HMXB_PARAMS = [
    dict(xi=1.49, gamma=1.60, L_cut=2.1e40, L_min=1e35),   # Mineo+12
    dict(xi=3.3, gamma=1.61, L_cut=2.1e40, L_min=1e35),    # GGS03
    dict(xi=2.0, gamma=2.10, L_cut=1e41, L_min=1e36),      # steep, alt cut
    dict(xi=1.0, gamma=1.00, L_cut=1e40, L_min=1e35),      # log special case
]

LMXB_PARAMS = [
    dict(),  # all defaults (Gilfanov 2004)
    dict(alpha1=0.9, alpha2=2.0, alpha3=4.5, L_b1=2e37, L_b2=4e38),
    dict(alpha1=1.0, alpha2=1.0, alpha3=1.0),  # degenerate: single slope across breaks
]


# ---------------------------------------------------------------------------
# N(>L) and expected_number vs quad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", HMXB_PARAMS)
def test_hmxb_Ngt_vs_quad(params):
    xlf = HMXBXLF(**params)
    for L in [xlf.L_min * 3, 1e37, 1e38, 1e39, 0.5 * xlf.L_cut]:
        if L >= xlf.L_cut:
            continue
        # numerical N(>L) = int_L^Lcut dN/dL dL
        num, _ = integrate.quad(lambda x: float(xlf.dN_dL(x)), L, xlf.L_cut,
                                limit=200)
        ana = float(xlf.N_gt(L)[0])
        assert ana == pytest.approx(num, rel=1e-4), f"L={L:.2e}"


@pytest.mark.parametrize("params", HMXB_PARAMS)
def test_hmxb_expected_number_vs_quad(params):
    xlf = HMXBXLF(**params)
    num, _ = integrate.quad(lambda x: float(xlf.dN_dL(x)), xlf.L_min, xlf.L_cut,
                            limit=200)
    assert xlf.expected_number() == pytest.approx(num, rel=1e-4)


@pytest.mark.parametrize("params", LMXB_PARAMS)
def test_lmxb_Ngt_vs_quad(params):
    xlf = LMXBXLF(**params)
    for L in [xlf.L_min * 3, 1e37, 1e38, 1e39, 1e40]:
        if L >= xlf.L_cut:
            continue
        # integrate the differential form across the (possibly broken) range
        num, _ = integrate.quad(lambda x: float(xlf.dN_dL(x)), L, xlf.L_cut,
                                limit=400,
                                points=[xlf.L_b1, xlf.L_b2])
        ana = float(xlf.N_gt(L)[0])
        assert ana == pytest.approx(num, rel=1e-4), f"L={L:.2e}"


@pytest.mark.parametrize("params", LMXB_PARAMS)
def test_lmxb_expected_number_vs_quad(params):
    xlf = LMXBXLF(**params)
    num, _ = integrate.quad(lambda x: float(xlf.dN_dL(x)), xlf.L_min, xlf.L_cut,
                            limit=400, points=[xlf.L_b1, xlf.L_b2])
    assert xlf.expected_number() == pytest.approx(num, rel=1e-4)


# ---------------------------------------------------------------------------
# inverse-CDF sampling vs the analytic CDF (KS test)
# ---------------------------------------------------------------------------
def _analytic_cdf(xlf, L):
    """CDF F(L) = 1 - N(>L)/N_tot over the support [L_min, L_cut]."""
    Ntot = xlf.expected_number()
    return 1.0 - xlf.N_gt(L) / Ntot


@pytest.mark.parametrize("params", HMXB_PARAMS)
def test_hmxb_sampling_ks_vs_cdf(params):
    xlf = HMXBXLF(**params)
    rng = np.random.default_rng(7)
    samples = xlf.sample(100_000, rng)
    assert samples.min() >= xlf.L_min * (1 - 1e-9)
    assert samples.max() <= xlf.L_cut * (1 + 1e-9)
    cdf = lambda L: np.asarray(_analytic_cdf(xlf, np.atleast_1d(L)))
    stat, p = stats.kstest(samples, cdf)
    assert p > 0.01, f"KS p={p:.3g}, stat={stat:.3g}"


@pytest.mark.parametrize("params", LMXB_PARAMS)
def test_lmxb_sampling_ks_vs_cdf(params):
    xlf = LMXBXLF(**params)
    rng = np.random.default_rng(7)
    samples = xlf.sample(100_000, rng)
    assert samples.min() >= xlf.L_min * (1 - 1e-9)
    assert samples.max() <= xlf.L_cut * (1 + 1e-9)
    cdf = lambda L: np.asarray(_analytic_cdf(xlf, np.atleast_1d(L)))
    stat, p = stats.kstest(samples, cdf)
    assert p > 0.01, f"KS p={p:.3g}, stat={stat:.3g}"


# ---------------------------------------------------------------------------
# inverse-CDF vs brute-force rejection sampling (two-sample KS)
# ---------------------------------------------------------------------------
def _rejection_sample(xlf, n, rng, oversample=60):
    """Brute-force rejection sampling from dN/dL on [L_min, L_cut] in log-L."""
    logLmin = np.log10(xlf.L_min)
    logLmax = np.log10(xlf.L_cut)
    # sampling density in log-L is proportional to L * dN/dL (Jacobian dL = L ln10 dlogL)
    logL_grid = np.linspace(logLmin, logLmax, 4000)
    L_grid = 10.0**logL_grid
    w = L_grid * xlf.dN_dL(L_grid)
    wmax = w.max() * 1.05
    out = []
    while len(out) < n:
        m = n * oversample
        logL = rng.uniform(logLmin, logLmax, m)
        L = 10.0**logL
        wprop = L * xlf.dN_dL(L)
        accept = rng.random(m) < (wprop / wmax)
        out.extend(L[accept].tolist())
    return np.array(out[:n])


@pytest.mark.parametrize("params", HMXB_PARAMS)
def test_hmxb_sampling_vs_rejection(params):
    xlf = HMXBXLF(**params)
    rng = np.random.default_rng(3)
    a = xlf.sample(40_000, rng)
    b = _rejection_sample(xlf, 40_000, rng)
    stat, p = stats.ks_2samp(a, b)
    assert p > 0.01, f"2-sample KS p={p:.3g}, stat={stat:.3g}"


@pytest.mark.parametrize("params", LMXB_PARAMS)
def test_lmxb_sampling_vs_rejection(params):
    xlf = LMXBXLF(**params)
    rng = np.random.default_rng(3)
    a = xlf.sample(40_000, rng)
    b = _rejection_sample(xlf, 40_000, rng)
    stat, p = stats.ks_2samp(a, b)
    assert p > 0.01, f"2-sample KS p={p:.3g}, stat={stat:.3g}"


# ---------------------------------------------------------------------------
# broken-power-law continuity at the breaks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", LMXB_PARAMS)
def test_lmxb_continuity_at_breaks(params):
    xlf = LMXBXLF(**params)
    for Lb in [xlf.L_b1, xlf.L_b2]:
        lo = float(xlf.dN_dL(Lb * (1 - 1e-7)))
        hi = float(xlf.dN_dL(Lb * (1 + 1e-7)))
        assert lo == pytest.approx(hi, rel=1e-4), f"break at {Lb:.2e}: {lo} vs {hi}"


# ---------------------------------------------------------------------------
# normalization linear scaling
# ---------------------------------------------------------------------------
def test_hmxb_scales_linearly_with_sfr(config):
    x1 = hmxb_from_config(config, SFR=1.0)
    x5 = hmxb_from_config(config, SFR=5.0)
    assert x5.expected_number() == pytest.approx(5.0 * x1.expected_number(), rel=1e-12)
    assert float(x5.N_gt(1e38)[0]) == pytest.approx(5.0 * float(x1.N_gt(1e38)[0]), rel=1e-12)


def test_lmxb_scales_linearly_with_mstar(config):
    m1 = lmxb_from_config(config, Mstar=1e10)
    m4 = lmxb_from_config(config, Mstar=4e10)
    assert m4.expected_number() == pytest.approx(4.0 * m1.expected_number(), rel=1e-12)
    assert float(m4.N_gt(1e37)[0]) == pytest.approx(4.0 * float(m1.N_gt(1e37)[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# published cross-checks (sanity anchors)
# ---------------------------------------------------------------------------
def test_lmxb_Ngt_1e37_matches_G04(config):
    """G04 Eq.11: N(>1e37) = 142.9 per 1e11 Msun (within the K1-vs-N inconsistency)."""
    lmxb = lmxb_from_config(config, Mstar=1e11)
    val = float(lmxb.N_gt(1e37)[0])
    # K1=440.4 and N=142.9 are independently quoted with ~6% errors; tolerate 5%
    assert val == pytest.approx(142.9, rel=0.05), val


def test_hmxb_Ngt_1e38_single_pl(config):
    """Single-PL N(>1e38)/SFR = xi/(gamma-1) modulo the high-L cutoff."""
    hmxb = hmxb_from_config(config, SFR=1.0)
    val = float(hmxb.N_gt(1e38)[0])
    # analytic pure-PL value is xi/(gamma-1)=2.483; cutoff lowers it slightly
    assert 2.3 < val < 2.5, val
