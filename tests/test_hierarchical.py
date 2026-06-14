"""Tests for the hierarchical stack (src/xlf_model/hierarchical.py).

Coverage (the Phase-3b validation):
  * a 1-galaxy survey's joint lnL EXACTLY equals that galaxy's single-galaxy lnL
    (the joint wrapper adds nothing for one member);
  * the 2-galaxy joint lnL equals the sum of the two per-galaxy lnL terms, and
    that sum is hand-reconstructed from the Poisson-process pieces
    (sum_i ln lambda - Lambda) for each galaxy;
  * a 0-detection galaxy contributes a finite -Lambda_g (no detections is
    informative, not a failure);
  * a tiny 5-galaxy seeded survey fit recovers the true gamma within 2 sigma
    (fast settings).
  * structural guards: mismatched priors are rejected; total_n_det/n_galaxies
    accounting is correct.
"""

import numpy as np
import pytest

from xlf_model.forward import DetectorPreset, run_forward
from xlf_model.hierarchical import (
    GalaxySpec,
    JointHMXBLikelihood,
    build_joint_likelihood,
    make_hmxb_xlf_builder_with_sfr,
    single_galaxy_likelihood,
)
from xlf_model.inference import LogLGrid, ObservationModel, Parameter, fit_xlf
from xlf_model.xlf import HMXBXLF


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------
L_CUT, L_MIN, L_REF = 2.1e40, 1.0e35, 1.0e38
BAND = "0.5-8 keV"


def _detector():
    return DetectorPreset(
        name="erosita_erass1", flux_limit_50=5.0e-14, completeness_width_dex=0.2,
        exposure_s=200.0, cts_to_flux=1.0e-12, band="0.5-2 keV",
    )


def _params():
    return [
        Parameter("log10_xi", -1.0, 2.0, "uniform"),
        Parameter("gamma", 0.8, 2.6, "uniform"),
    ]


def _simulate_galaxy(sfr, dist, seed):
    """Forward-simulate one galaxy and wrap it as a GalaxySpec."""
    det = _detector()
    xlf = HMXBXLF(xi=1.49, gamma=1.6, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF,
                  SFR=sfr, band=BAND)
    rng = np.random.default_rng(seed)
    res = run_forward(xlf, dist, det, rng, component="HMXB")
    L = res.table["L_obs"]
    L = L[L > 0]
    return GalaxySpec(L_obs=L, SFR=sfr, distance_Mpc=dist, detector=det,
                      name=f"g_{seed}")


# ---------------------------------------------------------------------------
# 1) joint of a 1-galaxy survey == single-galaxy lnL (EXACT)
# ---------------------------------------------------------------------------
def test_joint_one_galaxy_equals_single():
    """A 1-galaxy survey's joint lnL must EQUAL that galaxy's single lnL exactly.

    Both use the SAME builder (shared theta = (log10 xi, gamma) with the galaxy's
    SFR folded in), so the joint wrapper is the identity for one member -- this
    must hold bit-for-bit, not just approximately.
    """
    gal = _simulate_galaxy(sfr=10.0, dist=5.0, seed=101)
    assert gal.n_det >= 2, "need a non-degenerate galaxy for this test"
    params = _params()

    single = single_galaxy_likelihood(
        gal, params, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF, band=BAND,
    )
    joint = build_joint_likelihood(
        [gal], params, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF, band=BAND,
    )

    for theta in ([np.log10(1.49), 1.6], [0.3, 1.4], [-0.2, 2.0]):
        ll_single = single.log_likelihood(theta)
        ll_joint = joint.log_likelihood(theta)
        assert ll_joint == ll_single, f"theta={theta}: {ll_joint} != {ll_single}"


# ---------------------------------------------------------------------------
# 2) joint of 2 galaxies == sum of the two per-galaxy lnL (hand-check)
# ---------------------------------------------------------------------------
def test_joint_two_galaxies_sums_correctly():
    """The 2-galaxy joint lnL must equal the sum of the two single lnL terms,
    and that sum must equal a hand-reconstruction from the Poisson-process
    pieces (sum_i ln lambda_g - Lambda_g) for each galaxy."""
    g1 = _simulate_galaxy(sfr=8.0, dist=4.0, seed=201)
    g2 = _simulate_galaxy(sfr=3.0, dist=9.0, seed=202)
    assert g1.n_det >= 2 and g2.n_det >= 1
    params = _params()
    theta = [np.log10(1.49), 1.6]

    # the two single-galaxy likelihoods (same shared parametrisation)
    s1 = single_galaxy_likelihood(g1, params, L_cut=L_CUT, L_min=L_MIN,
                                  L_ref=L_REF, band=BAND)
    s2 = single_galaxy_likelihood(g2, params, L_cut=L_CUT, L_min=L_MIN,
                                  L_ref=L_REF, band=BAND)
    joint = build_joint_likelihood([g1, g2], params, L_cut=L_CUT, L_min=L_MIN,
                                   L_ref=L_REF, band=BAND)

    ll_sum = s1.log_likelihood(theta) + s2.log_likelihood(theta)
    ll_joint = joint.log_likelihood(theta)
    assert ll_joint == pytest.approx(ll_sum, rel=0, abs=1e-9)

    # HAND reconstruction from the Poisson-process pieces for each galaxy:
    #   ln L_g = sum_i ln lambda_g(L_i) - Lambda_g
    hand = 0.0
    for s in (s1, s2):
        ln_lam = s.log_intensity(theta)          # sum_i ln lambda_g(L_i)
        Lambda = s.expected_detected(theta)       # Lambda_g
        hand += float(np.sum(ln_lam) - Lambda)
    assert ll_joint == pytest.approx(hand, rel=0, abs=1e-9)

    # per-galaxy vector must match the two singles individually
    per = joint.per_galaxy_log_likelihood(theta)
    assert per[0] == pytest.approx(s1.log_likelihood(theta), abs=1e-12)
    assert per[1] == pytest.approx(s2.log_likelihood(theta), abs=1e-12)


# ---------------------------------------------------------------------------
# a 0-detection galaxy contributes a finite -Lambda (not a failure)
# ---------------------------------------------------------------------------
def test_zero_detection_galaxy_contributes_minus_lambda():
    """A galaxy with N_det=0 must contribute a finite ln L = -Lambda_g (the
    Poisson 'expected some, saw none' term), NOT -inf or NaN."""
    det = _detector()
    gal = GalaxySpec(L_obs=np.array([]), SFR=1.0, distance_Mpc=25.0,
                     detector=det)
    params = _params()
    s = single_galaxy_likelihood(gal, params, L_cut=L_CUT, L_min=L_MIN,
                                 L_ref=L_REF, band=BAND)
    theta = [np.log10(1.49), 1.6]
    ll = s.log_likelihood(theta)
    Lambda = s.expected_detected(theta)
    assert np.isfinite(ll)
    assert ll == pytest.approx(-Lambda, abs=1e-12)
    assert ll <= 0.0


# ---------------------------------------------------------------------------
# structural guards
# ---------------------------------------------------------------------------
def test_mismatched_priors_rejected():
    """All survey galaxies must share the identical parameter set."""
    g1 = _simulate_galaxy(sfr=8.0, dist=4.0, seed=301)
    p_a = _params()
    p_b = [Parameter("log10_xi", -1.0, 2.0), Parameter("DIFFERENT", 0.8, 2.6)]
    m1 = single_galaxy_likelihood(g1, p_a, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF)
    m2 = single_galaxy_likelihood(g1, p_b, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF)
    with pytest.raises(ValueError):
        JointHMXBLikelihood(members=[m1, m2], parameters=p_a)


def test_empty_survey_rejected():
    with pytest.raises(ValueError):
        build_joint_likelihood([], _params(), L_cut=L_CUT, L_min=L_MIN,
                               L_ref=L_REF)


def test_accounting():
    """total_n_det and n_galaxies must match the survey."""
    gals = [_simulate_galaxy(sfr=s, dist=d, seed=400 + i)
            for i, (s, d) in enumerate([(10.0, 4.0), (2.0, 8.0), (0.5, 15.0)])]
    joint = build_joint_likelihood(gals, _params(), L_cut=L_CUT, L_min=L_MIN,
                                   L_ref=L_REF)
    assert joint.n_galaxies == 3
    assert joint.total_n_det == sum(g.n_det for g in gals)


# ---------------------------------------------------------------------------
# the SFR-folding builder uses the galaxy's own SFR (consistency with forward)
# ---------------------------------------------------------------------------
def test_builder_folds_sfr():
    """make_hmxb_xlf_builder_with_sfr must build an XLF whose normalisation is
    xi * SFR (the galaxy's own SFR), matching how the forward model drew it."""
    SFR = 7.0
    build = make_hmxb_xlf_builder_with_sfr(SFR=SFR, L_cut=L_CUT, L_min=L_MIN,
                                           L_ref=L_REF, band=BAND)
    xlf = build([np.log10(1.49), 1.6])
    assert xlf.xi == pytest.approx(1.49)
    assert xlf.SFR == pytest.approx(SFR)
    # effective expected number scales with xi*SFR
    ref = HMXBXLF(xi=1.49, gamma=1.6, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF,
                  SFR=SFR, band=BAND)
    assert xlf.expected_number() == pytest.approx(ref.expected_number())


# ---------------------------------------------------------------------------
# 3) tiny 5-galaxy seeded survey fit recovers truth within 2 sigma
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_five_galaxy_fit_recovers_truth():
    """A 5-galaxy joint fit (fast UltraNest settings) recovers the true gamma
    (and per-SFR log10 xi) within 2 sigma. Seeded + bright-ish galaxies so the
    stack is well-constrained and the test is fast and deterministic."""
    gamma_true, xi_true = 1.6, 1.49
    log_xi_true = np.log10(xi_true)
    # bright-ish, nearby galaxies so the 5-galaxy stack constrains gamma well
    setups = [(20.0, 3.0), (15.0, 4.0), (10.0, 4.0), (25.0, 5.0), (12.0, 3.5)]
    gals = [_simulate_galaxy(sfr=s, dist=d, seed=500 + i)
            for i, (s, d) in enumerate(setups)]
    total = sum(g.n_det for g in gals)
    assert total >= 20, f"need enough detections to constrain the slope (got {total})"

    params = _params()
    joint = build_joint_likelihood(
        gals, params, L_cut=L_CUT, L_min=L_MIN, L_ref=L_REF, band=BAND,
        grid=LogLGrid(points_per_dex=120),
    )
    cfg = {"inference": {"sampler": {"n_live": 200}}}
    fr = fit_xlf(joint, cfg, sampler="ultranest", seed=7)

    s = fr.summary()
    g_med = s["gamma"]["median"]
    g_sig = 0.5 * (s["gamma"]["q84"] - s["gamma"]["q16"])
    x_med = s["log10_xi"]["median"]
    x_sig = 0.5 * (s["log10_xi"]["q84"] - s["log10_xi"]["q16"])

    assert abs(g_med - gamma_true) <= 2.0 * g_sig, (
        f"gamma {g_med:.3f}+/-{g_sig:.3f} not within 2 sigma of {gamma_true}")
    assert abs(x_med - log_xi_true) <= 2.0 * x_sig, (
        f"log10_xi {x_med:.3f}+/-{x_sig:.3f} not within 2 sigma of {log_xi_true}")
