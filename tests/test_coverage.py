"""Tests for the Phase-4 posterior-coverage machinery.

Covers the pure pieces of the coverage run + analysis (no sampler needed):
  * the central-credibility-level -> quantile mapping  c -> [(1-c)/2, (1+c)/2];
  * the Wilson binomial interval (hand value + monotone shrink with n);
  * the empirical-coverage counter on synthetic rows with KNOWN containment
    (a Gaussian-posterior toy yields ~diagonal coverage);
  * skipped (too-few-source) rows are excluded from the coverage denominator;
  * the run_coverage resume logic: re-running skips completed realizations and a
    truncated final JSONL line is tolerated.

Everything here is fast (no fits): the analysis/quantile helpers are imported
from the two scripts loaded as modules.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load_module(name, rel):
    path = os.path.join(_REPO, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RUN = _load_module("run_coverage", os.path.join("scripts", "run_coverage.py"))
AN = _load_module("analyze_coverage", os.path.join("scripts", "analyze_coverage.py"))


# ---------------------------------------------------------------------------
# central-level -> quantile mapping
# ---------------------------------------------------------------------------
def test_central_levels_to_quantiles():
    levels = [0.1, 0.5, 0.9]
    qlo, qhi = RUN._central_levels_to_quantiles(levels)
    # a central interval at level c spans [(1-c)/2, (1+c)/2]
    assert qlo == pytest.approx([0.45, 0.25, 0.05])
    assert qhi == pytest.approx([0.55, 0.75, 0.95])
    # the interval width equals the credibility level
    assert (qhi - qlo) == pytest.approx(levels)
    # symmetry about the median
    assert (qlo + qhi) == pytest.approx([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------
def test_wilson_interval_hand_value():
    # k=50 of n=100, z=1: centre ~0.5, half-width ~ z*sqrt(p(1-p)/n) corrected
    lo, hi = AN.wilson_interval(50, 100, z=1.0)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # symmetric about 0.5 for p=0.5
    assert (lo + hi) == pytest.approx(1.0, abs=1e-9)
    # width is close to the normal approx 2*0.05 = 0.1 but slightly wider
    assert (hi - lo) == pytest.approx(0.0995, abs=0.01)


def test_wilson_shrinks_with_n():
    w_small = AN.wilson_interval(20, 40)[1] - AN.wilson_interval(20, 40)[0]
    w_large = AN.wilson_interval(200, 400)[1] - AN.wilson_interval(200, 400)[0]
    assert w_large < w_small            # more trials -> tighter band


def test_wilson_well_behaved_at_extremes():
    # p=0 and p=1 must give finite, in-[0,1] bounds (the reason for Wilson)
    for k, n in ((0, 100), (100, 100)):
        lo, hi = AN.wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# empirical coverage on synthetic rows with KNOWN containment
# ---------------------------------------------------------------------------
def _make_gaussian_rows(n, levels, seed=0):
    """Synthetic coverage rows from a perfectly-calibrated Gaussian posterior.

    For each realization: truth ~ N(0,1) draw; the 'posterior' is N(truth, 1),
    so the central interval at level c is truth +/- z_c (z_c the half-Gaussian
    quantile).  By construction the empirical coverage must be ~ nominal.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    qlo, qhi = RUN._central_levels_to_quantiles(levels)
    rows = []
    for r in range(n):
        truth_g = float(rng.standard_normal())
        truth_x = float(rng.standard_normal())
        # posterior mean offset from truth by a standard-normal step (calibrated)
        mu_g = truth_g + float(rng.standard_normal())
        mu_x = truth_x + float(rng.standard_normal())
        g_lo = (mu_g + norm.ppf(qlo)).tolist()
        g_hi = (mu_g + norm.ppf(qhi)).tolist()
        x_lo = (mu_x + norm.ppf(qlo)).tolist()
        x_hi = (mu_x + norm.ppf(qhi)).tolist()
        rows.append({
            "status": "ok",
            "gamma_true": truth_g, "log10_xi_eff_true": truth_x,
            "gamma_ci_lo": g_lo, "gamma_ci_hi": g_hi,
            "log_xi_ci_lo": x_lo, "log_xi_ci_hi": x_hi,
        })
    return rows


def test_empirical_coverage_diagonal_for_calibrated_toy():
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = _make_gaussian_rows(4000, levels, seed=7)
    cov = AN.empirical_coverage(rows, levels)
    for pname in ("gamma", "log10_xi_eff"):
        emp = cov[pname]["emp"]
        # a calibrated posterior gives empirical ~ nominal (within sampling noise)
        assert np.max(np.abs(emp - np.asarray(levels))) < 0.04
        assert cov[pname]["n"] == 4000


def test_skipped_rows_excluded_from_denominator():
    levels = [0.5]
    rows = _make_gaussian_rows(100, levels, seed=1)
    # add 10 skipped rows that must NOT count toward n
    for _ in range(10):
        rows.append({"status": "skipped_too_few"})
    cov = AN.empirical_coverage(rows, levels)
    assert cov["gamma"]["n"] == 100          # only the 100 ok rows


def test_containment_counter_exact():
    """Hand-built rows: truth inside one level's interval, outside another."""
    levels = [0.5, 0.9]
    # gamma truth = 1.6; level-0.5 interval [1.5, 1.7] contains it,
    # level-0.9 interval [2.0, 2.2] does NOT.
    rows = [{
        "status": "ok",
        "gamma_true": 1.6, "log10_xi_eff_true": 0.0,
        "gamma_ci_lo": [1.5, 2.0], "gamma_ci_hi": [1.7, 2.2],
        "log_xi_ci_lo": [-0.1, -0.2], "log_xi_ci_hi": [0.1, 0.2],
    }]
    cov = AN.empirical_coverage(rows, levels)
    assert cov["gamma"]["k"].tolist() == [1, 0]      # covered at 0.5, not 0.9
    assert cov["log10_xi_eff"]["k"].tolist() == [1, 1]


# ---------------------------------------------------------------------------
# resume logic
# ---------------------------------------------------------------------------
def test_load_done_realizations_and_resume(tmp_path):
    jsonl = tmp_path / "coverage_results.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"realization": 0, "status": "ok"}) + "\n")
        fh.write(json.dumps({"realization": 1, "status": "ok"}) + "\n")
    done = RUN.load_done_realizations(str(jsonl))
    assert done == {0, 1}


def test_load_done_tolerates_truncated_last_line(tmp_path):
    jsonl = tmp_path / "coverage_results.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"realization": 0, "status": "ok"}) + "\n")
        fh.write('{"realization": 1, "stat')        # truncated by an interruption
    done = RUN.load_done_realizations(str(jsonl))
    assert done == {0}                                # partial line dropped


def test_build_specs_independent_seeds():
    import yaml

    with open(os.path.join(_REPO, "configs", "coverage.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    specs, SFR, lam_u = RUN.build_specs(cfg, n_realizations=5, sampler="ultranest")
    assert len(specs) == 5
    # all selection-aware, all share the tuned SFR, distinct sim seeds
    assert all(s.gamma == cfg["coverage"]["gamma_true"] for s in specs)
    assert len({s.sim_seed for s in specs}) == 5
    assert SFR > 0 and lam_u > 0
    # the coverage seed namespace is offset from the recovery suite (7e6) so the
    # draws are independent of the suite's realizations
    assert all(s.sim_seed >= int(cfg["seed_base"]) + 7_000_000 for s in specs)
