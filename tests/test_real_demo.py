"""Tests for the real-galaxy demonstration (optional extension).

These exercise the catalogue parser, the selection-threshold logic, and the
config plumbing -- all on a FROZEN fixture extract committed under
``tests/fixtures/`` (the two demo galaxies + a small NGC 2403 slice, ~38 kB).
No network access and no astroquery import are needed: the fixture stands in for
the full SFGALHMXB download.

The fixture preserves the key, citable cross-check used throughout the demo:
the per-galaxy ``source_flag == 1`` count reproduces M12 Table 1's ``N_XRB``.
"""

import os

import numpy as np
import pytest
import yaml

from xlf_model.real_data import (
    M12_GLOBAL_GAMMA,
    M12_PERGALAXY_GAMMA_MEAN,
    M12_PERGALAXY_GAMMA_RMS,
    M12_TABLE1,
    completeness_anchor_shift_dex,
    load_catalog_csv,
    parse_galaxy,
    threshold_luminosity,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE = os.path.join(_REPO, "tests", "fixtures", "sfgalhmxb_fixture.csv")
_CONFIG = os.path.join(_REPO, "configs", "real_demo.yaml")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog_csv(_FIXTURE)


@pytest.fixture(scope="module")
def real_cfg():
    with open(_CONFIG) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------
def test_fixture_exists_and_small():
    assert os.path.exists(_FIXTURE), "frozen fixture must be committed"
    size = os.path.getsize(_FIXTURE)
    assert size < 50_000, f"fixture must be <50 kB (got {size} B)"


def test_fixture_has_demo_galaxies(catalog):
    names = {str(g).strip() for g in catalog["galaxy_name"]}
    assert "NGC 5457" in names
    assert "NGC 4038/39" in names
    # required columns present
    for col in ("galaxy_name", "log_lx", "source_flag"):
        assert col in catalog.colnames


# ---------------------------------------------------------------------------
# Parser: the M12 N_XRB cross-check (source_flag == 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("galaxy", ["NGC 5457", "NGC 4038/39"])
def test_flag1_count_reproduces_M12_NXRB(catalog, galaxy):
    """source_flag == 1 count must equal M12 Table-1 N_XRB (the parse cross-check)."""
    rg = parse_galaxy(catalog, galaxy, flag_keep=(1,))
    assert rg.L_all.size == M12_TABLE1[galaxy].N_XRB


def test_parse_returns_physical_luminosities(catalog):
    rg = parse_galaxy(catalog, "NGC 5457", flag_keep=(1,))
    # luminosities are 10**log_lx, all positive, all finite
    assert np.all(rg.L_all > 0)
    assert np.all(np.isfinite(rg.L_all))
    # log_lx_all and L_all are consistent
    np.testing.assert_allclose(rg.L_all, 10.0 ** rg.log_lx_all, rtol=1e-12)


def test_flag_keep_none_keeps_all(catalog):
    rg_all = parse_galaxy(catalog, "NGC 5457", flag_keep=None)
    rg_f1 = parse_galaxy(catalog, "NGC 5457", flag_keep=(1,))
    # the unrestricted parse keeps strictly more sources (outer/CXB/bulge too)
    assert rg_all.L_all.size > rg_f1.L_all.size
    assert rg_all.n_total_catalog == rg_all.L_all.size  # no spatial cut applied


def test_unknown_galaxy_raises(catalog):
    with pytest.raises((KeyError, ValueError)):
        parse_galaxy(catalog, "NGC 9999")


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------
def test_threshold_luminosity_factor():
    log_Llim = 36.36  # NGC 5457
    L_thr = threshold_luminosity(log_Llim, factor=1.5)
    assert L_thr == pytest.approx(1.5 * 10.0 ** log_Llim)


def test_threshold_filters_fit_sample(catalog):
    rg = parse_galaxy(catalog, "NGC 5457", flag_keep=(1,), threshold_factor=1.5)
    # every fitted source is above the threshold; threshold above L_lim
    assert np.all(rg.L_fit >= rg.L_threshold)
    assert rg.L_threshold > 10.0 ** rg.meta.log_Llim
    # fit sample is a (proper) subset of all flag-1 sources
    assert rg.n_fit <= rg.L_all.size
    assert rg.n_fit > 0


def test_higher_threshold_keeps_fewer(catalog):
    rg15 = parse_galaxy(catalog, "NGC 5457", flag_keep=(1,), threshold_factor=1.5)
    rg30 = parse_galaxy(catalog, "NGC 5457", flag_keep=(1,), threshold_factor=3.0)
    assert rg30.n_fit <= rg15.n_fit
    assert rg30.L_threshold > rg15.L_threshold


# ---------------------------------------------------------------------------
# Completeness anchoring math
# ---------------------------------------------------------------------------
def test_completeness_anchor_shift_at_half_is_zero():
    # at K = 0.5 the erf C=0.5 point IS the limit -> zero shift
    assert completeness_anchor_shift_dex(0.2, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_completeness_anchor_reproduces_K06():
    """The anchored erf completeness must equal 0.6 exactly at the quoted L_lim."""
    from xlf_model.forward import completeness_erf, luminosity_to_flux

    width = 0.2
    meta = M12_TABLE1["NGC 5457"]
    L_lim = 10.0 ** meta.log_Llim
    F_lim = float(luminosity_to_flux(L_lim, meta.distance_Mpc))
    d = completeness_anchor_shift_dex(width, 0.6)
    F50 = F_lim / 10.0 ** d
    c = float(completeness_erf(F_lim, F50, width))
    assert c == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def test_config_loads_and_has_required_keys(real_cfg):
    assert "real" in real_cfg
    r = real_cfg["real"]
    for key in ("galaxies", "flag_keep", "threshold_factor",
                "completeness_k_anchor", "completeness_width_dex"):
        assert key in r, f"real_demo.yaml missing real.{key}"
    # the chosen galaxies must all have M12 metadata
    for gal in r["galaxies"]:
        assert gal in M12_TABLE1, f"{gal} lacks M12 Table-1 metadata"


def test_config_galaxies_parseable(catalog, real_cfg):
    """Every galaxy named in the config parses against the fixture/full catalogue."""
    r = real_cfg["real"]
    flag_keep = tuple(r["flag_keep"])
    for gal in r["galaxies"]:
        rg = parse_galaxy(catalog, gal, flag_keep=flag_keep,
                          threshold_factor=float(r["threshold_factor"]))
        assert rg.n_fit > 5, f"{gal}: need >5 sources above threshold (M12 criterion)"


def test_m12_reference_constants_sane():
    # guard the published reference values we compare against
    assert M12_GLOBAL_GAMMA == pytest.approx(1.60)
    assert M12_PERGALAXY_GAMMA_MEAN == pytest.approx(1.59)
    assert M12_PERGALAXY_GAMMA_RMS == pytest.approx(0.25)
