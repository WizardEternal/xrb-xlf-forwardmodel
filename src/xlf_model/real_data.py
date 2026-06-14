"""Real-galaxy demonstration: SFGALHMXB catalog access + selection model.

OPTIONAL EXTENSION (clearly labeled "demonstration on real data").  This module
supports the selection-aware HMXB-slope fit on 1-2 real public Chandra
point-source catalogues from the Mineo, Gilfanov & Sunyaev 2012 (M12) sample.
It is a *consistency check* against M12's published slopes, NOT a re-measurement
or a survey.

What this module does
---------------------
1. **Data access** (:func:`fetch_sfgalhmxb`): resumable/cached download of the
   public HEASARC table **SFGALHMXB** -- the M12 1055-source Chandra catalogue
   (``heasarc.gsfc.nasa.gov/w3browse/all/sfgalhmxb.html``) -- via
   ``astroquery.heasarc`` TAP, cached to ``data/real/`` (gitignored).  If the
   cache exists it is reused; the network is only touched on a cold cache.
   ``astroquery`` is an **extension-only dependency** (see
   ``requirements-extension.txt``); the core repo does not need it.

2. **Per-galaxy metadata** (:data:`M12_TABLE1`): distance + SFR + sensitivity
   limit + ``N_XRB`` transcribed from **M12 Table 1** (Mineo et al. 2012, MNRAS
   419, 2095; arXiv:1105.4610), p. 3.  Each entry cites the table.

3. **Catalogue parsing** (:func:`parse_galaxy`): select one galaxy's sources,
   optionally restrict to the HMXB-dominated spatial region
   (``source_flag == 1``; see "Selection" below), return the source
   luminosities (erg/s) ready for the fit machinery.

4. **Selection approximation** (:func:`completeness_anchor_shift_dex`,
   :func:`threshold_luminosity`): the documented erf-ramp completeness around the
   galaxy's quoted sensitivity limit, and the conservative fit threshold.

Selection function (caveat)
---------------------------
M12's per-galaxy incompleteness correction used the **Voss & Gilfanov 2006**
simulations (their function ``K(L)``), which we **cannot reproduce** here.  We
adopt a documented *approximation*:

* **Anchor.** M12 Table 1 tabulates, per galaxy, ``log L_lim`` -- the sensitivity
  limit, defined (Table 1 footnote f) as the luminosity where the incompleteness
  function ``K(L) = 0.6``.  This is the ONLY per-galaxy completeness luminosity
  M12 tabulate.  (M12 also *define* a "completeness luminosity" ``L_comp`` at
  ``K = 0.8`` in Section 4.3, but they do **not** tabulate it per galaxy.)  We
  therefore anchor our smooth erf completeness ramp so that completeness = 0.6 at
  ``L_lim`` -- i.e. we reuse the repo's :func:`forward.completeness_erf` but shift
  it so the quoted ``L_lim`` is the ``C = 0.6`` point rather than the ``C = 0.5``
  point.  The ramp width is a documented free choice (``width_dex``), set to the
  forward model's default; the result is insensitive to it because we fit only
  well above the limit (next bullet).

* **Spatial selection.** The SFGALHMXB ``source_flag`` (values 1/2/3) encodes the
  spatial region of M12 Fig. 2: ``1`` = inner HMXB-dominated region (the sources
  M12 used for the per-galaxy XLF -- the per-galaxy ``source_flag == 1`` count
  reproduces M12 Table 1's ``N_XRB`` exactly), ``2`` = outer region where CXB
  contamination is significant, ``3`` = excluded bulge.  By default we keep
  ``source_flag == 1`` (matching M12's XLF sample).

* **Conservative threshold.** M12 fit each galaxy's XLF only above
  ``L_th = 1.5 x L_comp`` (the 80%-completeness limit; M12 Section 7.4).  We do
  not have per-galaxy ``L_comp``, so we apply the same ``1.5x`` factor to the
  tabulated ``L_lim`` (``K = 0.6``).  Because ``L_comp >= L_lim``, our
  ``1.5 x L_lim`` is a *slightly more permissive* threshold than M12's; fitting
  above it keeps us in the regime where the exact completeness shape barely
  matters, which is the whole point of the conservative cut.

* **What this approximation cannot do.** It does not reproduce the true ``K(L)``
  shape (Voss & Gilfanov simulations), and it does not subtract CXB
  contamination per source (M12 modeled CXB statistically, as a fixed model
  component, in their ML fit).  The fit lives in the luminosity regime
  (log L ~ 36.5-38.5) where M12 found CXB to be a significant fraction of the
  point sources by number, and the HMXB-only likelihood here carries no CXB
  term.  The recovered absolute slopes are therefore not interpretable as a
  clean selection effect; this module demonstrates that the machinery runs
  end-to-end on real Chandra data, and the limitations are quantified in
  RESULTS.md and the README.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "M12_GALAXY",
    "M12_TABLE1",
    "fetch_sfgalhmxb",
    "load_catalog_csv",
    "parse_galaxy",
    "completeness_anchor_shift_dex",
    "threshold_luminosity",
    "RealGalaxy",
]


# ---------------------------------------------------------------------------
# Mineo+12 Table 1 per-galaxy metadata (transcribed; each value cited)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class M12_GALAXY:
    """Per-galaxy metadata from Mineo et al. 2012 (M12) Table 1, p. 3.

    Attributes
    ----------
    name : str
        Galaxy name as it appears in the SFGALHMXB ``galaxy_name`` column.
    distance_Mpc : float
        Redshift-independent distance, M12 Table 1 column ``D`` (Mpc).
    SFR : float
        Star-formation rate, M12 Table 1 column ``SFR`` (Msun/yr); Spitzer+GALEX.
    log_Llim : float
        Sensitivity limit ``log10 L_lim`` (erg/s), M12 Table 1 column ``log Llim``;
        defined (Table 1 footnote f) as where the incompleteness ``K(L) = 0.6``.
    N_XRB : int
        M12 Table 1 column ``N_XRB``: number of XRBs above the sensitivity limit
        in the HMXB-dominated region.  Reproduced exactly by the catalogue's
        ``source_flag == 1`` count -- the cross-check that ties our parse to M12.
    label : str
        Human label (common name) for plots/text.
    """

    name: str
    distance_Mpc: float
    SFR: float
    log_Llim: float
    N_XRB: int
    label: str = ""


# Two richest galaxies by N_XRB (= source_flag==1 count) with SFR + distance in
# M12 Table 1.  Choice rule (documented): "the two galaxies with the most
# catalogued HMXB-region sources (source_flag == 1, identical to M12 Table 1
# N_XRB) and SFR + distance quoted in M12 Table 1."  All values: M12 Table 1, p.3.
M12_TABLE1: dict[str, M12_GALAXY] = {
    "NGC 5457": M12_GALAXY(
        name="NGC 5457", distance_Mpc=6.7, SFR=1.5, log_Llim=36.36,
        N_XRB=96, label="NGC 5457 (M101)",
    ),
    "NGC 4038/39": M12_GALAXY(
        name="NGC 4038/39", distance_Mpc=13.8, SFR=5.4, log_Llim=36.92,
        N_XRB=83, label="NGC 4038/39 (the Antennae)",
    ),
    # A few more transcribed for convenience / alternative selections (all M12 T1):
    "NGC 5194": M12_GALAXY(
        name="NGC 5194", distance_Mpc=7.6, SFR=3.7, log_Llim=37.05,
        N_XRB=69, label="NGC 5194 (M51A)",
    ),
    "NGC 2403": M12_GALAXY(
        name="NGC 2403", distance_Mpc=3.1, SFR=0.52, log_Llim=36.16,
        N_XRB=42, label="NGC 2403",
    ),
}

# M12 global / per-galaxy slope results, for the consistency check (M12 Sec 7).
M12_GLOBAL_GAMMA = 1.60          # M12 single-PL global slope (gamma = 1.60 +/- 0.02; Sec 7.2)
M12_GLOBAL_GAMMA_ERR = 0.02
M12_PERGALAXY_GAMMA_MEAN = 1.59  # M12 Sec 7.4: <gamma> = 1.59 over individual galaxies
M12_PERGALAXY_GAMMA_RMS = 0.25   # M12 Sec 7.4: rms scatter of per-galaxy gamma


# ---------------------------------------------------------------------------
# Data access: resumable/cached SFGALHMXB fetch
# ---------------------------------------------------------------------------
# Column subset we keep (the SFGALHMXB schema, verified via TAP_SCHEMA):
SFGALHMXB_COLUMNS = (
    "galaxy_name", "galaxy_source_number", "name", "ra", "dec",
    "counts", "counts_error", "log_lx", "log_flux", "source_flag", "lii", "bii",
)


def _default_cache_path(repo_root: str | None = None) -> str:
    """Default cache CSV path: ``<repo>/data/real/sfgalhmxb_full.csv``."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "data", "real", "sfgalhmxb_full.csv")


def fetch_sfgalhmxb(cache_path: str | None = None, *, force: bool = False) -> str:
    """Fetch the SFGALHMXB catalogue to a CSV cache (resumable / skip-if-exists).

    Parameters
    ----------
    cache_path : str, optional
        Destination CSV.  Defaults to ``data/real/sfgalhmxb_full.csv``.
    force : bool
        Re-download even if the cache exists.

    Returns
    -------
    str
        The path to the cached CSV.

    Notes
    -----
    Uses ``astroquery.heasarc`` TAP (sync).  The query selects all rows of the
    public table; the result is ~1055 rows / ~100 kB so a sync query is fine.
    The cache is the data product downstream code reads -- the network is touched
    only when the cache is missing (or ``force=True``).
    """
    cache_path = cache_path or _default_cache_path()
    if os.path.exists(cache_path) and not force:
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Import here so astroquery stays an extension-only dependency.
    from astroquery.heasarc import Heasarc

    h = Heasarc()
    cols = ", ".join(SFGALHMXB_COLUMNS)
    query = f"SELECT {cols} FROM sfgalhmxb"
    table = h.tap.run_sync(query).to_table()
    table.write(cache_path, format="csv", overwrite=True)
    return cache_path


def load_catalog_csv(path: str):
    """Load a cached SFGALHMXB CSV into an astropy Table.

    Kept separate from :func:`fetch_sfgalhmxb` so tests can load a frozen fixture
    without any network access or astroquery import.
    """
    from astropy.table import Table

    return Table.read(path, format="csv")


# ---------------------------------------------------------------------------
# Selection approximation
# ---------------------------------------------------------------------------
# The forward model's erf completeness is C(F) = 0.5*(1+erf(log10(F/F50)/(sqrt2 w))).
# We anchor it so completeness = 0.6 at the *quoted* L_lim (M12's K=0.6 point),
# not 0.5.  Equivalently we place the C=0.5 flux limit slightly BELOW L_lim, by a
# fixed offset in dex that we compute once from the inverse erf.
from scipy.special import erfinv  # noqa: E402


def completeness_anchor_shift_dex(width_dex: float, k_anchor: float = 0.6) -> float:
    """Offset (dex) to move the erf C=0.5 point below the quoted ``K=k_anchor`` limit.

    The erf ramp has ``C(F) = 0.5*(1 + erf(d / (sqrt2 * width_dex)))`` with
    ``d = log10(F/F50)``.  We want ``C = k_anchor`` at ``F = F(L_lim)``.  Solving,
    ``d_anchor = sqrt2 * width_dex * erfinv(2*k_anchor - 1)``.  So
    ``log10 F50 = log10 F(L_lim) - d_anchor``: F50 sits ``d_anchor`` dex below the
    flux of the quoted limit.  Returns ``d_anchor`` (>= 0 for ``k_anchor >= 0.5``).
    """
    return float(np.sqrt(2.0) * float(width_dex) * erfinv(2.0 * float(k_anchor) - 1.0))


def threshold_luminosity(log_Llim: float, factor: float = 1.5) -> float:
    """Conservative fit threshold luminosity (erg/s) = ``factor * 10**log_Llim``.

    M12 (Section 7.4) used ``L_th = 1.5 x L_comp`` (the 80%-completeness limit).
    We apply the same factor to the tabulated ``L_lim`` (the K=0.6 limit), which
    is the only per-galaxy completeness luminosity M12 tabulate -- see the module
    docstring for the documented caveat.
    """
    return float(factor) * 10.0 ** float(log_Llim)


# ---------------------------------------------------------------------------
# Per-galaxy parse
# ---------------------------------------------------------------------------
@dataclass
class RealGalaxy:
    """One parsed real galaxy: metadata + selected source luminosities.

    Attributes
    ----------
    meta : M12_GALAXY
        The M12 Table-1 metadata.
    L_all : ndarray
        Luminosities (erg/s) of ALL catalogued sources for this galaxy after the
        spatial cut (``source_flag`` filter), before the luminosity threshold.
    L_fit : ndarray
        Luminosities (erg/s) above the conservative threshold -- the data handed
        to the fit.
    log_lx_all : ndarray
        ``log10 L`` of ``L_all`` (handy for plots).
    source_flag : ndarray
        The ``source_flag`` values kept (after the spatial cut), aligned to L_all.
    L_threshold : float
        The conservative threshold luminosity used (erg/s).
    n_total_catalog : int
        Number of catalogued sources for this galaxy BEFORE any cut.
    """

    meta: M12_GALAXY
    L_all: np.ndarray
    L_fit: np.ndarray
    log_lx_all: np.ndarray
    source_flag: np.ndarray
    L_threshold: float
    n_total_catalog: int

    @property
    def n_fit(self) -> int:
        return int(self.L_fit.size)


def parse_galaxy(
    catalog,
    galaxy: str,
    *,
    meta: M12_GALAXY | None = None,
    flag_keep=(1,),
    threshold_factor: float = 1.5,
) -> RealGalaxy:
    """Extract one galaxy's HMXB source luminosities and apply the selection.

    Parameters
    ----------
    catalog : astropy.table.Table
        The loaded SFGALHMXB table (from :func:`load_catalog_csv`).
    galaxy : str
        Galaxy name (``galaxy_name`` value), e.g. ``"NGC 5457"``.
    meta : M12_GALAXY, optional
        Table-1 metadata; defaults to ``M12_TABLE1[galaxy]``.
    flag_keep : tuple of int or None
        ``source_flag`` values to keep (spatial cut).  Default ``(1,)`` = the
        HMXB-dominated region (matches M12's XLF sample).  ``None`` keeps all.
    threshold_factor : float
        Multiple of ``L_lim`` for the conservative fit threshold (default 1.5,
        mirroring M12's ``L_th`` choice).

    Returns
    -------
    RealGalaxy
    """
    if meta is None:
        if galaxy not in M12_TABLE1:
            raise KeyError(f"no M12 Table-1 metadata for galaxy {galaxy!r}")
        meta = M12_TABLE1[galaxy]

    names = np.array([str(g).strip() for g in catalog["galaxy_name"]])
    sel = names == galaxy.strip()
    n_total = int(np.count_nonzero(sel))
    if n_total == 0:
        raise ValueError(f"galaxy {galaxy!r} not found in catalogue")

    log_lx = np.asarray(catalog["log_lx"][sel], dtype=float)
    flags = np.asarray(catalog["source_flag"][sel])

    # spatial cut
    if flag_keep is not None:
        keep = np.isin(flags, np.asarray(flag_keep))
        log_lx = log_lx[keep]
        flags = flags[keep]

    # drop any non-finite luminosities defensively
    finite = np.isfinite(log_lx)
    log_lx = log_lx[finite]
    flags = flags[finite]

    L_all = 10.0 ** log_lx
    L_thr = threshold_luminosity(meta.log_Llim, factor=threshold_factor)
    L_fit = L_all[L_all >= L_thr]

    return RealGalaxy(
        meta=meta,
        L_all=L_all,
        L_fit=L_fit,
        log_lx_all=log_lx,
        source_flag=flags,
        L_threshold=L_thr,
        n_total_catalog=n_total,
    )
