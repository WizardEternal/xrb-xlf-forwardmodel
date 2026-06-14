"""X-ray binary luminosity functions (XLFs).

Pure, vectorized implementations of the HMXB and LMXB luminosity functions used
by the forward model, with:

  * the differential XLF dN/dL,
  * the analytic cumulative N(>L),
  * the analytic expected total number over a luminosity range,
  * analytic (piecewise) inverse-CDF sampling.

All default parameters are verified against the primary papers. See
``configs/xlf_defaults.yaml`` for the cited values.

Conventions
-----------
* The HMXB XLF is written in the dimensionless luminosity ``L38 = L / 1e38``,
  following Grimm/Mineo:  dN/dL38 = xi * SFR * L38^(-gamma).  Internally we
  expose ``dN/dL`` (per erg/s) as well so both XLFs share one interface.
* Each XLF is a truncated power law / broken power law on a finite support
  ``[L_min, L_cut]``.  The high-L cutoff is implemented as a SHARP truncation
  (zero above ``L_cut``).  This keeps every quantity -- the integral, N(>L) and
  the inverse CDF -- exactly analytic and therefore exactly testable.  For the
  default parameters the cutoff sits orders of magnitude above the luminosities
  of interest, so the truncation has negligible effect on validation numbers.
* "Normalization" (``xi*SFR`` for HMXBs, ``K1*Mstar/Mref`` for LMXBs) multiplies
  the *shape* and sets the expected total count.  Sampling uses only the shape;
  the count is drawn separately (see ``forward.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "powerlaw_integral",
    "powerlaw_cumulative_above",
    "HMXBXLF",
    "LMXBXLF",
    "hmxb_from_config",
    "lmxb_from_config",
]


# ---------------------------------------------------------------------------
# Low-level analytic helpers for a single (untruncated-normalization) power law
# ---------------------------------------------------------------------------
def powerlaw_integral(a: float, b: float, slope: float) -> float:
    """Integral of ``x^(-slope)`` from ``a`` to ``b`` (a <= b, a > 0).

    Handles the logarithmic special case ``slope == 1`` analytically.
    """
    a = float(a)
    b = float(b)
    if a <= 0.0:
        raise ValueError("lower limit must be positive")
    if b < a:
        return 0.0
    if np.isclose(slope, 1.0):
        return float(np.log(b / a))
    p = 1.0 - slope
    return float((b**p - a**p) / p)


def powerlaw_cumulative_above(L, a: float, b: float, slope: float) -> np.ndarray:
    """Unnormalized number above ``L`` for a power law ``x^(-slope)`` on [a, b].

    Returns ``integral_{max(L,a)}^{b} x^(-slope) dx`` (clipped: 0 above b, full
    integral below a).  Vectorized over ``L``.
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    out = np.empty_like(L)
    lo = np.clip(L, a, b)
    if np.isclose(slope, 1.0):
        out = np.log(b / lo)
    else:
        p = 1.0 - slope
        out = (b**p - lo**p) / p
    out = np.where(L > b, 0.0, out)
    return out


# ---------------------------------------------------------------------------
# HMXB XLF -- single (truncated) power law
# ---------------------------------------------------------------------------
@dataclass
class HMXBXLF:
    """HMXB luminosity function: single power law with a high-L cutoff.

    Differential form (Grimm/Mineo convention, in L38 = L / L_ref):

        dN/dL38 = xi * SFR * L38^(-gamma)        for L_min <= L <= L_cut

    Parameters
    ----------
    xi : float
        Normalization per unit SFR (Msun/yr), in the dN/dL38 convention.
    gamma : float
        Differential power-law slope.
    L_cut : float
        High-L sharp truncation, erg/s.
    L_min : float
        Lower luminosity bound of the support, erg/s.
    L_ref : float
        Reference luminosity defining L38 (1e38 erg/s by default).
    SFR : float
        Star-formation rate, Msun/yr.  Scales the normalization linearly.
    band : str
        Bookkeeping label for the energy band (informational).
    """

    xi: float = 1.49
    gamma: float = 1.60
    L_cut: float = 2.1e40
    L_min: float = 1.0e35
    L_ref: float = 1.0e38
    SFR: float = 1.0
    band: str = "0.5-8 keV"

    # ----- differential -----
    def dN_dL38(self, L38) -> np.ndarray:
        """dN/dL38 at dimensionless luminosity L38 (zero outside support)."""
        L38 = np.asarray(L38, dtype=float)
        val = self.xi * self.SFR * np.power(L38, -self.gamma)
        lo38 = self.L_min / self.L_ref
        hi38 = self.L_cut / self.L_ref
        return np.where((L38 >= lo38) & (L38 <= hi38), val, 0.0)

    def dN_dL(self, L) -> np.ndarray:
        """dN/dL per erg/s (= dN/dL38 / L_ref), zero outside support."""
        L = np.asarray(L, dtype=float)
        return self.dN_dL38(L / self.L_ref) / self.L_ref

    # ----- normalization / counts -----
    @property
    def _norm(self) -> float:
        """Shape normalization in L38 units: A = xi * SFR."""
        return self.xi * self.SFR

    def expected_number(self, L_lo=None, L_hi=None) -> float:
        """Expected total number of sources with L in [L_lo, L_hi].

        Defaults: L_lo = L_min, L_hi = L_cut.  Computed analytically in L38.
        """
        L_lo = self.L_min if L_lo is None else float(L_lo)
        L_hi = self.L_cut if L_hi is None else float(L_hi)
        L_lo = max(L_lo, self.L_min)
        L_hi = min(L_hi, self.L_cut)
        if L_hi <= L_lo:
            return 0.0
        a = L_lo / self.L_ref
        b = L_hi / self.L_ref
        return self._norm * powerlaw_integral(a, b, self.gamma)

    def N_gt(self, L) -> np.ndarray:
        """Cumulative N(>L): expected number with luminosity above L."""
        L = np.atleast_1d(np.asarray(L, dtype=float))
        a = self.L_min / self.L_ref
        b = self.L_cut / self.L_ref
        L38 = L / self.L_ref
        return self._norm * powerlaw_cumulative_above(L38, a, b, self.gamma)

    # ----- sampling -----
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` luminosities (erg/s) via the analytic inverse CDF.

        The CDF over the support [L_min, L_cut] is the normalized integral of
        x^(-gamma); inverting it analytically gives the samples.
        """
        n = int(n)
        if n == 0:
            return np.empty(0, dtype=float)
        a = self.L_min / self.L_ref
        b = self.L_cut / self.L_ref
        u = rng.random(n)
        L38 = _inverse_cdf_powerlaw(u, a, b, self.gamma)
        return L38 * self.L_ref


def _inverse_cdf_powerlaw(u, a: float, b: float, slope: float) -> np.ndarray:
    """Analytic inverse CDF of a power law x^(-slope) truncated to [a, b].

    The CDF is F(x) = I(a, x) / I(a, b) with I the power-law integral.
    Inverting:
        slope != 1:  x = ( a^p + u * (b^p - a^p) )^(1/p),  p = 1 - slope
        slope == 1:  x = a * (b/a)^u
    """
    u = np.asarray(u, dtype=float)
    if np.isclose(slope, 1.0):
        return a * np.power(b / a, u)
    p = 1.0 - slope
    ap = a**p
    bp = b**p
    return np.power(ap + u * (bp - ap), 1.0 / p)


# ---------------------------------------------------------------------------
# LMXB XLF -- two-break (three-segment) broken power law
# ---------------------------------------------------------------------------
@dataclass
class LMXBXLF:
    """LMXB luminosity function: two-break broken power law, normalized to M*.

    Differential form (Gilfanov 2004), in physical L (erg/s):

        dN/dL ~ L^(-alpha1)             L_min <= L < L_b1
        dN/dL ~ L^(-alpha2)             L_b1  <= L < L_b2
        dN/dL ~ L^(-alpha3)             L_b2  <= L <= L_cut

    with continuity enforced at the two breaks.  The overall scale is set by
    ``K1`` = differential normalization at ``L_b1`` per ``mass_ref`` Msun:
    i.e. dN/dL evaluated just at L_b1 equals (K1 * Mstar / mass_ref) / L_b1
    in the standard G04 dimensionless-at-break convention.

    Normalization convention
    ------------------------
    G04 writes the XLF in units of L_38 = L / 1e38 erg/s (the same unit Grimm
    & Gilfanov use).  ``K1`` is the differential normalization of the middle
    (segment-2) power law anchored at the first break ``L_b1``:

        dN/dL_38 = K1 * (Mstar/mass_ref) * (L / L_b1)^(-alpha2)

    where L_38 = L / 1e38.  Segments 1 and 3 follow by continuity at the two
    breaks.  With the published K1 = 440.4 this yields N(>1e37) = 146.5 per
    1e11 Msun, within 2.5% of G04's independently-quoted N(>1e37) = 142.9 --
    the residual is the internal inconsistency between G04's separately quoted
    K1 (+/-25.9) and N(>1e37) (+/-8.4) values, both within their error bars.
    (Using L_37 = L/1e37 here would overshoot N(>1e37) by ~10x; the 1e38 unit
    is the correct G04 convention.)
    """

    alpha1: float = 1.0
    alpha2: float = 1.86
    alpha3: float = 4.8
    L_b1: float = 1.9e37
    L_b2: float = 5.0e38
    L_cut: float = 5.0e40
    K1: float = 440.4
    mass_ref: float = 1.0e11
    Mstar: float = 1.0e11
    L_min: float = 1.0e35
    L_ref_norm: float = 1.0e38  # G04 expresses the XLF in units of L38 = L/1e38
    band: str = "0.5-8 keV"

    # cached coefficients of dN/dL = C_i * L^(-alpha_i) per segment
    _C: np.ndarray = field(init=False, repr=False)
    _edges: np.ndarray = field(init=False, repr=False)
    _slopes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_coeffs()

    def _build_coeffs(self) -> None:
        """Compute per-segment coefficients C_i of dN/dL = C_i * L^(-alpha_i).

        Anchor: segment 2 normalization is fixed by K1 at L_b1.  The G04
        convention dN/dL_38 = K1 * (Mstar/mass_ref) * (L/L_b1)^(-alpha2), with
        L_38 = L / L_ref_norm (= L/1e38), means

            dN/dL = (1/L_ref_norm) * K1 * (Mstar/mass_ref) * (L/L_b1)^(-alpha2)
                  = C2 * L^(-alpha2),
            C2 = K1 * (Mstar/mass_ref) / L_ref_norm * L_b1^(alpha2).

        Then C1 and C3 follow from continuity at L_b1 and L_b2.
        """
        scale = self.K1 * (self.Mstar / self.mass_ref) / self.L_ref_norm
        C2 = scale * self.L_b1**self.alpha2
        # continuity at L_b1: C1 * L_b1^-a1 = C2 * L_b1^-a2
        C1 = C2 * self.L_b1 ** (self.alpha1 - self.alpha2)
        # continuity at L_b2: C3 * L_b2^-a3 = C2 * L_b2^-a2
        C3 = C2 * self.L_b2 ** (self.alpha3 - self.alpha2)
        self._C = np.array([C1, C2, C3], dtype=float)
        self._slopes = np.array([self.alpha1, self.alpha2, self.alpha3], dtype=float)
        self._edges = np.array(
            [self.L_min, self.L_b1, self.L_b2, self.L_cut], dtype=float
        )

    # ----- differential -----
    def dN_dL(self, L) -> np.ndarray:
        """dN/dL per erg/s; zero outside [L_min, L_cut].

        Preserves input shape (0-d array for scalar input), matching
        :meth:`HMXBXLF.dN_dL`.
        """
        L = np.asarray(L, dtype=float)
        Lf = np.atleast_1d(L)
        out = np.zeros_like(Lf)
        for i in range(3):
            lo, hi = self._edges[i], self._edges[i + 1]
            # include right edge only for the last segment to avoid double-count
            if i < 2:
                m = (Lf >= lo) & (Lf < hi)
            else:
                m = (Lf >= lo) & (Lf <= hi)
            out[m] = self._C[i] * np.power(Lf[m], -self._slopes[i])
        return out.reshape(L.shape)

    # ----- normalization / counts -----
    def expected_number(self, L_lo=None, L_hi=None) -> float:
        """Expected total number with L in [L_lo, L_hi] (default full support)."""
        L_lo = self.L_min if L_lo is None else float(L_lo)
        L_hi = self.L_cut if L_hi is None else float(L_hi)
        L_lo = max(L_lo, self.L_min)
        L_hi = min(L_hi, self.L_cut)
        if L_hi <= L_lo:
            return 0.0
        total = 0.0
        for i in range(3):
            lo = max(L_lo, self._edges[i])
            hi = min(L_hi, self._edges[i + 1])
            if hi <= lo:
                continue
            total += self._C[i] * powerlaw_integral(lo, hi, self._slopes[i])
        return float(total)

    def N_gt(self, L) -> np.ndarray:
        """Cumulative N(>L) for the broken power law (vectorized)."""
        L = np.atleast_1d(np.asarray(L, dtype=float))
        out = np.array([self.expected_number(L_lo=float(x)) for x in L])
        return out

    # ----- sampling -----
    def _segment_weights(self) -> np.ndarray:
        """Expected number in each of the three segments (over full support)."""
        w = np.empty(3, dtype=float)
        for i in range(3):
            lo, hi = self._edges[i], self._edges[i + 1]
            w[i] = self._C[i] * powerlaw_integral(lo, hi, self._slopes[i])
        return w

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` luminosities (erg/s) via the analytic piecewise inverse CDF.

        Procedure: choose a segment with probability proportional to its
        expected count, then sample within the segment with the analytic
        single-power-law inverse CDF.  This is exact (no rejection).
        """
        n = int(n)
        if n == 0:
            return np.empty(0, dtype=float)
        w = self._segment_weights()
        probs = w / w.sum()
        seg = rng.choice(3, size=n, p=probs)
        out = np.empty(n, dtype=float)
        u = rng.random(n)
        for i in range(3):
            m = seg == i
            if not np.any(m):
                continue
            lo, hi = self._edges[i], self._edges[i + 1]
            out[m] = _inverse_cdf_powerlaw(u[m], lo, hi, self._slopes[i])
        return out


# ---------------------------------------------------------------------------
# Config constructors
# ---------------------------------------------------------------------------
def hmxb_from_config(cfg: dict, SFR: float = 1.0, preset: str | None = None) -> HMXBXLF:
    """Build an :class:`HMXBXLF` from a parsed config dict.

    Parameters
    ----------
    cfg : dict
        The full parsed YAML config (``configs/xlf_defaults.yaml``).
    SFR : float
        Star-formation rate (Msun/yr).
    preset : str, optional
        ``"mineo12"`` or ``"ggs03"``.  Defaults to ``cfg['hmxb']['preset']``.
    """
    h = cfg["hmxb"]
    preset = preset or h["preset"]
    p = h[preset]
    return HMXBXLF(
        xi=float(p["xi"]),
        gamma=float(p["gamma"]),
        L_cut=float(p["L_cut"]),
        L_min=float(cfg["forward"]["L_min"]),
        L_ref=float(cfg["L_unit"]),
        SFR=float(SFR),
        band=str(p["band"]),
    )


def lmxb_from_config(cfg: dict, Mstar: float = 1.0e11) -> LMXBXLF:
    """Build an :class:`LMXBXLF` from a parsed config dict."""
    m = cfg["lmxb"]
    return LMXBXLF(
        alpha1=float(m["alpha1"]),
        alpha2=float(m["alpha2"]),
        alpha3=float(m["alpha3"]),
        L_b1=float(m["L_b1"]),
        L_b2=float(m["L_b2"]),
        L_cut=float(m["L_cut"]),
        K1=float(m["K1"]),
        mass_ref=float(m["K1_mass_ref"]),
        Mstar=float(Mstar),
        L_min=float(cfg["forward"]["L_min"]),
        band=str(m["band"]),
    )
