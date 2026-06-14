r"""Phase 3 (c): the L_X-SFR nonlinearity reproduction (Gilfanov+ 2004).

Monte-Carlo the TOTAL HMXB X-ray luminosity of a galaxy as a function of its
star-formation rate, and show the statistical nonlinearity at low SFR.

The physics (Gilfanov, Grimm & Sunyaev 2004, MNRAS 351, 1365; astro-ph/0312540)
-----------------------------------------------------------------------------
For a power-law HMXB XLF with slope gamma < 2 the *integral* that gives the
total luminosity is dominated by its bright end, i.e. by the single most
luminous source in the galaxy.  At high SFR the population is well sampled, the
brightest source sits near the cutoff every time, and <L_tot> follows the linear
Mineo+12 scaling L_X = 2.61e39 * SFR.  At LOW SFR the bright end is sparsely
populated -- whether the galaxy hosts one luminous HMXB or none is a coin-flip --
so:

  * the SCATTER of L_tot at fixed SFR blows up (it is set by the Poisson
    statistics of a handful of bright sources);
  * the MEDIAN (and mode) of L_tot fall BELOW the linear line and steepen:
    in the small-N regime L_tot ~ SFR^(1/(gamma-1)), super-linear, while the
    MEAN stays linear -- so mean and median diverge by a large factor.

This is the INTRINSIC statistical effect: HMXB only, NO detector cut, NO
measurement noise.  Each Monte-Carlo galaxy is a fresh Poisson draw from the
XLF; L_tot is the sum of the intrinsic luminosities.

Output
------
  * outputs/diagnostics/lxsfr_nonlinearity.png  (publication-quality, 2-panel)
  * the summary numbers printed to stdout (and pasted into RESULTS.md):
      - the SFR at which the L_tot scatter exceeds 0.3 dex;
      - the effective power-law index of the median L_tot below the break;
      - the mean-vs-median (mode) divergence factor at the lowest SFR.

Usage
-----
    .venv\Scripts\python.exe scripts\run_lxsfr_demo.py --config configs\lxsfr_demo.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xlf_model.forward import draw_population  # noqa: E402
from xlf_model.xlf import HMXBXLF, powerlaw_integral  # noqa: E402


# ---------------------------------------------------------------------------
# Monte-Carlo: total HMXB luminosity vs SFR
# ---------------------------------------------------------------------------
def monte_carlo_lxsfr(cfg, rng):
    """Return SFR grid + per-SFR L_tot statistics from the Monte-Carlo draws."""
    h = cfg["hmxb"]
    L_min = float(cfg["forward"]["L_min"])
    L_ref = float(cfg["L_unit"])
    xi = float(h["xi"])
    gamma = float(h["gamma"])
    L_cut = float(h["L_cut"])

    lx = cfg["lxsfr"]
    sfrs = np.logspace(np.log10(float(lx["sfr_min"])),
                       np.log10(float(lx["sfr_max"])),
                       int(lx["n_sfr"]))
    n_real = int(lx["n_realizations"])

    stats = {k: np.empty(len(sfrs)) for k in
             ("mean", "median", "p16", "p84", "scatter_dex", "p_zero", "mode")}

    for j, SFR in enumerate(sfrs):
        xlf = HMXBXLF(xi=xi, gamma=gamma, L_cut=L_cut, L_min=L_min,
                      L_ref=L_ref, SFR=SFR)
        Ltot = np.empty(n_real)
        for i in range(n_real):
            lum, _ = draw_population(xlf, rng)
            Ltot[i] = lum.sum()
        pos = Ltot[Ltot > 0]
        stats["mean"][j] = Ltot.mean()
        stats["median"][j] = np.median(Ltot)
        stats["p16"][j] = np.percentile(Ltot, 16)
        stats["p84"][j] = np.percentile(Ltot, 84)
        stats["p_zero"][j] = float(np.mean(Ltot == 0))
        if pos.size > 10:
            lg = np.log10(pos)
            stats["scatter_dex"][j] = 0.5 * (np.percentile(lg, 84)
                                             - np.percentile(lg, 16))
            # mode estimate: peak of a log-L histogram of the positive draws
            hcounts, hedges = np.histogram(lg, bins=30)
            stats["mode"][j] = 10.0 ** (0.5 * (hedges[np.argmax(hcounts)]
                                               + hedges[np.argmax(hcounts) + 1]))
        else:
            stats["scatter_dex"][j] = np.nan
            stats["mode"][j] = np.nan

    return sfrs, stats, (xi, gamma, L_cut, L_min, L_ref)


def analytic_mean_per_sfr(xi, gamma, L_cut, L_min, L_ref):
    """Closed-form <L_tot>/SFR = xi * L_ref * integral L38^(1-gamma) dL38."""
    a = L_min / L_ref
    b = L_cut / L_ref
    return xi * L_ref * powerlaw_integral(a, b, gamma - 1.0)


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
def headline_numbers(sfrs, stats, reference):
    """Compute the quantitative findings reported in RESULTS.md."""
    ls = np.log10(sfrs)
    scat = stats["scatter_dex"]
    med = stats["median"]
    mean = stats["mean"]

    # SFR where scatter crosses 0.3 dex (scatter decreases with SFR)
    sfr_03 = np.nan
    idx = np.where((scat[:-1] >= 0.3) & (scat[1:] < 0.3))[0]
    if len(idx):
        i = idx[0]
        frac = (0.3 - scat[i]) / (scat[i + 1] - scat[i])
        sfr_03 = 10.0 ** (ls[i] + frac * (ls[i + 1] - ls[i]))

    nl_lo = float(reference["nonlinear_sfr_lo"])
    # effective index of the MEDIAN below the break (small-N / discreteness regime)
    m = (sfrs >= 0.02) & (sfrs <= max(nl_lo * 0.7, 2.0))
    idx_lo = np.polyfit(np.log10(sfrs[m]), np.log10(med[m]), 1)[0] if m.sum() >= 3 else np.nan
    # linear-regime median slope, for contrast
    m2 = sfrs >= 10.0
    idx_hi = np.polyfit(np.log10(sfrs[m2]), np.log10(med[m2]), 1)[0] if m2.sum() >= 3 else np.nan

    return {
        "sfr_scatter_0p3": sfr_03,
        "median_index_lowSFR": idx_lo,
        "median_index_highSFR": idx_hi,
        "mean_over_median_lowest": float(mean[0] / med[0]) if med[0] > 0 else np.inf,
        "mean_per_sfr_high": float(np.mean((mean / sfrs)[sfrs > 20])),
    }


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong 2011 / Okabe-Ito)
C_MEAN = "#000000"      # black   -- mean
C_MED = "#0072B2"       # blue    -- median (mode tracer)
C_BAND = "#56B4E9"      # sky blue-- 16-84% band
C_LIN = "#D55E00"       # vermillion -- Mineo+12 linear line
C_BREAK = "#999999"     # grey    -- break region


def make_figure(sfrs, stats, reference, hn, gamma, mean_per_sfr_analytic, out):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.4, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
    )

    lin = float(reference["mineo12_slope"])
    nl_lo = float(reference["nonlinear_sfr_lo"])
    nl_hi = float(reference["nonlinear_sfr_hi"])

    # --- panel 1: L_tot vs SFR ---
    # 16-84% scatter band
    ax.fill_between(sfrs, stats["p16"], stats["p84"], color=C_BAND, alpha=0.35,
                    label=r"16--84\% scatter", zorder=1)
    # mean and median
    ax.plot(sfrs, stats["mean"], color=C_MEAN, lw=2.4, label=r"mean $\langle L_{\rm tot}\rangle$",
            zorder=4)
    ax.plot(sfrs, stats["median"], color=C_MED, lw=2.4, ls="-",
            label=r"median $L_{\rm tot}$ (mode tracer)", zorder=4)
    # the linear Mineo+12 reference
    L_lin = lin * sfrs
    ax.plot(sfrs, L_lin, color=C_LIN, lw=2.0, ls="--",
            label=r"Mineo+12 linear: $L_X = 2.61\times10^{39}\,\mathrm{SFR}$",
            zorder=3)

    # break region shading + annotation
    ax.axvspan(nl_lo, nl_hi, color=C_BREAK, alpha=0.18, zorder=0)
    ax.axvline(hn["sfr_scatter_0p3"], color=C_BREAK, lw=1.3, ls=":", zorder=2)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"total HMXB luminosity  $L_{\rm tot}$  [erg s$^{-1}$]  (0.5--8 keV)")
    ax.set_title("Statistical nonlinearity of $L_X$ vs SFR (HMXB only; "
                 "intrinsic, no detector cut)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    # annotate the flattening / scatter blow-up
    ax.annotate(
        "discreteness regime:\n"
        r"median $\propto$ SFR$^{%.2f}$ (super-linear)," % hn["median_index_lowSFR"]
        + "\nfalls below the linear line;\nscatter blows up",
        xy=(0.045, stats["median"][np.argmin(np.abs(sfrs - 0.045))]),
        xytext=(0.16, 1.5e36), fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0),
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
    )
    ax.annotate(
        r"scatter $>0.3$ dex" + "\n" + r"below SFR $\approx %.1f\,M_\odot$/yr" % hn["sfr_scatter_0p3"],
        xy=(hn["sfr_scatter_0p3"], 4e38), xytext=(8.0, 1.3e38),
        fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
    )
    # GGS04 theoretical small-N slope for reference
    theo = 1.0 / (gamma - 1.0)
    ax.text(0.012, 2.0e37,
            r"GGS04: $L_{\rm tot}\propto$ SFR$^{1/(\gamma-1)}=$ SFR$^{%.2f}$" % theo,
            fontsize=8.5, color=C_MED)

    # --- panel 2: scatter (dex) vs SFR ---
    ax2.plot(sfrs, stats["scatter_dex"], color=C_MED, lw=2.2,
             label=r"our $\log_{10} L_{\rm tot}$ scatter (statistical floor only)")
    ax2.axhline(0.3, color=C_LIN, lw=1.4, ls="--", label="0.3 dex")
    ax2.axhline(0.43, color="0.5", lw=1.0, ls=":",
                label=r"M12 observed scatter $\sigma=0.43$ dex, for scale"
                      "\n(includes non-statistical terms)")
    ax2.axvspan(nl_lo, nl_hi, color=C_BREAK, alpha=0.18)
    ax2.axvline(hn["sfr_scatter_0p3"], color=C_BREAK, lw=1.3, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"star-formation rate  SFR  [$M_\odot$ yr$^{-1}$]")
    ax2.set_ylabel("scatter [dex]")
    ax2.set_ylim(0, max(0.85, np.nanmax(stats["scatter_dex"]) * 1.1))
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "lxsfr_demo.yaml"))
    p.add_argument("--outdir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # matplotlib mathtext (no system LaTeX needed)
    matplotlib.rcParams["text.usetex"] = False

    rng = np.random.default_rng(int(cfg["seed"]))
    sfrs, stats, (xi, gamma, L_cut, L_min, L_ref) = monte_carlo_lxsfr(cfg, rng)
    mean_per_sfr = analytic_mean_per_sfr(xi, gamma, L_cut, L_min, L_ref)
    reference = cfg["reference"]
    hn = headline_numbers(sfrs, stats, reference)

    outdir = args.outdir or os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, cfg["run"]["figure"])
    make_figure(sfrs, stats, reference, hn, gamma, mean_per_sfr, out)

    print("=" * 70)
    print("L_X-SFR nonlinearity (Gilfanov+ 2004; astro-ph/0312540)")
    print(f"  HMXB XLF: xi={xi}, gamma={gamma}, L_cut={L_cut:.2e}, "
          f"L_min={L_min:.0e}  (0.5-8 keV, intrinsic)")
    print(f"  Monte-Carlo: {int(cfg['lxsfr']['n_realizations'])} galaxies x "
          f"{int(cfg['lxsfr']['n_sfr'])} SFR points")
    print("-" * 70)
    print(f"  analytic <L_tot>/SFR             = {mean_per_sfr:.3e} erg/s "
          f"(M12 Eq.20 target 2.61e39)")
    print(f"  MC mean <L_tot>/SFR (SFR>20)     = {hn['mean_per_sfr_high']:.3e} erg/s")
    print(f"  SFR where scatter exceeds 0.3 dex = {hn['sfr_scatter_0p3']:.2f} "
          f"Msun/yr  (GGS04 threshold ~4-5)")
    print(f"  effective median index (SFR<break)= {hn['median_index_lowSFR']:.3f} "
          f"(GGS04 small-N: 1/(gamma-1) = {1.0/(gamma-1.0):.3f})")
    print(f"  median index (SFR>10, linear)     = {hn['median_index_highSFR']:.3f}")
    print(f"  mean/median at SFR={sfrs[0]:.3f}        = {hn['mean_over_median_lowest']:.1f}x "
          f"(mode-vs-mean divergence)")
    print("-" * 70)
    print(f"Saved figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
