"""Single-galaxy XLF recovery demo (Phase 2).

Simulates ONE galaxy with the forward model (seeded), then fits its detected
HMXB luminosities with the **selection-aware** unbinned Poisson-process
likelihood and the **naive** (hard-flux-cut, no-completeness) likelihood, using
BOTH samplers (UltraNest primary, emcee fallback).  Prints truth vs posterior
median +/- 68% for each, and saves:

  * outputs/diagnostics/fit_single_corner.png  -- corner plot (both fits)
  * outputs/diagnostics/fit_single_xlf.png     -- true vs fitted XLF overlay
                                                  on the detected-L histogram.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\run_fit_single.py --config configs\\fit_single.yaml

Everything is reproducible from the config's ``seed``.
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

from xlf_model.forward import (  # noqa: E402
    detector_from_config,
    luminosity_to_flux,
    run_forward,
)
from xlf_model.inference import fit_xlf, make_hmxb_problem  # noqa: E402
from xlf_model.xlf import HMXBXLF, hmxb_from_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=os.path.join(_REPO, "configs", "fit_single.yaml"),
        help="path to the YAML config",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(_REPO, "outputs", "diagnostics"),
        help="directory for the diagnostic plots",
    )
    p.add_argument(
        "--sampler",
        default="both",
        choices=["ultranest", "emcee", "both"],
        help="which sampler(s) to run (default: both)",
    )
    return p.parse_args(argv)


def simulate_galaxy(cfg):
    """Simulate one galaxy; return (truth dict, L_obs array, detector, distance)."""
    gal = cfg["galaxy"]
    SFR = float(gal["SFR"])
    distance = float(gal["distance_Mpc"])
    absorption = float(gal.get("absorption_flux_factor",
                               cfg["forward"].get("absorption_flux_factor", 1.0)))
    detector = detector_from_config(cfg, name=gal["detector"])
    xlf = hmxb_from_config(cfg, SFR=SFR)

    rng = np.random.default_rng(int(cfg["seed"]))
    res = run_forward(xlf, distance, detector, rng,
                      absorption_flux_factor=absorption, component="HMXB")
    L_obs = res.table["L_obs"]
    L_obs = L_obs[L_obs > 0]  # drop zero-count back-conversions

    truth = {
        "log10_xi_eff": float(np.log10(xlf.xi * SFR)),
        "gamma": float(xlf.gamma),
        "xi": float(xlf.xi),
        "SFR": SFR,
        "L_cut": float(xlf.L_cut),
        "L_min": float(xlf.L_min),
        "L_ref": float(xlf.L_ref),
    }
    return truth, L_obs, detector, distance, absorption, res.funnel


def run_one_fit(L_obs, cfg, *, distance, detector, absorption,
                selection_aware, sampler, seed):
    """Build the likelihood and fit it; return a FitResult."""
    if selection_aware:
        L_fit = L_obs
    else:
        # Naive model: hard flux cut at the nominal limit. Sources below the cut
        # have P_det = 0 (ln lambda = -inf), so the naive fitter is handed only
        # the sources above its own cut -- exactly the data a naive analyst would
        # keep. This is the deliberate mistake whose bias we measure.
        flux = luminosity_to_flux(L_obs, distance) * absorption
        L_fit = L_obs[flux >= detector.flux_limit_50]

    like = make_hmxb_problem(
        L_fit, cfg, distance_Mpc=distance, detector=detector,
        absorption_flux_factor=absorption, selection_aware=selection_aware,
    )
    fr = fit_xlf(like, cfg, sampler=sampler, seed=seed)
    fr.extra["n_fit"] = int(L_fit.size)
    return fr


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    truth, L_obs, detector, distance, absorption, funnel = simulate_galaxy(cfg)
    truth_summary = {"log10_xi_eff": truth["log10_xi_eff"], "gamma": truth["gamma"]}

    print("=" * 66)
    print("XRB-XLF single-galaxy recovery demo (Phase 2)")
    print(f"  galaxy: SFR={cfg['galaxy']['SFR']} Msun/yr, D={distance} Mpc, "
          f"detector={detector.name}")
    print(f"  seed={cfg['seed']}   N_detected={L_obs.size}")
    print(f"  TRUTH: log10(xi_eff)={truth['log10_xi_eff']:.4f}  "
          f"gamma={truth['gamma']:.4f}")
    print("=" * 66)

    samplers = ["ultranest", "emcee"] if args.sampler == "both" else [args.sampler]

    # primary sampler for the printed/plotted posteriors
    primary = "ultranest" if "ultranest" in samplers else samplers[0]

    results = {}  # (sampler, mode) -> FitResult
    for s in samplers:
        for aware in (True, False):
            mode = "selection-aware" if aware else "naive"
            fr = run_one_fit(
                L_obs, cfg, distance=distance, detector=detector,
                absorption=absorption, selection_aware=aware,
                sampler=s, seed=1,
            )
            results[(s, aware)] = fr
            logz = f"  logZ={fr.logZ:.2f}" if fr.logZ is not None else ""
            print(f"\n[{s}] {mode} fit  (N_fit={fr.extra['n_fit']}, "
                  f"wall={fr.wall_time_s:.1f}s){logz}")
            print(fr.format_summary(truth=truth_summary))

    # --- diagnostics use the primary sampler ---
    fr_aware = results[(primary, True)]
    fr_naive = results[(primary, False)]
    _make_corner(fr_aware, fr_naive, truth_summary, args.outdir)
    _make_xlf_overlay(L_obs, truth, fr_aware, fr_naive, detector, distance,
                      absorption, cfg, args.outdir)

    print(f"\nSaved corner plot   -> {os.path.join(args.outdir, 'fit_single_corner.png')}")
    print(f"Saved XLF overlay   -> {os.path.join(args.outdir, 'fit_single_xlf.png')}")
    return 0


def _make_corner(fr_aware, fr_naive, truth, outdir):
    """Overlaid corner plot: selection-aware vs naive posteriors + truth."""
    import corner

    os.makedirs(outdir, exist_ok=True)
    labels = [r"$\log_{10}\,\xi_{\rm eff}$", r"$\gamma$"]
    truths = [truth["log10_xi_eff"], truth["gamma"]]

    # range: union of both posteriors with padding
    both = np.vstack([fr_aware.samples, fr_naive.samples])
    rng = [(both[:, i].min(), both[:, i].max()) for i in range(both.shape[1])]
    pad = [0.1 * (hi - lo) for lo, hi in rng]
    rng = [(lo - p, hi + p) for (lo, hi), p in zip(rng, pad)]

    fig = corner.corner(
        fr_aware.samples, labels=labels, truths=truths, range=rng,
        color="C0", truth_color="k", plot_datapoints=False,
        hist_kwargs={"density": True},
        levels=(0.68, 0.95), label_kwargs={"fontsize": 12},
    )
    corner.corner(
        fr_naive.samples, fig=fig, range=rng, color="C3",
        plot_datapoints=False, hist_kwargs={"density": True},
        levels=(0.68, 0.95),
    )
    # manual legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="C0", lw=2, label="selection-aware"),
        Line2D([0], [0], color="C3", lw=2, label="naive (hard cut)"),
        Line2D([0], [0], color="k", lw=1.5, ls="-", label="truth"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10,
               frameon=True, bbox_to_anchor=(0.98, 0.92))
    fig.suptitle("HMXB XLF recovery: selection-aware vs naive",
                 fontsize=13, y=1.02)
    out = os.path.join(outdir, "fit_single_corner.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _xlf_from_params(theta, truth):
    """Build an HMXBXLF from a (log10 xi_eff, gamma) sample (SFR folded in)."""
    return HMXBXLF(
        xi=10.0 ** float(theta[0]), gamma=float(theta[1]),
        L_cut=truth["L_cut"], L_min=truth["L_min"], L_ref=truth["L_ref"],
        SFR=1.0,
    )


def _make_xlf_overlay(L_obs, truth, fr_aware, fr_naive, detector, distance,
                      absorption, cfg, outdir):
    """True vs fitted XLF (dN/dL) overlay on the detected-L histogram."""
    os.makedirs(outdir, exist_ok=True)

    L_min, L_cut = truth["L_min"], truth["L_cut"]
    L_grid = np.logspace(np.log10(L_min), np.log10(L_cut), 400)

    xlf_true = _xlf_from_params([truth["log10_xi_eff"], truth["gamma"]], truth)
    dN_true = xlf_true.dN_dL(L_grid)

    def band(fr, color, label):
        # posterior band: 16/50/84 of dN/dL over samples
        samp = fr.samples
        idx = np.random.default_rng(0).choice(samp.shape[0],
                                              size=min(400, samp.shape[0]),
                                              replace=False)
        curves = np.array([_xlf_from_params(samp[i], truth).dN_dL(L_grid)
                           for i in idx])
        lo, med, hi = np.percentile(curves, [16, 50, 84], axis=0)
        ax.plot(L_grid, med, color=color, lw=2.0, label=label)
        ax.fill_between(L_grid, lo, hi, color=color, alpha=0.22)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    band(fr_aware, "C0", "selection-aware fit (68%)")
    band(fr_naive, "C3", "naive fit (68%)")
    ax.plot(L_grid, dN_true, color="k", lw=2.2, ls="--", label="truth")

    # detected-L histogram as dN/dL points (number / dL per bin), for context
    bins = np.logspace(np.log10(max(L_obs.min(), L_min)),
                       np.log10(L_obs.max()) + 0.1, 20)
    counts, edges = np.histogram(L_obs, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    dL = np.diff(edges)
    nz = counts > 0
    ax.plot(centres[nz], counts[nz] / dL[nz], "o", color="0.3", ms=5,
            label=f"detected L (N={L_obs.size})", zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"X-ray luminosity $L_X$ [erg s$^{-1}$] (0.5--8 keV)")
    ax.set_ylabel(r"$dN/dL$ [(erg s$^{-1}$)$^{-1}$]")
    ax.set_title("HMXB XLF: true vs fitted (selection-aware vs naive)")
    ax.legend(fontsize=9, loc="lower left")

    # mark the 50% completeness luminosity at this distance
    L_50 = detector.flux_limit_50 / absorption * (
        4.0 * np.pi * (distance * 3.0856775814913673e24) ** 2
    )
    ax.axvline(L_50, color="0.5", lw=1.0, ls=":", alpha=0.8)
    ax.text(L_50, ax.get_ylim()[1], " 50% limit", fontsize=8, va="top")

    fig.tight_layout()
    out = os.path.join(outdir, "fit_single_xlf.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
