"""Real-galaxy demonstration of the selection-aware HMXB-slope fit (EXTENSION).

OPTIONAL EXTENSION -- "demonstration on real data".

Applies the repo's existing selection-aware unbinned Poisson-process likelihood
(``xlf_model.inference``) to 1-2 real public Chandra point-source catalogues from
the Mineo, Gilfanov & Sunyaev 2012 (M12) sample, fetched from the HEASARC table
SFGALHMXB.  For each galaxy it runs BOTH the selection-aware fit and the naive
(hard-cut) fit, prints the recovered slope ``gamma`` against M12's published
values, and saves a per-galaxy diagnostic figure (observed XLF + fitted model
+- posterior band).

This is a CONSISTENCY CHECK, not a re-measurement: the per-galaxy incompleteness
is an erf-ramp APPROXIMATION anchored to M12's quoted sensitivity limit (M12 used
Voss & Gilfanov simulations we cannot reproduce), and CXB contamination is not
modeled per source.  See src/xlf_model/real_data.py and RESULTS.md for the full
caveats.  We fit only well above the sensitivity limit (1.5 x L_lim) to
minimize sensitivity to the approximation.

Usage
-----
    set OMP_NUM_THREADS=6
    .venv\\Scripts\\python.exe scripts\\run_real_demo.py --config configs\\real_demo.yaml

Everything is reproducible from the config + cached catalogue.
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

from xlf_model.forward import DetectorPreset, luminosity_to_flux  # noqa: E402
from xlf_model.inference import fit_xlf, make_hmxb_problem  # noqa: E402
from xlf_model.real_data import (  # noqa: E402
    M12_GLOBAL_GAMMA,
    M12_GLOBAL_GAMMA_ERR,
    M12_PERGALAXY_GAMMA_MEAN,
    M12_PERGALAXY_GAMMA_RMS,
    M12_TABLE1,
    completeness_anchor_shift_dex,
    fetch_sfgalhmxb,
    load_catalog_csv,
    parse_galaxy,
)
from xlf_model.xlf import HMXBXLF  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(_REPO, "configs", "real_demo.yaml"))
    p.add_argument("--outdir", default=os.path.join(_REPO, "outputs", "diagnostics"))
    p.add_argument("--catalog", default=None,
                   help="path to a cached SFGALHMXB CSV (default: data/real/, "
                        "fetched if missing)")
    p.add_argument("--sampler", default="ultranest",
                   choices=["ultranest", "emcee"])
    p.add_argument("--no-fetch", action="store_true",
                   help="error instead of downloading if the cache is missing")
    return p.parse_args(argv)


def galaxy_detector(meta, cfg) -> DetectorPreset:
    """Build a per-galaxy DetectorPreset whose erf completeness reproduces the
    M12 sensitivity limit.

    The forward model's completeness is ``C(F) = 0.5*(1+erf(log10(F/F50)/...))``.
    We choose ``F50`` so that completeness = ``k_anchor`` at the flux a source of
    luminosity ``L_lim`` produces at this galaxy's distance -- i.e. we anchor the
    ramp to M12's quoted K=0.6 sensitivity limit.  ``exposure_s`` / ``cts_to_flux``
    are irrelevant for the selection-aware likelihood (only completeness is used),
    but are set to harmless placeholders so the preset is well-formed.
    """
    rcfg = cfg["real"]
    width = float(rcfg["completeness_width_dex"])
    k_anchor = float(rcfg["completeness_k_anchor"])
    L_lim = 10.0 ** meta.log_Llim
    F_Llim = float(luminosity_to_flux(L_lim, meta.distance_Mpc))
    d_anchor = completeness_anchor_shift_dex(width, k_anchor)
    F50 = F_Llim / (10.0 ** d_anchor)
    return DetectorPreset(
        name=f"m12_{meta.name.replace(' ', '_').replace('/', '_')}",
        flux_limit_50=F50,
        completeness_width_dex=width,
        exposure_s=1.0,
        cts_to_flux=1.0,
        band="0.5-8 keV",
    )


def fit_galaxy(rg, cfg, detector, *, sampler, seed):
    """Run the selection-aware AND naive fit for one parsed galaxy.

    Returns a dict with both FitResults and the threshold flux used.
    """
    distance = rg.meta.distance_Mpc
    absorption = 1.0  # M12 luminosities are already de-absorbed

    out = {}
    # selection-aware: fit all sources above the conservative threshold, with the
    # erf completeness ramp (anchored to M12's K=0.6 limit) folded into the
    # likelihood normalisation Lambda(theta).
    like_aware = make_hmxb_problem(
        rg.L_fit, cfg, distance_Mpc=distance, detector=detector,
        absorption_flux_factor=absorption, selection_aware=True,
    )
    fr_aware = fit_xlf(like_aware, cfg, sampler=sampler, seed=seed)
    fr_aware.extra["n_fit"] = int(rg.L_fit.size)
    out["aware"] = fr_aware

    # naive: treat the conservative threshold as a SHARP completeness edge (the
    # typical analyst's choice -- no ramp, P_det = 1 above the threshold, 0
    # below).  We give the naive ObservationModel a detector whose hard cut sits
    # exactly at the fit threshold flux, so the naive Lambda integral spans the
    # same luminosity range as the data (a fair, like-for-like contrast).
    F_thr = float(luminosity_to_flux(rg.L_threshold, distance)) * absorption
    naive_detector = DetectorPreset(
        name=detector.name + "_naive",
        flux_limit_50=F_thr,
        completeness_width_dex=detector.completeness_width_dex,
        exposure_s=detector.exposure_s,
        cts_to_flux=detector.cts_to_flux,
        band=detector.band,
    )
    like_naive = make_hmxb_problem(
        rg.L_fit, cfg, distance_Mpc=distance, detector=naive_detector,
        absorption_flux_factor=absorption, selection_aware=False,
    )
    fr_naive = fit_xlf(like_naive, cfg, sampler=sampler, seed=seed)
    fr_naive.extra["n_fit"] = int(rg.L_fit.size)
    out["naive"] = fr_naive
    return out


def gamma_summary(fr):
    """(median, minus, plus) for the gamma parameter of a FitResult."""
    s = fr.summary()["gamma"]
    return s["median"], s["minus"], s["plus"]


def _xlf_from_sample(theta, L_cut, L_min, L_ref):
    return HMXBXLF(xi=10.0 ** float(theta[0]), gamma=float(theta[1]),
                   L_cut=L_cut, L_min=L_min, L_ref=L_ref, SFR=1.0)


def make_figure(galaxies, cfg, outdir):
    """One row per galaxy: observed XLF (binned dN/dL) + fitted model +- 68% band.

    galaxies : list of (RealGalaxy, fits_dict, detector)
    """
    os.makedirs(outdir, exist_ok=True)
    L_min = float(cfg["forward"]["L_min"])
    L_ref = float(cfg["L_unit"])
    preset = cfg["hmxb"][cfg["hmxb"]["preset"]]
    L_cut = float(preset["L_cut"])

    n = len(galaxies)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 5.6), squeeze=False)
    axes = axes[0]

    for ax, (rg, fits, detector) in zip(axes, galaxies):
        meta = rg.meta
        L_thr = rg.L_threshold
        L_grid = np.logspace(np.log10(L_thr), np.log10(rg.L_fit.max() * 2.0), 300)

        # observed dN/dL from the fitted sources (binned, for context)
        bins = np.logspace(np.log10(L_thr), np.log10(rg.L_fit.max()) + 0.1, 12)
        counts, edges = np.histogram(rg.L_fit, bins=bins)
        centres = np.sqrt(edges[:-1] * edges[1:])
        dL = np.diff(edges)
        nz = counts > 0
        ax.errorbar(centres[nz], counts[nz] / dL[nz],
                    yerr=np.sqrt(counts[nz]) / dL[nz], fmt="o", color="0.25",
                    ms=5, capsize=2, label=f"observed (N={rg.n_fit})", zorder=6)

        # fitted models: posterior 68% band for selection-aware and naive
        def band(fr, color, label):
            samp = fr.samples
            rng = np.random.default_rng(0)
            idx = rng.choice(samp.shape[0], size=min(400, samp.shape[0]),
                             replace=False)
            curves = np.array([
                _xlf_from_sample(samp[i], L_cut, L_min, L_ref).dN_dL(L_grid)
                for i in idx])
            lo, med, hi = np.percentile(curves, [16, 50, 84], axis=0)
            ax.plot(L_grid, med, color=color, lw=2.0, label=label)
            ax.fill_between(L_grid, lo, hi, color=color, alpha=0.22)

        band(fits["aware"], "C0", "selection-aware fit (68%)")
        band(fits["naive"], "C3", "naive fit (68%)")

        # M12 global-slope reference line (gamma=1.60), normalised to the data
        # at the threshold for visual comparison of the SHAPE/slope only.
        g_m12 = M12_GLOBAL_GAMMA
        ref_norm = (counts[nz][0] / dL[nz][0]) if nz.any() else 1.0
        L0 = centres[nz][0] if nz.any() else L_thr
        ref = ref_norm * (L_grid / L0) ** (-g_m12)
        ax.plot(L_grid, ref, color="k", lw=1.6, ls="--",
                label=fr"M12 slope $\gamma$={g_m12:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        gm, gmn, gmp = gamma_summary(fits["aware"])

        # reference vertical lines, with labels in axis-fraction y (robust on log)
        ax.axvline(10.0 ** meta.log_Llim, color="0.5", lw=1.0, ls=":")
        ax.annotate(r"$L_{\rm lim}$ (K=0.6)", xy=(10.0 ** meta.log_Llim, 0.04),
                    xycoords=("data", "axes fraction"), fontsize=8,
                    ha="right", rotation=90, va="bottom", color="0.4")
        ax.axvline(L_thr, color="0.7", lw=1.0, ls="-.")
        ax.annotate(r"$1.5\,L_{\rm lim}$ (fit threshold)", xy=(L_thr, 0.04),
                    xycoords=("data", "axes fraction"), fontsize=8,
                    ha="left", rotation=90, va="bottom", color="0.55")

        ax.set_xlabel(r"$L_X$ [erg s$^{-1}$] (0.5--8 keV)")
        ax.set_ylabel(r"$dN/dL$ [(erg s$^{-1}$)$^{-1}$]")
        ax.set_title(f"{meta.label}\n"
                     fr"aware $\gamma$ = {gm:.2f}$^{{+{gmp:.2f}}}_{{-{gmn:.2f}}}$"
                     fr"  (M12 $\langle\gamma\rangle$={M12_PERGALAXY_GAMMA_MEAN:.2f}"
                     fr"$\pm${M12_PERGALAXY_GAMMA_RMS:.2f})")
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("HMXB XLF slope on real Chandra data (Mineo+2012 sample) "
                 "-- demonstration on real data",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(outdir, "real_galaxy_fit.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # --- data: cached / resumable fetch ---
    if args.catalog is not None:
        cat_path = args.catalog
    elif args.no_fetch:
        cat_path = os.path.join(_REPO, "data", "real", "sfgalhmxb_full.csv")
        if not os.path.exists(cat_path):
            print(f"ERROR: catalogue cache missing at {cat_path} and --no-fetch set.")
            return 2
    else:
        cat_path = fetch_sfgalhmxb()
    catalog = load_catalog_csv(cat_path)

    rcfg = cfg["real"]
    flag_keep = tuple(rcfg["flag_keep"]) if rcfg.get("flag_keep") else None
    thr_factor = float(rcfg["threshold_factor"])

    print("=" * 74)
    print("XRB-XLF real-galaxy DEMONSTRATION (optional extension)")
    print("  data: HEASARC SFGALHMXB (Mineo, Gilfanov & Sunyaev 2012 sample)")
    print(f"  catalogue cache: {cat_path}")
    print(f"  M12 published: global gamma = {M12_GLOBAL_GAMMA:.2f} +/- "
          f"{M12_GLOBAL_GAMMA_ERR:.2f}; per-galaxy <gamma> = "
          f"{M12_PERGALAXY_GAMMA_MEAN:.2f} (rms {M12_PERGALAXY_GAMMA_RMS:.2f})")
    print("=" * 74)

    galaxies_out = []
    for gal in rcfg["galaxies"]:
        meta = M12_TABLE1[gal]
        rg = parse_galaxy(catalog, gal, meta=meta, flag_keep=flag_keep,
                          threshold_factor=thr_factor)
        detector = galaxy_detector(meta, cfg)
        fits = fit_galaxy(rg, cfg, detector, sampler=args.sampler,
                          seed=int(cfg["seed"]))
        galaxies_out.append((rg, fits, detector))

        gm_a, mn_a, mp_a = gamma_summary(fits["aware"])
        gm_n, mn_n, mp_n = gamma_summary(fits["naive"])
        print(f"\n{meta.label}")
        print(f"  D = {meta.distance_Mpc} Mpc, SFR = {meta.SFR} Msun/yr, "
              f"log Llim = {meta.log_Llim} (K=0.6)")
        print(f"  catalogue sources = {rg.n_total_catalog}, "
              f"flag{flag_keep} = {rg.L_all.size} (M12 N_XRB = {meta.N_XRB}), "
              f"above 1.5xLlim = {rg.n_fit}")
        print(f"  selection-aware gamma = {gm_a:.3f}  (-{mn_a:.3f} / +{mp_a:.3f})")
        print(f"  naive           gamma = {gm_n:.3f}  (-{mn_n:.3f} / +{mp_n:.3f})")
        print(f"  M12 global gamma = {M12_GLOBAL_GAMMA:.2f} +/- "
              f"{M12_GLOBAL_GAMMA_ERR:.2f}  "
              f"(per-galaxy rms {M12_PERGALAXY_GAMMA_RMS:.2f})")
        # consistency flag vs the per-galaxy scatter band
        dev = abs(gm_a - M12_GLOBAL_GAMMA)
        within = dev <= 2.0 * np.hypot(M12_PERGALAXY_GAMMA_RMS, mn_a + mp_a)
        print(f"  -> selection-aware within M12 per-galaxy scatter: "
              f"{'YES' if within else 'NO'} (|dgamma|={dev:.2f})")

    out = make_figure(galaxies_out, cfg, args.outdir)
    print(f"\nSaved figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
