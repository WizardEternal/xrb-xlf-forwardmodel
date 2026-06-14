r"""Phase 3 (b): the hierarchical stack: simulate a survey, fit it three ways.

Simulates a survey of ``n_galaxies`` galaxies sharing ONE global HMXB XLF
(per-SFR normalisation ``xi`` + slope ``gamma``) but each with its own
log-uniform (SFR, distance) and the shallow eROSITA preset, then fits the slope:

  (i)   JOINTLY  -- one shared ``theta = (log10 xi, gamma)``, summed unbinned
        Poisson-process likelihood over all galaxies;
  (ii)  the BEST single galaxy alone (the one with the most detected sources);
  (iii) the WORST-K stacked (the K fewest-detection galaxies) -- showing the
        stack recovering ``gamma`` from galaxies that individually constrain
        almost nothing.

Product figure ``outputs/diagnostics/hierarchical_stack.png``:
  * panel (a): posterior on ``gamma`` -- joint vs best-single (and worst-K),
    with the truth line;
  * panel (b): the per-galaxy N_det distribution (the "many in the
    few-detections regime" story), with the best galaxy marked.

Headline (printed + for RESULTS.md): the PRECISION GAIN
``sigma_gamma(best-single) / sigma_gamma(joint)`` and the total-N_det accounting.

Usage
-----
    .venv\Scripts\python.exe scripts\run_hierarchical.py --config configs\hierarchical.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import logging

for _name in ("ultranest", "ultranest.solvecompat"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False
    _lg.handlers = [logging.NullHandler()]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xlf_model.forward import detector_from_config, run_forward  # noqa: E402
from xlf_model.hierarchical import (  # noqa: E402
    GalaxySpec,
    build_joint_likelihood,
)
from xlf_model.inference import LogLGrid, Parameter, fit_xlf  # noqa: E402
from xlf_model.xlf import HMXBXLF  # noqa: E402

# colorblind-safe (Okabe-Ito)
C_JOINT = "#0072B2"      # blue       -- joint survey fit
C_BEST = "#D55E00"       # vermillion -- best single galaxy
C_WORST = "#009E73"      # green      -- worst-K stack
C_TRUTH = "#000000"      # black      -- truth
C_BAR = "#56B4E9"        # sky blue   -- N_det histogram


# ---------------------------------------------------------------------------
# Survey simulation
# ---------------------------------------------------------------------------
def simulate_survey(cfg: dict):
    """Draw the survey: per-galaxy (SFR, distance), simulate, collect GalaxySpecs.

    Returns ``(galaxies, detector, meta)`` where ``meta`` carries the truth and
    the XLF support for the fit.
    """
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    truth = cfg["truth"]
    xi_true = float(truth["xi"])
    gamma_true = float(truth["gamma"])
    L_cut = float(truth["L_cut"])
    band = str(truth.get("band", "0.5-8 keV"))

    sv = cfg["survey"]
    n_gal = int(sv["n_galaxies"])
    L_ref = float(cfg["L_unit"])
    L_min = float(cfg["forward"]["L_min"])
    absorption = float(sv.get("absorption_flux_factor",
                              cfg["forward"].get("absorption_flux_factor", 1.0)))
    detector = detector_from_config(cfg, name=sv["detector"])

    # log-uniform SFR and distance across the survey
    log_sfr = rng.uniform(np.log10(sv["sfr_min"]), np.log10(sv["sfr_max"]), n_gal)
    log_d = rng.uniform(np.log10(sv["distance_min_Mpc"]),
                        np.log10(sv["distance_max_Mpc"]), n_gal)
    SFRs = 10.0**log_sfr
    dists = 10.0**log_d

    galaxies = []
    for i in range(n_gal):
        xlf = HMXBXLF(xi=xi_true, gamma=gamma_true, L_cut=L_cut, L_min=L_min,
                      L_ref=L_ref, SFR=float(SFRs[i]), band=band)
        # independent per-galaxy RNG stream (deterministic from the survey seed)
        g_rng = np.random.default_rng(seed + 1000 + i)
        res = run_forward(xlf, float(dists[i]), detector, g_rng,
                          absorption_flux_factor=absorption, component="HMXB")
        L_obs = res.table["L_obs"]
        L_obs = L_obs[L_obs > 0]
        galaxies.append(GalaxySpec(
            L_obs=L_obs, SFR=float(SFRs[i]), distance_Mpc=float(dists[i]),
            detector=detector, absorption_flux_factor=absorption,
            name=f"g{i:02d}",
        ))

    meta = dict(xi_true=xi_true, gamma_true=gamma_true, L_cut=L_cut,
                L_min=L_min, L_ref=L_ref, band=band, absorption=absorption)
    return galaxies, detector, meta


# ---------------------------------------------------------------------------
# Shared priors -> Parameter list (first param relabelled to per-SFR log10 xi)
# ---------------------------------------------------------------------------
def shared_parameters(cfg: dict) -> list[Parameter]:
    pri = cfg["inference"]["priors"]
    lx = pri["log_xi_eff"]
    ga = pri["gamma"]
    return [
        Parameter("log10_xi", float(lx["lo"]), float(lx["hi"]),
                  lx.get("prior", "uniform")),
        Parameter("gamma", float(ga["lo"]), float(ga["hi"]),
                  ga.get("prior", "uniform")),
    ]


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------
def fit_subset(galaxies, params, meta, cfg, sampler, seed):
    """Build + fit the joint likelihood over a subset of galaxies."""
    like = build_joint_likelihood(
        galaxies, params, L_cut=meta["L_cut"], L_min=meta["L_min"],
        L_ref=meta["L_ref"], band=meta["band"], selection_aware=True,
        grid=LogLGrid(points_per_dex=int(cfg["inference"]["grid"]["points_per_dex"])),
    )
    fr = fit_xlf(like, cfg, sampler=sampler, seed=seed)
    return like, fr


def gamma_summary(fr):
    """(median, 68% half-width sigma, q16, q84) for gamma from a FitResult."""
    s = fr.summary()["gamma"]
    sigma = 0.5 * (s["q84"] - s["q16"])
    return s["median"], sigma, s["q16"], s["q84"]


# ---------------------------------------------------------------------------
# The product figure
# ---------------------------------------------------------------------------
def make_figure(results, n_dets, best_idx, gamma_true, out_png):
    """Two panels: gamma posteriors (joint vs best vs worst-K) + N_det histogram."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # ---- panel (a): gamma posterior samples as KDE-ish histograms ----
    order = [("joint", C_JOINT, "joint survey (all galaxies)"),
             ("best", C_BEST, "best single galaxy"),
             ("worst", C_WORST, "weak-tail stack (fewest-detection)")]
    gmin, gmax = gamma_true - 0.6, gamma_true + 0.6
    bins = np.linspace(gmin, gmax, 60)
    for key, color, label in order:
        if key not in results:
            continue
        fr = results[key]["fr"]
        g = fr.samples[:, fr.param_names.index("gamma")]
        med, sigma, q16, q84 = gamma_summary(fr)
        axa.hist(g, bins=bins, density=True, histtype="step", color=color,
                 lw=2.0, label=f"{label}\n  $\\gamma={med:.3f}\\pm{sigma:.3f}$")
    axa.axvline(gamma_true, color=C_TRUTH, lw=1.8, ls="--",
                label=fr"truth $\gamma={gamma_true}$")
    axa.set_xlim(gmin, gmax)
    axa.set_xlabel(r"HMXB slope  $\gamma$")
    axa.set_ylabel("posterior density")
    axa.set_title("(a) slope posterior: joint stack vs best single galaxy")
    axa.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    axa.grid(True, alpha=0.2)

    # ---- panel (b): per-galaxy N_det distribution ----
    n_dets = np.asarray(n_dets)
    nmax = int(n_dets.max())
    edges = np.arange(0, nmax + 2) - 0.5
    axb.hist(n_dets, bins=edges, color=C_BAR, edgecolor="k", lw=0.6,
             alpha=0.85)
    axb.axvline(n_dets[best_idx], color=C_BEST, lw=2.0, ls="-",
                label=f"best galaxy ($N_{{\\rm det}}={int(n_dets[best_idx])}$)")
    n_zero = int(np.sum(n_dets == 0))
    n_le5 = int(np.sum(n_dets <= 5))
    axb.set_xlabel(r"detected HMXBs per galaxy  $N_{\rm det}$")
    axb.set_ylabel("number of galaxies")
    axb.set_title("(b) survey $N_{\\rm det}$ distribution "
                  f"({len(n_dets)} galaxies)")
    axb.text(0.97, 0.78,
             f"$\\Sigma N_{{\\rm det}}={int(n_dets.sum())}$\n"
             f"{n_le5}/{len(n_dets)} have $N_{{\\rm det}}\\leq5$\n"
             f"{n_zero} with $N_{{\\rm det}}=0$",
             transform=axb.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    axb.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    axb.grid(True, alpha=0.2)

    fig.suptitle(
        "XRB-XLF hierarchical stack: pooling a survey of mostly-faint galaxies",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "hierarchical.yaml"))
    p.add_argument("--no-worst", action="store_true",
                   help="skip the optional worst-K stacked fit (iii)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    matplotlib.rcParams["text.usetex"] = False

    sampler = str(cfg["run"]["sampler"])
    seed = int(cfg["seed"])
    worst_k = int(cfg["run"].get("worst_k", 10))
    outdir = os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, cfg["run"]["figure"])

    # ---- simulate the survey ----
    galaxies, detector, meta = simulate_survey(cfg)
    params = shared_parameters(cfg)
    n_dets = np.array([g.n_det for g in galaxies])
    total_ndet = int(n_dets.sum())
    best_idx = int(np.argmax(n_dets))

    print("=" * 72)
    print("HIERARCHICAL STACK (Phase 3b)")
    print("=" * 72)
    print(f"  survey: {len(galaxies)} galaxies, eROSITA preset, "
          f"SFR log-uniform [{cfg['survey']['sfr_min']},{cfg['survey']['sfr_max']}], "
          f"D log-uniform [{cfg['survey']['distance_min_Mpc']},"
          f"{cfg['survey']['distance_max_Mpc']}] Mpc")
    print(f"  truth: xi={meta['xi_true']} (per SFR), gamma={meta['gamma_true']}")
    print(f"  per-galaxy N_det: min={n_dets.min()} max={n_dets.max()} "
          f"median={np.median(n_dets):.0f} total(SUM)={total_ndet}")
    print(f"  galaxies with N_det<=5: {int(np.sum(n_dets <= 5))}/{len(galaxies)}; "
          f"with N_det=0: {int(np.sum(n_dets == 0))}")
    print(f"  best galaxy: {galaxies[best_idx].name} "
          f"(N_det={n_dets[best_idx]}, SFR={galaxies[best_idx].SFR:.2f}, "
          f"D={galaxies[best_idx].distance_Mpc:.1f} Mpc)")
    print("-" * 72)

    results = {}

    # ---- (i) JOINT fit (all galaxies) ----
    print("fitting (i) JOINT survey (all galaxies) ...")
    _, fr_joint = fit_subset(galaxies, params, meta, cfg, sampler, seed + 1)
    gj, sj, qj16, qj84 = gamma_summary(fr_joint)
    results["joint"] = dict(fr=fr_joint, med=gj, sigma=sj)
    print(f"  joint:  gamma = {gj:.4f} +/- {sj:.4f}  "
          f"(68%: [{qj16:.4f}, {qj84:.4f}], {fr_joint.wall_time_s:.1f}s)")

    # ---- (ii) BEST single galaxy alone ----
    print("fitting (ii) BEST single galaxy alone ...")
    _, fr_best = fit_subset([galaxies[best_idx]], params, meta, cfg, sampler, seed + 2)
    gb, sb, qb16, qb84 = gamma_summary(fr_best)
    results["best"] = dict(fr=fr_best, med=gb, sigma=sb)
    print(f"  best:   gamma = {gb:.4f} +/- {sb:.4f}  "
          f"(68%: [{qb16:.4f}, {qb84:.4f}], {fr_best.wall_time_s:.1f}s)")

    # ---- (iii) WORST-K stacked (optional) ----
    # The point of (iii) is to show the stack RECOVERING gamma from galaxies that
    # individually constrain almost nothing.  Galaxies with N_det=0 contribute
    # only a -Lambda_g normalization term (no slope information at all), so a
    # pure-zero stack cannot constrain gamma -- it would demonstrate the floor,
    # not the pooling.  We therefore stack the K FEWEST-detection galaxies that
    # still have at least one detected source (the genuinely "weak but nonzero"
    # tail); if fewer than K such galaxies exist we take all of them.
    if not args.no_worst:
        nonzero_idx = np.where(n_dets > 0)[0]
        nonzero_sorted = nonzero_idx[np.argsort(n_dets[nonzero_idx])]
        worst_order = nonzero_sorted[:worst_k]
        worst_gals = [galaxies[i] for i in worst_order]
        worst_total = int(sum(g.n_det for g in worst_gals))
        print(f"fitting (iii) WORST-{len(worst_gals)} stacked "
              f"(fewest-detection galaxies with N_det>=1; "
              f"N_det per galaxy: {sorted(int(g.n_det) for g in worst_gals)}, "
              f"SUM={worst_total}) ...")
        _, fr_worst = fit_subset(worst_gals, params, meta, cfg, sampler, seed + 3)
        gw, sw, qw16, qw84 = gamma_summary(fr_worst)
        results["worst"] = dict(fr=fr_worst, med=gw, sigma=sw)
        print(f"  worst-{worst_k}: gamma = {gw:.4f} +/- {sw:.4f}  "
              f"(68%: [{qw16:.4f}, {qw84:.4f}], {fr_worst.wall_time_s:.1f}s)")

    # ---- precision gain ----
    gain = results["best"]["sigma"] / results["joint"]["sigma"]
    print("-" * 72)
    print("Precision gain on gamma")
    print(f"  sigma_gamma(best single) = {results['best']['sigma']:.4f}  "
          f"(N_det={n_dets[best_idx]})")
    print(f"  sigma_gamma(joint stack) = {results['joint']['sigma']:.4f}  "
          f"(N_det_total={total_ndet}, {len(galaxies)} galaxies)")
    print(f"  => precision gain factor = {gain:.2f}x tighter")
    naive_sqrt = np.sqrt(total_ndet / max(n_dets[best_idx], 1))
    print(f"  (naive sqrt(N) expectation if all sources were in one galaxy: "
          f"{naive_sqrt:.2f}x -- the stack pools across selection functions, "
          f"so the realized gain need not match this)")
    # bias check: both should bracket truth
    print(f"  truth gamma = {meta['gamma_true']}; "
          f"joint offset = {gj - meta['gamma_true']:+.4f} "
          f"({abs(gj - meta['gamma_true'])/sj:.2f} sigma_joint); "
          f"best offset = {gb - meta['gamma_true']:+.4f} "
          f"({abs(gb - meta['gamma_true'])/sb:.2f} sigma_best)")

    make_figure(results, n_dets, best_idx, meta["gamma_true"], out_png)
    print("-" * 72)
    print(f"Saved figure -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
