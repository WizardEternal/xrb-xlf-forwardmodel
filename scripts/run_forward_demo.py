"""Forward-model demo for a single galaxy.

Runs the XRB-XLF forward model for one demo galaxy (defaults from the config:
SFR = 1 Msun/yr, M* = 1e10 Msun, D = 10 Mpc, eROSITA eRASS1 preset), prints the
funnel accounting table for the HMXB and LMXB components and the combined
population, and saves a diagnostic plot (drawn vs detected luminosity histograms
with the completeness curve overlaid) to outputs/diagnostics/demo_xlf_draw.png.

Usage
-----
    python scripts/run_forward_demo.py --config configs/xlf_defaults.yaml

Everything is reproducible from the config's global seed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

# make src/ importable when run as a script
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xlf_model.forward import (  # noqa: E402
    completeness_erf,
    detector_from_config,
    flux_to_luminosity,
    format_funnel,
    run_forward,
)
from xlf_model.xlf import hmxb_from_config, lmxb_from_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=os.path.join(_REPO, "configs", "xlf_defaults.yaml"),
        help="path to the YAML config",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(_REPO, "outputs", "diagnostics"),
        help="directory for the diagnostic plot",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    gal = cfg["forward"]["demo_galaxy"]
    SFR = float(gal["SFR"])
    Mstar = float(gal["Mstar"])
    distance = float(gal["distance_Mpc"])
    absorption = float(cfg["forward"]["absorption_flux_factor"])
    seed = int(cfg["seed"])

    detector = detector_from_config(cfg, name=gal["detector"])
    hmxb = hmxb_from_config(cfg, SFR=SFR)
    lmxb = lmxb_from_config(cfg, Mstar=Mstar)

    # one Generator threads through everything -> fully reproducible
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("XRB-XLF forward-model demo")
    print(f"  galaxy: SFR={SFR} Msun/yr, M*={Mstar:.2e} Msun, D={distance} Mpc")
    print(f"  detector: {detector.name} (band {detector.band})")
    print(f"  F_lim(50%) = {detector.flux_limit_50:.2e} erg/s/cm^2,"
          f" ramp width = {detector.completeness_width_dex} dex")
    print(f"  HMXB preset: xi={hmxb.xi}, gamma={hmxb.gamma}, band {hmxb.band}")
    print(f"  seed = {seed}")
    print("=" * 60)

    res_h = run_forward(
        hmxb, distance, detector, rng,
        absorption_flux_factor=absorption, component="HMXB",
    )
    res_l = run_forward(
        lmxb, distance, detector, rng,
        absorption_flux_factor=absorption, component="LMXB",
    )

    print(format_funnel(res_h.funnel, title="HMXB funnel (Mineo+12)"))
    print(format_funnel(res_l.funnel, title="LMXB funnel (Gilfanov 2004)"))

    # combined funnel
    combined = {
        "expected_N": res_h.expected_N + res_l.expected_N,
        "n_drawn": res_h.funnel["n_drawn"] + res_l.funnel["n_drawn"],
        "n_above_limit": res_h.funnel["n_above_limit"] + res_l.funnel["n_above_limit"],
        "n_detected": res_h.funnel["n_detected"] + res_l.funnel["n_detected"],
        "flux_limit_50": detector.flux_limit_50,
    }
    print(format_funnel(combined, title="COMBINED funnel (HMXB + LMXB)"))

    # cross-check line
    n_gt_1e38 = float(hmxb.N_gt(1e38)[0])
    print(f"\nMineo cross-check: model N(>1e38)/SFR = {n_gt_1e38:.3f}"
          f"  (M12 Eq.22 target = 3.22)")
    n_gt_1e37_lmxb = float(lmxb.N_gt(1e37)[0]) / (Mstar / 1e11)
    print(f"Gilfanov cross-check: model N(>1e37)/1e11Msun = {n_gt_1e37_lmxb:.1f}"
          f"  (G04 Eq.11 target = 142.9)")

    _make_plot(cfg, res_h, res_l, detector, distance, absorption, args.outdir)
    print(f"\nSaved diagnostic plot to {os.path.join(args.outdir, 'demo_xlf_draw.png')}")
    return 0


def _make_plot(cfg, res_h, res_l, detector, distance, absorption, outdir):
    """Drawn vs detected luminosity histograms + completeness curve overlay."""
    os.makedirs(outdir, exist_ok=True)

    # use the exact drawn realizations carried by the ForwardResults
    drawn = np.concatenate([res_h.L_drawn, res_l.L_drawn])
    det_true = np.concatenate([res_h.table["L_true"], res_l.table["L_true"]])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    lo = np.log10(max(min(drawn.min(), 1e34), 1e33))
    hi = np.log10(max(drawn.max(), det_true.max() if det_true.size else 1e40)) + 0.2
    bins = np.logspace(lo, hi, 45)

    ax.hist(drawn, bins=bins, histtype="stepfilled", alpha=0.35, color="C0",
            label=f"drawn (N={drawn.size})")
    if det_true.size:
        ax.hist(det_true, bins=bins, histtype="step", lw=2.0, color="C3",
                label=f"detected (N={det_true.size})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"X-ray luminosity  $L_X$  [erg s$^{-1}$]  (0.5--8 keV, intrinsic)")
    ax.set_ylabel("number of sources per bin")
    ax.set_title("XRB-XLF forward model: drawn vs detected luminosities")

    # completeness curve as a function of luminosity (via flux at this distance)
    L_grid = np.logspace(lo, hi, 300)
    flux_grid = (L_grid / (4.0 * np.pi)) / ((distance * 3.0856775814913673e24) ** 2)
    flux_grid *= absorption
    comp = completeness_erf(flux_grid, detector.flux_limit_50,
                            detector.completeness_width_dex)
    ax2 = ax.twinx()
    ax2.plot(L_grid, comp, color="k", lw=1.8, ls="--",
             label="completeness $C(L)$")
    ax2.set_ylabel("completeness")
    ax2.set_ylim(-0.02, 1.05)

    # mark the 50% completeness luminosity
    L_50 = flux_to_luminosity(detector.flux_limit_50 / absorption, distance)
    ax2.axvline(L_50, color="k", lw=1.0, ls=":", alpha=0.7)
    ax2.text(L_50, 0.52, "  50% limit", fontsize=8, va="bottom")

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(outdir, "demo_xlf_draw.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
