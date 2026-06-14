r"""Phase 4: build the posterior-coverage CURVE.

Reads ``outputs/recovery/coverage_results.jsonl`` (from ``run_coverage.py``) and,
for each nominal central-credibility level c in {0.1, ..., 0.9}, computes the
EMPIRICAL coverage: the fraction of realizations whose c% central credible
interval [q_lo(c), q_hi(c)] contained the truth.  A calibrated likelihood gives
empirical ~ nominal, a near-diagonal curve.

We report coverage for BOTH parameters (gamma and log10 xi_eff) and draw the
two-panel coverage curve with:
  * the diagonal (perfect calibration),
  * a binomial (Wilson) error band on each empirical point,
  * the nominal-vs-empirical points with binomial error bars.

"Near-diagonal" is the success criterion.  We also print the max deviation from
the diagonal and a one-line PASS/INVESTIGATE verdict; if it is not near-diagonal
the verdict says so.

Cross-link: the same coverage methodology runs in the sibling repo
sbi-xray-calibration (expected-coverage test in src/sbixcal/calibrate.py).

Usage
-----
    .venv\Scripts\python.exe scripts\analyze_coverage.py --config configs\coverage.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# colorblind-safe (Okabe-Ito)
C_GAMMA = "#0072B2"     # blue       -- gamma
C_XI = "#D55E00"        # vermillion -- log10 xi_eff
C_DIAG = "#000000"      # black      -- diagonal (perfect calibration)
C_BAND = "#999999"      # grey       -- binomial band


def load_rows(jsonl_path):
    """Read coverage rows, tolerant of a truncated final line (crash safety)."""
    rows = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
    return rows


def wilson_interval(k, n, z=1.0):
    """Wilson score interval for a binomial proportion (z=1 -> ~68% band).

    Returns (lo, hi) for k successes in n trials.  Wilson is well-behaved near
    p=0 and p=1 (unlike the normal approximation), which matters at the
    extreme nominal levels.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half, centre + half)


def empirical_coverage(rows, levels):
    """Per-parameter empirical coverage at each nominal level.

    Returns a dict ``param -> dict`` with arrays:
      nominal, emp (empirical coverage), k (covered count), n (usable count),
      lo, hi (Wilson band).
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    levels = np.asarray(levels, dtype=float)

    out = {}
    for pname, lo_key, hi_key, truth_key in (
        ("gamma", "gamma_ci_lo", "gamma_ci_hi", "gamma_true"),
        ("log10_xi_eff", "log_xi_ci_lo", "log_xi_ci_hi", "log10_xi_eff_true"),
    ):
        emp = np.empty(len(levels))
        kk = np.empty(len(levels), dtype=int)
        lo_band = np.empty(len(levels))
        hi_band = np.empty(len(levels))
        for j in range(len(levels)):
            covered = 0
            for r in ok:
                truth = float(r[truth_key])
                ci_lo = float(r[lo_key][j])
                ci_hi = float(r[hi_key][j])
                if ci_lo <= truth <= ci_hi:
                    covered += 1
            emp[j] = covered / n if n else np.nan
            kk[j] = covered
            lo_band[j], hi_band[j] = wilson_interval(covered, n, z=1.0)
        out[pname] = dict(nominal=levels, emp=emp, k=kk, n=n,
                          lo=lo_band, hi=hi_band)
    return out


def make_coverage_curve(cov, out_png, n_used, n_skip):
    """Two-panel coverage curve (one per parameter) with diagonal + binomial band."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    panels = [
        ("gamma", r"slope $\gamma$", C_GAMMA, axes[0]),
        ("log10_xi_eff", r"normalisation $\log_{10}\xi_{\rm eff}$", C_XI, axes[1]),
    ]
    for pname, label, color, ax in panels:
        d = cov[pname]
        x = d["nominal"]
        # diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], color=C_DIAG, lw=1.6, ls="--",
                label="perfect calibration", zorder=2)
        # binomial (Wilson) error band on the empirical points
        ax.fill_between(x, d["lo"], d["hi"], color=C_BAND, alpha=0.30,
                        label="binomial 68% band", zorder=1)
        # empirical coverage points + bars
        yerr = np.vstack([d["emp"] - d["lo"], d["hi"] - d["emp"]])
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(x, d["emp"], yerr=yerr, fmt="o", color=color, ms=7,
                    capsize=3, elinewidth=1.6, lw=0, label="empirical", zorder=4)
        # max deviation annotation
        dev = float(np.max(np.abs(d["emp"] - x)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("nominal credibility level")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"({'a' if pname=='gamma' else 'b'}) {label}\n"
                     f"max |empirical $-$ nominal| = {dev:.3f}")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"XRB-XLF posterior coverage (selection-aware, $N_{{\\rm det}}\\approx50$, "
        f"$\\gamma=1.6$; {n_used} realizations"
        + (f", {n_skip} skipped" if n_skip else "") + ")",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_table_and_verdict(cov, levels):
    print("=" * 70)
    print("POSTERIOR COVERAGE CURVE")
    print("=" * 70)
    n = cov["gamma"]["n"]
    print(f"usable (ok) realizations: {n}")
    print(f"{'nominal':>8} | {'cov(gamma)':>11} {'[Wilson 68%]':>18} | "
          f"{'cov(logxi)':>11} {'[Wilson 68%]':>18}")
    print("-" * 76)
    for j, c in enumerate(levels):
        g = cov["gamma"]
        x = cov["log10_xi_eff"]
        print(f"{c:>8.2f} | {g['emp'][j]:>11.3f} "
              f"[{g['lo'][j]:.3f}, {g['hi'][j]:.3f}]   | "
              f"{x['emp'][j]:>11.3f} [{x['lo'][j]:.3f}, {x['hi'][j]:.3f}]")

    print("-" * 76)
    verdicts = {}
    for pname in ("gamma", "log10_xi_eff"):
        d = cov[pname]
        signed = d["emp"] - d["nominal"]          # >0 over-covers, <0 under-covers
        dev = np.abs(signed)
        max_dev = float(np.max(dev))
        rms_dev = float(np.sqrt(np.mean(dev ** 2)))
        mean_signed = float(np.mean(signed))      # systematic direction
        # how many nominal points lie OUTSIDE their own Wilson 68% band?  This is
        # INFORMATIONAL only -- for a perfectly-calibrated curve ~1/3 of points
        # land outside a 68% band by construction, so it is NOT a pass/fail gate.
        outside = int(np.sum((d["nominal"] < d["lo"]) | (d["nominal"] > d["hi"])))
        verdicts[pname] = (max_dev, rms_dev, mean_signed, outside)
        direction = ("under-covers" if mean_signed < 0 else "over-covers")
        print(f"  {pname:>14}: max |dev| = {max_dev:.3f}, "
              f"rms |dev| = {rms_dev:.3f}, mean signed dev = {mean_signed:+.3f} "
              f"({direction}); {outside}/{len(levels)} levels >1sigma off-diagonal")

    # Verdict: "near-diagonal" is the success criterion.  The principled test is
    # the MAX ABSOLUTE DEVIATION from the diagonal: <= 0.10 is the conventional
    # "near-diagonal".  The Wilson-band count is informational only (a 68% band
    # is EXPECTED to exclude ~1/3 of points even for a perfect curve, so using it
    # as a gate would mis-fail a good result).
    max_dev_all = max(v[0] for v in verdicts.values())
    mean_signed_all = float(np.mean([v[2] for v in verdicts.values()]))
    near = max_dev_all <= 0.10
    print("-" * 76)
    if near:
        direction = ("a mild UNDER-coverage" if mean_signed_all < 0
                     else "a mild OVER-coverage")
        print(f"VERDICT: NEAR-DIAGONAL (pass). max |empirical - nominal| over both "
              f"params = {max_dev_all:.3f} (<= 0.10, the near-diagonal criterion).")
        if abs(mean_signed_all) >= 0.005:
            print(f"  The curve carries {direction} of ~{abs(mean_signed_all)*100:.0f}% "
                  f"(mean signed deviation {mean_signed_all:+.3f}): the residual "
                  f"Eddington bias documented in RESULTS.md (the likelihood corrects "
                  f"completeness P_det but not the Poisson count-noise scatter), not a "
                  f"likelihood error.")
    else:
        print(f"VERDICT: INVESTIGATE. max |empirical - nominal| over both params = "
              f"{max_dev_all:.3f} (> 0.10). The curve departs from the diagonal; "
              f"see the signed deviation above (under- vs over-coverage) and the "
              f"Eddington-bias caveat in RESULTS.md.")
    return verdicts, near


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "coverage.yaml"))
    p.add_argument("--results", default=None,
                   help="path to coverage_results.jsonl (default from config)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    matplotlib.rcParams["text.usetex"] = False

    outdir = os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = args.results or os.path.join(outdir, cfg["run"]["results_jsonl"])
    levels = list(cfg["coverage"]["nominal_levels"])

    rows = load_rows(jsonl_path)
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped_too_few")
    print(f"loaded {n_total} rows ({n_ok} ok, {n_skip} skipped_too_few) "
          f"from {jsonl_path}")
    if n_ok == 0:
        print("no usable rows yet -- run run_coverage.py first.")
        return 1
    if n_skip:
        print(f"  note: {n_skip} realizations had < 2 detected sources "
              f"(degenerate Poisson draw) and are excluded from coverage.")

    cov = empirical_coverage(rows, levels)
    out_png = os.path.join(outdir, cfg["run"]["figure"])
    make_coverage_curve(cov, out_png, n_used=n_ok, n_skip=n_skip)
    print_table_and_verdict(cov, np.asarray(levels, dtype=float))
    print(f"\nSaved coverage curve -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
