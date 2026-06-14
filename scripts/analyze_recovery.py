r"""Phase 3 (a): analyse the recovery suite and draw the money plot.

Reads ``outputs/recovery/results.jsonl`` (produced by ``run_recovery_suite.py``)
and, for every (N_det target, true gamma, fitter) cell, computes the three
recovery diagnostics:

  * median bias of the recovered slope:  median_r(gamma_med - gamma_true);
  * 68% scatter of the recovered slope:  half the 16-84 percentile spread of
    gamma_med across the cell's realizations;
  * empirical coverage:  the fraction of realizations whose 68% credible
    interval [q16, q84] of gamma contains the truth (nominal 0.68).

Then it draws the MONEY-PLOT DRAFT (two panels):

  (a) recovered gamma (median +- 68% scatter) vs N_det for BOTH fitters, with
      the truth line, at gamma = 1.6 (the central slope);
  (b) empirical coverage vs N_det per fitter, with the nominal 0.68 line.

Output (final): ``outputs/recovery/money_plot.png``
(dpi 220, colorblind-safe Okabe-Ito).  A draft copy is also kept at
``outputs/recovery/money_plot_draft.png`` for provenance.  A per-cell summary
table is written to ``outputs/recovery/recovery_summary.csv`` and the summary
numbers (Delta-gamma bias at N_det=5/15/50, naive vs aware; the N_det where the
naive bias exceeds the aware 68% scatter; coverage) are printed to stdout.

Failed/skipped fits (degenerate Poisson draws with < 2 fittable sources) are
counted and reported with their status.  Each summarized cell reports its
actual usable n (which can be < n_realizations at the smallest N_det target).

This script works on a PARTIAL results file (whatever rows exist) so it can be
run while the suite is still completing; re-run it after the full run.

Usage
-----
    .venv\Scripts\python.exe scripts\analyze_recovery.py --config configs\recovery_suite.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# colorblind-safe (Okabe-Ito)
C_AWARE = "#0072B2"     # blue       -- selection-aware
C_NAIVE = "#D55E00"     # vermillion -- naive
C_TRUTH = "#000000"     # black      -- truth
C_NOM = "#999999"       # grey       -- nominal coverage


def load_rows(jsonl_path):
    """Read result rows, tolerant of a truncated final line (crash safety)."""
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


def percentile_scatter(x):
    """Half the 16-84 percentile spread (robust 1-sigma-equivalent scatter)."""
    if len(x) < 2:
        return np.nan
    lo, hi = np.percentile(x, [16, 84])
    return 0.5 * (hi - lo)


def report_failures(rows):
    """Count and describe failed/skipped fits per (N_det target, fitter).

    Returns ``(n_total, n_ok, n_skip, n_other, breakdown)`` where ``breakdown``
    maps ``(n_det_target, fitter) -> n_skipped`` for the degenerate-draw skips.
    Nothing is dropped silently: any status other than ``ok`` is accounted for.
    """
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped_too_few")
    n_other = n_total - n_ok - n_skip
    breakdown = defaultdict(int)
    other_rows = []
    for r in rows:
        st = r.get("status")
        if st == "skipped_too_few":
            breakdown[(int(r["n_det_target"]), str(r["fitter"]))] += 1
        elif st != "ok":
            other_rows.append(r)
    return n_total, n_ok, n_skip, n_other, dict(breakdown), other_rows


def summarize(rows):
    """Group OK rows by (N_det_target, gamma_true, fitter) -> diagnostics."""
    cells = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (int(r["n_det_target"]), round(float(r["gamma_true"]), 4),
               str(r["fitter"]))
        cells[key].append(r)

    summary = {}
    for key, rs in cells.items():
        n_target, gamma_true, fitter = key
        gmed = np.array([r["gamma_med"] for r in rs], dtype=float)
        q16 = np.array([r["gamma_q16"] for r in rs], dtype=float)
        q84 = np.array([r["gamma_q84"] for r in rs], dtype=float)
        nact = np.array([r["n_det_actual"] for r in rs], dtype=float)

        bias = float(np.median(gmed - gamma_true))
        scatter = float(percentile_scatter(gmed))
        # empirical coverage of the 68% interval
        covered = (q16 <= gamma_true) & (gamma_true <= q84)
        coverage = float(np.mean(covered))
        # binomial standard error on the coverage estimate
        n = len(rs)
        cov_err = float(np.sqrt(coverage * (1 - coverage) / n)) if n > 0 else np.nan

        summary[key] = {
            "n_det_target": n_target,
            "gamma_true": gamma_true,
            "fitter": fitter,
            "n_real": n,
            "n_det_actual_median": float(np.median(nact)),
            "gamma_med_median": float(np.median(gmed)),
            "bias": bias,
            "scatter68": scatter,
            "coverage68": coverage,
            "coverage_err": cov_err,
        }
    return summary


def write_csv(summary, out_csv):
    fields = ["n_det_target", "gamma_true", "fitter", "n_real",
              "n_det_actual_median", "gamma_med_median", "bias", "scatter68",
              "coverage68", "coverage_err"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in sorted(summary):
            w.writerow(summary[key])


def make_money_plot(summary, gamma_panel, out_png):
    """Two-panel money-plot draft at the central gamma."""
    targets = sorted({k[0] for k in summary})
    fitters = ["selection-aware", "naive"]
    colors = {"selection-aware": C_AWARE, "naive": C_NAIVE}
    markers = {"selection-aware": "o", "naive": "s"}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    # ---- panel (a): recovered gamma vs N_det at gamma = gamma_panel ----
    g = round(float(gamma_panel), 4)
    for fitter in fitters:
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for nt in targets:
            key = (nt, g, fitter)
            if key not in summary:
                continue
            s = summary[key]
            xs.append(s["n_det_actual_median"])
            ys.append(s["gamma_med_median"])
            # +-68% scatter band of the per-realization medians
            yerr_lo.append(s["scatter68"])
            yerr_hi.append(s["scatter68"])
        if not xs:
            continue
        xs = np.array(xs)
        # slight horizontal dodge so the two fitters' error bars don't overlap
        dodge = 1.04 if fitter == "naive" else 0.96
        axa.errorbar(
            xs * dodge, ys, yerr=[yerr_lo, yerr_hi],
            fmt=markers[fitter], color=colors[fitter], ms=7, lw=1.6,
            capsize=3, elinewidth=1.6, label=fitter, zorder=4,
        )
    axa.axhline(g, color=C_TRUTH, lw=1.8, ls="--", label=fr"truth $\gamma={g}$",
                zorder=2)
    axa.set_xscale("log")
    axa.set_xlabel(r"number of detected HMXBs  $N_{\rm det}$")
    axa.set_ylabel(r"recovered slope  $\gamma$  (median $\pm$ 68% scatter)")
    axa.set_title(f"(a) slope recovery vs $N_{{\\rm det}}$ "
                  f"(true $\\gamma = {g}$)")
    axa.legend(loc="best", fontsize=10, framealpha=0.92)
    axa.grid(True, which="both", alpha=0.2)

    # ---- panel (b): coverage vs N_det per fitter (at the same gamma) ----
    for fitter in fitters:
        xs, cov, cov_err = [], [], []
        for nt in targets:
            key = (nt, g, fitter)
            if key not in summary:
                continue
            s = summary[key]
            xs.append(s["n_det_actual_median"])
            cov.append(s["coverage68"])
            cov_err.append(s["coverage_err"])
        if not xs:
            continue
        xs = np.array(xs)
        dodge = 1.04 if fitter == "naive" else 0.96
        axb.errorbar(
            xs * dodge, cov, yerr=cov_err, fmt=markers[fitter],
            color=colors[fitter], ms=7, lw=1.6, capsize=3, elinewidth=1.6,
            label=fitter, zorder=4,
        )
    axb.axhline(0.68, color=C_NOM, lw=1.8, ls="--",
                label="nominal 68%", zorder=2)
    axb.set_xscale("log")
    axb.set_ylim(0.0, 1.02)
    axb.set_xlabel(r"number of detected HMXBs  $N_{\rm det}$")
    axb.set_ylabel("empirical coverage of the 68% interval")
    axb.set_title(f"(b) coverage vs $N_{{\\rm det}}$ (true $\\gamma = {g}$)")
    axb.legend(loc="best", fontsize=10, framealpha=0.92)
    axb.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "XRB-XLF recovery suite: selection-aware vs naive HMXB-slope fitting",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_headlines(summary, gamma_panel):
    g = round(float(gamma_panel), 4)
    print("=" * 78)
    print("RECOVERY SUITE SUMMARY")
    print("=" * 78)
    header = (f"{'N_tgt':>6} {'gamma':>6} {'fitter':>16} {'n':>4} "
              f"{'N_act':>7} {'bias':>9} {'scat68':>8} {'cov68':>7} {'cov_err':>7}")
    print(header)
    print("-" * len(header))
    for key in sorted(summary):
        s = summary[key]
        print(f"{s['n_det_target']:>6} {s['gamma_true']:>6.2f} "
              f"{s['fitter']:>16} {s['n_real']:>4} "
              f"{s['n_det_actual_median']:>7.0f} {s['bias']:>+9.4f} "
              f"{s['scatter68']:>8.4f} {s['coverage68']:>7.2f} "
              f"{s['coverage_err']:>7.2f}")

    # summary: Delta-gamma bias at N_det=5/15/50, naive vs aware, at gamma=1.6
    print("\n" + "=" * 78)
    print("SUMMARY NUMBERS")
    print("=" * 78)
    targets = sorted({k[0] for k in summary})
    for nt in (5, 15, 50):
        ka = (nt, g, "selection-aware")
        kn = (nt, g, "naive")
        if ka in summary and kn in summary:
            ba = summary[ka]["bias"]
            bn = summary[kn]["bias"]
            ca = summary[ka]["coverage68"]
            cn = summary[kn]["coverage68"]
            print(f"  N_det={nt:>3} (gamma={g}):")
            print(f"    selection-aware  bias(gamma) = {ba:+.4f}   "
                  f"coverage = {ca:.2f}")
            print(f"    naive            bias(gamma) = {bn:+.4f}   "
                  f"coverage = {cn:.2f}")
            print(f"    => |naive bias| - |aware bias| = "
                  f"{abs(bn) - abs(ba):+.4f}")

    # crossover: the N_det at which |naive bias| first exceeds the AWARE 68%
    # scatter -- i.e. where the naive systematic dominates the statistical error
    print(f"\n  Crossover (gamma={g}): N_det where |naive bias| > aware 68% "
          f"scatter")
    crossed = None
    for nt in targets:
        ka = (nt, g, "selection-aware")
        kn = (nt, g, "naive")
        if ka not in summary or kn not in summary:
            continue
        scat_a = summary[ka]["scatter68"]
        bias_n = abs(summary[kn]["bias"])
        flag = ""
        if bias_n > scat_a:
            flag = "  <-- naive bias exceeds aware scatter"
            if crossed is None:
                crossed = nt
        print(f"    N_det~{nt:>3}: |naive bias|={bias_n:.4f}  "
              f"vs aware scat68={scat_a:.4f}{flag}")
    if crossed is not None:
        print(f"    => first crossover at N_det ~ {crossed}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "recovery_suite.yaml"))
    p.add_argument("--results", default=None,
                   help="path to results.jsonl (default from config)")
    p.add_argument("--gamma-panel", type=float, default=1.6,
                   help="true gamma for the money-plot panels (default 1.6)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    matplotlib.rcParams["text.usetex"] = False

    outdir = os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = args.results or os.path.join(outdir, cfg["run"]["results_jsonl"])

    rows = load_rows(jsonl_path)
    n_total, n_ok, n_skip, n_other, breakdown, other_rows = report_failures(rows)
    print(f"loaded {n_total} rows ({n_ok} ok, {n_skip} skipped_too_few, "
          f"{n_other} other) from {jsonl_path}")

    # ---- explicit failed/skipped-fit accounting (counted with their status) ----
    print("-" * 78)
    print("FAILED / SKIPPED FITS (counted, not dropped)")
    print("-" * 78)
    if n_skip == 0 and n_other == 0:
        print("  none -- all fits succeeded.")
    else:
        if breakdown:
            print(f"  {n_skip} 'skipped_too_few' (degenerate Poisson draw, "
                  f"< 2 fittable sources -> a 2-parameter XLF is unconstrained):")
            for (nt, fitter) in sorted(breakdown):
                print(f"    N_det_target={nt:>3} {fitter:>16}: "
                      f"{breakdown[(nt, fitter)]} skipped")
            print("  (these reduce the usable n in the affected cell; the "
                  "per-cell 'n' column reflects it.)")
        if n_other:
            print(f"  {n_other} rows with an UNEXPECTED status -- investigate:")
            for r in other_rows[:10]:
                print(f"    {r.get('status')!r} at N_det_target="
                      f"{r.get('n_det_target')} gamma={r.get('gamma_true')} "
                      f"fitter={r.get('fitter')} realization={r.get('realization')}")
    if n_ok == 0:
        print("no usable rows yet -- run the suite first.")
        return 1

    summary = summarize(rows)
    out_csv = os.path.join(outdir, "recovery_summary.csv")
    write_csv(summary, out_csv)
    # final money plot + a draft copy kept for provenance
    out_png = os.path.join(outdir, "money_plot.png")
    out_draft = os.path.join(outdir, "money_plot_draft.png")
    make_money_plot(summary, args.gamma_panel, out_png)
    make_money_plot(summary, args.gamma_panel, out_draft)
    print_headlines(summary, args.gamma_panel)
    print(f"\nSaved summary table -> {out_csv}")
    print(f"Saved money plot     -> {out_png}")
    print(f"Saved draft copy     -> {out_draft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
