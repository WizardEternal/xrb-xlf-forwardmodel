r"""Run the whole XRB-XLF pipeline end to end, with stage-by-stage accounting.

Chains every stage of the repo in order and prints a funnel-style progress line
per stage (sources/fits in, artifacts out).
Each stage is also runnable standalone (see its own script); this is the single
``run_all`` entry point for the repo.

Stages
------
  1. forward demo        scripts/run_forward_demo.py   -> demo funnel + demo png
  2. single-galaxy fit   scripts/run_fit_single.py     -> corner + XLF overlay
  3. recovery suite      scripts/run_recovery_suite.py -> results.jsonl   [EXPENSIVE]
  4. analyze recovery    scripts/analyze_recovery.py   -> money_plot.png
  5. coverage run        scripts/run_coverage.py       -> coverage_results.jsonl [EXPENSIVE]
  6. analyze coverage    scripts/analyze_coverage.py   -> coverage_curve.png
  7. L_X-SFR nonlinearity scripts/run_lxsfr_demo.py     -> lxsfr_nonlinearity.png
  8. hierarchical stack  scripts/run_hierarchical.py   -> hierarchical_stack.png

The two EXPENSIVE stages (3, 5) are the parallel fitting runs.  Both are
crash-resumable: re-running skips already-completed fits, so by default
``run_all`` includes them and they no-op if their results tables are already
complete.  Use ``--skip-fits`` to run only the cheap stages + re-draw the
analysis figures from existing tables (the fast "regenerate everything" path).

Worker cap: the fitting stages honour ``--workers`` (capped at 6 to bound
memory use).

Usage
-----
    # everything (resumes the fits if their tables are incomplete):
    .venv\Scripts\python.exe scripts\run_all.py

    # cheap stages + redraw figures from existing result tables:
    .venv\Scripts\python.exe scripts\run_all.py --skip-fits

    # cap parallel fitting at K workers:
    .venv\Scripts\python.exe scripts\run_all.py --workers 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, os.path.join(_REPO, "src"))

import analyze_coverage  # noqa: E402
import analyze_recovery  # noqa: E402
import run_coverage  # noqa: E402
import run_fit_single  # noqa: E402
import run_forward_demo  # noqa: E402
import run_hierarchical  # noqa: E402
import run_lxsfr_demo  # noqa: E402
import run_recovery_suite  # noqa: E402


def _cfg(name):
    return os.path.join(_REPO, "configs", name)


def _count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def stage(n, title):
    print()
    print("#" * 76)
    print(f"# STAGE {n}: {title}")
    print("#" * 76)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-fits", action="store_true",
                   help="skip the two EXPENSIVE parallel fitting stages; only "
                        "run the cheap stages and redraw analysis figures from "
                        "existing result tables")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel workers for the fitting stages (cap 6)")
    args = p.parse_args(argv)
    workers = max(1, min(args.workers, 6))

    t0 = time.perf_counter()
    rec_jsonl = os.path.join(_REPO, "outputs", "recovery", "results.jsonl")
    cov_jsonl = os.path.join(_REPO, "outputs", "recovery", "coverage_results.jsonl")

    # 1. forward demo --------------------------------------------------------
    stage(1, "forward-model demo (funnel accounting + demo figure)")
    run_forward_demo.main(["--config", _cfg("xlf_defaults.yaml")])

    # 2. single-galaxy fit ---------------------------------------------------
    stage(2, "single-galaxy recovery (aware vs naive corner + XLF overlay)")
    run_fit_single.main(["--config", _cfg("fit_single.yaml")])

    # 3. recovery suite (EXPENSIVE) -----------------------------------------
    stage(3, "recovery suite -- the money-plot fits  [EXPENSIVE, resumable]")
    if args.skip_fits:
        have = _count_jsonl(rec_jsonl)
        print(f"  --skip-fits: leaving recovery suite as-is "
              f"({have} fits already in results.jsonl).")
    else:
        before = _count_jsonl(rec_jsonl)
        run_recovery_suite.main(["--config", _cfg("recovery_suite.yaml"),
                                 "--workers", str(workers)])
        after = _count_jsonl(rec_jsonl)
        print(f"  recovery-suite fits: {before} -> {after} "
              f"(+{after - before} this run)")

    # 4. analyze recovery -> money plot -------------------------------------
    stage(4, "analyze recovery -> money_plot.png")
    analyze_recovery.main(["--config", _cfg("recovery_suite.yaml")])

    # 5. coverage run (EXPENSIVE) -------------------------------------------
    stage(5, "posterior-coverage run  [EXPENSIVE, resumable]")
    if args.skip_fits:
        have = _count_jsonl(cov_jsonl)
        print(f"  --skip-fits: leaving coverage run as-is "
              f"({have} fits already in coverage_results.jsonl).")
    else:
        before = _count_jsonl(cov_jsonl)
        run_coverage.main(["--config", _cfg("coverage.yaml"),
                           "--workers", str(workers)])
        after = _count_jsonl(cov_jsonl)
        print(f"  coverage fits: {before} -> {after} (+{after - before} this run)")

    # 6. analyze coverage -> coverage curve ---------------------------------
    stage(6, "analyze coverage -> coverage_curve.png")
    analyze_coverage.main(["--config", _cfg("coverage.yaml")])

    # 7. L_X-SFR nonlinearity ------------------------------------------------
    stage(7, "L_X-SFR nonlinearity (Gilfanov+ 2004) -> lxsfr_nonlinearity.png")
    run_lxsfr_demo.main(["--config", _cfg("lxsfr_demo.yaml")])

    # 8. hierarchical stack --------------------------------------------------
    stage(8, "hierarchical survey stack -> hierarchical_stack.png")
    run_hierarchical.main(["--config", _cfg("hierarchical.yaml")])

    dt = time.perf_counter() - t0
    print()
    print("=" * 76)
    print(f"run_all complete in {dt/60.0:.1f} min")
    print("  committed README figures (the five synthetic figures):")
    for rel in ("outputs/recovery/money_plot.png",
                "outputs/recovery/coverage_curve.png",
                "outputs/diagnostics/lxsfr_nonlinearity.png",
                "outputs/diagnostics/hierarchical_stack.png",
                "outputs/diagnostics/demo_xlf_draw.png"):
        path = os.path.join(_REPO, rel)
        flag = "ok" if os.path.exists(path) else "MISSING"
        print(f"    [{flag:>7}] {rel}")
    print("  note: the sixth committed figure, real_galaxy_fit.png, is produced")
    print("        separately by run_real_demo.py (optional real-data extension).")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
