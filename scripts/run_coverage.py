r"""Phase 4: the dedicated posterior-coverage run.

The recovery suite (Phase 3a) stored only the 68% credible interval (q16/q84)
per fit, so it cannot build a coverage CURVE across credibility levels.  This
script runs many INDEPENDENT selection-aware fits at one representative config
(N_det ~ 50, gamma = 1.6) and, for each fit, records the central credible
interval [q_lo, q_hi] of EACH parameter at a ladder of nominal levels
{0.1, ..., 0.9}.  The analysis (``analyze_coverage.py``) then asks, per nominal
level c: what fraction of realizations had their c% central interval contain the
truth?  A calibrated likelihood gives empirical ~ nominal, a near-diagonal
coverage curve (the success criterion for this stage).

Why a dedicated run and not the suite rows
------------------------------------------
Coverage at level c needs the central interval [(1-c)/2, (1+c)/2] of the
posterior; the suite only saved the 0.16/0.84 quantiles.  Re-deriving multi-level
quantiles requires the posterior SAMPLES, which the suite did not persist (only
summaries).  Rather than re-run the whole 1500-fit grid, this run does >= 200
selection-aware fits at the single representative cell, enough to resolve the
multi-level coverage curve at that configuration.

Cross-link
----------
This is the SAME coverage methodology used by the sibling repo
sbi-xray-calibration (its expected-coverage test in src/sbixcal/calibrate.py):
draw datasets from the model, fit, and check that the q% credible region
contains the truth q% of the time.  Here the "model" is the XRB-XLF forward
model and the fit is the unbinned Poisson-process likelihood.

Crash-resumability
------------------
Every completed fit is appended as one JSON line to
``outputs/recovery/coverage_results.jsonl``, keyed by ``realization``.  On
restart we read existing keys and skip them.  JSONL is line-atomic: an interruption
can at worst truncate the final line, which is detected and dropped on the next
read.  Each row is re-runnable from its seed alone.

Parallelism (Windows-safe)
--------------------------
Fits run in a ``ProcessPoolExecutor`` with at most ``max_workers`` (default 6,
capped to bound memory use).  The worker
and its arguments are top-level and picklable (Windows spawn).  Only the MAIN
process writes the JSONL file.  Everything is under ``if __name__ == "__main__"``.

Usage
-----
    .venv\Scripts\python.exe scripts\run_coverage.py \
        --config configs\coverage.yaml [--n-realizations N] [--workers K]

Resume: just run the same command again -- completed keys are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

# Mute UltraNest's per-fit logging in every worker (same reason as the suite).
for _name in ("ultranest", "ultranest.solvecompat"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False
    _lg.handlers = [logging.NullHandler()]

from xlf_model.forward import (  # noqa: E402
    DetectorPreset,
    detector_from_config,
    run_forward,
)
from xlf_model.inference import (  # noqa: E402
    LogLGrid,
    ObservationModel,
    fit_xlf,
    make_hmxb_problem,
)
from xlf_model.xlf import HMXBXLF  # noqa: E402


# ---------------------------------------------------------------------------
# Tuning denominator: expected detected count at SFR = 1 (selection-aware)
# ---------------------------------------------------------------------------
def lambda_unit(gamma, xi, L_cut, L_min, L_ref, distance_Mpc, detector,
                absorption_flux_factor, grid):
    """E[N_det] at SFR=1 for the given gamma -- the tuning denominator."""
    xlf = HMXBXLF(xi=xi, gamma=gamma, L_cut=L_cut, L_min=L_min, L_ref=L_ref,
                  SFR=1.0)
    obs = ObservationModel(
        distance_Mpc=distance_Mpc, detector=detector,
        absorption_flux_factor=absorption_flux_factor, selection_aware=True,
    )
    L = grid.grid(xlf.L_min, xlf.L_cut)
    lam = np.asarray(xlf.dN_dL(L), dtype=float) * obs.p_det(L)
    return float(np.trapezoid(lam, L))


# ---------------------------------------------------------------------------
# A self-contained task spec for one realization (must be picklable)
# ---------------------------------------------------------------------------
@dataclass
class CovSpec:
    realization: int
    sampler: str
    sim_seed: int
    fit_seed: int
    SFR: float
    gamma: float
    xi_true: float
    L_cut: float
    L_min: float
    L_ref: float
    distance_Mpc: float
    detector: dict
    absorption_flux_factor: float
    priors: dict
    points_per_dex: int
    n_live: int
    nominal_levels: list


def _detector_from_spec(d: dict) -> DetectorPreset:
    return DetectorPreset(
        name=d["name"],
        flux_limit_50=float(d["flux_limit_50"]),
        completeness_width_dex=float(d["completeness_width_dex"]),
        exposure_s=float(d["exposure_s"]),
        cts_to_flux=float(d["cts_to_flux"]),
        band=str(d.get("band", "")),
    )


def _cfg_for_problem(spec: CovSpec) -> dict:
    return {
        "L_unit": spec.L_ref,
        "hmxb": {
            "preset": "mineo12",
            "mineo12": {
                "band": "0.5-8 keV",
                "xi": spec.xi_true,
                "gamma": spec.gamma,
                "L_cut": spec.L_cut,
            },
        },
        "forward": {
            "L_min": spec.L_min,
            "absorption_flux_factor": spec.absorption_flux_factor,
            "detectors": {spec.detector["name"]: spec.detector},
        },
        "inference": {
            "priors": spec.priors,
            "grid": {"points_per_dex": spec.points_per_dex},
            "sampler": {"n_live": spec.n_live},
        },
    }


def _central_levels_to_quantiles(levels):
    """Map central-credibility levels c -> the (lo, hi) quantile pair.

    A central credible interval at level c spans quantiles [(1-c)/2, (1+c)/2].
    Returns parallel arrays (qlo, qhi) the same length as ``levels``.
    """
    levels = np.asarray(levels, dtype=float)
    qlo = 0.5 * (1.0 - levels)
    qhi = 0.5 * (1.0 + levels)
    return qlo, qhi


# ---------------------------------------------------------------------------
# THE WORKER (top-level, picklable) -- simulate one galaxy + fit + record edges
# ---------------------------------------------------------------------------
def run_one_realization(spec: CovSpec) -> dict:
    """Simulate one galaxy, fit selection-aware, record central-interval edges.

    Deterministic from ``spec.sim_seed`` (galaxy) and ``spec.fit_seed``
    (sampler).  For each parameter and each nominal level c we store the central
    interval [q_lo, q_hi] so the analysis can test truth-containment.
    """
    detector = _detector_from_spec(spec.detector)
    L_ref = spec.L_ref

    xlf = HMXBXLF(
        xi=spec.xi_true, gamma=spec.gamma, L_cut=spec.L_cut,
        L_min=spec.L_min, L_ref=L_ref, SFR=spec.SFR,
    )
    rng = np.random.default_rng(spec.sim_seed)
    res = run_forward(
        xlf, spec.distance_Mpc, detector, rng,
        absorption_flux_factor=spec.absorption_flux_factor, component="HMXB",
    )
    L_obs = res.table["L_obs"]
    L_obs = L_obs[L_obs > 0]
    n_det_actual = int(L_obs.size)

    truth = {
        "log10_xi_eff": float(np.log10(spec.xi_true * spec.SFR)),
        "gamma": float(spec.gamma),
    }

    row = {
        "realization": int(spec.realization),
        "sampler": spec.sampler,
        "fitter": "selection-aware",
        "SFR": float(spec.SFR),
        "gamma_true": float(spec.gamma),
        "xi_true": float(spec.xi_true),
        "log10_xi_eff_true": truth["log10_xi_eff"],
        "distance_Mpc": float(spec.distance_Mpc),
        "detector": detector.name,
        "n_det_actual": n_det_actual,
        "sim_seed": int(spec.sim_seed),
        "fit_seed": int(spec.fit_seed),
        "nominal_levels": list(map(float, spec.nominal_levels)),
    }

    # selection-aware sees ALL detected sources (no hard cut)
    if L_obs.size < 2:
        row.update({"status": "skipped_too_few", "logZ": None,
                    "wall_time_s": 0.0})
        return row

    cfg = _cfg_for_problem(spec)
    like = make_hmxb_problem(
        L_obs, cfg, distance_Mpc=spec.distance_Mpc, detector=detector,
        absorption_flux_factor=spec.absorption_flux_factor,
        selection_aware=True,
        grid=LogLGrid(points_per_dex=spec.points_per_dex),
    )
    fr = fit_xlf(like, cfg, sampler=spec.sampler, seed=spec.fit_seed)

    qlo, qhi = _central_levels_to_quantiles(spec.nominal_levels)
    gi = fr.param_names.index("gamma")
    xi = fr.param_names.index("log10_xi_eff")
    g_samples = fr.samples[:, gi]
    x_samples = fr.samples[:, xi]

    # central-interval edges at each nominal level, per parameter
    g_lo = np.quantile(g_samples, qlo).tolist()
    g_hi = np.quantile(g_samples, qhi).tolist()
    x_lo = np.quantile(x_samples, qlo).tolist()
    x_hi = np.quantile(x_samples, qhi).tolist()

    row.update({
        "status": "ok",
        "logZ": (None if fr.logZ is None else float(fr.logZ)),
        "wall_time_s": float(fr.wall_time_s),
        "gamma_med": float(np.median(g_samples)),
        "log_xi_med": float(np.median(x_samples)),
        # central-interval edges (parallel to nominal_levels)
        "gamma_ci_lo": [float(v) for v in g_lo],
        "gamma_ci_hi": [float(v) for v in g_hi],
        "log_xi_ci_lo": [float(v) for v in x_lo],
        "log_xi_ci_hi": [float(v) for v in x_hi],
    })
    return row


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------
def load_done_realizations(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
        done.add(int(row["realization"]))
    return done


def build_specs(cfg: dict, n_realizations: int, sampler: str) -> list:
    cov = cfg["coverage"]
    gal = cfg["galaxy"]
    fwd = cfg["forward"]
    L_ref = float(cfg["L_unit"])
    L_min = float(fwd["L_min"])
    L_cut = float(gal["L_cut"])
    xi_true = float(gal["xi_true"])
    distance = float(gal["distance_Mpc"])
    gamma = float(cov["gamma_true"])
    n_target = int(cov["n_det_target"])
    levels = list(cov["nominal_levels"])
    absorption = float(gal.get("absorption_flux_factor",
                               fwd.get("absorption_flux_factor", 1.0)))
    det = detector_from_config(cfg, name=gal["detector"])
    det_dict = {
        "name": det.name, "flux_limit_50": det.flux_limit_50,
        "completeness_width_dex": det.completeness_width_dex,
        "exposure_s": det.exposure_s, "cts_to_flux": det.cts_to_flux,
        "band": det.band,
    }
    priors = cfg["inference"]["priors"]
    ppd = int(cfg["inference"]["grid"]["points_per_dex"])
    n_live = int(cfg["inference"]["sampler"]["n_live"])
    seed_base = int(cfg["seed_base"])

    grid_integrator = LogLGrid(points_per_dex=ppd)
    lam_u = lambda_unit(gamma, xi_true, L_cut, L_min, L_ref, distance, det,
                        absorption, grid_integrator)
    SFR = float(n_target) / float(lam_u)

    specs = []
    for r in range(n_realizations):
        # distinct seed namespace from the recovery suite (offset 7e6) so the
        # coverage realizations are independent draws, not a re-use of suite ones
        sim_seed = seed_base + 7_000_000 + r
        fit_seed = sim_seed + 1
        specs.append(CovSpec(
            realization=int(r),
            sampler=str(sampler),
            sim_seed=int(sim_seed),
            fit_seed=int(fit_seed),
            SFR=float(SFR),
            gamma=float(gamma),
            xi_true=xi_true,
            L_cut=L_cut,
            L_min=L_min,
            L_ref=L_ref,
            distance_Mpc=distance,
            detector=det_dict,
            absorption_flux_factor=absorption,
            priors=priors,
            points_per_dex=ppd,
            n_live=n_live,
            nominal_levels=levels,
        ))
    return specs, SFR, lam_u


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "coverage.yaml"))
    p.add_argument("--n-realizations", type=int, default=None,
                   help="override config n_realizations")
    p.add_argument("--workers", type=int, default=None,
                   help="override config max_workers (cap 6)")
    p.add_argument("--limit", type=int, default=None,
                   help="run at most this many NEW fits then stop (pilots)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    sampler = str(cfg["run"]["sampler"])
    n_real = args.n_realizations or int(cfg["coverage"]["n_realizations"])
    workers = args.workers or int(cfg["run"]["max_workers"])
    workers = max(1, min(workers, 6))           # hard cap at 6
    outdir = os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = os.path.join(outdir, cfg["run"]["results_jsonl"])

    specs, SFR, lam_u = build_specs(cfg, n_real, sampler)
    done = load_done_realizations(jsonl_path)
    todo = [s for s in specs if s.realization not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print("=" * 70)
    print("XRB-XLF posterior-coverage run (Phase 4)")
    print(f"  config: N_det_target={cfg['coverage']['n_det_target']} "
          f"gamma={cfg['coverage']['gamma_true']} selection-aware")
    print(f"  tuned SFR = {SFR:.3f} Msun/yr  (Lambda_unit = {lam_u:.3f})")
    print(f"  nominal levels: {cfg['coverage']['nominal_levels']}")
    print(f"  realizations: {n_real}   already done: {len(done)}   "
          f"to run now: {len(todo)}")
    print(f"  sampler: {sampler}   workers: {workers}")
    print(f"  results -> {jsonl_path}")
    print("=" * 70)
    if not todo:
        print("Nothing to do -- all realizations already complete.")
        return 0

    t0 = time.perf_counter()
    n_written = 0
    walls = []
    with open(jsonl_path, "a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one_realization, s): s for s in todo}
            for fut in as_completed(futs):
                row = fut.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())
                n_written += 1
                if row.get("status") == "ok":
                    walls.append(row["wall_time_s"])
                if n_written % 10 == 0 or n_written == len(todo):
                    elapsed = time.perf_counter() - t0
                    rate = n_written / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - n_written) / rate if rate > 0 else 0.0
                    print(f"  [{n_written:4d}/{len(todo)}] "
                          f"r={row['realization']} Nact={row.get('n_det_actual')} "
                          f"st={row.get('status')} | {elapsed:6.1f}s elapsed, "
                          f"ETA {eta:6.1f}s")

    elapsed = time.perf_counter() - t0
    print("-" * 70)
    print(f"wrote {n_written} fits in {elapsed:.1f}s "
          f"({elapsed / max(n_written,1):.2f}s/fit wall, {workers} workers)")
    if walls:
        walls = np.array(walls)
        print(f"per-fit sampler wall-clock: mean={walls.mean():.2f}s "
              f"median={np.median(walls):.2f}s max={walls.max():.2f}s "
              f"(n_ok={walls.size})")
    print(f"resume command: .venv\\Scripts\\python.exe scripts\\run_coverage.py "
          f"--config {os.path.relpath(args.config, _REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
