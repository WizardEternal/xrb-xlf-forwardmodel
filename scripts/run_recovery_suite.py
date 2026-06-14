"""Phase 3 (a): the single-galaxy recovery suite (the money-plot input).

For every cell of the grid

    target N_det  in  {5, 15, 50, 150, 500}
        x  true slope gamma  in  {1.4, 1.6, 1.8}
        x  fitter  in  {selection-aware, naive}
        x  realization  in  {0 .. n_realizations-1}

we simulate ONE galaxy tuned to yield ~ the target number of detected HMXBs,
fit its detected luminosities with the chosen likelihood, and record the
posterior median / 68% interval / std for (log10 xi_eff, gamma), plus the truth,
the actual N_det, logZ and the wall-clock time.  The analysis script
(``analyze_recovery.py``) turns these rows into the money plot.

The tuning rule (exact, closed-form)
------------------------------------
The expected detected count is

    Lambda(SFR) = integral dN/dL(L | xi, gamma, SFR) * P_det(L) dL
                = SFR * Lambda_unit(gamma, distance, detector)

because SFR multiplies the XLF normalization linearly.  So to hit a target
expected detected count we just divide:

    SFR_tuned = target_N_det / Lambda_unit(gamma, distance, detector).

This sets the *expected* N_det to the target; the *actual* N_det is a Poisson
draw recorded per fit.  ``Lambda_unit`` is computed once per (gamma) on the same
log-L grid the likelihood uses.

Crash-resumability
------------------
Every completed fit is appended as one JSON line to
``outputs/recovery/results.jsonl``, keyed by
``(n_det_target, gamma, fitter, realization, sampler)``.  On restart we read the
existing keys and skip them, so the suite resumes exactly where it stopped.
JSONL is line-atomic: an interruption can at worst truncate the final line, which is
detected and dropped on the next read.  Each row is also re-runnable from its
seed alone.

Parallelism (Windows-safe)
--------------------------
Fits run in a ``ProcessPoolExecutor`` with at most ``max_workers`` (default 6)
processes -- the worker function and its arguments are top-level and picklable
(Windows spawn).  Only the MAIN process writes the JSONL file (as futures
complete), so there is never a concurrent-write race.  The whole thing is under
the ``if __name__ == "__main__":`` guard.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\run_recovery_suite.py \\
        --config configs\\recovery_suite.yaml [--n-realizations N] [--workers K]

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

# Silence UltraNest's per-fit progress logging in every (worker) process: it
# attaches a StreamHandler bound to the real stderr at sampler-construction time,
# which the inference module's redirect_stderr cannot capture.  With ~1500 fits
# the chatter would bury the suite's own progress.  Disabling propagation +
# raising the level mutes it without touching the shared inference module.
for _name in ("ultranest", "ultranest.solvecompat"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False
    _lg.handlers = [logging.NullHandler()]

from xlf_model.forward import (  # noqa: E402
    DetectorPreset,
    detector_from_config,
    luminosity_to_flux,
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
# The cell key (defines crash-resume identity) and the result row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellKey:
    """Identity of one fit -- the crash-resume key."""

    n_det_target: int
    gamma: float
    fitter: str          # "selection-aware" | "naive"
    realization: int
    sampler: str

    def as_tuple(self):
        return (
            int(self.n_det_target),
            round(float(self.gamma), 6),
            str(self.fitter),
            int(self.realization),
            str(self.sampler),
        )


def _key_from_row(row: dict):
    return (
        int(row["n_det_target"]),
        round(float(row["gamma_true"]), 6),
        str(row["fitter"]),
        int(row["realization"]),
        str(row["sampler"]),
    )


# ---------------------------------------------------------------------------
# Lambda_unit: expected detected count at SFR=1 (the tuning denominator)
# ---------------------------------------------------------------------------
def lambda_unit(gamma, xi, L_cut, L_min, L_ref, distance_Mpc, detector,
                absorption_flux_factor, grid):
    """E[N_det] at SFR=1 for the given (gamma) -- the tuning denominator."""
    xlf = HMXBXLF(xi=xi, gamma=gamma, L_cut=L_cut, L_min=L_min, L_ref=L_ref,
                  SFR=1.0)
    obs = ObservationModel(
        distance_Mpc=distance_Mpc, detector=detector,
        absorption_flux_factor=absorption_flux_factor, selection_aware=True,
    )
    L = grid.grid(xlf.L_min, xlf.L_cut)
    lam = np.asarray(xlf.dN_dL(L), dtype=float) * obs.p_det(L)
    return float(np.trapezoid(lam, L))


def sfr_for_target(n_det_target, lam_unit):
    """SFR that makes E[N_det] equal the target (exact, linear in SFR)."""
    return float(n_det_target) / float(lam_unit)


# ---------------------------------------------------------------------------
# A self-contained task spec passed to a worker (must be picklable)
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    """Everything a worker needs to simulate one galaxy and run one fit.

    Held as plain Python scalars/dicts so it pickles cleanly across the Windows
    spawn boundary.
    """

    n_det_target: int
    gamma: float
    realization: int
    fitter: str               # "selection-aware" | "naive"
    sampler: str
    sim_seed: int             # seed for the galaxy realization
    fit_seed: int             # seed for the sampler
    SFR: float                # tuned SFR
    xi_true: float
    L_cut: float
    L_min: float
    L_ref: float
    distance_Mpc: float
    detector: dict            # serialised DetectorPreset fields
    absorption_flux_factor: float
    priors: dict
    points_per_dex: int
    n_live: int


def _detector_from_spec(d: dict) -> DetectorPreset:
    return DetectorPreset(
        name=d["name"],
        flux_limit_50=float(d["flux_limit_50"]),
        completeness_width_dex=float(d["completeness_width_dex"]),
        exposure_s=float(d["exposure_s"]),
        cts_to_flux=float(d["cts_to_flux"]),
        band=str(d.get("band", "")),
    )


def _cfg_for_problem(spec: TaskSpec) -> dict:
    """Reconstruct the minimal cfg dict that ``make_hmxb_problem`` reads."""
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


# ---------------------------------------------------------------------------
# THE WORKER (top-level, picklable) -- simulate one galaxy + run one fit
# ---------------------------------------------------------------------------
def run_one_cell(spec: TaskSpec) -> dict:
    """Simulate the galaxy for ``spec`` and run its one fit; return a result row.

    Deterministic from ``spec.sim_seed`` (galaxy) and ``spec.fit_seed``
    (sampler).  The selection-aware and naive fitters for the SAME realization
    share ``sim_seed`` (so they see the *same* simulated galaxy) but differ in
    ``fitter`` -- the only thing that changes is the likelihood's P_det.
    """
    detector = _detector_from_spec(spec.detector)
    L_ref = spec.L_ref

    # --- simulate the galaxy (intrinsic XLF -> detected L_obs) ---
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
    L_obs = L_obs[L_obs > 0]               # drop zero-count back-conversions
    n_det_actual = int(L_obs.size)

    cfg = _cfg_for_problem(spec)
    selection_aware = spec.fitter == "selection-aware"

    # the naive fitter only ever sees sources above its own hard cut
    if selection_aware:
        L_fit = L_obs
    else:
        flux = luminosity_to_flux(L_obs, spec.distance_Mpc) * spec.absorption_flux_factor
        L_fit = L_obs[flux >= detector.flux_limit_50]

    truth = {
        "log10_xi_eff": float(np.log10(spec.xi_true * spec.SFR)),
        "gamma": float(spec.gamma),
    }

    row = {
        # --- key ---
        "n_det_target": int(spec.n_det_target),
        "gamma_true": float(spec.gamma),
        "fitter": spec.fitter,
        "realization": int(spec.realization),
        "sampler": spec.sampler,
        # --- truth + tuning ---
        "SFR": float(spec.SFR),
        "xi_true": float(spec.xi_true),
        "log10_xi_eff_true": truth["log10_xi_eff"],
        "distance_Mpc": float(spec.distance_Mpc),
        "detector": detector.name,
        # --- realization ---
        "n_det_actual": n_det_actual,
        "n_fit": int(L_fit.size),
        "sim_seed": int(spec.sim_seed),
        "fit_seed": int(spec.fit_seed),
    }

    # degenerate galaxy (too few sources to fit a 2-parameter XLF): record and
    # skip the sampler.  n_fit < 2 cannot constrain (norm, slope).
    if L_fit.size < 2:
        row.update({
            "status": "skipped_too_few",
            "logZ": None, "wall_time_s": 0.0,
            "gamma_med": None, "gamma_minus": None, "gamma_plus": None,
            "gamma_std": None,
            "log_xi_med": None, "log_xi_minus": None, "log_xi_plus": None,
            "log_xi_std": None,
        })
        return row

    like = make_hmxb_problem(
        L_fit, cfg, distance_Mpc=spec.distance_Mpc, detector=detector,
        absorption_flux_factor=spec.absorption_flux_factor,
        selection_aware=selection_aware,
        grid=LogLGrid(points_per_dex=spec.points_per_dex),
    )
    fr = fit_xlf(like, cfg, sampler=spec.sampler, seed=spec.fit_seed)
    s = fr.summary()
    g = s["gamma"]
    x = s["log10_xi_eff"]
    gi = fr.param_names.index("gamma")
    xi = fr.param_names.index("log10_xi_eff")

    row.update({
        "status": "ok",
        "logZ": (None if fr.logZ is None else float(fr.logZ)),
        "wall_time_s": float(fr.wall_time_s),
        "gamma_med": float(g["median"]),
        "gamma_minus": float(g["minus"]),
        "gamma_plus": float(g["plus"]),
        "gamma_q16": float(g["q16"]),
        "gamma_q84": float(g["q84"]),
        "gamma_std": float(np.std(fr.samples[:, gi])),
        "log_xi_med": float(x["median"]),
        "log_xi_minus": float(x["minus"]),
        "log_xi_plus": float(x["plus"]),
        "log_xi_q16": float(x["q16"]),
        "log_xi_q84": float(x["q84"]),
        "log_xi_std": float(np.std(fr.samples[:, xi])),
    })
    return row


# ---------------------------------------------------------------------------
# Building the task list + resume bookkeeping
# ---------------------------------------------------------------------------
def load_done_keys(jsonl_path: str) -> set:
    """Read existing result rows; return the set of completed cell keys.

    Tolerant of a truncated final line (power-cut safety): a JSON parse error on
    the LAST line is ignored (that fit is simply re-run).
    """
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
            # only the final line may legitimately be partial after a crash
            if i == len(lines) - 1:
                continue
            raise
        done.add(_key_from_row(row))
    return done


def build_specs(cfg: dict, n_realizations: int, sampler: str) -> list:
    """Build all TaskSpecs for the grid (both fitters share each realization)."""
    grid_cfg = cfg["grid"]
    gal = cfg["galaxy"]
    fwd = cfg["forward"]
    L_ref = float(cfg["L_unit"])
    L_min = float(fwd["L_min"])
    L_cut = float(gal["L_cut"])
    xi_true = float(gal["xi_true"])
    distance = float(gal["distance_Mpc"])
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

    n_targets = list(grid_cfg["n_det_targets"])
    gammas = list(grid_cfg["gammas"])
    fitters = list(grid_cfg["fitters"])

    specs = []
    for gi, gamma in enumerate(gammas):
        lam_u = lambda_unit(
            gamma, xi_true, L_cut, L_min, L_ref, distance, det, absorption,
            grid_integrator,
        )
        for ti, n_target in enumerate(n_targets):
            SFR = sfr_for_target(n_target, lam_u)
            for r in range(n_realizations):
                # one simulated galaxy per (gamma, n_target, realization); BOTH
                # fitters see it via a shared sim_seed.  Seeds are deterministic
                # functions of the cell coordinates so resume is reproducible.
                sim_seed = (seed_base
                            + 1000000 * (gi + 1)
                            + 10000 * (ti + 1)
                            + r)
                for fitter in fitters:
                    fit_seed = sim_seed + (1 if fitter == "selection-aware" else 2)
                    specs.append(TaskSpec(
                        n_det_target=int(n_target),
                        gamma=float(gamma),
                        realization=int(r),
                        fitter=str(fitter),
                        sampler=str(sampler),
                        sim_seed=int(sim_seed),
                        fit_seed=int(fit_seed),
                        SFR=float(SFR),
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
                    ))
    return specs


def spec_key(spec: TaskSpec):
    return (int(spec.n_det_target), round(float(spec.gamma), 6),
            str(spec.fitter), int(spec.realization), str(spec.sampler))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",
                   default=os.path.join(_REPO, "configs", "recovery_suite.yaml"))
    p.add_argument("--n-realizations", type=int, default=None,
                   help="override config n_realizations (pilot uses 5)")
    p.add_argument("--workers", type=int, default=None,
                   help="override config max_workers (cap 6)")
    p.add_argument("--outdir", default=None,
                   help="override output dir (default from config)")
    p.add_argument("--limit", type=int, default=None,
                   help="run at most this many NEW fits then stop (for pilots)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    sampler = str(cfg["run"]["sampler"])
    n_real = args.n_realizations or int(cfg["grid"]["n_realizations"])
    workers = args.workers or int(cfg["run"]["max_workers"])
    workers = max(1, min(workers, 6))         # hard cap at 6
    outdir = args.outdir or os.path.join(_REPO, cfg["run"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = os.path.join(outdir, cfg["run"]["results_jsonl"])

    specs = build_specs(cfg, n_real, sampler)
    done = load_done_keys(jsonl_path)
    todo = [s for s in specs if spec_key(s) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print("=" * 70)
    print("XRB-XLF recovery suite (Phase 3a)")
    print(f"  grid: {len(cfg['grid']['n_det_targets'])} N_det x "
          f"{len(cfg['grid']['gammas'])} gamma x "
          f"{len(cfg['grid']['fitters'])} fitters x {n_real} realizations")
    print(f"  total fits: {len(specs)}   already done: {len(done)}   "
          f"to run now: {len(todo)}")
    print(f"  sampler: {sampler}   workers: {workers}")
    print(f"  results -> {jsonl_path}")
    print("=" * 70)
    if not todo:
        print("Nothing to do -- all cells already complete.")
        return 0

    t0 = time.perf_counter()
    n_written = 0
    walls = []
    # main process is the sole writer -> no concurrent-write race on the JSONL
    with open(jsonl_path, "a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one_cell, s): s for s in todo}
            for fut in as_completed(futs):
                row = fut.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())          # durability against power loss
                n_written += 1
                if row.get("status") == "ok":
                    walls.append(row["wall_time_s"])
                if n_written % 10 == 0 or n_written == len(todo):
                    elapsed = time.perf_counter() - t0
                    rate = n_written / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - n_written) / rate if rate > 0 else 0.0
                    print(f"  [{n_written:4d}/{len(todo)}] "
                          f"last={row['fitter'][:4]} N_t={row['n_det_target']} "
                          f"g={row['gamma_true']} Nact={row['n_det_actual']} "
                          f"| {elapsed:6.1f}s elapsed, ETA {eta:6.1f}s")

    elapsed = time.perf_counter() - t0
    print("-" * 70)
    print(f"wrote {n_written} fits in {elapsed:.1f}s "
          f"({elapsed / max(n_written,1):.2f}s/fit wall, "
          f"{workers} workers)")
    if walls:
        walls = np.array(walls)
        print(f"per-fit sampler wall-clock: mean={walls.mean():.2f}s "
              f"median={np.median(walls):.2f}s "
              f"min={walls.min():.2f}s max={walls.max():.2f}s "
              f"(n_ok={walls.size})")
        # extrapolation to the FULL grid (n=50)
        full_real = int(cfg["grid"]["n_realizations"])
        full_fits = (len(cfg["grid"]["n_det_targets"])
                     * len(cfg["grid"]["gammas"])
                     * len(cfg["grid"]["fitters"]) * full_real)
        # wall per fit observed in THIS run (includes parallel speedup)
        wall_per_fit_parallel = elapsed / max(n_written, 1)
        est_full = full_fits * wall_per_fit_parallel
        print(f"EXTRAPOLATION: full grid = {full_fits} fits; "
              f"at {wall_per_fit_parallel:.2f}s/fit ({workers}w) "
              f"=> ~{est_full/60.0:.1f} min ({est_full/3600.0:.2f} h)")
    print(f"resume command: .venv\\Scripts\\python.exe scripts\\run_recovery_suite.py "
          f"--config {os.path.relpath(args.config, _REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
