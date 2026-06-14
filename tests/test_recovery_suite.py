"""Tests for the Phase-3 recovery suite (scripts/run_recovery_suite.py).

Coverage:
  * the tuning rule is EXACT: Lambda is linear in SFR, so
    SFR = target / Lambda_unit makes E[N_det] == target;
  * build_specs produces the right grid (both fitters share a realization's
    sim_seed; fit_seeds differ);
  * a tiny-grid end-to-end run (1 N_det target x 1 gamma x both fitters x 2
    realizations) writes one keyed JSONL row per fit;
  * the RESUME logic: re-running skips completed keys (no duplicates); and if a
    single row is deleted, ONLY that one fit is recomputed.

The end-to-end test runs the sampler, so it is marked slow; it is kept to a
tiny grid (4 fits) with a small live-point count so it finishes in a few
seconds.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load_suite_module():
    """Import scripts/run_recovery_suite.py as a module (it is not a package).

    Registered in ``sys.modules`` BEFORE executing so the dataclass decorators
    can resolve forward-referenced annotations via the module dict.
    """
    path = os.path.join(_REPO, "scripts", "run_recovery_suite.py")
    spec = importlib.util.spec_from_file_location("run_recovery_suite", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_recovery_suite"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_suite_module()


# ---------------------------------------------------------------------------
# tiny config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def suite_cfg():
    """A 1-cell config: 1 N_det target x 1 gamma x both fitters."""
    with open(os.path.join(_REPO, "configs", "recovery_suite.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    # shrink to a tiny, fast grid
    cfg["grid"]["n_det_targets"] = [15]
    cfg["grid"]["gammas"] = [1.6]
    cfg["grid"]["n_realizations"] = 2
    cfg["inference"]["sampler"]["n_live"] = 80   # fast
    return cfg


# ---------------------------------------------------------------------------
# tuning rule: Lambda linear in SFR -> exact targeting
# ---------------------------------------------------------------------------
def test_tuning_rule_exact(suite_cfg):
    """SFR = target / Lambda_unit makes the expected detected count == target."""
    from xlf_model.forward import detector_from_config
    from xlf_model.inference import LogLGrid, ObservationModel
    from xlf_model.xlf import HMXBXLF

    cfg = suite_cfg
    gal = cfg["galaxy"]
    det = detector_from_config(cfg, name=gal["detector"])
    grid = LogLGrid(points_per_dex=int(cfg["inference"]["grid"]["points_per_dex"]))
    L_ref = float(cfg["L_unit"])
    L_min = float(cfg["forward"]["L_min"])
    L_cut = float(gal["L_cut"])
    xi = float(gal["xi_true"])
    dist = float(gal["distance_Mpc"])
    absorp = float(gal["absorption_flux_factor"])
    gamma = 1.6

    lam_u = R.lambda_unit(gamma, xi, L_cut, L_min, L_ref, dist, det, absorp, grid)
    for target in (5, 50, 500):
        SFR = R.sfr_for_target(target, lam_u)
        # recompute Lambda at the tuned SFR -> must equal target exactly
        xlf = HMXBXLF(xi=xi, gamma=gamma, L_cut=L_cut, L_min=L_min, L_ref=L_ref,
                      SFR=SFR)
        obs = ObservationModel(distance_Mpc=dist, detector=det,
                               absorption_flux_factor=absorp,
                               selection_aware=True)
        L = grid.grid(xlf.L_min, xlf.L_cut)
        lam = np.asarray(xlf.dN_dL(L)) * obs.p_det(L)
        Lambda = float(np.trapezoid(lam, L))
        assert Lambda == pytest.approx(float(target), rel=1e-6)


def test_build_specs_grid_and_seeds(suite_cfg):
    """build_specs makes 2 fitters x 2 realizations; the two fitters of one
    realization share sim_seed but have distinct fit_seeds."""
    specs = R.build_specs(suite_cfg, n_realizations=2, sampler="ultranest")
    assert len(specs) == 1 * 1 * 2 * 2  # N_det x gamma x fitters x realizations
    # group by realization
    by_real = {}
    for s in specs:
        by_real.setdefault(s.realization, []).append(s)
    for r, ss in by_real.items():
        fitters = {s.fitter for s in ss}
        assert fitters == {"selection-aware", "naive"}
        sim_seeds = {s.sim_seed for s in ss}
        assert len(sim_seeds) == 1                      # same galaxy
        fit_seeds = {s.fit_seed for s in ss}
        assert len(fit_seeds) == 2                      # distinct sampler seeds
    # the two realizations have different sim_seeds
    assert len({s.sim_seed for s in specs}) == 2


# ---------------------------------------------------------------------------
# end-to-end + resume logic (the crash-safety guarantee)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_end_to_end_and_resume(tmp_path, suite_cfg):
    """Run the tiny grid, then exercise the resume path:
       (1) first run writes 4 keyed rows (1 per fit);
       (2) re-running with all keys present writes NOTHING (no duplicates);
       (3) deleting ONE row and re-running recomputes ONLY that fit.
    """
    cfg = suite_cfg
    n_real = int(cfg["grid"]["n_realizations"])
    sampler = str(cfg["run"]["sampler"])
    jsonl = tmp_path / "results.jsonl"

    specs = R.build_specs(cfg, n_real, sampler)
    n_fits = len(specs)
    assert n_fits == 4

    def run_all(specs_to_run):
        """Mimic the script's main loop (serial: no pool needed for a smoke
        test, and it avoids spawn overhead)."""
        done = R.load_done_keys(str(jsonl))
        todo = [s for s in specs_to_run if R.spec_key(s) not in done]
        n_written = 0
        with open(jsonl, "a", encoding="utf-8") as out:
            for s in todo:
                row = R.run_one_cell(s)
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_written += 1
        return n_written, len(todo)

    # (1) first run: all 4 fits
    n_written, n_todo = run_all(specs)
    assert n_todo == 4 and n_written == 4
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    assert len(rows) == 4
    keys = {R._key_from_row(r) for r in rows}
    assert len(keys) == 4                       # all distinct, one per fit

    # (2) re-run with everything done -> nothing recomputed, no duplicates
    n_written2, n_todo2 = run_all(specs)
    assert n_todo2 == 0 and n_written2 == 0
    rows2 = [json.loads(l) for l in open(jsonl) if l.strip()]
    assert len(rows2) == 4                       # unchanged

    # (3) delete ONE row, re-run -> exactly that one fit recomputes
    target_key = R._key_from_row(rows[1])
    kept = [r for r in rows if R._key_from_row(r) != target_key]
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    assert len(kept) == 3

    done_after_delete = R.load_done_keys(str(jsonl))
    assert target_key not in done_after_delete
    assert len(done_after_delete) == 3

    n_written3, n_todo3 = run_all(specs)
    assert n_todo3 == 1 and n_written3 == 1      # ONLY the missing one
    rows3 = [json.loads(l) for l in open(jsonl) if l.strip()]
    assert len(rows3) == 4                       # back to 4, no duplicates
    keys3 = [R._key_from_row(r) for r in rows3]
    assert len(set(keys3)) == 4
    assert target_key in set(keys3)              # the deleted fit is back


def test_load_done_keys_tolerates_truncated_last_line(tmp_path):
    """A power-cut can truncate the FINAL JSONL line; load_done_keys must
    tolerate it (drop the partial line) rather than crash."""
    jsonl = tmp_path / "results.jsonl"
    good = {
        "n_det_target": 15, "gamma_true": 1.6, "fitter": "naive",
        "realization": 0, "sampler": "ultranest",
    }
    with open(jsonl, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write('{"n_det_target": 15, "gamma_true": 1.6, "fitt')  # truncated
    done = R.load_done_keys(str(jsonl))
    assert done == {(15, 1.6, "naive", 0, "ultranest")}
