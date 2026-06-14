"""xlf_model -- forward model + Bayesian recovery of XRB XLF parameters.

Phase 1 exposes the luminosity functions (:mod:`xlf_model.xlf`) and the forward
model (:mod:`xlf_model.forward`).  Phase 2 adds the inverse problem -- an
unbinned Poisson-process likelihood and samplers (:mod:`xlf_model.inference`).
"""

from .xlf import (
    HMXBXLF,
    LMXBXLF,
    hmxb_from_config,
    lmxb_from_config,
    powerlaw_cumulative_above,
    powerlaw_integral,
)
from .forward import (
    DetectorPreset,
    ForwardResult,
    completeness_erf,
    detector_from_config,
    draw_population,
    flux_to_luminosity,
    format_funnel,
    luminosity_to_flux,
    observe,
    run_forward,
)
from .inference import (
    FitResult,
    LogLGrid,
    ObservationModel,
    Parameter,
    PoissonProcessLikelihood,
    fit_xlf,
    hmxb_parameters_from_config,
    make_hmxb_problem,
    make_hmxb_xlf_builder,
)

__all__ = [
    "HMXBXLF",
    "LMXBXLF",
    "hmxb_from_config",
    "lmxb_from_config",
    "powerlaw_integral",
    "powerlaw_cumulative_above",
    "DetectorPreset",
    "ForwardResult",
    "detector_from_config",
    "draw_population",
    "observe",
    "run_forward",
    "luminosity_to_flux",
    "flux_to_luminosity",
    "completeness_erf",
    "format_funnel",
    # inference (Phase 2)
    "FitResult",
    "LogLGrid",
    "ObservationModel",
    "Parameter",
    "PoissonProcessLikelihood",
    "fit_xlf",
    "hmxb_parameters_from_config",
    "make_hmxb_problem",
    "make_hmxb_xlf_builder",
]
