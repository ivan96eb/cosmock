"""Lower-level utility helpers for cosmock internals and advanced users."""

from .optional import OptionalDependencyError, import_optional, require_healpy
from .quadrature import get_gh_nodes_weights
from .statistics import (
    empirical_cdf,
    get_binned_data,
    histogram_pdf_cubic,
    histogram_pdf_linear,
    histogram_pdf_quadratic,
    histogram_pdf_spline,
    histogramer2d,
)
from .validation import (
    MockValidationReport,
    SpectrumDiagnostics,
    spectrum_diagnostics,
    validate_cls,
    validate_kappa_maps,
    validate_pixwin,
)

__all__ = [
    "MockValidationReport",
    "OptionalDependencyError",
    "SpectrumDiagnostics",
    "empirical_cdf",
    "get_binned_data",
    "get_gh_nodes_weights",
    "histogram_pdf_cubic",
    "histogram_pdf_linear",
    "histogram_pdf_quadratic",
    "histogram_pdf_spline",
    "histogramer2d",
    "import_optional",
    "require_healpy",
    "spectrum_diagnostics",
    "validate_cls",
    "validate_kappa_maps",
    "validate_pixwin",
]
