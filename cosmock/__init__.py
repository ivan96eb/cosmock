"""Function-first scalar-field mock generation."""

from .types import FieldMockFit
from .workflow import fit_field_model, generate_field_mocks

__version__ = "0.0.1"

__all__ = [
    "FieldMockFit",
    "__version__",
    "fit_field_model",
    "generate_field_mocks",
]
