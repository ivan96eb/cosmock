"""Optional dependency helpers."""

from __future__ import annotations

from importlib import import_module


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is needed for a requested feature."""


def import_optional(module_name: str, *, extra: str, purpose: str):
    """Import an optional dependency or raise an actionable error."""

    try:
        return import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{purpose} requires the optional dependency '{module_name}'. "
            f"Install it with `pip install cosmock[{extra}]`."
        ) from exc


def require_healpy(purpose: str = "This operation"):
    """Return healpy, or raise an optional dependency error."""

    return import_optional("healpy", extra="healpix", purpose=purpose)
