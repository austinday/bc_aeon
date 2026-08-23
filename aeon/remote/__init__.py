"""Secure, mobile-friendly remote console for Aeon."""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Import lazily so ordinary Aeon installs do not require web dependencies."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
