"""Generic acquisition errors."""

from __future__ import annotations


class TransientAcquisitionError(RuntimeError):
    """Provider failure that may succeed on retry."""


class OversizedAcquisitionRequestError(RuntimeError):
    """Provider rejected a request because the batch was too large."""


class UnsupportedAcquisitionSourceError(RuntimeError):
    """Provider cannot produce a required acquisition source fact."""
