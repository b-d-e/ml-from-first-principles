"""
Machine Learning from First Principles:
    Implementing Machine Learning algorithms from first principles,
    without relying on Python libraries.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version(__name__)
except PackageNotFoundError:
    # Fallback version if package metadata is not found
    __version__ = "0.0.0-dev"

__all__ = ("__version__",)
