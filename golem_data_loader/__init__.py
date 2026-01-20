"""
GOLEM Data Loader Package

A robust, type-safe Python package for loading diagnostic data from the
GOLEM tokamak web server.
"""

from .golem_data_loader import (
    # Main class
    GolemDataLoader,
    # Data structures
    FastSpectrometryData,
    MiniSpectrometerData,
    FastCameraData,
    LoaderConfig,
    # Enums
    SpectroscopyLine,
    # Exceptions
    DataLoadError,
    FileNotFoundError,
    NetworkError,
    DataValidationError,
    # Convenience functions
    load_shot_data,
)

__version__ = "1.1.0"

__all__ = [
    "GolemDataLoader",
    "FastSpectrometryData",
    "MiniSpectrometerData",
    "FastCameraData",
    "LoaderConfig",
    "SpectroscopyLine",
    "DataLoadError",
    "FileNotFoundError",
    "NetworkError",
    "DataValidationError",
    "load_shot_data",
]
