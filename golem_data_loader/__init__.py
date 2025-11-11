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
    BasicDiagnosticsData,
    MirnovCoilsData,
    MHDRingData,
    PlasmaDetectionData,
    ShotInfo,
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

# Plotting utilities (optional import)
try:
    from . import plotting
except ImportError:
    plotting = None  # matplotlib not installed

__version__ = "1.2.0"

__all__ = [
    "GolemDataLoader",
    "FastSpectrometryData",
    "MiniSpectrometerData",
    "BasicDiagnosticsData",
    "MirnovCoilsData",
    "MHDRingData",
    "PlasmaDetectionData",
    "ShotInfo",
    "LoaderConfig",
    "SpectroscopyLine",
    "DataLoadError",
    "FileNotFoundError",
    "NetworkError",
    "DataValidationError",
    "load_shot_data",
]
