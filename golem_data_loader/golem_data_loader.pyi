"""
Type stub file for golem_data_loader module.

This file provides type hints for IDEs and type checkers.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import pandas as pd
import numpy as np
import logging

class DataLoadError(Exception): ...
class FileNotFoundError(DataLoadError): ...
class NetworkError(DataLoadError): ...
class DataValidationError(DataLoadError): ...

class SpectroscopyLine(Enum):
    H_ALPHA: SpectroscopyLine
    H_BETA: SpectroscopyLine
    HE_I: SpectroscopyLine
    WHOLE: SpectroscopyLine

    display_name: str
    filename: str

@dataclass
class FastSpectrometryData:
    label: str
    time: np.ndarray
    intensity: np.ndarray
    raw_dataframe: pd.DataFrame

@dataclass
class MiniSpectrometerData:
    spectra: np.ndarray
    wavelengths: np.ndarray
    temp_file_path: Optional[Path] = None

    def cleanup(self) -> None: ...

@dataclass
class FastCameraData:
    camera_type: str
    frames: np.ndarray
    frame_rate: float = 80000.0
    time: Optional[np.ndarray] = None

@dataclass
class LoaderConfig:
    base_url_template: str = ...
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 30.0

class GolemDataLoader:
    shot_number: int
    config: LoaderConfig
    base_url: str

    def __init__(
        self,
        shot_number: int,
        config: Optional[LoaderConfig] = None,
        log_level: int = logging.INFO,
    ) -> None: ...
    def load_fast_spectrometry(
        self, lines: Optional[List[SpectroscopyLine]] = None, validate: bool = True
    ) -> Dict[str, FastSpectrometryData]: ...
    def load_minispectrometer_h5(
        self, h5_filename: str = "IRVISUV_0.h5", keep_temp_file: bool = False
    ) -> MiniSpectrometerData: ...
    def load_fast_cameras(
        self, cameras: Optional[List[str]] = None, max_frames: Optional[int] = None
    ) -> Dict[str, FastCameraData]: ...
    def get_available_diagnostics(self) -> Dict[str, bool]: ...

def load_shot_data(
    shot_number: int,
    include_spectrometry: bool = True,
    include_minispectrometer: bool = True,
    include_fast_cameras: bool = False,
    **kwargs: Any,
) -> Tuple[
    Optional[Dict[str, FastSpectrometryData]],
    Optional[MiniSpectrometerData],
    Optional[Dict[str, FastCameraData]],
]: ...
