"""
GOLEM Data Loader Module

This module provides a robust interface for loading diagnostic data from the
GOLEM tokamak web server. It includes retry logic, error handling, and type-safe
data structures.

Example:
    >>> from golem_data_loader import GolemDataLoader
    >>> loader = GolemDataLoader(shot_number=50377)
    >>> spectrometry_data = loader.load_fast_spectrometry()
    >>> h5_data = loader.load_minispectrometer_h5()
"""

from __future__ import annotations

import time
import logging
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
from enum import Enum

import pandas as pd
import numpy as np
import h5py


# Configure module logger
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Base exception for data loading errors."""

    pass


class FileNotFoundError(DataLoadError):
    """Raised when a requested file is not found on the server."""

    pass


class NetworkError(DataLoadError):
    """Raised when network-related errors occur."""

    pass


class DataValidationError(DataLoadError):
    """Raised when loaded data fails validation."""

    pass


class SpectroscopyLine(Enum):
    """Enumeration of available fast spectrometry signals."""

    H_ALPHA = ("Hα", "U_Halpha.csv")
    H_BETA = ("Hβ", "U_Hbeta.csv")
    HE_I = ("He I", "U_HeI.csv")
    WHOLE = ("Whole", "U_whole.csv")

    def __init__(self, display_name: str, filename: str):
        self.display_name = display_name
        self.filename = filename


@dataclass
class FastSpectrometryData:
    """Container for fast spectrometry data from a single signal.

    Attributes:
        label: Human-readable label for the signal (e.g., "Hα")
        time: Time array in seconds
        intensity: Intensity values (arbitrary units)
        raw_dataframe: Original pandas DataFrame
    """

    label: str
    time: np.ndarray
    intensity: np.ndarray
    raw_dataframe: pd.DataFrame

    def __post_init__(self):
        """Validate data dimensions."""
        if len(self.time) != len(self.intensity):
            raise DataValidationError(
                f"Time and intensity arrays must have same length. "
                f"Got {len(self.time)} and {len(self.intensity)}"
            )


@dataclass
class MiniSpectrometerData:
    """Container for mini-spectrometer HDF5 data.

    Attributes:
        spectra: 2D array of spectral data [time, wavelength]
        wavelengths: 1D array of wavelength values
        temp_file_path: Path to temporary H5 file (for cleanup)
    """

    spectra: np.ndarray
    wavelengths: np.ndarray
    temp_file_path: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove temporary H5 file if it exists."""
        if self.temp_file_path and self.temp_file_path.exists():
            try:
                self.temp_file_path.unlink()
                logger.info(f"Cleaned up temporary file: {self.temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@dataclass
class BasicDiagnosticsData:
    """Container for basic diagnostics data.

    Attributes:
        toroidal_field: Bt - Toroidal magnetic field (T) vs time (s)
        plasma_current: Ip - Plasma current (kA) vs time (s)
        chamber_current: Ich - Chamber current (A) vs time (s)
        loop_voltage: U_loop - Loop voltage (V) vs time (s)
        raw_dataframes: Dictionary of original DataFrames
    """

    toroidal_field: Optional[FastSpectrometryData] = None
    plasma_current: Optional[FastSpectrometryData] = None
    chamber_current: Optional[FastSpectrometryData] = None
    loop_voltage: Optional[FastSpectrometryData] = None
    raw_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class MirnovCoilsData:
    """Container for Mirnov coils data.

    Attributes:
        coils: Dictionary mapping coil numbers to signal data
        raw_dataframes: Dictionary of original DataFrames
    """

    coils: Dict[int, FastSpectrometryData] = field(default_factory=dict)
    raw_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class MHDRingData:
    """Container for MHD ring data.

    Attributes:
        rings: Dictionary mapping ring numbers to signal data
        raw_dataframes: Dictionary of original DataFrames
    """

    rings: Dict[int, FastSpectrometryData] = field(default_factory=dict)
    raw_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PlasmaDetectionData:
    """Container for plasma detection signals.

    Attributes:
        bt_coil: Toroidal field coil signal
        int_bt_coil: Integrated toroidal field coil signal
        rog_coil: Rogowski coil signal
        int_rog_coil: Integrated Rogowski coil signal
        leyb_phot: Leybold photocell signal
        loop: Loop signal
        raw_dataframes: Dictionary of original DataFrames
    """

    bt_coil: Optional[FastSpectrometryData] = None
    int_bt_coil: Optional[FastSpectrometryData] = None
    rog_coil: Optional[FastSpectrometryData] = None
    int_rog_coil: Optional[FastSpectrometryData] = None
    leyb_phot: Optional[FastSpectrometryData] = None
    loop: Optional[FastSpectrometryData] = None
    raw_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class ShotInfo:
    """Container for shot metadata and information.

    Attributes:
        shot_number: GOLEM shot number
        logbook: Raw logbook text
        available_diagnostics: Dictionary of diagnostic availability
        timestamp: Shot timestamp if available
    """

    shot_number: int
    logbook: Optional[str] = None
    available_diagnostics: Dict[str, bool] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class LoaderConfig:
    """Configuration for the GOLEM data loader.

    Attributes:
        base_url_template: URL template for GOLEM shots
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
    """

    base_url_template: str = "http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/"
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 30.0


class GolemDataLoader:
    """
    Main class for loading GOLEM tokamak diagnostic data.

    This class provides methods to load various diagnostic data types from
    the GOLEM web server with built-in retry logic and error handling.

    Attributes:
        shot_number: GOLEM shot number
        config: Loader configuration

    Example:
        >>> loader = GolemDataLoader(50377)
        >>> data = loader.load_fast_spectrometry()
        >>> print(data.keys())
        dict_keys(['Hα', 'Hβ', 'He I', 'Whole'])
    """

    def __init__(
        self,
        shot_number: int,
        config: Optional[LoaderConfig] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the GOLEM data loader.

        Args:
            shot_number: GOLEM shot number to load data from
            config: Optional custom configuration
            log_level: Logging level (default: INFO)
        """
        self.shot_number = shot_number
        self.config = config or LoaderConfig()
        self.base_url = self.config.base_url_template.format(shot=shot_number)

        # Configure logging
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)

    def _fetch_url_with_retry(self, url: str, description: str = "data") -> bytes:
        """
        Fetch data from URL with retry logic.

        Args:
            url: URL to fetch
            description: Human-readable description for logging

        Returns:
            Raw bytes from the response

        Raises:
            NetworkError: If all retry attempts fail
            FileNotFoundError: If the resource is not found (404)
        """
        last_exception = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    f"Fetching {description} (attempt {attempt}/{self.config.max_retries}): {url}"
                )

                with urllib.request.urlopen(
                    url, timeout=self.config.timeout
                ) as response:
                    data = response.read()
                    logger.info(f"Successfully loaded {description} from {url}")
                    return data

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(f"Resource not found: {url}") from e
                last_exception = e
                logger.warning(
                    f"HTTP error {e.code} while fetching {description} "
                    f"(attempt {attempt}/{self.config.max_retries}): {e}"
                )

            except urllib.error.URLError as e:
                last_exception = e
                logger.warning(
                    f"URL error while fetching {description} "
                    f"(attempt {attempt}/{self.config.max_retries}): {e}"
                )

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Unexpected error while fetching {description} "
                    f"(attempt {attempt}/{self.config.max_retries}): {e}"
                )

            # Wait before retrying (except on last attempt)
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay)

        # All retries failed
        raise NetworkError(
            f"Failed to fetch {description} after {self.config.max_retries} attempts"
        ) from last_exception

    def load_fast_spectrometry(
        self, lines: Optional[List[SpectroscopyLine]] = None, validate: bool = True
    ) -> Dict[str, FastSpectrometryData]:
        """
        Load fast spectrometry CSV data.

        Args:
            lines: List of spectroscopy lines to load. If None, loads all available.
            validate: Whether to validate loaded data

        Returns:
            Dictionary mapping line names to FastSpectrometryData objects

        Raises:
            DataLoadError: If data cannot be loaded
            DataValidationError: If loaded data fails validation

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> data = loader.load_fast_spectrometry(
            ...     lines=[SpectroscopyLine.H_ALPHA, SpectroscopyLine.H_BETA]
            ... )
        """
        if lines is None:
            lines = list(SpectroscopyLine)

        results: Dict[str, FastSpectrometryData] = {}
        failed_loads: List[tuple[str, Exception]] = []

        for line in lines:
            try:
                url = f"{self.base_url}FastSpectrometry/{line.filename}"
                csv_data = self._fetch_url_with_retry(
                    url, f"{line.display_name} spectrometry"
                )

                # Parse CSV data
                from io import BytesIO

                df = pd.read_csv(BytesIO(csv_data))

                # Validate DataFrame structure
                if validate:
                    if df.shape[1] < 2:
                        raise DataValidationError(
                            f"Expected at least 2 columns, got {df.shape[1]}"
                        )
                    if df.shape[0] == 0:
                        raise DataValidationError("No data rows found")

                # Create structured data object
                spectrometry_data = FastSpectrometryData(
                    label=line.display_name,
                    time=df.iloc[:, 0].values,
                    intensity=df.iloc[:, 1].values,
                    raw_dataframe=df,
                )

                results[line.display_name] = spectrometry_data
                logger.info(
                    f"{line.display_name}: loaded {df.shape[0]} rows, "
                    f"columns = {list(df.columns)}"
                )

            except FileNotFoundError as e:
                logger.warning(f"File not found for {line.display_name}: {e}")
                failed_loads.append((line.display_name, e))

            except Exception as e:
                logger.error(f"Failed to load {line.display_name}: {e}")
                failed_loads.append((line.display_name, e))

        if not results and failed_loads:
            # All loads failed
            raise DataLoadError(
                f"Failed to load any spectrometry data. Errors: {failed_loads}"
            )

        if failed_loads:
            logger.warning(
                f"Partially loaded data. Failed: {[name for name, _ in failed_loads]}"
            )

        return results

    def load_minispectrometer_h5(
        self, h5_filename: str = "IRVISUV_0.h5", keep_temp_file: bool = False
    ) -> MiniSpectrometerData:
        """
        Load mini-spectrometer HDF5 data.

        Downloads the H5 file to a temporary location and extracts spectral data.

        Args:
            h5_filename: Name of the H5 file to load
            keep_temp_file: If True, don't delete temporary file (useful for debugging)

        Returns:
            MiniSpectrometerData object containing spectra and wavelengths

        Raises:
            DataLoadError: If data cannot be loaded
            DataValidationError: If H5 file structure is invalid

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> h5_data = loader.load_minispectrometer_h5()
            >>> print(h5_data.spectra.shape)
            >>> h5_data.cleanup()  # Clean up temp file when done
        """
        url = f"{self.base_url}MiniSpectrometer/DAS_raw_data_dir/{h5_filename}"

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            # Download H5 file
            logger.info(f"Downloading H5 file to {temp_path}")
            h5_data = self._fetch_url_with_retry(url, "mini-spectrometer H5 file")

            with open(temp_path, "wb") as f:
                f.write(h5_data)

            logger.info(f"Successfully downloaded H5 file ({len(h5_data)} bytes)")

            # Open and extract data
            with h5py.File(temp_path, "r") as h5_file:
                # Validate required datasets exist
                if "Spectra" not in h5_file:
                    raise DataValidationError("H5 file missing 'Spectra' dataset")
                if "Wavelengths" not in h5_file:
                    raise DataValidationError("H5 file missing 'Wavelengths' dataset")

                # Load data (convert to numpy arrays to avoid h5py reference issues)
                spectra = np.array(h5_file["Spectra"])
                wavelengths = np.array(h5_file["Wavelengths"][1:])  # Skip first element

                logger.info(
                    f"Loaded spectra: shape={spectra.shape}, "
                    f"wavelengths: shape={wavelengths.shape}"
                )

            # Create data object
            result = MiniSpectrometerData(
                spectra=spectra,
                wavelengths=wavelengths,
                temp_file_path=temp_path if keep_temp_file else None,
            )

            # Clean up temp file unless requested to keep it
            if not keep_temp_file:
                result.temp_file_path = temp_path
                result.cleanup()
                result.temp_file_path = None

            return result

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

            if isinstance(e, (DataLoadError, DataValidationError)):
                raise
            raise DataLoadError(f"Failed to load mini-spectrometer H5 data: {e}") from e

    def load_basic_diagnostics(self) -> BasicDiagnosticsData:
        """
        Load basic diagnostics data (Bt, Ip, Ich, U_loop).

        Returns:
            BasicDiagnosticsData object with available signals

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> basic = loader.load_basic_diagnostics()
            >>> if basic.plasma_current:
            ...     print(basic.plasma_current.time.shape)
        """
        result = BasicDiagnosticsData()
        base_path = f"{self.base_url}BasicDiagnostics/Results/"

        signals = {
            "Bt": ("toroidal_field", "Toroidal Field"),
            "Ip": ("plasma_current", "Plasma Current"),
            "Ich": ("chamber_current", "Chamber Current"),
            "U_loop": ("loop_voltage", "Loop Voltage"),
        }

        for filename, (attr_name, display_name) in signals.items():
            try:
                url = f"{base_path}{filename}.csv"
                csv_data = self._fetch_url_with_retry(
                    url, f"BasicDiagnostics/{filename}"
                )

                from io import BytesIO

                df = pd.read_csv(BytesIO(csv_data))

                if df.shape[1] >= 2:
                    signal_data = FastSpectrometryData(
                        label=display_name,
                        time=df.iloc[:, 0].values,
                        intensity=df.iloc[:, 1].values,
                        raw_dataframe=df,
                    )
                    setattr(result, attr_name, signal_data)
                    result.raw_dataframes[filename] = df
                    logger.info(f"Loaded {filename}: {df.shape[0]} points")

            except (FileNotFoundError, NetworkError) as e:
                logger.debug(f"{filename} not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        return result

    def load_mirnov_coils(self) -> MirnovCoilsData:
        """
        Load Limiter Mirnov Coils data.

        Returns:
            MirnovCoilsData object with available coil signals

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> mirnov = loader.load_mirnov_coils()
            >>> for coil_num, data in mirnov.coils.items():
            ...     print(f"Coil {coil_num}: {len(data.time)} points")
        """
        result = MirnovCoilsData()
        base_path = f"{self.base_url}LimiterMirnovCoils/"

        # Typical coil numbers (1, 5, 9, 13)
        for coil_num in [1, 5, 9, 13]:
            try:
                filename = f"U_mc{coil_num}.csv"
                url = f"{base_path}{filename}"
                csv_data = self._fetch_url_with_retry(url, f"Mirnov Coil {coil_num}")

                from io import BytesIO

                df = pd.read_csv(BytesIO(csv_data))

                if df.shape[1] >= 2:
                    signal_data = FastSpectrometryData(
                        label=f"Mirnov Coil {coil_num}",
                        time=df.iloc[:, 0].values,
                        intensity=df.iloc[:, 1].values,
                        raw_dataframe=df,
                    )
                    result.coils[coil_num] = signal_data
                    result.raw_dataframes[f"mc{coil_num}"] = df
                    logger.info(f"Loaded Mirnov Coil {coil_num}: {df.shape[0]} points")

            except (FileNotFoundError, NetworkError):
                logger.debug(f"Mirnov Coil {coil_num} not available")
            except Exception as e:
                logger.warning(f"Failed to load Mirnov Coil {coil_num}: {e}")

        return result

    def load_mhd_ring(self) -> MHDRingData:
        """
        Load MHD ring data.

        Returns:
            MHDRingData object with available ring signals

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> mhd = loader.load_mhd_ring()
            >>> for ring_num, data in mhd.rings.items():
            ...     print(f"Ring {ring_num}: {len(data.time)} points")
        """
        result = MHDRingData()
        base_path = f"{self.base_url}MHDring-TM/"

        for ring_num in range(1, 6):  # Rings 1-5
            try:
                filename = f"ring_{ring_num}.csv"
                url = f"{base_path}{filename}"
                csv_data = self._fetch_url_with_retry(url, f"MHD Ring {ring_num}")

                from io import BytesIO

                df = pd.read_csv(BytesIO(csv_data))

                if df.shape[1] >= 2:
                    signal_data = FastSpectrometryData(
                        label=f"MHD Ring {ring_num}",
                        time=df.iloc[:, 0].values,
                        intensity=df.iloc[:, 1].values,
                        raw_dataframe=df,
                    )
                    result.rings[ring_num] = signal_data
                    result.raw_dataframes[f"ring_{ring_num}"] = df
                    logger.info(f"Loaded MHD Ring {ring_num}: {df.shape[0]} points")

            except (FileNotFoundError, NetworkError):
                logger.debug(f"MHD Ring {ring_num} not available")
            except Exception as e:
                logger.warning(f"Failed to load MHD Ring {ring_num}: {e}")

        return result

    def load_plasma_detection(self) -> PlasmaDetectionData:
        """
        Load plasma detection signals.

        Returns:
            PlasmaDetectionData object with available signals

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> plasma_det = loader.load_plasma_detection()
            >>> if plasma_det.rog_coil:
            ...     print(plasma_det.rog_coil.time.shape)
        """
        result = PlasmaDetectionData()
        base_path = f"{self.base_url}PlasmaDetection/"

        signals = {
            "U_BtCoil": ("bt_coil", "BT Coil"),
            "U_IntBtCoil": ("int_bt_coil", "Integrated BT Coil"),
            "U_RogCoil": ("rog_coil", "Rogowski Coil"),
            "U_IntRogCoil": ("int_rog_coil", "Integrated Rogowski Coil"),
            "U_LeybPhot": ("leyb_phot", "Leybold Photocell"),
            "U_Loop": ("loop", "Loop"),
        }

        for filename, (attr_name, display_name) in signals.items():
            try:
                url = f"{base_path}{filename}.csv"
                csv_data = self._fetch_url_with_retry(
                    url, f"PlasmaDetection/{filename}"
                )

                from io import BytesIO

                df = pd.read_csv(BytesIO(csv_data))

                if df.shape[1] >= 2:
                    signal_data = FastSpectrometryData(
                        label=display_name,
                        time=df.iloc[:, 0].values,
                        intensity=df.iloc[:, 1].values,
                        raw_dataframe=df,
                    )
                    setattr(result, attr_name, signal_data)
                    result.raw_dataframes[filename] = df
                    logger.info(f"Loaded {filename}: {df.shape[0]} points")

            except (FileNotFoundError, NetworkError):
                logger.debug(f"{filename} not available")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        return result

    def load_shot_info(self) -> ShotInfo:
        """
        Load shot information and logbook.

        Returns:
            ShotInfo object with logbook and available diagnostics

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> info = loader.load_shot_info()
            >>> print(info.logbook[:200])
        """
        result = ShotInfo(shot_number=self.shot_number)

        # Load logbook
        try:
            url = f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/ShotLogbook"
            logbook_data = self._fetch_url_with_retry(url, "Shot logbook")
            result.logbook = logbook_data.decode("utf-8", errors="replace")
            logger.info(f"Loaded logbook: {len(result.logbook)} characters")
        except Exception as e:
            logger.warning(f"Could not load logbook: {e}")

        # Get available diagnostics
        result.available_diagnostics = self.get_available_diagnostics()

        return result

    def load_all_diagnostics(self) -> Dict[str, Any]:
        """
        Load all available diagnostics for this shot.

        Returns:
            Dictionary containing all loaded diagnostic data

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> all_data = loader.load_all_diagnostics()
            >>> print(all_data.keys())
        """
        all_data = {}

        try:
            all_data["basic"] = self.load_basic_diagnostics()
        except Exception as e:
            logger.error(f"Failed to load basic diagnostics: {e}")

        try:
            all_data["spectrometry"] = self.load_fast_spectrometry()
        except Exception as e:
            logger.error(f"Failed to load fast spectrometry: {e}")

        try:
            all_data["minispectrometer"] = self.load_minispectrometer_h5()
        except Exception as e:
            logger.error(f"Failed to load mini-spectrometer: {e}")

        try:
            all_data["mirnov"] = self.load_mirnov_coils()
        except Exception as e:
            logger.error(f"Failed to load Mirnov coils: {e}")

        try:
            all_data["mhd_ring"] = self.load_mhd_ring()
        except Exception as e:
            logger.error(f"Failed to load MHD ring: {e}")

        try:
            all_data["plasma_detection"] = self.load_plasma_detection()
        except Exception as e:
            logger.error(f"Failed to load plasma detection: {e}")

        try:
            all_data["info"] = self.load_shot_info()
        except Exception as e:
            logger.error(f"Failed to load shot info: {e}")

        return all_data

    def export_to_hdf5(
        self,
        output_path: str,
        include_diagnostics: Optional[List[str]] = None,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """
        Export all available diagnostics to a single HDF5 file.

        Creates a well-organized HDF5 file with groups for each diagnostic type.
        Includes metadata, timestamps, and all signal data.

        Parameters
        ----------
        output_path : str
            Path where the HDF5 file will be saved
        include_diagnostics : list of str, optional
            List of diagnostics to include. If None, includes all available.
            Options: 'basic', 'spectrometry', 'minispectrometer', 'mirnov',
                    'mhd_ring', 'plasma_detection', 'info'
        compression : str, optional
            HDF5 compression algorithm (default: 'gzip')
        compression_opts : int, optional
            Compression level 0-9 (default: 4)

        Examples
        --------
        >>> loader = GolemDataLoader(50377)
        >>> loader.export_to_hdf5('shot_50377_data.h5')
        >>> # Export only specific diagnostics
        >>> loader.export_to_hdf5('shot_50377_basic.h5',
        ...                       include_diagnostics=['basic', 'spectrometry'])

        Notes
        -----
        The HDF5 file structure:
        /metadata/shot_number
        /metadata/export_timestamp
        /basic/plasma_current/{time, intensity}
        /basic/toroidal_field/{time, intensity}
        /spectrometry/Hα/{time, intensity}
        /mhd_ring/ring_1/{time, intensity}
        etc.
        """
        import datetime

        logger.info(f"Exporting shot {self.shot_number} to HDF5: {output_path}")

        # Determine which diagnostics to load
        if include_diagnostics is None:
            include_diagnostics = [
                "basic",
                "spectrometry",
                "minispectrometer",
                "mirnov",
                "mhd_ring",
                "plasma_detection",
                "info",
            ]

        # Load diagnostics
        all_data = {}
        if "basic" in include_diagnostics:
            try:
                all_data["basic"] = self.load_basic_diagnostics()
                logger.info("Loaded basic diagnostics")
            except Exception as e:
                logger.warning(f"Failed to load basic diagnostics: {e}")

        if "spectrometry" in include_diagnostics:
            try:
                all_data["spectrometry"] = self.load_fast_spectrometry()
                logger.info("Loaded fast spectrometry")
            except Exception as e:
                logger.warning(f"Failed to load fast spectrometry: {e}")

        if "minispectrometer" in include_diagnostics:
            try:
                all_data["minispectrometer"] = self.load_minispectrometer_h5()
                logger.info("Loaded mini-spectrometer")
            except Exception as e:
                logger.warning(f"Failed to load mini-spectrometer: {e}")

        if "mirnov" in include_diagnostics:
            try:
                all_data["mirnov"] = self.load_mirnov_coils()
                logger.info("Loaded Mirnov coils")
            except Exception as e:
                logger.warning(f"Failed to load Mirnov coils: {e}")

        if "mhd_ring" in include_diagnostics:
            try:
                all_data["mhd_ring"] = self.load_mhd_ring()
                logger.info("Loaded MHD ring")
            except Exception as e:
                logger.warning(f"Failed to load MHD ring: {e}")

        if "plasma_detection" in include_diagnostics:
            try:
                all_data["plasma_detection"] = self.load_plasma_detection()
                logger.info("Loaded plasma detection")
            except Exception as e:
                logger.warning(f"Failed to load plasma detection: {e}")

        if "info" in include_diagnostics:
            try:
                all_data["info"] = self.load_shot_info()
                logger.info("Loaded shot info")
            except Exception as e:
                logger.warning(f"Failed to load shot info: {e}")

        # Create HDF5 file
        with h5py.File(output_path, "w") as f:
            # Metadata group
            meta = f.create_group("metadata")
            meta.attrs["shot_number"] = self.shot_number
            meta.attrs["export_timestamp"] = datetime.datetime.now().isoformat()
            meta.attrs["loader_version"] = "1.2.0"

            # Basic diagnostics
            if "basic" in all_data:
                basic = all_data["basic"]
                basic_grp = f.create_group("basic")

                if basic.plasma_current is not None:
                    ip_grp = basic_grp.create_group("plasma_current")
                    ip_grp.create_dataset(
                        "time",
                        data=basic.plasma_current.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ip_grp.create_dataset(
                        "intensity",
                        data=basic.plasma_current.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ip_grp.attrs["label"] = "Plasma Current"
                    ip_grp.attrs["units"] = "A"

                if basic.toroidal_field is not None:
                    bt_grp = basic_grp.create_group("toroidal_field")
                    bt_grp.create_dataset(
                        "time",
                        data=basic.toroidal_field.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    bt_grp.create_dataset(
                        "intensity",
                        data=basic.toroidal_field.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    bt_grp.attrs["label"] = "Toroidal Field"
                    bt_grp.attrs["units"] = "T"

                if basic.chamber_current is not None:
                    ich_grp = basic_grp.create_group("chamber_current")
                    ich_grp.create_dataset(
                        "time",
                        data=basic.chamber_current.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ich_grp.create_dataset(
                        "intensity",
                        data=basic.chamber_current.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ich_grp.attrs["label"] = "Chamber Current"
                    ich_grp.attrs["units"] = "A"

                if basic.loop_voltage is not None:
                    uloop_grp = basic_grp.create_group("loop_voltage")
                    uloop_grp.create_dataset(
                        "time",
                        data=basic.loop_voltage.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    uloop_grp.create_dataset(
                        "intensity",
                        data=basic.loop_voltage.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    uloop_grp.attrs["label"] = "Loop Voltage"
                    uloop_grp.attrs["units"] = "V"

            # Fast spectrometry
            if "spectrometry" in all_data:
                spec = all_data["spectrometry"]
                spec_grp = f.create_group("spectrometry")

                for line_name, line_data in spec.items():
                    line_grp = spec_grp.create_group(line_name.replace(" ", "_"))
                    line_grp.create_dataset(
                        "time",
                        data=line_data.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    line_grp.create_dataset(
                        "intensity",
                        data=line_data.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    line_grp.attrs["label"] = line_name
                    line_grp.attrs["units"] = "a.u."

            # MHD Ring
            if "mhd_ring" in all_data:
                mhd = all_data["mhd_ring"]
                mhd_grp = f.create_group("mhd_ring")

                for ring_num, ring_data in mhd.rings.items():
                    ring_grp = mhd_grp.create_group(f"ring_{ring_num}")
                    ring_grp.create_dataset(
                        "time",
                        data=ring_data.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ring_grp.create_dataset(
                        "intensity",
                        data=ring_data.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    ring_grp.attrs["label"] = f"MHD Ring {ring_num}"
                    ring_grp.attrs["units"] = "V"

            # Mirnov Coils
            if "mirnov" in all_data:
                mirnov = all_data["mirnov"]
                mirnov_grp = f.create_group("mirnov_coils")

                for coil_num, coil_data in mirnov.coils.items():
                    coil_grp = mirnov_grp.create_group(f"coil_{coil_num}")
                    coil_grp.create_dataset(
                        "time",
                        data=coil_data.time,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    coil_grp.create_dataset(
                        "intensity",
                        data=coil_data.intensity,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    coil_grp.attrs["label"] = f"Mirnov Coil {coil_num}"
                    coil_grp.attrs["units"] = "V"

            # Plasma Detection
            if "plasma_detection" in all_data:
                plasma = all_data["plasma_detection"]
                plasma_grp = f.create_group("plasma_detection")

                signals = [
                    ("bt_coil", plasma.bt_coil, "BT Coil"),
                    ("bt_coil_int", plasma.bt_coil_integrated, "BT Coil Integrated"),
                    ("rog_coil", plasma.rogowski_coil, "Rogowski Coil"),
                    (
                        "rog_coil_int",
                        plasma.rogowski_coil_integrated,
                        "Rogowski Coil Integrated",
                    ),
                    ("photocell", plasma.leybold_photocell, "Leybold Photocell"),
                    ("loop_signal", plasma.loop_signal, "Loop Signal"),
                ]

                for name, signal, label in signals:
                    if signal is not None:
                        sig_grp = plasma_grp.create_group(name)
                        sig_grp.create_dataset(
                            "time",
                            data=signal.time,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                        sig_grp.create_dataset(
                            "intensity",
                            data=signal.intensity,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                        sig_grp.attrs["label"] = label
                        sig_grp.attrs["units"] = "V"

            # Shot info
            if "info" in all_data:
                info = all_data["info"]
                info_grp = f.create_group("shot_info")
                info_grp.attrs["shot_number"] = info.shot_number
                if info.logbook_text:
                    info_grp.attrs["logbook"] = info.logbook_text
                if info.timestamp:
                    info_grp.attrs["timestamp"] = info.timestamp

                # Available diagnostics
                if info.available_diagnostics:
                    diag_grp = info_grp.create_group("available_diagnostics")
                    for diag, available in info.available_diagnostics.items():
                        diag_grp.attrs[diag] = available

        logger.info(f"Successfully exported shot {self.shot_number} to {output_path}")
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"File size: {file_size:.2f} MB")

    def get_available_diagnostics(self) -> Dict[str, bool]:
        """
        Check which diagnostics are available for this shot.

        Returns:
            Dictionary mapping diagnostic names to availability (True/False)

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> available = loader.get_available_diagnostics()
            >>> print(available)
            {'Hα': True, 'Hβ': True, 'He I': True, ...}
        """
        availability = {}

        # Check fast spectrometry files
        for line in SpectroscopyLine:
            url = f"{self.base_url}FastSpectrometry/{line.filename}"
            try:
                urllib.request.urlopen(url, timeout=5)
                availability[line.display_name] = True
            except:
                availability[line.display_name] = False

        # Check mini-spectrometer
        url = f"{self.base_url}MiniSpectrometer/DAS_raw_data_dir/IRVISUV_0.h5"
        try:
            urllib.request.urlopen(url, timeout=5)
            availability["MiniSpectrometer"] = True
        except:
            availability["MiniSpectrometer"] = False

        # Check basic diagnostics
        basic_files = ["Bt", "Ip", "Ich", "U_loop"]
        basic_available = False
        for filename in basic_files:
            url = f"{self.base_url}BasicDiagnostics/Results/{filename}.csv"
            try:
                urllib.request.urlopen(url, timeout=5)
                basic_available = True
                break
            except:
                pass
        availability["BasicDiagnostics"] = basic_available

        # Check Mirnov coils
        mirnov_available = False
        for coil_num in [1, 5, 9, 13]:
            url = f"{self.base_url}LimiterMirnovCoils/U_mc{coil_num}.csv"
            try:
                urllib.request.urlopen(url, timeout=5)
                mirnov_available = True
                break
            except:
                pass
        availability["MirnovCoils"] = mirnov_available

        # Check MHD ring
        mhd_available = False
        for ring_num in range(1, 6):
            url = f"{self.base_url}MHDring-TM/ring_{ring_num}.csv"
            try:
                urllib.request.urlopen(url, timeout=5)
                mhd_available = True
                break
            except:
                pass
        availability["MHDRing"] = mhd_available

        # Check plasma detection
        plasma_files = ["U_BtCoil", "U_RogCoil", "U_Loop"]
        plasma_available = False
        for filename in plasma_files:
            url = f"{self.base_url}PlasmaDetection/{filename}.csv"
            try:
                urllib.request.urlopen(url, timeout=5)
                plasma_available = True
                break
            except:
                pass
        availability["PlasmaDetection"] = plasma_available

        return availability


# Convenience function for quick data loading
def load_shot_data(
    shot_number: int,
    include_spectrometry: bool = True,
    include_minispectrometer: bool = True,
    **kwargs,
) -> tuple[Optional[Dict[str, FastSpectrometryData]], Optional[MiniSpectrometerData]]:
    """
    Convenience function to load all available data for a shot.

    Args:
        shot_number: GOLEM shot number
        include_spectrometry: Whether to load fast spectrometry data
        include_minispectrometer: Whether to load mini-spectrometer data
        **kwargs: Additional arguments passed to GolemDataLoader

    Returns:
        Tuple of (spectrometry_data, minispectrometer_data)
        Either can be None if not requested or if loading fails

    Example:
        >>> spectrometry, minispec = load_shot_data(50377)
        >>> if spectrometry:
        ...     print(f"Loaded {len(spectrometry)} spectrometry signals")
    """
    loader = GolemDataLoader(shot_number, **kwargs)

    spectrometry_data = None
    minispectrometer_data = None

    if include_spectrometry:
        try:
            spectrometry_data = loader.load_fast_spectrometry()
        except DataLoadError as e:
            logger.error(f"Failed to load spectrometry data: {e}")

    if include_minispectrometer:
        try:
            minispectrometer_data = loader.load_minispectrometer_h5()
        except DataLoadError as e:
            logger.error(f"Failed to load mini-spectrometer data: {e}")

    return spectrometry_data, minispectrometer_data
