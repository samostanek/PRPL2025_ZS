"""
GOLEM Data Loader Module

This module provides a robust interface for loading diagnostic data from the
GOLEM tokamak web server. It includes retry logic, error handling, and type-safe
data structures.

Supported diagnostics:
- Fast Spectrometry (CSV data)
- Mini-Spectrometer (HDF5 data)
- Fast Cameras (Radial and Vertical, 80000 fps)

Example:
    >>> from golem_data_loader import GolemDataLoader
    >>> loader = GolemDataLoader(shot_number=50377)
    >>> spectrometry_data = loader.load_fast_spectrometry()
    >>> h5_data = loader.load_minispectrometer_h5()
    >>> cameras = loader.load_fast_cameras()
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

    H_ALPHA = ("Hα", "DAS_raw_data_dir/ch8.csv")
    Cl_II = ("Hb 486nm", "DAS_raw_data_dir/ch5.csv")
    HE_I = ("He I 588nm", "DAS_raw_data_dir/ch7.csv")
    WHOLE = ("Whole", "DAS_raw_data_dir/ch6.csv")
    C_II = ("C II 514.5nm", "DAS_raw_data_dir/ch4.csv")
    N_II = ("N II 568.6nm", "DAS_raw_data_dir/ch3.csv")
    O_I = ("O I 777nm", "DAS_raw_data_dir/ch2.csv")
    He_I = ("He I 447.1nm", "DAS_raw_data_dir/ch1.csv")

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
class FastCameraData:
    """Container for fast camera data.

    Each frame is stored as a PNG file and captured at 80000 fps.
    Frames are 1 pixel tall (height dimension squeezed out).

    Attributes:
        camera_type: Type of camera ('radial' or 'vertical')
        frames: Array of frame data with shape [num_frames, width, channels] for RGB images
        frame_rate: Frame rate in fps (default: 80000)
        time: Time array in seconds for each frame
    """

    camera_type: str
    frames: np.ndarray
    frame_rate: float = 80000.0
    time: Optional[np.ndarray] = None

    def __post_init__(self):
        """Calculate time array if not provided."""
        if self.time is None:
            num_frames = self.frames.shape[0]
            self.time = np.arange(num_frames) / self.frame_rate


@dataclass
class PlasmaTiming:
    """Container for plasma detection timing data.

    Attributes:
        t_plasma_start_ms: Plasma start time in milliseconds
        t_plasma_end_ms: Plasma end time in milliseconds
        t_plasma_start: Plasma start time in seconds
        t_plasma_end: Plasma end time in seconds
    """

    t_plasma_start_ms: float
    t_plasma_end_ms: float
    t_plasma_start: float = field(init=False)
    t_plasma_end: float = field(init=False)

    def __post_init__(self):
        """Calculate second values."""
        self.t_plasma_start = self.t_plasma_start_ms * 1e-3
        self.t_plasma_end = self.t_plasma_end_ms * 1e-3


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
        self.plasma_timing: Optional[PlasmaTiming] = None

        # Configure logging
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)

        # Automatically load plasma timing
        try:
            self.plasma_timing = self.load_plasma_timing()
        except Exception as e:
            logger.warning(f"Could not load plasma timing: {e}")

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

    def load_fast_cameras(
        self, cameras: Optional[List[str]] = None, max_frames: Optional[int] = None
    ) -> Dict[str, FastCameraData]:
        """
        Load fast camera frame data.

        Each camera captures one-pixel-wide frames at 80000 fps.
        Frames are stored as individual PNG files (1.png, 2.png, etc.).

        Args:
            cameras: List of camera types to load ('radial', 'vertical').
                    If None, loads both cameras.
            max_frames: Maximum number of frames to load. If None, loads until
                       a frame is not found (assumes sequential numbering).

        Returns:
            Dictionary mapping camera type to FastCameraData objects

        Raises:
            DataLoadError: If data cannot be loaded
            DataValidationError: If loaded data fails validation

        Example:
            >>> loader = GolemDataLoader(50377)
            >>> cameras = loader.load_fast_cameras()
            >>> radial_data = cameras['radial']
            >>> print(radial_data.frames.shape)
        """
        if cameras is None:
            cameras = ["radial", "vertical"]

        results: Dict[str, FastCameraData] = {}
        failed_loads: List[tuple[str, Exception]] = []

        for camera_type in cameras:
            try:
                # Construct base URL based on camera type
                if camera_type.lower() == "radial":
                    base_url = f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/FastCameras/Camera_Radial/Frames/"
                    description = "Fast Camera (Radial)"
                elif camera_type.lower() == "vertical":
                    base_url = f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/FastCameras/Camera_Vertical/"
                    description = "Fast Camera (Vertical)"
                else:
                    raise ValueError(
                        f"Unknown camera type: {camera_type}. Must be 'radial' or 'vertical'"
                    )

                # Load PNG frames sequentially
                from io import BytesIO

                try:
                    from PIL import Image
                except ImportError:
                    raise DataLoadError(
                        "PIL/Pillow is required to load PNG frames. "
                        "Install with: pip install Pillow"
                    )

                frames_list = []
                frame_num = 1
                consecutive_failures = 0
                max_consecutive_failures = 5  # Stop after 5 consecutive missing frames

                logger.info(f"Loading {description} frames...")

                while True:
                    # Check if we've reached max_frames
                    if max_frames is not None and frame_num > max_frames:
                        break

                    # Check if we've had too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.info(
                            f"Stopped after {max_consecutive_failures} consecutive missing frames"
                        )
                        break

                    frame_url = f"{base_url}{frame_num}.png"

                    try:
                        # Try to fetch the frame
                        frame_bytes = self._fetch_url_with_retry(
                            frame_url, f"{description} frame {frame_num}"
                        )

                        # Load PNG image
                        img = Image.open(BytesIO(frame_bytes))
                        # Convert to numpy array, keeping all RGB channels
                        frame_array = np.array(img)

                        frames_list.append(frame_array)
                        consecutive_failures = 0  # Reset counter on success
                        frame_num += 1

                        # Log progress every 100 frames
                        if frame_num % 100 == 0:
                            logger.info(f"Loaded {frame_num} frames...")

                    except FileNotFoundError:
                        # Frame not found, increment failure counter
                        consecutive_failures += 1
                        frame_num += 1
                        continue

                    except Exception as e:
                        logger.warning(f"Error loading frame {frame_num}: {e}")
                        consecutive_failures += 1
                        frame_num += 1
                        continue

                if not frames_list:
                    raise DataLoadError(f"No frames found for {camera_type} camera")

                # Stack all frames into a single array
                # frames shape before squeeze: [num_frames, height, width, channels]
                # Since height is always 1, squeeze it out: [num_frames, width, channels]
                frames = np.array(frames_list)

                # Remove the height dimension (axis=1) since it's always 1
                if frames.ndim == 4 and frames.shape[1] == 1:
                    frames = frames.squeeze(axis=1)

                # Create structured data object
                camera_data = FastCameraData(
                    camera_type=camera_type.lower(), frames=frames, frame_rate=80000.0
                )

                results[camera_type.lower()] = camera_data
                logger.info(
                    f"{description}: loaded {frames.shape[0]} frames, "
                    f"frame size = {frames.shape[1:]} pixels"
                )

            except FileNotFoundError as e:
                logger.warning(f"File not found for {camera_type} camera: {e}")
                failed_loads.append((camera_type, e))

            except Exception as e:
                logger.error(f"Failed to load {camera_type} camera: {e}")
                failed_loads.append((camera_type, e))

        if not results and failed_loads:
            # All loads failed
            raise DataLoadError(
                f"Failed to load any camera data. Errors: {failed_loads}"
            )

        if failed_loads:
            logger.warning(
                f"Partially loaded data. Failed: {[name for name, _ in failed_loads]}"
            )

        return results

    def load_plasma_timing(self) -> PlasmaTiming:
        """
        Load plasma detection timing data.

        Returns:
            PlasmaTiming object containing plasma start and end times

        Raises:
            DataLoadError: If timing data cannot be loaded

        Example:
            >>> loader = GolemDataLoader(50930)
            >>> timing = loader.load_plasma_timing()
            >>> print(f"Plasma from {timing.t_plasma_start_ms:.2f} to {timing.t_plasma_end_ms:.2f} ms")
        """
        # Fetch plasma start time
        start_url = f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/PlasmaDetection/Results/t_plasma_start"
        end_url = f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/PlasmaDetection/Results/t_plasma_end"

        try:
            start_data = self._fetch_url_with_retry(start_url, "plasma start time")
            end_data = self._fetch_url_with_retry(end_url, "plasma end time")

            # Parse the values (they are already in milliseconds)
            t_plasma_start_ms = float(start_data.decode("utf-8").strip())
            t_plasma_end_ms = float(end_data.decode("utf-8").strip())

            timing = PlasmaTiming(
                t_plasma_start_ms=t_plasma_start_ms, t_plasma_end_ms=t_plasma_end_ms
            )

            logger.info(
                f"Loaded plasma timing: {timing.t_plasma_start_ms:.2f} - {timing.t_plasma_end_ms:.2f} ms"
            )

            return timing

        except Exception as e:
            raise DataLoadError(f"Failed to load plasma timing: {e}") from e

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

        # Check fast cameras
        camera_urls = {
            "FastCamera_Radial": f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/FastCameras/Camera_Radial/Frames/",
            "FastCamera_Vertical": f"http://golem.fjfi.cvut.cz/shots/{self.shot_number}/Diagnostics/FastCameras/Camera_Vertical/",
        }
        for camera_name, url in camera_urls.items():
            try:
                urllib.request.urlopen(url, timeout=5)
                availability[camera_name] = True
            except:
                availability[camera_name] = False

        return availability


# Convenience function for quick data loading
def load_shot_data(
    shot_number: int,
    include_spectrometry: bool = True,
    include_minispectrometer: bool = True,
    include_fast_cameras: bool = False,
    **kwargs,
) -> tuple[
    Optional[Dict[str, FastSpectrometryData]],
    Optional[MiniSpectrometerData],
    Optional[Dict[str, FastCameraData]],
]:
    """
    Convenience function to load all available data for a shot.

    Args:
        shot_number: GOLEM shot number
        include_spectrometry: Whether to load fast spectrometry data
        include_minispectrometer: Whether to load mini-spectrometer data
        include_fast_cameras: Whether to load fast camera data
        **kwargs: Additional arguments passed to GolemDataLoader

    Returns:
        Tuple of (spectrometry_data, minispectrometer_data, fast_camera_data)
        Any can be None if not requested or if loading fails

    Example:
        >>> spectrometry, minispec, cameras = load_shot_data(50377, include_fast_cameras=True)
        >>> if spectrometry:
        ...     print(f"Loaded {len(spectrometry)} spectrometry signals")
        >>> if cameras:
        ...     print(f"Loaded {len(cameras)} cameras")
    """
    loader = GolemDataLoader(shot_number, **kwargs)

    spectrometry_data = None
    minispectrometer_data = None
    fast_camera_data = None

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

    if include_fast_cameras:
        try:
            fast_camera_data = loader.load_fast_cameras()
        except DataLoadError as e:
            logger.error(f"Failed to load fast camera data: {e}")

    return spectrometry_data, minispectrometer_data, fast_camera_data
