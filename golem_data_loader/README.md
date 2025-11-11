# GOLEM Data Loader Module

A robust, type-safe Python package for loading diagnostic data from the GOLEM tokamak web server.

## 🆕 Version 1.1.0 - New Features

- ✅ **Extended Diagnostics**: Load BasicDiagnostics, Mirnov Coils, MHD Ring, and Plasma Detection
- ✅ **Shot Information**: Access shot logbooks and metadata
- ✅ **Command-Line Interface**: New `golem-cli` tool for quick data inspection
- ✅ **Load All Data**: Single function to load all available diagnostics
- ✅ **Export Capability**: Export data to CSV files via CLI

## Features

- ✅ **Comprehensive Data Loading**: Support for 7+ diagnostic types
- ✅ **Type-safe**: Full type annotations for excellent IDE support and type checking
- ✅ **Robust error handling**: Automatic retries with configurable delays
- ✅ **Flexible API**: Load all or specific diagnostics
- ✅ **Data validation**: Built-in validation of loaded data
- ✅ **Structured data**: Returns well-structured dataclasses instead of raw dictionaries
- ✅ **Comprehensive logging**: Detailed logging for debugging and monitoring
- ✅ **Clean resource management**: Automatic cleanup of temporary files
- ✅ **CLI Tool**: Command-line interface for quick data access
- ✅ **Backward compatible**: Easy integration with existing code

## Installation

No installation needed - just ensure the following dependencies are available:

```bash
pip install pandas numpy h5py
```

## Quick Start

### Basic Usage

```python
from golem_data_loader import GolemDataLoader

# Initialize loader for a specific shot
loader = GolemDataLoader(shot_number=50377)

# Load fast spectrometry data
spectrometry_data = loader.load_fast_spectrometry()

# Access data for specific spectroscopy line
h_alpha = spectrometry_data['Hα']
print(f"Time array shape: {h_alpha.time.shape}")
print(f"Intensity array shape: {h_alpha.intensity.shape}")

# Load mini-spectrometer H5 data
h5_data = loader.load_minispectrometer_h5()
print(f"Spectra shape: {h5_data.spectra.shape}")
print(f"Wavelengths shape: {h5_data.wavelengths.shape}")

# Clean up temporary files when done
h5_data.cleanup()
```

### Loading Specific Lines

```python
from golem_data_loader import GolemDataLoader, SpectroscopyLine

loader = GolemDataLoader(50377)

# Load only H-alpha and H-beta
data = loader.load_fast_spectrometry(
    lines=[SpectroscopyLine.H_ALPHA, SpectroscopyLine.H_BETA]
)
```

### Convenience Function

```python
from golem_data_loader import load_shot_data

# Load all available data at once
spectrometry, minispec = load_shot_data(50377)

if spectrometry:
    print(f"Loaded {len(spectrometry)} spectroscopy signals")

if minispec:
    print(f"Loaded spectra with shape {minispec.spectra.shape}")
```

### 🆕 Loading Other Diagnostics

```python
loader = GolemDataLoader(50377)

# Basic diagnostics (Bt, Ip, Ich, U_loop)
basic = loader.load_basic_diagnostics()
if basic.plasma_current:
    print(f"Plasma current: {len(basic.plasma_current.time)} points")

# Mirnov coils
mirnov = loader.load_mirnov_coils()
for coil_num, signal in mirnov.coils.items():
    print(f"Coil {coil_num}: {len(signal.time)} points")

# MHD ring
mhd = loader.load_mhd_ring()
print(f"Loaded {len(mhd.rings)} rings")

# Plasma detection
plasma = loader.load_plasma_detection()
if plasma.rog_coil:
    print(f"Rogowski coil: {len(plasma.rog_coil.time)} points")

# Shot information and logbook
info = loader.load_shot_info()
print(f"Shot #{info.shot_number}")
print(f"Logbook: {len(info.logbook)} characters")

# Load everything at once
all_data = loader.load_all_diagnostics()
print(f"Loaded {len(all_data)} diagnostic categories")
```

### 🆕 Using the CLI

```bash
# Show shot information
./golem-cli info 50377

# List available diagnostics
./golem-cli list 50377

# Load specific diagnostic
./golem-cli load 50377 basic
./golem-cli load 50377 spectrometry

# Export all data
./golem-cli export 50377 ./data/shot_50377
```

See `CLI_GUIDE.md` for complete CLI documentation.

### Checking Data Availability

```python
loader = GolemDataLoader(50377)
available = loader.get_available_diagnostics()

for diagnostic, is_available in available.items():
    status = "✓" if is_available else "✗"
    print(f"{status} {diagnostic}")
```

### Custom Configuration

```python
from golem_data_loader import GolemDataLoader, LoaderConfig
import logging

# Custom configuration
config = LoaderConfig(
    max_retries=5,           # Try 5 times before giving up
    retry_delay=3.0,         # Wait 3 seconds between retries
    timeout=60.0             # 60 second timeout for requests
)

loader = GolemDataLoader(
    shot_number=50377,
    config=config,
    log_level=logging.DEBUG  # Enable debug logging
)
```

## API Reference

### Classes

#### `GolemDataLoader`

Main class for loading GOLEM diagnostic data.

**Constructor:**

```python
GolemDataLoader(
    shot_number: int,
    config: Optional[LoaderConfig] = None,
    log_level: int = logging.INFO
)
```

**Methods:**

- `load_fast_spectrometry(lines=None, validate=True)` - Load fast spectrometry CSV data
- `load_minispectrometer_h5(h5_filename="IRVISUV_0.h5", keep_temp_file=False)` - Load mini-spectrometer HDF5 data
- `get_available_diagnostics()` - Check which diagnostics are available

#### `FastSpectrometryData`

Structured container for fast spectrometry data.

**Attributes:**

- `label: str` - Human-readable label (e.g., "Hα")
- `time: np.ndarray` - Time array in seconds
- `intensity: np.ndarray` - Intensity values (arbitrary units)
- `raw_dataframe: pd.DataFrame` - Original pandas DataFrame

#### `MiniSpectrometerData`

Structured container for mini-spectrometer data.

**Attributes:**

- `spectra: np.ndarray` - 2D array of spectral data [time, wavelength]
- `wavelengths: np.ndarray` - 1D array of wavelength values
- `temp_file_path: Optional[Path]` - Path to temporary H5 file

**Methods:**

- `cleanup()` - Remove temporary H5 file

#### `SpectroscopyLine` (Enum)

Enumeration of available spectroscopy lines.

**Values:**

- `H_ALPHA` - Hydrogen alpha line
- `H_BETA` - Hydrogen beta line
- `HE_I` - Helium I line
- `WHOLE` - Full spectrum

#### `LoaderConfig`

Configuration dataclass for the loader.

**Attributes:**

- `base_url_template: str` - URL template for GOLEM shots
- `max_retries: int` - Maximum retry attempts (default: 3)
- `retry_delay: float` - Delay between retries in seconds (default: 2.0)
- `timeout: float` - Request timeout in seconds (default: 30.0)

### Exceptions

- `DataLoadError` - Base exception for data loading errors
- `FileNotFoundError` - Resource not found on server (404)
- `NetworkError` - Network-related errors
- `DataValidationError` - Data validation failures

## Error Handling

The module uses a hierarchy of custom exceptions for precise error handling:

```python
from golem_data_loader import GolemDataLoader, FileNotFoundError, NetworkError

loader = GolemDataLoader(50377)

try:
    data = loader.load_fast_spectrometry()
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except NetworkError as e:
    print(f"Network error occurred: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Migration Guide

If you have existing code using manual data loading, migration is straightforward:

### Before:

```python
import urllib
import pandas as pd

shotno = 50377
base_url = f"http://golem.fjfi.cvut.cz/shots/{shotno}/Diagnostics/"

data = {}
for label, fname in files.items():
    url = base_url + "FastSpectrometry/" + fname
    df = pd.read_csv(url)
    data[label] = df
```

### After:

```python
from golem_data_loader import GolemDataLoader

loader = GolemDataLoader(50377)
spectrometry_data = loader.load_fast_spectrometry()

# For backward compatibility, extract raw DataFrames
data = {name: spec.raw_dataframe for name, spec in spectrometry_data.items()}
```

## Best Practices

1. **Use structured data**: Prefer accessing `.time` and `.intensity` attributes over DataFrame columns for type safety

2. **Handle partial failures**: The loader returns partial data if some files are unavailable

   ```python
   data = loader.load_fast_spectrometry()
   if 'Hα' in data:
       # Process H-alpha data
       pass
   ```

3. **Clean up resources**: Always cleanup temporary H5 files when done

   ```python
   h5_data = loader.load_minispectrometer_h5()
   try:
       # Use the data
       process_spectra(h5_data.spectra)
   finally:
       h5_data.cleanup()
   ```

4. **Use logging**: Enable appropriate logging level for debugging

   ```python
   import logging
   loader = GolemDataLoader(50377, log_level=logging.DEBUG)
   ```

5. **Check availability first**: For production code, check data availability before attempting to load
   ```python
   available = loader.get_available_diagnostics()
   if available['Hα']:
       data = loader.load_fast_spectrometry(lines=[SpectroscopyLine.H_ALPHA])
   ```

## Design Principles

This module follows several software engineering best practices:

- **Type Safety**: Comprehensive type annotations for better IDE support and fewer runtime errors
- **Separation of Concerns**: Clear separation between data loading, validation, and processing
- **Fail Fast**: Immediate validation with clear error messages
- **Resource Management**: Automatic cleanup of temporary files
- **Defensive Programming**: Extensive error handling and validation
- **Single Responsibility**: Each class has a focused purpose
- **Documentation**: Comprehensive docstrings and examples
- **Testability**: Pure functions and dependency injection for easy testing

## License

This module is provided as-is for use with GOLEM tokamak data analysis.

## Contributing

When extending this module:

1. Maintain type annotations for all public APIs
2. Add comprehensive docstrings with examples
3. Handle errors gracefully with appropriate exception types
4. Add logging at appropriate levels
5. Write validation tests for new data types
6. Update this README with new features
