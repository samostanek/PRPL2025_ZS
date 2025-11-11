# Quick Start Guide - GOLEM Data Loader

## Installation

No installation needed! The module is self-contained. Just ensure you have the dependencies:

```bash
pip install pandas numpy h5py
```

## 30-Second Quick Start

```python
from golem_data_loader import GolemDataLoader

# Load data from shot 50377
loader = GolemDataLoader(50377)
data = loader.load_fast_spectrometry()

# Access H-alpha data
h_alpha = data['Hα']
print(h_alpha.time.shape)      # Time array
print(h_alpha.intensity.shape) # Intensity array
```

## Common Tasks

### Load All Spectrometry Data

```python
loader = GolemDataLoader(50377)
data = loader.load_fast_spectrometry()

for name, signal in data.items():
    print(f"{name}: {len(signal.time)} data points")
```

### Load Specific Lines Only

```python
from golem_data_loader import SpectroscopyLine

loader = GolemDataLoader(50377)
data = loader.load_fast_spectrometry(
    lines=[SpectroscopyLine.H_ALPHA, SpectroscopyLine.H_BETA]
)
```

### Load Mini-Spectrometer Data

```python
loader = GolemDataLoader(50377)
h5_data = loader.load_minispectrometer_h5()

print(h5_data.spectra.shape)      # 2D array [time, wavelength]
print(h5_data.wavelengths.shape)  # Wavelength values

# Clean up when done
h5_data.cleanup()
```

### Check What's Available

```python
loader = GolemDataLoader(50377)
available = loader.get_available_diagnostics()

for diagnostic, is_available in available.items():
    print(f"{'✓' if is_available else '✗'} {diagnostic}")
```

### Load Everything at Once

```python
from golem_data_loader import load_shot_data

spectrometry, minispec = load_shot_data(50377)
```

### Handle Errors

```python
from golem_data_loader import GolemDataLoader, DataLoadError

try:
    loader = GolemDataLoader(50377)
    data = loader.load_fast_spectrometry()
except DataLoadError as e:
    print(f"Error: {e}")
```

### Custom Configuration

```python
from golem_data_loader import GolemDataLoader, LoaderConfig
import logging

config = LoaderConfig(
    max_retries=5,      # Try 5 times
    retry_delay=3.0,    # Wait 3 seconds between retries
    timeout=60.0        # 60 second timeout
)

loader = GolemDataLoader(
    shot_number=50377,
    config=config,
    log_level=logging.DEBUG
)
```

## Migrating Existing Code

### Old Code

```python
import urllib
import pandas as pd

base_url = f"http://golem.fjfi.cvut.cz/shots/{shotno}/Diagnostics/"
files = {"Hα": "U_Halpha.csv", "Hβ": "U_Hbeta.csv"}

data = {}
for label, fname in files.items():
    url = base_url + "FastSpectrometry/" + fname
    df = pd.read_csv(url)
    data[label] = df
```

### New Code

```python
from golem_data_loader import GolemDataLoader

loader = GolemDataLoader(shotno)
spectrometry_data = loader.load_fast_spectrometry()

# For backward compatibility
data = {name: spec.raw_dataframe for name, spec in spectrometry_data.items()}
```

## Available Spectroscopy Lines

```python
from golem_data_loader import SpectroscopyLine

SpectroscopyLine.H_ALPHA    # Hydrogen alpha
SpectroscopyLine.H_BETA     # Hydrogen beta
SpectroscopyLine.HE_I       # Helium I
SpectroscopyLine.WHOLE      # Full spectrum
```

## Data Access

### Structured Access (Recommended)

```python
h_alpha = data['Hα']
time = h_alpha.time           # NumPy array
intensity = h_alpha.intensity # NumPy array
```

### DataFrame Access (Backward Compatible)

```python
h_alpha = data['Hα']
df = h_alpha.raw_dataframe
time = df.iloc[:, 0]
intensity = df.iloc[:, 1]
```

## Common Issues

### Connection Timeout

```python
# Increase timeout
config = LoaderConfig(timeout=120.0)
loader = GolemDataLoader(50377, config=config)
```

### Data Not Available

```python
# Check availability first
loader = GolemDataLoader(50377)
if loader.get_available_diagnostics()['Hα']:
    data = loader.load_fast_spectrometry()
```

### Need More Logging

```python
import logging
loader = GolemDataLoader(50377, log_level=logging.DEBUG)
```

## Need More Help?

- Full documentation: `README_golem_loader.md`
- Example scripts: `example_usage.py`
- Notebook example: `fast_spectrometry_graphs.ipynb`

## Module Files

- `golem_data_loader.py` - Main module
- `golem_data_loader.pyi` - Type stubs (for IDE support)
- `README_golem_loader.md` - Full documentation
- `example_usage.py` - Usage examples
