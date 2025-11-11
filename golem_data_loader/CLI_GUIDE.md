# GOLEM CLI User Guide

## Overview

The GOLEM CLI provides a command-line interface for loading and inspecting GOLEM tokamak diagnostic data.

## Installation

The CLI is included in the `golem_data_loader` package. Make sure you have the required dependencies:

```bash
pip install pandas numpy h5py
```

## Usage

### Basic Syntax

```bash
./golem-cli <command> [options]
```

Or directly:

```bash
python3 golem_data_loader/cli.py <command> [options]
```

## Commands

### 1. `info` - Show Shot Information

Display comprehensive information about a shot, including available diagnostics and logbook.

**Syntax:**

```bash
./golem-cli info <shot_number>
```

**Example:**

```bash
./golem-cli info 50377
```

**Output:**

```
======================================================================
GOLEM Shot #50377 Information
======================================================================

Available Diagnostics:
----------------------------------------------------------------------

Fast Spectrometry:
  ✓ Hα
  ✓ Hβ
  ✓ He I
  ✓ Whole

Basic Diagnostics:
  ✓ BasicDiagnostics
  ✓ PlasmaDetection

...

Shot Logbook (first 1000 characters):
----------------------------------------------------------------------
14:47:24 #50376   PingAllDevices:
14:47:25 #50376   CheckAbilityToMakeDischarge:  Ping OK
...
```

### 2. `list` - List Available Diagnostics

Show which diagnostics are available for a specific shot.

**Syntax:**

```bash
./golem-cli list <shot_number>
```

**Example:**

```bash
./golem-cli list 50377
```

**Output:**

```
======================================================================
Available Diagnostics for Shot #50377
======================================================================

✓ Available          BasicDiagnostics
✓ Available          FastCameras
✓ Available          Hα
✓ Available          Hβ
✓ Available          He I
...

12/15 diagnostics available
```

### 3. `load` - Load Specific Diagnostic

Load and display summary information for a specific diagnostic type.

**Syntax:**

```bash
./golem-cli load <shot_number> <diagnostic_type>
```

**Diagnostic Types:**

- `basic` / `basicdiagnostics` - Basic diagnostics (Bt, Ip, Ich, U_loop)
- `spectrometry` / `fastspectrometry` - Fast spectrometry signals
- `minispectrometer` / `minispec` - Mini-spectrometer H5 data
- `mirnov` / `mirnovcoils` - Limiter Mirnov coils
- `mhd` / `mhdring` - MHD ring signals
- `plasma` / `plasmadetection` - Plasma detection signals
- `all` - Load all available diagnostics

**Examples:**

Load basic diagnostics:

```bash
./golem-cli load 50377 basic
```

Output:

```
Basic Diagnostics Data:
--------------------------------------------------
✓ Toroidal Field (Bt): 1234 points
✓ Plasma Current (Ip): 1234 points
✓ Chamber Current (Ich): 1234 points
✓ Loop Voltage: 1234 points
```

Load fast spectrometry:

```bash
./golem-cli load 50377 spectrometry
```

Load all diagnostics:

```bash
./golem-cli load 50377 all
```

### 4. `export` - Export All Data

Export all available diagnostic data to a directory as CSV files.

**Syntax:**

```bash
./golem-cli export <shot_number> <output_directory>
```

**Example:**

```bash
./golem-cli export 50377 ./data/shot_50377
```

**Output:**

```
Exporting shot 50377 data to ./data/shot_50377...

Exported 15 files:
  • basic_Bt.csv
  • basic_Ich.csv
  • basic_Ip.csv
  • basic_U_loop.csv
  • mirnov_coil_1.csv
  • mirnov_coil_5.csv
  ...

All data exported to: /path/to/data/shot_50377
```

### 5. `export-hdf5` - Export to HDF5 File ✨ NEW

Export all diagnostic data to a single HDF5 file for efficient storage and easy sharing.

**Syntax:**

```bash
./golem-cli export-hdf5 <shot_number> <output_file> [options]
```

**Options:**

- `-d, --diagnostics`: Comma-separated list of diagnostics to include (default: all)
- `-c, --compression-level`: HDF5 compression level 0-9 (default: 4)
- `-v, --verbose`: Enable verbose output

**Examples:**

```bash
# Export all diagnostics to HDF5
./golem-cli export-hdf5 50377 shot_50377.h5

# Export only specific diagnostics
./golem-cli export-hdf5 50377 shot_50377.h5 -d basic,spectrometry,mhd_ring

# Use maximum compression
./golem-cli export-hdf5 50377 shot_50377.h5 -c 9

# Verbose mode to see what's being exported
./golem-cli export-hdf5 50377 shot_50377.h5 -v
```

**Output:**

```
Exporting shot 50377 to HDF5 format...

======================================================================
✓ Successfully exported shot 50377
  Output file: shot_50377.h5
  File size: 3.11 MB
  Compression: gzip (level 4)
======================================================================

To read this HDF5 file in Python:
  >>> import h5py
  >>> f = h5py.File('shot_50377.h5', 'r')
  >>> list(f.keys())  # Show available groups
  >>> plasma_current = f['basic/plasma_current/intensity'][:]
  >>> f.close()
```

**HDF5 File Structure:**

The exported HDF5 file has the following organization:

```
/metadata/
  - shot_number (attribute)
  - export_timestamp (attribute)
  - loader_version (attribute)

/basic/
  plasma_current/{time, intensity}
  toroidal_field/{time, intensity}
  chamber_current/{time, intensity}
  loop_voltage/{time, intensity}

/spectrometry/
  Hα/{time, intensity}
  Hβ/{time, intensity}
  He_I/{time, intensity}
  Whole/{time, intensity}

/mhd_ring/
  ring_1/{time, intensity}
  ring_2/{time, intensity}
  ...

/mirnov_coils/
  coil_1/{time, intensity}
  ...

/plasma_detection/
  bt_coil/{time, intensity}
  ...

/shot_info/
  - logbook (attribute)
  - timestamp (attribute)
  available_diagnostics/...
```

**Reading HDF5 Files:**

```python
import h5py
import matplotlib.pyplot as plt

# Open file and explore structure
with h5py.File('shot_50377.h5', 'r') as f:
    print("Groups:", list(f.keys()))
    print("Shot number:", f['metadata'].attrs['shot_number'])

    # Load plasma current
    time = f['basic/plasma_current/time'][:]
    current = f['basic/plasma_current/intensity'][:]

    # Plot
    plt.plot(time * 1000, current / 1000)
    plt.xlabel('Time (ms)')
    plt.ylabel('Plasma Current (kA)')
    plt.show()
```

**Advantages of HDF5:**

- **Single file**: All diagnostics in one file vs. many CSV files
- **Efficient**: Compressed storage (3-5 MB vs 3.5 MB for CSV)
- **Fast**: Binary format is faster to read/write
- **Metadata**: Self-documenting with attributes
- **Standard**: Widely supported format (Python, MATLAB, IDL, etc.)

## Global Options

### Verbose Mode

Add `-v` or `--verbose` to any command for detailed logging:

```bash
./golem-cli info 50377 -v
./golem-cli load 50377 basic --verbose
```

## Examples

### Quick Shot Overview

```bash
# Get basic info
./golem-cli info 50377

# Check what's available
./golem-cli list 50377

# Load the most important data
./golem-cli load 50377 basic
```

### Data Analysis Workflow

```bash
# 1. Check shot info
./golem-cli info 50377

# 2. Load specific diagnostics
./golem-cli load 50377 spectrometry
./golem-cli load 50377 mirnov

# 3. Export to HDF5 for analysis (NEW!)
./golem-cli export-hdf5 50377 shot_50377.h5

# Or export to CSV directory
./golem-cli export 50377 ./analysis/shot_50377
```

### Batch Processing Multiple Shots

```bash
#!/bin/bash
# Export multiple shots to HDF5

for shot in 50375 50376 50377 50378; do
    echo "Exporting shot $shot..."
    ./golem-cli export-hdf5 $shot "shot_${shot}.h5" -c 6
done

echo "All shots exported!"
```

### Selective Data Export

```bash
# Export only basic diagnostics and spectroscopy
./golem-cli export-hdf5 50377 basic_spec.h5 -d basic,spectrometry

# Export only MHD data with high compression
./golem-cli export-hdf5 50377 mhd_only.h5 -d mhd_ring,mirnov -c 9
```

### Troubleshooting

```bash
# Use verbose mode to see what's happening
./golem-cli load 50377 basic -v

# Check if shot exists
./golem-cli list 50377
```

## Error Handling

The CLI provides clear error messages:

```bash
$ ./golem-cli load 50377 invalid_diagnostic
Unknown diagnostic: invalid_diagnostic

Available diagnostics: basic, spectrometry, minispectrometer,
mirnov, mhd, plasma, all
```

```bash
$ ./golem-cli info 99999
Error loading shot info: Failed to load any spectrometry data
```

## Exit Codes

- `0` - Success
- `1` - Error (diagnostic not found, network error, etc.)

## Integration with Scripts

The CLI can be easily integrated into bash scripts:

```bash
#!/bin/bash

# Process multiple shots
for shot in 50375 50376 50377; do
    echo "Processing shot $shot..."
    ./golem-cli export $shot "./data/shot_$shot"
done
```

## Python API vs CLI

For programmatic access, use the Python API instead:

```python
from golem_data_loader import GolemDataLoader

loader = GolemDataLoader(50377)
basic = loader.load_basic_diagnostics()
info = loader.load_shot_info()
```

The CLI is best for:

- Quick data inspection
- Manual data exploration
- Batch exports
- Shell scripts

The Python API is best for:

- Data analysis
- Custom processing
- Integration with other Python code
- Interactive exploration in Jupyter

## Help

Get help for any command:

```bash
./golem-cli --help
./golem-cli info --help
./golem-cli load --help
```

## See Also

- `README.md` - Full package documentation
- `QUICKSTART.md` - Quick start guide
- `test_extended_loader.py` - Example usage in Python
