# GOLEM Data Loader - Installation & Usage Guide

## Installation

### Prerequisites

The GOLEM Data Loader requires Python 3.7+ with the following packages:

- pandas
- numpy
- h5py

### Installing Dependencies

You have several options:

#### Option 1: Install in your current Python environment

```bash
pip install pandas numpy h5py
```

Or using pip3:

```bash
pip3 install pandas numpy h5py
```

#### Option 2: Use a virtual environment (Recommended)

```bash
# Create virtual environment
python3 -m venv golem_env

# Activate it
source golem_env/bin/activate  # On macOS/Linux
# or
golem_env\Scripts\activate  # On Windows

# Install dependencies
pip install pandas numpy h5py
```

#### Option 3: Use conda

```bash
conda create -n golem python=3.9 pandas numpy h5py
conda activate golem
```

## Using the Package

### In Python/Jupyter (Recommended for Analysis)

The package works perfectly in Jupyter notebooks where pandas/numpy are already installed:

```python
from golem_data_loader import GolemDataLoader

loader = GolemDataLoader(50377)
basic = loader.load_basic_diagnostics()
spectrometry = loader.load_fast_spectrometry()
info = loader.load_shot_info()
```

### Using the CLI

The CLI requires pandas to be installed. Here are your options:

#### Option A: Run with the correct Python environment

If you have pandas installed in a virtual environment or conda:

```bash
# Activate your environment first
source golem_env/bin/activate  # or conda activate golem

# Then run the CLI
./golem-cli info 50377
```

#### Option B: Run directly with Python

```bash
# Use the Python that has pandas installed
python3 golem_data_loader/cli.py info 50377
```

If you have multiple Python installations, specify the one with pandas:

```bash
# For example, if using conda
/path/to/conda/envs/golem/bin/python golem_data_loader/cli.py info 50377

# Or if using Jupyter's Python
/path/to/jupyter/python golem_data_loader/cli.py info 50377
```

#### Option C: Create a shell script wrapper

Create `run-golem-cli.sh`:

```bash
#!/bin/bash
# Activate environment and run CLI
source golem_env/bin/activate
python golem_data_loader/cli.py "$@"
```

Make it executable:

```bash
chmod +x run-golem-cli.sh
./run-golem-cli.sh info 50377
```

## Quick Test

To verify your installation works:

```python
# Test in Python
python3 << EOF
try:
    import pandas as pd
    import numpy as np
    import h5py
    print("✓ All dependencies installed!")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
EOF
```

## Usage Examples

### In Jupyter Notebook (Best for Analysis)

```python
from golem_data_loader import GolemDataLoader

# Load data
loader = GolemDataLoader(50377)

# Basic diagnostics
basic = loader.load_basic_diagnostics()
if basic.plasma_current:
    import matplotlib.pyplot as plt
    plt.plot(basic.plasma_current.time, basic.plasma_current.intensity)
    plt.xlabel('Time (s)')
    plt.ylabel('Plasma Current (kA)')
    plt.show()

# All diagnostics
all_data = loader.load_all_diagnostics()
print(f"Loaded {len(all_data)} diagnostic types")
```

### Using CLI (After Installing Dependencies)

```bash
# Show shot info
./golem-cli info 50377

# List diagnostics
./golem-cli list 50377

# Load specific data
./golem-cli load 50377 basic
./golem-cli load 50377 spectrometry

# Export all data
./golem-cli export 50377 ./data/shot_50377
```

### In Python Scripts

```python
#!/usr/bin/env python3
"""
Example: Load and analyze GOLEM data
"""

from golem_data_loader import GolemDataLoader

def main():
    loader = GolemDataLoader(50377)

    # Load diagnostics
    basic = loader.load_basic_diagnostics()
    info = loader.load_shot_info()

    # Print summary
    print(f"Shot #{info.shot_number}")
    if basic.plasma_current:
        print(f"Plasma current: {len(basic.plasma_current.time)} points")

    # Export to CSV
    if basic.toroidal_field:
        basic.toroidal_field.raw_dataframe.to_csv('Bt.csv', index=False)
        print("Exported Bt.csv")

if __name__ == '__main__':
    main()
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install pandas in your Python environment:

```bash
pip3 install pandas numpy h5py
```

Or use the Python from an environment where it's installed (like Jupyter).

### Error: "golem-cli: command not found"

**Solution:** Run from the project directory:

```bash
cd /path/to/PRPL2025_ZS
./golem-cli info 50377
```

Or add to your PATH:

```bash
export PATH="$PATH:/path/to/PRPL2025_ZS"
```

### CLI works in Jupyter but not in terminal

**Solution:** Jupyter uses a different Python environment. Either:

1. Install pandas in your system Python:

   ```bash
   pip3 install pandas numpy h5py
   ```

2. Or run CLI with Jupyter's Python:

   ```bash
   # Find Jupyter's Python
   which python  # (run this in Jupyter)

   # Use it for CLI
   /path/to/jupyter/python golem_data_loader/cli.py info 50377
   ```

## Recommended Workflow

1. **For Data Analysis**: Use Jupyter notebooks

   - Dependencies already installed
   - Interactive exploration
   - Easy plotting

2. **For Quick Inspection**: Use the CLI

   - After installing dependencies
   - Great for shell scripts
   - Fast data checks

3. **For Batch Processing**: Use Python scripts
   - Combine API with automation
   - Process multiple shots
   - Generate reports

## Getting Help

### CLI Help

```bash
./golem-cli --help
./golem-cli info --help
./golem-cli load --help
```

### Python Help

```python
from golem_data_loader import GolemDataLoader
help(GolemDataLoader)
help(GolemDataLoader.load_basic_diagnostics)
```

### Documentation

- `README.md` - Full API documentation
- `CLI_GUIDE.md` - Complete CLI guide
- `QUICKSTART.md` - Quick start guide
- `EXTENSION_SUMMARY.md` - New features overview

## Summary

**Best Practice:**

- Use the package in Jupyter notebooks for analysis (pandas already installed)
- Use the CLI after installing dependencies with `pip3 install pandas numpy h5py`
- The Python API and CLI provide the same data, choose based on your workflow

**Remember:** The CLI is a convenience tool. For programmatic access and analysis, the Python API in Jupyter is more powerful and flexible!
