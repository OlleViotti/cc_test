# cc_test
Testing of Claude Code

## Wind Field Tracking Analysis

This repository contains a comprehensive analysis notebook for tracking wind fields using ERA5 reanalysis data.

### Features

- **ERA5 Data Download**: Automatically downloads wind data from the Climate Data Store
- **Wind Field Tracking**: Implements advanced tracking algorithms to estimate advection velocity
- **Extended Domain**: Uses a grid that extends beyond measurement stations for proper tracking
- **Rich Visualizations**: Creates multiple maps showing:
  - Wind speed fields as colored contours
  - ERA5 wind vectors
  - Tracked advection velocity as arrows
  - Measurement station locations
  - Time evolution of wind patterns

### Setup

1. Install dependencies:
# ERA5 Wind Advection Analysis

A Python toolkit for downloading ERA5 reanalysis wind data and computing advection velocities to track how wind patterns move over time.

## Overview

This package provides tools to:

1. **Download ERA5 wind data** at wind turbine hub height (100m) on a full grid
2. **Compute advection velocity** - the velocity at which wind patterns move through space
3. **Analyze wind pattern movement** for wind energy forecasting and meteorological applications

### What is Advection Velocity?

**Advection velocity** is fundamentally different from **wind velocity**:

- **Wind velocity**: How fast the air itself is moving
- **Advection velocity**: How fast weather patterns (e.g., wind speed patterns) are moving through space

For example, a region of high wind speeds might be moving northward at 10 m/s (advection velocity), while the wind within that region is blowing eastward at 15 m/s (wind velocity).

Understanding advection velocity is crucial for:
- Wind turbine power forecasting
- Short-term wind prediction
- Understanding synoptic weather pattern movement

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

2. Ensure you have a valid CDS API key in `.cdsapirc` file

3. Run the notebook:
```bash
jupyter notebook example_analysis.ipynb
```

### Notebook Contents

The `example_analysis.ipynb` notebook demonstrates:

1. Defining measurement stations (two offshore wind farms in Denmark)
2. Setting up an extended domain for tracking
3. Downloading ERA5 wind data via CDS API
4. Implementing wind tracking algorithms (cross-correlation and optical flow)
5. Creating comprehensive visualizations of wind fields and advection
6. Analyzing time series at station locations
7. Computing and displaying summary statistics

### Tracking Methodology

The notebook uses two complementary approaches:

- **Bulk Advection**: Cross-correlation to estimate overall wind pattern movement
- **Optical Flow**: Spatially-varying velocity field using Lucas-Kanade method

The tracking grid extends 2° beyond the station boundaries to properly capture wind field evolution.
### 2. Configure CDS API credentials

To download ERA5 data, you need a free account with the Copernicus Climate Data Store:

1. Sign up at: https://cds.climate.copernicus.eu/
2. Get your API credentials from your profile page
3. Create `~/.cdsapirc` file with your credentials:

```bash
cp .cdsapirc.example ~/.cdsapirc
# Edit ~/.cdsapirc with your actual UID and API key
```

## Features

### Unified ERA5 Downloader

The repository now includes a unified ERA5 downloader (`wind_correlation_analysis/src/data_acquisition/era5_downloader.py`) that consolidates all ERA5 download functionality:

- **Support for multiple data types**:
  - Pressure-level data (for geostrophic wind calculation)
  - Single-level data (100m, 10m wind for advection analysis)

- **Intelligent caching**:
  - Automatically caches downloaded files to avoid duplicate downloads
  - Checks both time period and geographic domain before downloading
  - Cache metadata stored in `.era5_cache.json`
  - Can be disabled by setting `enable_cache=False`

- **Consistent API**:
  - Unified error handling
  - Better logging and progress reporting
  - Flexible date format support (datetime objects or ISO strings)

## Quick Start

### Option 1: Quick Test (No Download Required)

Test the advection computation with synthetic data:

```bash
python example_usage.py
# Choose option 1
```

### Option 2: Full Example with ERA5 Data

Download ERA5 data and compute advection velocities:

```bash
python example_usage.py
# Choose option 2
```

## Usage

### Download ERA5 Wind Data

```python
from era5_wind_advection import download_era5_wind_grid

# Download wind data at 100m height (typical wind turbine hub height)
download_era5_wind_grid(
    output_file='era5_wind_data.nc',
    start_date='2024-01-01',
    end_date='2024-01-02',
    time_hours=['00:00', '06:00', '12:00', '18:00'],
    area=[60, -5, 50, 10],  # [North, West, South, East] or None for global
    height=100  # meters (wind turbine hub height)
)
```

### Compute Advection Velocity

```python
from era5_wind_advection import compute_advection_grid_from_era5

# Compute advection velocity from downloaded ERA5 data
result = compute_advection_grid_from_era5(
    era5_file='era5_wind_data.nc',
    output_file='advection_velocity.nc',
    method='crosscorr',  # or 'optical_flow' for faster computation
    time_step_hours=1
)

# Access results
print(f"Mean advection speed: {result['advection_speed'].mean().values:.2f} m/s")
```

## Output Data

The output NetCDF file contains:

- `u_advection`: Eastward advection velocity (m/s)
- `v_advection`: Northward advection velocity (m/s)
- `advection_speed`: Magnitude of advection velocity (m/s)
- `wind_speed`: Original wind speed for reference (m/s)
- `u_wind`: Eastward wind component (m/s)
- `v_wind`: Northward wind component (m/s)

All variables are on the same lat/lon grid with time coordinates.

## Methods

Two methods are available for computing advection velocity:

### 1. Cross-Correlation Method (`method='crosscorr'`)

- **More accurate** but slower
- Tracks wind patterns by finding maximum correlation between consecutive time steps
- Uses local windows to compute displacement of patterns
- Recommended for research and detailed analysis

### 2. Optical Flow Method (`method='optical_flow'`)

- **Faster** but potentially less accurate
- Uses gradient-based optical flow equation
- Suitable for quick analysis or real-time applications

## Technical Details

### Algorithm Overview

1. **Download**: Fetch ERA5 100m wind components (u, v) from Copernicus CDS
2. **Wind Speed**: Compute wind speed magnitude at each grid point
3. **Pattern Tracking**: Track how wind speed patterns move between time steps
4. **Advection Velocity**: Compute velocity of pattern movement

### Cross-Correlation Method

For each local window in the wind field:
1. Extract window from time t
2. Search for best matching window at time t+1 (within search radius)
3. Compute displacement that maximizes correlation
4. Convert displacement to velocity: `v = displacement / time_step`

### Optical Flow Method

Based on the optical flow equation:
```
∂I/∂t + u·∂I/∂x + v·∂I/∂y = 0
```

Where:
- `I` is the wind speed field (intensity)
- `(u, v)` is the advection velocity
- Solved using gradient-based least squares approach

## File Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── era5_wind_advection.py        # Main module
├── example_usage.py              # Example scripts
└── .cdsapirc.example             # CDS API config template
```

## Requirements

- Python 3.8+
- cdsapi (for downloading ERA5 data)
- xarray (for working with NetCDF data)
- numpy (numerical computations)
- scipy (signal processing and optimization)
- netCDF4 (file I/O)

## Applications

This toolkit is useful for:

- **Wind Energy**: Forecasting wind power generation by understanding pattern movement
- **Meteorology**: Studying synoptic-scale weather pattern propagation
- **Aviation**: Understanding wind pattern changes for flight planning
- **Air Quality**: Tracking pollution pattern movement

## Notes

- ERA5 provides reanalysis data, which combines model data with observations
- The 100m wind height is typical for modern wind turbine hub heights
- Grid resolution depends on ERA5 data (typically 0.25° × 0.25°)
- Advection velocity is typically smaller than wind velocity
- For best results with cross-correlation, ensure sufficient temporal resolution

## References

- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
- Copernicus Climate Data Store: https://cds.climate.copernicus.eu/

## License

This project is provided as-is for research and educational purposes.

## Contributing

This is a test project for Claude Code. Contributions and suggestions are welcome!
