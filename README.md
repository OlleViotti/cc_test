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
