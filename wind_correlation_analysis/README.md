# Wind Farm Spatio-Temporal Correlation Analysis

A comprehensive framework for analyzing spatio-temporal correlations in wind power production across multiple locations, with normalized advection parameters.

## Overview

This project implements a sophisticated analysis methodology for understanding how wind patterns propagate across geographical regions. It normalizes time lags and directions using advection velocity to reveal directional dependencies in wind correlation patterns.

### Key Features

- **Multi-source Data Integration**: Automatically downloads and processes:
  - METAR wind speed observations (5-minute resolution)
  - ERA5 reanalysis data for geostrophic wind calculations

- **Spatial Analysis**: Calculates distances and bearings between all station pairs

- **Normalized Cross-Correlation**:
  - Time lags normalized by τ = distance/speed
  - Directions normalized by θ = advection_direction - bearing
  - Analysis binned by 30° direction intervals

- **Advanced Visualization**: Polar plots showing correlation patterns vs normalized direction and time lag

## Methodology

For each pair of wind measurement sites, the analysis:

1. **Finds advection velocity** (speed and direction):
   - Direction: Geostrophic wind from ERA5 (850 hPa)
   - Speed: Scaled geostrophic wind speed (proxy)

2. **Calculates spatial relationship**:
   - Distance between stations (Haversine formula)
   - Bearing from station 1 to station 2

3. **Normalizes parameters**:
   - τ = distance / advection_speed (normalized time lag)
   - θ = advection_direction - bearing (normalized direction)

4. **Bins data** by normalized direction (default 30° bins)

5. **Computes cross-correlation** for each direction bin

6. **Visualizes results** in polar plots with:
   - θ (normalized direction) as angular coordinate
   - τ (normalized time lag) as radial coordinate
   - Correlation coefficient as color

## Project Structure

```
wind_correlation_analysis/
├── config/
│   └── stations.yaml          # Station configuration
├── data/
│   ├── raw/
│   │   ├── metar/            # Downloaded METAR data
│   │   └── era5/             # Downloaded ERA5 data
│   └── processed/            # Processed datasets
├── src/
│   ├── data_acquisition/
│   │   ├── metar_downloader.py    # METAR data acquisition
│   │   └── era5_downloader.py     # ERA5 data acquisition
│   ├── analysis/
│   │   ├── spatial_utils.py       # Spatial calculations
│   │   └── correlation_analysis.py # Cross-correlation analysis
│   └── visualization/
│       └── polar_plots.py         # Polar plot generation
├── results/
│   ├── data/                 # Analysis results (CSV)
│   └── figures/              # Generated plots
├── notebooks/                # Jupyter notebooks
├── requirements.txt
├── run_analysis.py           # Main pipeline script
└── README.md
```

## Installation

### 1. Clone or download this repository

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up ERA5 data access

To download ERA5 data, you need a free account at the Copernicus Climate Data Store:

1. Register at: https://cds.climate.copernicus.eu/
2. Install the CDS API key:
   - Go to https://cds.climate.copernicus.eu/api-how-to
   - Copy your API key
   - Create `~/.cdsapirc` with:
     ```
     url: https://cds.climate.copernicus.eu/api/v2
     key: YOUR_UID:YOUR_API_KEY
     ```

**Note**: If you skip this step, the analysis will use synthetic ERA5 data for demonstration purposes.

## Usage

### Quick Start

Run the analysis with default settings (January 2024, Swedish stations):

```bash
python run_analysis.py
```

This will:
1. Download METAR data for configured stations
2. Generate synthetic ERA5 data (or download if credentials are set up)
3. Calculate station pair relationships
4. Perform correlation analysis
5. Generate polar plots and summary visualizations

### Custom Date Range

```bash
python run_analysis.py --start-date 2024-01-01 --end-date 2024-02-29
```

### Download Real ERA5 Data

```bash
python run_analysis.py --download-era5
```

*Requires CDS API credentials (see Installation step 3)*

### Custom Station Configuration

Edit `config/stations.yaml` to add/modify stations:

```yaml
stations:
  - id: "ESSB"
    name: "Stockholm-Bromma"
    lat: 59.3544
    lon: 17.9419
  # Add more stations...
```

## Configuration

### Station Configuration (`config/stations.yaml`)

Define measurement stations with ICAO codes and coordinates. The package includes Swedish airports by default.

### Analysis Parameters

Modify in `run_analysis.py` or via `CrossCorrelationAnalysis` class:

- `bin_width`: Direction bin width in degrees (default: 30°)
- `max_lag_hours`: Maximum time lag to analyze (default: 24 hours)

### Advection Speed Proxies

Three methods are available in `era5_downloader.py`:

1. **from_era5_wind_speed**: Uses 850 hPa wind with height scaling
2. **from_geostrophic_wind**: Uses geostrophic wind with scaling factor
3. **from_gradient_wind**: Advanced method considering curvature (placeholder)

Default: geostrophic wind with 0.7 scaling factor

## Output

### Data Files (`results/data/`)

CSV files with correlation results for each station pair:
- `correlation_STATION1_STATION2.csv`

Columns:
- `bin`: Direction bin number
- `bin_center_deg`: Bin center angle
- `n_samples`: Number of samples in bin
- `tau_mean_sec/hours`: Mean normalized time lag
- `max_correlation`: Peak correlation coefficient
- `lag_at_max_samples/hours`: Time lag at peak correlation

### Visualizations (`results/figures/`)

1. **Polar correlation plots**: `correlation_polar_STATION1_STATION2.png`
   - Scatter plot in polar coordinates
   - Angular axis: normalized direction θ
   - Radial axis: normalized time lag τ
   - Color: correlation coefficient

2. **Bin statistics**: `bin_statistics_STATION1_STATION2.png`
   - Four subplots showing:
     - Correlation by direction
     - Data availability
     - Normalized time lag
     - Lag at maximum correlation

3. **Summary**: `correlation_vs_distance.png`
   - Maximum correlation vs station pair distance

## Scientific Background

### Advection Concept

Wind patterns propagate across regions through atmospheric advection. By normalizing:
- **Time lag** by distance/speed: Accounts for different station separations
- **Direction** by bearing: Reveals whether wind patterns align with station geometry

### Expected Patterns

- **θ ≈ 0°**: Wind blowing from station 1 toward station 2
  - High correlation expected
  - Positive time lag (station 2 delayed)

- **θ ≈ 180°**: Wind blowing from station 2 toward station 1
  - High correlation expected
  - Negative time lag (station 1 delayed)

- **θ ≈ ±90°**: Wind perpendicular to station pair
  - Lower correlation expected
  - Near-zero time lag

### Applications

This methodology can be used for:
- Wind power forecasting
- Understanding mesoscale wind patterns
- Validating numerical weather prediction models
- Optimizing wind farm layouts
- Power grid management

## Future Enhancements

Planned features mentioned in the project description:

1. **Wind Ramp Tracking**: Track rapid wind changes and derive advection velocity directly from observations

2. **Multiple Advection Proxies**: Compare different methods for estimating advection speed

3. **Ensemble Analysis**: Incorporate uncertainty in advection parameters

4. **Machine Learning**: Predict correlations based on meteorological conditions

## Data Sources

### METAR Data
- Source: Iowa State ASOS/METAR archive
- URL: https://mesonet.agron.iastate.edu/request/download.phtml
- Resolution: Variable (typically 20-60 minutes), resampled to 5 minutes
- Coverage: Global airport network

### ERA5 Reanalysis
- Source: ECMWF Copernicus Climate Data Store
- URL: https://cds.climate.copernicus.eu/
- Resolution: Hourly, 0.25° × 0.25° grid
- Variables: Geopotential, wind components at pressure levels

## Citation

If you use this code in your research, please cite:

```
[Your Name/Institution] (2024). Wind Farm Spatio-Temporal Correlation Analysis.
GitHub repository: [URL]
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources (SYNOP, wind farm SCADA data)
- Advanced advection detection algorithms
- Statistical significance testing
- Performance optimization for large datasets

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

- Iowa State University for METAR data access
- ECMWF Copernicus Programme for ERA5 reanalysis data
- Swedish Meteorological and Hydrological Institute (SMHI) for inspiration

## Troubleshooting

### Common Issues

**Problem**: "No data received for station"
- **Solution**: Check ICAO code is correct and station has wind data for requested period

**Problem**: "cdsapi not installed" or "CDS API error"
- **Solution**: Either install cdsapi and set up credentials, or use synthetic ERA5 data (default behavior)

**Problem**: "Insufficient data for correlation analysis"
- **Solution**: Extend date range or check data quality at stations

**Problem**: Empty polar plots
- **Solution**: Check that wind speed and ERA5 data overlap in time. Verify time zones (all data should be in UTC).

### Debug Mode

For detailed logging, modify the logging level in `run_analysis.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Notes

- Analysis of 6 stations for 1 month: ~5-10 minutes (depending on download speed)
- Memory usage: Typically < 1 GB for monthly analysis
- Disk space: ~100 MB per month of data for 10 stations

## References

1. Vincent, C. L., et al. (2011). "Cross-correlation analysis of meteorological data for wind turbine power production."
2. Tarroja, B., et al. (2013). "Spatial and temporal analysis of electric wind generation intermittency."
3. Holttinen, H. (2005). "Impact of hourly wind power variations on the system operation in the Nordic countries."

---

**Version**: 0.1.0
**Last Updated**: 2024
**Status**: Active Development
