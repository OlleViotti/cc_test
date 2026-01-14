# Quick Start Guide

## Wind Farm Spatio-Temporal Correlation Analysis

This guide will help you get started with the wind correlation analysis framework in 5 minutes.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python test_installation.py
   ```

   This will check that all packages are installed and the framework is working correctly.

## Running Your First Analysis

### Option 1: Command Line (Fastest)

Run the complete analysis pipeline with default settings:

```bash
python run_analysis.py
```

This will:
- Download METAR wind data for 6 Swedish airports (January 2024)
- Generate synthetic ERA5 data for demonstration
- Calculate correlations for all station pairs
- Generate polar plots and visualizations
- Save results to `results/` directory

**Expected runtime:** 5-10 minutes

### Option 2: Jupyter Notebook (Interactive)

For interactive exploration:

```bash
jupyter notebook notebooks/example_analysis.ipynb
```

Follow the notebook cells to run each step of the analysis.

## Understanding the Results

After running the analysis, you'll find:

### 1. Polar Plots (`results/figures/correlation_polar_*.png`)

These show the main results:
- **Angular axis (θ)**: Normalized direction (advection direction - bearing)
- **Radial axis (τ)**: Normalized time lag (distance/speed) in hours
- **Color**: Cross-correlation coefficient (-1 to 1)

**Interpretation:**
- θ ≈ 0°: Wind blowing from station 1 toward station 2 → High correlation expected
- θ ≈ 180°: Wind blowing opposite direction → High correlation, opposite lag
- θ ≈ ±90°: Wind perpendicular → Low correlation

### 2. Direction Bin Statistics (`results/figures/bin_statistics_*.png`)

Four subplots showing:
- Correlation by direction bin
- Data availability (sample count)
- Normalized time lag by bin
- Time lag at maximum correlation

### 3. Data Files (`results/data/correlation_*.csv`)

CSV files with detailed results for each station pair and direction bin.

## Customization

### Change Date Range

```bash
python run_analysis.py --start-date 2024-02-01 --end-date 2024-02-29
```

### Add/Modify Stations

Edit `config/stations.yaml`:

```yaml
stations:
  - id: "ICAO_CODE"
    name: "Station Name"
    lat: 59.3544
    lon: 17.9419
```

Find ICAO codes at: https://airportcodes.aero/

### Use Real ERA5 Data

1. Set up CDS API credentials (see README)
2. Run with:
   ```bash
   python run_analysis.py --download-era5
   ```

## Understanding the Methodology

### What is being analyzed?

For each pair of wind measurement locations, we:

1. **Extract wind speeds** at both locations (5-minute resolution)
2. **Get advection direction** from geostrophic wind (ERA5)
3. **Calculate normalized parameters:**
   - τ = distance / advection_speed
   - θ = advection_direction - bearing
4. **Bin by direction** (30° bins)
5. **Compute cross-correlation** for each bin

### Why normalize?

Without normalization:
- Different station separations aren't comparable
- Advection speed varies with weather conditions
- Time lags depend on both distance and wind speed

With normalization:
- All station pairs can be compared
- Physical interpretation is clearer
- Reveals directional dependencies

## Next Steps

1. **Extend analysis period** to capture seasonal variations
2. **Add more stations** to study regional patterns
3. **Experiment with bin widths** (modify `bin_width` parameter)
4. **Compare different advection proxies** (edit `era5_downloader.py`)
5. **Implement wind ramp tracking** for data-driven advection detection

## Troubleshooting

### "No data received for station"
- Check that ICAO code is correct
- Verify station has wind observations for your date range
- Try a different time period

### "cdsapi not installed"
- Either: `pip install cdsapi` and set up credentials
- Or: Use synthetic ERA5 data (default behavior)

### Plots look strange
- Check data quality with `notebooks/example_analysis.ipynb`
- Verify time alignment between METAR and ERA5 data
- Ensure sufficient data in each direction bin

## Questions?

See the full [README.md](README.md) for:
- Detailed methodology
- Scientific background
- API documentation
- Advanced configuration options

## Example Output

After running `python run_analysis.py`, you should see:

```
================================================================================
WIND FARM SPATIO-TEMPORAL CORRELATION ANALYSIS
================================================================================
Analysis period: 2024-01-01 to 2024-01-31

================================================================================
STEP 1: Downloading METAR wind speed data
================================================================================
Downloading ESSB (attempt 1/3)
Downloaded 8760 records for ESSB
...

================================================================================
ANALYSIS COMPLETE!
================================================================================
Results saved to:
  - Data: results/data/
  - Figures: results/figures/
```

Enjoy analyzing wind correlations! 🌬️
