# Advection Velocity Estimation Methods

This document describes the two advection velocity estimation methods implemented for wind ramp tracking and comparison.

## Overview

Two methods have been implemented to estimate advection velocity:

1. **850 hPa Wind Field Method**: Uses the wind speed field at 850 hPa from ERA5 as a direct estimate of advection velocity
2. **Temporal Difference Optical Flow Method**: Applies optical flow to the temporal difference of wind fields (Δu(t) = u(t) - u(t-1))

## Methods Description

### Method 1: 850 hPa Wind Field

**Rationale**: The 850 hPa pressure level (approximately 1.5 km altitude) is above the planetary boundary layer and represents free-atmosphere flow. Wind at this level is less affected by surface friction and turbulence, making it a good proxy for the large-scale advection of weather patterns.

**Implementation**: `extract_850hpa_wind_advection()` in `era5_wind_advection.py`

**Advantages**:
- Simple and direct measurement from ERA5 data
- Represents large-scale atmospheric flow
- Less affected by local surface effects
- Well-established in meteorology

**Limitations**:
- May not capture small-scale advection patterns
- Assumes advection at surface follows upper-level winds
- Requires pressure-level ERA5 data (additional download)

**Data Requirements**:
- ERA5 pressure-level data with u and v wind components at 850 hPa

### Method 2: Temporal Difference Optical Flow (Ramp Tracking)

**Rationale**: Wind ramps manifest as rapid changes in wind speed. By computing Δu(t) = u(t) - u(t-1), we isolate regions of acceleration and deceleration. This method first **detects ramps** in the difference field, then applies optical flow specifically to track how these ramp patterns propagate through space.

**Implementation**: `compute_advection_velocity_temporal_difference()` in `era5_wind_advection.py`

**Algorithm Steps**:
1. Compute temporal differences at two consecutive times:
   - Δu(t) = u(t) - u(t-1) and Δu(t+1) = u(t+1) - u(t)
   - Δv(t) = v(t) - v(t-1) and Δv(t+1) = v(t+1) - v(t)
2. **Detect ramps** in the difference fields:
   - Identify regions where magnitude exceeds threshold (e.g., 1 m/s)
   - AND rate of change exceeds threshold (e.g., 1 m/s per time step)
   - Creates a `ramp_mask` showing where ramps exist
3. Apply spatial smoothing to reduce noise (Gaussian filter) while preserving ramp edges
4. Apply optical flow to track how the ramp patterns move from t to t+1
5. Apply edge masking based on gradient strength
6. Mask boundary regions to avoid edge artifacts
7. Combine masks: only report advection where ramps are present AND gradients are strong

**Advantages**:
- Specifically tracks wind ramp propagation (not just any changes)
- Captures both acceleration and deceleration ramps
- Can detect local-scale advection features
- Only requires single-level wind data
- More physically meaningful: tracks actual ramp features

**Limitations**:
- Requires 3 consecutive time steps (outputs n_times-2 instead of n_times-1)
- Sensitive to ramp detection thresholds
- Requires careful tuning of smoothing parameters
- Only provides advection estimates where ramps are present
- May miss advection in regions without strong ramps

**Data Requirements**:
- ERA5 single-level data (e.g., 100m wind) with hourly temporal resolution
- At least 3 consecutive time steps for each advection estimate

## Key Parameters

### Temporal Difference Method

- **ramp_magnitude_threshold** (default: 1.0 m/s): Minimum magnitude of change to be considered a ramp
  - Based on Δu = u(t) - u(t-1)
  - Higher values = only detect strong ramps
  - Lower values = detect weaker ramps but more noise
  - Recommended range: 0.5 - 2.0 m/s
  - Typical significant ramp: 1-3 m/s change over 1 hour

- **ramp_rate_threshold** (default: 1.0 m/s per time_step): Minimum rate of change to be considered a ramp
  - Filters out slow gradual changes
  - For hourly data: 1 m/s per hour
  - Higher values = only rapid ramps
  - Lower values = includes slower ramps
  - Recommended range: 0.5 - 2.0 m/s per time_step

- **smooth_sigma** (default: 2.0): Gaussian smoothing sigma for difference fields
  - Applied after ramp detection
  - Higher values = more smoothing, less noise but blurred ramp edges
  - Lower values = sharper features but noisier
  - Recommended range: 1.0 - 3.0

- **edge_threshold** (default: 0.1): Threshold for gradient-based masking
  - Regions with normalized gradients below this are masked
  - Higher values = more aggressive masking, fewer valid regions
  - Lower values = more coverage but potentially noisy regions
  - Recommended range: 0.05 - 0.2

## Special Considerations

### Ramp Detection

The temporal difference method first detects ramps, then tracks them. Proper tuning of ramp thresholds is critical:

- **ramp_magnitude_threshold**: Too high = miss weak ramps; too low = track noise as ramps
- **ramp_rate_threshold**: Too high = miss gradual ramps; too low = track slow changes

**Recommendation**: For wind turbine applications, start with:
- Magnitude: 1.0 m/s (detects changes significant for power output)
- Rate: 1.0 m/s/hour (distinguishes ramps from diurnal cycles)

Adjust based on your application and turbine power curve sensitivity.

### Smoothing

The difference field Δu can be noisy, especially in regions with weak gradients. Spatial smoothing (applied after ramp detection) helps but must be balanced:

- **Too little smoothing**: Noisy advection estimates, unreliable tracking
- **Too much smoothing**: Blurred ramp edges, reduced spatial resolution

**Recommendation**: Start with sigma=2.0 and adjust based on visual inspection of results.

### Boundary Effects

The difference field has edge artifacts when ramp patterns enter or exit the domain between timesteps. This is handled by:

1. **Edge masking**: Automatically masks 5 pixels from domain boundaries
2. **Gradient-based masking**: Masks regions with weak gradients (unreliable optical flow)
3. **Domain adjustment**: Consider extending the spatial domain if ramps frequently cross boundaries

### Sign and Direction

The difference field gives both positive (acceleration) and negative (deceleration) values. Ramps of both types are detected and tracked:

- **Positive Δu**: Wind speed increasing (up-ramps) - detected when magnitude and rate exceed thresholds
- **Negative Δu**: Wind speed decreasing (down-ramps) - detected when magnitude and rate exceed thresholds

The ramp detection uses **absolute magnitude** of the change, so both up-ramps and down-ramps are detected if they exceed the thresholds. The optical flow then tracks the spatial propagation of these ramp patterns, regardless of sign.

## Usage Example

```python
from era5_wind_advection import compute_advection_grid_from_era5

# Method 1: 850 hPa wind
advection_850hpa = compute_advection_grid_from_era5(
    era5_file="era5_wind_100m.nc",  # Not used but required
    output_file="advection_850hpa.nc",
    method='850hpa',
    era5_pressure_file="era5_pressure_levels.nc",
    pressure_level=850
)

# Method 2: Temporal difference
advection_temporal = compute_advection_grid_from_era5(
    era5_file="era5_wind_100m.nc",
    output_file="advection_temporal_diff.nc",
    method='temporal_difference',
    time_step_hours=1
)
```

## Wind Ramp Detection

Wind ramps are detected using the `RampDetector` class:

```python
from wind_correlation_analysis.src.analysis.ramp_detection import RampDetector

detector = RampDetector(
    time_window_hours=1.0,           # Time window for change detection
    magnitude_threshold_ms=4.0,      # Minimum change (m/s)
    rate_threshold_ms_per_hour=4.0,  # Minimum rate of change (m/s/hour)
    smooth_sigma=1.0                 # Smoothing for noise reduction
)

# Detect ramps in time series
ramps = detector.detect_ramps_timeseries(wind_speed_series)

# Detect ramps in gridded data
ramp_grid = detector.detect_ramps_grid(wind_speed_grid, times)
```

**Ramp Detection Parameters**:

- **time_window_hours**: Duration over which to compute wind speed changes
  - Smaller windows = detect rapid ramps
  - Larger windows = detect gradual ramps
  - Recommended: 0.5 - 2.0 hours for turbine-scale ramps

- **magnitude_threshold_ms**: Minimum absolute change in wind speed
  - Typical values: 3-5 m/s for significant ramps
  - Based on turbine power curve sensitivity

- **rate_threshold_ms_per_hour**: Minimum rate of change
  - Filters out slow gradual changes
  - Typical values: 3-6 m/s/hour

## Validation

### Comparing Methods

Use `AdvectionValidator` to compare the two methods:

```python
from wind_correlation_analysis.src.analysis.advection_validation import AdvectionValidator

validator = AdvectionValidator()

comparison = validator.compare_advection_methods(
    advection_data1=advection_850hpa,
    advection_data2=advection_temporal,
    method1_name="850 hPa Wind",
    method2_name="Temporal Difference",
    sample_points=[(59.35, 17.94), (59.65, 17.92)]  # Station locations
)

print(f"Speed correlation: {comparison['global_statistics']['speed_correlation']:.3f}")
print(f"Speed RMSE: {comparison['global_statistics']['speed_rmse']:.2f} m/s")
```

### Validating Against Observations

Validate advection estimates against observed ramp arrival times:

```python
# Prepare ramp arrival data
ramp_arrivals = pd.DataFrame({
    'station1': ['ESSB'],
    'station2': ['ESSA'],
    'ramp_time1': [pd.Timestamp('2024-01-01 12:00')],
    'ramp_time2': [pd.Timestamp('2024-01-01 12:30')],
    'distance_km': [25.0],
    'bearing_deg': [15.0]
})

station_locations = {
    'ESSB': (59.3544, 17.9419),
    'ESSA': (59.6519, 17.9186)
}

validation = validator.validate_against_ramp_arrivals(
    advection_data=advection_850hpa,
    ramp_arrivals=ramp_arrivals,
    station_locations=station_locations,
    method_name="850 hPa Wind"
)

print(f"Speed RMSE: {validation['speed_rmse_ms']:.2f} m/s")
print(f"Direction MAE: {validation['direction_mae_deg']:.1f} degrees")
```

## Visualization

Generate side-by-side comparison plots:

```python
from wind_correlation_analysis.src.analysis.advection_validation import AdvectionPlotter

plotter = AdvectionPlotter(figsize=(16, 10))

fig = plotter.plot_side_by_side_comparison(
    advection_data1=advection_850hpa,
    advection_data2=advection_temporal,
    time_index=0,  # First time step
    method1_name="850 hPa Wind",
    method2_name="Temporal Difference",
    output_file="comparison.png"
)
```

This creates a 6-panel comparison:
1. Method 1 speed field
2. Method 2 speed field
3. Difference field
4. Method 1 vector field
5. Method 2 vector field
6. Scatter plot comparison

## Complete Example

See `example_advection_comparison.py` for a complete working example that:
1. Downloads ERA5 data
2. Computes advection with both methods
3. Detects wind ramps
4. Compares methods
5. Validates against observations
6. Generates comparison plots

Run it with:
```bash
python example_advection_comparison.py
```

## Output Files

The analysis generates several output files:

- `advection_850hpa.nc`: 850 hPa advection velocity grid
  - Variables: u_advection, v_advection, advection_speed
- `advection_temporal_difference.nc`: Temporal difference advection grid
  - Variables: u_advection, v_advection, advection_speed, advection_mask, ramp_mask
  - `advection_mask`: Combined mask (ramps + strong gradients + not near boundaries)
  - `ramp_mask`: Where ramps were detected in the difference field
- `ramp_detection_summary.csv`: Detected ramps at all stations (time series analysis)
- `comparison_t*.png`: Side-by-side comparison plots at different times
- `validation_*.png`: Validation scatter plots and error distributions

## References

**Optical Flow for Atmospheric Motion**:
- Corpetti, T., & Mémin, É. (2008). "Stochastic Uncertainty Models for the Luminance Consistency Assumption". *IEEE Trans. Pattern Analysis and Machine Intelligence*.

**Wind Ramp Detection**:
- Gallego, C., et al. (2015). "A wavelet-based approach for large wind power ramp characterisation". *Wind Energy*.

**850 hPa Advection**:
- Holton, J. R. (2004). "An Introduction to Dynamic Meteorology" (4th ed.). Academic Press.

## Troubleshooting

### Problem: High RMSE between methods

**Possible causes**:
- Different spatial scales captured by each method
- Temporal difference method affected by noise
- 850 hPa wind not representative of surface advection

**Solutions**:
- Increase smoothing parameter in temporal difference method
- Compare at multiple times and locations
- Consider averaging over larger spatial regions

### Problem: Many NaN values in temporal difference method

**Possible causes**:
- Weak gradients in difference field
- Edge masking too aggressive
- Ramp detection thresholds too high (no ramps detected)
- Insufficient spatial domain

**Solutions**:
- Reduce edge_threshold parameter
- Reduce ramp_magnitude_threshold and ramp_rate_threshold
- Check ramp_mask output to see if ramps are being detected
- Increase spatial domain size
- Check that input data has sufficient temporal resolution

### Problem: No ramps detected (ramp_mask is all False)

**Possible causes**:
- Ramp thresholds too high for your data
- Temporal resolution too coarse (e.g., 6-hourly instead of hourly)
- Study period has no significant wind events
- Data is too smooth (already heavily filtered)

**Solutions**:
- Lower ramp_magnitude_threshold (try 0.5 m/s)
- Lower ramp_rate_threshold (try 0.5 m/s per time step)
- Check wind speed time series for actual ramp events
- Use higher temporal resolution data (hourly recommended)
- Verify wind speed variability is realistic

### Problem: Poor validation against observations

**Possible causes**:
- Station observations not representative of ERA5 grid scale
- Timing errors in ramp detection
- Local effects not captured in ERA5

**Solutions**:
- Use more stations for validation
- Adjust ramp detection thresholds
- Consider ensemble of advection estimates

## Future Enhancements

Potential improvements to consider:

1. **Multi-level averaging**: Combine winds from multiple pressure levels (850, 700, 500 hPa)
2. **Ensemble methods**: Blend multiple advection estimates with adaptive weighting
3. **Machine learning**: Train ML models to predict advection from observations
4. **Spectral methods**: Use FFT-based tracking for periodic patterns
5. **Lagrangian tracking**: Follow air parcel trajectories directly

## Contact

For questions or issues, please open an issue on the GitHub repository.
