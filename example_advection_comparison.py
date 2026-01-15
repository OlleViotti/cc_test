#!/usr/bin/env python3
"""
Example: Comparing Advection Velocity Estimation Methods

This script demonstrates how to:
1. Compute advection velocity using two methods:
   - 850 hPa wind from ERA5 pressure-level data
   - Optical flow on temporal difference fields (Δu = u(t) - u(t-1))
2. Detect wind ramps in the data
3. Compare the two methods
4. Validate against observed ramp arrival times at stations
5. Generate comparison plots

Author: Wind Correlation Analysis Team
Date: 2024
"""

import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wind_correlation_analysis'))

# Import modules
from era5_wind_advection import (
    download_era5_wind_grid,
    compute_advection_grid_from_era5,
    extract_850hpa_wind_advection
)
from wind_correlation_analysis.src.analysis.ramp_detection import (
    RampDetector,
    RampAdvectionTracker
)
from wind_correlation_analysis.src.analysis.advection_validation import (
    AdvectionValidator,
    AdvectionPlotter
)
from wind_correlation_analysis.src.data_acquisition.era5_downloader import ERA5Downloader
from wind_correlation_analysis.src.data_acquisition.metar_downloader import METARDownloader


def main():
    """
    Main function to run the advection comparison analysis.
    """
    print("=" * 80)
    print("ADVECTION VELOCITY COMPARISON ANALYSIS")
    print("=" * 80)

    # ========================================
    # Configuration
    # ========================================
    output_dir = Path("output/advection_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date range for analysis
    start_date = "2024-01-01"
    end_date = "2024-01-03"

    # Geographic area [North, West, South, East]
    # Stockholm region
    area = [60.5, 16.0, 58.5, 19.0]

    # Station locations for validation
    stations = {
        'ESSB': (59.3544, 17.9419),  # Stockholm-Bromma
        'ESSA': (59.6519, 17.9186),  # Stockholm-Arlanda
    }

    # ========================================
    # Step 1: Download/Load ERA5 Data
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: Downloading ERA5 Data")
    print("=" * 80)

    era5_single_file = output_dir / "era5_wind_100m.nc"
    era5_pressure_file = output_dir / "era5_pressure_levels.nc"

    # Download single-level data (100m wind)
    if not era5_single_file.exists():
        print("\nDownloading 100m wind data...")
        try:
            download_era5_wind_grid(
                output_file=str(era5_single_file),
                start_date=start_date,
                end_date=end_date,
                time_hours=['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                           '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                           '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                           '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                area=area,
                height=100
            )
        except Exception as e:
            print(f"Warning: Could not download single-level data: {e}")
            print("Please ensure you have CDS API credentials configured.")
            print("You can still run the comparison if you have existing data files.")
    else:
        print(f"Using existing file: {era5_single_file}")

    # Download pressure-level data (850 hPa)
    if not era5_pressure_file.exists():
        print("\nDownloading 850 hPa wind data...")
        try:
            downloader = ERA5Downloader(output_dir=str(output_dir))
            downloader.download_pressure_level_data(
                start_date=start_date,
                end_date=end_date,
                area=area,
                pressure_levels=[850],
                variables=['u_component_of_wind', 'v_component_of_wind'],
                time_hours=['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                           '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                           '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                           '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
            )
            # Rename to expected name
            downloaded_files = list(output_dir.glob("era5_pressure_*.nc"))
            if downloaded_files:
                os.rename(downloaded_files[0], era5_pressure_file)
        except Exception as e:
            print(f"Warning: Could not download pressure-level data: {e}")
            print("The 850 hPa method will not be available.")
    else:
        print(f"Using existing file: {era5_pressure_file}")

    # ========================================
    # Step 2: Compute Advection - Method 1 (850 hPa wind)
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: Computing Advection - Method 1 (850 hPa Wind)")
    print("=" * 80)

    advection_850hpa_file = output_dir / "advection_850hpa.nc"

    if era5_pressure_file.exists():
        print("\nComputing 850 hPa wind advection...")
        try:
            advection_850hpa = compute_advection_grid_from_era5(
                era5_file=str(era5_single_file),  # Not used but required
                output_file=str(advection_850hpa_file),
                method='850hpa',
                era5_pressure_file=str(era5_pressure_file),
                pressure_level=850
            )
            print(f"✓ Method 1 complete: {advection_850hpa_file}")
        except Exception as e:
            print(f"Error computing 850 hPa advection: {e}")
            advection_850hpa = None
    else:
        print("Skipping 850 hPa method (no pressure-level data)")
        advection_850hpa = None

    # ========================================
    # Step 3: Compute Advection - Method 2 (Temporal Difference)
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: Computing Advection - Method 2 (Temporal Difference)")
    print("=" * 80)

    advection_temporal_file = output_dir / "advection_temporal_difference.nc"

    if era5_single_file.exists():
        print("\nComputing temporal difference advection...")
        print("This method applies optical flow to Δu(t) = u(t) - u(t-1)")
        try:
            advection_temporal = compute_advection_grid_from_era5(
                era5_file=str(era5_single_file),
                output_file=str(advection_temporal_file),
                method='temporal_difference',
                time_step_hours=1
            )
            print(f"✓ Method 2 complete: {advection_temporal_file}")
        except Exception as e:
            print(f"Error computing temporal difference advection: {e}")
            advection_temporal = None
    else:
        print("Cannot compute temporal difference method (no single-level data)")
        advection_temporal = None

    # ========================================
    # Step 4: Detect Wind Ramps
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: Detecting Wind Ramps")
    print("=" * 80)

    if era5_single_file.exists():
        print("\nDetecting wind ramps in 100m wind field...")
        try:
            # Load wind data
            ds = xr.open_dataset(era5_single_file)

            # Compute wind speed
            u_var = [v for v in ds.variables if 'u' in v and 'wind' in v][0]
            v_var = [v for v in ds.variables if 'v' in v and 'wind' in v][0]
            wind_speed = np.sqrt(ds[u_var]**2 + ds[v_var]**2)

            # Initialize ramp detector
            ramp_detector = RampDetector(
                time_window_hours=1.0,
                magnitude_threshold_ms=4.0,
                rate_threshold_ms_per_hour=4.0,
                smooth_sigma=1.0
            )

            # Detect ramps at station locations
            ramp_results = {}
            for station_id, (lat, lon) in stations.items():
                print(f"\n  Detecting ramps at {station_id} ({lat:.2f}°N, {lon:.2f}°E)...")

                # Extract time series at station
                ws_station = wind_speed.sel(
                    latitude=lat, longitude=lon, method='nearest'
                ).to_series()

                # Detect ramps
                ramps = ramp_detector.detect_ramps_timeseries(
                    ws_station, return_details=True
                )

                ramp_results[station_id] = ramps

                if len(ramps) > 0:
                    print(f"    Found {len(ramps)} ramps:")
                    print(f"      Up-ramps: {sum(ramps['direction']=='up')}")
                    print(f"      Down-ramps: {sum(ramps['direction']=='down')}")

            # Save ramp detection results
            ramp_summary_file = output_dir / "ramp_detection_summary.csv"
            all_ramps = []
            for station_id, ramps in ramp_results.items():
                if len(ramps) > 0:
                    ramps_copy = ramps.copy()
                    ramps_copy['station'] = station_id
                    all_ramps.append(ramps_copy)

            if all_ramps:
                pd.concat(all_ramps).to_csv(ramp_summary_file, index=False)
                print(f"\n✓ Ramp detection complete: {ramp_summary_file}")

        except Exception as e:
            print(f"Error detecting ramps: {e}")
            ramp_results = {}
    else:
        print("Cannot detect ramps (no wind data)")
        ramp_results = {}

    # ========================================
    # Step 5: Compare Advection Methods
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 5: Comparing Advection Methods")
    print("=" * 80)

    if advection_850hpa is not None and advection_temporal is not None:
        print("\nComparing 850 hPa vs Temporal Difference methods...")

        try:
            validator = AdvectionValidator()

            # Sample points for comparison
            sample_points = list(stations.values())

            comparison_results = validator.compare_advection_methods(
                advection_data1=advection_850hpa,
                advection_data2=advection_temporal,
                method1_name="850 hPa Wind",
                method2_name="Temporal Difference",
                sample_points=sample_points
            )

            # Print comparison statistics
            print("\n--- Comparison Statistics ---")
            stats = comparison_results['global_statistics']
            print(f"  Speed Correlation: {stats['speed_correlation']:.3f}")
            print(f"  Speed RMSE: {stats['speed_rmse']:.2f} m/s")
            print(f"  U-component RMSE: {stats['u_rmse']:.2f} m/s")
            print(f"  V-component RMSE: {stats['v_rmse']:.2f} m/s")
            print(f"\n  850 hPa mean speed: {stats['speed1_mean']:.2f} ± {stats['speed1_std']:.2f} m/s")
            print(f"  Temporal Diff mean speed: {stats['speed2_mean']:.2f} ± {stats['speed2_std']:.2f} m/s")

            print("\n✓ Comparison complete")

        except Exception as e:
            print(f"Error comparing methods: {e}")
            comparison_results = None
    else:
        print("Cannot compare methods (one or both advection datasets missing)")
        comparison_results = None

    # ========================================
    # Step 6: Generate Comparison Plots
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: Generating Comparison Plots")
    print("=" * 80)

    if advection_850hpa is not None and advection_temporal is not None:
        print("\nCreating side-by-side comparison plots...")

        try:
            plotter = AdvectionPlotter(figsize=(16, 10))

            # Plot multiple time steps
            n_times = min(6, len(advection_850hpa.time))

            for t_idx in range(0, n_times, max(1, n_times // 3)):
                output_file = output_dir / f"comparison_t{t_idx:03d}.png"

                fig = plotter.plot_side_by_side_comparison(
                    advection_data1=advection_850hpa,
                    advection_data2=advection_temporal,
                    time_index=t_idx,
                    method1_name="850 hPa Wind",
                    method2_name="Temporal Difference",
                    output_file=str(output_file)
                )
                plt.close(fig)

                print(f"  ✓ Created: {output_file}")

            print("\n✓ Comparison plots complete")

        except Exception as e:
            print(f"Error generating plots: {e}")

    # ========================================
    # Step 7: Validate Against Ramp Arrivals
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 7: Validating Against Ramp Arrivals")
    print("=" * 80)

    # Create synthetic ramp arrival data for demonstration
    # In practice, this would come from actual observations
    if len(ramp_results) >= 2:
        print("\nMatching ramps between stations...")

        try:
            # Get ramps from two stations
            station_ids = list(ramp_results.keys())
            if len(station_ids) >= 2:
                station1 = station_ids[0]
                station2 = station_ids[1]
                ramps1 = ramp_results[station1]
                ramps2 = ramp_results[station2]

                if len(ramps1) > 0 and len(ramps2) > 0:
                    # Calculate distance and bearing between stations
                    from wind_correlation_analysis.src.analysis.spatial_utils import SpatialCalculations

                    lat1, lon1 = stations[station1]
                    lat2, lon2 = stations[station2]

                    distance_km = SpatialCalculations.haversine_distance(lat1, lon1, lat2, lon2)
                    bearing_deg = SpatialCalculations.calculate_bearing(lat1, lon1, lat2, lon2)

                    # Match ramps
                    tracker = RampAdvectionTracker(max_time_lag_hours=6.0)
                    matched_ramps = tracker.match_ramps_between_stations(
                        ramps1, ramps2, distance_km, bearing_deg
                    )

                    if len(matched_ramps) > 0:
                        print(f"\n  Found {len(matched_ramps)} matched ramps between {station1} and {station2}")

                        # Prepare for validation
                        validation_df = pd.DataFrame({
                            'station1': station1,
                            'station2': station2,
                            'ramp_time1': matched_ramps['station1_ramp_start'],
                            'ramp_time2': matched_ramps['station2_ramp_start'],
                            'distance_km': distance_km,
                            'bearing_deg': bearing_deg
                        })

                        # Validate both methods
                        for method_name, adv_data in [
                            ("850 hPa Wind", advection_850hpa),
                            ("Temporal Difference", advection_temporal)
                        ]:
                            if adv_data is not None:
                                print(f"\n  Validating {method_name}...")

                                validation = validator.validate_against_ramp_arrivals(
                                    advection_data=adv_data,
                                    ramp_arrivals=validation_df,
                                    station_locations=stations,
                                    method_name=method_name
                                )

                                if validation:
                                    print(f"    RMSE: {validation['speed_rmse_ms']:.2f} m/s")
                                    print(f"    Bias: {validation['speed_bias_ms']:.2f} m/s")
                                    print(f"    MAE: {validation['speed_mae_ms']:.2f} m/s")

                                    # Plot validation results
                                    output_file = output_dir / f"validation_{method_name.replace(' ', '_')}.png"
                                    fig = plotter.plot_validation_results(
                                        validation,
                                        output_file=str(output_file)
                                    )
                                    if fig:
                                        plt.close(fig)
                                        print(f"    ✓ Plot saved: {output_file}")

                        print("\n✓ Validation complete")
                    else:
                        print("  No matching ramps found between stations")

        except Exception as e:
            print(f"Error in validation: {e}")
    else:
        print("Insufficient ramp data for validation")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            print(f"  - {file.name}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
