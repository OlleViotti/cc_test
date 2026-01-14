"""
Example usage of ERA5 Wind Advection module

This script demonstrates how to:
1. Download ERA5 wind data for a specific region and time period
2. Compute advection velocities from the downloaded data
"""

from era5_wind_advection import (
    download_era5_wind_grid,
    compute_advection_grid_from_era5
)
from datetime import datetime


def example_download_and_compute_advection():
    """
    Complete example: download ERA5 data and compute advection velocity.
    """

    # Define parameters
    output_era5 = 'era5_wind_data.nc'
    output_advection = 'era5_advection_velocity.nc'

    # Time period
    start_date = '2024-01-01'
    end_date = '2024-01-02'
    time_hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

    # Geographic area (optional)
    # Format: [North, West, South, East] in degrees
    # Example: North Sea region
    area = [60, -5, 50, 10]  # North Sea
    # For global data, use: area = None

    # Wind turbine hub height
    height = 100  # meters (typical for modern wind turbines)

    print("=" * 70)
    print("ERA5 Wind Advection Analysis Example")
    print("=" * 70)

    # Step 1: Download ERA5 wind data
    print("\n[STEP 1] Downloading ERA5 wind data...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Time steps: {len(time_hours)} per day")
    print(f"  Height: {height}m (wind turbine hub height)")

    try:
        download_era5_wind_grid(
            output_file=output_era5,
            start_date=start_date,
            end_date=end_date,
            time_hours=time_hours,
            area=area,
            height=height
        )
    except Exception as e:
        print(f"\nError downloading data: {e}")
        print("\nNote: You need to configure CDS API credentials first:")
        print("1. Sign up at: https://cds.climate.copernicus.eu/")
        print("2. Create ~/.cdsapirc with your API key")
        print("\nSkipping download step for this example...")
        return

    # Step 2: Compute advection velocity
    print("\n[STEP 2] Computing advection velocity from wind data...")
    print("  This analyzes how wind patterns move over time")
    print("  (different from wind speed itself)")

    try:
        result_ds = compute_advection_grid_from_era5(
            era5_file=output_era5,
            output_file=output_advection,
            method='crosscorr',  # or 'optical_flow' for faster computation
            time_step_hours=3  # Time difference for advection computation
        )

        # Display results
        print("\n[RESULTS]")
        print(f"  Output file: {output_advection}")
        print(f"  Dimensions: {dict(result_ds.dims)}")
        print(f"  Variables: {list(result_ds.data_vars)}")

        # Print some statistics
        print("\n[STATISTICS]")
        print(f"  Mean advection speed: {result_ds['advection_speed'].mean().values:.2f} m/s")
        print(f"  Max advection speed: {result_ds['advection_speed'].max().values:.2f} m/s")
        print(f"  Mean wind speed: {result_ds['wind_speed'].mean().values:.2f} m/s")

        print("\n✓ Advection velocity computation complete!")

    except Exception as e:
        print(f"\nError computing advection: {e}")
        import traceback
        traceback.print_exc()


def example_quick_test():
    """
    Quick test with synthetic data (no download needed).
    """
    import numpy as np
    import xarray as xr
    from era5_wind_advection import compute_advection_velocity_crosscorr

    print("=" * 70)
    print("Quick Test: Advection Velocity Computation (Synthetic Data)")
    print("=" * 70)

    # Create synthetic wind field that moves to the right
    nx, ny = 100, 100
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)

    # Create a Gaussian blob
    center_x1, center_y = 3.0, 5.0
    center_x2 = 4.0  # Shifted right

    field1 = np.exp(-((X - center_x1)**2 + (Y - center_y)**2) / 0.5)
    field2 = np.exp(-((X - center_x2)**2 + (Y - center_y)**2) / 0.5)

    # Grid spacing and time step
    dx = 10000  # 10 km per grid point
    dy = 10000  # 10 km per grid point
    dt = 3600   # 1 hour

    # Calculate displacement in grid points
    # The coordinate space goes from 0-10 in 100 points
    coord_spacing = (x[1] - x[0])  # spacing in coordinate units
    displacement_coords = center_x2 - center_x1  # displacement in coordinate units
    displacement_gridpts = displacement_coords / coord_spacing  # displacement in grid points

    print("\nComputing advection velocity...")
    print(f"  Pattern shifted {displacement_gridpts:.1f} grid points ({displacement_gridpts * dx / 1000:.1f} km) to the east")
    print(f"  Time step: {dt / 3600:.1f} hours")

    u_adv, v_adv = compute_advection_velocity_crosscorr(
        field1, field2, dx, dy, dt, max_displacement=20
    )

    # Expected advection velocity (eastward)
    expected_u = displacement_gridpts * dx / dt  # displacement in grid points * spacing/gridpt / time
    computed_u = np.mean(u_adv[~np.isnan(u_adv)])

    computed_v = np.mean(v_adv[~np.isnan(v_adv)])

    print(f"\nResults:")
    print(f"  Expected eastward advection: {expected_u:.2f} m/s")
    print(f"  Computed eastward advection: {computed_u:.2f} m/s")
    print(f"  Northward advection: {computed_v:.2f} m/s (should be ~0)")

    if abs(computed_u - expected_u) < 5.0 and abs(computed_v) < 5.0:
        print("\n✓ Test passed! Advection velocity computed correctly.")
    else:
        print(f"\n✗ Test failed. Error: {abs(computed_u - expected_u):.2f} m/s")


if __name__ == '__main__':
    import sys

    print("\n")
    print("Choose an example to run:")
    print("1. Quick test with synthetic data (no download needed)")
    print("2. Full example: download ERA5 data and compute advection")
    print()

    choice = input("Enter choice (1 or 2, default=1): ").strip()

    if choice == '2':
        example_download_and_compute_advection()
    else:
        example_quick_test()
