"""
ERA5 Wind Advection Analysis

This module provides functions to:
1. Download ERA5 wind speed data on a full grid (via unified ERA5Downloader)
2. Compute advection velocity from wind pattern movement
"""

import sys
import os

# Add the wind_correlation_analysis package to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wind_correlation_analysis'))

from wind_correlation_analysis.src.data_acquisition.era5_downloader import ERA5Downloader
import xarray as xr
import numpy as np
from scipy import signal
from scipy.ndimage import shift, gaussian_filter
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
import warnings


def download_era5_wind_grid(
    output_file: str,
    start_date: str,
    end_date: str,
    time_hours: List[str],
    area: Optional[List[float]] = None,
    height: int = 100,
    variables: Optional[List[str]] = None
) -> str:
    """
    Download ERA5 wind speed data on a full grid using the unified ERA5Downloader.

    Parameters
    ----------
    output_file : str
        Path to save the downloaded NetCDF file
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    time_hours : list
        List of hours to download (e.g., ['00:00', '06:00', '12:00', '18:00'])
    area : list, optional
        Geographic area [North, West, South, East] in degrees
        Default is None (global)
    height : int, optional
        Height above ground level in meters (typical wind turbine hub height)
        Default is 100m
    variables : list, optional
        Variables to download. Default is auto-selected based on height

    Returns
    -------
    str
        Path to the downloaded file

    Notes
    -----
    This function uses the unified ERA5Downloader class to download wind data
    at wind turbine hub height (typically 80-100m). The data includes both
    u (eastward) and v (northward) wind components needed for advection
    velocity computation.

    You need to have a CDS API key configured (~/.cdsapirc) to use this function.
    Sign up at: https://cds.climate.copernicus.eu/
    """
    # Initialize the unified downloader
    # Extract the output directory from output_file
    output_dir = os.path.dirname(output_file) or '.'
    downloader = ERA5Downloader(output_dir=output_dir)

    # Download using the unified downloader
    downloaded_file = downloader.download_single_level_data(
        start_date=start_date,
        end_date=end_date,
        area=area,
        variables=variables,
        time_hours=time_hours,
        height=height
    )

    # Rename to the requested output file if different
    if downloaded_file != output_file:
        import shutil
        shutil.move(downloaded_file, output_file)
        print(f"Renamed to: {output_file}")

    return output_file


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute wind speed magnitude from u and v components.

    Parameters
    ----------
    u : np.ndarray
        U (eastward) component of wind
    v : np.ndarray
        V (northward) component of wind

    Returns
    -------
    np.ndarray
        Wind speed magnitude
    """
    return np.sqrt(u**2 + v**2)


def compute_advection_velocity_crosscorr(
    field1: np.ndarray,
    field2: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    max_displacement: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advection velocity using cross-correlation method.

    This function tracks how wind patterns move between two time steps
    by finding the displacement that maximizes cross-correlation.

    Parameters
    ----------
    field1 : np.ndarray
        Wind field at time t (2D array)
    field2 : np.ndarray
        Wind field at time t+dt (2D array)
    dx : float
        Grid spacing in x direction (meters or degrees)
    dy : float
        Grid spacing in y direction (meters or degrees)
    dt : float
        Time difference between fields (seconds)
    max_displacement : int, optional
        Maximum displacement to search in grid points (default: 20)

    Returns
    -------
    u_advection : np.ndarray
        Advection velocity in x direction
    v_advection : np.ndarray
        Advection velocity in y direction

    Notes
    -----
    The advection velocity represents how fast the wind patterns themselves
    are moving, which is different from the wind speed itself.
    """

    ny, nx = field1.shape
    u_advection = np.full((ny, nx), np.nan)
    v_advection = np.full((ny, nx), np.nan)

    # Window size for local cross-correlation
    window_size = 32
    half_window = window_size // 2

    # Stride should be smaller than window size for coverage
    stride = window_size // 2

    # Compute advection velocity using local cross-correlation
    for i in range(half_window, ny - half_window, stride):
        for j in range(half_window, nx - half_window, stride):
            # Extract local window from field1
            window1 = field1[i-half_window:i+half_window,
                            j-half_window:j+half_window]

            # Skip if window has no variation
            if window1.std() < 1e-6:
                continue

            # Search for best match in field2
            best_corr = -np.inf
            best_di = 0
            best_dj = 0

            for di in range(-max_displacement, max_displacement + 1):
                for dj in range(-max_displacement, max_displacement + 1):
                    # Extract shifted window from field2
                    i2_start = i - half_window + di
                    i2_end = i + half_window + di
                    j2_start = j - half_window + dj
                    j2_end = j + half_window + dj

                    # Check bounds
                    if (i2_start >= 0 and i2_end <= ny and
                        j2_start >= 0 and j2_end <= nx):

                        window2 = field2[i2_start:i2_end, j2_start:j2_end]

                        # Compute normalized cross-correlation
                        if window2.std() > 1e-6:
                            corr = np.corrcoef(window1.flatten(),
                                             window2.flatten())[0, 1]

                            if not np.isnan(corr) and corr > best_corr:
                                best_corr = corr
                                best_di = di
                                best_dj = dj

            # Compute advection velocity from displacement
            # Displacement is in grid points, convert to physical units
            u_adv = best_dj * dx / dt  # x-component (eastward)
            v_adv = best_di * dy / dt  # y-component (northward)

            # Assign to output grid (assign to window center region)
            i_start = max(0, i - stride // 2)
            i_end = min(ny, i + stride // 2)
            j_start = max(0, j - stride // 2)
            j_end = min(nx, j + stride // 2)

            u_advection[i_start:i_end, j_start:j_end] = u_adv
            v_advection[i_start:i_end, j_start:j_end] = v_adv

    # Fill any remaining NaN values with nearest neighbor interpolation
    from scipy.ndimage import distance_transform_edt

    for arr in [u_advection, v_advection]:
        mask = np.isnan(arr)
        if mask.any():
            ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
            arr[mask] = arr[tuple(ind[:, mask])]

    return u_advection, v_advection


def compute_advection_velocity_optical_flow(
    field1: np.ndarray,
    field2: np.ndarray,
    dx: float,
    dy: float,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advection velocity using optical flow method (gradient-based).

    This is a faster but potentially less accurate method compared to
    cross-correlation, based on the optical flow equation.

    Parameters
    ----------
    field1 : np.ndarray
        Wind field at time t (2D array)
    field2 : np.ndarray
        Wind field at time t+dt (2D array)
    dx : float
        Grid spacing in x direction (meters or degrees)
    dy : float
        Grid spacing in y direction (meters or degrees)
    dt : float
        Time difference between fields (seconds)

    Returns
    -------
    u_advection : np.ndarray
        Advection velocity in x direction
    v_advection : np.ndarray
        Advection velocity in y direction

    Notes
    -----
    Uses the optical flow equation: dI/dt + u*dI/dx + v*dI/dy = 0
    where I is the field intensity (wind speed).
    """

    # Compute temporal derivative
    dI_dt = (field2 - field1) / dt

    # Compute spatial gradients using central differences
    dI_dy, dI_dx = np.gradient(field1, dy, dx)

    # Solve for advection velocity using least squares
    # This is a simplified version; more sophisticated methods exist

    # Avoid division by zero
    grad_mag_sq = dI_dx**2 + dI_dy**2
    epsilon = 1e-10

    # Lucas-Kanade style solution
    # We need to solve: dI/dt + u*dI/dx + v*dI/dy = 0
    # This is one equation with two unknowns, so we use local averaging

    # For simplicity, use a regularized least-squares approach
    alpha = 0.1  # Regularization parameter

    u_advection = -(dI_dt * dI_dx) / (grad_mag_sq + alpha)
    v_advection = -(dI_dt * dI_dy) / (grad_mag_sq + alpha)

    # Smooth the result
    from scipy.ndimage import gaussian_filter
    u_advection = gaussian_filter(u_advection, sigma=2)
    v_advection = gaussian_filter(v_advection, sigma=2)

    return u_advection, v_advection


def extract_850hpa_wind_advection(
    era5_pressure_file: str,
    pressure_level: int = 850
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Extract wind at a specified pressure level (e.g., 850 hPa) as advection velocity estimate.

    Parameters
    ----------
    era5_pressure_file : str
        Path to ERA5 pressure-level NetCDF file
    pressure_level : int, optional
        Pressure level in hPa (default: 850)

    Returns
    -------
    u_advection : xr.DataArray
        U (eastward) component at specified pressure level
    v_advection : xr.DataArray
        V (northward) component at specified pressure level

    Notes
    -----
    The 850 hPa wind is commonly used as a proxy for advection velocity
    as it represents the free atmosphere above the boundary layer.
    """
    print(f"Loading ERA5 pressure-level data from {era5_pressure_file}...")
    ds = xr.open_dataset(era5_pressure_file)

    # Check for pressure/level dimension
    if 'level' in ds.dims:
        level_dim = 'level'
    elif 'pressure_level' in ds.dims:
        level_dim = 'pressure_level'
    elif 'isobaricInhPa' in ds.dims:
        level_dim = 'isobaricInhPa'
    else:
        raise ValueError(f"Could not find pressure level dimension. Dimensions: {list(ds.dims)}")

    # Check if requested level exists
    available_levels = ds[level_dim].values
    if pressure_level not in available_levels:
        raise ValueError(f"Pressure level {pressure_level} hPa not found. "
                        f"Available levels: {available_levels}")

    # Extract u and v components at the specified level
    u_var = None
    v_var = None

    for var in ds.variables:
        if 'u' in var.lower() and 'wind' in var.lower():
            u_var = var
        elif var == 'u':
            u_var = var
        if 'v' in var.lower() and 'wind' in var.lower():
            v_var = var
        elif var == 'v':
            v_var = var

    if u_var is None or v_var is None:
        raise ValueError(f"Could not find u/v wind components. Variables: {list(ds.variables)}")

    print(f"Found wind components: {u_var}, {v_var}")
    print(f"Extracting {pressure_level} hPa wind as advection velocity...")

    # Select the pressure level
    u_advection = ds[u_var].sel({level_dim: pressure_level})
    v_advection = ds[v_var].sel({level_dim: pressure_level})

    print(f"  U-component: mean={float(u_advection.mean()):.2f} m/s, "
          f"std={float(u_advection.std()):.2f} m/s")
    print(f"  V-component: mean={float(v_advection.mean()):.2f} m/s, "
          f"std={float(v_advection.std()):.2f} m/s")

    return u_advection, v_advection


def compute_advection_velocity_temporal_difference(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    smooth_sigma: float = 2.0,
    edge_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute advection velocity using optical flow on temporal difference fields.

    This method computes Δu(t) = u(t) - u(t-1) and Δv(t) = v(t) - v(t-1),
    applies smoothing and edge masking, then uses optical flow to track these
    difference patterns.

    Parameters
    ----------
    u_wind : np.ndarray
        U wind component at two consecutive times, shape (2, ny, nx)
        [u(t-1), u(t)]
    v_wind : np.ndarray
        V wind component at two consecutive times, shape (2, ny, nx)
        [v(t-1), v(t)]
    dx : float
        Grid spacing in x direction (meters)
    dy : float
        Grid spacing in y direction (meters)
    dt : float
        Time difference between fields (seconds)
    smooth_sigma : float, optional
        Sigma for Gaussian smoothing of difference fields (default: 2.0)
        Higher values = more smoothing
    edge_threshold : float, optional
        Threshold for edge masking based on gradient magnitude (default: 0.1)
        Regions with gradients below this threshold are masked

    Returns
    -------
    u_advection : np.ndarray
        Advection velocity in x direction (m/s)
    v_advection : np.ndarray
        Advection velocity in y direction (m/s)
    mask : np.ndarray
        Boolean mask indicating valid regions (True = valid)

    Notes
    -----
    The difference field Δu = u(t) - u(t-1) shows regions of acceleration
    (positive) and deceleration (negative). Optical flow applied to this
    field tracks how these patterns of change move through space.

    Smoothing helps reduce noise but too much will blur ramp edges.
    Edge masking removes regions with weak gradients where optical flow
    estimates are unreliable.
    """
    if u_wind.shape[0] != 2 or v_wind.shape[0] != 2:
        raise ValueError("u_wind and v_wind must have shape (2, ny, nx)")

    # Compute temporal differences
    delta_u = u_wind[1] - u_wind[0]  # Δu(t) = u(t) - u(t-1)
    delta_v = v_wind[1] - v_wind[0]  # Δv(t) = v(t) - v(t-1)

    # Compute magnitude of difference field (for masking)
    delta_magnitude = np.sqrt(delta_u**2 + delta_v**2)

    # Apply spatial smoothing to reduce noise
    if smooth_sigma > 0:
        delta_u_smooth = gaussian_filter(delta_u, sigma=smooth_sigma)
        delta_v_smooth = gaussian_filter(delta_v, sigma=smooth_sigma)
    else:
        delta_u_smooth = delta_u
        delta_v_smooth = delta_v

    # Create a combined intensity field from the difference vectors
    # Use the magnitude of the vector difference as the "intensity"
    intensity_field = np.sqrt(delta_u_smooth**2 + delta_v_smooth**2)

    # For optical flow, we need two consecutive time steps of this field
    # Since we only have one difference, we'll use the current wind field
    # and the difference field to simulate motion
    field1 = intensity_field  # Current difference pattern
    field2 = np.sqrt(u_wind[1]**2 + v_wind[1]**2)  # Current wind speed

    # Apply optical flow method
    u_adv, v_adv = compute_advection_velocity_optical_flow(
        field1, field2, dx, dy, dt
    )

    # Create edge mask based on gradient magnitude
    # Mask out regions where the difference field is too weak
    gradient_mag = np.sqrt(
        np.gradient(delta_u_smooth, dy, dx)[0]**2 +
        np.gradient(delta_u_smooth, dy, dx)[1]**2 +
        np.gradient(delta_v_smooth, dy, dx)[0]**2 +
        np.gradient(delta_v_smooth, dy, dx)[1]**2
    )

    # Normalize gradient magnitude
    if gradient_mag.max() > 0:
        gradient_mag_norm = gradient_mag / gradient_mag.max()
    else:
        gradient_mag_norm = gradient_mag

    # Create mask: True where gradients are strong enough
    mask = gradient_mag_norm > edge_threshold

    # Also mask regions near domain boundaries (5 pixels)
    mask[:5, :] = False
    mask[-5:, :] = False
    mask[:, :5] = False
    mask[:, -5:] = False

    # Apply mask to advection velocities
    u_advection = np.where(mask, u_adv, np.nan)
    v_advection = np.where(mask, v_adv, np.nan)

    return u_advection, v_advection, mask


def compute_advection_grid_from_era5(
    era5_file: str,
    output_file: str,
    method: str = 'crosscorr',
    time_step_hours: int = 1,
    era5_pressure_file: Optional[str] = None,
    pressure_level: int = 850
) -> xr.Dataset:
    """
    Compute advection velocity grid from ERA5 wind data.

    This is the main function that:
    1. Loads ERA5 wind data
    2. Computes wind speed at each grid point and time
    3. Tracks advection to create advection velocity grid

    Parameters
    ----------
    era5_file : str
        Path to ERA5 NetCDF file with wind data (single-level)
    output_file : str
        Path to save the output NetCDF file with advection velocities
    method : str, optional
        Method to compute advection:
        - 'crosscorr': Cross-correlation based (more accurate but slower)
        - 'optical_flow': Optical flow on wind speed field (faster)
        - 'temporal_difference': Optical flow on Δu(t) = u(t) - u(t-1)
        - '850hpa': Use 850 hPa wind as advection (requires era5_pressure_file)
        Default is 'crosscorr'
    time_step_hours : int, optional
        Time step in hours between consecutive fields for advection computation
        Default is 1 hour
    era5_pressure_file : str, optional
        Path to ERA5 pressure-level NetCDF file (required for '850hpa' method)
    pressure_level : int, optional
        Pressure level in hPa to use for advection (default: 850)

    Returns
    -------
    xr.Dataset
        Dataset containing advection velocities with coordinates

    Notes
    -----
    The output contains:
    - u_advection: Eastward advection velocity (m/s)
    - v_advection: Northward advection velocity (m/s)
    - advection_speed: Magnitude of advection velocity (m/s)
    - Also preserves original wind data for reference

    For 'temporal_difference' method:
    - Also includes 'advection_mask' showing valid regions

    For '850hpa' method:
    - Directly uses the wind at specified pressure level as advection
    """

    # Special handling for 850 hPa method
    if method == '850hpa':
        if era5_pressure_file is None:
            raise ValueError("era5_pressure_file must be provided for '850hpa' method")

        print(f"Using {pressure_level} hPa wind as advection velocity...")
        u_adv, v_adv = extract_850hpa_wind_advection(era5_pressure_file, pressure_level)

        # Calculate advection speed
        advection_speed = np.sqrt(u_adv**2 + v_adv**2)

        # Create output dataset
        output_ds = xr.Dataset(
            {
                'u_advection': u_adv,
                'v_advection': v_adv,
                'advection_speed': advection_speed,
            }
        )

        # Add attributes
        output_ds['u_advection'].attrs = {
            'long_name': f'{pressure_level} hPa eastward wind',
            'units': 'm/s',
            'description': f'U-component of wind at {pressure_level} hPa used as advection velocity'
        }
        output_ds['v_advection'].attrs = {
            'long_name': f'{pressure_level} hPa northward wind',
            'units': 'm/s',
            'description': f'V-component of wind at {pressure_level} hPa used as advection velocity'
        }
        output_ds['advection_speed'].attrs = {
            'long_name': f'{pressure_level} hPa wind speed',
            'units': 'm/s',
            'description': f'Wind speed at {pressure_level} hPa used as advection speed'
        }

        output_ds.attrs = {
            'title': f'ERA5 {pressure_level} hPa Wind as Advection Velocity',
            'method': method,
            'pressure_level': pressure_level,
            'created': datetime.now().isoformat(),
            'source_file': era5_pressure_file
        }

        # Save to file
        print(f"\nSaving results to {output_file}...")
        output_ds.to_netcdf(output_file)
        print(f"Advection speed statistics: mean={float(advection_speed.mean()):.2f} m/s, "
              f"max={float(advection_speed.max()):.2f} m/s")
        print("Done!")

        return output_ds

    print(f"Loading ERA5 data from {era5_file}...")
    ds = xr.open_dataset(era5_file)

    # Identify wind component variables (handle different naming conventions)
    u_var = None
    v_var = None

    for var in ds.variables:
        if 'u10' in var or ('u' in var and 'wind' in var):
            u_var = var
        if 'v10' in var or ('v' in var and 'wind' in var):
            v_var = var

    if u_var is None or v_var is None:
        raise ValueError(f"Could not find wind components in dataset. Variables: {list(ds.variables)}")

    print(f"Found wind components: {u_var}, {v_var}")

    # Get wind components
    u_wind = ds[u_var]
    v_wind = ds[v_var]

    # Compute wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)

    # Get grid information
    if 'latitude' in ds.coords:
        lat = ds['latitude'].values
        lon = ds['longitude'].values
    elif 'lat' in ds.coords:
        lat = ds['lat'].values
        lon = ds['lon'].values
    else:
        raise ValueError("Could not find latitude/longitude coordinates")

    time = ds['time'].values

    # Compute grid spacing
    # For lat/lon grids, spacing in degrees
    dy = abs(np.mean(np.diff(lat)))
    dx = abs(np.mean(np.diff(lon)))

    # Convert to meters approximately (at mid-latitude)
    mid_lat = np.mean(lat)
    meters_per_degree_lat = 111000  # approximately
    meters_per_degree_lon = 111000 * np.cos(np.radians(mid_lat))

    dy_meters = dy * meters_per_degree_lat
    dx_meters = dx * meters_per_degree_lon

    print(f"Grid spacing: {dx:.3f}° lon ({dx_meters:.0f} m), {dy:.3f}° lat ({dy_meters:.0f} m)")

    # Compute time step in seconds
    dt_seconds = time_step_hours * 3600

    # Initialize arrays for advection velocity
    n_times = len(time)
    n_lat = len(lat)
    n_lon = len(lon)

    u_advection_all = np.zeros((n_times - 1, n_lat, n_lon))
    v_advection_all = np.zeros((n_times - 1, n_lat, n_lon))

    # For temporal_difference method, also store masks
    if method == 'temporal_difference':
        mask_all = np.zeros((n_times - 1, n_lat, n_lon), dtype=bool)

    print(f"\nComputing advection velocity using {method} method...")
    print(f"Processing {n_times - 1} time step pairs...")

    # Compute advection velocity for each consecutive time pair
    for t in range(n_times - 1):
        if t % 10 == 0:
            print(f"Processing time step {t+1}/{n_times-1}...")

        if method == 'temporal_difference':
            # For this method, we need the actual u and v components
            u_t0 = u_wind.isel(time=t).values
            u_t1 = u_wind.isel(time=t+1).values
            v_t0 = v_wind.isel(time=t).values
            v_t1 = v_wind.isel(time=t+1).values

            # Stack into arrays
            u_pair = np.array([u_t0, u_t1])
            v_pair = np.array([v_t0, v_t1])

            u_adv, v_adv, mask = compute_advection_velocity_temporal_difference(
                u_pair, v_pair, dx_meters, dy_meters, dt_seconds
            )
            mask_all[t, :, :] = mask
        else:
            # For other methods, use wind speed
            field1 = wind_speed.isel(time=t).values
            field2 = wind_speed.isel(time=t+1).values

            if method == 'crosscorr':
                u_adv, v_adv = compute_advection_velocity_crosscorr(
                    field1, field2, dx_meters, dy_meters, dt_seconds
                )
            elif method == 'optical_flow':
                u_adv, v_adv = compute_advection_velocity_optical_flow(
                    field1, field2, dx_meters, dy_meters, dt_seconds
                )
            else:
                raise ValueError(f"Unknown method: {method}. "
                               f"Valid methods: 'crosscorr', 'optical_flow', 'temporal_difference', '850hpa'")

        u_advection_all[t, :, :] = u_adv
        v_advection_all[t, :, :] = v_adv

    # Compute advection speed magnitude
    advection_speed = np.sqrt(u_advection_all**2 + v_advection_all**2)

    print(f"\nAdvection velocity statistics:")
    print(f"  U-component: mean={np.nanmean(u_advection_all):.2f} m/s, "
          f"std={np.nanstd(u_advection_all):.2f} m/s")
    print(f"  V-component: mean={np.nanmean(v_advection_all):.2f} m/s, "
          f"std={np.nanstd(v_advection_all):.2f} m/s")
    print(f"  Speed: mean={np.nanmean(advection_speed):.2f} m/s, "
          f"max={np.nanmax(advection_speed):.2f} m/s")

    # Create output dataset
    time_advection = time[:-1]  # One less time step

    data_vars = {
        'u_advection': (['time', 'latitude', 'longitude'], u_advection_all),
        'v_advection': (['time', 'latitude', 'longitude'], v_advection_all),
        'advection_speed': (['time', 'latitude', 'longitude'], advection_speed),
        'wind_speed': (['time', 'latitude', 'longitude'],
                      wind_speed.isel(time=slice(0, -1)).values),
        'u_wind': (['time', 'latitude', 'longitude'],
                  u_wind.isel(time=slice(0, -1)).values),
        'v_wind': (['time', 'latitude', 'longitude'],
                  v_wind.isel(time=slice(0, -1)).values),
    }

    # Add mask for temporal_difference method
    if method == 'temporal_difference':
        data_vars['advection_mask'] = (['time', 'latitude', 'longitude'], mask_all)

    output_ds = xr.Dataset(
        data_vars,
        coords={
            'time': time_advection,
            'latitude': lat,
            'longitude': lon,
        }
    )

    # Add attributes
    output_ds['u_advection'].attrs = {
        'long_name': 'Eastward advection velocity',
        'units': 'm/s',
        'description': 'Velocity of wind pattern movement in eastward direction'
    }
    output_ds['v_advection'].attrs = {
        'long_name': 'Northward advection velocity',
        'units': 'm/s',
        'description': 'Velocity of wind pattern movement in northward direction'
    }
    output_ds['advection_speed'].attrs = {
        'long_name': 'Advection velocity magnitude',
        'units': 'm/s',
        'description': 'Speed of wind pattern movement'
    }

    if method == 'temporal_difference':
        output_ds['advection_mask'].attrs = {
            'long_name': 'Valid advection region mask',
            'description': 'Boolean mask indicating regions with reliable advection estimates (True=valid)'
        }

    output_ds.attrs = {
        'title': 'ERA5 Wind Advection Velocity',
        'method': method,
        'time_step_hours': time_step_hours,
        'created': datetime.now().isoformat(),
        'source_file': era5_file
    }

    # Save to file
    print(f"\nSaving results to {output_file}...")
    output_ds.to_netcdf(output_file)
    print("Done!")

    return output_ds


if __name__ == '__main__':
    # Example usage
    print("ERA5 Wind Advection Module")
    print("This module provides functions to download ERA5 wind data and compute advection velocities.")
    print("\nSee example_usage.py for usage examples.")
