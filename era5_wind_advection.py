"""
ERA5 Wind Advection Analysis

This module provides functions to:
1. Download ERA5 wind speed data on a full grid (via unified ERA5Downloader)
2. Compute advection velocity from wind pattern movement
"""

import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import signal
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift

# Add the wind_correlation_analysis package to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wind_correlation_analysis'))

from wind_correlation_analysis.src.data_acquisition.era5_downloader import ERA5Downloader

# =============================================================================
# Constants
# =============================================================================

# Geographic constants
METERS_PER_DEGREE_LAT = 111000  # Approximate meters per degree latitude

# Cross-correlation parameters
CROSSCORR_WINDOW_SIZE = 32  # Default window size for local cross-correlation (grid points)
CROSSCORR_WINDOW_KM = 150  # Physical size of correlation window in km (for dynamic sizing)
CROSSCORR_MIN_WINDOW = 8  # Minimum window size in grid points
CROSSCORR_MAX_WINDOW = 64  # Maximum window size in grid points
CROSSCORR_STD_THRESHOLD = 1e-6  # Minimum std dev to compute correlation

# Optical flow parameters
OPTICAL_FLOW_ALPHA_BASE = 0.1  # Base regularization parameter for optical flow
OPTICAL_FLOW_GRADIENT_EPSILON = 1e-10  # Avoid division by zero
OPTICAL_FLOW_SMOOTH_SIGMA = 2  # Gaussian smoothing sigma for optical flow

# Temporal difference method parameters
BOUNDARY_MASK_PIXELS = 5  # Pixels to mask near domain boundaries


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
    max_displacement: int = 20,
    grid_spacing_km: Optional[float] = None
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
    grid_spacing_km : float, optional
        Grid spacing in kilometers. If provided, window size is dynamically
        calculated to span approximately CROSSCORR_WINDOW_KM (150 km).
        This ensures consistent physical coverage across different grid
        resolutions. If None, uses the default CROSSCORR_WINDOW_SIZE.

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

    # Calculate window size based on physical distance if grid spacing provided
    if grid_spacing_km is not None and grid_spacing_km > 0:
        # Calculate window size to span approximately CROSSCORR_WINDOW_KM
        window_size = int(CROSSCORR_WINDOW_KM / grid_spacing_km)
        # Clamp to reasonable bounds
        window_size = max(CROSSCORR_MIN_WINDOW, min(CROSSCORR_MAX_WINDOW, window_size))
        # Ensure even number for symmetric windows
        window_size = window_size + (window_size % 2)
    else:
        window_size = CROSSCORR_WINDOW_SIZE

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
            if window1.std() < CROSSCORR_STD_THRESHOLD:
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
                        if window2.std() > CROSSCORR_STD_THRESHOLD:
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
    dt: float,
    adaptive_alpha: bool = True
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
    adaptive_alpha : bool, optional
        If True, use adaptive regularization that scales with local gradient
        strength. Strong gradients get less regularization (more trust in
        gradient), weak gradients get more (prevent noise amplification).
        Default is True.

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

    # Compute gradient magnitude squared
    grad_mag_sq = dI_dx**2 + dI_dy**2

    # Lucas-Kanade style solution with regularization
    # We need to solve: dI/dt + u*dI/dx + v*dI/dy = 0
    # This is one equation with two unknowns, so we use regularization

    if adaptive_alpha:
        # Adaptive regularization: scale alpha inversely with gradient strength
        # Strong gradients get less regularization (more trust in gradient)
        # Weak gradients get more regularization (prevent noise amplification)
        valid_grads = grad_mag_sq[grad_mag_sq > OPTICAL_FLOW_GRADIENT_EPSILON]
        if len(valid_grads) > 0:
            grad_percentile_90 = np.percentile(valid_grads, 90)
        else:
            grad_percentile_90 = 1.0

        # Alpha scales up where gradients are weak relative to strong regions
        alpha = OPTICAL_FLOW_ALPHA_BASE * (
            1 + grad_percentile_90 / (grad_mag_sq + OPTICAL_FLOW_GRADIENT_EPSILON)
        )
    else:
        # Fixed regularization
        alpha = OPTICAL_FLOW_ALPHA_BASE

    # Compute advection velocities
    u_advection = -(dI_dt * dI_dx) / (grad_mag_sq + alpha)
    v_advection = -(dI_dt * dI_dy) / (grad_mag_sq + alpha)

    # Smooth the result
    u_advection = gaussian_filter(u_advection, sigma=OPTICAL_FLOW_SMOOTH_SIGMA)
    v_advection = gaussian_filter(v_advection, sigma=OPTICAL_FLOW_SMOOTH_SIGMA)

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
    edge_threshold: float = 0.1,
    ramp_magnitude_threshold: float = 1.0,
    ramp_rate_threshold: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute advection velocity by tracking ramps in temporal difference fields.

    This method:
    1. Computes temporal differences Δu(t) = u(t) - u(t-1) and Δv(t) = v(t) - v(t-1)
    2. Detects ramps (significant changes) in the difference field
    3. Applies smoothing to reduce noise while preserving ramp edges
    4. Applies edge masking to remove unreliable regions
    5. Uses optical flow to track how these ramp patterns propagate through space

    Parameters
    ----------
    u_wind : np.ndarray
        U wind component at three consecutive times, shape (3, ny, nx)
        [u(t-1), u(t), u(t+1)] - need three times to compute two difference fields
    v_wind : np.ndarray
        V wind component at three consecutive times, shape (3, ny, nx)
        [v(t-1), v(t), v(t+1)]
    dx : float
        Grid spacing in x direction (meters)
    dy : float
        Grid spacing in y direction (meters)
    dt : float
        Time difference between consecutive fields (seconds)
    smooth_sigma : float, optional
        Sigma for Gaussian smoothing of difference fields (default: 2.0)
        Higher values = more smoothing but blurred ramp edges
    edge_threshold : float, optional
        Threshold for edge masking based on gradient magnitude (default: 0.1)
        Regions with gradients below this threshold are masked
    ramp_magnitude_threshold : float, optional
        Minimum magnitude of change to be considered a ramp (m/s, default: 1.0)
    ramp_rate_threshold : float, optional
        Minimum rate of change to be considered a ramp (m/s per time_step, default: 1.0)

    Returns
    -------
    u_advection : np.ndarray
        Advection velocity in x direction (m/s)
    v_advection : np.ndarray
        Advection velocity in y direction (m/s)
    mask : np.ndarray
        Boolean mask indicating valid regions (True = valid)
    ramp_mask : np.ndarray
        Boolean mask indicating detected ramp regions (True = ramp present)

    Notes
    -----
    The difference field Δu = u(t) - u(t-1) shows regions of acceleration
    (positive) and deceleration (negative). By detecting ramps in this field
    and applying optical flow, we track how these patterns of change propagate.

    This is more physically meaningful than applying optical flow everywhere,
    as it specifically tracks wind ramp advection.
    """
    if u_wind.shape[0] != 3 or v_wind.shape[0] != 3:
        raise ValueError("u_wind and v_wind must have shape (3, ny, nx) for temporal difference method")

    ny, nx = u_wind.shape[1], u_wind.shape[2]

    # Compute temporal differences at two consecutive time steps
    # This gives us two snapshots of the difference field to track with optical flow
    delta_u_t0 = u_wind[1] - u_wind[0]  # Δu(t) = u(t) - u(t-1)
    delta_v_t0 = v_wind[1] - v_wind[0]  # Δv(t) = v(t) - v(t-1)

    delta_u_t1 = u_wind[2] - u_wind[1]  # Δu(t+1) = u(t+1) - u(t)
    delta_v_t1 = v_wind[2] - v_wind[1]  # Δv(t+1) = v(t+1) - v(t)

    # Compute magnitude of difference fields
    delta_mag_t0 = np.sqrt(delta_u_t0**2 + delta_v_t0**2)
    delta_mag_t1 = np.sqrt(delta_u_t1**2 + delta_v_t1**2)

    # Compute rate of change (magnitude per unit time)
    delta_rate_t0 = delta_mag_t0 / dt
    delta_rate_t1 = delta_mag_t1 / dt

    # STEP 1: Detect ramps in the difference field
    # A ramp is where both magnitude and rate exceed thresholds
    ramp_mask_t0 = (delta_mag_t0 >= ramp_magnitude_threshold) & \
                   (delta_rate_t0 >= ramp_rate_threshold)
    ramp_mask_t1 = (delta_mag_t1 >= ramp_magnitude_threshold) & \
                   (delta_rate_t1 >= ramp_rate_threshold)

    # Combined ramp mask (union of ramps at both times)
    ramp_mask = ramp_mask_t0 | ramp_mask_t1

    # STEP 2: Apply spatial smoothing to the difference fields (only where ramps exist)
    if smooth_sigma > 0:
        # Smooth the entire field first
        delta_mag_t0_smooth = gaussian_filter(delta_mag_t0, sigma=smooth_sigma)
        delta_mag_t1_smooth = gaussian_filter(delta_mag_t1, sigma=smooth_sigma)

        # For direction, smooth u and v components separately
        delta_u_t0_smooth = gaussian_filter(delta_u_t0, sigma=smooth_sigma)
        delta_v_t0_smooth = gaussian_filter(delta_v_t0, sigma=smooth_sigma)
        delta_u_t1_smooth = gaussian_filter(delta_u_t1, sigma=smooth_sigma)
        delta_v_t1_smooth = gaussian_filter(delta_v_t1, sigma=smooth_sigma)
    else:
        delta_mag_t0_smooth = delta_mag_t0
        delta_mag_t1_smooth = delta_mag_t1
        delta_u_t0_smooth = delta_u_t0
        delta_v_t0_smooth = delta_v_t0
        delta_u_t1_smooth = delta_u_t1
        delta_v_t1_smooth = delta_v_t1

    # STEP 3: Create intensity fields for optical flow
    # Use the smoothed magnitude of the difference field
    field_t0 = delta_mag_t0_smooth
    field_t1 = delta_mag_t1_smooth

    # STEP 4: Apply optical flow to track ramp movement
    # This tracks how the difference pattern moves between t0 and t1
    u_adv, v_adv = compute_advection_velocity_optical_flow(
        field_t0, field_t1, dx, dy, dt
    )

    # STEP 5: Create edge mask based on gradient magnitude
    # Mask out regions where the difference field has weak gradients
    gradient_mag = np.sqrt(
        np.gradient(delta_u_t0_smooth, dy, dx)[0]**2 +
        np.gradient(delta_u_t0_smooth, dy, dx)[1]**2 +
        np.gradient(delta_v_t0_smooth, dy, dx)[0]**2 +
        np.gradient(delta_v_t0_smooth, dy, dx)[1]**2
    )

    # Normalize gradient magnitude
    if gradient_mag.max() > 0:
        gradient_mag_norm = gradient_mag / gradient_mag.max()
    else:
        gradient_mag_norm = gradient_mag

    # Create edge mask: True where gradients are strong enough
    edge_mask = gradient_mag_norm > edge_threshold

    # STEP 6: Combine masks
    # Only trust advection estimates where:
    # 1. Ramps are present (ramp_mask)
    # 2. Gradients are strong (edge_mask)
    # 3. Not near domain boundaries

    # Mask regions near domain boundaries
    boundary_mask = np.ones_like(ramp_mask, dtype=bool)
    boundary_mask[:BOUNDARY_MASK_PIXELS, :] = False
    boundary_mask[-BOUNDARY_MASK_PIXELS:, :] = False
    boundary_mask[:, :BOUNDARY_MASK_PIXELS] = False
    boundary_mask[:, -BOUNDARY_MASK_PIXELS:] = False

    # Combined validity mask
    valid_mask = ramp_mask & edge_mask & boundary_mask

    # STEP 7: Apply mask to advection velocities
    # Only report advection where we have valid ramp tracking
    u_advection = np.where(valid_mask, u_adv, np.nan)
    v_advection = np.where(valid_mask, v_adv, np.nan)

    return u_advection, v_advection, valid_mask, ramp_mask


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
    - Also includes 'advection_mask' showing valid regions (ramps + strong gradients)
    - Also includes 'ramp_mask' showing where ramps were detected
    - This method detects ramps in Δu(t)=u(t)-u(t-1), then tracks their propagation
    - Requires 3 consecutive time steps, so output has n_times-2 time steps

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
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(mid_lat))

    dy_meters = dy * METERS_PER_DEGREE_LAT
    dx_meters = dx * meters_per_degree_lon

    print(f"Grid spacing: {dx:.3f}° lon ({dx_meters:.0f} m), {dy:.3f}° lat ({dy_meters:.0f} m)")

    # Compute time step in seconds
    dt_seconds = time_step_hours * 3600

    # Initialize arrays for advection velocity
    n_times = len(time)
    n_lat = len(lat)
    n_lon = len(lon)

    # For temporal_difference method, we need 3 consecutive times, so we get n_times-2 outputs
    # For other methods, we need 2 consecutive times, so we get n_times-1 outputs
    if method == 'temporal_difference':
        n_output_times = n_times - 2
        u_advection_all = np.zeros((n_output_times, n_lat, n_lon))
        v_advection_all = np.zeros((n_output_times, n_lat, n_lon))
        mask_all = np.zeros((n_output_times, n_lat, n_lon), dtype=bool)
        ramp_mask_all = np.zeros((n_output_times, n_lat, n_lon), dtype=bool)
    else:
        n_output_times = n_times - 1
        u_advection_all = np.zeros((n_output_times, n_lat, n_lon))
        v_advection_all = np.zeros((n_output_times, n_lat, n_lon))

    print(f"\nComputing advection velocity using {method} method...")
    print(f"Processing {n_output_times} time step pairs...")

    # Compute advection velocity for each consecutive time pair
    if method == 'temporal_difference':
        # Need 3 consecutive times for this method
        for t in range(n_output_times):
            if t % 10 == 0:
                print(f"Processing time step {t+1}/{n_output_times}...")

            # Extract three consecutive time steps
            u_t0 = u_wind.isel(time=t).values
            u_t1 = u_wind.isel(time=t+1).values
            u_t2 = u_wind.isel(time=t+2).values
            v_t0 = v_wind.isel(time=t).values
            v_t1 = v_wind.isel(time=t+1).values
            v_t2 = v_wind.isel(time=t+2).values

            # Stack into arrays [t, t+1, t+2]
            u_triplet = np.array([u_t0, u_t1, u_t2])
            v_triplet = np.array([v_t0, v_t1, v_t2])

            u_adv, v_adv, mask, ramp_mask = compute_advection_velocity_temporal_difference(
                u_triplet, v_triplet, dx_meters, dy_meters, dt_seconds
            )

            u_advection_all[t, :, :] = u_adv
            v_advection_all[t, :, :] = v_adv
            mask_all[t, :, :] = mask
            ramp_mask_all[t, :, :] = ramp_mask
    else:
        # Standard methods need 2 consecutive times
        for t in range(n_output_times):
            if t % 10 == 0:
                print(f"Processing time step {t+1}/{n_output_times}...")

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
    # For temporal_difference: time dimension is n_times-2 (need 3 consecutive times)
    # For other methods: time dimension is n_times-1 (need 2 consecutive times)
    if method == 'temporal_difference':
        time_advection = time[1:-1]  # Middle times (exclude first and last)
        time_slice_start = 1
        time_slice_end = -1
    else:
        time_advection = time[:-1]  # All times except last
        time_slice_start = 0
        time_slice_end = -1

    data_vars = {
        'u_advection': (['time', 'latitude', 'longitude'], u_advection_all),
        'v_advection': (['time', 'latitude', 'longitude'], v_advection_all),
        'advection_speed': (['time', 'latitude', 'longitude'], advection_speed),
        'wind_speed': (['time', 'latitude', 'longitude'],
                      wind_speed.isel(time=slice(time_slice_start, time_slice_end)).values),
        'u_wind': (['time', 'latitude', 'longitude'],
                  u_wind.isel(time=slice(time_slice_start, time_slice_end)).values),
        'v_wind': (['time', 'latitude', 'longitude'],
                  v_wind.isel(time=slice(time_slice_start, time_slice_end)).values),
    }

    # Add masks for temporal_difference method
    if method == 'temporal_difference':
        data_vars['advection_mask'] = (['time', 'latitude', 'longitude'], mask_all)
        data_vars['ramp_mask'] = (['time', 'latitude', 'longitude'], ramp_mask_all)

        # Print ramp detection statistics
        total_points = mask_all.size
        valid_points = mask_all.sum()
        ramp_points = ramp_mask_all.sum()
        print(f"\nRamp detection statistics:")
        print(f"  Total grid points: {total_points}")
        print(f"  Points with valid advection: {valid_points} ({100*valid_points/total_points:.1f}%)")
        print(f"  Points with detected ramps: {ramp_points} ({100*ramp_points/total_points:.1f}%)")

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
            'description': 'Boolean mask indicating regions with reliable advection estimates (True=valid). '
                          'Combines ramp detection, gradient strength, and boundary masking.'
        }
        output_ds['ramp_mask'].attrs = {
            'long_name': 'Wind ramp detection mask',
            'description': 'Boolean mask indicating regions where wind ramps were detected in the '
                          'temporal difference field (True=ramp present). Ramps are significant changes '
                          'in wind speed exceeding magnitude and rate thresholds.'
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
