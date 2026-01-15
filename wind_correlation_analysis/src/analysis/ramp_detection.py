"""
Wind Ramp Detection Module

This module provides functions to detect wind ramps (rapid changes in wind speed)
in time series data, which are critical events for wind power forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class RampDetector:
    """
    Detect wind ramps in time series data.

    A ramp is defined as a rapid change in wind speed over a specified time window.
    Both positive ramps (increases) and negative ramps (decreases) are detected.

    Default parameters use "research" preset values optimized for comprehensive
    detection of all notable wind speed changes. For operational use cases,
    consider adjusting thresholds:
    - Grid operations: magnitude=4.0, rate=4.0 (detect significant power changes)
    - Short-term forecast: magnitude=2.0, rate=3.0 (detect smaller, faster changes)
    - Offshore: magnitude=5.0, rate=5.0 (higher wind speeds, larger ramps)
    """

    def __init__(
        self,
        time_window_hours: float = 2.0,
        magnitude_threshold_ms: float = 2.5,
        rate_threshold_ms_per_hour: float = 1.5,
        smooth_sigma: float = 1.5
    ):
        """
        Initialize ramp detector.

        Default values use "research" preset for comprehensive ramp detection.

        Parameters:
        -----------
        time_window_hours : float
            Time window for computing wind speed changes (hours).
            Default: 2.0 (research preset, catches slower ramps)
        magnitude_threshold_ms : float
            Minimum change in wind speed to be considered a ramp (m/s).
            Default: 2.5 (research preset, more sensitive)
        rate_threshold_ms_per_hour : float
            Minimum rate of change to be considered a ramp (m/s per hour).
            Default: 1.5 (research preset, catches gradual ramps)
        smooth_sigma : float
            Sigma for Gaussian smoothing of time series (0 = no smoothing).
            Default: 1.5 (research preset, moderate smoothing)
        """
        self.time_window_hours = time_window_hours
        self.magnitude_threshold = magnitude_threshold_ms
        self.rate_threshold = rate_threshold_ms_per_hour
        self.smooth_sigma = smooth_sigma

    def detect_ramps_timeseries(
        self,
        wind_speed: pd.Series,
        return_details: bool = False
    ) -> pd.DataFrame:
        """
        Detect ramps in a wind speed time series.

        Parameters:
        -----------
        wind_speed : pd.Series
            Wind speed time series with DatetimeIndex
        return_details : bool
            If True, return detailed information for each ramp

        Returns:
        --------
        pd.DataFrame
            DataFrame with ramp events, columns:
            - ramp_start: Start time of ramp
            - ramp_end: End time of ramp
            - magnitude: Change in wind speed (m/s)
            - rate: Rate of change (m/s per hour)
            - direction: 'up' or 'down'
            - peak_time: Time of maximum rate of change (if return_details=True)
        """
        # Remove NaN values
        ws = wind_speed.dropna()

        if len(ws) < 10:
            logger.warning("Insufficient data for ramp detection")
            return pd.DataFrame()

        # Apply smoothing if requested
        if self.smooth_sigma > 0:
            ws_smooth = pd.Series(
                gaussian_filter1d(ws.values, sigma=self.smooth_sigma),
                index=ws.index
            )
        else:
            ws_smooth = ws

        # Determine time resolution
        time_diffs = np.diff(ws.index.values).astype('timedelta64[s]').astype(float)
        dt_seconds = np.median(time_diffs)
        dt_hours = dt_seconds / 3600

        # Calculate window size in samples
        window_samples = int(self.time_window_hours / dt_hours)
        if window_samples < 2:
            window_samples = 2

        # Compute rolling change in wind speed
        ws_diff = ws_smooth.diff(window_samples)

        # Convert to rate (m/s per hour)
        ws_rate = ws_diff / (window_samples * dt_hours)

        # Detect ramp candidates
        ramp_candidates = []

        # Positive ramps (wind speed increases)
        positive_mask = (ws_diff >= self.magnitude_threshold) & \
                       (ws_rate >= self.rate_threshold)

        # Negative ramps (wind speed decreases)
        negative_mask = (ws_diff <= -self.magnitude_threshold) & \
                       (ws_rate <= -self.rate_threshold)

        # Find contiguous ramp periods
        for direction, mask in [('up', positive_mask), ('down', negative_mask)]:
            mask_array = mask.values if hasattr(mask, 'values') else np.asarray(mask)

            # Handle case where there are no True values
            if not np.any(mask_array):
                continue

            # Find where mask changes from False to True (ramp start)
            # and from True to False (ramp end)
            mask_diff = np.diff(mask_array.astype(int))
            starts = np.where(mask_diff == 1)[0] + 1  # +1 because diff shifts indices
            ends = np.where(mask_diff == -1)[0] + 1

            # Handle edge cases for mask starting or ending with True
            # If mask starts with True, the first ramp starts at index 0
            if mask_array[0]:
                starts = np.insert(starts, 0, 0)
            # If mask ends with True, the last ramp ends at the final index
            if mask_array[-1]:
                ends = np.append(ends, len(mask_array))

            # Ensure starts and ends are properly paired
            if len(starts) != len(ends):
                logger.warning(f"Mismatched ramp starts ({len(starts)}) and ends ({len(ends)})")
                # Take minimum to avoid index errors
                n_pairs = min(len(starts), len(ends))
                starts = starts[:n_pairs]
                ends = ends[:n_pairs]

            # Process each ramp
            for start_idx, end_idx in zip(starts, ends):
                # Ensure indices are within bounds
                end_idx = min(end_idx, len(ws.index) - 1)
                if start_idx >= len(ws.index) or start_idx >= end_idx:
                    continue

                ramp_start = ws.index[start_idx]
                ramp_end = ws.index[end_idx]

                # Calculate ramp characteristics
                ws_start = ws_smooth.iloc[start_idx]
                ws_end = ws_smooth.iloc[end_idx]
                magnitude = ws_end - ws_start

                # Time duration
                duration_hours = (ramp_end - ramp_start).total_seconds() / 3600
                if duration_hours == 0:
                    continue

                rate = magnitude / duration_hours

                ramp_info = {
                    'ramp_start': ramp_start,
                    'ramp_end': ramp_end,
                    'magnitude': magnitude,
                    'rate': rate,
                    'direction': direction,
                    'duration_hours': duration_hours
                }

                if return_details:
                    # Find peak rate within ramp period
                    ramp_rates = ws_rate.iloc[start_idx:end_idx+1]
                    if len(ramp_rates) > 0:
                        if direction == 'up':
                            peak_idx = ramp_rates.idxmax()
                        else:
                            peak_idx = ramp_rates.idxmin()
                        ramp_info['peak_time'] = peak_idx
                        ramp_info['peak_rate'] = ramp_rates.loc[peak_idx]

                ramp_candidates.append(ramp_info)

        if len(ramp_candidates) == 0:
            return pd.DataFrame()

        # Create DataFrame
        ramps_df = pd.DataFrame(ramp_candidates)

        # Sort by start time
        ramps_df = ramps_df.sort_values('ramp_start').reset_index(drop=True)

        logger.info(f"Detected {len(ramps_df)} ramps: "
                   f"{sum(ramps_df['direction']=='up')} up-ramps, "
                   f"{sum(ramps_df['direction']=='down')} down-ramps")

        return ramps_df

    def detect_ramps_grid(
        self,
        wind_speed_grid: np.ndarray,
        times: np.ndarray,
        return_grid: bool = False
    ) -> Dict:
        """
        Detect ramps in a gridded wind speed dataset.

        Parameters:
        -----------
        wind_speed_grid : np.ndarray
            Wind speed grid with shape (time, lat, lon)
        times : np.ndarray
            Array of datetime values corresponding to time dimension
        return_grid : bool
            If True, return a binary ramp mask grid

        Returns:
        --------
        dict
            Dictionary containing:
            - ramp_mask: Binary mask indicating ramp times (if return_grid=True)
            - ramp_count: Number of ramps at each grid point
            - ramp_stats: Statistics about detected ramps
        """
        n_times, n_lat, n_lon = wind_speed_grid.shape

        # Initialize output
        ramp_count = np.zeros((n_lat, n_lon))

        if return_grid:
            ramp_mask = np.zeros_like(wind_speed_grid, dtype=bool)

        # Detect ramps at each grid point
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series
                ws_point = pd.Series(
                    wind_speed_grid[:, i, j],
                    index=pd.DatetimeIndex(times)
                )

                # Detect ramps
                ramps = self.detect_ramps_timeseries(ws_point, return_details=False)

                ramp_count[i, j] = len(ramps)

                if return_grid and len(ramps) > 0:
                    # Mark ramp times in the mask
                    for _, ramp in ramps.iterrows():
                        mask_times = (times >= ramp['ramp_start']) & \
                                    (times <= ramp['ramp_end'])
                        ramp_mask[mask_times, i, j] = True

        result = {
            'ramp_count': ramp_count,
            'ramp_stats': {
                'total_grid_points': n_lat * n_lon,
                'points_with_ramps': np.sum(ramp_count > 0),
                'max_ramps_per_point': np.max(ramp_count),
                'mean_ramps_per_point': np.mean(ramp_count)
            }
        }

        if return_grid:
            result['ramp_mask'] = ramp_mask

        return result


class RampAdvectionTracker:
    """
    Track the advection of wind ramps across space.

    This class identifies ramp events at multiple locations and estimates
    the advection velocity by tracking ramp arrival times.
    """

    def __init__(self, max_time_lag_hours: float = 6.0):
        """
        Initialize ramp advection tracker.

        Parameters:
        -----------
        max_time_lag_hours : float
            Maximum time lag to consider for ramp matching (hours)
        """
        self.max_time_lag_hours = max_time_lag_hours

    def match_ramps_between_stations(
        self,
        ramps1: pd.DataFrame,
        ramps2: pd.DataFrame,
        distance_km: float,
        bearing_deg: float
    ) -> pd.DataFrame:
        """
        Match ramps between two stations to estimate advection.

        Parameters:
        -----------
        ramps1 : pd.DataFrame
            Ramps detected at station 1
        ramps2 : pd.DataFrame
            Ramps detected at station 2
        distance_km : float
            Distance between stations
        bearing_deg : float
            Bearing from station 1 to station 2

        Returns:
        --------
        pd.DataFrame
            Matched ramps with estimated advection velocities
        """
        matches = []

        max_lag_seconds = self.max_time_lag_hours * 3600

        for _, ramp1 in ramps1.iterrows():
            # Look for matching ramps in station 2
            # Same direction and within time window
            candidates = ramps2[
                (ramps2['direction'] == ramp1['direction']) &
                (ramps2['ramp_start'] >= ramp1['ramp_start']) &
                (ramps2['ramp_start'] <= ramp1['ramp_start'] + pd.Timedelta(seconds=max_lag_seconds))
            ]

            if len(candidates) == 0:
                continue

            # Find best match based on magnitude similarity
            mag_diff = np.abs(candidates['magnitude'] - ramp1['magnitude'])
            best_idx = mag_diff.idxmin()
            ramp2 = candidates.loc[best_idx]

            # Calculate time lag
            time_lag_seconds = (ramp2['ramp_start'] - ramp1['ramp_start']).total_seconds()

            if time_lag_seconds <= 0:
                continue

            # Estimate advection speed
            advection_speed_ms = (distance_km * 1000) / time_lag_seconds

            matches.append({
                'station1_ramp_start': ramp1['ramp_start'],
                'station2_ramp_start': ramp2['ramp_start'],
                'time_lag_hours': time_lag_seconds / 3600,
                'distance_km': distance_km,
                'bearing_deg': bearing_deg,
                'advection_speed_ms': advection_speed_ms,
                'direction': ramp1['direction'],
                'magnitude1': ramp1['magnitude'],
                'magnitude2': ramp2['magnitude']
            })

        if len(matches) == 0:
            return pd.DataFrame()

        return pd.DataFrame(matches)


if __name__ == "__main__":
    print("Wind Ramp Detection Module")
    print("Provides tools for detecting and tracking wind ramps")
