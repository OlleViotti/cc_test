"""
Cross-correlation analysis for wind farm pairs
with spatio-temporal normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import signal
from scipy.interpolate import interp1d
import logging

from .spatial_utils import (
    SpatialCalculations,
    CircularStatistics,
    NormalizationUtils
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossCorrelationAnalysis:
    """
    Perform cross-correlation analysis between wind farm pairs
    """

    def __init__(self, bin_width: float = 30.0, max_lag_hours: float = 24.0):
        """
        Initialize correlation analysis

        Parameters:
        -----------
        bin_width : float
            Width of direction bins in degrees (default 30°)
        max_lag_hours : float
            Maximum time lag to consider in hours (default 24h)
        """
        self.bin_width = bin_width
        self.max_lag_hours = max_lag_hours
        self.n_bins = int(360 / bin_width)

    def calculate_cross_correlation(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        max_lag: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cross-correlation between two signals

        Parameters:
        -----------
        signal1, signal2 : np.ndarray
            Input signals (must be same length)
        max_lag : int, optional
            Maximum lag in number of samples

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            lags (in sample indices), correlation values
        """
        # Normalize signals (zero mean, unit variance)
        s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

        # Calculate cross-correlation
        correlation = signal.correlate(s1, s2, mode='full', method='auto')
        correlation = correlation / len(s1)

        # Calculate lags
        lags = signal.correlation_lags(len(s1), len(s2), mode='full')

        # Limit to max_lag if specified
        if max_lag is not None:
            mask = np.abs(lags) <= max_lag
            lags = lags[mask]
            correlation = correlation[mask]

        return lags, correlation

    def analyze_station_pair(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        era5_data1: pd.DataFrame,
        era5_data2: pd.DataFrame,
        distance_km: float,
        bearing_deg: float,
        station1_id: str,
        station2_id: str
    ) -> pd.DataFrame:
        """
        Analyze cross-correlation for a station pair

        Parameters:
        -----------
        data1, data2 : pd.DataFrame
            Wind speed data for each station (5-min resolution)
            Must have 'wind_speed_ms' column and timestamp index
        era5_data1, era5_data2 : pd.DataFrame
            ERA5 data with geostrophic wind direction and advection speed
            Must have timestamp index
        distance_km : float
            Distance between stations
        bearing_deg : float
            Bearing from station 1 to station 2
        station1_id, station2_id : str
            Station identifiers

        Returns:
        --------
        pd.DataFrame
            Results with columns: bin, bin_center, tau_mean, correlation, lag_samples, lag_hours
        """
        logger.info(f"Analyzing {station1_id} - {station2_id}")

        # Merge datasets
        # First, align wind speed data
        df = pd.merge(
            data1[['wind_speed_ms']].rename(columns={'wind_speed_ms': 'ws1'}),
            data2[['wind_speed_ms']].rename(columns={'wind_speed_ms': 'ws2'}),
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Merge with ERA5 data (use average between two stations)
        era5_merged = pd.merge(
            era5_data1[['geostrophic_direction', 'geostrophic_speed']].add_suffix('_1'),
            era5_data2[['geostrophic_direction', 'geostrophic_speed']].add_suffix('_2'),
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Average geostrophic speed
        era5_merged['advection_speed'] = (
            era5_merged['geostrophic_speed_1'] + era5_merged['geostrophic_speed_2']
        ) / 2

        # Circular mean for direction
        era5_merged['advection_direction'] = era5_merged.apply(
            lambda row: CircularStatistics.circular_mean(
                np.array([row['geostrophic_direction_1'], row['geostrophic_direction_2']])
            ),
            axis=1
        )

        # Resample ERA5 to 5-min (forward fill)
        era5_resampled = era5_merged[['advection_speed', 'advection_direction']].resample('5T').ffill()

        # Merge with wind speed data
        df = pd.merge(df, era5_resampled, left_index=True, right_index=True, how='inner')

        # Remove NaN values
        df = df.dropna()

        if len(df) < 100:
            logger.warning(f"Insufficient data for {station1_id} - {station2_id}: {len(df)} samples")
            return pd.DataFrame()

        # Calculate normalized parameters
        df['tau'] = df['advection_speed'].apply(
            lambda speed: NormalizationUtils.normalize_time_lag(distance_km, speed)
        )

        df['theta'] = df.apply(
            lambda row: NormalizationUtils.normalize_direction(
                row['advection_direction'],
                bearing_deg
            ),
            axis=1
        )

        # Assign direction bins
        df['bin'] = df['theta'].apply(
            lambda theta: NormalizationUtils.assign_direction_bin(theta, self.bin_width)
        )

        # Calculate cross-correlation for each bin
        results = []

        for bin_num in range(self.n_bins):
            df_bin = df[df['bin'] == bin_num]

            if len(df_bin) < 50:  # Minimum samples per bin
                continue

            # Get wind speed signals
            ws1 = df_bin['ws1'].values
            ws2 = df_bin['ws2'].values

            # Calculate max lag in samples (5-min resolution)
            max_lag_samples = int(self.max_lag_hours * 60 / 5)

            # Calculate cross-correlation
            lags, corr = self.calculate_cross_correlation(ws1, ws2, max_lag=max_lag_samples)

            # Find peak correlation
            max_corr_idx = np.argmax(corr)
            max_corr = corr[max_corr_idx]
            lag_at_max = lags[max_corr_idx]

            # Calculate mean tau for this bin
            tau_mean = df_bin['tau'].mean()

            # Bin center angle
            bin_center = NormalizationUtils.get_bin_centers(self.bin_width)[bin_num]

            # Convert lag from samples to hours
            lag_hours = lag_at_max * 5 / 60  # 5-min samples to hours

            results.append({
                'bin': bin_num,
                'bin_center_deg': bin_center,
                'n_samples': len(df_bin),
                'tau_mean_sec': tau_mean,
                'tau_mean_hours': tau_mean / 3600,
                'max_correlation': max_corr,
                'lag_at_max_samples': lag_at_max,
                'lag_at_max_hours': lag_hours,
                'lags': lags,
                'correlation_values': corr
            })

        results_df = pd.DataFrame(results)

        logger.info(f"Completed {station1_id} - {station2_id}: {len(results)} bins with data")

        return results_df

    def analyze_all_pairs(
        self,
        wind_data: Dict[str, pd.DataFrame],
        era5_data: Dict[str, pd.DataFrame],
        station_pairs: pd.DataFrame
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Analyze all station pairs

        Parameters:
        -----------
        wind_data : Dict[str, pd.DataFrame]
            Dictionary of wind speed data for each station
        era5_data : Dict[str, pd.DataFrame]
            Dictionary of ERA5 data for each station
        station_pairs : pd.DataFrame
            DataFrame with station pair information

        Returns:
        --------
        Dict[Tuple[str, str], pd.DataFrame]
            Dictionary mapping (station1_id, station2_id) to results
        """
        results_dict = {}

        for idx, row in station_pairs.iterrows():
            station1_id = row['station1_id']
            station2_id = row['station2_id']

            # Check if data exists for both stations
            if station1_id not in wind_data or station2_id not in wind_data:
                logger.warning(f"Missing wind data for {station1_id} or {station2_id}")
                continue

            if station1_id not in era5_data or station2_id not in era5_data:
                logger.warning(f"Missing ERA5 data for {station1_id} or {station2_id}")
                continue

            # Analyze pair
            results = self.analyze_station_pair(
                data1=wind_data[station1_id],
                data2=wind_data[station2_id],
                era5_data1=era5_data[station1_id],
                era5_data2=era5_data[station2_id],
                distance_km=row['distance_km'],
                bearing_deg=row['bearing_deg'],
                station1_id=station1_id,
                station2_id=station2_id
            )

            if not results.empty:
                results_dict[(station1_id, station2_id)] = results

        return results_dict


class CorrelationMetrics:
    """
    Calculate various correlation metrics
    """

    @staticmethod
    def calculate_peak_correlation(
        correlation: np.ndarray,
        lags: np.ndarray
    ) -> Dict[str, float]:
        """
        Find peak correlation and its characteristics

        Parameters:
        -----------
        correlation : np.ndarray
            Correlation values
        lags : np.ndarray
            Lag values

        Returns:
        --------
        Dict[str, float]
            Dictionary with peak correlation metrics
        """
        max_idx = np.argmax(correlation)

        return {
            'peak_correlation': correlation[max_idx],
            'lag_at_peak': lags[max_idx],
            'peak_sharpness': correlation[max_idx] - np.mean(correlation)
        }

    @staticmethod
    def calculate_correlation_decay(
        correlation: np.ndarray,
        lags: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate correlation decay time (e-folding time)

        Parameters:
        -----------
        correlation : np.ndarray
            Correlation values
        lags : np.ndarray
            Lag values (in samples)
        threshold : float
            Threshold for defining decay (default 0.5)

        Returns:
        --------
        float
            Decay time in samples
        """
        # Find zero-lag correlation
        zero_idx = np.argmin(np.abs(lags))
        corr_zero = correlation[zero_idx]

        # Find where correlation drops below threshold * corr_zero
        target = threshold * corr_zero

        # Search in positive lags
        positive_lags = lags[lags >= 0]
        positive_corr = correlation[lags >= 0]

        decay_idx = np.where(positive_corr < target)[0]

        if len(decay_idx) > 0:
            return positive_lags[decay_idx[0]]
        else:
            return np.nan


if __name__ == "__main__":
    print("Cross-correlation analysis module loaded")
    print(f"Default bin width: 30°")
    print(f"Number of bins: {int(360/30)}")
