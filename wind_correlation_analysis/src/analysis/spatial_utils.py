"""
Spatial utility functions for wind farm correlation analysis
"""

import numpy as np
from typing import Tuple
import pandas as pd


class SpatialCalculations:
    """
    Calculate spatial relationships between locations
    """

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate great circle distance between two points using Haversine formula

        Parameters:
        -----------
        lat1, lon1 : float
            Latitude and longitude of first point (degrees)
        lat2, lon2 : float
            Latitude and longitude of second point (degrees)

        Returns:
        --------
        float
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        distance = SpatialCalculations.EARTH_RADIUS_KM * c

        return distance

    @staticmethod
    def bearing(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate initial bearing from point 1 to point 2

        Parameters:
        -----------
        lat1, lon1 : float
            Latitude and longitude of first point (degrees)
        lat2, lon2 : float
            Latitude and longitude of second point (degrees)

        Returns:
        --------
        float
            Bearing in degrees (0-360, where 0 is North)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        dlon = lon2_rad - lon1_rad

        # Calculate bearing
        x = np.sin(dlon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - \
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        initial_bearing = np.arctan2(x, y)

        # Convert to degrees and normalize to 0-360
        bearing_deg = (np.degrees(initial_bearing) + 360) % 360

        return bearing_deg

    @staticmethod
    def calculate_station_pairs(
        stations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate distance and bearing for all station pairs

        Parameters:
        -----------
        stations : pd.DataFrame
            DataFrame with columns: id, name, lat, lon

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: station1, station2, distance_km, bearing_deg
        """
        pairs = []

        for i, row1 in stations.iterrows():
            for j, row2 in stations.iterrows():
                if i < j:  # Avoid duplicates and self-pairs
                    distance = SpatialCalculations.haversine_distance(
                        row1['lat'], row1['lon'],
                        row2['lat'], row2['lon']
                    )
                    bearing_deg = SpatialCalculations.bearing(
                        row1['lat'], row1['lon'],
                        row2['lat'], row2['lon']
                    )

                    pairs.append({
                        'station1_id': row1['id'],
                        'station1_name': row1['name'],
                        'station1_lat': row1['lat'],
                        'station1_lon': row1['lon'],
                        'station2_id': row2['id'],
                        'station2_name': row2['name'],
                        'station2_lat': row2['lat'],
                        'station2_lon': row2['lon'],
                        'distance_km': distance,
                        'bearing_deg': bearing_deg
                    })

        return pd.DataFrame(pairs)


class CircularStatistics:
    """
    Circular statistics for handling angular data (wind directions)
    """

    @staticmethod
    def circular_mean(angles: np.ndarray, weights: np.ndarray = None) -> float:
        """
        Calculate circular mean of angles

        Parameters:
        -----------
        angles : np.ndarray
            Angles in degrees
        weights : np.ndarray, optional
            Weights for each angle

        Returns:
        --------
        float
            Circular mean in degrees (0-360)
        """
        angles_rad = np.radians(angles)

        if weights is None:
            weights = np.ones_like(angles)

        # Calculate mean of unit vectors
        sin_mean = np.average(np.sin(angles_rad), weights=weights)
        cos_mean = np.average(np.cos(angles_rad), weights=weights)

        mean_angle = np.arctan2(sin_mean, cos_mean)

        # Convert to degrees and normalize
        mean_deg = (np.degrees(mean_angle) + 360) % 360

        return mean_deg

    @staticmethod
    def circular_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """
        Calculate circular difference between angles

        Parameters:
        -----------
        angle1, angle2 : np.ndarray
            Angles in degrees

        Returns:
        --------
        np.ndarray
            Angular difference in degrees (-180 to 180)
        """
        diff = angle1 - angle2

        # Normalize to -180 to 180
        diff = (diff + 180) % 360 - 180

        return diff

    @staticmethod
    def circular_std(angles: np.ndarray) -> float:
        """
        Calculate circular standard deviation

        Parameters:
        -----------
        angles : np.ndarray
            Angles in degrees

        Returns:
        --------
        float
            Circular standard deviation in degrees
        """
        angles_rad = np.radians(angles)

        # Calculate mean resultant length
        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))

        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)

        # Circular standard deviation
        circular_var = 1 - R
        circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else 0

        return np.degrees(circular_std)


class NormalizationUtils:
    """
    Utilities for normalizing time lags and directions
    """

    @staticmethod
    def normalize_time_lag(
        distance_km: float,
        advection_speed_ms: float
    ) -> float:
        """
        Normalize time lag by distance/speed

        tau = distance / speed

        Parameters:
        -----------
        distance_km : float
            Distance between stations in km
        advection_speed_ms : float
            Advection speed in m/s

        Returns:
        --------
        float
            Normalized time lag in seconds
        """
        distance_m = distance_km * 1000
        tau = distance_m / advection_speed_ms if advection_speed_ms > 0 else np.nan

        return tau

    @staticmethod
    def normalize_direction(
        advection_direction: float,
        bearing: float
    ) -> float:
        """
        Normalize direction as circular difference

        theta = advection_direction - bearing

        Parameters:
        -----------
        advection_direction : float
            Wind/advection direction in degrees (0-360)
        bearing : float
            Bearing from station 1 to station 2 in degrees (0-360)

        Returns:
        --------
        float
            Normalized direction in degrees (-180 to 180)
        """
        theta = CircularStatistics.circular_difference(
            advection_direction,
            bearing
        )

        return theta

    @staticmethod
    def assign_direction_bin(
        normalized_direction: float,
        bin_width: float = 30.0
    ) -> int:
        """
        Assign normalized direction to a bin

        Parameters:
        -----------
        normalized_direction : float
            Normalized direction in degrees (-180 to 180)
        bin_width : float
            Width of each bin in degrees (default 30)

        Returns:
        --------
        int
            Bin number (0 to n_bins-1)
        """
        # Shift to 0-360 range
        direction_positive = (normalized_direction + 360) % 360

        # Calculate bin
        bin_number = int(direction_positive / bin_width)

        # Handle edge case
        n_bins = int(360 / bin_width)
        if bin_number >= n_bins:
            bin_number = 0

        return bin_number

    @staticmethod
    def get_bin_centers(bin_width: float = 30.0) -> np.ndarray:
        """
        Get center angles for each direction bin

        Parameters:
        -----------
        bin_width : float
            Width of each bin in degrees

        Returns:
        --------
        np.ndarray
            Array of bin center angles in degrees
        """
        n_bins = int(360 / bin_width)
        bin_centers = np.arange(0, 360, bin_width) + bin_width / 2

        return bin_centers


if __name__ == "__main__":
    # Example usage
    print("Testing spatial calculations...")

    # Example stations
    stations_df = pd.DataFrame([
        {'id': 'ESSB', 'name': 'Stockholm', 'lat': 59.3544, 'lon': 17.9419},
        {'id': 'ESGG', 'name': 'Göteborg', 'lat': 57.6628, 'lon': 12.2798},
        {'id': 'ESMS', 'name': 'Malmö', 'lat': 55.5363, 'lon': 13.3762},
    ])

    # Calculate all pairs
    pairs = SpatialCalculations.calculate_station_pairs(stations_df)
    print("\nStation pairs:")
    print(pairs[['station1_name', 'station2_name', 'distance_km', 'bearing_deg']])

    # Test circular statistics
    print("\n\nTesting circular statistics...")
    angles = np.array([350, 10, 5, 355])
    mean_angle = CircularStatistics.circular_mean(angles)
    print(f"Circular mean of {angles}: {mean_angle:.1f}°")

    # Test normalization
    print("\n\nTesting normalization...")
    distance = 300  # km
    speed = 15  # m/s
    tau = NormalizationUtils.normalize_time_lag(distance, speed)
    print(f"Normalized time lag for {distance}km at {speed}m/s: {tau:.0f} seconds ({tau/3600:.2f} hours)")

    advection_dir = 270  # West
    bearing = 315  # Northwest
    theta = NormalizationUtils.normalize_direction(advection_dir, bearing)
    print(f"Normalized direction (advection={advection_dir}°, bearing={bearing}°): {theta:.1f}°")
