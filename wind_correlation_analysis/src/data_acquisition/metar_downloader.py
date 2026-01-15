"""
METAR data downloader for Swedish stations
Uses Iowa State ASOS/METAR archive
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Dict, Optional
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class METARDownloader:
    """Download and process METAR data from Iowa State archive"""

    BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    KNOTS_TO_MS = 0.514444  # Conversion factor from knots to m/s

    def __init__(self, output_dir: str = "data/raw/metar"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_station_data(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        retry_attempts: int = 3
    ) -> pd.DataFrame:
        """
        Download METAR data for a single station

        Parameters:
        -----------
        station_id : str
            ICAO station identifier (e.g., 'ESSB')
        start_date : datetime
            Start of data period
        end_date : datetime
            End of data period
        retry_attempts : int
            Number of retry attempts for failed downloads

        Returns:
        --------
        pd.DataFrame
            DataFrame with timestamp, wind speed, wind direction
        """
        params = {
            'station': station_id,
            'data': 'all',
            'year1': start_date.year,
            'month1': start_date.month,
            'day1': start_date.day,
            'year2': end_date.year,
            'month2': end_date.month,
            'day2': end_date.day,
            'tz': 'UTC',
            'format': 'comma',
            'latlon': 'yes',
            'elev': 'yes',
            'missing': 'null',
            'trace': '0.0001',
            'direct': 'yes'
        }

        for attempt in range(retry_attempts):
            try:
                logger.info(f"Downloading {station_id} (attempt {attempt + 1}/{retry_attempts})")
                response = requests.get(self.BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                # Parse CSV data
                df = pd.read_csv(StringIO(response.text), parse_dates=['valid'])

                if df.empty:
                    logger.warning(f"No data received for {station_id}")
                    return pd.DataFrame()

                # Rename columns for consistency
                df = df.rename(columns={
                    'valid': 'timestamp',
                    'drct': 'wind_direction',
                    'sknt': 'wind_speed_knots'
                })

                # Convert wind speed from knots to m/s
                df['wind_speed_ms'] = df['wind_speed_knots'] * self.KNOTS_TO_MS

                # Select relevant columns
                cols = ['timestamp', 'wind_speed_ms', 'wind_direction', 'lat', 'lon']
                df = df[cols].copy()

                # Remove missing values
                df = df.dropna(subset=['wind_speed_ms', 'wind_direction'])

                # Set timestamp as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp').sort_index()

                logger.info(f"Downloaded {len(df)} records for {station_id}")
                return df

            except Exception as e:
                logger.error(f"Error downloading {station_id}: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to download {station_id} after {retry_attempts} attempts")
                    return pd.DataFrame()

    def resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to 5-minute averages

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with wind data

        Returns:
        --------
        pd.DataFrame
            Resampled dataframe
        """
        # For wind direction, we need circular mean
        df_resampled = pd.DataFrame()

        # Wind speed - simple mean
        df_resampled['wind_speed_ms'] = df['wind_speed_ms'].resample('5min').mean()

        # Wind direction - circular mean
        # Convert to radians, then to u and v components
        wind_dir_rad = np.radians(df['wind_direction'])
        u = -df['wind_speed_ms'] * np.sin(wind_dir_rad)
        v = -df['wind_speed_ms'] * np.cos(wind_dir_rad)

        # Resample u and v components
        u_mean = u.resample('5min').mean()
        v_mean = v.resample('5min').mean()

        # Convert back to direction
        df_resampled['wind_direction'] = (np.degrees(np.arctan2(u_mean, v_mean)) + 180) % 360

        # Keep lat/lon (first value)
        if 'lat' in df.columns:
            df_resampled['lat'] = df['lat'].resample('5min').first()
            df_resampled['lon'] = df['lon'].resample('5min').first()

        # Remove NaN values
        df_resampled = df_resampled.dropna()

        return df_resampled

    def download_multiple_stations(
        self,
        stations: List[Dict],
        start_date: datetime,
        end_date: datetime,
        resample: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stations

        Parameters:
        -----------
        stations : List[Dict]
            List of station dictionaries with 'id', 'name', 'lat', 'lon'
        start_date : datetime
            Start of data period
        end_date : datetime
            End of data period
        resample : bool
            Whether to resample to 5-minute averages

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping station IDs to dataframes
        """
        results = {}

        for station in stations:
            station_id = station['id']
            df = self.download_station_data(station_id, start_date, end_date)

            if not df.empty:
                if resample:
                    df = self.resample_to_5min(df)

                # Save to file
                output_file = self.output_dir / f"{station_id}_metar.csv"
                df.to_csv(output_file)
                logger.info(f"Saved {station_id} to {output_file}")

                results[station_id] = df

            # Be nice to the server
            time.sleep(1)

        return results


if __name__ == "__main__":
    # Example usage
    downloader = METARDownloader()

    # Download one month of data for testing
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)

    stations = [
        {'id': 'ESSB', 'name': 'Stockholm-Bromma', 'lat': 59.3544, 'lon': 17.9419},
        {'id': 'ESGG', 'name': 'Göteborg-Landvetter', 'lat': 57.6628, 'lon': 12.2798},
    ]

    data = downloader.download_multiple_stations(stations, start, end)

    for station_id, df in data.items():
        print(f"\n{station_id}: {len(df)} records")
        print(df.head())
