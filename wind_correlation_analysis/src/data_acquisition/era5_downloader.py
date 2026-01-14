"""
ERA5 data downloader for geostrophic wind calculation
Uses Copernicus Climate Data Store (CDS) API
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERA5Downloader:
    """Download and process ERA5 reanalysis data"""

    def __init__(self, output_dir: str = "data/raw/era5"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_era5_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Dict[str, float],
        variables: list = None,
        pressure_levels: list = None
    ) -> str:
        """
        Download ERA5 data from CDS

        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
        bbox : Dict[str, float]
            Bounding box with 'north', 'south', 'east', 'west'
        variables : list
            List of variables to download
        pressure_levels : list
            Pressure levels in hPa

        Returns:
        --------
        str
            Path to downloaded file
        """
        try:
            import cdsapi
        except ImportError:
            logger.error("cdsapi not installed. Install with: pip install cdsapi")
            logger.info("You also need to set up CDS API key: https://cds.climate.copernicus.eu/api-how-to")
            raise

        if variables is None:
            variables = [
                'geopotential',
                'u_component_of_wind',
                'v_component_of_wind',
            ]

        if pressure_levels is None:
            pressure_levels = ['850', '925', '1000']

        # Generate list of dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]

        # Build request
        request = {
            'product_type': 'reanalysis',
            'variable': variables,
            'pressure_level': pressure_levels,
            'year': list(set([d.year for d in dates])),
            'month': list(set([d.month for d in dates])),
            'day': list(set([d.day for d in dates])),
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': [bbox['north'], bbox['west'], bbox['south'], bbox['east']],
            'format': 'netcdf',
        }

        output_file = self.output_dir / f"era5_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"

        c = cdsapi.Client()

        logger.info(f"Downloading ERA5 data to {output_file}")
        logger.info("This may take several minutes...")

        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                request,
                str(output_file)
            )
            logger.info(f"Download complete: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error downloading ERA5 data: {e}")
            raise

    def calculate_geostrophic_wind(
        self,
        geopotential: xr.DataArray,
        latitude: np.ndarray,
        longitude: np.ndarray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate geostrophic wind from geopotential field

        Parameters:
        -----------
        geopotential : xr.DataArray
            Geopotential field in m²/s²
        latitude : np.ndarray
            Latitude coordinates
        longitude : np.ndarray
            Longitude coordinates

        Returns:
        --------
        Tuple[xr.DataArray, xr.DataArray]
            u and v components of geostrophic wind in m/s
        """
        # Constants
        omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        R_earth = 6.371e6  # Earth's radius (m)

        # Calculate Coriolis parameter
        f = 2 * omega * np.sin(np.deg2rad(latitude))

        # Convert geopotential to geopotential height
        g = 9.80665  # Standard gravity (m/s²)
        Z = geopotential / g

        # Calculate gradients
        # dZ/dy (northward gradient)
        dy = R_earth * np.deg2rad(np.gradient(latitude))
        dZ_dy = np.gradient(Z, axis=-2) / dy[:, np.newaxis]

        # dZ/dx (eastward gradient)
        dx = R_earth * np.deg2rad(np.gradient(longitude)) * np.cos(np.deg2rad(latitude))[:, np.newaxis]
        dZ_dx = np.gradient(Z, axis=-1) / dx

        # Geostrophic wind components
        # u_g = -(g/f) * dZ/dy
        # v_g = (g/f) * dZ/dx
        u_g = -(g / f[:, np.newaxis]) * dZ_dy
        v_g = (g / f[:, np.newaxis]) * dZ_dx

        return u_g, v_g

    def process_era5_file(
        self,
        filepath: str,
        pressure_level: int = 850
    ) -> xr.Dataset:
        """
        Process ERA5 file and calculate geostrophic wind

        Parameters:
        -----------
        filepath : str
            Path to ERA5 netCDF file
        pressure_level : int
            Pressure level in hPa for geostrophic wind calculation

        Returns:
        --------
        xr.Dataset
            Processed dataset with geostrophic wind
        """
        logger.info(f"Processing ERA5 file: {filepath}")

        # Load data
        ds = xr.open_dataset(filepath)

        # Select pressure level
        if 'level' in ds.dims:
            ds = ds.sel(level=pressure_level)

        # Calculate geostrophic wind from geopotential if available
        if 'z' in ds.variables:
            logger.info("Calculating geostrophic wind from geopotential")

            lat = ds.latitude.values
            lon = ds.longitude.values

            # Initialize arrays for geostrophic wind
            u_g_list = []
            v_g_list = []

            for t in range(len(ds.time)):
                Z = ds.z.isel(time=t).values
                u_g, v_g = self.calculate_geostrophic_wind(Z, lat, lon)
                u_g_list.append(u_g)
                v_g_list.append(v_g)

            # Add to dataset
            ds['u_geostrophic'] = (['time', 'latitude', 'longitude'], np.array(u_g_list))
            ds['v_geostrophic'] = (['time', 'latitude', 'longitude'], np.array(v_g_list))

            # Calculate geostrophic wind speed and direction
            ds['geostrophic_speed'] = np.sqrt(ds['u_geostrophic']**2 + ds['v_geostrophic']**2)
            ds['geostrophic_direction'] = (np.arctan2(ds['u_geostrophic'], ds['v_geostrophic']) * 180 / np.pi + 180) % 360

        # Save processed file
        output_file = Path(filepath).parent / f"{Path(filepath).stem}_processed.nc"
        ds.to_netcdf(output_file)
        logger.info(f"Saved processed data to {output_file}")

        return ds

    def extract_station_timeseries(
        self,
        ds: xr.Dataset,
        lat: float,
        lon: float,
        method: str = 'nearest'
    ) -> pd.DataFrame:
        """
        Extract time series at a specific location

        Parameters:
        -----------
        ds : xr.Dataset
            ERA5 dataset
        lat : float
            Latitude of location
        lon : float
            Longitude of location
        method : str
            Interpolation method ('nearest' or 'linear')

        Returns:
        --------
        pd.DataFrame
            Time series at the location
        """
        # Select nearest point or interpolate
        ds_point = ds.sel(latitude=lat, longitude=lon, method=method)

        # Convert to dataframe
        df = ds_point.to_dataframe()

        # Reset index to make time a column
        df = df.reset_index()

        # Keep only relevant columns
        cols = ['time']
        if 'geostrophic_speed' in df.columns:
            cols.extend(['geostrophic_speed', 'geostrophic_direction'])
        if 'u' in df.columns:
            cols.extend(['u', 'v'])
        if 'u_geostrophic' in df.columns:
            cols.extend(['u_geostrophic', 'v_geostrophic'])

        df = df[cols]
        df = df.rename(columns={'time': 'timestamp'})
        df = df.set_index('timestamp')

        return df


class AdvectionSpeedProxy:
    """
    Calculate advection speed proxy from various sources
    """

    @staticmethod
    def from_era5_wind_speed(
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        height_factor: float = 1.5
    ) -> np.ndarray:
        """
        Use ERA5 wind speed as advection speed proxy

        A typical approach is to use 850 hPa wind speed multiplied by a factor
        to account for the wind speed at turbine hub height (~100m)

        Parameters:
        -----------
        u_wind : np.ndarray
            U component of wind
        v_wind : np.ndarray
            V component of wind
        height_factor : float
            Scaling factor for height adjustment (default 1.5)

        Returns:
        --------
        np.ndarray
            Advection speed in m/s
        """
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        return wind_speed * height_factor

    @staticmethod
    def from_geostrophic_wind(
        u_geostrophic: np.ndarray,
        v_geostrophic: np.ndarray,
        scaling_factor: float = 0.7
    ) -> np.ndarray:
        """
        Use geostrophic wind speed as advection speed proxy

        Geostrophic wind is typically faster than surface wind,
        so we scale it down

        Parameters:
        -----------
        u_geostrophic : np.ndarray
            U component of geostrophic wind
        v_geostrophic : np.ndarray
            V component of geostrophic wind
        scaling_factor : float
            Scaling factor (default 0.7)

        Returns:
        --------
        np.ndarray
            Advection speed in m/s
        """
        geostrophic_speed = np.sqrt(u_geostrophic**2 + v_geostrophic**2)
        return geostrophic_speed * scaling_factor

    @staticmethod
    def from_gradient_wind(
        pressure_field: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray
    ) -> np.ndarray:
        """
        Calculate advection speed from pressure gradient

        This is a more sophisticated approach that accounts for
        curvature effects

        Parameters:
        -----------
        pressure_field : np.ndarray
            Surface pressure field
        lat : np.ndarray
            Latitude array
        lon : np.ndarray
            Longitude array

        Returns:
        --------
        np.ndarray
            Advection speed in m/s
        """
        # This is a placeholder for more sophisticated calculation
        # Would require pressure field and radius of curvature
        pass


if __name__ == "__main__":
    # Example usage
    downloader = ERA5Downloader()

    # Note: This requires CDS API credentials
    # Set up at: https://cds.climate.copernicus.eu/api-how-to

    print("ERA5 Downloader initialized")
    print("To use this, you need to:")
    print("1. Create account at https://cds.climate.copernicus.eu/")
    print("2. Install cdsapi: pip install cdsapi")
    print("3. Set up ~/.cdsapirc with your credentials")
