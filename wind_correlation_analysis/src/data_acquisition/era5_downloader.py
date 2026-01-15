"""
Unified ERA5 data downloader for wind analysis
Supports both pressure-level and single-level data downloads
Uses Copernicus Climate Data Store (CDS) API
Includes caching to avoid duplicate downloads
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class ERA5Downloader:
    """
    Unified ERA5 data downloader supporting multiple use cases:
    - Pressure-level data for geostrophic wind calculation
    - Single-level data (100m, 10m) for wind advection analysis

    Features:
    - Intelligent caching to avoid duplicate downloads
    - Support for both pressure-level and single-level datasets
    - Consistent error handling and logging
    """

    def __init__(self, output_dir: str = "data/raw/era5", enable_cache: bool = True):
        """
        Initialize ERA5 downloader

        Parameters:
        -----------
        output_dir : str
            Directory to save downloaded files
        enable_cache : bool
            Whether to enable caching (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cds_client = None
        self.enable_cache = enable_cache

        # Cache metadata file to track downloaded files
        self.cache_metadata_file = self.output_dir / ".era5_cache.json"
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load cache metadata from JSON file"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}. Starting fresh.")
                self.cache_metadata = {}
        else:
            self.cache_metadata = {}

    def _save_cache_metadata(self):
        """Save cache metadata to JSON file"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")

    def _generate_cache_key(self, request_params: Dict) -> str:
        """
        Generate a unique cache key from request parameters

        Parameters:
        -----------
        request_params : dict
            Dictionary of request parameters

        Returns:
        --------
        str
            MD5 hash of the sorted parameters
        """
        # Sort the parameters to ensure consistent hashing
        sorted_params = json.dumps(request_params, sort_keys=True)
        cache_key = hashlib.md5(sorted_params.encode()).hexdigest()
        return cache_key

    def _check_cache(self, cache_key: str, dataset_type: str) -> Optional[str]:
        """
        Check if a file with matching parameters exists in cache

        Parameters:
        -----------
        cache_key : str
            Cache key for the request
        dataset_type : str
            Type of dataset ('pressure' or 'single')

        Returns:
        --------
        str or None
            Path to cached file if exists and valid, None otherwise
        """
        if not self.enable_cache:
            return None

        if cache_key in self.cache_metadata:
            cached_file = Path(self.cache_metadata[cache_key]['file_path'])

            # Check if the file still exists
            if cached_file.exists():
                logger.info(f"✓ Found cached file: {cached_file}")
                logger.info(f"  Dataset: {self.cache_metadata[cache_key].get('dataset', 'unknown')}")
                logger.info(f"  Downloaded: {self.cache_metadata[cache_key].get('download_date', 'unknown')}")
                return str(cached_file)
            else:
                logger.warning(f"Cached file no longer exists: {cached_file}")
                # Remove from cache metadata
                del self.cache_metadata[cache_key]
                self._save_cache_metadata()

        return None

    def _add_to_cache(self, cache_key: str, file_path: str, request_params: Dict, dataset_name: str):
        """
        Add a downloaded file to the cache metadata

        Parameters:
        -----------
        cache_key : str
            Cache key for the request
        file_path : str
            Path to the downloaded file
        request_params : dict
            Request parameters used for download
        dataset_name : str
            Name of the dataset (e.g., 'reanalysis-era5-pressure-levels')
        """
        if not self.enable_cache:
            return

        self.cache_metadata[cache_key] = {
            'file_path': file_path,
            'dataset': dataset_name,
            'download_date': datetime.now().isoformat(),
            'request_params': request_params
        }
        self._save_cache_metadata()
        logger.info(f"✓ Added to cache: {file_path}")

    def _get_cds_client(self):
        """Get or create CDS API client with error handling"""
        if self._cds_client is None:
            try:
                import cdsapi
                self._cds_client = cdsapi.Client()
                logger.info("CDS API client initialized successfully")
            except ImportError:
                logger.error("cdsapi not installed. Install with: pip install cdsapi")
                logger.info("You also need to set up CDS API key: https://cds.climate.copernicus.eu/api-how-to")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize CDS API client: {e}")
                logger.info("Please check your ~/.cdsapirc configuration file")
                raise
        return self._cds_client

    def download_pressure_level_data(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        bbox: Dict[str, float],
        variables: Optional[List[str]] = None,
        pressure_levels: Optional[List[str]] = None,
        time_hours: Optional[List[str]] = None
    ) -> str:
        """
        Download ERA5 pressure-level data (for geostrophic wind calculation)

        Parameters:
        -----------
        start_date : datetime or str
            Start date (datetime or 'YYYY-MM-DD' format)
        end_date : datetime or str
            End date (datetime or 'YYYY-MM-DD' format)
        bbox : Dict[str, float]
            Bounding box with 'north', 'south', 'east', 'west'
        variables : list, optional
            List of variables to download (default: geopotential + wind components)
        pressure_levels : list, optional
            Pressure levels in hPa (default: ['850', '925', '1000'])
        time_hours : list, optional
            List of hours to download (default: all 24 hours)

        Returns:
        --------
        str
            Path to downloaded file
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        if variables is None:
            variables = [
                'geopotential',
                'u_component_of_wind',
                'v_component_of_wind',
            ]

        if pressure_levels is None:
            pressure_levels = ['850', '925', '1000']

        if time_hours is None:
            time_hours = [f'{h:02d}:00' for h in range(24)]

        # Generate list of dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Build request
        request = {
            'product_type': 'reanalysis',
            'variable': variables,
            'pressure_level': pressure_levels,
            'year': [str(y) for y in sorted(set([d.year for d in dates]))],
            'month': [f'{m:02d}' for m in sorted(set([d.month for d in dates]))],
            'day': [f'{d:02d}' for d in sorted(set([d.day for d in dates]))],
            'time': time_hours,
            'area': [bbox['north'], bbox['west'], bbox['south'], bbox['east']],
            'format': 'netcdf',
        }

        # Generate cache key and check cache
        cache_key = self._generate_cache_key(request)
        cached_file = self._check_cache(cache_key, 'pressure')

        if cached_file is not None:
            logger.info("Using cached file instead of downloading")
            return cached_file

        output_file = self.output_dir / f"era5_pressure_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"

        c = self._get_cds_client()

        logger.info(f"Downloading ERA5 pressure-level data to {output_file}")
        logger.info(f"  Dataset: reanalysis-era5-pressure-levels")
        logger.info(f"  Variables: {variables}")
        logger.info(f"  Pressure levels: {pressure_levels}")
        logger.info(f"  Time range: {start_date} to {end_date}")
        logger.info("This may take several minutes...")

        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                request,
                str(output_file)
            )
            logger.info(f"Download complete: {output_file}")

            # Add to cache
            self._add_to_cache(cache_key, str(output_file), request, 'reanalysis-era5-pressure-levels')

            return str(output_file)

        except Exception as e:
            logger.error(f"Error downloading ERA5 pressure-level data: {e}")
            raise

    def download_single_level_data(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        area: Optional[List[float]] = None,
        variables: Optional[List[str]] = None,
        time_hours: Optional[List[str]] = None,
        height: int = 100
    ) -> str:
        """
        Download ERA5 single-level data (for wind advection analysis)

        Parameters:
        -----------
        start_date : datetime or str
            Start date (datetime or 'YYYY-MM-DD' format)
        end_date : datetime or str
            End date (datetime or 'YYYY-MM-DD' format)
        area : list, optional
            Geographic area [North, West, South, East] in degrees (None for global)
        variables : list, optional
            Variables to download (auto-selected based on height if None)
        time_hours : list, optional
            List of hours to download (e.g., ['00:00', '06:00'])
        height : int, optional
            Height above ground in meters (100 or 10, default: 100)

        Returns:
        --------
        str
            Path to downloaded file
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # Auto-select variables based on height
        if variables is None:
            if height == 100:
                variables = [
                    '100m_u_component_of_wind',
                    '100m_v_component_of_wind',
                ]
            elif height == 10:
                variables = [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                ]
            else:
                logger.warning(f"Height {height}m not directly available. Using 100m wind data.")
                variables = [
                    '100m_u_component_of_wind',
                    '100m_v_component_of_wind',
                ]

        if time_hours is None:
            time_hours = [f'{h:02d}:00' for h in range(24)]

        # Prepare request parameters
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'date': f'{start_date.strftime("%Y-%m-%d")}/{end_date.strftime("%Y-%m-%d")}',
            'time': time_hours,
        }

        # Add area if specified
        if area is not None:
            request['area'] = area

        # Generate cache key and check cache
        cache_key = self._generate_cache_key(request)
        cached_file = self._check_cache(cache_key, 'single')

        if cached_file is not None:
            logger.info("Using cached file instead of downloading")
            return cached_file

        output_file = self.output_dir / f"era5_single_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"

        c = self._get_cds_client()

        logger.info(f"Downloading ERA5 single-level data to {output_file}")
        logger.info(f"  Dataset: reanalysis-era5-single-level")
        logger.info(f"  Variables: {variables}")
        logger.info(f"  Height: {height}m")
        logger.info(f"  Time range: {start_date} to {end_date}")
        if area:
            logger.info(f"  Area: {area}")
        else:
            logger.info("  Area: Global")
        logger.info("This may take several minutes...")

        try:
            c.retrieve(
                'reanalysis-era5-single-level',
                request,
                str(output_file)
            )
            logger.info(f"Download complete: {output_file}")

            # Add to cache
            self._add_to_cache(cache_key, str(output_file), request, 'reanalysis-era5-single-level')

            return str(output_file)

        except Exception as e:
            logger.error(f"Error downloading ERA5 single-level data: {e}")
            raise

    def download_era5_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Dict[str, float],
        variables: list = None,
        pressure_levels: list = None
    ) -> str:
        """
        Download ERA5 pressure-level data (legacy method for backward compatibility)

        This method is maintained for backward compatibility.
        For new code, use download_pressure_level_data() or download_single_level_data()
        """
        return self.download_pressure_level_data(
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            variables=variables,
            pressure_levels=pressure_levels
        )

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
