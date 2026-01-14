"""
ERA5 Data Downloader Module

This module provides functionality to download ERA5 reanalysis data from the
Copernicus Climate Data Store (CDS) using the CDS API.
"""

import cdsapi
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERA5Downloader:
    """
    A class to handle ERA5 data downloads from the Copernicus Climate Data Store.
    """

    def __init__(self, cdsapirc_path: Optional[str] = None):
        """
        Initialize the ERA5 downloader.

        Args:
            cdsapirc_path: Path to .cdsapirc file. If None, uses default location.
        """
        if cdsapirc_path and os.path.exists(cdsapirc_path):
            os.environ['CDSAPI_RC'] = cdsapirc_path
            logger.info(f"Using CDS API config from: {cdsapirc_path}")

        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
            raise

    def download_era5_reanalysis(
        self,
        output_path: str,
        variables: List[str],
        years: List[Union[str, int]],
        months: List[Union[str, int]],
        days: List[Union[str, int]],
        times: List[str],
        area: Optional[List[float]] = None,
        pressure_levels: Optional[List[str]] = None,
        product_type: str = 'reanalysis',
        data_format: str = 'netcdf'
    ) -> str:
        """
        Download ERA5 reanalysis data.

        Args:
            output_path: Path where the downloaded file will be saved
            variables: List of variables to download (e.g., ['10m_u_component_of_wind', '10m_v_component_of_wind'])
            years: List of years (e.g., ['2020', '2021'])
            months: List of months (e.g., ['01', '02', '03'])
            days: List of days (e.g., ['01', '02', ..., '31'])
            times: List of times (e.g., ['00:00', '12:00'])
            area: Bounding box [North, West, South, East] in degrees. If None, downloads global data.
            pressure_levels: List of pressure levels for 3D variables (e.g., ['500', '850'])
            product_type: Type of product ('reanalysis' or 'ensemble_members')
            data_format: Output format ('netcdf' or 'grib')

        Returns:
            Path to the downloaded file
        """
        # Prepare the request
        request = {
            'product_type': product_type,
            'variable': variables,
            'year': [str(y) for y in years],
            'month': [str(m).zfill(2) for m in months],
            'day': [str(d).zfill(2) for d in days],
            'time': times,
            'format': data_format
        }

        # Add area if specified
        if area:
            request['area'] = area

        # Determine dataset based on whether pressure levels are requested
        if pressure_levels:
            dataset = 'reanalysis-era5-pressure-levels'
            request['pressure_level'] = pressure_levels
            logger.info(f"Downloading ERA5 pressure level data for levels: {pressure_levels}")
        else:
            dataset = 'reanalysis-era5-single-levels'
            logger.info("Downloading ERA5 single level data")

        logger.info(f"Variables: {variables}")
        logger.info(f"Time range: {years[0]}-{months[0]}-{days[0]} to {years[-1]}-{months[-1]}-{days[-1]}")
        logger.info(f"Output path: {output_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        try:
            # Submit the request
            logger.info("Submitting request to CDS API...")
            self.client.retrieve(dataset, request, output_path)
            logger.info(f"Download completed successfully: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def download_wind_data(
        self,
        output_path: str,
        years: List[Union[str, int]],
        months: List[Union[str, int]],
        days: List[Union[str, int]],
        times: Optional[List[str]] = None,
        area: Optional[List[float]] = None,
        height: str = '10m'
    ) -> str:
        """
        Convenience method to download wind data (U and V components).

        Args:
            output_path: Path where the downloaded file will be saved
            years: List of years
            months: List of months
            days: List of days
            times: List of times. If None, downloads all hours.
            area: Bounding box [North, West, South, East]. If None, downloads global data.
            height: Wind height ('10m' or '100m')

        Returns:
            Path to the downloaded file
        """
        if times is None:
            times = [f"{h:02d}:00" for h in range(24)]

        if height == '10m':
            variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']
        elif height == '100m':
            variables = ['100m_u_component_of_wind', '100m_v_component_of_wind']
        else:
            raise ValueError(f"Unsupported height: {height}. Use '10m' or '100m'")

        logger.info(f"Downloading {height} wind data (U and V components)")
        return self.download_era5_reanalysis(
            output_path=output_path,
            variables=variables,
            years=years,
            months=months,
            days=days,
            times=times,
            area=area
        )

    def download_temperature_data(
        self,
        output_path: str,
        years: List[Union[str, int]],
        months: List[Union[str, int]],
        days: List[Union[str, int]],
        times: Optional[List[str]] = None,
        area: Optional[List[float]] = None,
        variable: str = '2m_temperature'
    ) -> str:
        """
        Convenience method to download temperature data.

        Args:
            output_path: Path where the downloaded file will be saved
            years: List of years
            months: List of months
            days: List of days
            times: List of times. If None, downloads all hours.
            area: Bounding box [North, West, South, East]. If None, downloads global data.
            variable: Temperature variable ('2m_temperature' or 'skin_temperature')

        Returns:
            Path to the downloaded file
        """
        if times is None:
            times = [f"{h:02d}:00" for h in range(24)]

        logger.info(f"Downloading {variable} data")
        return self.download_era5_reanalysis(
            output_path=output_path,
            variables=[variable],
            years=years,
            months=months,
            days=days,
            times=times,
            area=area
        )


def download_from_config(config: Dict) -> str:
    """
    Download ERA5 data using a configuration dictionary.

    Args:
        config: Dictionary containing download configuration

    Returns:
        Path to the downloaded file
    """
    downloader = ERA5Downloader(cdsapirc_path=config.get('cdsapirc_path'))

    return downloader.download_era5_reanalysis(
        output_path=config['output_path'],
        variables=config['variables'],
        years=config['years'],
        months=config['months'],
        days=config['days'],
        times=config['times'],
        area=config.get('area'),
        pressure_levels=config.get('pressure_levels'),
        product_type=config.get('product_type', 'reanalysis'),
        data_format=config.get('format', 'netcdf')
    )


if __name__ == "__main__":
    # Example usage
    downloader = ERA5Downloader(cdsapirc_path='.cdsapirc')

    # Download 10m wind data for a specific region and time period
    downloader.download_wind_data(
        output_path='output/wind_data.nc',
        years=['2023'],
        months=['01'],
        days=['01', '02', '03'],
        times=['00:00', '06:00', '12:00', '18:00'],
        area=[60, -10, 50, 2],  # Example: UK region [N, W, S, E]
        height='10m'
    )
