# ERA5 Data Downloader

A Python tool for downloading ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).

## Overview

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate. This tool provides a convenient interface to download ERA5 data using the CDS API.

## Features

- Download ERA5 single-level and pressure-level data
- Support for multiple variables, time periods, and geographic regions
- Convenience methods for common data types (wind, temperature, pressure)
- Configuration file support for complex downloads
- Command-line interface for quick downloads
- Comprehensive logging

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. CDS API Credentials

You need to register for a free account at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) and obtain your API key.

The `.cdsapirc` file should be in the project root with the following format:

```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

**Note:** The `.cdsapirc` file is already configured in this project.

## Usage

### Method 1: Using the Command-Line Script

#### Download using a configuration file:

```bash
python download_era5.py --config config_example.json
```

#### Download wind data for a specific period:

```bash
python download_era5.py --preset wind --output wind_2023.nc --year 2023 --month 01 --area 60,-10,50,2
```

#### Download temperature data for multiple months:

```bash
python download_era5.py --preset temperature --output temp_2023_q1.nc --year 2023 --month 01,02,03
```

#### Download custom variables:

```bash
python download_era5.py \
  --output custom_data.nc \
  --variables 10m_u_component_of_wind 10m_v_component_of_wind 2m_temperature \
  --year 2023 \
  --month 01,02 \
  --day 01,15 \
  --time 00:00,12:00 \
  --area 60,-10,50,2
```

### Method 2: Using the Python Module

```python
from era5_downloader import ERA5Downloader

# Initialize the downloader
downloader = ERA5Downloader(cdsapirc_path='.cdsapirc')

# Download wind data
downloader.download_wind_data(
    output_path='output/wind_data.nc',
    years=['2023'],
    months=['01', '02', '03'],
    days=['01', '02', '03'],
    times=['00:00', '06:00', '12:00', '18:00'],
    area=[60, -10, 50, 2],  # [North, West, South, East]
    height='10m'
)

# Download temperature data
downloader.download_temperature_data(
    output_path='output/temperature_data.nc',
    years=['2023'],
    months=['01'],
    days=['01'],
    times=['00:00', '12:00']
)

# Download custom variables
downloader.download_era5_reanalysis(
    output_path='output/custom_data.nc',
    variables=['surface_pressure', 'total_precipitation'],
    years=['2023'],
    months=['01'],
    days=['01', '02', '03'],
    times=['00:00', '06:00', '12:00', '18:00'],
    area=[60, -10, 50, 2]
)
```

### Method 3: Using a Configuration File

Create a JSON configuration file (see `config_example.json`):

```json
{
  "cdsapirc_path": ".cdsapirc",
  "output_path": "output/era5_data.nc",
  "variables": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
  "years": ["2023"],
  "months": ["01", "02"],
  "days": ["01", "02", "03"],
  "times": ["00:00", "06:00", "12:00", "18:00"],
  "area": [60, -10, 50, 2],
  "format": "netcdf"
}
```

Then run:

```python
from era5_downloader import download_from_config
import json

with open('config_example.json', 'r') as f:
    config = json.load(f)

download_from_config(config)
```

## Command-Line Arguments

- `--config`: Path to JSON configuration file
- `--cdsapirc`: Path to .cdsapirc file (default: .cdsapirc)
- `--output`, `-o`: Output file path
- `--preset`: Use a preset configuration (wind, temperature, pressure)
- `--variables`: Variables to download (space-separated)
- `--year`: Years (e.g., 2023 or 2020 2021 2022)
- `--month`: Months (comma-separated, e.g., 01,02,03)
- `--day`: Days (comma-separated, e.g., 01,02,03)
- `--time`: Times (comma-separated, e.g., 00:00,12:00)
- `--area`: Bounding box as N,W,S,E (e.g., 60,-10,50,2)
- `--pressure-levels`: Pressure levels (comma-separated, e.g., 500,850,1000)
- `--format`: Output format (netcdf or grib)

## Common ERA5 Variables

### Wind Variables
- `10m_u_component_of_wind`, `10m_v_component_of_wind`
- `100m_u_component_of_wind`, `100m_v_component_of_wind`

### Temperature Variables
- `2m_temperature`
- `2m_dewpoint_temperature`
- `skin_temperature`

### Pressure Variables
- `surface_pressure`
- `mean_sea_level_pressure`

### Precipitation Variables
- `total_precipitation`
- `convective_precipitation`

### Cloud Variables
- `total_cloud_cover`
- `low_cloud_cover`
- `high_cloud_cover`

### Soil Variables
- `soil_temperature_level_1` through `soil_temperature_level_4`
- `volumetric_soil_water_layer_1` through `volumetric_soil_water_layer_4`

For a complete list of available variables, visit the [ERA5 data documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels).

## Geographic Areas

The `area` parameter defines a bounding box as `[North, West, South, East]` in degrees:

- **Global**: Omit the area parameter
- **Europe**: `[72, -12, 35, 42]`
- **North America**: `[72, -170, 15, -50]`
- **UK**: `[60, -10, 50, 2]`

## Output Format

Downloaded data is saved in NetCDF format by default, which can be opened using:

```python
import xarray as xr

# Open the dataset
ds = xr.open_dataset('output/wind_data.nc')

# View the data
print(ds)

# Access variables
u_wind = ds['u10']  # 10m u-component of wind
v_wind = ds['v10']  # 10m v-component of wind

# Calculate wind speed
wind_speed = (u_wind**2 + v_wind**2)**0.5
```

## Notes

- Downloads can take time depending on the amount of data requested
- The CDS API has request limits and queuing systems
- Large requests may be queued and processed later
- NetCDF files can be large; ensure sufficient disk space

## Troubleshooting

### Authentication Error

If you get an authentication error, check that:
1. Your `.cdsapirc` file is correctly formatted
2. Your API key is valid
3. You've accepted the ERA5 license terms on the CDS website

### Download Fails

- Check your internet connection
- Verify the variables and time periods are valid
- Check CDS service status at https://cds.climate.copernicus.eu/

### Large Downloads

For very large downloads:
- Consider splitting into smaller time periods
- Use spatial subsetting with the `area` parameter
- Reduce temporal resolution by selecting specific times

## License

This tool is provided as-is for downloading ERA5 data. ERA5 data is provided by the Copernicus Climate Change Service and is subject to their terms and conditions.

## Resources

- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [CDS API Documentation](https://cds.climate.copernicus.eu/api-how-to)
