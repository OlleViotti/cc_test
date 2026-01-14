#!/usr/bin/env python3
"""
Main script for downloading ERA5 data.

This script provides a command-line interface for downloading ERA5 reanalysis data
from the Copernicus Climate Data Store.
"""

import argparse
import json
import sys
from datetime import datetime
from era5_downloader import ERA5Downloader, download_from_config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download ERA5 reanalysis data from Copernicus Climate Data Store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using a configuration file
  python download_era5.py --config config_example.json

  # Download wind data for a specific period
  python download_era5.py --preset wind --output wind_2023.nc --year 2023 --month 01 --area 60,-10,50,2

  # Download temperature data
  python download_era5.py --preset temperature --output temp_2023.nc --year 2023 --month 01,02,03

Common ERA5 variables:
  - 10m_u_component_of_wind, 10m_v_component_of_wind
  - 100m_u_component_of_wind, 100m_v_component_of_wind
  - 2m_temperature, 2m_dewpoint_temperature
  - surface_pressure, mean_sea_level_pressure
  - total_precipitation, total_cloud_cover
  - soil_temperature_level_1, soil_temperature_level_2
        """
    )

    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--cdsapirc', type=str, default='.cdsapirc',
                        help='Path to .cdsapirc file (default: .cdsapirc)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--preset', type=str, choices=['wind', 'temperature', 'pressure'],
                        help='Use a preset configuration')
    parser.add_argument('--variables', type=str, nargs='+',
                        help='Variables to download (space-separated)')
    parser.add_argument('--year', type=str, nargs='+', help='Years (e.g., 2023 or 2020 2021 2022)')
    parser.add_argument('--month', type=str, help='Months (comma-separated, e.g., 01,02,03)')
    parser.add_argument('--day', type=str, help='Days (comma-separated, e.g., 01,02,03)')
    parser.add_argument('--time', type=str, help='Times (comma-separated, e.g., 00:00,12:00)')
    parser.add_argument('--area', type=str,
                        help='Bounding box as N,W,S,E (e.g., 60,-10,50,2)')
    parser.add_argument('--pressure-levels', type=str,
                        help='Pressure levels (comma-separated, e.g., 500,850,1000)')
    parser.add_argument('--format', type=str, choices=['netcdf', 'grib'], default='netcdf',
                        help='Output format (default: netcdf)')

    return parser.parse_args()


def get_all_days():
    """Return all days of the month."""
    return [f"{d:02d}" for d in range(1, 32)]


def get_all_times():
    """Return all hours of the day."""
    return [f"{h:02d}:00" for h in range(24)]


def build_config_from_args(args):
    """Build configuration dictionary from command-line arguments."""
    config = {
        'cdsapirc_path': args.cdsapirc,
        'output_path': args.output,
        'format': args.format
    }

    # Handle presets
    if args.preset == 'wind':
        config['variables'] = ['10m_u_component_of_wind', '10m_v_component_of_wind']
    elif args.preset == 'temperature':
        config['variables'] = ['2m_temperature']
    elif args.preset == 'pressure':
        config['variables'] = ['surface_pressure', 'mean_sea_level_pressure']
    elif args.variables:
        config['variables'] = args.variables
    else:
        raise ValueError("Either --preset or --variables must be specified")

    # Years
    if not args.year:
        raise ValueError("--year is required")
    config['years'] = args.year

    # Months
    if args.month:
        config['months'] = args.month.split(',')
    else:
        config['months'] = [f"{m:02d}" for m in range(1, 13)]

    # Days
    if args.day:
        config['days'] = args.day.split(',')
    else:
        config['days'] = get_all_days()

    # Times
    if args.time:
        config['times'] = args.time.split(',')
    else:
        config['times'] = get_all_times()

    # Area
    if args.area:
        try:
            config['area'] = [float(x) for x in args.area.split(',')]
            if len(config['area']) != 4:
                raise ValueError("Area must have exactly 4 values: N,W,S,E")
        except ValueError as e:
            raise ValueError(f"Invalid area format: {e}")

    # Pressure levels
    if args.pressure_levels:
        config['pressure_levels'] = args.pressure_levels.split(',')

    return config


def main():
    """Main function."""
    args = parse_arguments()

    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Override with command-line arguments if provided
            if args.output:
                config['output_path'] = args.output
            if args.cdsapirc:
                config['cdsapirc_path'] = args.cdsapirc
        else:
            if not args.output:
                print("Error: --output is required when not using --config")
                sys.exit(1)
            config = build_config_from_args(args)

        # Display configuration
        print("\n" + "="*60)
        print("ERA5 Download Configuration")
        print("="*60)
        print(f"Output path: {config['output_path']}")
        print(f"Variables: {', '.join(config['variables'])}")
        print(f"Years: {', '.join(config['years'])}")
        print(f"Months: {', '.join(config['months'])}")
        print(f"Days: {len(config['days'])} days")
        print(f"Times: {len(config['times'])} time steps per day")
        if 'area' in config:
            print(f"Area (N,W,S,E): {config['area']}")
        else:
            print("Area: Global")
        if 'pressure_levels' in config:
            print(f"Pressure levels: {', '.join(config['pressure_levels'])}")
        print(f"Format: {config.get('format', 'netcdf')}")
        print("="*60 + "\n")

        # Download data
        print(f"Starting download at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_file = download_from_config(config)
        print(f"\nDownload completed successfully!")
        print(f"Data saved to: {output_file}")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
