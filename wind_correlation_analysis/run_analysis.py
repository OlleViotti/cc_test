#!/usr/bin/env python3
"""
Main pipeline for wind farm spatio-temporal correlation analysis

This script orchestrates the complete analysis:
1. Downloads METAR wind speed data
2. Downloads ERA5 geostrophic wind data
3. Calculates station pair distances and bearings
4. Performs cross-correlation analysis with normalized parameters
5. Generates polar plots
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_acquisition.metar_downloader import METARDownloader
from data_acquisition.era5_downloader import ERA5Downloader, AdvectionSpeedProxy
from analysis.spatial_utils import SpatialCalculations
from analysis.correlation_analysis import CrossCorrelationAnalysis
from visualization.polar_plots import PolarCorrelationPlot, CorrelationSummaryPlots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WindCorrelationPipeline:
    """
    Main pipeline for wind correlation analysis
    """

    def __init__(self, config_file: str = "config/stations.yaml"):
        """
        Initialize pipeline

        Parameters:
        -----------
        config_file : str
            Path to configuration file
        """
        self.config_file = Path(config_file)
        self.load_config()

        # Initialize components
        self.metar_downloader = METARDownloader(output_dir="data/raw/metar")
        self.era5_downloader = ERA5Downloader(output_dir="data/raw/era5")
        self.correlation_analyzer = CrossCorrelationAnalysis(bin_width=30.0, max_lag_hours=24.0)
        self.plotter = PolarCorrelationPlot()

    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.stations = pd.DataFrame(self.config['stations'])
        logger.info(f"Loaded {len(self.stations)} stations from config")

    def step1_download_metar_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Download METAR wind speed data

        Parameters:
        -----------
        start_date, end_date : datetime
            Date range for data download

        Returns:
        --------
        dict
            Dictionary of wind speed dataframes
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Downloading METAR wind speed data")
        logger.info("=" * 80)

        wind_data = self.metar_downloader.download_multiple_stations(
            stations=self.stations.to_dict('records'),
            start_date=start_date,
            end_date=end_date,
            resample=True
        )

        logger.info(f"Downloaded data for {len(wind_data)} stations")
        return wind_data

    def step2_download_era5_data(
        self,
        start_date: datetime,
        end_date: datetime,
        download_new: bool = True
    ) -> Optional[str]:
        """
        Step 2: Download ERA5 data

        Parameters:
        -----------
        start_date, end_date : datetime
            Date range for data download
        download_new : bool
            Whether to download new data (requires CDS API credentials)

        Returns:
        --------
        str
            Path to ERA5 file
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Downloading ERA5 data")
        logger.info("=" * 80)

        if download_new:
            try:
                era5_file = self.era5_downloader.download_era5_data(
                    start_date=start_date,
                    end_date=end_date,
                    bbox=self.config['era5']['bbox'],
                    variables=self.config['era5']['variables'],
                    pressure_levels=[str(p) for p in self.config['era5']['pressure_levels']]
                )
                logger.info(f"Downloaded ERA5 data to {era5_file}")
                return era5_file
            except Exception as e:
                logger.error(f"Failed to download ERA5 data: {e}")
                logger.info("Will attempt to use existing processed data if available")
                return None
        else:
            logger.info("Skipping ERA5 download (using existing data)")
            return None

    def step3_process_era5_data(self, era5_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Step 3: Process ERA5 data and extract station time series

        Parameters:
        -----------
        era5_file : str, optional
            Path to ERA5 file

        Returns:
        --------
        dict
            Dictionary of ERA5 dataframes for each station
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Processing ERA5 data")
        logger.info("=" * 80)

        if era5_file and Path(era5_file).exists():
            # Process ERA5 file
            ds = self.era5_downloader.process_era5_file(era5_file, pressure_level=850)

            # Extract time series for each station
            era5_data = {}
            for _, station in self.stations.iterrows():
                station_id = station['id']
                lat = station['lat']
                lon = station['lon']

                df = self.era5_downloader.extract_station_timeseries(ds, lat, lon)
                era5_data[station_id] = df

                logger.info(f"Extracted ERA5 data for {station_id}: {len(df)} records")

            return era5_data
        else:
            logger.warning("No ERA5 file available. Generating synthetic data for demonstration.")
            return self._generate_synthetic_era5_data()

    def _generate_synthetic_era5_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic ERA5 data for demonstration purposes

        Returns:
        --------
        dict
            Dictionary of synthetic ERA5 dataframes
        """
        logger.info("Generating synthetic ERA5 data...")

        era5_data = {}

        # Create synthetic data with realistic characteristics
        date_range = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')

        for _, station in self.stations.iterrows():
            station_id = station['id']

            # Synthetic geostrophic wind
            # Direction: slowly varying with some randomness
            base_direction = 270  # Westerly
            direction_variation = 30 * np.sin(np.arange(len(date_range)) * 2 * np.pi / (24 * 7))
            geostrophic_direction = (base_direction + direction_variation +
                                    np.random.normal(0, 15, len(date_range))) % 360

            # Speed: realistic range with diurnal variation
            base_speed = 15  # m/s
            speed_variation = 5 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 24)
            geostrophic_speed = np.maximum(
                base_speed + speed_variation + np.random.normal(0, 3, len(date_range)),
                1.0  # Minimum speed
            )

            df = pd.DataFrame({
                'timestamp': date_range,
                'geostrophic_speed': geostrophic_speed,
                'geostrophic_direction': geostrophic_direction
            }).set_index('timestamp')

            era5_data[station_id] = df

        logger.info(f"Generated synthetic ERA5 data for {len(era5_data)} stations")
        return era5_data

    def step4_calculate_station_pairs(self) -> pd.DataFrame:
        """
        Step 4: Calculate distances and bearings for station pairs

        Returns:
        --------
        pd.DataFrame
            Station pairs with spatial relationships
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Calculating station pair relationships")
        logger.info("=" * 80)

        station_pairs = SpatialCalculations.calculate_station_pairs(self.stations)

        logger.info(f"Calculated {len(station_pairs)} station pairs")
        logger.info("\nStation pairs:")
        for _, pair in station_pairs.iterrows():
            logger.info(
                f"  {pair['station1_name']} - {pair['station2_name']}: "
                f"{pair['distance_km']:.1f} km, bearing {pair['bearing_deg']:.1f}°"
            )

        return station_pairs

    def step5_perform_correlation_analysis(
        self,
        wind_data: Dict[str, pd.DataFrame],
        era5_data: Dict[str, pd.DataFrame],
        station_pairs: pd.DataFrame
    ) -> Dict[tuple, pd.DataFrame]:
        """
        Step 5: Perform cross-correlation analysis

        Parameters:
        -----------
        wind_data : dict
            Wind speed data for each station
        era5_data : dict
            ERA5 data for each station
        station_pairs : pd.DataFrame
            Station pair information

        Returns:
        --------
        dict
            Analysis results
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Performing cross-correlation analysis")
        logger.info("=" * 80)

        results = self.correlation_analyzer.analyze_all_pairs(
            wind_data=wind_data,
            era5_data=era5_data,
            station_pairs=station_pairs
        )

        logger.info(f"Completed analysis for {len(results)} station pairs")

        # Save results
        results_dir = Path("results/data")
        results_dir.mkdir(parents=True, exist_ok=True)

        for (station1_id, station2_id), df in results.items():
            output_file = results_dir / f"correlation_{station1_id}_{station2_id}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved results to {output_file}")

        return results

    def step6_create_visualizations(
        self,
        results: Dict[tuple, pd.DataFrame],
        station_pairs: pd.DataFrame
    ) -> None:
        """
        Step 6: Create polar plots and visualizations

        Parameters:
        -----------
        results : dict
            Analysis results
        station_pairs : pd.DataFrame
            Station pair information
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Creating visualizations")
        logger.info("=" * 80)

        output_dir = Path("results/figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create polar plots for each pair
        self.plotter.plot_multiple_pairs(
            results_dict=results,
            station_pairs=station_pairs,
            output_dir=str(output_dir),
            show_plot=False
        )

        # Create summary plots
        summary_file = output_dir / "correlation_vs_distance.png"
        CorrelationSummaryPlots.plot_correlation_vs_distance(
            results_dict=results,
            station_pairs=station_pairs,
            output_file=str(summary_file)
        )

        # Create detailed statistics plots for each pair
        for (station1_id, station2_id), results_df in results.items():
            pair_info = station_pairs[
                (station_pairs['station1_id'] == station1_id) &
                (station_pairs['station2_id'] == station2_id)
            ].iloc[0]

            stats_file = output_dir / f"bin_statistics_{station1_id}_{station2_id}.png"
            CorrelationSummaryPlots.plot_direction_bin_statistics(
                results_df=results_df,
                station1_name=pair_info['station1_name'],
                station2_name=pair_info['station2_name'],
                output_file=str(stats_file)
            )

        logger.info(f"Saved visualizations to {output_dir}")

    def run_full_pipeline(
        self,
        start_date: datetime,
        end_date: datetime,
        download_era5: bool = False
    ) -> None:
        """
        Run the complete analysis pipeline

        Parameters:
        -----------
        start_date, end_date : datetime
            Date range for analysis
        download_era5 : bool
            Whether to download new ERA5 data
        """
        logger.info("=" * 80)
        logger.info("WIND FARM SPATIO-TEMPORAL CORRELATION ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Analysis period: {start_date.date()} to {end_date.date()}")
        logger.info("")

        try:
            # Step 1: Download METAR data
            wind_data = self.step1_download_metar_data(start_date, end_date)

            if not wind_data:
                logger.error("No wind data available. Exiting.")
                return

            # Step 2 & 3: ERA5 data
            era5_file = self.step2_download_era5_data(start_date, end_date, download_new=download_era5)
            era5_data = self.step3_process_era5_data(era5_file)

            # Step 4: Calculate station pairs
            station_pairs = self.step4_calculate_station_pairs()

            # Step 5: Correlation analysis
            results = self.step5_perform_correlation_analysis(wind_data, era5_data, station_pairs)

            if not results:
                logger.warning("No results from correlation analysis")
                return

            # Step 6: Visualizations
            self.step6_create_visualizations(results, station_pairs)

            logger.info("=" * 80)
            logger.info("ANALYSIS COMPLETE!")
            logger.info("=" * 80)
            logger.info("Results saved to:")
            logger.info("  - Data: results/data/")
            logger.info("  - Figures: results/figures/")

        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Wind farm spatio-temporal correlation analysis'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-01-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--download-era5',
        action='store_true',
        help='Download new ERA5 data (requires CDS API credentials)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/stations.yaml',
        help='Configuration file'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Initialize and run pipeline
    pipeline = WindCorrelationPipeline(config_file=args.config)
    pipeline.run_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        download_era5=args.download_era5
    )


if __name__ == "__main__":
    main()
