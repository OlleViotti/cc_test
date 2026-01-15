"""
Advection Velocity Validation Module

This module provides functions to compare different advection velocity estimation methods
and validate them against observed wind ramp arrival times at measurement stations.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


class AdvectionValidator:
    """
    Compare and validate different advection velocity estimation methods.
    """

    def __init__(self):
        """Initialize advection validator."""
        pass

    def compare_advection_methods(
        self,
        advection_data1: xr.Dataset,
        advection_data2: xr.Dataset,
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        sample_points: Optional[List[Tuple[float, float]]] = None
    ) -> Dict:
        """
        Compare two advection velocity estimation methods.

        Parameters:
        -----------
        advection_data1 : xr.Dataset
            First advection dataset (e.g., 850 hPa wind)
        advection_data2 : xr.Dataset
            Second advection dataset (e.g., temporal difference optical flow)
        method1_name : str
            Name of first method
        method2_name : str
            Name of second method
        sample_points : list of (lat, lon) tuples, optional
            Specific points to sample for comparison

        Returns:
        --------
        dict
            Dictionary containing comparison metrics and statistics
        """
        logger.info(f"Comparing {method1_name} vs {method2_name}")

        # Extract advection components and speeds
        u1 = advection_data1['u_advection']
        v1 = advection_data1['v_advection']
        speed1 = advection_data1['advection_speed']

        u2 = advection_data2['u_advection']
        v2 = advection_data2['v_advection']
        speed2 = advection_data2['advection_speed']

        # Align datasets in time
        common_times = np.intersect1d(u1.time.values, u2.time.values)
        if len(common_times) == 0:
            raise ValueError("No common time steps between the two datasets")

        u1 = u1.sel(time=common_times)
        v1 = v1.sel(time=common_times)
        speed1 = speed1.sel(time=common_times)

        u2 = u2.sel(time=common_times)
        v2 = v2.sel(time=common_times)
        speed2 = speed2.sel(time=common_times)

        # Compute statistics
        results = {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'n_timesteps': len(common_times),
        }

        # Global statistics
        u1_flat = u1.values.flatten()
        u2_flat = u2.values.flatten()
        v1_flat = v1.values.flatten()
        v2_flat = v2.values.flatten()
        speed1_flat = speed1.values.flatten()
        speed2_flat = speed2.values.flatten()

        # Remove NaN values for correlation
        valid_mask = ~(np.isnan(u1_flat) | np.isnan(u2_flat) |
                       np.isnan(v1_flat) | np.isnan(v2_flat))

        if valid_mask.sum() > 0:
            u_corr, u_pval = pearsonr(u1_flat[valid_mask], u2_flat[valid_mask])
            v_corr, v_pval = pearsonr(v1_flat[valid_mask], v2_flat[valid_mask])

            valid_speed_mask = ~(np.isnan(speed1_flat) | np.isnan(speed2_flat))
            if valid_speed_mask.sum() > 0:
                speed_corr, speed_pval = pearsonr(
                    speed1_flat[valid_speed_mask],
                    speed2_flat[valid_speed_mask]
                )
            else:
                speed_corr, speed_pval = np.nan, np.nan
        else:
            u_corr, u_pval = np.nan, np.nan
            v_corr, v_pval = np.nan, np.nan
            speed_corr, speed_pval = np.nan, np.nan

        results['global_statistics'] = {
            'u_correlation': u_corr,
            'v_correlation': v_corr,
            'speed_correlation': speed_corr,
            'u_rmse': np.sqrt(np.nanmean((u1_flat - u2_flat)**2)),
            'v_rmse': np.sqrt(np.nanmean((v1_flat - v2_flat)**2)),
            'speed_rmse': np.sqrt(np.nanmean((speed1_flat - speed2_flat)**2)),
            'speed1_mean': np.nanmean(speed1_flat),
            'speed2_mean': np.nanmean(speed2_flat),
            'speed1_std': np.nanstd(speed1_flat),
            'speed2_std': np.nanstd(speed2_flat),
        }

        # Time series comparison at sample points
        if sample_points is not None:
            point_comparisons = []

            for lat, lon in sample_points:
                try:
                    u1_pt = u1.sel(latitude=lat, longitude=lon, method='nearest')
                    v1_pt = v1.sel(latitude=lat, longitude=lon, method='nearest')
                    speed1_pt = speed1.sel(latitude=lat, longitude=lon, method='nearest')

                    u2_pt = u2.sel(latitude=lat, longitude=lon, method='nearest')
                    v2_pt = v2.sel(latitude=lat, longitude=lon, method='nearest')
                    speed2_pt = speed2.sel(latitude=lat, longitude=lon, method='nearest')

                    point_comparisons.append({
                        'lat': lat,
                        'lon': lon,
                        'u1': u1_pt.values,
                        'v1': v1_pt.values,
                        'speed1': speed1_pt.values,
                        'u2': u2_pt.values,
                        'v2': v2_pt.values,
                        'speed2': speed2_pt.values,
                        'time': common_times
                    })
                except Exception as e:
                    logger.warning(f"Could not extract data at ({lat}, {lon}): {e}")

            results['point_comparisons'] = point_comparisons

        logger.info(f"Speed correlation: {speed_corr:.3f}, RMSE: {results['global_statistics']['speed_rmse']:.2f} m/s")

        return results

    def validate_against_ramp_arrivals(
        self,
        advection_data: xr.Dataset,
        ramp_arrivals: pd.DataFrame,
        station_locations: Dict[str, Tuple[float, float]],
        method_name: str = "Advection Method"
    ) -> Dict:
        """
        Validate advection estimates against observed ramp arrival times.

        Parameters:
        -----------
        advection_data : xr.Dataset
            Advection velocity dataset
        ramp_arrivals : pd.DataFrame
            DataFrame with ramp arrival times at stations
            Columns: station1, station2, ramp_time1, ramp_time2, distance_km, bearing_deg
        station_locations : dict
            Dictionary mapping station IDs to (lat, lon) tuples
        method_name : str
            Name of advection method being validated

        Returns:
        --------
        dict
            Validation metrics comparing predicted vs observed advection
        """
        logger.info(f"Validating {method_name} against ramp arrivals")

        validation_results = []

        for idx, row in ramp_arrivals.iterrows():
            station1 = row['station1']
            station2 = row['station2']
            ramp_time1 = pd.Timestamp(row['ramp_time1'])
            ramp_time2 = pd.Timestamp(row['ramp_time2'])
            distance_km = row['distance_km']
            bearing_deg = row['bearing_deg']

            # Observed advection speed
            time_lag_hours = (ramp_time2 - ramp_time1).total_seconds() / 3600
            if time_lag_hours <= 0:
                continue

            observed_speed_ms = (distance_km * 1000) / (time_lag_hours * 3600)

            # Get predicted advection speed at station 1 location at ramp time
            try:
                lat1, lon1 = station_locations[station1]

                # Select advection data at ramp time (or closest time)
                adv_at_time = advection_data.sel(time=ramp_time1, method='nearest')

                # Extract at station location
                u_adv = float(adv_at_time['u_advection'].sel(
                    latitude=lat1, longitude=lon1, method='nearest'
                ))
                v_adv = float(adv_at_time['v_advection'].sel(
                    latitude=lat1, longitude=lon1, method='nearest'
                ))

                predicted_speed_ms = np.sqrt(u_adv**2 + v_adv**2)

                # Calculate direction of advection
                predicted_direction = np.degrees(np.arctan2(v_adv, u_adv))

                # Normalize to 0-360
                if predicted_direction < 0:
                    predicted_direction += 360

                # Direction difference from bearing
                direction_error = np.abs(predicted_direction - bearing_deg)
                if direction_error > 180:
                    direction_error = 360 - direction_error

                validation_results.append({
                    'station1': station1,
                    'station2': station2,
                    'ramp_time': ramp_time1,
                    'observed_speed_ms': observed_speed_ms,
                    'predicted_speed_ms': predicted_speed_ms,
                    'speed_error_ms': predicted_speed_ms - observed_speed_ms,
                    'speed_error_percent': 100 * (predicted_speed_ms - observed_speed_ms) / observed_speed_ms,
                    'bearing_deg': bearing_deg,
                    'predicted_direction': predicted_direction,
                    'direction_error_deg': direction_error,
                    'distance_km': distance_km,
                    'time_lag_hours': time_lag_hours
                })

            except Exception as e:
                logger.warning(f"Could not validate ramp {station1}->{station2}: {e}")
                continue

        if len(validation_results) == 0:
            logger.warning("No validation results obtained")
            return {}

        validation_df = pd.DataFrame(validation_results)

        # Calculate summary statistics
        summary = {
            'method_name': method_name,
            'n_ramps': len(validation_df),
            'speed_bias_ms': validation_df['speed_error_ms'].mean(),
            'speed_mae_ms': validation_df['speed_error_ms'].abs().mean(),
            'speed_rmse_ms': np.sqrt((validation_df['speed_error_ms']**2).mean()),
            'speed_correlation': validation_df[['observed_speed_ms', 'predicted_speed_ms']].corr().iloc[0, 1],
            'direction_mae_deg': validation_df['direction_error_deg'].mean(),
            'validation_data': validation_df
        }

        logger.info(f"Validation complete: RMSE = {summary['speed_rmse_ms']:.2f} m/s, "
                   f"Bias = {summary['speed_bias_ms']:.2f} m/s")

        return summary


class AdvectionPlotter:
    """
    Create comparison plots for advection velocity methods.
    """

    def __init__(self, figsize: Tuple[float, float] = (16, 10)):
        """
        Initialize plotter.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.figsize = figsize

    def plot_side_by_side_comparison(
        self,
        advection_data1: xr.Dataset,
        advection_data2: xr.Dataset,
        time_index: int,
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        output_file: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        """
        Create side-by-side comparison plots of advection velocity fields.

        Parameters:
        -----------
        advection_data1 : xr.Dataset
            First advection dataset
        advection_data2 : xr.Dataset
            Second advection dataset
        time_index : int
            Time index to plot
        method1_name : str
            Name of first method
        method2_name : str
            Name of second method
        output_file : str, optional
            Path to save figure
        vmin, vmax : float, optional
            Color scale limits for speed (m/s)
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Select time
        speed1 = advection_data1['advection_speed'].isel(time=time_index)
        u1 = advection_data1['u_advection'].isel(time=time_index)
        v1 = advection_data1['v_advection'].isel(time=time_index)

        speed2 = advection_data2['advection_speed'].isel(time=time_index)
        u2 = advection_data2['u_advection'].isel(time=time_index)
        v2 = advection_data2['v_advection'].isel(time=time_index)

        # Determine color scale
        if vmin is None:
            vmin = min(float(speed1.min()), float(speed2.min()))
        if vmax is None:
            vmax = max(float(speed1.max()), float(speed2.max()))

        # Plot 1: Method 1 speed
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.pcolormesh(
            speed1.longitude, speed1.latitude, speed1.values,
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax1.set_title(f'{method1_name}\nAdvection Speed')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(im1, ax=ax1, label='Speed (m/s)')

        # Plot 2: Method 2 speed
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.pcolormesh(
            speed2.longitude, speed2.latitude, speed2.values,
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax2.set_title(f'{method2_name}\nAdvection Speed')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(im2, ax=ax2, label='Speed (m/s)')

        # Plot 3: Difference
        ax3 = fig.add_subplot(gs[0, 2])
        diff = speed1.values - speed2.values
        vdiff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im3 = ax3.pcolormesh(
            speed1.longitude, speed1.latitude, diff,
            cmap='RdBu_r', vmin=-vdiff, vmax=vdiff
        )
        ax3.set_title(f'Difference\n({method1_name} - {method2_name})')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(im3, ax=ax3, label='Speed difference (m/s)')

        # Plot 4: Method 1 vectors (subsampled)
        ax4 = fig.add_subplot(gs[1, 0])
        subsample = 5  # Plot every 5th vector
        ax4.pcolormesh(
            speed1.longitude, speed1.latitude, speed1.values,
            cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.5
        )
        ax4.quiver(
            speed1.longitude.values[::subsample],
            speed1.latitude.values[::subsample],
            u1.values[::subsample, ::subsample],
            v1.values[::subsample, ::subsample],
            scale=50, scale_units='inches'
        )
        ax4.set_title(f'{method1_name}\nVector Field')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')

        # Plot 5: Method 2 vectors (subsampled)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.pcolormesh(
            speed2.longitude, speed2.latitude, speed2.values,
            cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.5
        )
        ax5.quiver(
            speed2.longitude.values[::subsample],
            speed2.latitude.values[::subsample],
            u2.values[::subsample, ::subsample],
            v2.values[::subsample, ::subsample],
            scale=50, scale_units='inches'
        )
        ax5.set_title(f'{method2_name}\nVector Field')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')

        # Plot 6: Scatter plot
        ax6 = fig.add_subplot(gs[1, 2])
        speed1_flat = speed1.values.flatten()
        speed2_flat = speed2.values.flatten()
        valid = ~(np.isnan(speed1_flat) | np.isnan(speed2_flat))
        ax6.scatter(speed1_flat[valid], speed2_flat[valid], alpha=0.3, s=1)
        ax6.plot([vmin, vmax], [vmin, vmax], 'r--', label='1:1 line')
        ax6.set_xlabel(f'{method1_name} (m/s)')
        ax6.set_ylabel(f'{method2_name} (m/s)')
        ax6.set_title('Speed Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Add time information
        time_str = pd.Timestamp(speed1.time.values).strftime('%Y-%m-%d %H:%M')
        fig.suptitle(f'Advection Velocity Comparison - {time_str}', fontsize=14, fontweight='bold')

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_file}")

        return fig

    def plot_validation_results(
        self,
        validation_results: Dict,
        output_file: Optional[str] = None
    ):
        """
        Plot validation results against ramp observations.

        Parameters:
        -----------
        validation_results : dict
            Results from validate_against_ramp_arrivals
        output_file : str, optional
            Path to save figure
        """
        if 'validation_data' not in validation_results:
            logger.warning("No validation data to plot")
            return None

        df = validation_results['validation_data']
        method_name = validation_results['method_name']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Predicted vs Observed Speed
        ax1 = axes[0, 0]
        ax1.scatter(df['observed_speed_ms'], df['predicted_speed_ms'], alpha=0.6)
        lims = [
            min(df['observed_speed_ms'].min(), df['predicted_speed_ms'].min()),
            max(df['observed_speed_ms'].max(), df['predicted_speed_ms'].max())
        ]
        ax1.plot(lims, lims, 'r--', label='1:1 line')
        ax1.set_xlabel('Observed Speed (m/s)')
        ax1.set_ylabel('Predicted Speed (m/s)')
        ax1.set_title('Advection Speed: Predicted vs Observed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add statistics
        corr = validation_results['speed_correlation']
        rmse = validation_results['speed_rmse_ms']
        ax1.text(0.05, 0.95, f'R = {corr:.3f}\nRMSE = {rmse:.2f} m/s',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Speed Error Distribution
        ax2 = axes[0, 1]
        ax2.hist(df['speed_error_ms'], bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='r', linestyle='--', label='Zero error')
        ax2.axvline(df['speed_error_ms'].mean(), color='g', linestyle='--',
                   label=f'Mean = {df["speed_error_ms"].mean():.2f} m/s')
        ax2.set_xlabel('Speed Error (m/s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Speed Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Direction Error
        ax3 = axes[1, 0]
        ax3.scatter(df['bearing_deg'], df['direction_error_deg'], alpha=0.6)
        ax3.set_xlabel('Station Bearing (degrees)')
        ax3.set_ylabel('Direction Error (degrees)')
        ax3.set_title('Direction Error vs Bearing')
        ax3.grid(True, alpha=0.3)
        mae_dir = validation_results['direction_mae_deg']
        ax3.text(0.05, 0.95, f'MAE = {mae_dir:.1f}°',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 4: Error vs Distance
        ax4 = axes[1, 1]
        ax4.scatter(df['distance_km'], df['speed_error_ms'].abs(), alpha=0.6)
        ax4.set_xlabel('Station Distance (km)')
        ax4.set_ylabel('Absolute Speed Error (m/s)')
        ax4.set_title('Speed Error vs Station Distance')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Validation Results: {method_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved validation plot to {output_file}")

        return fig


if __name__ == "__main__":
    print("Advection Validation Module")
    print("Provides tools for comparing and validating advection velocity methods")
