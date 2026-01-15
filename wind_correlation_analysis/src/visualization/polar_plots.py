"""
Polar plot visualization for wind correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PolarCorrelationPlot:
    """
    Create polar plots for correlation analysis results
    """

    def __init__(self, figsize: Tuple[float, float] = (10, 10), dpi: int = 100):
        """
        Initialize polar plot generator

        Parameters:
        -----------
        figsize : Tuple[float, float]
            Figure size in inches
        dpi : int
            Resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_correlation_polar(
        self,
        results_df: pd.DataFrame,
        station1_name: str,
        station2_name: str,
        distance_km: float,
        bearing_deg: float,
        output_file: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Create polar plot of correlation vs normalized direction

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from CrossCorrelationAnalysis with columns:
            bin_center_deg, tau_mean_hours, max_correlation
        station1_name, station2_name : str
            Station names
        distance_km : float
            Distance between stations
        bearing_deg : float
            Bearing from station 1 to station 2
        output_file : str, optional
            Path to save figure
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='polar')

        # Convert bin centers to radians
        theta = np.deg2rad(results_df['bin_center_deg'].values)

        # Radial axis: normalized time lag (tau)
        r = results_df['tau_mean_hours'].values

        # Color: correlation value
        c = results_df['max_correlation'].values

        # Create scatter plot
        scatter = ax.scatter(
            theta, r, c=c,
            cmap='RdYlBu_r',
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidth=1,
            vmin=-1, vmax=1
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, fraction=0.046)
        cbar.set_label('Cross-correlation coefficient', fontsize=12)

        # Set theta direction (clockwise, 0 at North)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Labels
        ax.set_xlabel('Normalized Direction θ (degrees)', fontsize=12, labelpad=20)
        ax.set_ylabel('Normalized Time Lag τ (hours)', fontsize=12, labelpad=30)

        # Title
        title = f'Spatio-temporal Wind Correlation\n{station1_name} - {station2_name}\n'
        title += f'Distance: {distance_km:.1f} km, Bearing: {bearing_deg:.1f}°'
        ax.set_title(title, fontsize=14, pad=20)

        # Grid
        ax.grid(True, alpha=0.3)

        # Add reference arrow for bearing
        ax.annotate(
            '',
            xy=(np.deg2rad(bearing_deg), ax.get_ylim()[1] * 0.9),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->',
                lw=2,
                color='green',
                alpha=0.6
            )
        )
        ax.text(
            np.deg2rad(bearing_deg + 10),
            ax.get_ylim()[1] * 0.95,
            'Station bearing',
            fontsize=10,
            color='green'
        )

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {output_file}")

        if show_plot:
            plt.show()

        return fig

    def plot_correlation_heatmap_polar(
        self,
        results_df: pd.DataFrame,
        station1_name: str,
        station2_name: str,
        output_file: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Create polar heatmap of correlation

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results with correlation_values and lags for each bin
        station1_name, station2_name : str
            Station names
        output_file : str, optional
            Path to save figure
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='polar')

        # Create meshgrid for polar coordinates
        n_bins = len(results_df)
        max_lag_idx = max([len(row['lags']) for _, row in results_df.iterrows()])

        # Initialize correlation matrix
        corr_matrix = np.zeros((n_bins, max_lag_idx))
        theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)
        lag_values = None

        for idx, row in results_df.iterrows():
            lags = row['lags']
            corr = row['correlation_values']

            if lag_values is None:
                lag_values = lags

            # Ensure same length
            n_lags = min(len(corr), max_lag_idx)
            corr_matrix[idx, :n_lags] = corr[:n_lags]

        # Convert lags to hours for radial axis
        lag_hours = lag_values * 5 / 60  # 5-min samples to hours

        # Create polar mesh
        Theta, R = np.meshgrid(theta_edges, lag_hours)

        # Plot
        pcm = ax.pcolormesh(
            Theta, R, corr_matrix.T,
            cmap='RdYlBu_r',
            vmin=-1, vmax=1,
            shading='auto'
        )

        # Colorbar
        cbar = plt.colorbar(pcm, ax=ax, pad=0.1, fraction=0.046)
        cbar.set_label('Cross-correlation', fontsize=12)

        # Set theta direction
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Title
        title = f'Cross-correlation Heatmap\n{station1_name} - {station2_name}'
        ax.set_title(title, fontsize=14, pad=20)

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {output_file}")

        if show_plot:
            plt.show()

        return fig

    def plot_multiple_pairs(
        self,
        results_dict: Dict[Tuple[str, str], pd.DataFrame],
        station_pairs: pd.DataFrame,
        output_dir: str,
        show_plot: bool = False
    ) -> None:
        """
        Create plots for multiple station pairs

        Parameters:
        -----------
        results_dict : Dict[Tuple[str, str], pd.DataFrame]
            Dictionary of results for each station pair
        station_pairs : pd.DataFrame
            Station pair information
        output_dir : str
            Directory to save plots
        show_plot : bool
            Whether to display plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for (station1_id, station2_id), results_df in results_dict.items():
            # Get station info
            pair_info = station_pairs[
                (station_pairs['station1_id'] == station1_id) &
                (station_pairs['station2_id'] == station2_id)
            ].iloc[0]

            station1_name = pair_info['station1_name']
            station2_name = pair_info['station2_name']
            distance_km = pair_info['distance_km']
            bearing_deg = pair_info['bearing_deg']

            # Create scatter plot
            output_file = output_path / f"correlation_polar_{station1_id}_{station2_id}.png"
            self.plot_correlation_polar(
                results_df,
                station1_name,
                station2_name,
                distance_km,
                bearing_deg,
                output_file=str(output_file),
                show_plot=show_plot
            )

            plt.close()

            logger.info(f"Created plot for {station1_id} - {station2_id}")


class CorrelationSummaryPlots:
    """
    Create summary plots for overall correlation analysis
    """

    @staticmethod
    def plot_correlation_vs_distance(
        results_dict: Dict[Tuple[str, str], pd.DataFrame],
        station_pairs: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> Figure:
        """
        Plot maximum correlation vs distance between stations

        Parameters:
        -----------
        results_dict : Dict[Tuple[str, str], pd.DataFrame]
            Dictionary of results
        station_pairs : pd.DataFrame
            Station pair information
        output_file : str, optional
            Path to save figure

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        distances = []
        max_corrs = []
        labels = []

        for (station1_id, station2_id), results_df in results_dict.items():
            pair_info = station_pairs[
                (station_pairs['station1_id'] == station1_id) &
                (station_pairs['station2_id'] == station2_id)
            ].iloc[0]

            distance = pair_info['distance_km']
            max_corr = results_df['max_correlation'].max()

            distances.append(distance)
            max_corrs.append(max_corr)
            labels.append(f"{station1_id}-{station2_id}")

        # Scatter plot
        ax.scatter(distances, max_corrs, s=100, alpha=0.6)

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (distances[i], max_corrs[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Maximum Cross-correlation', fontsize=12)
        ax.set_title('Correlation Decay with Distance', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            logger.info(f"Saved summary plot to {output_file}")

        return fig

    @staticmethod
    def plot_direction_bin_statistics(
        results_df: pd.DataFrame,
        station1_name: str,
        station2_name: str,
        output_file: Optional[str] = None
    ) -> Figure:
        """
        Plot statistics by direction bin

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results for a single station pair
        station1_name, station2_name : str
            Station names
        output_file : str, optional
            Path to save figure

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Correlation by bin
        ax1 = axes[0, 0]
        ax1.bar(
            results_df['bin_center_deg'],
            results_df['max_correlation'],
            width=30,
            alpha=0.7,
            color='steelblue'
        )
        ax1.set_xlabel('Direction Bin Center (°)', fontsize=10)
        ax1.set_ylabel('Max Correlation', fontsize=10)
        ax1.set_title('Correlation by Direction', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sample count by bin
        ax2 = axes[0, 1]
        ax2.bar(
            results_df['bin_center_deg'],
            results_df['n_samples'],
            width=30,
            alpha=0.7,
            color='coral'
        )
        ax2.set_xlabel('Direction Bin Center (°)', fontsize=10)
        ax2.set_ylabel('Number of Samples', fontsize=10)
        ax2.set_title('Data Availability by Direction', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time lag by bin
        ax3 = axes[1, 0]
        ax3.bar(
            results_df['bin_center_deg'],
            results_df['tau_mean_hours'],
            width=30,
            alpha=0.7,
            color='mediumseagreen'
        )
        ax3.set_xlabel('Direction Bin Center (°)', fontsize=10)
        ax3.set_ylabel('Mean τ (hours)', fontsize=10)
        ax3.set_title('Normalized Time Lag by Direction', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Lag at max correlation
        ax4 = axes[1, 1]
        ax4.bar(
            results_df['bin_center_deg'],
            results_df['lag_at_max_hours'],
            width=30,
            alpha=0.7,
            color='mediumpurple'
        )
        ax4.set_xlabel('Direction Bin Center (°)', fontsize=10)
        ax4.set_ylabel('Lag at Max Corr (hours)', fontsize=10)
        ax4.set_title('Time Lag at Maximum Correlation', fontsize=12)
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            f'Direction Bin Statistics: {station1_name} - {station2_name}',
            fontsize=14,
            y=1.00
        )

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            logger.info(f"Saved bin statistics plot to {output_file}")

        return fig


if __name__ == "__main__":
    print("Polar plot visualization module loaded")
