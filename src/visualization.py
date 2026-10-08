"""
src/visualization.py
====================
ResearchVisualizationSuite — Publication-grade IEEE-compatible figure generation
for the automotive portfolio research project.

Part of the research project:
"Optimising Automotive Portfolios for Net-Zero Transition,
Urban Mobility, and Risk Mitigation."

CRISP-DM Phase: Evaluation — Visual Reporting
"""

from __future__ import annotations

import pathlib
import sys
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")  # Non-interactive backend for headless/server environments.

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR: pathlib.Path = pathlib.Path("reports") / "figures"

# ---------------------------------------------------------------------------
# IEEE-compatible style configuration
# ---------------------------------------------------------------------------
IEEE_STYLE: dict = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

FIGURE_DPI: int = 300
COLORMAP_DIVERGING: str = "RdBu_r"


class ResearchVisualizationSuite:
    """Generates publication-grade figures for the automotive research portfolio.

    All figures are saved as 300 DPI PNG files to ``/reports/figures/``.
    Each method produces exactly one self-contained figure file and applies
    IEEE-compatible visual styling including labeled axes, titles, tight layout,
    and legends where applicable.

    Attributes
    ----------
    figures_dir : pathlib.Path
        Directory to which all PNG output files are written.

    Examples
    --------
    >>> suite = ResearchVisualizationSuite()
    >>> suite.plot_skewness_distributions(engineered_df)
    >>> suite.plot_engine_downsizing_regression(engineered_df)
    >>> suite.plot_correlation_heatmap(engineered_df)
    """

    figures_dir: pathlib.Path

    def __init__(self, figures_dir: pathlib.Path = FIGURES_DIR) -> None:
        """Initialise the visualization suite and ensure the output directory exists.

        Parameters
        ----------
        figures_dir : pathlib.Path, optional
            Destination directory for figure exports.
            Defaults to ``reports/figures``.

        Returns
        -------
        None
        """
        self.figures_dir = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_skewness_distributions(self, df: pd.DataFrame) -> None:
        """Render KDE-overlaid histograms annotated with Fisher-Pearson skewness.

        For each of the three target columns — ``price``, ``horsepower``, and
        ``symboling`` — a subplot is produced containing:

        * A histogram of the numeric distribution.
        * A Kernel Density Estimate curve overlaid on the histogram.
        * An annotation showing the Fisher-Pearson skewness coefficient
          computed via :func:`scipy.stats.skew`.

        The combined figure is saved to
        ``reports/figures/skewness_distributions.png`` at 300 DPI.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered automotive DataFrame.  Must contain columns
            ``'price'``, ``'horsepower'``, and ``'symboling'``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any of the three target columns is absent from ``df``.
        """
        target_columns: List[str] = ["price", "horsepower", "symboling"]
        missing: List[str] = [c for c in target_columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"ResearchVisualizationSuite: columns missing for skewness plot: {missing}"
            )

        with plt.rc_context(IEEE_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig.suptitle(
                "Distribution Skewness Analysis — Automotive Portfolio",
                fontsize=12,
                fontweight="bold",
            )

            for ax, column in zip(axes, target_columns):
                series: pd.Series = pd.to_numeric(df[column], errors="coerce").dropna()

                skewness_value: float = float(stats.skew(series))

                ax.hist(
                    series,
                    bins=20,
                    density=True,
                    alpha=0.55,
                    color="#3b82d4",
                    edgecolor="#1f2328",
                    linewidth=0.4,
                    label="Histogram",
                )

                kde_x: np.ndarray = np.linspace(series.min(), series.max(), 300)
                kde = stats.gaussian_kde(series)
                ax.plot(kde_x, kde(kde_x), color="#c0392b", linewidth=2.0, label="KDE")

                ax.annotate(
                    f"Skewness: {skewness_value:.3f}",
                    xy=(0.97, 0.93),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f8fa", alpha=0.8),
                )

                ax.set_title(column.replace("-", " ").title())
                ax.set_xlabel(column)
                ax.set_ylabel("Density")
                ax.legend(loc="upper left")

            plt.tight_layout()
            output_path: pathlib.Path = self.figures_dir / "skewness_distributions.png"
            fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)

        print(
            f"[ResearchVisualizationSuite] Saved: {output_path.resolve()}",
            file=sys.stdout,
            flush=True,
        )

    def plot_engine_downsizing_regression(self, df: pd.DataFrame) -> None:
        """Render bivariate regression scatterplot: engine size vs fuel consumption.

        Data points are color-coded by aspiration profile reconstructed from
        the one-hot columns ``aspiration_std`` and ``aspiration_turbo``.
        A linear regression fit line is overlaid for each aspiration group.

        The figure is saved to
        ``reports/figures/engine_downsizing_regression.png`` at 300 DPI.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered automotive DataFrame.  Must contain columns
            ``'engine-size'``, ``'city-L/100km'``, ``'aspiration_std'``, and
            ``'aspiration_turbo'``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any required column is absent from ``df``.
        """
        required: List[str] = ["engine-size", "city-L/100km", "aspiration_std", "aspiration_turbo"]
        missing: List[str] = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"ResearchVisualizationSuite: columns missing for regression plot: {missing}"
            )

        plot_df: pd.DataFrame = df[required].copy()
        plot_df["engine-size"] = pd.to_numeric(plot_df["engine-size"], errors="coerce")
        plot_df["city-L/100km"] = pd.to_numeric(plot_df["city-L/100km"], errors="coerce")
        plot_df = plot_df.dropna(subset=["engine-size", "city-L/100km"])

        # Reconstruct categorical label from one-hot columns.
        plot_df["aspiration_label"] = np.where(
            plot_df["aspiration_turbo"] == 1, "Turbo", "Standard"
        )

        aspiration_groups: dict[str, dict] = {
            "Standard": {"color": "#3b82d4", "marker": "o"},
            "Turbo": {"color": "#c0392b", "marker": "^"},
        }

        with plt.rc_context(IEEE_STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.suptitle(
                "Engine Downsizing Regression — Engine Size vs Fuel Consumption",
                fontsize=12,
                fontweight="bold",
            )

            for label, style in aspiration_groups.items():
                group: pd.DataFrame = plot_df[plot_df["aspiration_label"] == label]
                if group.empty:
                    continue

                x_vals: np.ndarray = group["engine-size"].to_numpy()
                y_vals: np.ndarray = group["city-L/100km"].to_numpy()

                ax.scatter(
                    x_vals,
                    y_vals,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.65,
                    s=45,
                    label=f"{label} (n={len(group)})",
                    zorder=3,
                )

                # Linear regression overlay.
                slope, intercept, _r, _p, _se = stats.linregress(x_vals, y_vals)
                x_line: np.ndarray = np.linspace(x_vals.min(), x_vals.max(), 200)
                y_line: np.ndarray = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )

            ax.set_xlabel("Engine Size (cc)")
            ax.set_ylabel("City Fuel Consumption (L/100km)")
            ax.legend(title="Aspiration")
            plt.tight_layout()

            output_path: pathlib.Path = self.figures_dir / "engine_downsizing_regression.png"
            fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)

        print(
            f"[ResearchVisualizationSuite] Saved: {output_path.resolve()}",
            file=sys.stdout,
            flush=True,
        )

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Render a Pearson Product-Moment Correlation Matrix Heatmap.

        Computes and visualises the pairwise Pearson correlation matrix for the
        following five variables: ``curb-weight``, ``width``, ``horsepower``,
        ``symboling``, and ``normalized-losses``.

        Each cell is annotated with its rounded correlation coefficient.
        A diverging high-contrast colormap is applied with a colorbar.

        The figure is saved to
        ``reports/figures/correlation_heatmap.png`` at 300 DPI.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered automotive DataFrame.  Must contain all five
            target columns.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any of the five target columns is absent from ``df``.
        """
        heatmap_columns: List[str] = [
            "curb-weight",
            "width",
            "horsepower",
            "symboling",
            "normalized-losses",
        ]
        missing: List[str] = [c for c in heatmap_columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"ResearchVisualizationSuite: columns missing for heatmap: {missing}"
            )

        numeric_subset: pd.DataFrame = df[heatmap_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        correlation_matrix: pd.DataFrame = numeric_subset.corr(method="pearson")

        with plt.rc_context(IEEE_STYLE):
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.suptitle(
                "Pearson Correlation Matrix — Key Engineering Variables",
                fontsize=12,
                fontweight="bold",
            )

            sns.heatmap(
                correlation_matrix,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap=COLORMAP_DIVERGING,
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.5,
                linecolor="#e5e7eb",
                square=True,
                cbar_kws={"shrink": 0.82, "label": "Pearson r"},
            )

            ax.set_title("")  # suptitle is used instead
            ax.set_xticklabels(
                [label.replace("-", "\n") for label in heatmap_columns],
                rotation=0,
                fontsize=8,
            )
            ax.set_yticklabels(
                [label.replace("-", "\n") for label in heatmap_columns],
                rotation=0,
                fontsize=8,
            )

            plt.tight_layout()
            output_path: pathlib.Path = self.figures_dir / "correlation_heatmap.png"
            fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)

        print(
            f"[ResearchVisualizationSuite] Saved: {output_path.resolve()}",
            file=sys.stdout,
            flush=True,
        )
