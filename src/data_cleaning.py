"""
src/data_cleaning.py
====================
DataSanitizationEngine — Iterative Conditional Median Profiling strategy
for automotive EDA datasets.

Part of the research project:
"Optimising Automotive Portfolios for Net-Zero Transition,
Urban Mobility, and Risk Mitigation."

CRISP-DM Phase: Data Preparation — Cleaning
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMNS: List[str] = ["normalized-losses", "bore", "stroke"]
GROUP_COLUMN: str = "make"


class DataSanitizationEngine:
    """Applies Iterative Conditional Median Profiling to impute sparse columns.

    The engine imputes missing values in designated actuarial and engineering
    metric columns using a two-tier strategy:

    1. **Brand-specific conditional median** — within each manufacturer group
       the median of non-null observations is computed and used to fill NaN
       entries.  This preserves intra-brand variance structure.

    2. **Global dataset-level fallback** — when an entire manufacturer slice
       contains only NaN values for a target column, the global (dataset-wide)
       median of that column is substituted, guaranteeing zero residual NaN
       values after the sanitization pass.

    Every imputation decision is logged to stdout for full auditability.

    Attributes
    ----------
    target_columns : list[str]
        Names of the sparse columns that require imputation.
    group_column : str
        Name of the column used as the grouping key (manufacturer brand).
    _global_medians : dict[str, float]
        Dataset-level fallback medians computed during ``fit_transform``.

    Examples
    --------
    >>> engine = DataSanitizationEngine()
    >>> clean_df = engine.fit_transform(raw_df)
    """

    target_columns: List[str]
    group_column: str
    _global_medians: dict[str, float]

    def __init__(
        self,
        target_columns: List[str] | None = None,
        group_column: str = GROUP_COLUMN,
    ) -> None:
        """Initialise the sanitization engine with column configuration.

        Parameters
        ----------
        target_columns : list[str] | None, optional
            Columns to impute.  Defaults to ``['normalized-losses', 'bore',
            'stroke']`` when ``None``.
        group_column : str, optional
            Column name used for manufacturer-level grouping.
            Defaults to ``'make'``.

        Returns
        -------
        None
        """
        self.target_columns = target_columns if target_columns is not None else TARGET_COLUMNS
        self.group_column = group_column
        self._global_medians: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_global_medians(self, df: pd.DataFrame) -> None:
        """Pre-compute and cache the dataset-level median for each target column.

        Parameters
        ----------
        df : pd.DataFrame
            The raw input DataFrame prior to any imputation.

        Returns
        -------
        None
        """
        for column in self.target_columns:
            numeric_series: pd.Series = pd.to_numeric(df[column], errors="coerce")
            global_median: float = float(numeric_series.median())
            self._global_medians[column] = global_median
            print(
                f"[GLOBAL MEDIAN] column='{column}' "
                f"global_median={global_median:.4f}",
                file=sys.stdout,
                flush=True,
            )

    def _impute_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Impute a single target column using conditional median profiling.

        For each manufacturer group the brand-specific median is applied first.
        If the entire brand group is null, the pre-computed global median is
        used as a fallback.

        Parameters
        ----------
        df : pd.DataFrame
            Working copy of the DataFrame (may contain NaN entries).
        column : str
            The name of the column to impute.

        Returns
        -------
        pd.DataFrame
            DataFrame with the specified column fully imputed.

        Raises
        ------
        KeyError
            If ``column`` or ``self.group_column`` is absent from ``df``.
        """
        df[column] = pd.to_numeric(df[column], errors="coerce")
        global_fallback: float = self._global_medians[column]

        for make_label, group_index in df.groupby(self.group_column).groups.items():
            group_slice: pd.Series = df.loc[group_index, column]
            null_mask: pd.Series = group_slice.isna()

            if not null_mask.any():
                # No missing values in this brand group — nothing to do.
                continue

            brand_median_value: float = float(group_slice.median())

            if np.isnan(brand_median_value):
                # Entire brand group is null — apply global fallback.
                fill_value: float = global_fallback
                strategy_label: str = "GLOBAL_FALLBACK"
            else:
                fill_value = brand_median_value
                strategy_label = "BRAND_CONDITIONAL_MEDIAN"

            df.loc[group_index[null_mask], column] = fill_value

            null_count: int = int(null_mask.sum())
            print(
                f"[IMPUTE] column='{column}' make='{make_label}' "
                f"strategy={strategy_label} "
                f"fill_value={fill_value:.4f} "
                f"cells_filled={null_count}",
                file=sys.stdout,
                flush=True,
            )

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize the DataFrame by imputing all designated sparse columns.

        Applies Iterative Conditional Median Profiling sequentially to each
        target column, then returns the fully imputed DataFrame.  The input
        DataFrame is copied before modification to preserve the original.

        Parameters
        ----------
        df : pd.DataFrame
            Raw automotive dataset.  Must contain all columns listed in
            ``self.target_columns`` and ``self.group_column``.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with zero residual NaN values in the target columns.

        Raises
        ------
        ValueError
            If any required column is missing from ``df``.
        """
        missing_cols: list[str] = [
            col
            for col in self.target_columns + [self.group_column]
            if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"DataSanitizationEngine: required columns missing from DataFrame: "
                f"{missing_cols}"
            )

        working_df: pd.DataFrame = df.copy()

        print(
            "\n[DataSanitizationEngine] Starting Iterative Conditional Median "
            "Profiling pass...",
            file=sys.stdout,
            flush=True,
        )

        self._compute_global_medians(working_df)

        for column in self.target_columns:
            print(
                f"\n[DataSanitizationEngine] Processing column: '{column}'",
                file=sys.stdout,
                flush=True,
            )
            working_df = self._impute_column(working_df, column)

            residual_nulls: int = int(working_df[column].isna().sum())
            print(
                f"[DataSanitizationEngine] Column '{column}' — "
                f"residual NaN count after imputation: {residual_nulls}",
                file=sys.stdout,
                flush=True,
            )

        print(
            "\n[DataSanitizationEngine] Sanitization pass complete.\n",
            file=sys.stdout,
            flush=True,
        )

        return working_df
