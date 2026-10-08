"""
src/engineering.py
==================
FeatureEngineer — Deterministic feature transformation pipeline for
automotive EDA datasets.

Part of the research project:
"Optimising Automotive Portfolios for Net-Zero Transition,
Urban Mobility, and Risk Mitigation."

CRISP-DM Phase: Data Preparation — Feature Engineering
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Conversion constants
# ---------------------------------------------------------------------------
MPG_TO_L100KM_FACTOR: float = 235.215
"""Conversion multiplier: miles-per-gallon  →  litres per 100 km."""


class FeatureEngineer:
    """Applies deterministic feature transformations to a sanitized automotive DataFrame.

    Transformations produce three categories of derived features:

    1. **Carbon proxy vectors** — volumetric fuel consumption columns
       ``city-L/100km`` and ``highway-L/100km``, converted from MPG using the
       standard imperial-to-metric inversion formula.

    2. **Categorical encoding flags** — binary one-hot indicator columns for
       ``aspiration`` (``aspiration_std``, ``aspiration_turbo``) and
       ``fuel-system`` (prefixed ``fuel_system_<type>``).

    3. **Performance efficiency coefficient** — scalar ``perf_efficiency``
       computed as ``horsepower / city-L/100km``.

    Attributes
    ----------
    mpg_conversion_factor : float
        Constant used in MPG → L/100km conversion (235.215).

    Examples
    --------
    >>> engineer = FeatureEngineer()
    >>> engineered_df = engineer.transform(clean_df)
    """

    mpg_conversion_factor: float

    def __init__(self, mpg_conversion_factor: float = MPG_TO_L100KM_FACTOR) -> None:
        """Initialise the feature engineer.

        Parameters
        ----------
        mpg_conversion_factor : float, optional
            Conversion constant for MPG → L/100km.  Defaults to 235.215.

        Returns
        -------
        None
        """
        self.mpg_conversion_factor = mpg_conversion_factor

    # ------------------------------------------------------------------
    # Private transformation helpers
    # ------------------------------------------------------------------

    def _add_fuel_consumption_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volumetric fuel consumption columns from MPG columns.

        Applies the inversion formula:
            L/100km = 235.215 / mpg

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing numeric ``city-mpg`` and ``highway-mpg``
            columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with two new columns appended:
            ``city-L/100km`` and ``highway-L/100km``.

        Raises
        ------
        KeyError
            If either ``city-mpg`` or ``highway-mpg`` is absent from ``df``.
        """
        city_mpg: pd.Series = pd.to_numeric(df["city-mpg"], errors="coerce")
        highway_mpg: pd.Series = pd.to_numeric(df["highway-mpg"], errors="coerce")

        df["city-L/100km"] = self.mpg_conversion_factor / city_mpg
        df["highway-L/100km"] = self.mpg_conversion_factor / highway_mpg

        print(
            f"[FeatureEngineer] Carbon proxy vectors created: "
            f"'city-L/100km', 'highway-L/100km'",
            file=sys.stdout,
            flush=True,
        )
        return df

    def _encode_aspiration(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the ``aspiration`` column into binary indicator columns.

        Produces exactly two columns: ``aspiration_std`` and
        ``aspiration_turbo``.  The original ``aspiration`` column is dropped.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a string ``aspiration`` column with values
            ``'std'`` or ``'turbo'``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``aspiration_std``, ``aspiration_turbo`` appended
            and ``aspiration`` removed.

        Raises
        ------
        KeyError
            If ``aspiration`` is absent from ``df``.
        """
        aspiration_dummies: pd.DataFrame = pd.get_dummies(
            df["aspiration"],
            prefix="aspiration",
            dtype=int,
        )

        # Guarantee both columns exist even if a category is absent in this slice.
        for expected_col in ("aspiration_std", "aspiration_turbo"):
            if expected_col not in aspiration_dummies.columns:
                aspiration_dummies[expected_col] = 0

        df = pd.concat([df, aspiration_dummies[["aspiration_std", "aspiration_turbo"]]], axis=1)
        df = df.drop(columns=["aspiration"])

        print(
            "[FeatureEngineer] One-hot encoded: 'aspiration' -> "
            "'aspiration_std', 'aspiration_turbo'",
            file=sys.stdout,
            flush=True,
        )
        return df

    def _encode_fuel_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the ``fuel-system`` column into binary indicator columns.

        Each unique fuel system type found in the dataset produces one column
        with the prefix ``fuel_system_``.  The original ``fuel-system`` column
        is dropped.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a string ``fuel-system`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``fuel_system_<type>`` columns appended and
            ``fuel-system`` removed.

        Raises
        ------
        KeyError
            If ``fuel-system`` is absent from ``df``.
        """
        fuel_dummies: pd.DataFrame = pd.get_dummies(
            df["fuel-system"],
            prefix="fuel_system",
            dtype=int,
        )

        new_cols: List[str] = list(fuel_dummies.columns)
        df = pd.concat([df, fuel_dummies], axis=1)
        df = df.drop(columns=["fuel-system"])

        print(
            f"[FeatureEngineer] One-hot encoded: 'fuel-system' -> {new_cols}",
            file=sys.stdout,
            flush=True,
        )
        return df

    def _add_performance_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the scalar performance efficiency coefficient.

        Defined as::

            perf_efficiency = horsepower / city-L/100km

        Division by zero is handled by returning NaN for affected rows
        via ``numpy.where``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains ``horsepower`` and
            ``city-L/100km`` columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with the new ``perf_efficiency`` column appended.

        Raises
        ------
        KeyError
            If ``horsepower`` or ``city-L/100km`` is absent from ``df``.
        """
        horsepower: pd.Series = pd.to_numeric(df["horsepower"], errors="coerce")
        city_l100km: pd.Series = pd.to_numeric(df["city-L/100km"], errors="coerce")

        df["perf_efficiency"] = np.where(
            city_l100km == 0.0,
            np.nan,
            horsepower / city_l100km,
        )

        print(
            "[FeatureEngineer] Performance efficiency coefficient created: "
            "'perf_efficiency' = horsepower / city-L/100km",
            file=sys.stdout,
            flush=True,
        )
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all deterministic feature transformations to the sanitized DataFrame.

        Transformations are applied in the following order:

        1. Fuel consumption vectors (MPG → L/100km).
        2. One-hot encoding of ``aspiration``.
        3. One-hot encoding of ``fuel-system``.
        4. Performance efficiency coefficient.

        Parameters
        ----------
        df : pd.DataFrame
            Sanitized automotive dataset.  Must contain: ``city-mpg``,
            ``highway-mpg``, ``aspiration``, ``fuel-system``, ``horsepower``.

        Returns
        -------
        pd.DataFrame
            Fully feature-engineered DataFrame with all new columns appended
            and source columns replaced as specified.

        Raises
        ------
        ValueError
            If any required source column is missing from ``df``.
        """
        required_source_columns: List[str] = [
            "city-mpg",
            "highway-mpg",
            "aspiration",
            "fuel-system",
            "horsepower",
        ]
        missing_cols: List[str] = [
            col for col in required_source_columns if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"FeatureEngineer: required columns missing from DataFrame: "
                f"{missing_cols}"
            )

        working_df: pd.DataFrame = df.copy()

        print(
            "\n[FeatureEngineer] Starting deterministic feature transformation pass...",
            file=sys.stdout,
            flush=True,
        )

        working_df = self._add_fuel_consumption_vectors(working_df)
        working_df = self._encode_aspiration(working_df)
        working_df = self._encode_fuel_system(working_df)
        working_df = self._add_performance_efficiency(working_df)

        print(
            "[FeatureEngineer] Feature transformation pass complete.\n",
            file=sys.stdout,
            flush=True,
        )

        return working_df
