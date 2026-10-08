"""
app.py
======
Gradio web application — complete pipeline entry point for:
"Optimising Automotive Portfolios for Net-Zero Transition,
Urban Mobility, and Risk Mitigation."

Zero-configuration deployment on Hugging Face Spaces.
Run with: python app.py

Pipeline sequence on startup:
    1. Build 40-row synthetic automotive DataFrame (seed=42).
    2. DataSanitizationEngine.fit_transform()   — imputation pass.
    3. FeatureEngineer.transform()              — feature engineering pass.
    4. ResearchVisualizationSuite               — export all three figures.
    5. Launch Gradio Blocks interface.
"""

from __future__ import annotations

import io
import pathlib
import sys

# ---------------------------------------------------------------------------
# Force UTF-8 stdout so Unicode in Markdown strings never crashes on Windows.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Ensure src/ is importable from any working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import gradio as gr

from data_cleaning import DataSanitizationEngine
from engineering import FeatureEngineer
from visualization import ResearchVisualizationSuite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR: pathlib.Path = _PROJECT_ROOT / "reports" / "figures"
HEATMAP_PATH: pathlib.Path = FIGURES_DIR / "correlation_heatmap.png"
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Mock dataset factory
# ---------------------------------------------------------------------------

def build_mock_dataframe() -> pd.DataFrame:
    """Construct a reproducible 40-row synthetic automotive DataFrame.

    All column values are generated with realistic domain ranges using a
    fixed NumPy random seed (42) to guarantee full reproducibility.

    The mock dataset includes:
    * Six distinct manufacturer names.
    * Both ``'std'`` and ``'turbo'`` aspiration values.
    * Four distinct fuel-system types.
    * Intentional NaN entries in ``normalized-losses``, ``bore``, and
      ``stroke`` to exercise the sanitization engine.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        A 40-row synthetic DataFrame with all required pipeline columns.
    """
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED)

    n_rows: int = 40

    makes: list[str] = [
        "toyota", "honda", "bmw", "volkswagen", "mazda", "subaru"
    ]
    aspirations: list[str] = ["std", "turbo"]
    fuel_systems: list[str] = ["mpfi", "2bbl", "idi", "spdi"]

    make_col: list[str] = [makes[i % len(makes)] for i in range(n_rows)]
    aspiration_col: list[str] = [
        "turbo" if i % 5 == 0 else "std" for i in range(n_rows)
    ]
    fuel_system_col: list[str] = [fuel_systems[i % len(fuel_systems)] for i in range(n_rows)]

    engine_sizes: np.ndarray = rng.integers(90, 280, size=n_rows).astype(float)
    horsepower: np.ndarray = rng.integers(70, 210, size=n_rows).astype(float)
    city_mpg: np.ndarray = rng.integers(18, 42, size=n_rows).astype(float)
    highway_mpg: np.ndarray = city_mpg + rng.integers(3, 12, size=n_rows)
    curb_weight: np.ndarray = rng.integers(1800, 3600, size=n_rows).astype(float)
    width: np.ndarray = rng.uniform(60.0, 72.0, size=n_rows)
    symboling: np.ndarray = rng.integers(-2, 4, size=n_rows).astype(float)
    price: np.ndarray = rng.uniform(6500.0, 38000.0, size=n_rows)
    bore: np.ndarray = rng.uniform(2.5, 3.9, size=n_rows)
    stroke: np.ndarray = rng.uniform(2.5, 4.2, size=n_rows)
    normalized_losses: np.ndarray = rng.uniform(65.0, 256.0, size=n_rows)

    # Introduce intentional NaN entries to exercise imputation (≈18% sparsity).
    nan_indices_nl: np.ndarray = rng.choice(n_rows, size=7, replace=False)
    nan_indices_bore: np.ndarray = rng.choice(n_rows, size=5, replace=False)
    nan_indices_stroke: np.ndarray = rng.choice(n_rows, size=6, replace=False)

    normalized_losses[nan_indices_nl] = np.nan
    bore[nan_indices_bore] = np.nan
    stroke[nan_indices_stroke] = np.nan

    mock_data: dict = {
        "make": make_col,
        "normalized-losses": normalized_losses,
        "bore": bore,
        "stroke": stroke,
        "city-mpg": city_mpg,
        "highway-mpg": highway_mpg,
        "engine-size": engine_sizes,
        "horsepower": horsepower,
        "curb-weight": curb_weight,
        "width": width,
        "symboling": symboling,
        "price": price,
        "aspiration": aspiration_col,
        "fuel-system": fuel_system_col,
    }

    return pd.DataFrame(mock_data)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """Execute the full CRISP-DM data preparation and visualization pipeline.

    Steps performed:
    1. Build synthetic mock DataFrame.
    2. Sanitize via :class:`DataSanitizationEngine`.
    3. Engineer features via :class:`FeatureEngineer`.
    4. Generate and save all three research figures via
       :class:`ResearchVisualizationSuite`.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        The fully processed and feature-engineered DataFrame, ready for
        use by the Gradio interface callbacks.
    """
    print("\n" + "=" * 60, flush=True)
    print("  Automotive Portfolio Pipeline -- Starting...", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Build mock dataset.
    raw_df: pd.DataFrame = build_mock_dataframe()
    print(f"[Pipeline] Mock dataset built: {raw_df.shape[0]} rows x {raw_df.shape[1]} cols")

    # Step 2: Sanitization.
    sanitizer: DataSanitizationEngine = DataSanitizationEngine()
    sanitized_df: pd.DataFrame = sanitizer.fit_transform(raw_df)

    # Step 3: Feature engineering.
    engineer: FeatureEngineer = FeatureEngineer()
    engineered_df: pd.DataFrame = engineer.transform(sanitized_df)

    # Step 4: Visualization exports.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suite: ResearchVisualizationSuite = ResearchVisualizationSuite(figures_dir=FIGURES_DIR)
    suite.plot_skewness_distributions(engineered_df)
    suite.plot_engine_downsizing_regression(engineered_df)
    suite.plot_correlation_heatmap(engineered_df)

    print("=" * 60, flush=True)
    print("  Automotive Portfolio Pipeline -- Complete.", flush=True)
    print("=" * 60 + "\n", flush=True)

    return engineered_df


# ---------------------------------------------------------------------------
# Execute pipeline at import time (required for Hugging Face Spaces compat.)
# ---------------------------------------------------------------------------
_PROCESSED_DF: pd.DataFrame = run_pipeline()

# Pre-compute base values used by the Gradio callback.
_BASE_CITY_L100KM: float = float(_PROCESSED_DF["city-L/100km"].mean())
_BASE_HIGHWAY_MPG: float = float(_PROCESSED_DF["highway-mpg"].mean())
_BASE_NORMALIZED_LOSSES: float = float(_PROCESSED_DF["normalized-losses"].mean())


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def compute_portfolio_summary(
    powertrain_config: str,
    mass_reduction_pct: int,
    adas_active: bool,
) -> str:
    """Compute and render an analytical portfolio summary as a Markdown table.

    Applies deterministic adjustment rules to the base processed-dataset means:

    * **Down-Sized + Turbo** selected → city fuel consumption reduced by 22%.
    * **Mass reduction slider** → highway MPG scaled up proportionally.
    * **ADAS active** → normalized losses reduced by 15%.

    Parameters
    ----------
    powertrain_config : str
        Selected powertrain label.  One of ``"Legacy ICE"`` or
        ``"Down-Sized + Turbo"``.
    mass_reduction_pct : int
        Lightweight composite mass reduction percentage (0–30).
    adas_active : bool
        Whether the ADAS safety package is active.

    Returns
    -------
    str
        A Markdown-formatted analytical summary table string.
    """
    city_l100km: float = _BASE_CITY_L100KM
    highway_mpg: float = _BASE_HIGHWAY_MPG
    normalized_losses: float = _BASE_NORMALIZED_LOSSES

    # Rule 1 — Powertrain selection: 22% city fuel reduction for turbo.
    if powertrain_config == "Down-Sized + Turbo":
        city_l100km = city_l100km * (1.0 - 0.22)
        powertrain_note: str = "✓ 22% city fuel efficiency gain applied"
    else:
        powertrain_note = "Baseline — no efficiency adjustment"

    # Rule 2 — Mass reduction: proportional highway MPG uplift.
    adjusted_highway_mpg: float = highway_mpg * (1.0 + mass_reduction_pct / 100.0)

    # Rule 3 — ADAS: 15% normalized-losses reduction.
    if adas_active:
        normalized_losses = normalized_losses * (1.0 - 0.15)
        adas_note: str = "✓ 15% actuarial risk reduction applied"
    else:
        adas_note = "ADAS inactive — baseline risk profile"

    markdown_output: str = f"""
## 📊 Automotive Portfolio Analytical Summary

### Configuration
| Parameter | Value |
|---|---|
| Powertrain Configuration | **{powertrain_config}** |
| Composite Mass Reduction | **{mass_reduction_pct}%** |
| ADAS Safety Package | **{"Active" if adas_active else "Inactive"}** |

---

### Adjusted Performance Metrics
| Metric | Adjusted Value | Unit |
|---|---|---|
| City Fuel Consumption | **{city_l100km:.2f}** | L/100km |
| Highway Fuel Economy | **{adjusted_highway_mpg:.1f}** | MPG |
| Normalized Risk Loss Index | **{normalized_losses:.1f}** | Index |

---

### Adjustment Logic Applied
| Rule | Status |
|---|---|
| Powertrain Efficiency | {powertrain_note} |
| Mass Reduction Uplift | Highway MPG × (1 + {mass_reduction_pct}/100) = **{adjusted_highway_mpg:.1f} MPG** |
| ADAS Risk Mitigation | {adas_note} |

---
*Source: Synthetic mock dataset (n=40) — CRISP-DM pipeline, seed=42*
"""
    return markdown_output.strip()


# ---------------------------------------------------------------------------
# Gradio Blocks interface
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface layout.

    The layout consists of two panels:

    * **Left panel** — Interactive controllers (Dropdown, Slider, Checkbox).
    * **Right panel** — Markdown output + static correlation heatmap image.

    Parameters
    ----------
    None

    Returns
    -------
    gr.Blocks
        Configured, ready-to-launch Gradio Blocks application instance.
    """
    heatmap_image_path: str | None = (
        str(HEATMAP_PATH) if HEATMAP_PATH.exists() else None
    )

    with gr.Blocks(
        title="Automotive Portfolio Optimisation — Net-Zero Research Dashboard",
        theme=gr.themes.Base(),
    ) as demo:

        gr.Markdown(
            """
            # 🚗 Automotive Portfolio Optimisation
            ## Net-Zero Transition · Urban Mobility · Risk Mitigation
            *CRISP-DM Research Dashboard — IEEE-Standard Publication Output*
            ---
            """
        )

        with gr.Row():
            # ---------------------------------------------------------------
            # Left Panel — Interactive Controllers
            # ---------------------------------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration Controllers")

                powertrain_dropdown: gr.Dropdown = gr.Dropdown(
                    choices=["Legacy ICE", "Down-Sized + Turbo"],
                    value="Legacy ICE",
                    label="Powertrain Configuration",
                    info="Select the target powertrain architecture for analysis.",
                )

                mass_slider: gr.Slider = gr.Slider(
                    minimum=0,
                    maximum=30,
                    step=1,
                    value=0,
                    label="Lightweight Composite Mass Reduction (%)",
                    info="Projected structural mass saving from composite materials.",
                )

                adas_checkbox: gr.Checkbox = gr.Checkbox(
                    value=False,
                    label="ADAS Safety Package Active",
                    info="Activates actuarial risk reduction (−15% normalized losses).",
                )

                run_button: gr.Button = gr.Button(
                    "Run Portfolio Analysis",
                    variant="primary",
                )

            # ---------------------------------------------------------------
            # Right Panel — Analytical Outputs
            # ---------------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analytical Summary")

                summary_output: gr.Markdown = gr.Markdown(
                    value=compute_portfolio_summary("Legacy ICE", 0, False),
                )

                gr.Markdown("### 🗺️ Correlation Heatmap — Key Engineering Variables")

                heatmap_image: gr.Image = gr.Image(
                    value=heatmap_image_path,
                    label="Pearson Correlation Matrix (Curb-Weight · Width · HP · Symboling · NL)",
                    show_label=True,
                    interactive=False,
                )

        # -------------------------------------------------------------------
        # Event wiring — button click triggers computation.
        # -------------------------------------------------------------------
        run_button.click(
            fn=compute_portfolio_summary,
            inputs=[powertrain_dropdown, mass_slider, adas_checkbox],
            outputs=[summary_output],
        )

        # Live update on any controller change (no explicit button press needed).
        for controller in (powertrain_dropdown, mass_slider, adas_checkbox):
            controller.change(
                fn=compute_portfolio_summary,
                inputs=[powertrain_dropdown, mass_slider, adas_checkbox],
                outputs=[summary_output],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    interface: gr.Blocks = build_interface()
    interface.launch(
        server_name="0.0.0.0",
        share=False,
    )
