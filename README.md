# Automotive Portfolio Optimisation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.36.1-orange?logo=gradio&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![CRISP--DM](https://img.shields.io/badge/Workflow-CRISP--DM-blueviolet)

> **Optimising Automotive Portfolios for Net-Zero Transition, Urban Mobility, and Risk Mitigation**
>
> A publication-grade data science pipeline and interactive dashboard for automotive portfolio analysis, engineered to IEEE academic standards and structured following the Cookiecutter Data Science architecture pattern.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Data](#data)
- [Methodology / How It Works](#methodology--how-it-works)
- [Results / Output](#results--output)
- [Contributing](#contributing)
- [License](#license)
- [Contact / Author](#contact--author)

---

## Project Overview

The automotive industry is undergoing a structural transformation driven by net-zero emissions mandates, urban congestion policy, and rapidly shifting consumer demand. This project delivers a full end-to-end research-grade data pipeline and interactive optimisation dashboard that enables analysts, engineers, and portfolio strategists to:

- Quantify the fuel efficiency and carbon exposure of a multi-manufacturer vehicle portfolio
- Identify high-risk configurations using actuarial loss metrics and safety indicators
- Simulate the impact of powertrain downsizing, lightweighting, and advanced driver assistance systems (ADAS) on fleet performance
- Produce publication-ready figures and structured analytical reports compliant with IEEE Transactions formatting standards

The pipeline is implemented entirely in Python, follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) workflow from raw data ingestion to interactive deployment, and is structured for zero-configuration deployment on Hugging Face Spaces via Gradio.

---

## Features

| Feature | Description |
|---|---|
| **Iterative Conditional Median Profiling** | Brand-specific NaN imputation with global fallback — preserves intra-manufacturer variance structure |
| **Carbon Proxy Vectorisation** | Converts MPG fuel economy figures to volumetric L/100 km consumption vectors for carbon footprint analysis |
| **One-Hot Encoding Pipeline** | Deterministic categorical encoding of powertrain configuration (`aspiration`) and fuel delivery system (`fuel-system`) |
| **Performance Efficiency Coefficient** | Scalar ratio `horsepower / city-L/100km` — measures power output per unit of urban fuel consumption |
| **Publication-Grade Visualisations** | Three IEEE-compatible 300 DPI figures: skewness distributions, engine downsizing regression, and Pearson correlation heatmap |
| **Interactive Gradio Dashboard** | Real-time portfolio scenario simulation with powertrain selector, mass-reduction slider, and ADAS toggle |
| **Reproducible Pipeline** | Fixed random seed (`numpy.random.seed(42)`), pinned dependency versions, idempotent workspace initialiser |
| **HF Spaces Compatible** | Configured for zero-configuration deployment on Hugging Face Spaces (`server_name="0.0.0.0"`) |

---

## Project Structure

```
Automotive-Portfolio-Optimisation/
│
├── setup_project.py               # Workspace initialiser — creates directory tree and writes requirements.txt
├── app.py                         # Pipeline entry point — Gradio dashboard + full CRISP-DM pipeline execution
├── requirements.txt               # Pinned production dependencies
├── ameda.csv                      # Source automotive dataset (raw engineering and actuarial metrics)
├── README.md                      # This file
│
├── src/
│   ├── data_cleaning.py           # DataSanitizationEngine — Iterative Conditional Median Profiling imputer
│   ├── engineering.py             # FeatureEngineer — carbon proxy vectors, OHE, performance coefficients
│   └── visualization.py          # ResearchVisualizationSuite — three IEEE-grade publication figures
│
├── data/
│   ├── 1_raw/                     # Raw data directory (created by setup_project.py)
│   └── 2_processed/               # Processed / feature-engineered data outputs
│
└── reports/
    ├── paper.tex                  # IEEE LaTeX source — full academic paper (compile-ready, 6 sections, 14 refs)
    ├── paper.docx                 # Word document — IMRaD-structured academic paper (23 headings, 1 table)
    ├── presentation.pptx          # 10-slide PowerPoint deck with embedded figures and speaker notes
    ├── presentation_script.txt    # Verbatim presenter scripts for all 10 slides
    └── figures/
        ├── skewness_distributions.png        # Fisher-Pearson skewness KDE/histogram panel
        ├── correlation_heatmap.png           # Pearson correlation matrix (5 key engineering variables)
        ├── engine_downsizing_regression.png  # Engine size vs fuel consumption regression by aspiration
        ├── curb_weight_regression.png        # Curb weight regression figure
        └── symboling_horsepower_bar.png      # Risk symboling vs horsepower bar chart
```

---

## Tech Stack & Dependencies

### Core Language

| Component | Version |
|---|---|
| Python | 3.10+ |

### Python Libraries

| Library | Version | Role |
|---|---|---|
| `gradio` | 4.36.1 | Interactive web dashboard and HF Spaces deployment |
| `pandas` | 2.2.2 | DataFrame operations, groupBy imputation, data wrangling |
| `numpy` | 1.26.4 | Numerical computation, array operations, random seed control |
| `scipy` | 1.13.1 | Statistical functions — `scipy.stats.skew` for skewness coefficients |
| `matplotlib` | 3.9.0 | Figure rendering, subplot composition, 300 DPI PNG export |
| `seaborn` | 0.13.2 | KDE overlays, regression plots, annotated heatmaps |

### Supporting Tools

| Tool | Purpose |
|---|---|
| `pathlib.Path` | All file I/O — no raw string paths anywhere in the codebase |
| `sys.path` insertion | Resolves `src/` module imports from repository root |

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- Git

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/earthyk/Automotive-Portfolio-Optimisation.git
cd Automotive-Portfolio-Optimisation
```

**2. Create and activate a virtual environment** (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Install pinned dependencies**

```bash
pip install -r requirements.txt
```

**4. Initialise the workspace directory structure**

```bash
python setup_project.py
```

This creates all required directories (`data/1_raw/`, `data/2_processed/`, `src/`, `reports/figures/`) if they do not already exist and confirms each action to stdout.

---

## Usage

### Run the Full Pipeline and Launch the Dashboard

```bash
python app.py
```

This single command:
1. Generates a reproducible synthetic automotive dataset (40+ rows, 6+ manufacturers, seed 42)
2. Runs `DataSanitizationEngine.fit_transform()` — imputes all NaN values
3. Runs `FeatureEngineer.transform()` — applies carbon proxy conversion, OHE, and efficiency scoring
4. Runs all three `ResearchVisualizationSuite` methods — saves 300 DPI PNG figures to `reports/figures/`
5. Launches the Gradio interactive dashboard at `http://localhost:7860`

### Dashboard Controls

Once the interface is running, the left panel exposes three scenario controls:

| Control | Type | Options / Range |
|---|---|---|
| **Powertrain Configuration** | Dropdown | `Legacy ICE`, `Down-Sized + Turbo` |
| **Lightweight Composite Mass Reduction (%)** | Slider | 0 – 30, step 1 |
| **ADAS Safety Package Active** | Checkbox | True / False |

The right panel dynamically renders a Markdown table showing adjusted fuel efficiency, carbon consumption rate, and normalised actuarial loss scores based on the selected scenario parameters.

### Individual Module Usage

Each `src/` module is independently importable:

```python
import sys
sys.path.insert(0, ".")

import pandas as pd
from src.data_cleaning import DataSanitizationEngine
from src.engineering import FeatureEngineer
from src.visualization import ResearchVisualizationSuite

# Load your DataFrame
df = pd.read_csv("ameda.csv")

# Stage 1 — Imputation
engine = DataSanitizationEngine()
df_clean = engine.fit_transform(df)

# Stage 2 — Feature Engineering
fe = FeatureEngineer()
df_engineered = fe.transform(df_clean)

# Stage 3 — Visualisation
viz = ResearchVisualizationSuite()
viz.plot_skewness_distributions(df_engineered)
viz.plot_engine_downsizing_regression(df_engineered)
viz.plot_correlation_heatmap(df_engineered)
```

---

## Data

### Source File

| File | Format | Rows | Description |
|---|---|---|---|
| `ameda.csv` | CSV | ~200 | Automotive engineering and actuarial dataset with vehicle specifications, fuel economy, insurance risk scores, and manufacturer metadata |

### Key Columns

| Column | Type | Description |
|---|---|---|
| `make` | Categorical | Vehicle manufacturer / brand |
| `symboling` | Integer | Actuarial risk rating (−3 to +3; higher = higher insurance risk) |
| `normalized-losses` | Float | Relative average loss payment per insured vehicle year (sparse) |
| `aspiration` | Categorical | Powertrain induction type: `std` (naturally aspirated) or `turbo` |
| `fuel-system` | Categorical | Fuel delivery system type (e.g., `mpfi`, `2bbl`, `idi`) |
| `engine-size` | Integer | Engine displacement in cubic centimetres |
| `horsepower` | Float | Peak engine output in brake horsepower |
| `city-mpg` | Integer | EPA city fuel economy in miles per gallon |
| `highway-mpg` | Integer | EPA highway fuel economy in miles per gallon |
| `curb-weight` | Integer | Vehicle kerb weight in pounds |
| `width` | Float | Vehicle body width in inches |
| `bore` | Float | Engine cylinder bore diameter in inches (sparse) |
| `stroke` | Float | Engine piston stroke length in inches (sparse) |
| `price` | Float | Manufacturer suggested retail price in USD |

### Derived Columns (post-engineering)

| Column | Formula | Description |
|---|---|---|
| `city-L/100km` | `235.215 / city-mpg` | Urban volumetric fuel consumption — carbon proxy vector |
| `highway-L/100km` | `235.215 / highway-mpg` | Highway volumetric fuel consumption |
| `perf_efficiency` | `horsepower / city-L/100km` | Power output per unit urban fuel consumption |
| `aspiration_std` | OHE | Binary flag: 1 = naturally aspirated |
| `aspiration_turbo` | OHE | Binary flag: 1 = turbocharged |
| `fuel_system_*` | OHE | One binary column per unique fuel system type |

---

## Methodology / How It Works

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow across five sequential stages:

```
Raw Data → [1] Data Sanitisation → [2] Feature Engineering → [3] Visualisation → [4] Analysis → [5] Interactive Deployment
```

### Stage 1 — Data Sanitisation (`src/data_cleaning.py`)

**Iterative Conditional Median Profiling (ICMP)** is applied to three sparse actuarial and engineering metric columns: `normalized-losses`, `bore`, and `stroke`.

For each target column:
1. The dataset is grouped by `make` (vehicle manufacturer)
2. A **brand-specific conditional median** is computed from non-null values within each group
3. Missing values are filled using their manufacturer's group median — this preserves intra-brand variance structure rather than collapsing all variation to a single global statistic
4. If an entire manufacturer group contains only null values for a column, a **global dataset-level fallback median** is applied to prevent residual NaN propagation
5. Every imputation decision is logged to stdout with column name, manufacturer, and strategy applied

### Stage 2 — Feature Engineering (`src/engineering.py`)

Three deterministic transformations are applied to the sanitised DataFrame:

1. **Carbon Proxy Vectorisation** — MPG values are converted to L/100 km using the exact conversion factor `235.215 / mpg`, producing `city-L/100km` and `highway-L/100km` as direct surrogates for urban carbon intensity
2. **Categorical Encoding** — `aspiration` and `fuel-system` are one-hot encoded, with the original source columns dropped after encoding
3. **Performance Efficiency Coefficient** — a scalar `perf_efficiency = horsepower / city-L/100km` is computed, with division-by-zero handled gracefully via NaN substitution

### Stage 3 — Visualisation (`src/visualization.py`)

Three publication-grade figures are generated at 300 DPI PNG resolution using an IEEE-compatible visual style:

1. **Skewness Distributions** — Fisher-Pearson skewness coefficients for `price`, `horsepower`, and `symboling`, each rendered as a histogram with overlaid KDE curve and skewness annotation
2. **Engine Downsizing Regression** — bivariate scatterplot of `engine-size` vs `city-L/100km`, colour-coded by aspiration profile, with per-group linear regression fit lines
3. **Pearson Correlation Heatmap** — annotated matrix for five key engineering variables (`curb-weight`, `width`, `horsepower`, `symboling`, `normalized-losses`) using a diverging high-contrast colormap

### Stage 4 — Scenario Analysis (Dashboard Logic in `app.py`)

The Gradio dashboard applies deterministic portfolio scenario rules to the processed dataset means:

- **Down-Sized + Turbo** selected → mean `city-L/100km` reduced by **22%** (downsizing efficiency gain)
- **Mass Reduction Slider** at value *s* → adjusted highway MPG = `base_highway_mpg × (1 + s/100)`
- **ADAS Package Active** → mean `normalized-losses` reduced by **15%** (actuarial risk reduction credit)

---

## Results / Output

Running `python app.py` produces the following artefacts:

### Figures (`reports/figures/`)

| File | Description |
|---|---|
| `skewness_distributions.png` | Skewness panel — price, horsepower, and symboling distributions |
| `engine_downsizing_regression.png` | Engine size vs L/100 km regression by aspiration group |
| `correlation_heatmap.png` | Pearson correlation matrix for 5 engineering variables |
| `curb_weight_regression.png` | Curb weight regression figure |
| `symboling_horsepower_bar.png` | Risk symboling vs horsepower bar chart |

### Academic Reports (`reports/`)

| File | Format | Description |
|---|---|---|
| `paper.tex` | LaTeX | Compile-ready IEEE-format academic paper — 6 sections, 4 numbered equations, 14 BibTeX references |
| `paper.docx` | Word | IMRaD-structured academic paper — 23 headings, 1 formatted table, 14 references |
| `presentation.pptx` | PowerPoint | 10-slide deck with embedded 300 DPI figures and speaker notes on every slide |
| `presentation_script.txt` | Plain text | Verbatim presenter scripts for all 10 slides |

### Interactive Dashboard

A Gradio web interface at `http://localhost:7860` presenting:
- Real-time portfolio scenario output table (fuel consumption, highway MPG, normalised losses)
- Static correlation heatmap panel
- Configurable powertrain, mass reduction, and ADAS scenario controls

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes — ensure all functions carry full NumPy-style docstrings and explicit type annotations
4. Run any existing tests or validate the pipeline end-to-end via `python app.py`
5. Commit with a clear, descriptive message

   ```bash
   git commit -m "Add: brief description of change"
   ```

6. Push to your fork and open a Pull Request against the `main` branch

### Code Standards

- All Python functions and methods must include NumPy-style docstrings (`Parameters`, `Returns`, `Raises`)
- All function signatures must carry explicit type annotations
- All file I/O must use `pathlib.Path` — no raw string paths
- Variable names must follow PEP 8 `snake_case` conventions
- All random operations must use fixed seed `42` for reproducibility

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 earthyk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact / Author

**earthyk**
GitHub: [https://github.com/earthyk](https://github.com/earthyk)
Repository: [https://github.com/earthyk/Automotive-Portfolio-Optimisation](https://github.com/earthyk/Automotive-Portfolio-Optimisation)

---

*Built with Python · Gradio · Pandas · NumPy · SciPy · Matplotlib · Seaborn*
*CRISP-DM Workflow · Cookiecutter Data Science Architecture · IEEE Publication Standard*
