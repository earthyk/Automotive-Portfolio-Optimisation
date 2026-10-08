"""
setup_project.py
================
Programmatic workspace initializer for the research project:
"Optimising Automotive Portfolios for Net-Zero Transition,
Urban Mobility, and Risk Mitigation."

Follows CRISP-DM workspace conventions and the Cookiecutter Data Science
directory structure. Idempotent — safe to re-run on an existing project.
"""

from __future__ import annotations

import pathlib
import sys


# ---------------------------------------------------------------------------
# Directory manifest
# ---------------------------------------------------------------------------
REQUIRED_DIRECTORIES: list[pathlib.Path] = [
    pathlib.Path("data") / "1_raw",
    pathlib.Path("data") / "2_processed",
    pathlib.Path("src"),
    pathlib.Path("reports") / "figures",
]

# ---------------------------------------------------------------------------
# requirements.txt content — pinned, mutually-compatible mid-2024 versions
# ---------------------------------------------------------------------------
REQUIREMENTS_CONTENT: str = """\
gradio==4.36.1
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.0
seaborn==0.13.2
"""


def create_directories(directories: list[pathlib.Path]) -> None:
    """Create all required project directories if they do not already exist.

    Parameters
    ----------
    directories : list[pathlib.Path]
        Ordered list of directory paths to create, relative to the current
        working directory.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If a directory cannot be created due to a filesystem permission error.
    """
    for directory in directories:
        if directory.exists():
            print(f"[SKIP]    Directory already exists: {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[CREATED] Directory created:        {directory}")


def write_requirements(destination: pathlib.Path, content: str) -> None:
    """Write the pinned requirements manifest to the project root.

    Parameters
    ----------
    destination : pathlib.Path
        Target file path for requirements.txt.
    content : str
        Full text content to write, including newline-terminated package specs.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written due to a filesystem permission error.
    """
    destination.write_text(content, encoding="utf-8")
    print(f"[WRITTEN] requirements.txt written to: {destination.resolve()}")


def main() -> None:
    """Entry point: initialize workspace directories and write requirements.txt.

    Returns
    -------
    None
    """
    print("=" * 60)
    print("  Automotive Portfolio Research — Workspace Initializer")
    print("=" * 60)

    create_directories(REQUIRED_DIRECTORIES)

    requirements_path: pathlib.Path = pathlib.Path("requirements.txt")
    write_requirements(requirements_path, REQUIREMENTS_CONTENT)

    print("=" * 60)
    print("  Workspace initialization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    sys.exit(0)
