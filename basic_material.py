"""
basic_material.py

Shared utilities for 16.06.
Students should not modify this file.

All environment setup is opt in via setup_environment().
"""

__version__ = "16.06-hw1"

import sys
import os
import importlib.util
from pathlib import Path

import numpy as np

# -------------------------------
# Version and environment helpers
# -------------------------------

PYTHON_VERSION = sys.version_info
NEW_ROOT_LOCUS_COLOR_STRING = PYTHON_VERSION >= (3, 11)

R2D = 180 / np.pi
RPS2HZ = 1 / (2 * np.pi)

# -------------------------------
# Paths
# -------------------------------

DATA_DIR = Path("./data")
FIG_DIR = Path("./figs")

# -------------------------------
# Public setup function
# -------------------------------

def setup_environment(
    *,
    verbose=True,
    set_plot_style=True,
    create_dirs=True,
    check_packages=True
):
    """
    Perform course standard environment setup.

    Parameters
    ----------
    verbose : bool
        Print Python and SymPy versions.
    set_plot_style : bool
        Apply course matplotlib style.
    create_dirs : bool
        Create ./data and ./figs if missing.
    check_packages : bool
        Check that required packages are installed.
    """

    if verbose:
        from platform import python_version
        import sympy as sym
        print("Running Python:", python_version())
        print("Running SymPy:", sym.__version__)

    if set_plot_style:
        _set_plot_style()

    if create_dirs:
        DATA_DIR.mkdir(exist_ok=True)
        FIG_DIR.mkdir(exist_ok=True)

    if check_packages:
        _check_required_packages()

# -------------------------------
# Plotting style
# -------------------------------

def _set_plot_style():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.serif"] = "cmr14"
    rcParams.update({"font.size": 18})

    plt.rcParams["figure.figsize"] = [8, 5.0]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True

    SMALL = 10
    MEDIUM = 14
    BIG = 18

    plt.rc("font", size=SMALL)
    plt.rc("axes", titlesize=SMALL)
    plt.rc("axes", labelsize=SMALL)
    plt.rc("xtick", labelsize=SMALL)
    plt.rc("ytick", labelsize=SMALL)
    plt.rc("legend", fontsize=SMALL)
    plt.rc("figure", titlesize=BIG)

# -------------------------------
# Package checks
# -------------------------------

def _require_package(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Required package '{name}' not found. "
            "Please install it following the course instructions."
        )

def _check_required_packages():
    _require_package("matplotlib")
    _require_package("numpy")
    _require_package("scipy")
    _require_package("sympy")
    _require_package("control")

# -------------------------------
# Numeric helpers
# -------------------------------

def U(t):
    """
    Unit step function.

    Parameters
    ----------
    t : array_like

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    u = np.zeros_like(t)
    u[t >= 0] = 1
    return u

# -------------------------------
# Plot helpers
# -------------------------------

def nicegrid(ax, hh=2):
    """
    Apply standard grid styling to one or more axes.
    """
    try:
        for a in ax:
            _jgrid(a, hh)
    except TypeError:
        _jgrid(ax, hh)

def _jgrid(ax, hh=2):
    import matplotlib.ticker as ticker

    ax.grid(True, which="major", color="#666666", linestyle=":")
    ax.grid(True, which="minor", color="#999999", linestyle=":", alpha=0.2)

    if ax.get_yscale() != "log":
        ax.axhline(y=0, color="k", linestyle="-", lw=1)
    else:
        ax.axhline(y=1, color="k", linestyle="--", lw=1)

    if ax.get_xscale() != "log":
        ax.axvline(x=0, color="k", linestyle="-", lw=1)

    ax.minorticks_on()

def caption(txt, fig, xloc=0.5, yloc=-0.05):
    """
    Add a caption below a figure.
    """
    fig.text(xloc, yloc, txt, ha="center", size=14, color="blue")

# -------------------------------
# Line style presets
# -------------------------------

LOOSELY_DOTTED = (0, (1, 10))
DENSELY_DOTTED = (0, (1, 1))
LOOSELY_DASHED = (0, (5, 10))
DENSELY_DASHED = (0, (5, 1))
LOOSELY_DASHDOTTED = (0, (3, 10, 1, 10))
DENSELY_DASHDOTTED = (0, (3, 1, 1, 1))