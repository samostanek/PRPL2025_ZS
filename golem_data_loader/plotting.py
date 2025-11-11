"""
Plotting utilities for GOLEM tokamak diagnostic data.

This module provides publication-quality plotting functions for common
GOLEM diagnostics including plasma current, spectroscopy, MHD activity,
and more.

Author: GOLEM Data Loader Team
Version: 1.2.0
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def plot_basic_diagnostics(
    basic_data,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot basic diagnostics (Bt, Ip, Ich, U_loop) in a 4-panel layout.

    Parameters
    ----------
    basic_data : BasicDiagnosticsData
        Basic diagnostics data object
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Overall figure title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object

    Examples
    --------
    >>> from golem_data_loader import GolemDataLoader
    >>> from golem_data_loader.plotting import plot_basic_diagnostics
    >>> loader = GolemDataLoader(50377)
    >>> basic = loader.load_basic_diagnostics()
    >>> fig = plot_basic_diagnostics(basic, title="Shot 50377 - Basic Diagnostics")
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Toroidal Field
    if basic_data.toroidal_field is not None:
        axes[0].plot(
            basic_data.toroidal_field.time * 1000,
            basic_data.toroidal_field.intensity,
            "b-",
            linewidth=1.5,
        )
        axes[0].set_ylabel("Bt (T)", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Toroidal Magnetic Field", fontsize=11, fontweight="bold")

    # Plasma Current
    if basic_data.plasma_current is not None:
        axes[1].plot(
            basic_data.plasma_current.time * 1000,
            basic_data.plasma_current.intensity / 1000,  # Convert to kA
            "r-",
            linewidth=1.5,
        )
        axes[1].set_ylabel("Ip (kA)", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Plasma Current", fontsize=11, fontweight="bold")
        axes[1].axhline(0, color="k", linestyle="--", alpha=0.3)

    # Chamber Current
    if basic_data.chamber_current is not None:
        axes[2].plot(
            basic_data.chamber_current.time * 1000,
            basic_data.chamber_current.intensity,
            "g-",
            linewidth=1.5,
        )
        axes[2].set_ylabel("Ich (A)", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title("Chamber Current", fontsize=11, fontweight="bold")

    # Loop Voltage
    if basic_data.loop_voltage is not None:
        axes[3].plot(
            basic_data.loop_voltage.time * 1000,
            basic_data.loop_voltage.intensity,
            "m-",
            linewidth=1.5,
        )
        axes[3].set_ylabel("U_loop (V)", fontweight="bold")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title("Loop Voltage", fontsize=11, fontweight="bold")

    axes[3].set_xlabel("Time (ms)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_plasma_current(
    basic_data,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    annotate_peak: bool = True,
) -> plt.Figure:
    """
    Plot plasma current with optional peak annotation.

    Parameters
    ----------
    basic_data : BasicDiagnosticsData
        Basic diagnostics data object
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure
    annotate_peak : bool, optional
        Whether to annotate the peak current

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if basic_data.plasma_current is None:
        logger.warning("No plasma current data available")
        return fig

    time_ms = basic_data.plasma_current.time * 1000
    current_ka = basic_data.plasma_current.intensity / 1000

    ax.plot(time_ms, current_ka, "r-", linewidth=2, label="Plasma Current")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.fill_between(time_ms, current_ka, alpha=0.3, color="red")

    if annotate_peak:
        peak_idx = np.argmax(np.abs(current_ka))
        peak_time = time_ms[peak_idx]
        peak_current = current_ka[peak_idx]
        ax.plot(peak_time, peak_current, "b*", markersize=15, label="Peak")
        ax.annotate(
            f"Peak: {peak_current:.2f} kA\nat {peak_time:.2f} ms",
            xy=(peak_time, peak_current),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Plasma Current (kA)", fontsize=12, fontweight="bold")
    ax.set_title(title or "Plasma Current", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_spectrometry(
    spectrometry_data: Dict,
    lines: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot spectroscopic emission lines.

    Parameters
    ----------
    spectrometry_data : dict
        Dictionary of spectroscopy data (from load_fast_spectrometry)
    lines : list of str, optional
        List of emission lines to plot. If None, plots all available
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object

    Examples
    --------
    >>> spec = loader.load_fast_spectrometry()
    >>> fig = plot_spectrometry(spec, lines=['Hα', 'Hβ', 'He I'])
    """
    if lines is None:
        lines = list(spectrometry_data.keys())

    n_lines = len(lines)
    if n_lines == 0:
        logger.warning("No spectroscopy data to plot")
        return plt.figure()

    fig, axes = plt.subplots(n_lines, 1, figsize=figsize, sharex=True)
    if n_lines == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(range(n_lines))

    for i, line_name in enumerate(lines):
        if line_name not in spectrometry_data:
            logger.warning(f"Line {line_name} not found in data")
            continue

        line_data = spectrometry_data[line_name]
        axes[i].plot(
            line_data.time * 1000,
            line_data.intensity,
            color=colors[i],
            linewidth=1.5,
            label=line_name,
        )
        axes[i].set_ylabel("Intensity (a.u.)", fontweight="bold")
        axes[i].set_title(f"{line_name} Emission", fontsize=11, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_mhd_activity(
    mhd_data,
    rings: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    plot_rms: bool = True,
) -> plt.Figure:
    """
    Plot MHD ring sensor data showing magnetic fluctuations.

    Parameters
    ----------
    mhd_data : MHDRingData
        MHD ring data object
    rings : list of int, optional
        List of ring numbers to plot. If None, plots all available
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure
    plot_rms : bool, optional
        Whether to include RMS values in legend

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    if rings is None:
        rings = sorted(mhd_data.rings.keys())

    n_rings = len(rings)
    if n_rings == 0:
        logger.warning("No MHD ring data to plot")
        return plt.figure()

    # Create main overview plot + individual detail plots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rings + 1, 1, height_ratios=[2] + [1] * n_rings)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    # Main overview
    ax_main = plt.subplot(gs[0])
    colors = plt.cm.tab10(range(n_rings))

    for i, ring_num in enumerate(rings):
        if ring_num not in mhd_data.rings:
            continue

        ring_data = mhd_data.rings[ring_num]
        rms = np.sqrt(np.mean(ring_data.intensity**2))

        label = f"Ring {ring_num}"
        if plot_rms:
            label += f" (RMS: {rms:.2f} V)"

        ax_main.plot(
            ring_data.time * 1000,
            ring_data.intensity,
            color=colors[i],
            linewidth=1,
            alpha=0.8,
            label=label,
        )

    ax_main.set_ylabel("Signal (V)", fontweight="bold")
    ax_main.set_title("MHD Rings Overview", fontsize=12, fontweight="bold")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper right", ncol=n_rings)
    ax_main.axhline(0, color="k", linestyle="--", alpha=0.3)

    # Individual ring details
    for i, ring_num in enumerate(rings):
        if ring_num not in mhd_data.rings:
            continue

        ax = plt.subplot(gs[i + 1])
        ring_data = mhd_data.rings[ring_num]
        rms = np.sqrt(np.mean(ring_data.intensity**2))
        std = np.std(ring_data.intensity)

        ax.plot(
            ring_data.time * 1000,
            ring_data.intensity,
            color=colors[i],
            linewidth=1,
            alpha=0.8,
        )
        ax.set_ylabel("Signal (V)", fontsize=9, fontweight="bold")
        ax.set_title(
            f"Ring {ring_num} (RMS: {rms:.3f} V, σ: {std:.3f} V)",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)

        if i == n_rings - 1:
            ax.set_xlabel("Time (ms)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_plasma_overview(
    basic_data,
    spectrometry_data: Optional[Dict] = None,
    mhd_data=None,
    figsize: Tuple[int, int] = (16, 12),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive overview plot combining multiple diagnostics.

    Parameters
    ----------
    basic_data : BasicDiagnosticsData
        Basic diagnostics data
    spectrometry_data : dict, optional
        Spectroscopy data dictionary
    mhd_data : MHDRingData, optional
        MHD ring data
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Overall plot title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    fig = plt.figure(figsize=figsize)

    # Determine number of rows based on available data
    n_rows = 2  # Always have Bt and Ip
    if spectrometry_data:
        n_rows += 1
    if mhd_data:
        n_rows += 1

    gs = gridspec.GridSpec(n_rows, 1)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    row = 0

    # Toroidal Field
    if basic_data.toroidal_field is not None:
        ax = plt.subplot(gs[row])
        ax.plot(
            basic_data.toroidal_field.time * 1000,
            basic_data.toroidal_field.intensity,
            "b-",
            linewidth=2,
        )
        ax.set_ylabel("Bt (T)", fontweight="bold")
        ax.set_title("Toroidal Magnetic Field", fontweight="bold")
        ax.grid(True, alpha=0.3)
        row += 1

    # Plasma Current
    if basic_data.plasma_current is not None:
        ax = plt.subplot(gs[row])
        ax.plot(
            basic_data.plasma_current.time * 1000,
            basic_data.plasma_current.intensity / 1000,
            "r-",
            linewidth=2,
        )
        ax.set_ylabel("Ip (kA)", fontweight="bold")
        ax.set_title("Plasma Current", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        row += 1

    # Spectroscopy (if available)
    if spectrometry_data:
        ax = plt.subplot(gs[row])
        for line_name, line_data in list(spectrometry_data.items())[:3]:  # Max 3 lines
            ax.plot(
                line_data.time * 1000,
                line_data.intensity,
                linewidth=1.5,
                label=line_name,
                alpha=0.8,
            )
        ax.set_ylabel("Intensity (a.u.)", fontweight="bold")
        ax.set_title("Spectroscopy", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        row += 1

    # MHD Activity (if available)
    if mhd_data:
        ax = plt.subplot(gs[row])
        colors = plt.cm.tab10(range(len(mhd_data.rings)))
        for i, (ring_num, ring_data) in enumerate(sorted(mhd_data.rings.items())):
            ax.plot(
                ring_data.time * 1000,
                ring_data.intensity,
                color=colors[i],
                linewidth=1,
                alpha=0.7,
                label=f"Ring {ring_num}",
            )
        ax.set_ylabel("MHD Signal (V)", fontweight="bold")
        ax.set_title("MHD Activity", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        row += 1

    # Set x-label on bottom plot
    ax.set_xlabel("Time (ms)", fontweight="bold", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_mirnov_coils(
    mirnov_data,
    coils: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Mirnov coil data showing magnetic fluctuations.

    Parameters
    ----------
    mirnov_data : MirnovCoilsData
        Mirnov coils data object
    coils : list of int, optional
        List of coil numbers to plot. If None, plots all available
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    if coils is None:
        coils = sorted(mirnov_data.coils.keys())

    n_coils = len(coils)
    if n_coils == 0:
        logger.warning("No Mirnov coil data to plot")
        return plt.figure()

    fig, axes = plt.subplots(
        n_coils + 1,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2] + [1] * n_coils},
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Overview plot
    colors = plt.cm.tab10(range(n_coils))
    for i, coil_num in enumerate(coils):
        if coil_num not in mirnov_data.coils:
            continue

        coil_data = mirnov_data.coils[coil_num]
        axes[0].plot(
            coil_data.time * 1000,
            coil_data.intensity,
            color=colors[i],
            linewidth=1,
            alpha=0.8,
            label=f"Coil {coil_num}",
        )

    axes[0].set_ylabel("Signal (V)", fontweight="bold")
    axes[0].set_title("Mirnov Coils Overview", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=n_coils)
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.3)

    # Individual coil plots
    for i, coil_num in enumerate(coils):
        if coil_num not in mirnov_data.coils:
            continue

        coil_data = mirnov_data.coils[coil_num]
        axes[i + 1].plot(
            coil_data.time * 1000, coil_data.intensity, color=colors[i], linewidth=1
        )
        axes[i + 1].set_ylabel("Signal (V)", fontsize=9, fontweight="bold")
        axes[i + 1].set_title(f"Coil {coil_num}", fontsize=10, fontweight="bold")
        axes[i + 1].grid(True, alpha=0.3)
        axes[i + 1].axhline(0, color="k", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (ms)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


# Convenience function
def quick_plot(shot_number: int, save_dir: Optional[str] = None, show: bool = True):
    """
    Quickly generate all standard plots for a shot.

    Parameters
    ----------
    shot_number : int
        Shot number to plot
    save_dir : str, optional
        Directory to save plots. If None, plots are not saved
    show : bool, optional
        Whether to display the plots

    Examples
    --------
    >>> from golem_data_loader.plotting import quick_plot
    >>> quick_plot(50377, save_dir='./plots/', show=False)
    """
    from golem_data_loader import GolemDataLoader
    import os

    logger.info(f"Generating plots for shot {shot_number}")
    loader = GolemDataLoader(shot_number)

    # Load data
    basic = loader.load_basic_diagnostics()
    spec = loader.load_fast_spectrometry()
    mhd = loader.load_mhd_ring()

    # Create save paths
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basic_path = os.path.join(save_dir, f"shot_{shot_number}_basic.png")
        spec_path = os.path.join(save_dir, f"shot_{shot_number}_spectrometry.png")
        mhd_path = os.path.join(save_dir, f"shot_{shot_number}_mhd.png")
        overview_path = os.path.join(save_dir, f"shot_{shot_number}_overview.png")
    else:
        basic_path = spec_path = mhd_path = overview_path = None

    # Generate plots
    plot_basic_diagnostics(
        basic,
        title=f"Shot {shot_number} - Basic Diagnostics",
        save_path=basic_path,
        show=show,
    )

    if spec:
        plot_spectrometry(
            spec,
            title=f"Shot {shot_number} - Spectroscopy",
            save_path=spec_path,
            show=show,
        )

    if mhd:
        plot_mhd_activity(
            mhd,
            title=f"Shot {shot_number} - MHD Activity",
            save_path=mhd_path,
            show=show,
        )

    plot_plasma_overview(
        basic,
        spec,
        mhd,
        title=f"Shot {shot_number} - Complete Overview",
        save_path=overview_path,
        show=show,
    )

    logger.info(f"Completed plotting for shot {shot_number}")
