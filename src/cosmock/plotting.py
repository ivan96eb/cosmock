"""Plotting helpers for calibration and mock validation."""

from __future__ import annotations

import numpy as np

from ._optional import import_optional


def _plt():
    return import_optional("matplotlib.pyplot", extra="plot", purpose="Plotting")


def ratio_ploter(average_ratio, N_bins, save_path=None, ellmin=10, ymin=0.95, ymax=1.05, dashed_vert=2 * 256):
    """Plot average mock-to-target power-spectrum ratios."""

    plt = _plt()
    fig, axes = plt.subplots(N_bins, N_bins, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ell = np.arange(average_ratio.shape[2])
    for i in range(N_bins):
        for j in range(N_bins):
            if j > i:
                axes[i, j].axis("off")
                continue
            ax = axes[i, j]
            ell_plot = ell[ellmin:]
            avg_plot = average_ratio[i, j, ellmin:]
            ax.semilogx(ell_plot, avg_plot, label="Average ratio", lw=0.8, color="blue")
            ax.axhline(1, color="black", linestyle="--", lw=1, alpha=0.5)
            ax.set_xlabel(r"$\ell$", fontsize=11)
            ax.set_ylabel(r"$\langle C_\ell^{\rm mock}/(W_\ell^2C_\ell^{\rm true})\rangle$", fontsize=11)
            ax.set_title(f"Bin {i + 1}" if i == j else f"Bins ({i + 1},{j + 1})", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylim(ymin, ymax)
            if dashed_vert is not None:
                ax.axvline(dashed_vert, linestyle="dashed", color="black", lw=1, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def hist_comparisson_plotter(kappa_sim, kappa_lr_pix, kappa_mock, kappa_mock_pix, N_bins, N, save_path=None):
    """Plot log-scale histogram comparisons for simulation and mock maps."""

    plt = _plt()
    fig, axes = plt.subplots(2, N_bins // 2, figsize=(10, 10))
    axes = np.asarray(axes).flatten()
    for i in range(N_bins):
        ax = axes[i]
        counts, bins, _ = ax.hist(
            kappa_sim[i], bins=50, histtype="step", linewidth=2.5, label="Simulation", color="black"
        )
        ax.hist(kappa_lr_pix[i], bins=bins, histtype="step", linewidth=2, label="Pix + LP sim", color="red")
        ax.hist(kappa_mock[i], bins=bins, histtype="step", linewidth=2, label=f"G{N} mock", color="blue", linestyle="dashed")
        ax.hist(
            kappa_mock_pix[i],
            bins=bins,
            histtype="step",
            linewidth=2,
            label=f"G{N} mock Pix + LP",
            color="orange",
            linestyle="dashed",
        )
        if np.any(counts > 0):
            ax.set_ylim(1e-1, None)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_title(f"Z-bin {i + 1}", fontsize=18, pad=15)
        if i == 0:
            ax.legend(fontsize=13, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def hist_comparisson_plotter_linear(kappa_sim, kappa_lr_pix, kappa_mock, kappa_mock_pix, N_bins, N, save_path=None):
    """Plot linear-scale histogram comparisons for simulation and mock maps."""

    plt = _plt()
    fig, axes = plt.subplots(2, N_bins // 2, figsize=(10, 10))
    axes = np.asarray(axes).flatten()
    for i in range(N_bins):
        ax = axes[i]
        if i in [2, 3]:
            hist_range = (kappa_sim[i].min(), 0.018)
        else:
            hist_range = (kappa_sim[i].min(), 0.01)
        _, bins, _ = ax.hist(
            kappa_sim[i],
            bins=50,
            histtype="step",
            linewidth=2.5,
            label="Simulation",
            color="black",
            range=hist_range,
        )
        ax.hist(kappa_lr_pix[i], bins=bins, histtype="step", linewidth=2, label="Pix + LP sim", color="red")
        ax.hist(kappa_mock[i], bins=bins, histtype="step", linewidth=2, label=f"G{N} mock", color="blue", linestyle="dashed")
        ax.hist(
            kappa_mock_pix[i],
            bins=bins,
            histtype="step",
            linewidth=2,
            label=f"G{N} mock Pix + LP",
            color="orange",
            linestyle="dashed",
        )
        ax.set_ylim(1e-1, None)
        ax.set_xlabel(r"$\kappa$", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_title(f"Z-bin {i + 1}", fontsize=18, pad=15)
        if i == 0:
            ax.legend(fontsize=13, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def ratio_ploter_comp(average_ratio, average_ratio_filt, N_bins, save_path=None, ellmin=10, ymin=0.95, ymax=1.05, dashed_vert=2 * 256):
    """Plot raw and filtered average mock-to-target power-spectrum ratios."""

    plt = _plt()
    fig, axes = plt.subplots(N_bins, N_bins, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ell = np.arange(average_ratio.shape[2])
    for i in range(N_bins):
        for j in range(N_bins):
            if j > i:
                axes[i, j].axis("off")
                continue
            ax = axes[i, j]
            ell_plot = ell[ellmin:]
            ax.semilogx(ell_plot, average_ratio[i, j, ellmin:], label="Average ratio", lw=0.8, color="blue")
            ax.semilogx(
                ell_plot,
                average_ratio_filt[i, j, ellmin:],
                label="Average ratio filtered",
                lw=0.8,
                color="red",
            )
            ax.axhline(1, color="black", linestyle="--", lw=1, alpha=0.5)
            ax.set_xlabel(r"$\ell$", fontsize=11)
            ax.set_ylabel(r"$\langle C_\ell^{\rm mock}/(W_\ell^2C_\ell^{\rm true})\rangle$", fontsize=11)
            ax.set_title(f"Bin {i + 1}" if i == j else f"Bins ({i + 1},{j + 1})", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylim(ymin, ymax)
            if dashed_vert is not None:
                ax.axvline(dashed_vert, linestyle="dashed", color="black", lw=1, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

