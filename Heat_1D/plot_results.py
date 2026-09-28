from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call

from FTP_pinn import ParameterDict, TrainingResult


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8


BLUE_MAIN = "#0F4D92"
BLUE_SECONDARY = "#3775BA"
RED_STRONG = "#B64342"
NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#767676"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_publication_figure(fig: plt.Figure, base_path: Path) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        base_path.with_suffix(".svg"),
        base_path.with_suffix(".pdf"),
        base_path.with_suffix(".tiff"),
        base_path.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def evaluate_field(
    model: torch.nn.Module,
    parameters: ParameterDict,
    buffers: dict[str, torch.Tensor],
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    space_points: int = 101,
    time_points: int = 101,
) -> dict[str, np.ndarray]:
    x = torch.linspace(0.0, 1.0, space_points, device=device)
    t = torch.linspace(0.0, 1.0, time_points, device=device)
    xt = torch.cartesian_prod(x, t)
    with torch.no_grad():
        prediction = functional_call(model, (parameters, buffers), (xt,))
        exact = exact_solution(xt)

    prediction_grid = prediction.reshape(space_points, time_points)
    exact_grid = exact.reshape(space_points, time_points)
    error_grid = prediction_grid - exact_grid
    return {
        "x": x.cpu().numpy(),
        "t": t.cpu().numpy(),
        "prediction": prediction_grid.cpu().numpy(),
        "exact": exact_grid.cpu().numpy(),
        "error": error_grid.cpu().numpy(),
    }


def write_field_source_data(field: dict[str, np.ndarray], output_dir: Path) -> None:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "field_data.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "t", "prediction", "exact", "signed_error", "absolute_error"])
        for x_index, x_value in enumerate(field["x"]):
            for t_index, t_value in enumerate(field["t"]):
                prediction = field["prediction"][x_index, t_index]
                exact = field["exact"][x_index, t_index]
                error = field["error"][x_index, t_index]
                writer.writerow(
                    [x_value, t_value, prediction, exact, error, abs(error)]
                )


def plot_solution_results(
    field: dict[str, np.ndarray], output_dir: Path
) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 4.6), layout="constrained")
    grid = fig.add_gridspec(
        3, 2, width_ratios=[1.55, 1.0], hspace=0.12, wspace=0.12
    )
    ax_field = fig.add_subplot(grid[:, 0])
    slice_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]

    image = ax_field.pcolormesh(
        field["x"],
        field["t"],
        field["prediction"].T,
        cmap="viridis",
        shading="auto",
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax_field, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Predicted $u_\theta(x,t)$")
    ax_field.set_xlabel(r"Space $x$")
    ax_field.set_ylabel(r"Time $t$")
    ax_field.set_title("FTP-PINN solution field", loc="left", fontweight="bold")
    add_panel_label(ax_field, "a")

    selected_times = [0.0, 0.5, 1.0]
    for panel_index, (ax, selected_time) in enumerate(
        zip(slice_axes, selected_times), start=1
    ):
        time_index = int(np.argmin(np.abs(field["t"] - selected_time)))
        ax.plot(
            field["x"],
            field["exact"][:, time_index],
            color=NEUTRAL_DARK,
            linewidth=1.4,
            linestyle="--",
            label="Exact",
        )
        ax.plot(
            field["x"],
            field["prediction"][:, time_index],
            color=BLUE_MAIN,
            linewidth=1.6,
            label="FTP-PINN",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.05)
        if panel_index == 2:
            ax.set_ylabel(r"$u(x,t)$")
        ax.text(
            0.98,
            0.76,
            rf"$t={selected_time:.1f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=NEUTRAL_DARK,
        )
        if panel_index == 1:
            ax.legend(loc="upper left", ncol=2, handlelength=2.0)
        if panel_index < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Space $x$")
        add_panel_label(ax, chr(ord("a") + panel_index))

    return save_publication_figure(fig, output_dir / "ftp_pinn_results")


def plot_error_results(
    field: dict[str, np.ndarray], output_dir: Path
) -> list[Path]:
    absolute_error = np.abs(field["error"])
    rmse_by_time = np.sqrt(np.mean(field["error"] ** 2, axis=0))
    max_error_by_time = np.max(absolute_error, axis=0)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "error_by_time.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["t", "spatial_rmse", "maximum_absolute_error"])
        writer.writerows(zip(field["t"], rmse_by_time, max_error_by_time))

    fig = plt.figure(figsize=(7.2, 4.2), layout="constrained")
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.45, 1.0], hspace=0.18, wspace=0.18
    )
    ax_error = fig.add_subplot(grid[:, 0])
    ax_rmse = fig.add_subplot(grid[0, 1])
    ax_max = fig.add_subplot(grid[1, 1])

    image = ax_error.pcolormesh(
        field["x"],
        field["t"],
        absolute_error.T,
        cmap="magma",
        shading="auto",
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax_error, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Absolute error $|u_\theta-u|$")
    ax_error.set_xlabel(r"Space $x$")
    ax_error.set_ylabel(r"Time $t$")
    ax_error.set_title("Pointwise error", loc="left", fontweight="bold")
    add_panel_label(ax_error, "a")

    ax_rmse.plot(
        field["t"], 1.0e3 * rmse_by_time, color=BLUE_MAIN, linewidth=1.7
    )
    ax_rmse.set_ylabel(r"RMSE ($10^{-3}$)")
    ax_rmse.tick_params(labelbottom=False)
    ax_rmse.set_xlim(0.0, 1.0)
    add_panel_label(ax_rmse, "b")

    ax_max.plot(
        field["t"],
        1.0e3 * max_error_by_time,
        color=RED_STRONG,
        linewidth=1.7,
    )
    ax_max.set_xlabel(r"Time $t$")
    ax_max.set_ylabel(r"Max. error ($10^{-3}$)")
    ax_max.set_xlim(0.0, 1.0)
    add_panel_label(ax_max, "c")

    return save_publication_figure(fig, output_dir / "ftp_pinn_error")


def plot_loss_curves(
    iterations: np.ndarray,
    condition_loss: np.ndarray,
    physics_loss: np.ndarray,
    total_loss: np.ndarray,
    feasibility_tolerance: float,
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 4.5), layout="constrained")
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.55, 1.0], hspace=0.10, wspace=0.10
    )
    ax_pde = fig.add_subplot(grid[:, 0])
    ax_total = fig.add_subplot(grid[0, 1])
    ax_condition = fig.add_subplot(grid[1, 1])

    ax_pde.semilogy(
        iterations, physics_loss, color=BLUE_MAIN, linewidth=1.8
    )
    ax_pde.set_xlabel("FTP iteration")
    ax_pde.set_ylabel(r"PDE loss $L_r$")
    ax_pde.set_title(r"PDE loss $L_r$", loc="left", fontweight="bold")
    add_panel_label(ax_pde, "a")

    ax_total.semilogy(
        iterations, total_loss, color=NEUTRAL_DARK, linewidth=1.5
    )
    ax_total.set_ylabel(r"Total loss $L_c+L_r$")
    ax_total.set_title("Total loss", loc="left", fontweight="bold")
    ax_total.tick_params(labelbottom=False)
    relative_gap = np.max(
        condition_loss / np.maximum(physics_loss, np.finfo(float).tiny)
    )
    if relative_gap > 0.0:
        relative_exponent = int(np.floor(np.log10(relative_gap)))
        relative_mantissa = relative_gap / 10.0**relative_exponent
        relative_gap_label = (
            rf"{relative_mantissa:.1f}\times10^{{{relative_exponent}}}"
        )
    else:
        relative_gap_label = "0"
    ax_total.text(
        0.97,
        0.08,
        rf"max $L_c/L_r={relative_gap_label}$",
        transform=ax_total.transAxes,
        ha="right",
        va="bottom",
        color=NEUTRAL_MID,
    )
    add_panel_label(ax_total, "b")

    positive_condition_loss = np.maximum(
        condition_loss, np.finfo(float).tiny
    )
    ax_condition.semilogy(
        iterations,
        positive_condition_loss,
        color=RED_STRONG,
        linewidth=1.3,
    )
    tolerance_loss = 0.5 * feasibility_tolerance**2
    ax_condition.axhline(
        tolerance_loss,
        color=NEUTRAL_MID,
        linewidth=1.0,
        linestyle=":",
    )
    ax_condition.text(
        0.97,
        tolerance_loss,
        r"$\frac{1}{2}\tau_{\mathrm{feas}}^2$",
        transform=ax_condition.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color=NEUTRAL_MID,
    )
    ax_condition.set_xlabel("FTP iteration")
    ax_condition.set_ylabel(r"Condition loss $L_c$")
    ax_condition.set_title("Condition feasibility", loc="left", fontweight="bold")
    add_panel_label(ax_condition, "c")

    return save_publication_figure(fig, output_dir / "ftp_pinn_losses")


def plot_loss_history(
    training: TrainingResult,
    feasibility_tolerance: float,
    output_dir: Path,
) -> list[Path]:
    iterations = np.array(
        [0] + [record.iteration + 1 for record in training.history], dtype=int
    )
    condition_loss = np.array(
        [float(training.initial_state.loss_c.item())]
        + [record.condition_loss for record in training.history]
    )
    physics_loss = np.array(
        [float(training.initial_state.loss_r.item())]
        + [record.physics_loss for record in training.history]
    )
    total_loss = np.array(
        [
            float(
                (training.initial_state.loss_c + training.initial_state.loss_r).item()
            )
        ]
        + [record.total_loss for record in training.history]
    )

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "loss_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["iteration", "condition_loss", "pde_loss", "total_loss"]
        )
        writer.writerows(zip(iterations, condition_loss, physics_loss, total_loss))

    return plot_loss_curves(
        iterations,
        condition_loss,
        physics_loss,
        total_loss,
        feasibility_tolerance,
        output_dir,
    )


def write_figure_notes(output_dir: Path) -> Path:
    note_path = output_dir / "figure_notes.txt"
    note_path.write_text(
        "Core conclusion: FTP-PINN approximates the heat-equation solution while "
        "maintaining the discrete conditions and reducing the PDE residual.\n"
        "Backend: Python/matplotlib only.\n"
        "Total loss definition: L_total = L_c + L_r.\n"
        "Uncertainty: one random seed; no confidence interval is claimed.\n"
        "Validation: independent dense spatial-temporal grids are used for the "
        "solution and error figures.\n",
        encoding="utf-8",
    )
    return note_path


def generate_all_figures(
    model: torch.nn.Module,
    parameters: ParameterDict,
    buffers: dict[str, torch.Tensor],
    training: TrainingResult,
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    feasibility_tolerance: float,
    device: torch.device,
    output_dir: str | Path = "figures",
) -> dict[str, list[Path] | Path]:
    output_path = Path(output_dir)
    field = evaluate_field(
        model, parameters, buffers, exact_solution, device
    )
    write_field_source_data(field, output_path)
    return {
        "results": plot_solution_results(field, output_path),
        "error": plot_error_results(field, output_path),
        "losses": plot_loss_history(
            training, feasibility_tolerance, output_path
        ),
        "notes": write_figure_notes(output_path),
    }
