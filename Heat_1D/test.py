from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from functions import (
    DEFAULT_DTYPE,
    SPACE_INTERVAL,
    TIME_INTERVAL,
    exact_heat_solution,
    set_seed,
)
from main import METHODS, PINN


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


METHOD_LABELS = {
    "Standard_PINN": "Standard PINN",
    "NTK_LM": "NTK-LM",
    "AL_PINN": "AL-PINN",
    "HARMONIC": "HARMONIC",
    "ConFIG": "ConFIG",
    "FTP": "FTP",
}
METHOD_COLORS = {
    "Standard_PINN": "#484878",
    "NTK_LM": "#6877A8",
    "AL_PINN": "#91A0C5",
    "HARMONIC": "#8172A4",
    "ConFIG": "#AF8FAB",
    "FTP": "#C25B7C",
}
METHOD_LINESTYLES = {
    "Standard_PINN": "-",
    "NTK_LM": "--",
    "AL_PINN": ":",
    "HARMONIC": "-.",
    "ConFIG": (0, (5, 2, 1, 2)),
    "FTP": "-",
}
WARMUP_COLOR = "#767676"
WARMUP_SHADE = "#EEEEF4"
LOSS_COLUMNS = {
    "condition": "loss_c",
    "pde": "loss_r",
    "total": "loss",
}
ELAPSED_TIME_PATTERN = re.compile(
    r"^\s*elapsed\s+time\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
CURVE_POINTS = 201
METRIC_NAMES = ["mse", "rmse", "mae", "max_abs_error", "relative_l2"]


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all heat-equation PINNs on one shared dense grid."
        )
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results") / "heat"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "heat" / "evaluation",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    parser.add_argument("--space-points", type=int, default=201)
    parser.add_argument("--time-points", type=int, default=201)
    return parser.parse_args()


def make_dense_grid(device, space_points, time_points):
    if space_points < 2 or time_points < 2:
        raise ValueError("space-points and time-points must both be at least 2")
    x = torch.linspace(
        *SPACE_INTERVAL,
        space_points,
        device=device,
        dtype=DEFAULT_DTYPE,
    )
    t = torch.linspace(
        *TIME_INTERVAL,
        time_points,
        device=device,
        dtype=DEFAULT_DTYPE,
    )
    return x, t, torch.cartesian_prod(x, t)


def calculate_metrics(prediction, exact):
    error = prediction - exact
    mse = torch.mean(error.square())
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(torch.mean(error.abs()).item()),
        "max_abs_error": float(error.abs().max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(exact)).item()
        ),
    }


def load_model_prediction(checkpoint, device, xt_dense):
    model = PINN().to(device)
    state_dict = torch.load(
        checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        prediction = model(xt_dense)

    expected_shape = (len(xt_dense),)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"{checkpoint} produced shape {tuple(prediction.shape)}; "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(prediction).all():
        raise ValueError(f"{checkpoint} produced non-finite predictions")
    return prediction


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def read_loss_table(path, columns):
    table = pd.read_excel(path)
    missing_columns = [column for column in columns if column not in table]
    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {', '.join(missing_columns)}"
        )
    table = table.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if table.empty:
        raise ValueError(f"{path} contains no loss records")
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    loss_columns = [column for column in columns if column.startswith("loss")]
    if (table[loss_columns].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(f"{path} contains negative loss values")
    return table


def read_elapsed_time(path):
    match = ELAPSED_TIME_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Could not parse elapsed time from {path}")
    elapsed_time = float(match.group(1))
    if not np.isfinite(elapsed_time) or elapsed_time <= 0.0:
        raise ValueError(f"Elapsed time in {path} must be finite and positive")
    return elapsed_time


def load_training_artifacts(args):
    required_files = []
    for method in args.methods:
        for seed in args.seeds:
            run_dir = args.results_root / method / f"seed_{seed}"
            required_files.extend(
                [
                    run_dir / "warmup_loss_c.xlsx",
                    run_dir / "loss_history.xlsx",
                    run_dir / "elapsed_time.txt",
                ]
            )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "The requested training comparison is incomplete; missing artifacts:\n"
            f"{missing_text}"
        )

    warmup_by_seed = {}
    method_histories = {method: [] for method in args.methods}
    time_records = []

    for method in args.methods:
        for seed in args.seeds:
            run_dir = args.results_root / method / f"seed_{seed}"
            warmup = read_loss_table(
                run_dir / "warmup_loss_c.xlsx", ["epoch", "loss_c"]
            )
            method_history = read_loss_table(
                run_dir / "loss_history.xlsx", ["loss_c", "loss_r", "loss"]
            )

            if seed in warmup_by_seed:
                reference = warmup_by_seed[seed].to_numpy(dtype=float)
                candidate = warmup.to_numpy(dtype=float)
                if reference.shape != candidate.shape or not np.allclose(
                    reference, candidate, rtol=1.0e-12, atol=0.0
                ):
                    raise ValueError(
                        "Warm-up histories differ between methods for "
                        f"seed {seed}; the comparison no longer has a shared start."
                    )
            else:
                warmup_by_seed[seed] = warmup

            method_histories[method].append(method_history)
            time_records.append(
                {
                    "method": method,
                    "seed": seed,
                    "elapsed_time_s": read_elapsed_time(
                        run_dir / "elapsed_time.txt"
                    ),
                }
            )

    warmup_histories = [warmup_by_seed[seed] for seed in args.seeds]
    return pd.DataFrame(time_records), warmup_histories, method_histories


def summarize_training_times(training_times):
    rows = []
    for method, group in training_times.groupby("method", sort=False):
        values = group["elapsed_time_s"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "runs": len(values),
                "mean_s": float(np.mean(values)),
                "std_s": float(np.std(values, ddof=0)),
                "median_s": float(np.median(values)),
                "q1_s": float(np.quantile(values, 0.25)),
                "q3_s": float(np.quantile(values, 0.75)),
                "min_s": float(np.min(values)),
                "max_s": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def summarize_loss_runs(histories, column):
    progress = np.linspace(0.0, 1.0, CURVE_POINTS)
    resampled = []
    for history in histories:
        values = history[column].to_numpy(dtype=float)
        source_progress = np.linspace(0.0, 1.0, len(values))
        positive_values = values[values > 0.0]
        if positive_values.size:
            floor = max(
                float(np.min(positive_values)) * 0.1,
                np.finfo(float).tiny,
            )
        else:
            floor = np.finfo(float).tiny
        log_values = np.log10(np.maximum(values, floor))
        resampled.append(
            np.power(10.0, np.interp(progress, source_progress, log_values))
        )

    values = np.vstack(resampled)
    return pd.DataFrame(
        {
            "progress": progress,
            "median": np.median(values, axis=0),
            "q1": np.quantile(values, 0.25, axis=0),
            "q3": np.quantile(values, 0.75, axis=0),
            "runs": values.shape[0],
        }
    )


def build_loss_curve_source_data(
    warmup_histories, method_histories, methods
):
    frames = []
    warmup_summary = summarize_loss_runs(warmup_histories, "loss_c")
    warmup_summary["phase_progress"] = warmup_summary["progress"] - 1.0
    warmup_summary["phase"] = "warm-up"
    warmup_summary["method"] = "Shared warm-up"
    for component in ("condition", "total"):
        component_frame = warmup_summary.copy()
        component_frame["component"] = component
        frames.append(component_frame)

    for method in methods:
        for component, column in LOSS_COLUMNS.items():
            summary = summarize_loss_runs(method_histories[method], column)
            summary["phase_progress"] = summary["progress"]
            summary["phase"] = "method"
            summary["method"] = method
            summary["component"] = component
            frames.append(summary)

    source_data = pd.concat(frames, ignore_index=True)
    return source_data.loc[
        :,
        [
            "phase",
            "method",
            "component",
            "progress",
            "phase_progress",
            "median",
            "q1",
            "q3",
            "runs",
        ],
    ]


def evaluate_models(args, device, xt_dense, exact):
    expected_checkpoints = [
        args.results_root / method / f"seed_{seed}" / "model.pt"
        for method in args.methods
        for seed in args.seeds
    ]
    missing = [path for path in expected_checkpoints if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "The requested comparison is incomplete; missing checkpoints:\n"
            f"{missing_text}"
        )

    records = []
    for method in args.methods:
        for seed in args.seeds:
            checkpoint = (
                args.results_root / method / f"seed_{seed}" / "model.pt"
            )
            prediction = load_model_prediction(
                checkpoint, device, xt_dense
            )

            record = {"method": method, "seed": seed}
            record.update(calculate_metrics(prediction, exact))
            records.append(record)
            print(
                f"{method:>13s} seed={seed:2d}: "
                f"relative L2={record['relative_l2']:.6e}, "
                f"RMSE={record['rmse']:.6e}"
            )
    return pd.DataFrame(records)


def summarize_metrics(metrics):
    rows = []
    for method, group in metrics.groupby("method", sort=False):
        row = {"method": method, "runs": len(group)}
        for metric_name in METRIC_NAMES:
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"] = float(np.std(values, ddof=0))
            row[f"{metric_name}_min"] = float(np.min(values))
            row[f"{metric_name}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def select_representative_models(metrics, methods):
    required_columns = {"method", "seed", *METRIC_NAMES}
    missing_columns = sorted(required_columns.difference(metrics.columns))
    if missing_columns:
        raise ValueError(
            "Metrics table is missing columns: "
            + ", ".join(missing_columns)
        )

    records = []
    for method in methods:
        subset = metrics.loc[metrics["method"] == method].copy()
        if subset.empty:
            raise ValueError(f"No evaluation metrics found for {method}")
        values = subset[METRIC_NAMES].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite evaluation metrics found for {method}")

        median_relative_l2 = float(np.median(subset["relative_l2"]))
        distances = np.abs(
            subset["relative_l2"].to_numpy(dtype=float)
            - median_relative_l2
        )
        minimum_distance = float(np.min(distances))
        tolerance = max(
            np.finfo(float).eps,
            abs(median_relative_l2) * 1.0e-12,
        )
        candidates = subset.loc[
            np.isclose(
                distances,
                minimum_distance,
                rtol=1.0e-12,
                atol=tolerance,
            )
        ].sort_values("seed", kind="mergesort")
        selected = candidates.iloc[0]

        record = {
            "method": method,
            "seed": int(selected["seed"]),
            "median_relative_l2": median_relative_l2,
            "distance_to_median": abs(
                float(selected["relative_l2"]) - median_relative_l2
            ),
        }
        for metric_name in METRIC_NAMES:
            record[metric_name] = float(selected[metric_name])
        records.append(record)
    return pd.DataFrame(records)


def evaluate_representative_fields(
    args, representatives, device, xt_dense, exact, space_points, time_points
):
    expected_size = space_points * time_points
    if len(xt_dense) != expected_size or exact.numel() != expected_size:
        raise ValueError(
            "Dense-grid dimensions do not match the evaluation tensors"
        )
    if not torch.isfinite(exact).all():
        raise ValueError("Exact solution contains non-finite values")

    fields = {}
    for row in representatives.itertuples(index=False):
        checkpoint = (
            args.results_root
            / row.method
            / f"seed_{row.seed}"
            / "model.pt"
        )
        prediction = load_model_prediction(checkpoint, device, xt_dense)
        error = prediction - exact
        fields[row.method] = {
            "seed": int(row.seed),
            "relative_l2": float(row.relative_l2),
            "prediction": prediction.reshape(
                space_points, time_points
            ).cpu().numpy(),
            "signed_error": error.reshape(
                space_points, time_points
            ).cpu().numpy(),
            "absolute_error": error.abs().reshape(
                space_points, time_points
            ).cpu().numpy(),
        }
    return fields


def plot_comparison(metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(metrics["method"]))
    colors = [METHOD_COLORS[method] for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    for index, method in enumerate(methods):
        subset = metrics[metrics["method"] == method].sort_values("seed")
        axes[0].plot(
            subset["seed"],
            subset["relative_l2"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=colors[index],
            label=method,
        )
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Accuracy for each run")

    distributions = [
        metrics.loc[metrics["method"] == method, "relative_l2"].to_numpy()
        for method in methods
    ]
    boxplot = axes[1].boxplot(
        distributions,
        tick_labels=methods,
        patch_artist=True,
        showmeans=True,
    )
    for index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(colors[index])
        box.set_alpha(0.45)
    for index, values in enumerate(distributions, start=1):
        axes[1].scatter(
            np.full(len(values), index),
            values,
            s=16,
            color=colors[index - 1],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    axes[1].set_ylabel(r"Relative $L_2$ error")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Accuracy distribution across seeds")

    png_path = output_dir / "method_accuracy_comparison.png"
    pdf_path = output_dir / "method_accuracy_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_field_grid(panel_count, max_columns):
    if panel_count < 1:
        raise ValueError("At least one field panel is required")
    columns = min(max_columns, panel_count)
    rows = (panel_count + columns - 1) // columns
    width = 7.2 if columns >= 3 else 2.4 * columns + 0.5
    height = max(2.6, 2.05 * rows + 0.5)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, height),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes.reshape(-1), rows, columns


def style_field_axis(
    ax, panel_index, panel_count, columns, x_values, t_values, title
):
    column = panel_index % columns
    if panel_index + columns >= panel_count:
        ax.set_xlabel(r"Space $x$")
    if column == 0:
        ax.set_ylabel(r"Time $t$")
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(float(t_values[0]), float(t_values[-1]))
    ax.set_title(title, fontsize=7, loc="left", fontweight="bold")
    add_panel_label(ax, chr(ord("a") + panel_index))


def plot_model_test_results(
    fields, methods, x_values, t_values, exact_grid, output_dir
):
    panels = [("Exact solution", exact_grid)]
    for method in methods:
        field = fields[method]
        panels.append(
            (
                f"{METHOD_LABELS[method]}\n"
                f"seed {field['seed']}, rel. $L_2$="
                f"{field['relative_l2']:.2e}",
                field["prediction"],
            )
        )

    all_values = np.concatenate(
        [np.asarray(values, dtype=float).ravel() for _, values in panels]
    )
    if not np.isfinite(all_values).all():
        raise ValueError("Prediction figure contains non-finite values")
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if value_min == value_max:
        padding = max(abs(value_min) * 0.05, 1.0)
        value_min -= padding
        value_max += padding

    panel_count = len(panels)
    fig, axes, _, columns = make_field_grid(panel_count, max_columns=4)
    image = None
    for panel_index, (title, values) in enumerate(panels):
        ax = axes[panel_index]
        image = ax.pcolormesh(
            x_values,
            t_values,
            np.asarray(values).T,
            cmap="viridis",
            shading="auto",
            rasterized=True,
            vmin=value_min,
            vmax=value_max,
        )
        style_field_axis(
            ax,
            panel_index,
            panel_count,
            columns,
            x_values,
            t_values,
            title,
        )

    unused_axes = axes[panel_count:]
    if len(unused_axes):
        unused_axes[0].text(
            0.05,
            0.55,
            "Representative run\nclosest to the median\nrelative $L_2$ error",
            transform=unused_axes[0].transAxes,
            ha="left",
            va="center",
            color="#4D4D4D",
            linespacing=1.5,
        )
    for ax in unused_axes:
        ax.set_axis_off()

    colorbar = fig.colorbar(
        image,
        ax=list(axes[:panel_count]),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(r"Solution $u(x,t)$")
    return save_publication_figure(
        fig, output_dir / "model_test_results"
    )


def make_absolute_error_norm(fields, methods):
    errors = [
        np.asarray(fields[method]["absolute_error"], dtype=float)
        for method in methods
    ]
    all_errors = np.concatenate([error.ravel() for error in errors])
    if not np.isfinite(all_errors).all():
        raise ValueError("Absolute-error figure contains non-finite values")
    if (all_errors < 0.0).any():
        raise ValueError("Absolute-error fields must be non-negative")

    maximum_error = float(np.max(all_errors))
    if maximum_error <= 0.0:
        return Normalize(vmin=0.0, vmax=1.0), 0.0, "linear"

    positive_errors = all_errors[all_errors > 0.0]
    error_floor = max(
        float(np.min(positive_errors)), maximum_error * 1.0e-6
    )
    if error_floor >= maximum_error:
        return (
            Normalize(vmin=0.0, vmax=maximum_error),
            0.0,
            "linear",
        )
    return (
        LogNorm(vmin=error_floor, vmax=maximum_error),
        error_floor,
        "logarithmic",
    )


def plot_model_absolute_error(
    fields, methods, x_values, t_values, output_dir
):
    norm, error_floor, scale_name = make_absolute_error_norm(
        fields, methods
    )
    panel_count = len(methods)
    fig, axes, _, columns = make_field_grid(panel_count, max_columns=3)
    image = None
    for panel_index, method in enumerate(methods):
        field = fields[method]
        values = field["absolute_error"]
        displayed_values = (
            np.maximum(values, error_floor)
            if scale_name == "logarithmic"
            else values
        )
        ax = axes[panel_index]
        image = ax.pcolormesh(
            x_values,
            t_values,
            displayed_values.T,
            cmap="magma",
            shading="auto",
            rasterized=True,
            norm=norm,
        )
        title = (
            f"{METHOD_LABELS[method]}\n"
            f"seed {field['seed']}, rel. $L_2$="
            f"{field['relative_l2']:.2e}"
        )
        style_field_axis(
            ax,
            panel_index,
            panel_count,
            columns,
            x_values,
            t_values,
            title,
        )

    for ax in axes[panel_count:]:
        ax.set_axis_off()
    colorbar = fig.colorbar(
        image,
        ax=list(axes[:panel_count]),
        fraction=0.025,
        pad=0.02,
        extend="min" if scale_name == "logarithmic" else "neither",
    )
    colorbar.set_label(r"Absolute error $|u_\theta-u|$")
    paths = save_publication_figure(
        fig, output_dir / "model_absolute_error"
    )
    return paths, error_floor, scale_name


def plot_time_panel(ax, training_times, methods):
    for position, method in enumerate(methods):
        subset = training_times.loc[
            training_times["method"] == method
        ].sort_values("seed")
        values = subset["elapsed_time_s"].to_numpy(dtype=float)
        color = METHOD_COLORS[method]
        jitter = (
            np.linspace(-0.13, 0.13, len(values))
            if len(values) > 1
            else np.zeros(1)
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.scatter(
            values,
            position + jitter,
            s=15,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.plot(
            [q1, q3],
            [position, position],
            color=color,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [median, median],
            [position - 0.17, position + 0.17],
            color="#272727",
            linewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            f"{median:.3g} s",
            (median, position),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=6,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_yticks(
        np.arange(len(methods)),
        [METHOD_LABELS[method] for method in methods],
    )
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.set_xlabel("Method training time (s)\n(shared warm-up excluded)")
    seed_count = training_times["seed"].nunique()
    ax.set_title(
        f"Training cost ($n$ = {seed_count} seeds)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", which="major", color="#D9D9D9", linewidth=0.6)
    add_panel_label(ax, "a")


def plot_loss_panel(ax, source_data, methods, component, title, label):
    ax.axvspan(-1.0, 0.0, color=WARMUP_SHADE, zorder=0)
    ax.axvline(0.0, color="#A0A0A0", linewidth=0.8, zorder=1)

    warmup = source_data.loc[
        (source_data["phase"] == "warm-up")
        & (source_data["component"] == component)
    ]
    if not warmup.empty:
        ax.fill_between(
            warmup["phase_progress"],
            warmup["q1"],
            warmup["q3"],
            color=WARMUP_COLOR,
            alpha=0.15,
            linewidth=0.0,
        )
        ax.plot(
            warmup["phase_progress"],
            warmup["median"],
            color=WARMUP_COLOR,
            linestyle="--",
            linewidth=1.2,
        )
    else:
        ax.text(
            -0.5,
            0.5,
            "Not evaluated\nin warm-up",
            transform=ax.get_xaxis_transform(),
            color=WARMUP_COLOR,
            fontsize=6,
            ha="center",
            va="center",
        )

    for method in methods:
        curve = source_data.loc[
            (source_data["phase"] == "method")
            & (source_data["method"] == method)
            & (source_data["component"] == component)
        ]
        color = METHOD_COLORS[method]
        ax.fill_between(
            curve["phase_progress"],
            curve["q1"],
            curve["q3"],
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
        ax.plot(
            curve["phase_progress"],
            curve["median"],
            color=color,
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.5 if method == "FTP" else 1.15,
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_yscale("log")
    ax.set_xticks(
        [-1.0, 0.0, 1.0],
        ["Warm-up\nstart", "Method\nstart", "Method\nend"],
    )
    ax.set_ylabel("Loss")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.6)
    add_panel_label(ax, label)


def save_publication_figure(fig, base_path):
    paths = [
        base_path.with_suffix(".svg"),
        base_path.with_suffix(".pdf"),
        base_path.with_suffix(".tiff"),
        base_path.with_suffix(".png"),
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_representative_field_data(
    fields, methods, x_values, t_values, exact_grid, output_dir
):
    source_data = {
        "x": np.asarray(x_values),
        "t": np.asarray(t_values),
        "exact": np.asarray(exact_grid),
    }
    for method in methods:
        source_data[f"{method}_prediction"] = fields[method]["prediction"]
        source_data[f"{method}_signed_error"] = fields[method][
            "signed_error"
        ]
        source_data[f"{method}_absolute_error"] = fields[method][
            "absolute_error"
        ]

    output_path = output_dir / "representative_field_data.npz"
    np.savez_compressed(output_path, **source_data)
    return output_path


def write_model_figure_notes(
    output_dir,
    representatives,
    space_points,
    time_points,
    error_floor,
    error_scale,
):
    selected_runs = ", ".join(
        f"{row.method}=seed {row.seed}"
        for row in representatives.itertuples(index=False)
    )
    if error_scale == "logarithmic":
        error_scale_text = (
            "shared logarithmic; values below "
            f"{error_floor:.6e} are shown at the color-scale floor"
        )
    else:
        error_scale_text = "shared linear (logarithmic scaling was degenerate)"

    notes_path = output_dir / "model_figure_notes.txt"
    notes_path.write_text(
        "Model test-result figure definitions\n"
        "====================================\n"
        "Core conclusion: all methods are evaluated against the same exact "
        "heat-equation solution, while their pointwise error distributions "
        "remain directly comparable.\n"
        "Representative run: the evaluated seed closest to each method's "
        "median relative L2 error; the smaller seed breaks numerical ties.\n"
        f"Selected runs: {selected_runs}.\n"
        f"Evaluation grid: {space_points} spatial points x "
        f"{time_points} temporal points.\n"
        "Prediction scale: shared linear limits across the exact solution "
        "and all representative predictions.\n"
        f"Absolute-error scale: {error_scale_text}.\n"
        "Processing: no spatial smoothing or local image adjustment is "
        "applied.\n",
        encoding="utf-8",
    )
    return notes_path


def plot_training_comparison(
    training_times, loss_curve_source_data, methods, output_dir
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.18, 1.18],
        hspace=0.48,
        wspace=0.55,
    )
    ax_time = fig.add_subplot(grid[:, 0])
    ax_total = fig.add_subplot(grid[0, 1:])
    ax_condition = fig.add_subplot(grid[1, 1])
    ax_pde = fig.add_subplot(grid[1, 2])

    plot_time_panel(ax_time, training_times, methods)
    plot_loss_panel(
        ax_total,
        loss_curve_source_data,
        methods,
        component="total",
        title="Total objective",
        label="b",
    )
    plot_loss_panel(
        ax_condition,
        loss_curve_source_data,
        methods,
        component="condition",
        title="Condition loss",
        label="c",
    )
    plot_loss_panel(
        ax_pde,
        loss_curve_source_data,
        methods,
        component="pde",
        title="PDE loss",
        label="d",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.8 if method == "FTP" else 1.3,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    handles.append(
        Patch(facecolor=WARMUP_SHADE, edgecolor="none", label="Shared warm-up")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.66, 0.995),
        ncol=min(4, len(handles)),
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.84)
    return save_publication_figure(
        fig, output_dir / "training_efficiency_and_loss"
    )


def write_training_figure_notes(output_dir, seeds):
    notes_path = output_dir / "training_figure_notes.txt"
    notes_path.write_text(
        "Training comparison definitions\n"
        "===============================\n"
        f"Independent runs: {len(seeds)} random seeds ({', '.join(map(str, seeds))}).\n"
        "Center and spread: median and interquartile range across seeds.\n"
        "Loss alignment: each phase is independently normalized to [0, 1]; "
        "curves are interpolated in log-loss space.\n"
        "Warm-up: optimizes the condition loss only. The warm-up total loss "
        "therefore equals the condition loss; PDE loss is not evaluated.\n"
        "Training time: elapsed method-training time saved by main.py; the "
        "shared warm-up is excluded.\n",
        encoding="utf-8",
    )
    return notes_path


def main():
    args = get_args()
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_times, warmup_histories, method_histories = (
        load_training_artifacts(args)
    )
    x, t, xt_dense = make_dense_grid(
        device, args.space_points, args.time_points
    )
    exact = exact_heat_solution(xt_dense)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_values = x.cpu().numpy()
    t_values = t.cpu().numpy()
    exact_grid = exact.reshape(len(x), len(t)).cpu().numpy()
    np.savez_compressed(
        args.output_dir / "dense_evaluation_grid.npz",
        x=x_values,
        t=t_values,
        exact=exact_grid,
    )

    metrics = evaluate_models(args, device, xt_dense, exact)
    representatives = select_representative_models(metrics, args.methods)
    representative_fields = evaluate_representative_fields(
        args,
        representatives,
        device,
        xt_dense,
        exact,
        len(x),
        len(t),
    )
    summary = summarize_metrics(metrics)
    training_time_summary = summarize_training_times(training_times)
    loss_curve_source_data = build_loss_curve_source_data(
        warmup_histories, method_histories, args.methods
    )
    metrics_path = args.output_dir / "metrics_by_seed.csv"
    summary_path = args.output_dir / "metrics_summary.csv"
    training_times_path = args.output_dir / "training_time_by_seed.csv"
    training_time_summary_path = (
        args.output_dir / "training_time_summary.csv"
    )
    loss_curve_source_path = args.output_dir / "training_loss_source_data.csv"
    representatives_path = args.output_dir / "representative_models.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    training_times.to_csv(training_times_path, index=False)
    training_time_summary.to_csv(training_time_summary_path, index=False)
    loss_curve_source_data.to_csv(loss_curve_source_path, index=False)
    representatives.to_csv(representatives_path, index=False)
    representative_field_path = save_representative_field_data(
        representative_fields,
        args.methods,
        x_values,
        t_values,
        exact_grid,
        args.output_dir,
    )
    figure_paths = plot_comparison(metrics, args.output_dir)
    model_result_figure_paths = plot_model_test_results(
        representative_fields,
        args.methods,
        x_values,
        t_values,
        exact_grid,
        args.output_dir,
    )
    (
        model_error_figure_paths,
        error_floor,
        error_scale,
    ) = plot_model_absolute_error(
        representative_fields,
        args.methods,
        x_values,
        t_values,
        args.output_dir,
    )
    model_notes_path = write_model_figure_notes(
        args.output_dir,
        representatives,
        len(x),
        len(t),
        error_floor,
        error_scale,
    )
    training_figure_paths = plot_training_comparison(
        training_times,
        loss_curve_source_data,
        args.methods,
        args.output_dir,
    )
    training_notes_path = write_training_figure_notes(
        args.output_dir, args.seeds
    )

    print("\nSummary (relative L2 mean +/- population std):")
    for row in summary.itertuples(index=False):
        print(
            f"{row.method:>13s}: {row.relative_l2_mean:.6e} "
            f"+/- {row.relative_l2_std:.6e}"
        )
    print(f"\nSaved per-run metrics: {metrics_path}")
    print(f"Saved aggregate metrics: {summary_path}")
    print(f"Saved comparison figures: {figure_paths[0]}, {figure_paths[1]}")
    print(f"Saved representative models: {representatives_path}")
    print(f"Saved representative field data: {representative_field_path}")
    print(
        "Saved model test-result figures: "
        + ", ".join(str(path) for path in model_result_figure_paths)
    )
    print(
        "Saved model absolute-error figures: "
        + ", ".join(str(path) for path in model_error_figure_paths)
    )
    print(f"Saved model figure notes: {model_notes_path}")
    print("\nTraining time (median [IQR], seconds; warm-up excluded):")
    for row in training_time_summary.itertuples(index=False):
        print(
            f"{row.method:>13s}: {row.median_s:.6g} "
            f"[{row.q1_s:.6g}, {row.q3_s:.6g}]"
        )
    print(f"Saved per-run training times: {training_times_path}")
    print(f"Saved training-time summary: {training_time_summary_path}")
    print(f"Saved plotted loss source data: {loss_curve_source_path}")
    print(
        "Saved training comparison figures: "
        + ", ".join(str(path) for path in training_figure_paths)
    )
    print(f"Saved training figure notes: {training_notes_path}")


if __name__ == "__main__":
    main()
