import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import torch

from main import PINN, METHODS, RESULTS_ROOT, condition_residual, physics_residual
from functions import (
    PRESSURE_POINT, exact_kovasznay_solution, cell_center_grid,
    boundary_grid, make_points, set_seed,
)



LABELS = {"Standard_PINN": "Standard PINN", "NTK_LM": "NTK-LM", "AL_PINN": "AL-PINN",
          "HARMONIC": "HARMONIC", "ConFIG": "ConFIG", "FTP": "FTP"}
COLORS = dict(zip(METHODS, ("#767676", "#3775BA", "#B64342", "#42949E", "#9A4D8E", "#0F4D92")))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})
VELOCITY_L2_TOL = 1e-3
BOUNDARY_RMS_TOL = 1e-4


def evaluate_model(model, device, grid_size=81, pde_grid_size=33):
    xy = cell_center_grid(device, grid_size)
    boundary = boundary_grid(device, grid_size, cell_centered=True)
    xy_pressure = torch.tensor([PRESSURE_POINT], device=device)
    with torch.no_grad():
        exact = exact_kovasznay_solution(xy)
        predicted = model(xy)
        difference = predicted - exact
        velocity_l2 = (difference[:, :2].norm() / exact[:, :2].norm()).item()
        relative_l2 = (difference.norm(dim=0) / exact.norm(dim=0)).cpu().tolist()
        boundary_rms = (model(boundary)[:, :2] - exact_kovasznay_solution(boundary)[:, :2]).square().mean().sqrt().item()
        pressure_reference_error = (model(xy_pressure)[:, 2] - exact_kovasznay_solution(xy_pressure)[:, 2]).abs().item()
    # 分批计算二阶导数，避免把独立评估网格放入训练。
    pde_points = cell_center_grid(device, pde_grid_size)
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    residuals = []
    for batch in pde_points.split(256):
        residuals.append(physics_residual(model, parameters, buffers, batch).detach().reshape(-1, 3))
    residual = torch.cat(residuals)
    pde_rms = residual.square().mean(dim=0).sqrt().cpu().tolist()
    metrics = {
        "velocity_relative_l2": velocity_l2,
        "u_relative_l2": relative_l2[0], "v_relative_l2": relative_l2[1],
        "pressure_relative_l2": relative_l2[2],
        "velocity_rmse": difference[:, :2].square().mean().sqrt().item(),
        "velocity_max_abs": difference[:, :2].abs().max().item(),
        "pressure_max_abs": difference[:, 2].abs().max().item(),
        "boundary_velocity_rms": boundary_rms,
        "pressure_reference_abs": pressure_reference_error,
        "momentum_x_rms": pde_rms[0], "momentum_y_rms": pde_rms[1],
        "continuity_rms": pde_rms[2],
        "pde_rms": residual.square().mean().sqrt().item(),
        "evaluation_grid_size": grid_size, "pde_grid_size": pde_grid_size,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite independent evaluation metric")
    fields = {"exact": exact.cpu().numpy(), "predicted": predicted.cpu().numpy()}
    return metrics, fields


def summarize_metrics(frame, methods, columns):
    """每个方法、每项指标一行；标准差使用样本标准差（ddof=1）。"""
    rows = []
    for method in methods:
        completed = frame[(frame["method"] == method) & (frame["status"] == "completed")]
        for column in columns:
            values = completed[column].dropna()
            rows.append({
                "method": method, "metric": column, "count": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
                "median": values.median(), "q25": values.quantile(0.25),
                "q75": values.quantile(0.75), "p90": values.quantile(0.90),
                "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows)


def save_figure(fig, output):
    # 与 Helmholtz_3D/test.py 一致，同时保存位图和矢量图。
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison(frame, methods, panels, filename, seed_count, title):
    """同一幅图汇总全部种子：散点为运行结果，粗线为 IQR，黑线为中位数。"""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (column, label) in zip(axes.flat, panels):
        labels = []
        for index, method in enumerate(methods):
            values = frame.loc[
                (frame["method"] == method) & (frame["status"] == "completed"), column
            ].dropna().to_numpy()
            labels.append(f"{LABELS[method]}\n{len(values)}/{seed_count}")
            if len(values):
                x = index + np.linspace(-0.17, 0.17, len(values))
                ax.scatter(x, np.maximum(values, 1e-20), color=COLORS[method], s=20, alpha=0.75)
                q1, median, q3 = np.maximum(np.quantile(values, [0.25, 0.5, 0.75]), 1e-20)
                ax.plot([index, index], [q1, q3], color=COLORS[method], lw=5, alpha=0.5)
                ax.plot([index - 0.18, index + 0.18], [median, median], color="black", lw=1.2)
        ax.set_xticks(range(len(methods)), labels, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel=label, yscale="log", xlim=(-0.5, len(methods) - 0.5))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{title}\nDots = completed runs; bars = IQR; black line = median; labels = completed/requested")
    save_figure(fig, filename)


def plot_fields(field_sums, exact, args, output):
    """适配三维解场图：每行对应 u/v/p，各列为不同方法的跨种子均值。"""
    methods = [method for method in args.methods if field_sums[method]["count"]]
    if not methods:
        return
    n = args.grid_size
    axis = (np.arange(n) + 0.5) / n
    arrays = {"axis": axis, "exact": exact.reshape(n, n, 3)}
    for method in methods:
        data = field_sums[method]
        mean = data["prediction"] / data["count"]
        variance = np.maximum(data["prediction_squared"] / data["count"] - mean**2, 0.0)
        arrays[f"{method}_mean"] = mean.reshape(n, n, 3)
        arrays[f"{method}_mean_abs_error"] = (data["abs_error"] / data["count"]).reshape(n, n, 3)
        # 跨种子总体标准差，只作离散度数据，不是均值置信区间。
        arrays[f"{method}_std"] = np.sqrt(variance).reshape(n, n, 3)
        arrays[f"{method}_count"] = np.array(data["count"])
    np.savez_compressed(output / "mean_fields.npz", **arrays)

    for error_plot in (False, True):
        columns = methods if error_plot else ["exact", *methods]
        fig, axes = plt.subplots(3, len(columns), figsize=(2.15 * len(columns), 7.2),
                                 squeeze=False, constrained_layout=True)
        for row, component in enumerate(("u", "v", "p")):
            values = [
                arrays["exact"][:, :, row] if method == "exact" else
                arrays[f"{method}_{'mean_abs_error' if error_plot else 'mean'}"][:, :, row]
                for method in columns
            ]
            if error_plot:
                norm = LogNorm(vmin=1e-8, vmax=max(1e-7, max(float(value.max()) for value in values)))
            else:
                low, high = min(float(value.min()) for value in values), max(float(value.max()) for value in values)
                norm = Normalize(vmin=low, vmax=high if high > low else low + 1e-12)
            for col, (method, value) in enumerate(zip(columns, values)):
                ax = axes[row, col]
                im = ax.pcolormesh(
                    axis, axis, np.maximum(value, 1e-8).T if error_plot else value.T,
                    cmap="magma" if error_plot else "RdBu_r", norm=norm,
                    shading="auto", rasterized=True,
                )
                ax.set_aspect("equal")
                if row == 0:
                    ax.set_title("Exact" if method == "exact" else
                                 f"{LABELS[method]}\nn={field_sums[method]['count']}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{component}\ny")
                if row == 2:
                    ax.set_xlabel("x")
            fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.85,
                         label=f"Mean |{component} - {component}*|" if error_plot else component)
        fig.suptitle("Across completed seeds: " + (
            "mean absolute errors (color floor 1e-8)" if error_plot else "mean velocity and pressure fields"))
        save_figure(fig, output / ("mean_errors" if error_plot else "mean_solutions"))


def plot_histories(args, output):
    """与三维脚本一样，每个方法一个面板，在面板内叠加所有可用种子的历史。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    counts = []
    for method, ax in zip(args.methods, axes.ravel()):
        available = 0
        warmup_count = 0
        record_counts = []
        for seed in args.seeds:
            run_dir = args.results_root / method / f"seed_{seed}"
            csv_path, xlsx_path = run_dir / "loss_history.csv", run_dir / "loss_history.xlsx"
            if csv_path.exists():
                history = pd.read_csv(csv_path)
            elif xlsx_path.exists():
                history = pd.read_excel(xlsx_path)
            else:
                continue
            values = history["loss"].to_numpy(dtype=float)
            if not len(values) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid loss history: {run_dir}")
            ax.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-20),
                    color=COLORS[method], alpha=0.45, lw=0.8)
            record_counts.append(len(values))
            available += 1
            warmup_path = run_dir / "warmup_loss_c.xlsx"
            if method == "FTP" and warmup_path.exists():
                warmup_count += 1
        ax.set(yscale="log", xlabel="Recorded loss evaluation", ylabel="Total loss")
        ax.set_title(f"{LABELS[method]}: {available}/{len(args.seeds)} histories" +
                     (" (after warmup)" if method == "FTP" else ""))
        ax.grid(alpha=0.2)
        counts.append({
            "method": method, "requested": len(args.seeds), "available_histories": available,
            "missing_histories": len(args.seeds) - available,
            "records_min": min(record_counts) if record_counts else np.nan,
            "records_median": np.median(record_counts) if record_counts else np.nan,
            "records_max": max(record_counts) if record_counts else np.nan,
            "ftp_warmup_histories": warmup_count,
        })
    for ax in axes.ravel()[len(args.methods):]:
        ax.set_visible(False)
    fig.suptitle("All available seeds per method; evaluation counts are not equal-time axes")
    save_figure(fig, output / "training_losses")
    pd.DataFrame(counts).to_csv(output / "loss_record_counts.csv", index=False)


def compare_ftp(frame, summary, seeds):
    """沿用三维算例的优势判据；种子不齐全时只报告已有配对汇总。"""
    verdict = {
        "formal_seed_set": sorted(seeds) == list(range(10)), "criterion_met": False,
        "criterion": "All 10 FTP runs complete; lower median and P90 velocity relative L2 "
                     "than the best complete baseline, with at least 8/10 paired wins",
    }
    completed = frame[frame["status"] == "completed"]
    ftp = completed[completed["method"] == "FTP"]
    candidates = summary[
        (summary["method"] != "FTP") & (summary["metric"] == "velocity_relative_l2")
        & (summary["count"] == len(seeds))
    ]
    if ftp.empty or candidates.empty:
        verdict["reason"] = "No completed FTP runs or no complete baseline"
        return verdict
    best = candidates.sort_values(["median", "method"]).iloc[0]
    baseline = completed[completed["method"] == best["method"]]
    pairs = ftp[["seed", "velocity_relative_l2"]].merge(
        baseline[["seed", "velocity_relative_l2"]], on="seed", suffixes=("_ftp", "_baseline"))
    ftp_values, baseline_values = pairs["velocity_relative_l2_ftp"], pairs["velocity_relative_l2_baseline"]
    fmedian, fp90 = np.quantile(ftp_values, [0.5, 0.9])
    bmedian, bp90 = np.quantile(baseline_values, [0.5, 0.9])
    wins = int((ftp_values < baseline_values).sum())
    verdict.update({
        "baseline": best["method"], "paired_runs": len(pairs), "ftp_wins": wins,
        "ftp_paired_median": float(fmedian), "baseline_paired_median": float(bmedian),
        "ftp_paired_p90": float(fp90), "baseline_paired_p90": float(bp90),
        "baseline_complete_median": float(best["median"]),
        "baseline_complete_p90": float(best["p90"]),
    })
    complete = verdict["formal_seed_set"] and len(pairs) == 10
    verdict["criterion_met"] = bool(complete and wins >= 8 and fmedian < bmedian and fp90 < bp90)
    verdict["reason"] = ("Complete formal comparison" if complete else
                         "Incomplete formal seed set; paired statistics are descriptive only")
    return verdict


def write_report(summary, common_summary, status, common_seeds, verdict, args, filename):
    def comparison_table(statistics):
        lines = [
            "| 方法 | 样本数 | 速度相对 L2 | 压力相对 L2 | 边界速度 RMS | PDE RMS | 总耗时/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in args.methods:
            values = statistics[statistics["method"] == method].set_index("metric")
            row = [LABELS[method], str(int(values.loc["velocity_relative_l2", "count"]))]
            for metric in ("velocity_relative_l2", "pressure_relative_l2",
                           "boundary_velocity_rms", "pde_rms", "elapsed_seconds"):
                mean, std = values.loc[metric, ["mean", "std"]]
                row.append(f"{mean:.3e} ± {std:.3e}" if pd.notna(std)
                           else (f"{mean:.3e}（单样本）" if pd.notna(mean) else "无结果"))
            lines.append("| " + " | ".join(row) + " |")
        return lines

    lines = [
        "# 二维 Navier–Stokes：随机种子综合对比", "",
        "参照三维 Helmholtz 的统计和展示方式，求解与误差图改为 u、v、p 的跨种子汇总。", "",
        f"计划种子：{', '.join(map(str, args.seeds))}，每种方法 {len(args.seeds)} 次运行。",
        f"独立解场网格 {args.grid_size}×{args.grid_size}；每边 {args.grid_size} 个中心点；"
        f"PDE 网格 {args.pde_grid_size}×{args.pde_grid_size}。", "",
        "## 完成情况、独立精度与尾部误差", "",
        "| 方法 | 完成/计划 | 失败 | 缺失/未完成 | 损失达标 | 速度精度达标 | 速度 L2 中位数 | 速度 L2 P90 | 时间中位数/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in status.to_dict("records"):
        values = summary[summary["method"] == row["method"]].set_index("metric")
        numbers = [values.loc["velocity_relative_l2", "median"],
                   values.loc["velocity_relative_l2", "p90"], values.loc["elapsed_seconds", "median"]]
        formatted = [f"{value:.4g}" if pd.notna(value) else "—" for value in numbers]
        lines.append(
            f"| {LABELS[row['method']]} | {row['completed']}/{row['requested']} | {row['failed']} | "
            f"{row['missing_or_incomplete']} | {row['reached_loss_tol']} | {row['accuracy_pass']} | "
            + " | ".join(formatted) + " |"
        )
    lines += [
        "", f"参照三维判据，将独立速度精度达标定义为：速度向量相对 L2 ≤ {VELOCITY_L2_TOL:g} "
        f"且边界速度 RMS ≤ {BOUNDARY_RMS_TOL:g}；这不是压力精度的达标判据。"
        "损失达标依据保存模型重新计算的训练损失。", "",
        "## 各方法已完成结果：均值 ± 样本标准差", "",
        "统计仅覆盖完成并通过模型重载检查的运行。失败和缺失保留在计划数及达标率分母中，"
        "不为其填入零误差或虚构耗时。", "",
    ]
    lines += comparison_table(summary)
    lines += ["", "## 相同种子集合上的综合对比", ""]
    if common_seeds:
        lines += [f"共同完成集合：{', '.join(map(str, common_seeds))}。"
                  "下表所有方法使用相同种子集合；上表仍报告各方法全部完成结果及失败数。", ""]
        lines += comparison_table(common_summary)
    else:
        lines += ["没有所有方法均完成的共同种子。"]
    lines += [
        "", "## FTP 优势判据", "",
        "参照三维脚本：正式种子 0–9 全部完成，FTP 的速度相对 L2 中位数和 P90 "
        "均低于最佳完整基线，且至少 8/10 次配对获胜。该判据只针对速度相对 L2。",
    ]
    if "baseline" in verdict:
        lines += [
            f"以完整种子集速度相对 L2 中位数最低的方法 {LABELS[verdict['baseline']]} 为基线。"
            f"已有共同完成运行中，FTP 获胜 {verdict['ftp_wins']}/{verdict['paired_runs']}。",
            f"在已有配对集合上，FTP/基线的中位数为 {verdict['ftp_paired_median']:.4g}/"
            f"{verdict['baseline_paired_median']:.4g}，P90 为 {verdict['ftp_paired_p90']:.4g}/"
            f"{verdict['baseline_paired_p90']:.4g}。",
        ]
        if verdict["formal_seed_set"] and verdict["paired_runs"] == 10:
            lines.append("正式优势判据" + ("达到。" if verdict["criterion_met"] else "未达到。"))
        else:
            lines.append("正式种子结果不完整，当前配对统计仅作描述，不能确认完整十种子的优势判据。")
    else:
        lines.append("没有足够的已完成 FTP 运行或完整基线，无法判断。")
    if not args.no_plots:
        lines += [
            "", "## 综合图", "",
            "![误差分布](comparison.png)", "",
            "![最终损失与训练成本](training_comparison.png)", "",
            "![各方法多种子训练历史](training_losses.png)", "",

        ]
    if not args.no_plots and status["completed"].sum() > 0:
        lines += ["![跨种子平均解](mean_solutions.png)", "",
                  "![跨种子平均绝对误差](mean_errors.png)", ""]
    lines += [
        "", "## 图表与统计口径", "",
        "- 分布图：散点为每次完成运行，粗线为 25%–75% 分位区间，黑线为中位数。"
        "零值只在对数图中显示为 1e-20，表格保持原值。",
        "- 解场按各方法完成种子求平均，图头标注样本数。误差图先对每个种子取绝对误差再求平均，"
        "避免正负误差在平均解中抵消；各方法同一分量共用色标。图中误差下限为 1e-8，NPZ 不截断。",
        "- mean_fields.npz 保存解析解、各方法平均解、平均绝对误差、"
        "跨种子总体标准差和样本数；该标准差不是均值置信区间。",
        "- 训练历史每个方法一个面板，叠加请求种子中所有可用历史，不生成单个种子的图。"
        "如失败运行保留了历史，也会显示；缺失数量另存 loss_record_counts.csv。",
        "- 训练曲线横轴为原始损失记录序号，L-BFGS 包含线搜索评价，"
        "各方法不能按横轴解读为等迭代成本或等时间。FTP 曲线从条件预训练之后开始。",
        "- 时间包含 FTP 预训练和主体训练，取原运行记录，排除独立评估和文件保存；"
        "各方法仍使用原学习率和训练协议，不属于等时间或等预训练预算消融。",
        "- metrics_summary.csv 和 common_seeds_summary.csv 含样本数、均值、样本标准差、"
        "中位数、Q25、Q75、P90、最小值和最大值；run_status_summary.csv 含完成、失败与达标率。",
        "- metrics_by_seed.csv 仅作为底层复核数据，报告和图按方法综合呈现。"
        "ftp_advantage.json 保存汇总判据，evaluation_config.json 保存本次评估设置。",
        "- 图形同时提供 PNG、PDF、SVG。",
        "",
    ]
    filename.write_text("\n".join(lines), encoding="utf-8")


def get_args():
    parser = argparse.ArgumentParser(description="Compare methods over ten Kovasznay random seeds.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        help="Summary output directory; default: <results-root>/evaluation")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--pde-grid-size", type=int, default=33)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.grid_size < 2 or args.pde_grid_size < 2:
        parser.error("evaluation grid sizes must be at least 2")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("duplicate seeds or methods are not allowed")
    return args


def main():
    args = get_args()
    set_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir or args.results_root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    field_sums = {method: {"count": 0} for method in args.methods}
    exact_field = None
    common_config = None
    config_keys = (
        "problem", "reynolds_number", "domain", "steps", "width", "depth", "loss_tol",
        "sampling_sha256", "interior_points", "boundary_points_per_edge",
        "config_learning_rate", "config_final_learning_rate",
    )
    for seed in args.seeds:
        initial_hash = None
        for method in args.methods:
            run_dir = args.results_root / method / f"seed_{seed}"
            result_file = run_dir / "run_result.json"
            record = {"method": method, "seed": seed}
            if not result_file.exists():
                records.append({**record, "status": "missing_or_running"})
                continue
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result["status"] != "completed":
                records.append({**record, "status": result["status"]})
                continue
            if not all((run_dir / name).exists() for name in ("run_config.json", "model.pt")):
                records.append({**record, "status": "missing_artifacts"})
                continue
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            # 不同种子允许初始化不同，其余实验配置必须一致。
            signature = {key: config[key] for key in config_keys}
            if common_config is not None and signature != common_config:
                raise ValueError(f"Inconsistent comparison configuration at {run_dir}")
            common_config = signature
            if initial_hash is not None and config["initial_parameters_sha256"] != initial_hash:
                raise ValueError(f"Inconsistent initial model for seed {seed}: {run_dir}")
            initial_hash = config["initial_parameters_sha256"]
            model = PINN(config["width"], config["depth"]).to(device)
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
            model.eval()
            metrics, fields = evaluate_model(model, device, args.grid_size, args.pde_grid_size)
            if not args.no_plots:
                data = field_sums[method]
                exact_field = fields["exact"]
                prediction = fields["predicted"]
                if data["count"] == 0:
                    data.update({key: np.zeros_like(prediction) for key in
                                 ("prediction", "prediction_squared", "abs_error")})
                data["prediction"] += prediction
                data["prediction_squared"] += prediction**2
                data["abs_error"] += np.abs(prediction - exact_field)
                data["count"] += 1
            boundary, interior = make_points(device, config["interior_points"], config["boundary_points_per_edge"])
            anchor = torch.tensor([PRESSURE_POINT], device=device)
            parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
            c = condition_residual(model, parameters, buffers, boundary,
                                   exact_kovasznay_solution(boundary), anchor, exact_kovasznay_solution(anchor)[:, 2])
            r = physics_residual(model, parameters, buffers, interior)
            loss_c, loss_r = 0.5 * c.square().mean().item(), 0.5 * r.square().mean().item()
            for name, value in (("loss_c", loss_c), ("loss_r", loss_r)):
                if not np.isclose(value, result[name], rtol=1e-8, atol=1e-12):
                    raise ValueError(f"Saved model does not match recorded {name}: {run_dir}")
            records.append({
                **record, "status": "completed", **metrics,
                "loss_c": loss_c, "loss_r": loss_r, "loss": loss_c + loss_r,
                "reached_loss_tol": loss_c + loss_r <= config["loss_tol"],
                "accuracy_pass": metrics["velocity_relative_l2"] <= VELOCITY_L2_TOL
                                 and metrics["boundary_velocity_rms"] <= BOUNDARY_RMS_TOL,
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_memory_mib": result["peak_memory_bytes"] / 1024**2,
            })
            del model, c, r, parameters
        print(f"Evaluated {len(records)}/{len(args.methods) * len(args.seeds)} runs.", flush=True)

    columns = [
        "velocity_relative_l2", "u_relative_l2", "v_relative_l2", "pressure_relative_l2",
        "velocity_rmse", "velocity_max_abs", "pressure_max_abs", "boundary_velocity_rms",
        "pressure_reference_abs", "momentum_x_rms", "momentum_y_rms", "continuity_rms",
        "pde_rms", "loss_c", "loss_r", "loss", "elapsed_seconds", "peak_memory_mib",
    ]
    frame = pd.DataFrame(records)
    # 即使全部运行缺失或失败，也生成完整的状态表和空统计。
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    if "reached_loss_tol" not in frame:
        frame["reached_loss_tol"] = False
    if "accuracy_pass" not in frame:
        frame["accuracy_pass"] = False
    status_rows = []
    completed_sets = []
    for method in args.methods:
        selected = frame[frame["method"] == method]
        completed = selected[selected["status"] == "completed"]
        failed = int((selected["status"] == "error").sum())
        converged = int(completed["reached_loss_tol"].eq(True).sum())
        status_rows.append({
            "method": method, "requested": len(args.seeds), "completed": len(completed),
            "failed": failed, "missing_or_incomplete": len(args.seeds) - len(completed) - failed,
            "reached_loss_tol": converged, "convergence_rate": converged / len(args.seeds),
            "accuracy_pass": int(completed["accuracy_pass"].eq(True).sum()),
            "accuracy_pass_rate": completed["accuracy_pass"].eq(True).sum() / len(args.seeds),
        })
        completed_sets.append(set(completed["seed"]))
    status = pd.DataFrame(status_rows)
    common_seeds = sorted(set.intersection(*completed_sets))
    summary = summarize_metrics(frame, args.methods, columns)
    common_summary = summarize_metrics(frame[frame["seed"].isin(common_seeds)], args.methods, columns)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    common_summary.to_csv(output / "common_seeds_summary.csv", index=False)
    status.to_csv(output / "run_status_summary.csv", index=False)

    accuracy_panels = [
        ("velocity_relative_l2", "Velocity relative L2"),
        ("u_relative_l2", "u relative L2"), ("v_relative_l2", "v relative L2"),
        ("pressure_relative_l2", "Pressure relative L2"),
        ("boundary_velocity_rms", "Boundary velocity RMS"), ("pde_rms", "PDE RMS"),
    ]
    training_panels = [
        ("loss_c", "Final condition loss"), ("loss_r", "Final PDE loss"),
        ("loss", "Final total loss"), ("elapsed_seconds", "Training time (s)"),
        ("peak_memory_mib", "Peak allocated memory (MiB)"),
        ("pressure_reference_abs", "Pressure reference error"),
    ]
    verdict = compare_ftp(frame, summary, args.seeds)
    (output / "ftp_advantage.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    evaluation_config = {
        "seeds": args.seeds, "methods": args.methods,
        "grid_size": args.grid_size, "boundary_points_per_edge": args.grid_size,
        "pde_grid_size": args.pde_grid_size, "grid_type": "cell_centres",
        "velocity_relative_l2_threshold": VELOCITY_L2_TOL,
        "boundary_velocity_rms_threshold": BOUNDARY_RMS_TOL,
        "field_aggregation": "mean prediction and mean absolute error over each method's completed seeds",
        "common_completed_seeds": common_seeds,
        "plots_generated": not args.no_plots,
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_plots:
        plot_comparison(frame, args.methods, accuracy_panels, output / "comparison",
                        len(args.seeds), "Kovasznay: accuracy across random seeds")
        plot_comparison(frame, args.methods, training_panels, output / "training_comparison",
                        len(args.seeds), "Kovasznay: final losses and training cost")
        plot_fields(field_sums, exact_field, args, output)
        plot_histories(args, output)
    write_report(summary, common_summary, status, common_seeds, verdict, args, output / "REPORT.md")
    print("\n" + status.to_string(index=False))
    print("\nVelocity relative L2 statistics:")
    print(summary[summary["metric"] == "velocity_relative_l2"].drop(columns="metric").to_string(index=False))
    print(f"Aggregate comparison saved to {output}", flush=True)


if __name__ == "__main__":
    main()
