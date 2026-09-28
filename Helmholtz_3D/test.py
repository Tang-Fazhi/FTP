"""Independent 3D accuracy evaluation and all-seed comparison (no seed selection)."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import torch

from functions import cell_center_grid, cube_faces, exact_helmholtz_solution, set_seed
from main import METHODS, PINN, RESULTS_ROOT, physics_residual, tensor_hash, write_json


LABELS = {"Standard_PINN": "Standard PINN", "NTK_LM": "NTK-LM", "AL_PINN": "AL-PINN", "HARMONIC": "HARMONIC", "ConFIG": "ConFIG", "FTP": "FTP"}
COLORS = dict(zip(METHODS, ("#767676", "#3775BA", "#B64342", "#42949E", "#9A4D8E", "#0F4D92")))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--volume-points-per-axis", type=int, default=81)
    parser.add_argument("--boundary-points-per-axis", type=int, default=81)
    parser.add_argument("--pde-points-per-axis", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pde-batch-size", type=int, default=256)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.volume_points_per_axis, args.boundary_points_per_axis, args.pde_points_per_axis) < 2:
        parser.error("evaluation axes require at least two points")
    if min(args.batch_size, args.pde_batch_size) < 1:
        parser.error("batch sizes must be positive")
    args.output_dir = args.output_dir or args.results_root / "evaluation"
    return args


def predict(model, points, batch_size):
    with torch.no_grad():
        return torch.cat([model(batch).detach() for batch in points.split(batch_size)])


def load_model(run, device):
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "run_result.json").read_text(encoding="utf-8"))
    model = PINN(config["width"], config["depth"]).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
    model.eval()
    if tensor_hash(model.parameters()) != result["saved_parameters_sha256"]:
        raise ValueError(f"Saved model hash mismatch: {run}")
    return model, config, result


def evaluate(args, device):
    volume = cell_center_grid(device, args.volume_points_per_axis)
    faces = cube_faces(device, args.boundary_points_per_axis, cell_centered=True)
    physics_points = cell_center_grid(device, args.pde_points_per_axis)
    exact = exact_helmholtz_solution(volume)
    exact_boundary = exact_helmholtz_solution(faces)
    rows = []
    identities = {}
    sampling_hashes = set()
    for method in args.methods:
        for seed in args.seeds:
            run = args.results_root / method / f"seed_{seed}"
            row = {"method": method, "seed": seed, "status": "missing", "accuracy_pass": False, "converged": False}
            result_path = run / "run_result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                row.update({key: result[key] for key in ("status", "elapsed_time_s", "converged", "final_loss", "final_loss_c", "final_loss_r", "loss_records", "warmup_time_s") if key in result})
                if result["status"] == "completed":
                    model, config, _ = load_model(run, device)
                    identity = config["initial_parameters_sha256"]
                    if seed in identities and identities[seed] != identity:
                        raise ValueError(f"Initial model differs across methods for seed {seed}")
                    identities[seed] = identity
                    sampling_hashes.add(config["sampling_sha256"])
                    error = predict(model, volume, args.batch_size) - exact
                    boundary_error = predict(model, faces, args.batch_size) - exact_boundary
                    params = {k: v.detach() for k, v in model.named_parameters()}
                    residuals = torch.cat([physics_residual(model, params, {}, batch).detach() for batch in physics_points.split(args.pde_batch_size)])
                    row.update({
                        "relative_l2": (error.norm() / exact.norm()).item(),
                        "rmse": error.square().mean().sqrt().item(),
                        "max_abs_error": error.abs().max().item(),
                        "boundary_rms": boundary_error.square().mean().sqrt().item(),
                        "pde_rms": residuals.square().mean().sqrt().item(),
                        "peak_gpu_gib": result["peak_gpu_bytes"] / 1024 ** 3,
                    })
                    row["accuracy_pass"] = row["relative_l2"] <= 1e-3 and row["boundary_rms"] <= 1e-4
                    if not all(math.isfinite(row[key]) for key in ("relative_l2", "rmse", "max_abs_error", "boundary_rms", "pde_rms")):
                        raise FloatingPointError(f"Non-finite evaluation: {run}")
                    print(f"EVAL {method} seed={seed} relL2={row['relative_l2']:.6e} boundary={row['boundary_rms']:.6e}", flush=True)
            rows.append(row)
    if len(sampling_hashes) > 1:
        raise ValueError("Sampling differs between completed runs")
    return pd.DataFrame(rows)


def summarize(metrics):
    rows = []
    for method, group in metrics.groupby("method", sort=False):
        completed = group[group.status == "completed"]
        row = {"method": method, "requested": len(group), "completed": len(completed), "errors": int((group.status == "error").sum()), "missing_or_running": int((~group.status.isin(["completed", "error"])).sum()), "converged": int(group.converged.sum()), "accuracy_pass": int(group.accuracy_pass.sum())}
        for metric in ("relative_l2", "rmse", "max_abs_error", "boundary_rms", "pde_rms", "elapsed_time_s", "peak_gpu_gib"):
            if len(completed):
                q1, median, q3, p90 = np.quantile(completed[metric], [0.25, 0.5, 0.75, 0.9])
                row.update({f"{metric}_median": median, f"{metric}_q1": q1, f"{metric}_q3": q3, f"{metric}_p90": p90})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_ftp(metrics, summary, seeds):
    candidates = summary[(summary.method != "FTP") & (summary.completed == len(seeds))]
    result = {"formal_seed_set": sorted(seeds) == list(range(10)), "criterion": "Lower median and p90 relative L2 than the best complete baseline, with >=8/10 paired wins", "criterion_met": False}
    ftp = metrics[(metrics.method == "FTP") & (metrics.status == "completed")]
    if candidates.empty or len(ftp) != len(seeds):
        result["reason"] = "Insufficient complete runs"
        return result, pd.DataFrame()
    best = candidates.sort_values(["relative_l2_median", "method"]).iloc[0]
    baseline = metrics[(metrics.method == best.method) & (metrics.status == "completed")]
    pairs = ftp[["seed", "relative_l2"]].merge(baseline[["seed", "relative_l2"]], on="seed", suffixes=("_ftp", "_baseline"))
    pairs["baseline"] = best.method
    pairs["ftp_win"] = pairs.relative_l2_ftp < pairs.relative_l2_baseline
    pairs["baseline_over_ftp"] = pairs.relative_l2_baseline / pairs.relative_l2_ftp
    fmedian, fp90 = np.quantile(ftp.relative_l2, [0.5, 0.9])
    result.update({"baseline": best.method, "paired_seeds": len(pairs), "ftp_wins": int(pairs.ftp_win.sum()), "ftp_median_relative_l2": fmedian, "baseline_median_relative_l2": best.relative_l2_median, "ftp_p90_relative_l2": fp90, "baseline_p90_relative_l2": best.relative_l2_p90})
    result["criterion_met"] = bool(result["formal_seed_set"] and len(pairs) == 10 and result["ftp_wins"] >= 8 and fmedian < best.relative_l2_median and fp90 < best.relative_l2_p90)
    return result, pairs


def save_figure(fig, output):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison(metrics, summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, metric, label in zip(axes, ("relative_l2", "elapsed_time_s", "boundary_rms"), ("Relative $L_2$ error", "Full training time (s)", "Boundary RMS error")):
        for i, row in enumerate(summary.itertuples()):
            completed = metrics[(metrics.method == row.method) & (metrics.status == "completed")].sort_values("seed")
            if completed.empty:
                continue
            values = completed[metric].to_numpy()
            x = i + np.linspace(-0.17, 0.17, len(values))
            ax.scatter(x, values, color=COLORS[row.method], s=19, alpha=0.75)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.plot([i, i], [q1, q3], color=COLORS[row.method], lw=5, alpha=0.5)
            ax.plot([i - 0.18, i + 0.18], [median, median], color="black", lw=1.2)
        ax.set_xticks(range(len(summary)), [LABELS[m] for m in summary.method], rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("All seeds; dots = runs, bars = IQR, black line = median")
    save_figure(fig, output / "comparison")


def plot_slices(args, device):
    # Fix the seed before seeing rankings. Missing seed 0 is never replaced by a better run.
    methods = [m for m in args.methods if (args.results_root / m / "seed_0" / "model.pt").exists()]
    if not methods:
        return
    axis = torch.linspace(0, 1, 81, device=device)
    xy = torch.cartesian_prod(axis, axis)
    arrays = {"axis": axis.cpu().numpy(), "z": np.array([0.25, 0.5, 0.75])}
    for z in arrays["z"]:
        points = torch.cat((xy, torch.full((len(xy), 1), float(z), device=device)), dim=1)
        arrays[f"exact_z{z}"] = exact_helmholtz_solution(points).reshape(81, 81).cpu().numpy()
        for method in methods:
            model, _, _ = load_model(args.results_root / method / "seed_0", device)
            arrays[f"{method}_z{z}"] = predict(model, points, args.batch_size).reshape(81, 81).cpu().numpy()
    np.savez_compressed(args.output_dir / "seed_0_slices.npz", **arrays)
    for error_plot in (False, True):
        columns = methods if error_plot else ["exact", *methods]
        fig, axes = plt.subplots(3, len(columns), figsize=(2.15 * len(columns), 6.6), squeeze=False, constrained_layout=True)
        values = []
        for z in arrays["z"]:
            for method in columns:
                value = arrays[f"{method}_z{z}"]
                values.append(np.abs(value - arrays[f"exact_z{z}"]) if error_plot else value)
        norm = LogNorm(vmin=1e-8, vmax=max(1e-7, max(float(v.max()) for v in values))) if error_plot else Normalize(vmin=min(float(v.min()) for v in values), vmax=max(float(v.max()) for v in values))
        for i, z in enumerate(arrays["z"]):
            for j, method in enumerate(columns):
                ax = axes[i, j]
                value = values[i * len(columns) + j]
                im = ax.pcolormesh(arrays["axis"], arrays["axis"], np.maximum(value, 1e-8).T if error_plot else value.T, cmap="magma" if error_plot else "RdBu_r", norm=norm, shading="auto", rasterized=True)
                ax.set_aspect("equal")
                if i == 0:
                    ax.set_title("Exact" if method == "exact" else LABELS[method], fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"z = {z}\ny")
                if i == 2:
                    ax.set_xlabel("x")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Absolute error (floor 1e-8)" if error_plot else "Acoustic field u")
        fig.suptitle("Fixed seed 0: " + ("absolute errors" if error_plot else "solution slices"))
        save_figure(fig, args.output_dir / ("seed_0_errors" if error_plot else "seed_0_solutions"))


def plot_histories(args):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    records = []
    for method, ax in zip(args.methods, axes.ravel()):
        for seed in args.seeds:
            path = args.results_root / method / f"seed_{seed}" / "loss_history.csv"
            if path.exists():
                frame = pd.read_csv(path)
                ax.plot(np.arange(len(frame)), frame.loss, color=COLORS[method], alpha=0.45, lw=0.8)
                records.append({"method": method, "seed": seed, "records": len(frame)})
        ax.set_yscale("log")
        ax.set_xlabel("Recorded loss evaluation")
        ax.set_ylabel("Total loss")
        ax.set_title(LABELS[method] + (" (after warmup)" if method == "FTP" else ""))
        ax.grid(alpha=0.2)
    fig.suptitle("All available seeds; evaluation counts are method-specific, not equal-time axes")
    save_figure(fig, args.output_dir / "training_losses")
    pd.DataFrame(records).to_csv(args.output_dir / "loss_record_counts.csv", index=False)


def write_report(args, summary, verdict):
    lines = ["# 三维声学 Helmholtz 实测", "", "方程：Δu + 16u = 0，区域 (0,1)^3，六面施加解析 Dirichlet 条件。", "", f"本报告包含请求的 {len(args.seeds)} 个种子：{args.seeds}。统计仅使用完成运行的模型，异常和缺失单列。", "", "| 方法 | 完成/请求 | 损失达标 | 独立精度达标 | 相对 L2 中位数 | 相对 L2 P90 | 时间中位数(s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        number = lambda key: f"{row[key]:.4g}" if key in row and pd.notna(row[key]) else "—"
        lines.append(f"| {LABELS[row['method']]} | {row['completed']}/{row['requested']} | {row['converged']} | {row['accuracy_pass']} | {number('relative_l2_median')} | {number('relative_l2_p90')} | {number('elapsed_time_s_median')} |")
    lines += ["", "独立精度达标：相对 L2 ≤ 1e-3 且六面边界 RMS ≤ 1e-4。损失达标依据保存模型重新计算的训练损失。", "", "FTP 优势判据：相对最佳完整基线，中位数和 P90 均更低，且正式种子 0–9 至少 8 次配对获胜。", ""]
    if "baseline" in verdict:
        lines.append(f"比较基线为 {LABELS[verdict['baseline']]}，FTP 配对获胜 {verdict['ftp_wins']}/{verdict['paired_seeds']}；正式优势判据{'达到' if verdict['criterion_met'] else '未达到'}。")
    else:
        lines.append("尚无足够的完整运行支持优势判断。")
    lines += ["", "时间包含 FTP 条件预训练和主体训练，并在 CUDA 同步后计时；不包含保存文件和独立评估。保留原方法特有的预训练、学习率及优化步骤，因此结论适用于该完整流程。", "", "损失曲线横轴为各方法记录的损失评价次数；L-BFGS 包含线搜索评价，不能解释为统一迭代成本或达标时间。", "", "独立网格采用体域单元中心、六面单元中心和独立 PDE 点；只在 seed 0 画固定截面，不筛选最佳种子。", "", "详细指标见 metrics_by_seed.csv、metrics_summary.csv 和 ftp_paired_comparison.csv；异常见各运行目录 Error.txt。"]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = get_args()
    torch.set_num_threads(1)
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args, device)
    summary = summarize(metrics)
    verdict, pairs = compare_ftp(metrics, summary, args.seeds)
    metrics.to_csv(args.output_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "ftp_paired_comparison.csv", index=False)
    write_json(args.output_dir / "ftp_advantage.json", verdict)
    write_json(args.output_dir / "evaluation_config.json", {"seeds": args.seeds, "methods": args.methods, "volume_points_per_axis": args.volume_points_per_axis, "boundary_points_per_axis": args.boundary_points_per_axis, "pde_points_per_axis": args.pde_points_per_axis, "volume_grid": "cell_centres", "boundary_grid": "six face cell centres", "relative_l2_threshold": 1e-3, "boundary_rms_threshold": 1e-4, "slice_seed": 0})
    write_report(args, summary, verdict)
    if not args.no_plots:
        plot_comparison(metrics, summary, args.output_dir)
        plot_slices(args, device)
        plot_histories(args)
    print(summary.to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
