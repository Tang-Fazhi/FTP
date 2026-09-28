import argparse
import gc
import hashlib
import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

from AL_PINN import AL_PINN
from FTP_pinn import FTP
from NTK_LM import NTK_LM
from Standard_PINN import Standard_PINN
from HARMONIC import HARMONIC
from ConFIG_pinn import ConFIG
from functions import (
    REYNOLDS_NUMBER, VISCOSITY, PRESSURE_POINT,
    exact_kovasznay_solution, make_points, set_seed, warm_start_conditions,
)


METHODS = ("Standard_PINN", "NTK_LM", "AL_PINN", "HARMONIC", "ConFIG", "FTP")
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "kovasznay"


class PINN(nn.Module):
    def __init__(self, width=30, depth=3):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be at least 1")
        layers = []
        input_features = 2
        for _ in range(depth):
            layers.extend((nn.Linear(input_features, width), nn.Tanh()))
            input_features = width
        layers.append(nn.Linear(input_features, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, xy):
        return self.network(xy)


def condition_residual(model, parameters, buffers, xy_boundary, boundary_values,
                       xy_pressure, pressure_value):
    # 四条边只给速度；一个压力参考值消除压力的任意常数。
    predicted = functional_call(model, (parameters, buffers), (xy_boundary,))
    velocity = (predicted[:, :2] - boundary_values[:, :2]).reshape(-1)
    predicted_pressure = functional_call(model, (parameters, buffers), (xy_pressure,))[:, 2]
    pressure = predicted_pressure - pressure_value
    return torch.cat((velocity, pressure))


def physics_residual(model, parameters, buffers, xy_interior):
    def solution_at_point(current_parameters, xy):
        value = functional_call(model, (current_parameters, buffers), (xy.unsqueeze(0),))
        return value.squeeze(0)

    input_gradient = jacrev(solution_at_point, argnums=1)
    first = vmap(input_gradient, in_dims=(None, 0))(parameters, xy_interior)

    # 压力只需要一阶导数，二阶导数仅计算 u、v。
    def velocity_gradient(current_parameters, xy):
        return input_gradient(current_parameters, xy)[:2]

    input_hessian = jacrev(velocity_gradient, argnums=1)
    second = vmap(input_hessian, in_dims=(None, 0))(parameters, xy_interior)
    values = functional_call(model, (parameters, buffers), (xy_interior,))
    u, v = values[:, 0], values[:, 1]
    u_x, u_y = first[:, 0, 0], first[:, 0, 1]
    v_x, v_y = first[:, 1, 0], first[:, 1, 1]
    p_x, p_y = first[:, 2, 0], first[:, 2, 1]
    laplace_u = second[:, 0, 0, 0] + second[:, 0, 1, 1]
    laplace_v = second[:, 1, 0, 0] + second[:, 1, 1, 1]
    momentum_x = u * u_x + v * u_y + p_x - VISCOSITY * laplace_u
    momentum_y = u * v_x + v * v_y + p_y - VISCOSITY * laplace_v
    continuity = u_x + v_y
    return torch.stack((momentum_x, momentum_y, continuity), dim=1).reshape(-1)


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for value in tensors:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(filename, values):
    filename.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def get_args():
    parser = argparse.ArgumentParser(description="Compare PINN optimizers on 2D steady Navier-Stokes (Kovasznay flow).")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--interior-points", type=int, default=256)
    parser.add_argument("--boundary-points-per-edge", type=int, default=9)
    parser.add_argument("--config-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--config-final-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--loss-tol", type=float, default=1.0e-7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    if args.steps < 10 or args.threads < 1 or not math.isfinite(args.loss_tol) or args.loss_tol <= 0:
        parser.error("steps >= 10, threads >= 1, and finite loss-tol > 0 are required")
    if any(not math.isfinite(rate) or rate <= 0 for rate in (args.config_learning_rate, args.config_final_learning_rate)):
        parser.error("learning rates must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("duplicate seeds or methods are not allowed")
    return args


def main():
    args = get_args()
    # 原算例的基线优化器使用 CUDA 显存接口，沿用这一运行环境。
    if not torch.cuda.is_available():
        raise RuntimeError("The copied Transfer_2D comparison adapters require CUDA.")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda")
    # 显式在 CUDA 上初始化，与原算例依赖的默认设备行为一致。
    torch.set_default_device(device)
    for method in args.methods:
        for seed in args.seeds:
            output = args.results_root / method / f"seed_{seed}"
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Results already exist: {output}. Use other seeds or --results-root.")
    failures = []

    for seed in args.seeds:
        set_seed(seed)
        xy_boundary, xy_interior = make_points(device, args.interior_points, args.boundary_points_per_edge)
        boundary_values = exact_kovasznay_solution(xy_boundary)
        xy_pressure = torch.tensor([PRESSURE_POINT], device=device)
        pressure_value = exact_kovasznay_solution(xy_pressure)[:, 2]
        pinn = PINN(width=args.width, depth=args.depth).to(device)
        initial_hash = tensor_hash(pinn.parameters())
        sampling_hash = tensor_hash((xy_boundary, xy_interior, xy_pressure))

        for method in args.methods:
            model = deepcopy(pinn)
            buffers = dict(model.named_buffers())

            def condition_fn(current_parameters):
                c = condition_residual(model, current_parameters, buffers, xy_boundary,
                                       boundary_values, xy_pressure, pressure_value)
                return c / math.sqrt(c.numel())

            def physics_fn(current_parameters):
                r = physics_residual(model, current_parameters, buffers, xy_interior)
                return r / math.sqrt(r.numel())

            output_dir = args.results_root / method / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "problem": "steady_navier_stokes_2d_kovasznay",
                "equations": ["u*u_x+v*u_y+p_x-nu*(u_xx+u_yy)=0",
                              "u*v_x+v*v_y+p_y-nu*(v_xx+v_yy)=0", "u_x+v_y=0"],
                "reynolds_number": REYNOLDS_NUMBER, "viscosity": VISCOSITY,
                "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "boundary": "exact u,v on four edges; one pressure reference",
                "pressure_point": PRESSURE_POINT, "pressure_value": pressure_value.item(),
                "method": method, "seed": seed, "steps": args.steps,
                "width": args.width, "depth": args.depth,
                "parameter_count": sum(p.numel() for p in pinn.parameters()), "dtype": "float64",
                "sampling_scheme": "cell_centered_interior_and_deduplicated_endpoint_edges",
                "interior_points": args.interior_points,
                "boundary_points_per_edge": args.boundary_points_per_edge,
                "boundary_point_count": len(xy_boundary),
                "condition_residual_count": 2 * len(xy_boundary) + 1,
                "physics_residual_count": 3 * len(xy_interior),
                "residual_normalization": "each vector divided by sqrt(its scalar residual count)",
                "loss_tol": args.loss_tol,
                "config_learning_rate": args.config_learning_rate,
                "config_final_learning_rate": args.config_final_learning_rate,
                "ftp_warmup_adam_steps": 1000 if method == "FTP" else 0,
                "ftp_warmup_lbfgs_max_iter": 1000 if method == "FTP" else 0,
                "threads": args.threads, "device": torch.cuda.get_device_name(),
                "torch_version": str(torch.__version__),
                "initial_parameters_sha256": initial_hash, "sampling_sha256": sampling_hash,
            }
            write_json(output_dir / "run_config.json", config)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            warmup_time, warmup_peak = 0.0, 0
            print(f"\n============ Start Kovasznay: {method}, seed {seed} ============", flush=True)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    if method == "Standard_PINN":
                        model, loss_history, _, peak = Standard_PINN(
                            model, condition_fn, physics_fn, adam_steps=int(args.steps * 0.9),
                            lbfgs_steps=int(args.steps * 0.1),
                            learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
                    elif method == "NTK_LM":
                        model, loss_history, _, peak = NTK_LM(
                            model, condition_fn, physics_fn, lm_steps=args.steps, loss_tol=args.loss_tol)
                    elif method == "AL_PINN":
                        model, loss_history, _, peak = AL_PINN(
                            model, condition_fn, physics_fn, adam_steps=args.steps,
                            learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
                    elif method == "HARMONIC":
                        model, loss_history, _, peak = HARMONIC(
                            model, condition_fn, physics_fn, steps=args.steps,
                            learning_rate=args.config_learning_rate, loss_tol=args.loss_tol)
                    elif method == "ConFIG":
                        model, loss_history, _, peak = ConFIG(
                            model, condition_fn, physics_fn, steps=args.steps,
                            learning_rate=args.config_learning_rate,
                            final_learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
                    else:
                        warm_started = perf_counter()
                        initial, final, warmup_history, _ = warm_start_conditions(
                            model, condition_fn, adam_steps=1000, lbfgs_steps=1000)
                        torch.cuda.synchronize()
                        warmup_time = perf_counter() - warm_started
                        warmup_peak = torch.cuda.max_memory_allocated()
                        print(f"Condition warmup: {initial:.6e} -> {final:.6e}", flush=True)
                        model, loss_history, _, peak = FTP(
                            model, condition_fn, physics_fn, max_iteration=args.steps, loss_tol=args.loss_tol)
                    torch.cuda.synchronize()
                    elapsed_time = perf_counter() - started
                    peak = max(peak, warmup_peak, torch.cuda.max_memory_allocated())
                    final_parameters = dict(model.named_parameters())
                    c, r = condition_fn(final_parameters), physics_fn(final_parameters)
                    loss_c, loss_r = (0.5 * c.square().sum()).item(), (0.5 * r.square().sum()).item()
                    if not all(math.isfinite(value) for value in (loss_c, loss_r)):
                        raise RuntimeError("Non-finite final training loss")
                    if method == "FTP":
                        pd.DataFrame(warmup_history, columns=["epoch", "loss_c"]).to_excel(
                            output_dir / "warmup_loss_c.xlsx", index=False)
                    pd.DataFrame(loss_history, columns=["loss_c", "loss_r", "loss"]).to_excel(
                        output_dir / "loss_history.xlsx", index=False)
                    torch.save(model.state_dict(), output_dir / "model.pt")
                    result = {
                        "status": "completed", "loss_c": loss_c, "loss_r": loss_r,
                        "loss": loss_c + loss_r, "reached_loss_tol": loss_c + loss_r <= args.loss_tol,
                        "history_records": len(loss_history), "elapsed_seconds": elapsed_time,
                        "warmup_seconds": warmup_time, "peak_memory_bytes": peak,
                    }
                    write_json(output_dir / "run_result.json", result)
                    (output_dir / "elapsed_time.txt").write_text(f"{elapsed_time:.6f}\n", encoding="utf-8")
                    print(f"DONE {method}: loss={loss_c + loss_r:.6e}, time={elapsed_time:.2f}s", flush=True)
                except Exception as error:
                    write_json(output_dir / "run_result.json", {"status": "error", "error": str(error)})
                    (output_dir / "Error.txt").write_text(str(error), encoding="utf-8")
                    failures.append(f"{method}/seed_{seed}: {error}")
                    print(f"FAILED {failures[-1]}", flush=True)
            write_json(output_dir / "warnings.json", list(dict.fromkeys(str(w.message) for w in caught)))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError("Some runs failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
