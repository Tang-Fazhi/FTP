"""Six unchanged Transfer_2D optimizers applied to a 3D acoustic Helmholtz BVP."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import traceback
import warnings

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
from functions import WAVE_NUMBER, exact_helmholtz_solution, make_points, set_seed, warm_start_conditions


METHODS = ("Standard_PINN", "NTK_LM", "AL_PINN", "HARMONIC", "ConFIG", "FTP")
CASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = CASE_DIR / "results" / "helmholtz"


class PINN(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        layers = []
        input_features = 3
        for _ in range(depth):
            layers.extend((nn.Linear(input_features, width), nn.Tanh()))
            input_features = width
        layers.append(nn.Linear(input_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, xyz):
        return self.network(xyz).squeeze(-1)


def condition_residual(model, parameters, buffers, xyz_boundary, boundary_values):
    return functional_call(model, (parameters, buffers), (xyz_boundary,)) - boundary_values


def physics_residual(model, parameters, buffers, xyz_interior):
    def solution_at_point(current_parameters, xyz):
        return functional_call(model, (current_parameters, buffers), (xyz.unsqueeze(0),)).squeeze()

    input_hessian = jacrev(jacrev(solution_at_point, argnums=1), argnums=1)
    second_derivatives = vmap(input_hessian, in_dims=(None, 0))(parameters, xyz_interior)
    laplacian = second_derivatives.diagonal(dim1=-2, dim2=-1).sum(-1)
    values = functional_call(model, (parameters, buffers), (xyz_interior,))
    return laplacian + WAVE_NUMBER ** 2 * values


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    return {str(p.relative_to(CASE_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CASE_DIR.rglob("*.py")) if "results" not in p.parts}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare six PINN optimizers on 3D acoustic Helmholtz.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--config-learning-rate", type=float, default=1e-3)
    parser.add_argument("--config-final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--interior-points", type=int, default=1728)
    parser.add_argument("--boundary-points-per-axis", type=int, default=5)
    parser.add_argument("--loss-tol", type=float, default=1e-7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--resume", action="store_true", help="Skip completed/error runs with matching configuration; retry interrupted runs.")
    args = parser.parse_args(argv)
    if args.steps < 10 or args.threads < 1 or args.loss_tol <= 0:
        parser.error("steps >= 10, threads >= 1, and loss-tol > 0 are required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("duplicate seeds/methods are not allowed")
    return args


def run_method(method, model, c, r, args):
    if method == "Standard_PINN":
        return Standard_PINN(model, c, r, adam_steps=int(args.steps * 0.9), lbfgs_steps=int(args.steps * 0.1), learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
    if method == "NTK_LM":
        return NTK_LM(model, c, r, lm_steps=args.steps, loss_tol=args.loss_tol)
    if method == "AL_PINN":
        return AL_PINN(model, c, r, adam_steps=args.steps, learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
    if method == "HARMONIC":
        return HARMONIC(model, c, r, steps=args.steps, learning_rate=args.config_learning_rate, loss_tol=args.loss_tol)
    if method == "ConFIG":
        return ConFIG(model, c, r, steps=args.steps, learning_rate=args.config_learning_rate, final_learning_rate=args.config_final_learning_rate, loss_tol=args.loss_tol)
    return FTP(model, c, r, max_iteration=args.steps, loss_tol=args.loss_tol)


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The unchanged Transfer_2D comparison adapters require CUDA.")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda")
    hashes = source_hashes()
    for seed in args.seeds:
        set_seed(seed)
        boundary, interior = make_points(device, args.interior_points, args.boundary_points_per_axis)
        boundary_values = exact_helmholtz_solution(boundary)
        pinn = PINN(args.width, args.depth).to(device)
        initial_hash = tensor_hash(pinn.parameters())
        sampling_hash = tensor_hash((boundary, interior))
        for method in args.methods:
            output = args.results_root / method / f"seed_{seed}"
            config = {
                "problem": "acoustic_helmholtz_3d", "equation": "u_xx+u_yy+u_zz+16*u=0",
                "wave_number": WAVE_NUMBER, "source": 0.0,
                "exact_solution": "sin(4*(x+y+z)/sqrt(3))+0.5*cos(4*(x-y)/sqrt(2))+0.25*sin(4*z)",
                "domain": {axis: [0.0, 1.0] for axis in "xyz"}, "boundary": "Dirichlet on all six faces",
                "method": method, "seed": seed, "steps": args.steps, "width": args.width, "depth": args.depth,
                "dtype": "float64", "loss_tol": args.loss_tol,
                "interior_points": args.interior_points, "boundary_points_per_axis": args.boundary_points_per_axis,
                "interior_point_count": len(interior), "boundary_point_count": len(boundary),
                "sampling_scheme": "cell_centered_interior_and_deduplicated_endpoint_faces",
                "config_learning_rate": args.config_learning_rate, "config_final_learning_rate": args.config_final_learning_rate,
                "ftp_warmup_adam_steps": 1000 if method == "FTP" else 0,
                "ftp_warmup_lbfgs_max_iter": 1000 if method == "FTP" else 0,
                "threads": args.threads, "device": torch.cuda.get_device_name(),
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "initial_parameters_sha256": initial_hash, "sampling_sha256": sampling_hash, "source_sha256": hashes,
            }
            if output.exists() and any(output.iterdir()):
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite {output}; use --resume or a different results root")
                previous = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
                if previous != config:
                    raise ValueError(f"Resume configuration/source mismatch: {output}")
                result_path = output / "run_result.json"
                if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8"))["status"] in ("completed", "error"):
                    print(f"SKIP {method} seed={seed}", flush=True)
                    continue
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "run_config.json", config)
            write_json(output / "run_result.json", {"status": "running", "method": method, "seed": seed})
            model = deepcopy(pinn)
            buffers = dict(model.named_buffers())
            progress = {"last": perf_counter(), "calls": 0}

            def condition_fn(parameters):
                progress["calls"] += 1
                if perf_counter() - progress["last"] >= 30:
                    print(f"PROGRESS {method} seed={seed} condition_evaluations={progress['calls']}", flush=True)
                    progress["last"] = perf_counter()
                return condition_residual(model, parameters, buffers, boundary, boundary_values) / math.sqrt(len(boundary))

            def physics_fn(parameters):
                return physics_residual(model, parameters, buffers, interior) / math.sqrt(len(interior))

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            warmup_time, warmup_peak, warmup_history = 0.0, 0, []
            print(f"START {method} seed={seed}", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    if method == "FTP":
                        warm_started = perf_counter()
                        warm_initial, warm_final, warmup_history, _ = warm_start_conditions(model, condition_fn, adam_steps=1000, lbfgs_steps=1000)
                        torch.cuda.synchronize()
                        warmup_time = perf_counter() - warm_started
                        warmup_peak = torch.cuda.max_memory_allocated()
                        print(f"WARMUP FTP seed={seed} initial={warm_initial:.6e} final={warm_final:.6e}", flush=True)
                    model, loss_history, method_time, peak = run_method(method, model, condition_fn, physics_fn, args)
                    torch.cuda.synchronize()
                    elapsed = perf_counter() - started
                    training_peak = max(int(peak), warmup_peak, torch.cuda.max_memory_allocated())
                    # Recompute at the saved model: some original adapters log a pre-update loss.
                    parameters = {k: v.detach() for k, v in model.named_parameters()}
                    final_c, final_r = condition_fn(parameters), physics_fn(parameters)
                    final_losses = [0.5 * final_c.square().sum().item(), 0.5 * final_r.square().sum().item()]
                    final_total = sum(final_losses)
                    if not math.isfinite(final_total):
                        raise FloatingPointError("Non-finite final residuals")
                    table = pd.DataFrame(loss_history, columns=["loss_c", "loss_r", "loss"])
                    table.to_excel(output / "loss_history.xlsx", index=False)
                    table.to_csv(output / "loss_history.csv", index=False)
                    if method == "FTP":
                        pd.DataFrame(warmup_history, columns=["epoch", "loss_c"]).to_excel(output / "warmup_loss_c.xlsx", index=False)
                    torch.save(model.state_dict(), output / "model.pt")
                    result = {
                        "status": "completed", "method": method, "seed": seed,
                        "converged": final_total <= args.loss_tol,
                        "final_loss_c": final_losses[0], "final_loss_r": final_losses[1], "final_loss": final_total,
                        "loss_records": len(loss_history), "elapsed_time_s": elapsed,
                        "method_reported_time_s": method_time, "warmup_time_s": warmup_time,
                        "peak_gpu_bytes": training_peak, "warmup_peak_gpu_bytes": warmup_peak,
                        "saved_parameters_sha256": tensor_hash(model.parameters()),
                    }
                    (output / "elapsed_time.txt").write_text(f"Problem: 3D acoustic Helmholtz\nmethod: {method}\nseed: {seed}\nelapsed time: {elapsed}\npeak GPU memory: {training_peak / 1024**2:.6f} MiB\n", encoding="utf-8")
                    print(f"DONE {method} seed={seed} loss={final_total:.6e} seconds={elapsed:.2f}", flush=True)
                except Exception:
                    error = traceback.format_exc()
                    (output / "Error.txt").write_text(error, encoding="utf-8")
                    result = {"status": "error", "method": method, "seed": seed, "elapsed_time_s": perf_counter() - started, "error": error}
                    print(f"ERROR {method} seed={seed}\n{error}", flush=True)
            result["warnings"] = [str(item.message) for item in caught]
            write_json(output / "run_result.json", result)
            (output / "warnings.txt").write_text("\n".join(result["warnings"]), encoding="utf-8")
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
