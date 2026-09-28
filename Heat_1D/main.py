import argparse
import math
import os
from copy import deepcopy
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap
from AL_PINN import AL_PINN
from FTP_pinn import FTP, ParameterDict
from NTK_LM import NTK_LM
from Standard_PINN import Standard_PINN
from HARMONIC import HARMONIC
from ConFIG_pinn import ConFIG
from functions import THERMAL_DIFFUSIVITY, make_points, set_seed, warm_start_conditions


METHODS = ("Standard_PINN", "NTK_LM", "AL_PINN", "HARMONIC", "ConFIG", "FTP")
RESULTS_ROOT = os.path.join("results", "heat")


class PINN(nn.Module):
    def __init__(self, width=15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, xt):
        return self.network(xt).squeeze(-1)


def condition_residual(
    model, parameters, buffers, x_initial, t_boundary
):
    xt_initial = torch.stack(
        (x_initial, torch.zeros_like(x_initial)), dim=1
    )
    xt_left = torch.stack(
        (torch.zeros_like(t_boundary), t_boundary), dim=1
    )
    xt_right = torch.stack(
        (torch.ones_like(t_boundary), t_boundary), dim=1
    )

    u_initial = functional_call(
        model, (parameters, buffers), (xt_initial,)
    )
    u_left = functional_call(model, (parameters, buffers), (xt_left,))
    u_right = functional_call(model, (parameters, buffers), (xt_right,))
    initial_target = torch.sin(math.pi * x_initial)

    # Initial condition and homogeneous Dirichlet boundary conditions.
    return torch.cat((u_initial - initial_target, u_left, u_right))


def physics_residual(model, parameters, buffers, xt_interior):
    def solution_at_point(current_parameters: ParameterDict, xt):
        value = functional_call(
            model, (current_parameters, buffers), (xt.unsqueeze(0),)
        )
        return value.squeeze()

    input_gradient = jacrev(solution_at_point, argnums=1)
    input_hessian = jacrev(input_gradient, argnums=1)
    first_derivatives = vmap(input_gradient, in_dims=(None, 0))(
        parameters, xt_interior
    )
    second_derivatives = vmap(input_hessian, in_dims=(None, 0))(
        parameters, xt_interior
    )

    u_t = first_derivatives[:, 1]
    u_xx = second_derivatives[:, 0, 0]
    return u_t - THERMAL_DIFFUSIVITY * u_xx


def get_args():
    parser = argparse.ArgumentParser(description="Compare PINN optimizers on the one-dimensional heat equation.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--config-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--config-final-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_initial, t_boundary, xt_interior = make_points(device)

    for seed in args.seeds:
        set_seed(seed)
        pinn = PINN().to(device)
        warmup_buffers = dict(pinn.named_buffers())

        def warmup_condition_fn(current_parameters):
            c = condition_residual(
                pinn,
                current_parameters,
                warmup_buffers,
                x_initial,
                t_boundary,
            )
            c = c / math.sqrt(c.numel())
            return c

        warmup_initial, warmup_final, warmup_history = warm_start_conditions(
            pinn, warmup_condition_fn
        )
        print(
            f"Seed {seed} condition warmup: "
            f"{warmup_initial:.6e} -> {warmup_final:.6e}"
        )

        for method in args.methods:
            model = deepcopy(pinn)
            buffers = dict(model.named_buffers())

            def condition_fn(current_parameters):
                c = condition_residual(model, current_parameters, buffers, x_initial, t_boundary)
                c = c / math.sqrt(c.numel())
                return c

            def physics_fn(current_parameters):
                r = physics_residual(model, current_parameters, buffers, xt_interior)
                r = r / math.sqrt(r.numel())
                return r

            output_dir = os.path.join(
                RESULTS_ROOT, method, f"seed_{seed}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(
                f"\n============ Start heat: {method} "
                f"with seed {seed} ============\n"
            )

            if method == "Standard_PINN":
                model, loss_history, elapsed_time, peak = Standard_PINN(
                    model,
                    condition_fn,
                    physics_fn,
                    adam_steps=int(args.steps * 0.9),
                    lbfgs_steps=int(args.steps * 0.1),
                    learning_rate=args.config_final_learning_rate
                )
            elif method == "NTK_LM":
                model, loss_history, elapsed_time, peak = NTK_LM(model, condition_fn, physics_fn, lm_steps=args.steps)
            elif method == "AL_PINN":
                model, loss_history, elapsed_time, peak = AL_PINN(model, condition_fn, physics_fn, adam_steps=args.steps, learning_rate=args.config_final_learning_rate)
            elif method == "HARMONIC":
                model, loss_history, elapsed_time, peak = HARMONIC(
                    model,
                    condition_fn,
                    physics_fn,
                    steps=args.steps,
                    learning_rate=args.config_learning_rate,
                )
            elif method == "ConFIG":
                model, loss_history, elapsed_time, peak = ConFIG(
                    model,
                    condition_fn,
                    physics_fn,
                    steps=args.steps,
                    learning_rate=args.config_learning_rate,
                    final_learning_rate=args.config_final_learning_rate,
                )
            else:
                model, loss_history, elapsed_time, peak = FTP(
                    model,
                    condition_fn,
                    physics_fn,
                    max_iteration=args.steps,
                )

            pd.DataFrame(
                warmup_history, columns=["epoch", "loss_c"]
            ).to_excel(
                os.path.join(output_dir, "warmup_loss_c.xlsx"),
                index=False,
            )
            pd.DataFrame(
                loss_history, columns=["loss_c", "loss_r", "loss"]
            ).to_excel(
                os.path.join(output_dir, "loss_history.xlsx"),
                index=False,
            )
            torch.save(
                model.state_dict(), os.path.join(output_dir, "model.pt")
            )
            with open(
                os.path.join(output_dir, "elapsed_time.txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "Problem: heat\n"
                    f"method: {method}\n"
                    f"seed: {seed}\n"
                    + f"elapsed time: {elapsed_time}\n"
                    + f"peak GPU memory: {peak / 1024**2:.6f} MiB"
                )


if __name__ == "__main__":
    main()
