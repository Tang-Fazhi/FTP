import math
import os
import random

import numpy as np
import torch


DEFAULT_DTYPE = torch.float64
THERMAL_DIFFUSIVITY = 0.01
SPACE_INTERVAL = (0.0, 1.0)
TIME_INTERVAL = (0.0, 1.0)


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_default_dtype(DEFAULT_DTYPE)


def make_points(device):
    """Create the same heat-equation training grid used by the FTP demo."""
    x_initial = torch.linspace(
        *SPACE_INTERVAL, 7, device=device, dtype=DEFAULT_DTYPE
    )[1:-1]
    t_boundary = torch.linspace(
        *TIME_INTERVAL, 3, device=device, dtype=DEFAULT_DTYPE
    )
    x_interior = torch.linspace(
        0.05, 0.95, 8, device=device, dtype=DEFAULT_DTYPE
    )
    t_interior = torch.linspace(
        0.05, 0.95, 8, device=device, dtype=DEFAULT_DTYPE
    )
    xt_interior = torch.cartesian_prod(x_interior, t_interior)
    return x_initial, t_boundary, xt_interior


def exact_heat_solution(xt):
    x = xt[:, 0]
    t = xt[:, 1]
    return torch.exp(-THERMAL_DIFFUSIVITY * math.pi**2 * t) * torch.sin(
        math.pi * x
    )


def warm_start_conditions(
    model,
    condition_fn,
    adam_steps=10000,
    lbfgs_steps=1000,
    learning_rate=5.0e-3,
):
    """Pre-train the initial and boundary conditions before method comparison."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_loss = float("nan")
    loss_history = []

    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        condition = condition_fn(parameters)
        loss = 0.5 * torch.dot(condition, condition)
        loss_history.append([step, loss.item()])
        if step == 0:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_steps,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        line_search_fn="strong_wolfe",
    )
    step_index = [adam_steps]

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        current_parameters = dict(model.named_parameters())
        current_condition = condition_fn(current_parameters)
        current_loss = 0.5 * torch.dot(current_condition, current_condition)
        loss_history.append([step_index[0], current_loss.item()])
        step_index[0] += 1
        current_loss.backward()
        return current_loss

    lbfgs.step(closure)

    final_parameters = dict(model.named_parameters())
    final_condition = condition_fn(final_parameters)
    final_loss = float(
        (0.5 * torch.dot(final_condition, final_condition)).item()
    )
    return initial_loss, final_loss, loss_history
