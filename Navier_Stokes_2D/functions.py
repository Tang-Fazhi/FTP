import math
import os
import random
from time import time
import numpy as np
import torch


DEFAULT_DTYPE = torch.float64
X_INTERVAL = (0.0, 1.0)
Y_INTERVAL = (0.0, 1.0)
REYNOLDS_NUMBER = 40.0
VISCOSITY = 1.0 / REYNOLDS_NUMBER
DECAY_RATE = REYNOLDS_NUMBER / 2.0 - math.sqrt(REYNOLDS_NUMBER**2 / 4.0 + 4.0 * math.pi**2)
PRESSURE_POINT = (1.0, 0.0)


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


def exact_kovasznay_solution(xy):
    """Return (u, v, p) for the steady incompressible Navier-Stokes equations."""
    x, y = xy.unbind(-1)
    exponential = torch.exp(DECAY_RATE * x)
    u = 1.0 - exponential * torch.cos(2.0 * math.pi * y)
    v = DECAY_RATE / (2.0 * math.pi) * exponential * torch.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - exponential.square())
    return torch.stack((u, v, p), dim=-1)


def cell_center_grid(device, points_per_axis):
    if points_per_axis < 1:
        raise ValueError("points_per_axis must be at least 1")
    axis = (torch.arange(points_per_axis, device=device, dtype=DEFAULT_DTYPE) + 0.5) / points_per_axis
    x = X_INTERVAL[0] + (X_INTERVAL[1] - X_INTERVAL[0]) * axis
    y = Y_INTERVAL[0] + (Y_INTERVAL[1] - Y_INTERVAL[0]) * axis
    return torch.cartesian_prod(x, y)


def boundary_grid(device, points_per_edge, *, cell_centered=False):
    if points_per_edge < 2:
        raise ValueError("points_per_edge must be at least 2")
    if cell_centered:
        axis = (torch.arange(points_per_edge, device=device, dtype=DEFAULT_DTYPE) + 0.5) / points_per_edge
    else:
        axis = torch.linspace(0.0, 1.0, points_per_edge, device=device, dtype=DEFAULT_DTYPE)
    x = X_INTERVAL[0] + (X_INTERVAL[1] - X_INTERVAL[0]) * axis
    y = Y_INTERVAL[0] + (Y_INTERVAL[1] - Y_INTERVAL[0]) * axis
    left = torch.stack((torch.full_like(y, X_INTERVAL[0]), y), dim=1)
    right = torch.stack((torch.full_like(y, X_INTERVAL[1]), y), dim=1)
    bottom = torch.stack((x, torch.full_like(x, Y_INTERVAL[0])), dim=1)
    top = torch.stack((x, torch.full_like(x, Y_INTERVAL[1])), dim=1)
    points = torch.cat((left, right, bottom, top), dim=0)
    return points if cell_centered else torch.unique(points, dim=0)


def make_points(device, interior_points=256, boundary_points_per_edge=9):
    """Fixed interior cell centres and four endpoint edges, with unique corners."""
    if interior_points < 1:
        raise ValueError("interior_points must be at least 1")
    points_per_axis = math.isqrt(interior_points)
    if points_per_axis**2 != interior_points:
        raise ValueError("interior_points must be a perfect square")
    return boundary_grid(device, boundary_points_per_edge), cell_center_grid(device, points_per_axis)


def warm_start_conditions(
    model,
    condition_fn,
    adam_steps=10000,
    lbfgs_steps=1000,
    learning_rate=5.0e-3,
):
    """Pre-train the boundary conditions before method comparison."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_loss = float("nan")
    loss_history = []
    start_time = time()
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
    elapsed_time = time() - start_time
    final_parameters = dict(model.named_parameters())
    final_condition = condition_fn(final_parameters)
    final_loss = float(
        (0.5 * torch.dot(final_condition, final_condition)).item()
    )
    return initial_loss, final_loss, loss_history, elapsed_time
