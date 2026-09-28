import math
import os
import random
from time import time
import numpy as np
import torch


DEFAULT_DTYPE = torch.float64
X_INTERVAL = (0.0, 1.0)
Y_INTERVAL = (0.0, 1.0)
Z_INTERVAL = (0.0, 1.0)
WAVE_NUMBER = 4.0


def exact_helmholtz_solution(xyz):
    """Three plane waves with wave-vector norm 4, spanning all spatial axes."""
    x, y, z = xyz.unbind(-1)
    return (
        torch.sin(WAVE_NUMBER * (x + y + z) / math.sqrt(3.0))
        + 0.5 * torch.cos(WAVE_NUMBER * (x - y) / math.sqrt(2.0))
        + 0.25 * torch.sin(WAVE_NUMBER * z)
    )


def cell_center_grid(device, points_per_axis):
    if points_per_axis < 1:
        raise ValueError("points_per_axis must be positive")
    axis = (torch.arange(points_per_axis, device=device, dtype=DEFAULT_DTYPE) + 0.5) / points_per_axis
    return torch.cartesian_prod(axis, axis, axis)


def cube_faces(device, points_per_axis, *, cell_centered=False):
    if points_per_axis < 2:
        raise ValueError("boundary points_per_axis must be at least 2")
    if cell_centered:
        axis = (torch.arange(points_per_axis, device=device, dtype=DEFAULT_DTYPE) + 0.5) / points_per_axis
    else:
        axis = torch.linspace(0.0, 1.0, points_per_axis, device=device, dtype=DEFAULT_DTYPE)
    uv = torch.cartesian_prod(axis, axis)
    faces = []
    for dimension in range(3):
        for value in (0.0, 1.0):
            points = torch.empty(len(uv), 3, device=device, dtype=DEFAULT_DTYPE)
            points[:, dimension] = value
            points[:, [d for d in range(3) if d != dimension]] = uv
            faces.append(points)
    points = torch.cat(faces)
    return points if cell_centered else torch.unique(points, dim=0)


def make_points(device, interior_points=1728, boundary_points_per_axis=5):
    """Fixed cell-centred interior grid and six endpoint faces, with unique edges."""
    if interior_points < 1:
        raise ValueError("interior_points must be positive")
    n = round(interior_points ** (1.0 / 3.0))
    if n ** 3 != interior_points:
        raise ValueError("interior_points must be a perfect cube")
    return cube_faces(device, boundary_points_per_axis), cell_center_grid(device, n)


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
