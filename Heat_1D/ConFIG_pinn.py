"""ConFIG adapter for the shared heat-equation PINN experiment."""

import math
from time import time

import torch

from ConFIG.conflictfree.grad_operator import ConFIGOperator
from ConFIG.conflictfree.utils import get_gradient_vector


def _cosine_learning_rate_scale(
    step, steps, warmup_steps, learning_rate, final_learning_rate
):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (steps - warmup_steps)
    cosine = 0.5 * (math.cos(math.pi * progress) + 1.0)
    final_scale = final_learning_rate / learning_rate
    return 1.0 - (1.0 - cosine) * (1.0 - final_scale)


def ConFIG(
    model,
    condition_fn,
    physics_fn,
    steps=10000,
    learning_rate=1.0e-3,
    final_learning_rate=1.0e-4,
):
    """Train with the official two-loss ConFIG gradient operator.

    The shared condition and physics losses are treated as the two tasks. The
    model, residual functions, starting point, and result serialization remain
    owned by the comparison framework.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if learning_rate <= 0.0 or final_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    operator = ConFIGOperator()
    warmup_steps = min(100, int(steps * 0.01))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_learning_rate_scale(
            step,
            steps,
            warmup_steps,
            learning_rate,
            final_learning_rate,
        ),
    )
    loss_history = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time()

    for _ in range(steps):
        optimizer.zero_grad()
        parameters = dict(model.named_parameters())
        physics = physics_fn(parameters)
        loss_r = 0.5 * torch.dot(physics, physics)
        loss_r.backward()
        physics_gradient = get_gradient_vector(
            model, none_grad_mode="zero"
        )

        optimizer.zero_grad()
        parameters = dict(model.named_parameters())
        condition = condition_fn(parameters)
        loss_c = 0.5 * torch.dot(condition, condition)
        loss_c.backward()
        condition_gradient = get_gradient_vector(
            model, none_grad_mode="zero"
        )

        loss = loss_c + loss_r
        loss_history.append(
            [loss_c.item(), loss_r.item(), loss.item()]
        )
        gradients = torch.stack((physics_gradient, condition_gradient))
        operator.update_gradient(
            model, gradients, losses=(loss_r, loss_c)
        )
        optimizer.step()
        scheduler.step()

    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated()
    return model, loss_history, elapsed_time, peak
