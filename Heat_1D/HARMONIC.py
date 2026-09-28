"""HARMONIC adapter for the shared heat-equation PINN experiment."""

from time import time

import torch

from HARMONIC_SRC.src.optimizer.harmonic import HARMONIC as HARMONICOptimizer


def HARMONIC(
    model,
    condition_fn,
    physics_fn,
    steps=10000,
    learning_rate=1.0e-3,
):
    """Train ``model`` by harmonizing condition and physics gradients."""
    optimizer = HARMONICOptimizer(
        torch.optim.Adam(model.parameters(), lr=learning_rate), model
    )
    loss_history = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time()

    def closure(*, skip_backward=False):
        optimizer.zero_grad(set_to_none=False)
        parameters = dict(model.named_parameters())
        condition = condition_fn(parameters)
        physics = physics_fn(parameters)
        loss_c = 0.5 * torch.dot(condition, condition)
        loss_r = 0.5 * torch.dot(physics, physics)
        loss = loss_c + loss_r
        optimizer.losses = (loss_c, loss_r)
        loss_history.append(
            [loss_c.item(), loss_r.item(), loss.item()]
        )
        if not skip_backward:
            loss.backward()
        return loss

    for _ in range(steps):
        optimizer.step(closure)

    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated()
    return model, loss_history, elapsed_time, peak
