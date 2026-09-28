import torch
import math
from time import time


def AL_PINN(
    model,
    condition_fn,
    physics_fn,
    adam_steps=10000,
    beta=1.0,
    learning_rate=1.0e-4,
    multiplier_learning_rate=1.0e-4,
    loss_tol=1.0e-10,
):
    parameters = dict(model.named_parameters())
    initial_condition = condition_fn(parameters)
    multipliers = torch.zeros_like(
        initial_condition.detach(), requires_grad=True
    )
    del initial_condition

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": learning_rate},
            {"params": [multipliers], "lr": multiplier_learning_rate},
        ]
    )
    loss_history = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time()

    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        condition = condition_fn(parameters)
        physics = physics_fn(parameters)
        # condition = condition / math.sqrt(condition.numel())
        # physics = physics / math.sqrt(physics.numel())
        loss_c = 0.5 * torch.dot(condition, condition)
        loss_r = 0.5 * torch.dot(physics, physics)
        comparable_loss = loss_c + loss_r
        augmented_loss = (
            loss_r
            + beta * loss_c
            + torch.dot(multipliers, condition)
        )

        loss_history.append(
            [loss_c.item(), loss_r.item(), comparable_loss.item()]
        )

        if comparable_loss.item() <= loss_tol:
            elapsed_time = time() - start_time
            peak = torch.cuda.max_memory_allocated()
            return model, loss_history, elapsed_time, peak

        augmented_loss.backward()
        if multipliers.grad is None:
            raise RuntimeError("AL-PINN multiplier gradient is missing")
        multipliers.grad.mul_(-1.0)
        optimizer.step()

    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated()
    return model, loss_history, elapsed_time, peak
