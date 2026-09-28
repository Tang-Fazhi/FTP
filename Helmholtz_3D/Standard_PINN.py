import torch
import math
from time import time


class _LossToleranceReached(Exception):
    pass


def Standard_PINN(model, condition_fn, physics_fn, adam_steps=1000, lbfgs_steps=200, learning_rate=5.0e-3, loss_tol=1e-10):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time()
    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        c = condition_fn(parameters)
        # c = c / math.sqrt(c.numel())
        r = physics_fn(parameters)
        # r = r / math.sqrt(r.numel())
        loss_c = 0.5 * torch.dot(c, c)
        loss_r = 0.5 * torch.dot(r, r)
        loss = loss_c + loss_r
        loss_history.append([loss_c.item(), loss_r.item(), loss.item()])
        if loss <= loss_tol:
            elapsed_time = time() - start_time
            peak = torch.cuda.max_memory_allocated()
            return model, loss_history, elapsed_time, peak
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

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        current_parameters = dict(model.named_parameters())
        current_c = condition_fn(current_parameters)
        #current_c = current_c / math.sqrt(current_c.numel())
        current_r = physics_fn(current_parameters)
        #current_r = current_r / math.sqrt(current_r.numel())
        current_loss_c = 0.5 * torch.dot(current_c, current_c)
        current_loss_r = 0.5 * torch.dot(current_r, current_r)
        current_loss = current_loss_c + current_loss_r
        loss_history.append([current_loss_c.item(), current_loss_r.item(), current_loss.item()])
        if current_loss.item() <= loss_tol:
            raise _LossToleranceReached
        current_loss.backward()
        return current_loss

    try:
        lbfgs.step(closure)
    except _LossToleranceReached:
        pass
    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated()
    return model, loss_history, elapsed_time, peak
