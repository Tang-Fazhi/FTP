"""Levenberg-Marquardt training for the shared PINN residual system.

The current problem supplies two residual functions:

    c(theta): initial and boundary-condition residuals
    r(theta): PDE residuals

LM minimises the same nonlinear least-squares objective as Standard_PINN,
0.5 * ||c||^2 + 0.5 * ||r||^2.  The implementation follows the damped
Gauss-Newton and optional geodesic-acceleration scheme used by burgers_LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.func import jacfwd, jacrev


ParameterDict = Dict[str, Tensor]
ResidualFunction = Callable[[ParameterDict], Tensor]
LossRecord = List[float]


@dataclass
class LMConfig:
    lambda0: float = 1.0
    Lambda: float = 1.0e8
    tol: float = 1.0e-4
    alpha: float = 0.75
    max_inner: int = 12
    geodesic: bool = True
    fd_step: float = 0.1
    jac_mode: str = "jacrev"
    chunk_size: int = 2000


class LMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        condition_fn: ResidualFunction,
        physics_fn: ResidualFunction,
        config: LMConfig,
    ) -> None:
        self.model = model
        self.condition_fn = condition_fn
        self.physics_fn = physics_fn
        self.config = config
        self.lambda_value = float(config.lambda0)

        named_parameters = list(model.named_parameters())
        self.names = [name for name, _ in named_parameters]
        self.shapes = [parameter.shape for _, parameter in named_parameters]
        self.numels = [parameter.numel() for _, parameter in named_parameters]
        self.n_parameters = sum(self.numels)

    def get_flat_parameters(self) -> Tensor:
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self.model.parameters()]
        )

    def unflatten_parameters(self, flat_parameters: Tensor) -> ParameterDict:
        parameters = {}
        start = 0
        for name, shape, numel in zip(self.names, self.shapes, self.numels):
            parameters[name] = flat_parameters[start:start + numel].view(shape)
            start += numel
        return parameters

    def set_flat_parameters(self, flat_parameters: Tensor) -> None:
        parameters = self.unflatten_parameters(flat_parameters)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.copy_(parameters[name])

    def residual_components(
        self, flat_parameters: Tensor
    ) -> Tuple[Tensor, Tensor]:
        parameters = self.unflatten_parameters(flat_parameters)
        condition = self.condition_fn(parameters).reshape(-1)
        physics = self.physics_fn(parameters).reshape(-1)
        return condition, physics

    def residual_vector(self, flat_parameters: Tensor) -> Tensor:
        condition, physics = self.residual_components(flat_parameters)
        return torch.cat((condition, physics))

    @staticmethod
    def loss_record(condition: Tensor, physics: Tensor) -> LossRecord:
        loss_c = 0.5 * torch.dot(condition, condition)
        loss_r = 0.5 * torch.dot(physics, physics)
        return [
            float(loss_c.item()),
            float(loss_r.item()),
            float((loss_c + loss_r).item()),
        ]

    def current_loss_record(self) -> LossRecord:
        condition, physics = self.residual_components(self.get_flat_parameters())
        return self.loss_record(condition, physics)

    def jacobian(self, flat_parameters: Tensor) -> Tensor:
        if self.config.jac_mode == "jacrev":
            return jacrev(
                self.residual_vector,
                chunk_size=self.config.chunk_size,
            )(flat_parameters)
        if self.config.jac_mode == "jacfwd":
            return jacfwd(self.residual_vector)(flat_parameters)
        raise ValueError(f"Unknown jacobian mode: {self.config.jac_mode}")

    @staticmethod
    def solve(system: Tensor, right_hand_side: Tensor) -> Tensor:
        try:
            cholesky = torch.linalg.cholesky(system)
            return torch.cholesky_solve(
                right_hand_side.unsqueeze(1), cholesky
            ).squeeze(1)
        except RuntimeError:
            return torch.linalg.lstsq(
                system, right_hand_side.unsqueeze(1)
            ).solution.squeeze(1)

    def step(self) -> LossRecord:
        config = self.config
        flat_parameters = self.get_flat_parameters()
        condition, physics = self.residual_components(flat_parameters)
        residual = torch.cat((condition, physics))
        old_loss_record = self.loss_record(condition, physics)
        old_total_loss = old_loss_record[2]

        jacobian = self.jacobian(flat_parameters)
        gradient = jacobian.T @ residual
        gauss_newton = jacobian.T @ jacobian
        identity = torch.eye(
            self.n_parameters,
            device=flat_parameters.device,
            dtype=flat_parameters.dtype,
        )

        lambda_value = self.lambda_value
        for _ in range(config.max_inner + 1):
            system = gauss_newton + lambda_value * identity
            velocity = -self.solve(system, gradient)
            parameter_step = velocity

            if config.geodesic:
                h = config.fd_step
                residual_forward = self.residual_vector(
                    flat_parameters + h * velocity
                )
                jacobian_velocity = jacobian @ velocity
                directional_second_derivative = (
                    2.0
                    / (h * h)
                    * (residual_forward - residual - h * jacobian_velocity)
                )
                acceleration = -self.solve(
                    system,
                    jacobian.T @ directional_second_derivative,
                )
                if 2.0 * acceleration.norm() <= config.alpha * velocity.norm():
                    parameter_step = velocity + 0.5 * acceleration

            trial_parameters = flat_parameters + parameter_step
            trial_condition, trial_physics = self.residual_components(
                trial_parameters
            )
            trial_loss_record = self.loss_record(trial_condition, trial_physics)
            trial_total_loss = trial_loss_record[2]

            predicted_reduction = 0.5 * torch.dot(
                velocity, lambda_value * velocity - gradient
            ).item()
            if predicted_reduction > 0.0:
                gain_ratio = (
                    old_total_loss - trial_total_loss
                ) / predicted_reduction
            else:
                gain_ratio = -1.0

            if gain_ratio > config.tol and trial_total_loss < old_total_loss:
                self.set_flat_parameters(trial_parameters)
                self.lambda_value = max(
                    lambda_value / 3.0, 1.0 / config.Lambda
                )
                return trial_loss_record

            lambda_value = min(2.0 * lambda_value, config.Lambda)

        self.lambda_value = lambda_value
        return old_loss_record


def run_lm(model, condition_fn, physics_fn, n_iter=200, config=None, verbose=False):
    trainer = LMTrainer(
        model,
        condition_fn,
        physics_fn,
        config if config is not None else LMConfig(),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_time = time()
    loss_history = [trainer.current_loss_record()]

    for iteration in range(1, n_iter + 1):
        loss_record = trainer.step()
        loss_history.append(loss_record)
        if verbose:
            print(
                f"LM iteration {iteration:04d}: "
                f"condition={loss_record[0]:.6e}, "
                f"physics={loss_record[1]:.6e}, "
                f"total={loss_record[2]:.6e}, "
                f"lambda={trainer.lambda_value:.3e}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return model, loss_history, elapsed_time, peak


def NTK_LM(model, condition_fn, physics_fn, lm_steps=200, config=None):
    return run_lm(
        model=model,
        condition_fn=condition_fn,
        physics_fn=physics_fn,
        n_iter=lm_steps,
        config=config,
    )
