from __future__ import annotations

import math
import operator
import warnings
from importlib.util import find_spec
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch
from torch.func import jacrev, jvp, vjp


__all__ = ["FTP"]


_DTYPE = torch.float64
_ParameterDict = dict[str, torch.Tensor]
_ResidualFunction = Callable[[_ParameterDict], torch.Tensor]


@dataclass(frozen=True)
class _Settings:
    max_iterations: int
    loss_tolerance: float
    feasibility_tolerance: float = 1.0e-5
    feasibility_stationarity_tolerance: float = 1.0e-8
    initial_feasibility_steps: int = 200
    restoration_steps: int = 6
    restoration_forcing: float = 1.0e-1
    restoration_tolerance_floor: float = 1.0e-12
    beta2: float = 0.9
    metric_epsilon: float = 1.0e-12
    min_inverse_metric: float = 1.0e-3
    max_inverse_metric: float = 1.0e3
    rank_hysteresis_factor: float = 2.0
    armijo_constant: float = 1.0e-4
    max_feasibility_backtracks: int = 20
    stationarity_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-12
    tangent_tolerance_max: float = 1.0e-4
    tangent_tolerance_min: float = 1.0e-12
    tangent_forcing: float = 1.0e-1
    linear_tolerance_max: float = 1.0e-1
    linear_tolerance_min: float = 1.0e-12
    linear_forcing: float = 1.0e-1
    cg_max_iterations: int = 500
    # Eager Heat CPU/CUDA benchmarks favor immediate checks over masked blocks.
    cg_block_size: int = 1
    lm_damping_initial: float = 1.0e-5
    lm_damping_min: float = 1.0e-12
    lm_damping_max: float = 1.0e12
    trust_accept_ratio: float = 1.0e-1
    trust_bad_ratio: float = 2.5e-1
    trust_good_ratio: float = 7.5e-1
    damping_increase: float = 4.0
    damping_decrease: float = 5.0e-1
    max_lm_attempts: int = 8
    max_consecutive_rejections: int = 3


@dataclass(frozen=True)
class _ParameterLayout:

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    starts: tuple[int, ...]
    stops: tuple[int, ...]
    size: int
    device: torch.device

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, torch.Tensor]) -> _ParameterLayout:
        if not parameters:
            raise ValueError("FTP 要求模型至少包含一个参数")

        names: list[str] = []
        shapes: list[torch.Size] = []
        starts: list[int] = []
        stops: list[int] = []
        devices: set[torch.device] = set()
        offset = 0
        for name, value in parameters.items():
            if value.dtype != _DTYPE:
                raise ValueError("FTP 的所有模型参数必须使用 torch.float64")
            if not torch.is_floating_point(value):
                raise ValueError(f"模型参数 {name!r} 必须是浮点张量")
            names.append(name)
            shapes.append(value.shape)
            starts.append(offset)
            offset += value.numel()
            stops.append(offset)
            devices.add(value.device)

        if len(devices) != 1:
            raise ValueError("所有模型参数必须位于同一设备")
        return cls(
            names=tuple(names),
            shapes=tuple(shapes),
            starts=tuple(starts),
            stops=tuple(stops),
            size=offset,
            device=next(iter(devices)),
        )

    def flatten(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([values[name].reshape(-1) for name in self.names])

    def flatten_jacobian(
        self, jacobian: Mapping[str, torch.Tensor], residual_count: int
    ) -> torch.Tensor:
        return torch.cat(
            [jacobian[name].reshape(residual_count, -1) for name in self.names],
            dim=1,
        )

    def unflatten(self, vector: torch.Tensor) -> _ParameterDict:
        if vector.shape != (self.size,):
            raise ValueError("扁平参数向量的长度必须等于模型参数数 p")
        return {
            name: vector[start:stop].reshape(shape)
            for name, shape, start, stop in zip(
                self.names, self.shapes, self.starts, self.stops, strict=True
            )
        }

    def add(
        self, parameters: Mapping[str, torch.Tensor], update: torch.Tensor
    ) -> _ParameterDict:
        if update.shape != (self.size,):
            raise ValueError("参数更新向量的长度必须等于模型参数数 p")
        return {
            name: (
                parameters[name] + update[start:stop].reshape(shape)
            ).detach()
            for name, shape, start, stop in zip(
                self.names, self.shapes, self.starts, self.stops, strict=True
            )
        }


@dataclass(frozen=True)
class _ConditionSVD:
    """B=J_c H^{-1/2} 的薄 SVD 及其数值有效秩。"""

    left: torch.Tensor
    singular_values: torch.Tensor
    right_transpose: torch.Tensor
    scale: torch.Tensor
    rank: int
    rank_tolerance: float
    rank_is_stable: bool
    condition_count: int
    parameter_count: int

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return vector
        right = self.right_transpose[: self.rank]
        return vector - right.T @ (right @ vector)

    @property
    def first_discarded_singular_value(self) -> float:
        if self.rank >= self.singular_values.numel():
            return 0.0
        return float(self.singular_values[self.rank].detach().item())


@dataclass(frozen=True)
class _ConditionState:
    residual: torch.Tensor
    jacobian: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class _ConditionMeasures:
    residual_norm: float
    effective_residual_norm: float
    stationarity_norm: float
    relative_stationarity: float
    unattainable_residual_norm: float


@dataclass(frozen=True)
class _FeasibilityResult:
    parameters: _ParameterDict
    success: bool
    measures: _ConditionMeasures
    iterations: int
    rank: int
    reason: str
    state: _ConditionState


class _PhysicsOperator(Protocol):
    backend: str

    def jvp(self, vector: torch.Tensor) -> torch.Tensor: ...

    def vjp(self, vector: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class _CallablePhysicsOperator:
    backend: str
    jvp_function: Callable[[torch.Tensor], torch.Tensor]
    vjp_function: Callable[[torch.Tensor], torch.Tensor]

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        return self.jvp_function(vector)

    def vjp(self, vector: torch.Tensor) -> torch.Tensor:
        return self.vjp_function(vector)


@dataclass(frozen=True)
class _PhysicsState:
    residual: torch.Tensor
    loss: torch.Tensor
    gradient: torch.Tensor
    operator: _PhysicsOperator


@dataclass(frozen=True)
class _CGResult:
    solution: torch.Tensor
    relative_residual: float
    iterations: int
    success: bool


@dataclass(frozen=True)
class _LMStep:
    direction: torch.Tensor
    predicted_reduction: float
    linear_residual: float
    tangent_defect: float
    success: bool


@dataclass(frozen=True)
class _TrialResult:
    parameters: _ParameterDict
    accepted: bool
    condition_loss: float
    physics_loss: float
    reduction_ratio: float
    rank: int
    condition_state: _ConditionState | None = None


class _AdaptiveMetric:
    def __init__(self, settings: _Settings) -> None:
        self._settings = settings
        self._step = 0
        self._second_moment: torch.Tensor | None = None

    def update(self, gradient: torch.Tensor) -> torch.Tensor:
        detached = gradient.detach()
        if self._second_moment is None:
            self._second_moment = torch.zeros_like(detached)
        self._step += 1
        beta2 = self._settings.beta2
        self._second_moment = (
            beta2 * self._second_moment
            + (1.0 - beta2) * detached.square()
        )
        corrected = self._second_moment / (1.0 - beta2**self._step)
        return torch.rsqrt(corrected + self._settings.metric_epsilon).clamp(
            self._settings.min_inverse_metric,
            self._settings.max_inverse_metric,
        )


class _RankWarningTracker:
    def __init__(self) -> None:
        self._last_key: tuple[int, int, int, bool] | None = None

    def report(
        self, factorization: _ConditionSVD, measures: _ConditionMeasures
    ) -> None:
        incomplete = factorization.rank < factorization.condition_count
        if not incomplete and factorization.rank_is_stable:
            self._last_key = None
            return
        key = (
            factorization.condition_count,
            factorization.parameter_count,
            factorization.rank,
            factorization.rank_is_stable,
        )
        if key == self._last_key:
            return
        self._last_key = key
        warnings.warn(
            "条件 Jacobian 仅在数值有效子空间内受保护："
            f"q={factorization.condition_count}, "
            f"p={factorization.parameter_count}, "
            f"rank={factorization.rank}, "
            f"rank_tolerance={factorization.rank_tolerance:.3e}, "
            f"first_discarded_sigma="
            f"{factorization.first_discarded_singular_value:.3e}, "
            f"unattainable_residual="
            f"{measures.unattainable_residual_norm:.3e}, "
            f"rank_stable={factorization.rank_is_stable}",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_residual(
    name: str, residual: torch.Tensor, device: torch.device
) -> None:
    if not isinstance(residual, torch.Tensor):
        raise TypeError(f"{name}必须返回 torch.Tensor")
    if residual.ndim != 1:
        raise ValueError(f"{name}必须返回一维残差向量")
    if residual.numel() == 0:
        raise ValueError(f"{name}不能返回空残差向量")
    if residual.dtype != _DTYPE:
        raise ValueError(f"{name}必须返回 torch.float64 残差")
    if residual.device != device:
        raise ValueError(f"{name}残差必须与模型参数位于同一设备")
    if not bool(torch.isfinite(residual).all().item()):
        raise ValueError(f"{name}残差包含 NaN 或 Inf")


def _prepare_residual_functions(
    parameters: _ParameterDict,
    condition_function: _ResidualFunction,
    physics_function: _ResidualFunction,
    layout: _ParameterLayout,
) -> tuple[_ResidualFunction, _ResidualFunction]:
    """CUDA 上尝试编译并验证函数变换；失败时安全回退 eager。"""
    if layout.device.type != "cuda" or not hasattr(torch, "compile"):
        return condition_function, physics_function
    if find_spec("triton") is None:
        warnings.warn(
            "当前环境缺少 torch.compile 的 Triton CUDA 优化后端，"
            "已使用吞吐更高的 eager 路径",
            RuntimeWarning,
            stacklevel=3,
        )
        return condition_function, physics_function

    try:
        compiled_condition = torch.compile(condition_function, dynamic=False, mode="max-autotune")
        compiled_physics = torch.compile(physics_function, dynamic=False, mode="max-autotune")
        c = compiled_condition(parameters)
        _validate_residual("condition_fn", c, layout.device)
        r, pullback = vjp(compiled_physics, parameters)
        _validate_residual("physics_fn", r, layout.device)
        pullback(r)
        zero_tangent = {
            name: torch.zeros_like(parameters[name]) for name in layout.names
        }
        jvp(compiled_physics, (parameters,), (zero_tangent,))
        vjp(compiled_condition, parameters)
        return compiled_condition, compiled_physics
    except Exception as error:
        if isinstance(error, torch.OutOfMemoryError):
            torch.cuda.empty_cache()
        warnings.warn(
            "torch.compile 或其自动微分变换不可用，已回退 eager："
            f"{type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
        return condition_function, physics_function


def _residual_and_jacobian(
    parameters: _ParameterDict,
    residual_function: _ResidualFunction,
    layout: _ParameterLayout,
    name: str,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a deterministic residual and its Jacobian in one AD transform."""
    def residual_with_aux(current: _ParameterDict):
        residual = residual_function(current)
        return residual, residual

    jacobian_tree, residual = jacrev(
        residual_with_aux, has_aux=True, chunk_size=chunk_size
    )(parameters)
    _validate_residual(name, residual, layout.device)
    jacobian = layout.flatten_jacobian(jacobian_tree, residual.numel())
    if jacobian.shape != (residual.numel(), layout.size):
        raise RuntimeError(f"{name} Jacobian shape does not match residuals and parameters")
    if not bool(torch.isfinite(jacobian).all().item()):
        raise ValueError(f"{name} Jacobian contains NaN or Inf")
    return residual, jacobian


def _condition_state(
    parameters: _ParameterDict,
    condition_function: _ResidualFunction,
    layout: _ParameterLayout,
) -> _ConditionState:
    residual, jacobian = _residual_and_jacobian(
        parameters, condition_function, layout, "condition_fn"
    )
    return _ConditionState(
        residual=residual,
        jacobian=jacobian,
        loss=0.5 * torch.dot(residual, residual),
    )


def _factor_condition_jacobian(
    jacobian: torch.Tensor,
    inverse_metric: torch.Tensor,
    previous_rank: int | None,
    hysteresis_factor: float,
) -> _ConditionSVD:
    if jacobian.ndim != 2:
        raise ValueError("条件 Jacobian 必须是二维矩阵")
    condition_count, parameter_count = jacobian.shape
    if condition_count == 0 or parameter_count == 0:
        raise ValueError("条件 Jacobian 的两个维度都必须为正")
    if inverse_metric.shape != (parameter_count,):
        raise ValueError("H^{-1} 对角线长度必须等于参数数 p")
    if not bool(
        (torch.isfinite(inverse_metric).all() & (inverse_metric > 0).all()).item()
    ):
        raise ValueError("H^{-1} 对角线必须是有限正数")

    scale = torch.sqrt(inverse_metric)
    weighted = jacobian.detach() * scale.unsqueeze(0)
    left, singular_values, right_transpose = torch.linalg.svd(
        weighted, full_matrices=False
    )
    sigma_max = singular_values[0]
    tolerance_tensor = (
        max(condition_count, parameter_count)
        * torch.finfo(jacobian.dtype).eps
        * sigma_max
    )
    strict_rank = int((singular_values > tolerance_tensor).sum().item())
    lower = tolerance_tensor / hysteresis_factor
    upper = tolerance_tensor * hysteresis_factor
    definitely_retained = int((singular_values > upper).sum().item())
    possibly_retained = int((singular_values > lower).sum().item())
    if previous_rank is None:
        rank = strict_rank
    else:
        bounded_previous = min(max(previous_rank, 0), singular_values.numel())
        rank = min(
            max(bounded_previous, definitely_retained), possibly_retained
        )
    ambiguous = bool(
        (
            (singular_values >= lower) & (singular_values <= upper)
        ).any().item()
    )
    return _ConditionSVD(
        left=left,
        singular_values=singular_values,
        right_transpose=right_transpose,
        scale=scale,
        rank=rank,
        rank_tolerance=float(tolerance_tensor.detach().item()),
        rank_is_stable=not ambiguous,
        condition_count=condition_count,
        parameter_count=parameter_count,
    )


def _condition_measures(
    residual: torch.Tensor, factorization: _ConditionSVD
) -> _ConditionMeasures:
    coordinates = factorization.left.T @ residual
    retained = coordinates[: factorization.rank]
    effective_norm_tensor = torch.linalg.vector_norm(retained)
    residual_norm_tensor = torch.linalg.vector_norm(residual)
    unexplained_squared = torch.clamp(
        residual_norm_tensor.square() - effective_norm_tensor.square(), min=0.0
    )
    stationarity_tensor = torch.linalg.vector_norm(
        factorization.singular_values * coordinates
    )
    scale = factorization.singular_values[0] * residual_norm_tensor
    relative = stationarity_tensor / (
        scale + torch.finfo(residual.dtype).eps
    )
    values = torch.stack(
        (
            residual_norm_tensor,
            effective_norm_tensor,
            stationarity_tensor,
            relative,
            torch.sqrt(unexplained_squared),
        )
    ).detach().cpu().tolist()
    return _ConditionMeasures(*values)


def _condition_correction(
    residual: torch.Tensor, factorization: _ConditionSVD
) -> torch.Tensor:
    if factorization.rank == 0:
        return torch.zeros_like(factorization.scale)
    rank = factorization.rank
    coordinates = (
        factorization.left[:, :rank].T @ residual
    ) / factorization.singular_values[:rank]
    scaled_update = -(factorization.right_transpose[:rank].T @ coordinates)
    return factorization.scale * scaled_update


def _restore_feasibility(
    parameters: _ParameterDict,
    condition_function: _ResidualFunction,
    layout: _ParameterLayout,
    inverse_metric: torch.Tensor,
    tolerance: float,
    max_steps: int,
    settings: _Settings,
    warning_tracker: _RankWarningTracker,
    previous_rank: int | None = None,
) -> _FeasibilityResult:
    current = parameters
    current_rank = previous_rank
    reason = "maximum_iterations"

    for iteration in range(max_steps + 1):
        state = _condition_state(current, condition_function, layout)
        factorization = _factor_condition_jacobian(
            state.jacobian,
            inverse_metric,
            current_rank,
            settings.rank_hysteresis_factor,
        )
        current_rank = factorization.rank
        measures = _condition_measures(state.residual, factorization)
        warning_tracker.report(factorization, measures)
        absolute_feasible = measures.residual_norm <= tolerance
        effective_feasible = (
            measures.effective_residual_norm <= tolerance
            and measures.relative_stationarity
            <= settings.feasibility_stationarity_tolerance
        )
        if absolute_feasible or effective_feasible:
            reason = (
                "absolute_feasibility"
                if absolute_feasible
                else "effective_stationarity"
            )
            return _FeasibilityResult(
                current, True, measures, iteration, factorization.rank, reason, state
            )
        if iteration == max_steps:
            break

        correction = _condition_correction(state.residual, factorization)
        correction_h_norm = torch.linalg.vector_norm(
            correction / factorization.scale
        )
        if not bool(
            (
                torch.isfinite(correction).all()
                & (correction_h_norm > 0.0)
            ).item()
        ):
            reason = "zero_or_nonfinite_correction"
            break

        objective = float(state.loss.detach().item())
        directional_decrease = measures.effective_residual_norm**2
        step_scale = 1.0
        accepted = False
        for _ in range(settings.max_feasibility_backtracks + 1):
            candidate = layout.add(current, step_scale * correction)
            candidate_residual = condition_function(candidate)
            _validate_residual(
                "condition_fn", candidate_residual, layout.device
            )
            candidate_objective = float(
                (0.5 * torch.dot(candidate_residual, candidate_residual)).item()
            )
            armijo_bound = (
                objective
                - settings.armijo_constant
                * step_scale
                * directional_decrease
            )
            if candidate_objective <= armijo_bound:
                current = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            reason = "armijo_line_search_failed"
            break

    # Every exit above leaves current at the already evaluated state.
    return _FeasibilityResult(
        current,
        absolute_feasible or effective_feasible,
        measures,
        min(max_steps, iteration),
        factorization.rank,
        (
            "absolute_feasibility"
            if absolute_feasible
            else "effective_stationarity"
            if effective_feasible
            else reason
        ),
        state,
    )


def _matrix_free_physics_state(
    parameters: _ParameterDict,
    physics_function: _ResidualFunction,
    layout: _ParameterLayout,
) -> _PhysicsState:
    residual, pullback = vjp(physics_function, parameters)
    _validate_residual("physics_fn", residual, layout.device)
    (gradient_tree,) = pullback(residual)
    gradient = layout.flatten(gradient_tree)
    if not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("物理梯度包含 NaN 或 Inf")

    def jacobian_vector_product(vector: torch.Tensor) -> torch.Tensor:
        tangent = layout.unflatten(vector)
        _, product = jvp(
            physics_function,
            (parameters,),
            (tangent,),
        )
        return product

    def vector_jacobian_product(vector: torch.Tensor) -> torch.Tensor:
        (product_tree,) = pullback(vector)
        return layout.flatten(product_tree)

    operator_impl = _CallablePhysicsOperator(
        backend="matrix_free",
        jvp_function=jacobian_vector_product,
        vjp_function=vector_jacobian_product,
    )
    return _PhysicsState(
        residual=residual,
        loss=0.5 * torch.dot(residual, residual),
        gradient=gradient,
        operator=operator_impl,
    )


def _prefer_explicit_backend(
    residual_count: int,
    layout: _ParameterLayout,
    rank: int,
    settings: _Settings,
) -> tuple[bool, int | None]:
    if layout.device.type != "cuda":
        return False, None
    free_bytes, _ = torch.cuda.mem_get_info(layout.device)
    explicit_bytes = residual_count * layout.size * torch.finfo(_DTYPE).bits // 8
    tangent_dimension = max(1, layout.size - rank)
    operator_budget = min(settings.cg_max_iterations, tangent_dimension)
    fits_memory = explicit_bytes <= int(0.35 * free_bytes)
    fits_work = residual_count <= 4 * operator_budget
    if not (fits_memory and fits_work):
        return False, None
    bytes_per_row = layout.size * torch.finfo(_DTYPE).bits // 8
    rows_per_chunk = max(1, int(0.05 * free_bytes) // bytes_per_row)
    return True, min(residual_count, rows_per_chunk)


def _explicit_operator(jacobian: torch.Tensor) -> _PhysicsOperator:
    return _CallablePhysicsOperator(
        backend="explicit",
        jvp_function=lambda vector: jacobian @ vector,
        vjp_function=lambda vector: jacobian.T @ vector,
    )


def _explicit_physics_state(
    parameters: _ParameterDict,
    physics_function: _ResidualFunction,
    layout: _ParameterLayout,
    residual_count: int,
    chunk_size: int,
) -> _PhysicsState:
    residual, jacobian = _residual_and_jacobian(
        parameters, physics_function, layout, "physics_fn", chunk_size
    )
    if residual.numel() != residual_count:
        raise ValueError("physics_fn 的残差维数 m 在训练过程中发生变化")
    operator_impl = _explicit_operator(jacobian)
    gradient = operator_impl.vjp(residual)
    if not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("物理梯度包含 NaN 或 Inf")
    return _PhysicsState(
        residual=residual,
        loss=0.5 * torch.dot(residual, residual),
        gradient=gradient,
        operator=operator_impl,
    )


def _forcing_tolerance(
    measure: float, maximum: float, minimum: float, forcing: float
) -> float:
    scaled = measure / (1.0 + measure) if measure > 0.0 else 0.0
    return max(minimum, min(maximum, forcing * scaled))


def _tangent_defect(
    direction: torch.Tensor, factorization: _ConditionSVD
) -> float:
    if factorization.rank == 0:
        return 0.0
    scaled_direction = direction / factorization.scale
    right_coordinates = (
        factorization.right_transpose[: factorization.rank] @ scaled_direction
    )
    numerator = torch.linalg.vector_norm(
        factorization.singular_values[: factorization.rank]
        * right_coordinates
    )
    denominator = (
        torch.linalg.vector_norm(
            factorization.singular_values[: factorization.rank]
        )
        * torch.linalg.vector_norm(scaled_direction)
        + 1.0e-30
    )
    return float((numerator / denominator).detach().item())


def _projected_cg_synchronous(
    linear_operator: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    project: Callable[[torch.Tensor], torch.Tensor],
    relative_tolerance: float,
    max_iterations: int,
) -> _CGResult:
    """小型显式 GEMM 路径：逐次检查比设备掩码具有更低开销。"""
    right_norm = torch.linalg.vector_norm(right_hand_side)
    if bool((right_norm == 0.0).item()):
        return _CGResult(torch.zeros_like(right_hand_side), 0.0, 0, True)

    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    target = relative_tolerance * right_norm
    iterations = 0
    failed = False

    for iteration in range(max_iterations):
        operator_direction = linear_operator(direction)
        curvature = torch.dot(direction, operator_direction)
        if not bool(
            (torch.isfinite(curvature) & (curvature > 0.0)).item()
        ):
            failed = True
            break
        step_length = residual_squared / curvature
        solution = solution + step_length * direction
        residual = residual - step_length * operator_direction
        iterations = iteration + 1
        if bool((torch.linalg.vector_norm(residual) <= target).item()):
            break
        next_residual_squared = torch.dot(residual, residual)
        direction = residual + (
            next_residual_squared / residual_squared
        ) * direction
        residual_squared = next_residual_squared

    solution = project(solution)
    true_residual = linear_operator(solution) - right_hand_side
    relative_residual = float(
        (
            torch.linalg.vector_norm(true_residual) / right_norm
        ).detach().item()
    )
    success = (
        not failed
        and math.isfinite(relative_residual)
        and relative_residual <= relative_tolerance
    )
    return _CGResult(solution, relative_residual, iterations, success)


def _projected_cg(
    linear_operator: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    project: Callable[[torch.Tensor], torch.Tensor],
    relative_tolerance: float,
    max_iterations: int,
    block_size: int,
) -> _CGResult:
    """CPU checks each step; CUDA optionally supports fixed device-side blocks."""
    if block_size == 1 or right_hand_side.device.type == "cpu":
        return _projected_cg_synchronous(
            linear_operator,
            right_hand_side,
            project,
            relative_tolerance,
            max_iterations,
        )
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    right_norm = torch.sqrt(residual_squared)
    if bool((right_norm == 0.0).item()):
        return _CGResult(solution, 0.0, 0, True)

    target_squared = (relative_tolerance * right_norm).square()
    active = residual_squared > target_squared
    failed = torch.zeros((), dtype=torch.bool, device=right_hand_side.device)
    iterations = 0

    for block_start in range(0, max_iterations, block_size):
        block_iterations = min(block_size, max_iterations - block_start)
        for _ in range(block_iterations):
            operator_direction = linear_operator(direction)
            curvature = torch.dot(direction, operator_direction)
            valid = (
                active
                & torch.isfinite(curvature)
                & torch.isfinite(residual_squared)
                & (curvature > 0.0)
                & (residual_squared > 0.0)
            )
            safe_curvature = torch.where(
                valid, curvature, torch.ones_like(curvature)
            )
            step_length = torch.where(
                valid,
                residual_squared / safe_curvature,
                torch.zeros_like(curvature),
            )
            candidate_solution = solution + step_length * direction
            candidate_residual = residual - step_length * operator_direction
            candidate_squared = torch.dot(
                candidate_residual, candidate_residual
            )
            finite_candidate = (
                torch.isfinite(candidate_squared)
                & torch.isfinite(candidate_solution).all()
                & torch.isfinite(candidate_residual).all()
            )
            keep = valid & finite_candidate
            reached = keep & (candidate_squared <= target_squared)
            failed = failed | (active & ~keep)
            solution = torch.where(keep, candidate_solution, solution)
            residual = torch.where(keep, candidate_residual, residual)
            safe_previous = torch.where(
                keep,
                residual_squared,
                torch.ones_like(residual_squared),
            )
            beta = torch.where(
                keep & ~reached,
                candidate_squared / safe_previous,
                torch.zeros_like(candidate_squared),
            )
            direction = torch.where(
                keep & ~reached,
                candidate_residual + beta * direction,
                torch.zeros_like(direction),
            )
            residual_squared = torch.where(
                keep, candidate_squared, residual_squared
            )
            active = active & keep & ~reached
            iterations += 1
        if not bool(active.item()):
            break

    solution = project(solution)
    true_residual = linear_operator(solution) - right_hand_side
    relative_residual = float(
        (
            torch.linalg.vector_norm(true_residual) / right_norm
        ).detach().item()
    )
    success = (
        not bool(failed.item())
        and math.isfinite(relative_residual)
        and relative_residual <= relative_tolerance
    )
    return _CGResult(solution, relative_residual, iterations, success)


def _solve_lm_step(
    physics: _PhysicsState,
    factorization: _ConditionSVD,
    damping: float,
    linear_tolerance: float,
    settings: _Settings,
) -> _LMStep:
    scaled_gradient = factorization.scale * physics.gradient
    right_hand_side = -factorization.project(scaled_gradient)

    def normal_operator(vector: torch.Tensor) -> torch.Tensor:
        tangent_vector = factorization.project(vector)
        physical_product = physics.operator.jvp(
            factorization.scale * tangent_vector
        )
        gauss_newton_product = factorization.scale * physics.operator.vjp(
            physical_product
        )
        return factorization.project(gauss_newton_product) + (
            damping * tangent_vector
        )

    cg = _projected_cg(
        normal_operator,
        right_hand_side,
        factorization.project,
        linear_tolerance,
        settings.cg_max_iterations,
        1 if physics.operator.backend == "explicit" else settings.cg_block_size,
    )
    scaled_direction = factorization.project(cg.solution)
    direction = factorization.scale * scaled_direction
    linearized_residual = physics.residual + physics.operator.jvp(direction)
    predicted_tensor = physics.loss - 0.5 * torch.dot(
        linearized_residual, linearized_residual
    )
    tangent_defect = _tangent_defect(direction, factorization)
    predicted = float(predicted_tensor.detach().item())
    finite = bool(
        (
            torch.isfinite(direction).all()
            & torch.isfinite(linearized_residual).all()
            & torch.isfinite(predicted_tensor)
        ).item()
    )
    return _LMStep(
        direction=direction,
        predicted_reduction=predicted,
        linear_residual=cg.relative_residual,
        tangent_defect=tangent_defect,
        success=finite and cg.success,
    )


def _evaluate_trial(
    parameters: _ParameterDict,
    step: _LMStep,
    condition_function: _ResidualFunction,
    physics_function: _ResidualFunction,
    layout: _ParameterLayout,
    inverse_metric: torch.Tensor,
    old_physics_loss: float,
    previous_rank: int,
    settings: _Settings,
    warning_tracker: _RankWarningTracker,
) -> _TrialResult:
    step_h_norm_squared = float(
        torch.sum(step.direction.square() / inverse_metric).detach().item()
    )
    restoration_tolerance = max(
        settings.restoration_tolerance_floor,
        min(
            settings.feasibility_tolerance,
            settings.restoration_forcing * step_h_norm_squared,
        ),
    )
    trial_parameters = layout.add(parameters, step.direction)
    restored = _restore_feasibility(
        trial_parameters,
        condition_function,
        layout,
        inverse_metric,
        restoration_tolerance,
        settings.restoration_steps,
        settings,
        warning_tracker,
        previous_rank,
    )
    if not restored.success:
        return _TrialResult(
            parameters,
            False,
            float("nan"),
            old_physics_loss,
            float("-inf"),
            previous_rank,
        )

    candidate_residual = physics_function(restored.parameters)
    _validate_residual("physics_fn", candidate_residual, layout.device)
    candidate_loss = float(
        (0.5 * torch.dot(candidate_residual, candidate_residual)).detach().item()
    )
    actual_reduction = old_physics_loss - candidate_loss
    ratio = (
        actual_reduction / step.predicted_reduction
        if step.predicted_reduction > 0.0
        else float("-inf")
    )
    accepted = bool(
        step.predicted_reduction > 0.0
        and math.isfinite(ratio)
        and ratio >= settings.trust_accept_ratio
    )
    condition_loss = 0.5 * restored.measures.residual_norm**2
    return _TrialResult(
        restored.parameters if accepted else parameters,
        accepted,
        condition_loss,
        candidate_loss if accepted else old_physics_loss,
        ratio,
        restored.rank,
        restored.state if accepted else None,
    )


def _loss_row(condition_loss: torch.Tensor, physics_loss: torch.Tensor) -> list[float]:
    return (
        torch.stack(
            (condition_loss, physics_loss, condition_loss + physics_loss)
        )
        .detach()
        .cpu()
        .tolist()
    )


def _train(
    parameters: _ParameterDict,
    condition_function: _ResidualFunction,
    physics_function: _ResidualFunction,
    layout: _ParameterLayout,
    settings: _Settings,
) -> tuple[_ParameterDict, list[list[float]]]:
    parameters = {name: value.detach() for name, value in parameters.items()}
    warning_tracker = _RankWarningTracker()
    identity_metric = torch.ones(
        layout.size, dtype=_DTYPE, device=layout.device
    )
    initial_feasibility = _restore_feasibility(
        parameters,
        condition_function,
        layout,
        identity_metric,
        settings.feasibility_tolerance,
        settings.initial_feasibility_steps,
        settings,
        warning_tracker,
    )
    if not initial_feasibility.success:
        raise RuntimeError(
            "初始可行性恢复失败："
            f"||c||={initial_feasibility.measures.residual_norm:.3e}, "
            f"effective_residual="
            f"{initial_feasibility.measures.effective_residual_norm:.3e}, "
            f"relative_stationarity="
            f"{initial_feasibility.measures.relative_stationarity:.3e}, "
            f"reason={initial_feasibility.reason}"
        )

    current = initial_feasibility.parameters
    current_condition = initial_feasibility.state
    initial_condition = current_condition.residual
    initial_physics = physics_function(current)
    _validate_residual("condition_fn", initial_condition, layout.device)
    _validate_residual("physics_fn", initial_physics, layout.device)
    initial_condition_loss = 0.5 * torch.dot(initial_condition, initial_condition)
    initial_physics_loss = 0.5 * torch.dot(initial_physics, initial_physics)
    physics_residual_count = initial_physics.numel()
    history = [_loss_row(initial_condition_loss, initial_physics_loss)]
    if history[-1][2] <= settings.loss_tolerance:
        return current, history

    metric = _AdaptiveMetric(settings)
    previous_rank = initial_feasibility.rank
    damping = min(
        settings.lm_damping_max,
        max(settings.lm_damping_min, settings.lm_damping_initial),
    )
    consecutive_rejections = 0

    for _ in range(settings.max_iterations):
        # Parameters are unchanged since restoration; only the metric is updated.
        condition = current_condition
        use_explicit, chunk_size = _prefer_explicit_backend(
            physics_residual_count,
            layout,
            previous_rank,
            settings,
        )
        explicit_oom: str | None = None
        if use_explicit and chunk_size is not None:
            try:
                physics = _explicit_physics_state(
                    current,
                    physics_function,
                    layout,
                    physics_residual_count,
                    chunk_size,
                )
            except torch.OutOfMemoryError as error:
                explicit_oom = str(error)
        else:
            physics = _matrix_free_physics_state(
                current, physics_function, layout
            )
        if explicit_oom is not None:
            torch.cuda.empty_cache()
            warnings.warn(
                "显式物理 Jacobian 超出可用显存，"
                "已回退矩阵自由后端："
                f"{explicit_oom}",
                RuntimeWarning,
                stacklevel=3,
            )
            physics = _matrix_free_physics_state(
                current, physics_function, layout
            )
        if physics.residual.numel() != physics_residual_count:
            raise ValueError("physics_fn 的残差维数 m 在训练过程中发生变化")

        inverse_metric = metric.update(physics.gradient)
        factorization = _factor_condition_jacobian(
            condition.jacobian,
            inverse_metric,
            previous_rank,
            settings.rank_hysteresis_factor,
        )
        previous_rank = factorization.rank
        measures = _condition_measures(condition.residual, factorization)
        warning_tracker.report(factorization, measures)

        projected_gradient = factorization.project(
            factorization.scale * physics.gradient
        )
        stationarity = float(
            torch.linalg.vector_norm(projected_gradient).detach().item()
        )
        if (
            measures.effective_residual_norm
            <= settings.feasibility_tolerance
            and measures.relative_stationarity
            <= settings.feasibility_stationarity_tolerance
            and stationarity <= settings.stationarity_tolerance
        ):
            break
        if stationarity <= settings.step_tolerance:
            break

        tangent_tolerance = _forcing_tolerance(
            stationarity,
            settings.tangent_tolerance_max,
            settings.tangent_tolerance_min,
            settings.tangent_forcing,
        )
        linear_tolerance = _forcing_tolerance(
            stationarity,
            settings.linear_tolerance_max,
            settings.linear_tolerance_min,
            settings.linear_forcing,
        )
        old_physics_loss = float(physics.loss.detach().item())
        accepted_trial: _TrialResult | None = None
        small_step = False

        for _ in range(settings.max_lm_attempts):
            step = _solve_lm_step(
                physics,
                factorization,
                damping,
                linear_tolerance,
                settings,
            )
            step_h_norm = float(
                torch.linalg.vector_norm(
                    step.direction / factorization.scale
                ).detach().item()
            )
            if (
                not step.success
                or step.predicted_reduction <= 0.0
                or step.tangent_defect > tangent_tolerance
            ):
                damping = min(
                    settings.lm_damping_max,
                    damping * settings.damping_increase,
                )
                continue
            if step_h_norm <= settings.step_tolerance:
                small_step = True
                break

            trial = _evaluate_trial(
                current,
                step,
                condition_function,
                physics_function,
                layout,
                inverse_metric,
                old_physics_loss,
                factorization.rank,
                settings,
                warning_tracker,
            )
            if trial.accepted:
                accepted_trial = trial
                current = trial.parameters
                assert trial.condition_state is not None
                current_condition = trial.condition_state
                previous_rank = trial.rank
                if trial.reduction_ratio >= settings.trust_good_ratio:
                    damping = max(
                        settings.lm_damping_min,
                        damping * settings.damping_decrease,
                    )
                elif trial.reduction_ratio < settings.trust_bad_ratio:
                    damping = min(
                        settings.lm_damping_max,
                        damping * settings.damping_increase,
                    )
                break
            damping = min(
                settings.lm_damping_max,
                damping * settings.damping_increase,
            )

        if small_step:
            break

        if accepted_trial is None:
            consecutive_rejections += 1
            row = _loss_row(condition.loss, physics.loss)
        else:
            consecutive_rejections = 0
            row = [
                accepted_trial.condition_loss,
                accepted_trial.physics_loss,
                accepted_trial.condition_loss + accepted_trial.physics_loss,
            ]
        history.append(row)

        if accepted_trial is not None and row[2] <= settings.loss_tolerance:
            break
        if consecutive_rejections >= settings.max_consecutive_rejections:
            warnings.warn(
                "连续 LM 试探均被拒绝，FTP 已在最后一个可行参数处停止",
                RuntimeWarning,
                stacklevel=3,
            )
            break

    return current, history


def _copy_parameters_to_model(
    model: torch.nn.Module, parameters: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, target in model.named_parameters():
            target.copy_(parameters[name])


def FTP(model, condition_fn, physics_fn, max_iteration, loss_tol=1.0e-10):
    """在条件数值流形上以切空间 LM 训练模型，并保持旧四元组接口。

    Residual callbacks must be deterministic functions of the supplied parameters:
    keep samples and buffers fixed during one FTP call. Condition states are
    reused only while the parameter point is unchanged; the weighted SVD is
    recomputed after every metric update.
    """
    max_iterations = operator.index(max_iteration)
    loss_tolerance = float(loss_tol)
    if max_iterations < 0:
        raise ValueError("max_iteration 必须是非负整数")
    if not math.isfinite(loss_tolerance) or loss_tolerance < 0.0:
        raise ValueError("loss_tol 必须是有限非负数")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model 必须是 torch.nn.Module")
    if not callable(condition_fn) or not callable(physics_fn):
        raise TypeError("condition_fn 和 physics_fn 必须可调用")

    model.to(dtype=_DTYPE)
    parameters = dict(model.named_parameters())
    layout = _ParameterLayout.from_parameters(parameters)
    settings = _Settings(max_iterations, loss_tolerance)
    is_cuda = layout.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(layout.device)
        torch.cuda.reset_peak_memory_stats(layout.device)
    start = perf_counter()

    condition_function, physics_function = _prepare_residual_functions(
        parameters,
        condition_fn,
        physics_fn,
        layout,
    )
    trained_parameters, loss_history = _train(
        parameters,
        condition_function,
        physics_function,
        layout,
        settings,
    )
    _copy_parameters_to_model(model, trained_parameters)

    if is_cuda:
        torch.cuda.synchronize(layout.device)
    elapsed_time = perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(layout.device)) if is_cuda else 0
    )
    return model, loss_history, elapsed_time, peak_memory
