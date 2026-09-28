from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import time

import torch
from torch.func import jacrev, vjp


DEFAULT_DTYPE = torch.float64
PHYSICS_JACOBIAN_CHUNK_SIZE = 512
ParameterDict = dict[str, torch.Tensor]
ResidualFunction = Callable[[ParameterDict], torch.Tensor]


@dataclass(frozen=True)
class ResidualEvaluation:
    c: torch.Tensor
    r: torch.Tensor
    loss_c: torch.Tensor
    loss_r: torch.Tensor
    jacobian_c: torch.Tensor
    jacobian_r: torch.Tensor
    gradient_r: torch.Tensor


@dataclass(frozen=True)
class ConditionDiagnostics:
    condition_count: int
    parameter_count: int
    effective_rank: int
    rank_tolerance: float
    lambda_min: float
    lambda_max: float
    condition_number: float
    is_full_row_rank: bool
    is_near_singular: bool
    gram_matrix: torch.Tensor | None
    singular_values: torch.Tensor
    spectral_gap_ratio: float = float("inf")
    rank_is_stable: bool = True


@dataclass(frozen=True)
class ConditionDecomposition:
    """B=J_c H^{-1/2} 的一次截断 SVD；所有条件算子共享此对象。"""

    left_singular_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_singular_vectors_transpose: torch.Tensor
    sqrt_inverse_metric: torch.Tensor
    effective_rank: int
    rank_tolerance: float
    spectral_gap_ratio: float
    rank_is_stable: bool


@dataclass(frozen=True)
class FTPState:
    c: torch.Tensor
    r: torch.Tensor
    loss_c: torch.Tensor
    loss_r: torch.Tensor
    jacobian_c: torch.Tensor
    gradient_r: torch.Tensor


@dataclass(frozen=True)
class MetricUpdate:
    inverse_diagonal: torch.Tensor
    clipped_fraction: float


@dataclass(frozen=True)
class ConditionCorrection:
    update: torch.Tensor
    gram_matrix: torch.Tensor | None
    damping: float
    used_pseudoinverse: bool
    effective_rank: int = 0


@dataclass(frozen=True)
class FeasibilityResult:
    parameters: ParameterDict
    residual_norm: float
    iterations: int
    success: bool
    correction_norm: float
    damping: float
    projected_residual_norm: float = float("inf")
    stationarity_norm: float = float("inf")
    relative_stationarity: float = float("inf")
    attainable_condition_norm: float = float("inf")
    effective_rank: int = 0
    rank_tolerance: float = 0.0
    rank_is_stable: bool = True
    absolute_feasible: bool = False
    termination_reason: str = "unknown"


@dataclass(frozen=True)
class ProjectionResult:
    direction: torch.Tensor
    gram_matrix: torch.Tensor | None
    damping: float
    used_pseudoinverse: bool
    effective_rank: int = 0


@dataclass(frozen=True)
class ProjectedStationarity:
    projection: ProjectionResult
    norm: float
    tangent_defect: float
    tangent_tolerance: float
    success: bool


@dataclass(frozen=True)
class FeasibleLMStep:
    direction: torch.Tensor
    multipliers: torch.Tensor
    linearized_residual: torch.Tensor
    predicted_reduction: float
    linear_solve_residual: float
    tangent_defect: float
    success: bool
    used_least_squares: bool
    tangent_iterations: int = 0


@dataclass(frozen=True)
class LMTrialResult:
    parameters: ParameterDict
    accepted: bool
    physics_loss: float
    condition_norm: float
    restoration_steps: int
    restoration_tolerance: float
    actual_reduction: float
    predicted_reduction: float
    reduction_ratio: float
    projected_condition_norm: float = float("inf")
    constraint_stationarity_norm: float = float("inf")
    relative_constraint_stationarity: float = float("inf")
    attainable_condition_norm: float = float("inf")
    effective_rank: int = 0
    rank_tolerance: float = 0.0


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    condition_norm: float
    condition_loss: float
    physics_loss: float
    total_loss: float
    step_size: float
    accepted: bool
    tangent_defect: float
    lambda_min: float
    condition_number: float
    projection_damping: float
    restoration_steps: int
    backtracks: int
    clipped_fraction: float
    lm_damping: float
    predicted_reduction: float
    reduction_ratio: float
    linear_solve_residual: float
    linear_solve_tolerance: float
    stationarity_norm: float
    tangent_tolerance: float
    restoration_tolerance: float
    effective_rank: int = 0
    rank_tolerance: float = 0.0
    rank_is_stable: bool = True
    spectral_gap_ratio: float = float("inf")
    projected_condition_norm: float = float("inf")
    constraint_stationarity_norm: float = float("inf")
    relative_constraint_stationarity: float = float("inf")
    attainable_condition_norm: float = float("inf")


@dataclass(frozen=True)
class FTPConfig:
    max_iterations: int = 40
    feasibility_tolerance: float = 1.0e-8
    initial_feasibility_steps: int = 200
    restoration_steps: int = 6
    restoration_relative_damping: float = 1.0e-8
    projection_relative_damping: float = 1.0e-8
    beta2: float = 0.9
    metric_epsilon: float = 1.0e-12
    min_inverse_metric: float = 1.0e-3
    max_inverse_metric: float = 1.0e3
    initial_step_size: float = 1.0e-5
    max_step_size: float = 1.0
    lm_damping_initial: float = 1.0e-3
    lm_damping_min: float = 1.0e-12
    lm_damping_max: float = 1.0e12
    trust_accept_ratio: float = 1.0e-1
    trust_bad_ratio: float = 2.5e-1
    trust_good_ratio: float = 7.5e-1
    damping_increase: float = 4.0
    damping_decrease: float = 5.0e-1
    max_lm_attempts: int = 8
    restoration_forcing: float = 1.0e-1
    restoration_tolerance_floor: float = 1.0e-12
    tangent_tolerance_max: float = 1.0e-4
    tangent_tolerance_min: float = 1.0e-12
    tangent_forcing: float = 1.0e-1
    max_projection_attempts: int = 12
    linear_solve_tolerance_max: float = 1.0e-1
    linear_solve_tolerance_min: float = 1.0e-12
    linear_solve_forcing: float = 1.0e-1
    condition_tolerance: float = 1.0e-8
    stationarity_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-12
    physics_loss_tolerance: float = 1.0e-8
    enable_normalized_physics_stopping: bool = False
    max_consecutive_rejections: int = 3
    condition_warning_threshold: float = 1.0e12
    feasibility_stationarity_tolerance: float = 1.0e-8
    rank_hysteresis_factor: float = 2.0
    armijo_constant: float = 1.0e-4
    max_feasibility_backtracks: int = 20
    tangent_cg_max_iterations: int = 500


@dataclass(frozen=True)
class TrainingResult:
    parameters: ParameterDict
    history: tuple[TrainingRecord, ...]
    initial_feasibility: FeasibilityResult
    initial_state: FTPState
    final_state: FTPState
    termination_reason: str


class AdaptiveDiagonalMetric:
    def __init__(
        self,
        beta2: float,
        epsilon: float,
        min_inverse: float,
        max_inverse: float,
    ) -> None:
        self.beta2 = beta2
        self.epsilon = epsilon
        self.min_inverse = min_inverse
        self.max_inverse = max_inverse
        self.step = 0
        self.second_moment: torch.Tensor | None = None

    def update(self, gradient_r: torch.Tensor) -> MetricUpdate:
        gradient = gradient_r.detach()
        if self.second_moment is None:
            self.second_moment = torch.zeros_like(gradient)

        self.step += 1
        self.second_moment = (
            self.beta2 * self.second_moment
            + (1.0 - self.beta2) * gradient.square()
        )
        corrected_moment = self.second_moment / (1.0 - self.beta2**self.step)

        # H_k^{-1} 对应文档 (3.5)-(3.7)
        raw_inverse = torch.rsqrt(corrected_moment + self.epsilon)
        inverse_diagonal = raw_inverse.clamp(self.min_inverse, self.max_inverse)
        clipped_fraction = float(
            ((raw_inverse < self.min_inverse) | (raw_inverse > self.max_inverse))
            .to(gradient.dtype)
            .mean()
            .item()
        )
        return MetricUpdate(inverse_diagonal, clipped_fraction)


def flatten_parameter_jacobian(
    jacobian: Mapping[str, torch.Tensor], residual_count: int
) -> torch.Tensor:
    """将参数 PyTree Jacobian 展平为 residual_count × p 矩阵。"""
    return torch.cat(
        [block.reshape(residual_count, -1) for block in jacobian.values()], dim=1
    )


def flatten_parameter_vector(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """按参数字典顺序展平并拼接向量。"""
    return torch.cat([value.reshape(-1) for value in values.values()])


def validate_float64_parameters(parameters: ParameterDict) -> None:
    """强制所有参数位于同一设备并使用 float64。"""
    if not parameters:
        raise ValueError("参数字典不能为空")
    devices = {value.device for value in parameters.values()}
    if len(devices) != 1:
        raise ValueError("所有模型参数必须位于同一设备")
    if any(value.dtype != DEFAULT_DTYPE for value in parameters.values()):
        raise ValueError("FTP 的所有模型参数必须使用 torch.float64")


def add_parameter_vector(
    parameters: ParameterDict, update: torch.Tensor
) -> ParameterDict:
    """返回 theta+update 的函数式参数字典。"""
    parameter_count = sum(value.numel() for value in parameters.values())
    if update.shape != (parameter_count,):
        raise ValueError("参数更新向量的长度必须等于 p")

    updated: ParameterDict = {}
    offset = 0
    for name, value in parameters.items():
        next_offset = offset + value.numel()
        block = update[offset:next_offset].reshape_as(value)
        updated[name] = (value + block).detach()
        offset = next_offset
    return updated


def copy_parameters_to_model(
    model: torch.nn.Module, parameters: ParameterDict
) -> None:
    """将函数式参数复制回 nn.Module。"""
    with torch.no_grad():
        for name, target in model.named_parameters():
            target.copy_(parameters[name])


def evaluate_residual_system(
    parameters: ParameterDict,
    condition_residual: ResidualFunction,
    physics_residual: ResidualFunction,
) -> ResidualEvaluation:
    """计算 c、r、对应损失、Jacobian 与物理梯度。"""
    c = condition_residual(parameters)
    r = physics_residual(parameters)
    if c.dtype != DEFAULT_DTYPE or r.dtype != DEFAULT_DTYPE:
        raise ValueError("条件残差和物理残差必须使用 torch.float64")
    if c.device != r.device:
        raise ValueError("条件残差和物理残差必须位于同一设备")

    # J_c = Dc，J_r = Dr，对应文档 (1.2)
    jacobian_c_tree = jacrev(condition_residual)(parameters)
    jacobian_r_tree = jacrev(
        physics_residual, chunk_size=PHYSICS_JACOBIAN_CHUNK_SIZE
    )(parameters)
    jacobian_c = flatten_parameter_jacobian(jacobian_c_tree, c.numel())
    jacobian_r = flatten_parameter_jacobian(jacobian_r_tree, r.numel())

    # L_c、L_r 对应文档 (1.1)
    loss_c = 0.5 * torch.dot(c, c)
    loss_r = 0.5 * torch.dot(r, r)

    # g_r = J_r^T r，对应文档 (1.4)
    gradient_r = jacobian_r.T @ r

    return ResidualEvaluation(
        c=c,
        r=r,
        loss_c=loss_c,
        loss_r=loss_r,
        jacobian_c=jacobian_c,
        jacobian_r=jacobian_r,
        gradient_r=gradient_r,
    )


def evaluate_ftp_state(
    parameters: ParameterDict,
    condition_residual: ResidualFunction,
    physics_residual: ResidualFunction,
) -> FTPState:
    """计算训练所需状态，不显式构造 J_r。"""
    c = condition_residual(parameters)
    if c.dtype != DEFAULT_DTYPE:
        raise ValueError("条件残差必须使用 torch.float64")
    jacobian_c_tree = jacrev(condition_residual)(parameters)
    jacobian_c = flatten_parameter_jacobian(jacobian_c_tree, c.numel())

    r, physics_pullback = vjp(physics_residual, parameters)
    if r.dtype != DEFAULT_DTYPE or r.device != c.device:
        raise ValueError("物理残差必须与条件残差同设备且使用 torch.float64")
    (gradient_r_tree,) = physics_pullback(r)
    gradient_r = flatten_parameter_vector(gradient_r_tree)

    return FTPState(
        c=c,
        r=r,
        loss_c=0.5 * torch.dot(c, c),
        loss_r=0.5 * torch.dot(r, r),
        jacobian_c=jacobian_c,
        gradient_r=gradient_r,
    )


def decompose_condition_jacobian(
    condition_jacobian: torch.Tensor,
    inverse_metric_diagonal: torch.Tensor | None = None,
    previous_rank: int | None = None,
    rank_hysteresis_factor: float = 2.0,
) -> ConditionDecomposition:
    """直接分解 B=J_c H^{-1/2}，不形成法方程或 Gram 矩阵。"""
    if condition_jacobian.ndim != 2:
        raise ValueError("condition_jacobian 必须是二维矩阵")

    jacobian = condition_jacobian.detach()
    condition_count, parameter_count = jacobian.shape
    if condition_count == 0 or parameter_count == 0:
        raise ValueError("J_c 的两个维度都必须为正")
    if condition_count > parameter_count:
        raise ValueError(
            f"条件数 q={condition_count} 不能大于参数数 p={parameter_count}"
        )
    if not torch.isfinite(jacobian).all():
        raise ValueError("J_c 包含 NaN 或 Inf")
    if jacobian.dtype != DEFAULT_DTYPE:
        raise ValueError("条件 Jacobian 必须使用 torch.float64")

    if inverse_metric_diagonal is None:
        inverse_metric_diagonal = torch.ones(
            parameter_count, dtype=jacobian.dtype, device=jacobian.device
        )
    else:
        inverse_metric_diagonal = inverse_metric_diagonal.detach()
        if inverse_metric_diagonal.shape != (parameter_count,):
            raise ValueError("H^{-1} 对角线的长度必须等于参数数 p")
        if not torch.isfinite(inverse_metric_diagonal).all() or torch.any(
            inverse_metric_diagonal <= 0
        ):
            raise ValueError("H^{-1} 对角线必须是有限正数")
        if (
            inverse_metric_diagonal.dtype != jacobian.dtype
            or inverse_metric_diagonal.device != jacobian.device
        ):
            raise ValueError("H^{-1} 对角线必须与 J_c 同设备、同 dtype")

    if rank_hysteresis_factor < 1.0:
        raise ValueError("秩滞回因子必须不小于 1")

    sqrt_inverse_metric = torch.sqrt(inverse_metric_diagonal)
    weighted_jacobian = jacobian * sqrt_inverse_metric.unsqueeze(0)
    left_vectors, singular_values, right_vectors_transpose = torch.linalg.svd(
        weighted_jacobian, full_matrices=False
    )
    rank_tolerance_tensor = (
        max(condition_count, parameter_count)
        * torch.finfo(jacobian.dtype).eps
        * singular_values[0]
    )
    strict_rank = int((singular_values > rank_tolerance_tensor).sum().item())

    lower_threshold = rank_tolerance_tensor / rank_hysteresis_factor
    upper_threshold = rank_tolerance_tensor * rank_hysteresis_factor
    definitely_retained = int((singular_values > upper_threshold).sum().item())
    possibly_retained = int((singular_values > lower_threshold).sum().item())
    if previous_rank is None:
        effective_rank = strict_rank
    else:
        bounded_previous_rank = min(max(previous_rank, 0), condition_count)
        effective_rank = min(
            max(bounded_previous_rank, definitely_retained), possibly_retained
        )

    if effective_rank == 0 or effective_rank == condition_count:
        spectral_gap_ratio = float("inf")
    else:
        spectral_gap_ratio = float(
            (singular_values[effective_rank - 1] / singular_values[effective_rank])
            .detach()
            .item()
        )
    ambiguous_rank = bool(
        ((singular_values >= lower_threshold) & (singular_values <= upper_threshold))
        .any()
        .item()
    )
    rank_is_stable = not ambiguous_rank

    return ConditionDecomposition(
        left_singular_vectors=left_vectors,
        singular_values=singular_values,
        right_singular_vectors_transpose=right_vectors_transpose,
        sqrt_inverse_metric=sqrt_inverse_metric,
        effective_rank=effective_rank,
        rank_tolerance=float(rank_tolerance_tensor.detach().item()),
        spectral_gap_ratio=spectral_gap_ratio,
        rank_is_stable=rank_is_stable,
    )


def diagnose_condition_jacobian(
    condition_jacobian: torch.Tensor,
    inverse_metric_diagonal: torch.Tensor | None = None,
    condition_warning_threshold: float = 1.0e12,
    decomposition: ConditionDecomposition | None = None,
) -> ConditionDiagnostics:
    """由 B 的共享 SVD 诊断数值秩和谱，绝不形成 B B^T。"""
    if decomposition is None:
        decomposition = decompose_condition_jacobian(
            condition_jacobian, inverse_metric_diagonal
        )
    condition_count, parameter_count = condition_jacobian.shape
    singular_values = decomposition.singular_values
    effective_rank = decomposition.effective_rank
    lambda_max = float(singular_values[0].square().detach().item())
    lambda_min = float(singular_values[-1].square().detach().item())
    is_full_row_rank = effective_rank == condition_count
    condition_number = (
        lambda_max / lambda_min
        if is_full_row_rank and lambda_min > 0.0
        else float("inf")
    )

    return ConditionDiagnostics(
        condition_count=condition_count,
        parameter_count=parameter_count,
        effective_rank=effective_rank,
        rank_tolerance=decomposition.rank_tolerance,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_number=condition_number,
        is_full_row_rank=is_full_row_rank,
        is_near_singular=(
            not is_full_row_rank or condition_number >= condition_warning_threshold
        ),
        gram_matrix=None,
        singular_values=singular_values,
        spectral_gap_ratio=decomposition.spectral_gap_ratio,
        rank_is_stable=decomposition.rank_is_stable,
    )


def damped_condition_correction(
    condition_residual: torch.Tensor,
    decomposition: ConditionDecomposition,
) -> ConditionCorrection:
    """计算 H-最小范数截断 Gauss--Newton 校正。"""
    rank = decomposition.effective_rank
    if rank == 0:
        update = torch.zeros_like(decomposition.sqrt_inverse_metric)
    else:
        left = decomposition.left_singular_vectors[:, :rank]
        right_transpose = decomposition.right_singular_vectors_transpose[:rank, :]
        singular_values = decomposition.singular_values[:rank]
        scaled_coordinates = (left.T @ condition_residual) / singular_values
        scaled_update = -(right_transpose.T @ scaled_coordinates)
        update = decomposition.sqrt_inverse_metric * scaled_update
    return ConditionCorrection(
        update=update,
        gram_matrix=None,
        damping=0.0,
        used_pseudoinverse=True,
        effective_rank=rank,
    )


def condition_optimality_measures(
    condition_residual: torch.Tensor,
    decomposition: ConditionDecomposition,
) -> tuple[float, float, float, float]:
    """返回值域残差、平稳性、相对平稳性与不可消残差范数。"""
    rank = decomposition.effective_rank
    left_coordinates = decomposition.left_singular_vectors.T @ condition_residual
    projected = left_coordinates[:rank]
    discarded = left_coordinates[rank:]
    projected_norm = torch.linalg.vector_norm(projected)
    attainable_norm = torch.linalg.vector_norm(discarded)
    stationarity = torch.linalg.vector_norm(
        decomposition.singular_values * left_coordinates
    )
    residual_norm = torch.linalg.vector_norm(condition_residual)
    scale = decomposition.singular_values[0] * residual_norm
    epsilon = torch.finfo(condition_residual.dtype).eps
    relative_stationarity = stationarity / (scale + epsilon)
    return (
        float(projected_norm.detach().item()),
        float(stationarity.detach().item()),
        float(relative_stationarity.detach().item()),
        float(attainable_norm.detach().item()),
    )


def restore_feasibility(
    parameters: ParameterDict,
    condition_residual: ResidualFunction,
    inverse_metric_diagonal: torch.Tensor,
    tolerance: float,
    max_steps: int,
    relative_damping: float,
    stationarity_tolerance: float = 1.0e-8,
    rank_hysteresis_factor: float = 2.0,
    armijo_constant: float = 1.0e-4,
    max_backtracks: int = 20,
    previous_rank: int | None = None,
) -> FeasibilityResult:
    """以截断 GN 恢复数值可行性；绝对可行或数值平稳均成功。"""
    del relative_damping  # 为兼容旧调用保留；截断 SVD 不使用 Gram 阻尼。
    current_parameters = parameters
    total_correction_norm = 0.0
    current_rank = previous_rank
    last_decomposition: ConditionDecomposition | None = None
    last_measures = (float("inf"), float("inf"), float("inf"), float("inf"))
    termination_reason = "maximum_iterations"

    for iteration in range(max_steps + 1):
        c = condition_residual(current_parameters)
        residual_norm = float(torch.linalg.vector_norm(c).item())
        jacobian_tree = jacrev(condition_residual)(current_parameters)
        jacobian = flatten_parameter_jacobian(jacobian_tree, c.numel())
        decomposition = decompose_condition_jacobian(
            jacobian,
            inverse_metric_diagonal,
            current_rank,
            rank_hysteresis_factor,
        )
        current_rank = decomposition.effective_rank
        last_decomposition = decomposition
        measures = condition_optimality_measures(c, decomposition)
        last_measures = measures
        projected_norm, stationarity_norm, relative_stationarity, attainable_norm = (
            measures
        )
        absolute_feasible = residual_norm <= tolerance
        numerically_stationary = (
            projected_norm <= tolerance
            and relative_stationarity <= stationarity_tolerance
        )
        if absolute_feasible or numerically_stationary:
            termination_reason = (
                "absolute_feasibility"
                if absolute_feasible
                else "numerical_stationarity"
            )
            return FeasibilityResult(
                current_parameters,
                residual_norm,
                iteration,
                True,
                total_correction_norm,
                0.0,
                projected_norm,
                stationarity_norm,
                relative_stationarity,
                attainable_norm,
                decomposition.effective_rank,
                decomposition.rank_tolerance,
                decomposition.rank_is_stable,
                absolute_feasible,
                termination_reason,
            )
        if iteration == max_steps:
            break

        correction = damped_condition_correction(c, decomposition)
        correction_h_norm = torch.linalg.vector_norm(
            correction.update / decomposition.sqrt_inverse_metric
        )
        if not torch.isfinite(correction.update).all() or correction_h_norm == 0.0:
            termination_reason = "zero_or_nonfinite_correction"
            break

        accepted_correction = False
        step_scale = 1.0
        current_objective = 0.5 * residual_norm**2
        directional_decrease = projected_norm**2
        for _ in range(max_backtracks + 1):
            scaled_update = step_scale * correction.update
            candidate = add_parameter_vector(current_parameters, scaled_update)
            candidate_c = condition_residual(candidate)
            candidate_objective = 0.5 * float(torch.dot(candidate_c, candidate_c).item())
            armijo_bound = (
                current_objective
                - armijo_constant * step_scale * directional_decrease
            )
            if torch.isfinite(candidate_c).all() and candidate_objective <= armijo_bound:
                current_parameters = candidate
                total_correction_norm += float(
                    (step_scale * correction_h_norm).detach().item()
                )
                accepted_correction = True
                break
            step_scale *= 0.5

        if not accepted_correction:
            termination_reason = "armijo_line_search_failed"
            break

    final_c = condition_residual(current_parameters)
    final_norm = float(torch.linalg.vector_norm(final_c).item())
    if last_decomposition is None:
        raise RuntimeError("可行性恢复未能构造条件 Jacobian 分解")
    projected_norm, stationarity_norm, relative_stationarity, attainable_norm = (
        last_measures
    )
    absolute_feasible = final_norm <= tolerance
    numerically_stationary = (
        projected_norm <= tolerance
        and relative_stationarity <= stationarity_tolerance
    )
    return FeasibilityResult(
        current_parameters,
        final_norm,
        min(max_steps, iteration),
        absolute_feasible or numerically_stationary,
        total_correction_norm,
        0.0,
        projected_norm,
        stationarity_norm,
        relative_stationarity,
        attainable_norm,
        last_decomposition.effective_rank,
        last_decomposition.rank_tolerance,
        last_decomposition.rank_is_stable,
        absolute_feasible,
        (
            "absolute_feasibility"
            if absolute_feasible
            else "numerical_stationarity"
            if numerically_stationary
            else termination_reason
        ),
    )


def projected_physics_direction(
    gradient_r: torch.Tensor,
    decomposition: ConditionDecomposition,
) -> ProjectionResult:
    """用共享 V_k 在 H-尺度坐标中施加正交切空间投影。"""
    scaled_gradient = decomposition.sqrt_inverse_metric * gradient_r
    projected_scaled_gradient = project_scaled_tangent(
        scaled_gradient, decomposition
    )
    direction = -decomposition.sqrt_inverse_metric * projected_scaled_gradient
    return ProjectionResult(
        direction=direction,
        gram_matrix=None,
        damping=0.0,
        used_pseudoinverse=True,
        effective_rank=decomposition.effective_rank,
    )


def project_scaled_tangent(
    vector: torch.Tensor, decomposition: ConditionDecomposition
) -> torch.Tensor:
    """计算 Q_k vector，其中 Q_k=I-V_k V_k^T。"""
    rank = decomposition.effective_rank
    if rank == 0:
        return vector
    right_transpose = decomposition.right_singular_vectors_transpose[:rank, :]
    return vector - right_transpose.T @ (right_transpose @ vector)


def relative_tangent_defect(
    direction: torch.Tensor,
    decomposition: ConditionDecomposition,
    numerical_epsilon: float = 1.0e-30,
) -> float:
    """仅在共享数值行空间 B_k 上计算尺度不变的切向缺陷。"""
    scaled_direction = direction / decomposition.sqrt_inverse_metric
    rank = decomposition.effective_rank
    if rank == 0:
        return 0.0
    right_coordinates = (
        decomposition.right_singular_vectors_transpose[:rank, :]
        @ scaled_direction
    )
    numerator = torch.linalg.vector_norm(
        decomposition.singular_values[:rank] * right_coordinates
    )
    denominator = (
        torch.linalg.vector_norm(decomposition.singular_values[:rank])
        * torch.linalg.vector_norm(scaled_direction)
        + numerical_epsilon
    )
    return float((numerator / denominator).item())


def forcing_tolerance(
    measure: float,
    maximum: float,
    minimum: float,
    forcing: float,
) -> float:
    """将当前最优性度量转换为有上下界的 forcing sequence。"""
    if not 0.0 < minimum <= maximum:
        raise ValueError("forcing sequence 要求 0 < minimum <= maximum")
    scaled_measure = measure / (1.0 + measure) if measure > 0.0 else 0.0
    return max(minimum, min(maximum, forcing * scaled_measure))


def evaluate_projected_stationarity(
    gradient_r: torch.Tensor,
    decomposition: ConditionDecomposition,
    config: FTPConfig,
) -> ProjectedStationarity:
    """计算 ||Q_k H^{-1/2}g||，投影与切向检验共享同一 V_k。"""
    projection = projected_physics_direction(gradient_r, decomposition)
    stationarity = torch.linalg.vector_norm(
        projection.direction / decomposition.sqrt_inverse_metric
    )
    stationarity_norm = float(stationarity.item())
    tangent_tolerance = forcing_tolerance(
        stationarity_norm,
        config.tangent_tolerance_max,
        config.tangent_tolerance_min,
        config.tangent_forcing,
    )
    tangent_defect = relative_tangent_defect(projection.direction, decomposition)
    return ProjectedStationarity(
        projection=projection,
        norm=stationarity_norm,
        tangent_defect=tangent_defect,
        tangent_tolerance=tangent_tolerance,
        success=tangent_defect <= tangent_tolerance,
    )


def solve_projected_lm_system(
    weighted_physics_jacobian: torch.Tensor,
    physics_residual: torch.Tensor,
    decomposition: ConditionDecomposition,
    damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, float, int]:
    """投影 CG 求解 (Q C^T C Q+lambda I)z=-Q C^T r。"""
    gradient = project_scaled_tangent(
        weighted_physics_jacobian.T @ physics_residual, decomposition
    )
    right_hand_side = -gradient
    right_norm = torch.linalg.vector_norm(right_hand_side)
    if right_norm == 0.0:
        return torch.zeros_like(right_hand_side), 0.0, 0

    def operator(vector: torch.Tensor) -> torch.Tensor:
        tangent_vector = project_scaled_tangent(vector, decomposition)
        gauss_newton_product = weighted_physics_jacobian.T @ (
            weighted_physics_jacobian @ tangent_vector
        )
        return project_scaled_tangent(gauss_newton_product, decomposition) + (
            damping * tangent_vector
        )

    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    search_direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    target = relative_tolerance * right_norm
    iterations = 0

    for iteration in range(max_iterations):
        operator_search = operator(search_direction)
        curvature = torch.dot(search_direction, operator_search)
        if not torch.isfinite(curvature) or curvature <= 0.0:
            break
        step_length = residual_squared / curvature
        solution = solution + step_length * search_direction
        residual = residual - step_length * operator_search
        iterations = iteration + 1
        if torch.linalg.vector_norm(residual) <= target:
            break
        next_residual_squared = torch.dot(residual, residual)
        search_direction = residual + (
            next_residual_squared / residual_squared
        ) * search_direction
        residual_squared = next_residual_squared

    solution = project_scaled_tangent(solution, decomposition)
    relative_residual = float(
        (torch.linalg.vector_norm(operator(solution) - right_hand_side) / right_norm)
        .detach()
        .item()
    )
    return solution, relative_residual, iterations


def solve_feasible_lm_step(
    state: ResidualEvaluation,
    decomposition: ConditionDecomposition,
    damping: float,
    linear_solve_tolerance: float,
    max_cg_iterations: int = 500,
) -> FeasibleLMStep:
    """用共享数值切空间求解阻尼 Gauss--Newton 子问题。"""
    parameter_count = state.gradient_r.numel()
    condition_count = state.c.numel()
    if decomposition.sqrt_inverse_metric.shape != (parameter_count,):
        raise ValueError("条件分解的尺度向量长度必须等于参数数 p")
    if damping <= 0.0:
        raise ValueError("LM 阻尼必须为正数")

    weighted_physics_jacobian = (
        state.jacobian_r * decomposition.sqrt_inverse_metric.unsqueeze(0)
    )
    scaled_direction, relative_solve_residual, tangent_iterations = (
        solve_projected_lm_system(
            weighted_physics_jacobian,
            state.r,
            decomposition,
            damping,
            linear_solve_tolerance,
            max_cg_iterations,
        )
    )
    direction = decomposition.sqrt_inverse_metric * scaled_direction
    linearized_residual = state.r + state.jacobian_r @ direction
    scaled_dual_residual = (
        weighted_physics_jacobian.T @ linearized_residual
        + damping * scaled_direction
    )
    rank = decomposition.effective_rank
    multipliers = torch.zeros(
        condition_count,
        dtype=state.c.dtype,
        device=state.c.device,
    )
    if rank > 0:
        right_transpose = decomposition.right_singular_vectors_transpose[:rank, :]
        multiplier_coordinates = -(
            right_transpose @ scaled_dual_residual
        ) / decomposition.singular_values[:rank]
        multipliers = (
            decomposition.left_singular_vectors[:, :rank]
            @ multiplier_coordinates
        )
    predicted_reduction_tensor = (
        state.loss_r
        - 0.5 * torch.dot(linearized_residual, linearized_residual)
    )
    tangent_defect = relative_tangent_defect(direction, decomposition)
    predicted_reduction = float(predicted_reduction_tensor.item())
    finite = bool(
        torch.isfinite(direction).all()
        and torch.isfinite(multipliers).all()
        and torch.isfinite(linearized_residual).all()
        and torch.isfinite(predicted_reduction_tensor)
    )

    return FeasibleLMStep(
        direction=direction,
        multipliers=multipliers,
        linearized_residual=linearized_residual,
        predicted_reduction=predicted_reduction,
        linear_solve_residual=relative_solve_residual,
        tangent_defect=tangent_defect,
        success=finite and relative_solve_residual <= linear_solve_tolerance,
        used_least_squares=False,
        tangent_iterations=tangent_iterations,
    )


def evaluate_feasible_lm_trial(
    parameters: ParameterDict,
    condition_residual: ResidualFunction,
    physics_residual: ResidualFunction,
    step: FeasibleLMStep,
    inverse_metric_diagonal: torch.Tensor,
    old_physics_loss: float,
    config: FTPConfig,
    previous_rank: int | None = None,
) -> LMTrialResult:
    """恢复 LM 试探点的可行性，并用实际/预测下降比判定接受。"""
    step_h_norm_squared = float(
        torch.sum(step.direction.square() / inverse_metric_diagonal).item()
    )
    restoration_tolerance = max(
        config.restoration_tolerance_floor,
        min(
            config.feasibility_tolerance,
            config.restoration_forcing * step_h_norm_squared,
        ),
    )
    trial_parameters = add_parameter_vector(parameters, step.direction)
    restored = restore_feasibility(
        trial_parameters,
        condition_residual,
        inverse_metric_diagonal,
        restoration_tolerance,
        config.restoration_steps,
        config.restoration_relative_damping,
        config.feasibility_stationarity_tolerance,
        config.rank_hysteresis_factor,
        config.armijo_constant,
        config.max_feasibility_backtracks,
        previous_rank,
    )

    candidate_loss = old_physics_loss
    actual_reduction = float("-inf")
    reduction_ratio = float("-inf")
    finite_candidate = False
    if restored.success:
        candidate_r = physics_residual(restored.parameters)
        finite_candidate = bool(torch.isfinite(candidate_r).all())
        if finite_candidate:
            candidate_loss = float(
                (0.5 * torch.dot(candidate_r, candidate_r)).item()
            )
            actual_reduction = old_physics_loss - candidate_loss
            if step.predicted_reduction > 0.0:
                reduction_ratio = actual_reduction / step.predicted_reduction

    accepted = bool(
        restored.success
        and finite_candidate
        and step.predicted_reduction > 0.0
        and math.isfinite(reduction_ratio)
        and reduction_ratio >= config.trust_accept_ratio
    )
    return LMTrialResult(
        parameters=restored.parameters if accepted else parameters,
        accepted=accepted,
        physics_loss=candidate_loss if accepted else old_physics_loss,
        condition_norm=(
            restored.residual_norm
            if accepted
            else float(torch.linalg.vector_norm(condition_residual(parameters)).item())
        ),
        restoration_steps=restored.iterations,
        restoration_tolerance=restoration_tolerance,
        actual_reduction=actual_reduction,
        predicted_reduction=step.predicted_reduction,
        reduction_ratio=reduction_ratio,
        projected_condition_norm=restored.projected_residual_norm,
        constraint_stationarity_norm=restored.stationarity_norm,
        relative_constraint_stationarity=restored.relative_stationarity,
        attainable_condition_norm=restored.attainable_condition_norm,
        effective_rank=restored.effective_rank,
        rank_tolerance=restored.rank_tolerance,
    )


def train_ftp(
    parameters: ParameterDict,
    condition_residual: ResidualFunction,
    physics_residual: ResidualFunction,
    config: FTPConfig,
) -> TrainingResult:
    validate_float64_parameters(parameters)
    parameters = {name: value.detach() for name, value in parameters.items()}
    parameter_vector = flatten_parameter_vector(parameters)
    identity_inverse_metric = torch.ones_like(parameter_vector)
    initial_feasibility = restore_feasibility(
        parameters,
        condition_residual,
        identity_inverse_metric,
        config.feasibility_tolerance,
        config.initial_feasibility_steps,
        config.restoration_relative_damping,
        config.feasibility_stationarity_tolerance,
        config.rank_hysteresis_factor,
        config.armijo_constant,
        config.max_feasibility_backtracks,
    )
    if not initial_feasibility.success:
        raise RuntimeError(
            "初始可行性子问题既未达到绝对容差，也未达到数值平稳："
            f"||c||={initial_feasibility.residual_norm:.3e}, "
            f"||U_k^T c||={initial_feasibility.projected_residual_norm:.3e}, "
            f"relative_stationarity="
            f"{initial_feasibility.relative_stationarity:.3e}, "
            f"reason={initial_feasibility.termination_reason}"
        )

    current_parameters = initial_feasibility.parameters
    initial_state = evaluate_ftp_state(
        current_parameters, condition_residual, physics_residual
    )
    metric = AdaptiveDiagonalMetric(
        config.beta2,
        config.metric_epsilon,
        config.min_inverse_metric,
        config.max_inverse_metric,
    )
    history: list[TrainingRecord] = []
    lm_damping = min(
        config.lm_damping_max,
        max(config.lm_damping_min, config.lm_damping_initial),
    )
    consecutive_rejections = 0
    termination_reason = "maximum_iterations"
    previous_rank = initial_feasibility.effective_rank

    for iteration in range(config.max_iterations):
        state = evaluate_residual_system(
            current_parameters, condition_residual, physics_residual
        )
        old_physics_loss = float(state.loss_r.item())
        metric_update = metric.update(state.gradient_r)
        decomposition = decompose_condition_jacobian(
            state.jacobian_c,
            metric_update.inverse_diagonal,
            previous_rank,
            config.rank_hysteresis_factor,
        )
        previous_rank = decomposition.effective_rank
        diagnostics = diagnose_condition_jacobian(
            state.jacobian_c,
            metric_update.inverse_diagonal,
            config.condition_warning_threshold,
            decomposition,
        )
        stationarity = evaluate_projected_stationarity(
            state.gradient_r,
            decomposition,
            config,
        )
        condition_norm = float(torch.linalg.vector_norm(state.c).item())
        (
            projected_condition_norm,
            constraint_stationarity_norm,
            relative_constraint_stationarity,
            attainable_condition_norm,
        ) = condition_optimality_measures(state.c, decomposition)
        pde_mse = 2.0 * old_physics_loss

        # 数值可行域只包含 Range(B_k)；不可消分量单独诊断，不阻塞 KKT 判据。
        if (
            projected_condition_norm <= config.condition_tolerance
            and relative_constraint_stationarity
            <= config.feasibility_stationarity_tolerance
            and stationarity.norm <= config.stationarity_tolerance
        ):
            termination_reason = "kkt_stationarity"
            break
        if (
                config.enable_normalized_physics_stopping
                and projected_condition_norm <= config.condition_tolerance
                and relative_constraint_stationarity
                <= config.feasibility_stationarity_tolerance
                and pde_mse <= config.physics_loss_tolerance
        ):
            termination_reason = "normalized_physics_residual"
            break
        if stationarity.norm <= config.step_tolerance:
            termination_reason = (
                "small_step_stationary"
                if projected_condition_norm <= config.condition_tolerance
                else "small_step_infeasible"
            )
            break
        if not stationarity.success:
            termination_reason = "tangent_projection_failed"
            break

        linear_solve_tolerance = forcing_tolerance(
            stationarity.norm,
            config.linear_solve_tolerance_max,
            config.linear_solve_tolerance_min,
            config.linear_solve_forcing,
        )
        accepted_trial: LMTrialResult | None = None
        last_trial: LMTrialResult | None = None
        last_step: FeasibleLMStep | None = None
        damping_used = lm_damping
        attempts = 0

        # 每次拒绝都增大阻尼并重新求解切空间子问题，而不是缩放旧方向。
        for attempt in range(config.max_lm_attempts):
            attempts = attempt + 1
            damping_used = lm_damping
            lm_step = solve_feasible_lm_step(
                state,
                decomposition,
                damping_used,
                linear_solve_tolerance,
                config.tangent_cg_max_iterations,
            )
            last_step = lm_step
            step_h_norm = float(
                torch.sqrt(
                    torch.sum(
                        lm_step.direction.square()
                        / metric_update.inverse_diagonal
                    )
                ).item()
            )
            if (
                not lm_step.success
                or lm_step.predicted_reduction <= 0.0
                or lm_step.tangent_defect > stationarity.tangent_tolerance
            ):
                lm_damping = min(
                    config.lm_damping_max,
                    lm_damping * config.damping_increase,
                )
                continue
            if step_h_norm <= config.step_tolerance:
                termination_reason = (
                    "small_step_stationary"
                    if projected_condition_norm <= config.condition_tolerance
                    else "small_step_infeasible"
                )
                break

            trial = evaluate_feasible_lm_trial(
                current_parameters,
                condition_residual,
                physics_residual,
                lm_step,
                metric_update.inverse_diagonal,
                old_physics_loss,
                config,
                decomposition.effective_rank,
            )
            last_trial = trial
            if trial.accepted:
                accepted_trial = trial
                current_parameters = trial.parameters
                consecutive_rejections = 0
                if trial.reduction_ratio >= config.trust_good_ratio:
                    lm_damping = max(
                        config.lm_damping_min,
                        lm_damping * config.damping_decrease,
                    )
                elif trial.reduction_ratio < config.trust_bad_ratio:
                    lm_damping = min(
                        config.lm_damping_max,
                        lm_damping * config.damping_increase,
                    )
                break

            lm_damping = min(
                config.lm_damping_max,
                lm_damping * config.damping_increase,
            )

        if termination_reason.startswith("small_step_"):
            break

        accepted = accepted_trial is not None
        if not accepted:
            consecutive_rejections += 1

        record_trial = accepted_trial or last_trial
        record_condition_norm = (
            record_trial.condition_norm if record_trial is not None else condition_norm
        )
        record_physics_loss = (
            record_trial.physics_loss
            if record_trial is not None
            else old_physics_loss
        )
        record_restoration_steps = (
            record_trial.restoration_steps if record_trial is not None else 0
        )
        record_restoration_tolerance = (
            record_trial.restoration_tolerance
            if record_trial is not None
            else config.feasibility_tolerance
        )
        record_predicted_reduction = (
            last_step.predicted_reduction if last_step is not None else 0.0
        )
        record_reduction_ratio = (
            record_trial.reduction_ratio if record_trial is not None else float("nan")
        )
        record_linear_residual = (
            last_step.linear_solve_residual if last_step is not None else float("inf")
        )
        record_tangent_defect = (
            last_step.tangent_defect
            if last_step is not None
            else stationarity.tangent_defect
        )
        record_projected_condition_norm = (
            accepted_trial.projected_condition_norm
            if accepted_trial is not None
            else projected_condition_norm
        )
        record_constraint_stationarity_norm = (
            accepted_trial.constraint_stationarity_norm
            if accepted_trial is not None
            else constraint_stationarity_norm
        )
        record_relative_constraint_stationarity = (
            accepted_trial.relative_constraint_stationarity
            if accepted_trial is not None
            else relative_constraint_stationarity
        )
        record_attainable_condition_norm = (
            accepted_trial.attainable_condition_norm
            if accepted_trial is not None
            else attainable_condition_norm
        )
        record_effective_rank = diagnostics.effective_rank
        record_rank_tolerance = diagnostics.rank_tolerance

        history.append(
            TrainingRecord(
                iteration=iteration,
                condition_norm=record_condition_norm,
                condition_loss=0.5 * record_condition_norm**2,
                physics_loss=record_physics_loss,
                total_loss=(
                    0.5 * record_condition_norm**2
                    + record_physics_loss
                ),
                step_size=1.0 if accepted else 0.0,
                accepted=accepted,
                tangent_defect=record_tangent_defect,
                lambda_min=diagnostics.lambda_min,
                condition_number=diagnostics.condition_number,
                projection_damping=stationarity.projection.damping,
                restoration_steps=record_restoration_steps,
                backtracks=max(0, attempts - 1),
                clipped_fraction=metric_update.clipped_fraction,
                lm_damping=damping_used,
                predicted_reduction=record_predicted_reduction,
                reduction_ratio=record_reduction_ratio,
                linear_solve_residual=record_linear_residual,
                linear_solve_tolerance=linear_solve_tolerance,
                stationarity_norm=stationarity.norm,
                tangent_tolerance=stationarity.tangent_tolerance,
                restoration_tolerance=record_restoration_tolerance,
                effective_rank=record_effective_rank,
                rank_tolerance=record_rank_tolerance,
                rank_is_stable=diagnostics.rank_is_stable,
                spectral_gap_ratio=diagnostics.spectral_gap_ratio,
                projected_condition_norm=record_projected_condition_norm,
                constraint_stationarity_norm=record_constraint_stationarity_norm,
                relative_constraint_stationarity=(
                    record_relative_constraint_stationarity
                ),
                attainable_condition_norm=record_attainable_condition_norm,
            )
        )

        if consecutive_rejections >= config.max_consecutive_rejections:
            termination_reason = "trust_region_rejected"
            break

    final_state = evaluate_ftp_state(
        current_parameters, condition_residual, physics_residual
    )
    return TrainingResult(
        parameters=current_parameters,
        history=tuple(history),
        initial_feasibility=initial_feasibility,
        initial_state=initial_state,
        final_state=final_state,
        termination_reason=termination_reason,
    )

def FTP(model, condition_fn, physics_fn, max_iteration):
    model.to(dtype=DEFAULT_DTYPE)
    parameters = dict(model.named_parameters())
    config = FTPConfig(
        max_iterations=max_iteration,
        feasibility_tolerance=1.0e-8,
        initial_step_size=1.0,
        max_step_size=1.0,
        lm_damping_initial=1.0e-5,
        projection_relative_damping=1.0e-9,
        restoration_relative_damping=1.0e-9,
    )
    torch.cuda.reset_peak_memory_stats()
    start_time = time()
    training = train_ftp(parameters, condition_fn, physics_fn, config)
    copy_parameters_to_model(model, training.parameters)
    elapsed_time = time() - start_time
    peak = torch.cuda.max_memory_allocated()

    initial_loss_c = float(training.initial_state.loss_c.item())
    initial_loss_r = float(training.initial_state.loss_r.item())
    loss_history = [
        [
            initial_loss_c,
            initial_loss_r,
            initial_loss_c + initial_loss_r,
        ]
    ]
    loss_history.extend(
        [
            record.condition_loss,
            record.physics_loss,
            record.total_loss,
        ]
        for record in training.history
    )

    return model, loss_history, elapsed_time, peak
