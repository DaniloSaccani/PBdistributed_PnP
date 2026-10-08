"""Distributed certified REN, bounded MAD readout and PnP recertification."""
from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from math import isfinite, log
from torch import nn
from torch.nn import functional
from typing import Any
from .plant import topology_library


# Certificate


@dataclass(frozen=True)
class Interconnection:
    M_ve: torch.Tensor
    M_vz: torch.Tensor
    M_uz: torch.Tensor
    node_ids: tuple[int, ...]
    n_sizes: tuple[int, ...]
    q_sizes: tuple[int, ...]
    r_sizes: tuple[int, ...]
    input_slices: dict[int, slice]
    residual_indices: dict[int, tuple[int, ...]]
    communication_rows: dict[int, int]
    edge_weight: float
    topology_edges: tuple[tuple[int, int], ...]
    registered_edge_weights: torch.Tensor

    @property
    def active_nodes(self) -> tuple[int, ...]:
        return self.node_ids

    @property
    def active_residual_indices(self) -> tuple[int, ...]:
        return tuple(k for i in self.node_ids for k in self.residual_indices[i])


@dataclass(frozen=True)
class Bounds:
    hbar: torch.Tensor
    rho_z: torch.Tensor
    rho_v0: torch.Tensor
    rho_v1: torch.Tensor
    cbar: torch.Tensor


@dataclass(frozen=True)
class Gains:
    alpha: torch.Tensor
    c: torch.Tensor
    gamma: torch.Tensor


def _positive(value: float, name: str) -> None:
    if not (float(value) > 0 and torch.isfinite(torch.tensor(value))):
        raise ValueError(f"{name} must be finite and strictly positive")


def _validated_edge_weights(registry: Any, edge_weights: torch.Tensor,
                            *, device: Any = None) -> torch.Tensor:
    if not isinstance(edge_weights, torch.Tensor):
        raise TypeError("Edge weights must be a float64 tensor in permanent registry order")
    if edge_weights.dtype != torch.float64:
        raise TypeError("Edge weights must have dtype torch.float64")
    if edge_weights.shape != (len(registry.edges),):
        raise ValueError("Edge weights must have one entry per permanent registered edge")
    if not bool(torch.isfinite(edge_weights).all() & (edge_weights > 0).all()):
        raise ValueError("Every registered edge weight must be finite and strictly positive")
    return edge_weights if device is None else edge_weights.to(device=device)


def build_interconnection(registry: Any, topology: Any, a: float = 0.25,
                         *, edge_weights: torch.Tensor | None = None,
                         device: Any = None) -> Interconnection:
    """Inject each allocated owner coordinate once; mix current neighbor z."""
    _positive(a, "edge weight")
    registry.validate_topology(topology)
    nodes = tuple(sorted(topology.active_nodes))
    if not nodes or not set(nodes).issubset(registry.node_ids):
        raise ValueError("An interconnection needs registered active DGUs")
    if edge_weights is None:
        edge_weights = torch.full((len(registry.edges),), float(a),
                                  dtype=torch.float64, device=device)
    # A realization records the weights used for its matrices. The clone keeps
    # their gradient path without aliasing a mutable caller-owned tensor.
    weights = _validated_edge_weights(registry, edge_weights, device=device).clone()
    device = weights.device
    edge_positions = {edge: k for k, edge in enumerate(registry.edges)}
    residual_indices = {i: tuple(registry.owned_indices(i)) for i in nodes}
    n_sizes = tuple(len(residual_indices[i]) for i in nodes)
    q_sizes = tuple(n + 1 for n in n_sizes)
    n, q, count = sum(n_sizes), sum(q_sizes), len(nodes)
    mve = torch.zeros(q, n, dtype=torch.float64, device=device)
    mvz = torch.zeros(q, count, dtype=torch.float64, device=device)
    muz = torch.eye(count, dtype=torch.float64, device=device)
    input_slices, communication_rows = {}, {}
    output_index = {node: position for position, node in enumerate(nodes)}
    row = column = 0
    for node, ni, qi in zip(nodes, n_sizes, q_sizes):
        input_slices[node] = slice(row, row + qi)
        communication_rows[node] = row + ni
        mve[row:row + ni, column:column + ni] = torch.eye(
            ni, dtype=torch.float64, device=device)
        for neighbor in registry.neighbors(topology, node):
            if neighbor not in output_index or neighbor == node:
                raise ValueError("Communication must use active, distinct neighbors")
            edge = (min(node, neighbor), max(node, neighbor))
            mvz[row + ni, output_index[neighbor]] = weights[edge_positions[edge]]
        row += qi
        column += ni
    return Interconnection(mve, mvz, muz, nodes, n_sizes, q_sizes,
                           (1,) * count, input_slices, residual_indices,
                           communication_rows, float(a), tuple(sorted(topology.edges)),
                           weights)


def general_bounds(interconnection: Interconnection, gamma_R: float,
                   *, cbar0: float = 1.0) -> Bounds:
    """Compute bounds from matrix entries, independently of graph degrees."""
    _positive(gamma_R, "network gain")
    _positive(cbar0, "empty-bound fallback")
    me, mz, mu = (interconnection.M_ve, interconnection.M_vz,
                  interconnection.M_uz)
    if me.ndim != 2 or mz.shape[0] != me.shape[0] or mu.shape[1] != mz.shape[1]:
        raise ValueError("Incompatible interconnection matrix dimensions")
    if me.shape[0] != sum(interconnection.q_sizes) or mz.shape[1] != sum(interconnection.r_sizes):
        raise ValueError("Local block sizes do not match the matrices")
    if not all(bool(torch.isfinite(matrix).all()) for matrix in (me, mz, mu)):
        raise ValueError("Interconnection matrices must be finite")
    nonzero = me != 0
    if (not bool(((me == 0) | (me == 1)).all())
            or not bool((nonzero.sum(dim=0) == 1).all())
            or not bool((nonzero.sum(dim=1) <= 1).all())):
        raise ValueError("M_ve must be a unit coordinate injection")
    gram = mu.T @ mu
    h = torch.diagonal(gram)
    if not torch.allclose(gram, torch.diag(h), atol=1e-12, rtol=0):
        raise ValueError("M_uz columns must be orthogonal")
    direct = nonzero.any(dim=1)
    row_sums, column_sums = mz.abs().sum(dim=1), mz.abs().sum(dim=0)
    hbars, rz, rv0, rv1, cbars = [], [], [], [], []
    row = output = 0
    zero = me.new_zeros(())
    target_squared = me.new_tensor(float(gamma_R) ** 2)
    for qi, ri in zip(interconnection.q_sizes, interconnection.r_sizes):
        own_direct = direct[row:row + qi]
        rows = row_sums[row:row + qi]
        bound0 = rows[~own_direct].max() if bool((~own_direct).any()) else zero
        bound1 = rows[own_direct].max() if bool(own_direct.any()) else zero
        choices = []
        if bool(own_direct.any()):
            choices.append(target_squared / (1 + bound1 * target_squared))
        if bool(bound0 > 0):
            choices.append(1 / bound0)
        cbars.append(torch.stack(choices).min() if choices else me.new_tensor(cbar0))
        hbars.append(h[output:output + ri].max())
        rz.append(column_sums[output:output + ri].max())
        rv0.append(bound0)
        rv1.append(bound1)
        row += qi
        output += ri
    return Bounds(*(torch.stack(values) for values in (hbars, rz, rv0, rv1, cbars)))


def specialized_bounds(registry: Any, topology: Any, a: float,
                       gamma_R: float) -> Bounds:
    """Closed-form specialization for separate residual/communication rows."""
    _positive(a, "edge weight")
    _positive(gamma_R, "network gain")
    degree = torch.tensor([len(registry.neighbors(topology, i))
                           for i in sorted(topology.active_nodes)], dtype=torch.float64)
    coupling = float(a) * degree
    reciprocal = torch.where(degree > 0, 1 / (float(a) * degree.clamp_min(1)),
                             torch.full_like(degree, float("inf")))
    cbar = torch.minimum(torch.full_like(degree, float(gamma_R) ** 2), reciprocal)
    return Bounds(torch.ones_like(degree), coupling, coupling.clone(),
                  torch.zeros_like(degree), cbar)


def weighted_specialized_bounds(registry: Any, topology: Any,
                                edge_weights: torch.Tensor,
                                gamma_R: float) -> Bounds:
    """Positive symmetric edge specialization with actual incident weight sums."""
    _positive(gamma_R, "network gain")
    registry.validate_topology(topology)
    weights = _validated_edge_weights(registry, edge_weights)
    active_edges = set(topology.edges)
    coupling = torch.stack([
        weights[[k for k, edge in enumerate(registry.edges)
                 if edge in active_edges and node in edge]].sum()
        for node in sorted(topology.active_nodes)
    ])
    if not bool(torch.isfinite(coupling).all()):
        raise FloatingPointError("Incident edge-weight sum is not finite")
    # Avoid an evaluated reciprocal of zero, even in torch.where's unused
    # branch, so isolated nodes also have a finite zero weight derivative.
    positive = coupling > 0
    safe = torch.where(positive, coupling, torch.ones_like(coupling))
    reciprocal = torch.where(positive, 1 / safe,
                             torch.full_like(coupling, float("inf")))
    cbar = torch.minimum(torch.full_like(coupling, float(gamma_R) ** 2), reciprocal)
    return Bounds(torch.ones_like(coupling), coupling, coupling.clone(),
                  torch.zeros_like(coupling), cbar)


def gain_parameters(interconnection: Interconnection, eta_alpha: torch.Tensor,
                    eta_c: torch.Tensor, gamma_R: float = 0.005,
                    epsilon: float = 0.02) -> Gains:
    _positive(epsilon, "certificate epsilon")
    bounds = general_bounds(interconnection, gamma_R)
    eta_alpha = torch.as_tensor(eta_alpha, dtype=torch.float64,
                                device=interconnection.M_ve.device)
    eta_c = torch.as_tensor(eta_c, dtype=torch.float64,
                            device=interconnection.M_ve.device)
    expected = (len(interconnection.node_ids),)
    if eta_alpha.shape != expected or eta_c.shape != expected:
        raise ValueError("Gain free variables must have one entry per active DGU")
    if not bool(torch.isfinite(eta_alpha).all() & torch.isfinite(eta_c).all()):
        raise FloatingPointError("Nonfinite gain free variables")
    fraction = torch.sigmoid(eta_c)
    if bool(((fraction <= 0) | (fraction >= 1)).any()):
        raise FloatingPointError("Sigmoid reached a floating-point endpoint; no projection applied")
    alpha = bounds.hbar + bounds.rho_z + eta_alpha.square() + float(epsilon)
    c = bounds.cbar * fraction
    gamma = torch.sqrt(c / alpha)
    if not bool(torch.isfinite(gamma).all() & (gamma > 0).all()):
        raise FloatingPointError("Local gain cannot be represented as a finite positive value")
    return Gains(alpha, c, gamma)


def network_certificate(interconnection: Interconnection, gains: Gains,
                        gamma_R: float) -> torch.Tensor:
    """Return the full symmetric G.T @ Q @ G, with no eigenvalue clipping."""
    _positive(gamma_R, "network gain")
    me, mz, mu = interconnection.M_ve, interconnection.M_vz, interconnection.M_uz
    expected = (len(interconnection.q_sizes),)
    if any(value.shape != expected for value in (gains.alpha, gains.c, gains.gamma)):
        raise ValueError("Gain vector and local block counts disagree")
    if not all(bool(torch.isfinite(value).all() & (value > 0).all())
               for value in (gains.alpha, gains.c, gains.gamma)):
        raise ValueError("Certificate gains must be finite and positive")
    q_repeats = torch.tensor(interconnection.q_sizes, device=me.device)
    r_repeats = torch.tensor(interconnection.r_sizes, device=me.device)
    # Use the actual prescribed local gains, not merely the auxiliary c values.
    piv = torch.diag(torch.repeat_interleave(gains.alpha * gains.gamma.square(), q_repeats))
    piz = torch.diag(torch.repeat_interleave(gains.alpha, r_repeats))
    zz = mz.T @ piv @ mz - piz + mu.T @ mu
    ze = mz.T @ piv @ me
    ee = me.T @ piv @ me - float(gamma_R) ** 2 * torch.eye(
        me.shape[1], dtype=me.dtype, device=me.device)
    matrix = torch.cat((torch.cat((zz, ze), dim=1),
                        torch.cat((ze.T, ee), dim=1)), dim=0)
    return (matrix + matrix.T) / 2


# Ren


@dataclass(frozen=True)
class RENState:
    x: torch.Tensor
    b: torch.Tensor


@dataclass(frozen=True)
class RENRealization:
    matrices: dict[str, torch.Tensor]
    P: torch.Tensor
    gamma: torch.Tensor


class CertifiedREN(nn.Module):
    def __init__(self, input_dim: int, state_dim: int = 3, width: int = 3,
                 *, construction_epsilon: float = 0.01):
        super().__init__()
        if (min(input_dim, state_dim, width) < 1
                or not isfinite(construction_epsilon) or construction_epsilon <= 0):
            raise ValueError("REN dimensions and construction epsilon must be positive")
        self.input_dim, self.state_dim, self.width = input_dim, state_dim, width
        self.construction_epsilon = float(construction_epsilon)
        total = 2 * state_dim + width

        def free(*shape: int) -> nn.Parameter:
            return nn.Parameter(0.1 * torch.randn(shape, dtype=torch.float64))

        self.metric_factor = free(total, total)
        self.state_skew = free(state_dim, state_dim)
        self.input_to_state = free(state_dim, input_dim)
        self.input_to_nonlinearity = free(width, input_dim)
        self.state_to_output = free(1, state_dim)
        self.nonlinearity_to_output = free(1, width)
        self.feedthrough_vector = free(input_dim)

    def realize(self, gamma: Any) -> RENRealization:
        """Rebuild differentiable matrices from the current free parameters."""
        q, n, width = self.input_dim, self.state_dim, self.width
        gamma = torch.as_tensor(gamma, dtype=torch.float64,
                                device=self.metric_factor.device)
        if gamma.numel() != 1 or not bool(torch.isfinite(gamma).all() & (gamma > 0).all()):
            raise ValueError("A REN realization needs one finite positive local gain")
        gamma = gamma.reshape(())
        if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters()):
            raise FloatingPointError("REN free parameters must be finite")
        identity_input = torch.eye(q, dtype=torch.float64, device=gamma.device)
        vector = self.feedthrough_vector
        direct = (vector / torch.sqrt(1 + vector.square().sum())).unsqueeze(0)
        input_slack = identity_input - direct.T @ direct
        # Cholesky failure is a numerical-contract failure, never repaired by projection.
        torch.linalg.cholesky(input_slack)
        c2, d21 = self.state_to_output, self.nonlinearity_to_output
        b2, d12 = self.input_to_state, self.input_to_nonlinearity
        supply_cross = torch.cat((-c2.T @ direct,
                                  -d21.T @ direct - d12,
                                  b2), dim=0)
        output_stack = torch.cat((c2.T, d21.T,
                                  c2.new_zeros(n, 1)), dim=0)
        total = 2 * n + width
        metric = (self.metric_factor.T @ self.metric_factor
                  + self.construction_epsilon * torch.eye(
                      total, dtype=torch.float64, device=gamma.device)
                  + supply_cross @ torch.linalg.solve(input_slack, supply_cross.T)
                  + output_stack @ output_stack.T)
        metric = (metric + metric.T) / 2
        linear = slice(0, n)
        nonlinear = slice(n, n + width)
        following = slice(n + width, total)
        p_cal = metric[following, following]
        torch.linalg.cholesky(p_cal)
        e = (metric[linear, linear] + p_cal
             + self.state_skew - self.state_skew.T) / 2
        storage_matrix = e.T @ torch.linalg.solve(p_cal, e)
        storage_matrix = (storage_matrix + storage_matrix.T) / 2
        if not bool(torch.isfinite(storage_matrix).all()):
            raise FloatingPointError("Nonfinite REN storage matrix")
        torch.linalg.cholesky(storage_matrix)
        nonlinear_metric = metric[nonlinear, nonlinear]
        matrices = {
            "E": e,
            "F": metric[following, linear],
            "B1": metric[following, nonlinear],
            "B2": gamma * b2,
            "C1": -metric[nonlinear, linear],
            "D11": -torch.tril(nonlinear_metric, diagonal=-1),
            "D12": gamma * d12,
            "C2": c2,
            "D21": d21,
            "D22": gamma * direct,
            "Lambda": torch.diagonal(nonlinear_metric) / 2,
        }
        if not all(bool(torch.isfinite(value).all()) for value in matrices.values()):
            raise FloatingPointError("Nonfinite derived REN matrices")
        return RENRealization(matrices, storage_matrix, gamma)

    def zero_state(self, batch_shape: tuple[int, ...] = ()) -> RENState:
        prototype = self.metric_factor
        return RENState(prototype.new_zeros(*batch_shape, self.state_dim),
                        prototype.new_zeros(batch_shape))

    def step(self, v: torch.Tensor, state: RENState,
             realization: RENRealization) -> tuple[torch.Tensor, RENState]:
        if v.shape[-1] != self.input_dim or state.x.shape[-1] != self.state_dim:
            raise ValueError("REN input or state has the wrong final dimension")
        matrices = realization.matrices
        activation_values = []
        for j in range(self.width):
            value = (functional.linear(state.x, matrices["C1"][j])
                     + functional.linear(v, matrices["D12"][j]))
            if j:
                value = value + functional.linear(torch.stack(activation_values, dim=-1),
                                                   matrices["D11"][j, :j])
            activation_values.append(torch.tanh(value / matrices["Lambda"][j]))
        activation = torch.stack(activation_values, dim=-1)
        rhs = (functional.linear(state.x, matrices["F"])
               + functional.linear(activation, matrices["B1"])
               + functional.linear(v, matrices["B2"]))
        next_x = torch.linalg.solve(matrices["E"], rhs.unsqueeze(-1)).squeeze(-1)
        next_buffer = (functional.linear(state.x, matrices["C2"])
                       + functional.linear(activation, matrices["D21"])
                       + functional.linear(v, matrices["D22"])).squeeze(-1)
        return state.b, RENState(next_x, next_buffer)

    @staticmethod
    def storage(state: RENState, realization: RENRealization) -> torch.Tensor:
        return torch.einsum("...i,ij,...j->...", state.x, realization.P, state.x) + state.b.square()


# Mad


class DirectionNetwork(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("Direction-network dimensions must be positive")
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim, 1, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(features)).squeeze(-1)


def physical_features(voltage_errors: torch.Tensor, registry: Any,
                      topology: Any) -> torch.Tensor:
    """Own voltage error and sum of active neighbors' errors; no event labels."""
    if voltage_errors.shape[-1] != len(registry.node_ids):
        raise ValueError("Voltage-error vector must use the permanent DGU axis")
    active = set(topology.active_nodes)
    positions = {node: k for k, node in enumerate(registry.node_ids)}
    zero = torch.zeros_like(voltage_errors[..., 0])
    features = []
    for node in registry.node_ids:
        if node in active:
            own = voltage_errors[..., positions[node]]
            neighbors = registry.neighbors(topology, node)
            neighbor_sum = sum((voltage_errors[..., positions[j]] for j in neighbors), zero)
            features.append(torch.stack((own, neighbor_sum), dim=-1))
        else:
            features.append(torch.stack((zero, zero), dim=-1))
    return torch.stack(features, dim=-2)


def readout(z: torch.Tensor, direction: torch.Tensor,
            node_mask: torch.Tensor) -> torch.Tensor:
    """Apply exact scalar magnitudes; direction is the final tanh output."""
    if z.shape != direction.shape:
        raise ValueError("Magnitude and direction vectors must have matching shapes")
    if not bool(torch.isfinite(direction).all() & (direction.abs() <= 1).all()):
        raise ValueError("MAD directions must be finite and bounded by one")
    if not bool(torch.isfinite(node_mask).all() & ((node_mask == 0) | (node_mask == 1)).all()):
        raise ValueError("The actuator mask must contain only zero or one")
    if node_mask.shape != (z.shape[-1],):
        raise ValueError("The actuator mask must have one entry per permanent DGU")
    return z.abs() * direction * node_mask


# Controller


@dataclass(frozen=True)
class NetworkRealization:
    interconnection: Interconnection
    bounds: Bounds
    gains: Gains
    locals: dict[int, RENRealization]
    certificate: torch.Tensor


@dataclass(frozen=True)
class ControllerStep:
    z: torch.Tensor
    u: torch.Tensor
    direction: torch.Tensor
    next_states: dict[int, RENState]
    communication: dict[int, torch.Tensor]
    features: torch.Tensor
    local_inputs: dict[int, torch.Tensor]


class DistributedController(nn.Module):
    def __init__(self, registry: Any, gamma_R: float = 0.005, a: float = 0.25,
                 epsilon: float = 0.02, state_dim: int = 3, width: int = 3,
                 hidden_dim: int = 8, learn_edge_weights: bool = False,
                 edge_weight_bounds=None):
        super().__init__()
        self.registry = registry
        self.gamma_R, self.a, self.epsilon = float(gamma_R), float(a), float(epsilon)
        if not isfinite(self.a) or self.a <= 0:
            raise ValueError("initial edge weight must be finite and positive")
        self.learn_edge_weights = bool(learn_edge_weights)
        self.edge_weight_bounds = None
        if edge_weight_bounds is not None:
            if len(edge_weight_bounds) != 2:
                raise ValueError("Edge bounds require a lower and upper value")
            lower, upper = map(float, edge_weight_bounds)
            if (not self.learn_edge_weights or not all(map(isfinite, (lower, upper)))
                    or not 0 < lower < self.a < upper):
                raise ValueError("Learned edge initialization must lie strictly within positive bounds")
            self.edge_weight_bounds = (lower, upper)
        if self.learn_edge_weights:
            if self.edge_weight_bounds is None:
                self.log_edge_weights = nn.Parameter(torch.full((len(registry.edges),), log(self.a),
                                                               dtype=torch.float64))
            else:
                lower, upper = self.edge_weight_bounds
                value = log((self.a-lower)/(upper-self.a))
                self.edge_logits = nn.Parameter(torch.full((len(registry.edges),), value, dtype=torch.float64))
        self.rens = nn.ModuleDict({str(i): CertifiedREN(len(registry.owned_indices(i)) + 1,
                                                      state_dim, width)
                                  for i in registry.node_ids})
        self.directions = nn.ModuleDict({str(i): DirectionNetwork(2, hidden_dim)
                                        for i in registry.node_ids})
        self.eta_alpha = nn.Parameter(torch.full((len(registry.node_ids),), 0.1,
                                                 dtype=torch.float64))
        self.eta_c = nn.Parameter(torch.zeros(len(registry.node_ids), dtype=torch.float64))

    def effective_edge_weights(self) -> torch.Tensor:
        """Permanent undirected weights; optimized only BETWEEN rollouts."""
        if self.learn_edge_weights:
            if self.edge_weight_bounds is None:
                weights = torch.exp(self.log_edge_weights)
            else:
                lower, upper = self.edge_weight_bounds
                fraction = torch.sigmoid(self.edge_logits)
                if not bool(((fraction > 0) & (fraction < 1)).all()):
                    raise FloatingPointError("Bounded edge sigmoid reached a floating-point endpoint")
                weights = lower+(upper-lower)*fraction
                if not bool(((weights > lower) & (weights < upper)).all()):
                    raise FloatingPointError("Represented edge weight reached a decision-bound endpoint")
        else:
            weights = self.eta_alpha.new_full((len(self.registry.edges),), self.a)
        if not bool(torch.isfinite(weights).all() & (weights > 0).all()):
            raise FloatingPointError("edge weights are not representable as finite positive numbers")
        return weights

    def check_frozen_edge_weights(self, expected: torch.Tensor) -> None:
        with torch.no_grad():
            if not torch.equal(self.effective_edge_weights(), expected):
                raise RuntimeError("edge weights changed inside a rollout or a cached realization; "
                                   "rebuild only between independent rollouts")

    def interconnection(self, topology: Any) -> Interconnection:
        return build_interconnection(self.registry, topology, self.a,
                                     edge_weights=self.effective_edge_weights(),
                                     device=self.eta_alpha.device)

    def realize(self, topology: Any) -> NetworkRealization:
        interconnection = self.interconnection(topology)
        positions = {node: k for k, node in enumerate(self.registry.node_ids)}
        active_positions = [positions[node] for node in interconnection.node_ids]
        bounds = general_bounds(interconnection, self.gamma_R)
        gains = gain_parameters(interconnection, self.eta_alpha[active_positions],
                                self.eta_c[active_positions], self.gamma_R, self.epsilon)
        locals_ = {node: self.rens[str(node)].realize(gains.gamma[k])
                   for k, node in enumerate(interconnection.node_ids)}
        return NetworkRealization(interconnection, bounds, gains, locals_,
                                  network_certificate(interconnection, gains, self.gamma_R))

    def initial_state(self, batch_shape: tuple[int, ...] = ()) -> dict[int, RENState]:
        return {i: self.rens[str(i)].zero_state(batch_shape) for i in self.registry.node_ids}

    def step(self, ehat: torch.Tensor, voltage_errors: torch.Tensor, topology: Any,
             states: dict[int, RENState], realization: NetworkRealization,
             mode: str = "mad") -> ControllerStep:
        if mode not in ("mad", "direct", "baseline"):
            raise ValueError("Controller mode must be mad, direct, or baseline")
        if ehat.shape[-1] != self.registry.state_dim:
            raise ValueError("Residual must use the full permanent canonical state layout")
        interconnection = realization.interconnection
        self.check_frozen_edge_weights(interconnection.registered_edge_weights.detach())
        if tuple(sorted(topology.active_nodes)) != interconnection.node_ids:
            raise ValueError("Realization and topology have different active DGUs")
        if tuple(sorted(topology.edges)) != interconnection.topology_edges:
            raise ValueError("Realization and topology have different communication edges")
        active = set(interconnection.node_ids)
        positions = {node: k for k, node in enumerate(self.registry.node_ids)}
        zero = ehat.new_zeros(ehat.shape[:-1])
        z = torch.stack([states[node].b if node in active else zero
                         for node in self.registry.node_ids], dim=-1)
        z_active = z[..., [positions[i] for i in interconnection.node_ids]]
        e_active = ehat[..., list(interconnection.active_residual_indices)]
        v_active = (functional.linear(z_active, interconnection.M_vz)
                    + functional.linear(e_active, interconnection.M_ve))
        next_states = dict(states)
        local_inputs, communication = {}, {}
        for node in interconnection.node_ids:
            v = v_active[..., interconnection.input_slices[node]]
            local_inputs[node] = v
            communication[node] = v_active[..., interconnection.communication_rows[node]]
            _, next_states[node] = self.rens[str(node)].step(v, states[node],
                                                           realization.locals[node])
        features = physical_features(voltage_errors, self.registry, topology)
        direction = torch.stack([self.directions[str(node)](features[..., positions[node], :])
                                 if node in active else zero
                                 for node in self.registry.node_ids], dim=-1)
        node_mask = ehat.new_tensor([float(node in active) for node in self.registry.node_ids])
        if mode == "mad":
            u = readout(z, direction, node_mask)
        elif mode == "direct":
            u = z * node_mask
        else:
            u = torch.zeros_like(z)
        return ControllerStep(z, u, direction, next_states, communication, features, local_inputs)


# Current mad


FEATURE_SCHEMA = {
    "version": 1,
    "order": ["own_voltage_error", "sum_active_neighbor_voltage_errors",
              "measured_converter_current", "sum_outward_active_incident_line_currents"],
    "units": ["V", "V", "A", "A"],
    "line_sign": "permanent tail:+I; permanent head:-I",
    "source": "current observed X; no load/equilibrium/prediction/residual features",
    "normalization": "none; raw SI features", "inactive": "four exact zeros",
}


def current_features(physical_state, voltage_errors, registry, topology):
    """Four local features; incident measurements do not duplicate REN residuals."""
    if physical_state.shape[-1] != registry.state_dim:
        raise ValueError("Measured physical state must use the permanent canonical layout")
    if physical_state.shape[:-1] != voltage_errors.shape[:-1]:
        raise ValueError("Physical and voltage observations must have identical batch axes")
    voltage = physical_features(voltage_errors, registry, topology)
    active, edges = set(topology.active_nodes), set(topology.edges)
    zero = physical_state.new_zeros(physical_state.shape[:-1])
    rows = []
    for position, node in enumerate(registry.node_ids):
        if node in active:
            converter = physical_state[..., registry.node_indices(node)[1]]
            outward = sum(((1. if edge[0] == node else -1.)
                           * physical_state[..., registry.edge_index(edge)]
                           for edge in registry.incident_edges(node) if edge in edges), zero)
            row = torch.stack((voltage[..., position, 0], voltage[..., position, 1],
                               converter, outward), dim=-1)
        else:
            row = torch.stack((zero, zero, zero, zero), dim=-1)
        rows.append(row)
    result = torch.stack(rows, dim=-2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Active measured MAD features must be finite")
    return result


class CurrentAwareController(DistributedController):
    mad_feature_dim = 4
    feature_definition = "raw measured (own Verror, active-neighbor Verror sum, It, outward incident current sum)"
    mad_feature_schema = FEATURE_SCHEMA

    def __init__(self, registry, **kwargs):
        super().__init__(registry, **kwargs)
        hidden = kwargs.get("hidden_dim", 8)
        self.directions = nn.ModuleDict({str(i): DirectionNetwork(4, hidden)
                                        for i in registry.node_ids})

    def step(self, *args, **kwargs):
        raise ValueError("CurrentAwareController requires step_observed with physical_state=X")

    def step_observed(self, ehat, voltage_errors, topology, states, realization,
                      *, physical_state, mode="mad"):
        if mode not in ("mad", "direct", "baseline"):
            raise ValueError("Controller mode must be mad, direct, or baseline")
        if ehat.shape[-1] != self.registry.state_dim:
            raise ValueError("Residual must use the permanent canonical layout")
        interconnection = realization.interconnection
        self.check_frozen_edge_weights(interconnection.registered_edge_weights.detach())
        if (tuple(sorted(topology.active_nodes)) != interconnection.node_ids
                or tuple(sorted(topology.edges)) != interconnection.topology_edges):
            raise ValueError("Realization and actual physical communication graph differ")
        active = set(interconnection.node_ids)
        positions = {node: k for k, node in enumerate(self.registry.node_ids)}
        zero = ehat.new_zeros(ehat.shape[:-1])
        # Same certified operations/timing as DistributedController.step.
        z = torch.stack([states[node].b if node in active else zero
                         for node in self.registry.node_ids], dim=-1)
        z_active = z[..., [positions[i] for i in interconnection.node_ids]]
        e_active = ehat[..., list(interconnection.active_residual_indices)]
        v_active = (functional.linear(z_active, interconnection.M_vz)
                    + functional.linear(e_active, interconnection.M_ve))
        next_states, local_inputs, communication = dict(states), {}, {}
        for node in interconnection.node_ids:
            v = v_active[..., interconnection.input_slices[node]]
            local_inputs[node] = v
            communication[node] = v_active[..., interconnection.communication_rows[node]]
            _, next_states[node] = self.rens[str(node)].step(v, states[node],
                                                           realization.locals[node])
        features = current_features(physical_state, voltage_errors, self.registry, topology)
        direction = torch.stack([self.directions[str(node)](features[..., positions[node], :])
                                 if node in active else zero
                                 for node in self.registry.node_ids], dim=-1)
        node_mask = ehat.new_tensor([float(node in active) for node in self.registry.node_ids])
        if mode == "mad":
            u = readout(z, direction, node_mask)
        elif mode == "direct":
            u = z*node_mask
        else:
            u = torch.zeros_like(z)
        return ControllerStep(z, u, direction, next_states, communication, features, local_inputs)


# Pnp


def affected_sets(old_plant, new_plant, controller=None):
    """A uses physical model/neighborhood changes; I uses certificate quantities.

    With a controller, compare actual certificate quantities, including weights.
    The degree-only fallback applies only to the fixed equal-weight architecture.
    A current-component change counts as a local physical-model change in A,
    although its shifted Jacobian and certificate quantities do not change.
    """
    r = old_plant.registry
    old = set(old_plant.topology.active_nodes)
    new = set(new_plant.topology.active_nodes)
    quantities = None
    if controller is not None:
        before_ic = controller.interconnection(old_plant.topology)
        after_ic = controller.interconnection(new_plant.topology)
        before_b = general_bounds(before_ic, controller.gamma_R)
        after_b = general_bounds(after_ic, controller.gamma_R)
        quantities = ({i: (before_b.hbar[k]+before_b.rho_z[k], before_b.cbar[k])
                       for k, i in enumerate(before_ic.node_ids)},
                      {i: (after_b.hbar[k]+after_b.rho_z[k], after_b.cbar[k])
                       for k, i in enumerate(after_ic.node_ids)})
    affected, influence = set(new-old), set(new-old)
    for i in sorted(old & new):
        before = r.neighbors(old_plant.topology, i)
        after = r.neighbors(new_plant.topology, i)
        model_changed = (old_plant.loads[i-1] != new_plant.loads[i-1]
                         or old_plant.zip_load != new_plant.zip_load
                         or old_plant.h != new_plant.h)
        if before != after or model_changed:
            affected.add(i)
        certificate_changed = (len(before) != len(after) if quantities is None else
                               any(not torch.equal(x.detach(), y.detach())
                                   for x, y in zip(quantities[0][i], quantities[1][i])))
        if certificate_changed:
            influence.add(i)
    if not influence <= affected:
        raise AssertionError("certificate influence must be a subset of affected nodes")
    return tuple(sorted(affected)), tuple(sorted(influence))


def network_storage(controller, states, realization):
    terms = [realization.gains.alpha[j]
             * controller.rens[str(i)].storage(states[i], realization.locals[i])
             for j, i in enumerate(realization.interconnection.active_nodes)]
    return torch.stack(terms).sum()


def transition_controller(controller, old_plant, new_plant, states, old_realization):
    """Reinstantiate prescribed gains, reset only newly activated/removed memory.

    Surviving state objects are retained without detach/copy, including gradients.
    Surviving realizations outside I are reused exactly, except for the explicit
    learned-edge gradient refresh on changed incident support. Network matrices
    always reflect the new physical graph, including equal-degree rewiring.
    """
    controller.check_frozen_edge_weights(old_realization.interconnection.registered_edge_weights.detach())
    affected, influence = affected_sets(old_plant, new_plant, controller=controller)
    old_nodes = set(old_plant.topology.active_nodes)
    new_nodes = set(new_plant.topology.active_nodes)
    next_states = controller.initial_state()
    for i in old_nodes & new_nodes:
        next_states[i] = states[i]
    realization = controller.realize(new_plant.topology)
    gradient_refresh = {i for i in (old_nodes & new_nodes)-set(influence)
                        if controller.learn_edge_weights
                        and controller.registry.neighbors(old_plant.topology, i)
                        != controller.registry.neighbors(new_plant.topology, i)}
    # Equal weighted sums imply equal gain VALUES, but a changed incident edge
    # can change which decision variable the gain depends on. Preserve the new
    # autograd graph in that case; free parameters and dynamic states stay put.
    for i in (old_nodes & new_nodes)-set(influence)-gradient_refresh:
        realization.locals[i] = old_realization.locals[i]
    before = network_storage(controller, states, old_realization)
    after = network_storage(controller, next_states, realization)
    report = {
        "A": list(affected), "I": list(influence),
        "gradient_refresh_nodes": sorted(gradient_refresh),
        "old_topology": old_plant.topology.name,
        "new_topology": new_plant.topology.name,
        "surviving_nodes": sorted(old_nodes & new_nodes),
        "reset_nodes": sorted(old_nodes ^ new_nodes),
        "surviving_state_objects_preserved": all(next_states[i] is states[i]
                                                 for i in old_nodes & new_nodes),
        "parameters_updated": False,
        "controller_state_before": {str(i): {"x": states[i].x.detach(), "b": states[i].b.detach()}
                                    for i in sorted(old_nodes)},
        "controller_state_after": {str(i): {"x": next_states[i].x.detach(), "b": next_states[i].b.detach()}
                                   for i in sorted(new_nodes)},
        "storage_before": float(before.detach()),
        "storage_after": float(after.detach()),
        "storage_jump": float((after-before).detach()),
        "arbitrary_switching_certificate": False,
    }
    return next_states, realization, report


# Certification


def ren_sector_certificate(realization):
    """Return reproducible local storage/sector evidence for one realization.

    For t=[x;w;v], let L=E^-1[F B1 B2], O=[C2 D21 D22], and
    Q=diag(P,0,gamma^2 I). The nonlinear equation obeys

        t' S t = 2 w' (C1 x + D11 w + D12 v - Lambda w) >= 0.

    Hence D=Q-L'PL-O'O-S >= 0 proves Delta(x'Px)<=gamma^2||v||^2-y^2.
    The published output is z=b, b_next=y, so adding b^2 to storage gives
    Delta(x'Px+b^2)<=gamma^2||v||^2-z^2 without a second message delay.

    Both physical and congruence-scaled matrices are returned.  The congruence
    normalizes the diagonal supply to diag(I,0,I); it changes no controller or
    residual coordinates. ``verified`` requires strict positive numerical
    eigenvalues, a positive storage, and the actual acyclic sector structure.
    This is a check of these finite matrices, not an interval-arithmetic proof.
    """
    matrices = {key: value.detach().to(device="cpu", dtype=torch.float64)
                for key, value in realization.matrices.items()}
    P = realization.P.detach().to(device="cpu", dtype=torch.float64)
    gamma = realization.gamma.detach().to(device="cpu", dtype=torch.float64).reshape(())
    required = {"E", "F", "B1", "B2", "C1", "D11", "D12", "C2", "D21", "D22", "Lambda"}
    if set(matrices) != required:
        raise ValueError("REN checker requires the complete realized matrix set")
    if not all(bool(torch.isfinite(value).all()) for value in (*matrices.values(), P, gamma)):
        raise ValueError("REN matrices, storage and gain must be finite")
    n, width, q = P.shape[0], matrices["B1"].shape[1], matrices["B2"].shape[1]
    shapes = {"E": (n, n), "F": (n, n), "B1": (n, width), "B2": (n, q),
              "C1": (width, n), "D11": (width, width), "D12": (width, q),
              "C2": (1, n), "D21": (1, width), "D22": (1, q), "Lambda": (width,)}
    if P.shape != (n, n) or any(matrices[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("REN checker matrix dimensions are inconsistent")
    if not bool(gamma > 0):
        raise ValueError("The prescribed local gain must be strictly positive")
    symmetry_error = float((P-P.T).abs().max())
    symmetric_P = (P+P.T)/2
    storage_eigenvalues, storage_vectors = torch.linalg.eigh(symmetric_P)
    if not bool((storage_eigenvalues > 0).all()):
        raise ValueError("REN storage is not positive definite")
    dynamics = torch.linalg.solve(matrices["E"],
                                  torch.cat((matrices["F"], matrices["B1"], matrices["B2"]), dim=1))
    output = torch.cat((matrices["C2"], matrices["D21"], matrices["D22"]), dim=1)
    supply = torch.block_diag(P, torch.zeros(width, width, dtype=P.dtype),
                              gamma.square()*torch.eye(q, dtype=P.dtype))
    sector = torch.zeros_like(supply)
    xs, ws, vs = slice(0, n), slice(n, n+width), slice(n+width, n+width+q)
    sector[ws, xs], sector[xs, ws] = matrices["C1"], matrices["C1"].T
    sector[ws, ws] = matrices["D11"]+matrices["D11"].T-2*torch.diag(matrices["Lambda"])
    sector[ws, vs], sector[vs, ws] = matrices["D12"], matrices["D12"].T
    dissipation = supply-dynamics.T@P@dynamics-output.T@output-sector
    dissipation = (dissipation+dissipation.T)/2
    storage_inverse_sqrt = ((storage_vectors/storage_eigenvalues.sqrt().unsqueeze(0))
                            @storage_vectors.T)
    congruence = torch.block_diag(storage_inverse_sqrt, torch.eye(width, dtype=P.dtype),
                                 torch.eye(q, dtype=P.dtype)/gamma)
    scaled = congruence.T@dissipation@congruence
    scaled = (scaled+scaled.T)/2
    eigenvalues, scaled_eigenvalues = torch.linalg.eigvalsh(dissipation), torch.linalg.eigvalsh(scaled)
    acyclic_error = float(torch.triu(matrices["D11"]).abs().max())
    sector_min = float(matrices["Lambda"].min())
    verified = (symmetry_error == 0 and acyclic_error == 0 and sector_min > 0
                and bool((eigenvalues > 0).all()) and bool((scaled_eigenvalues > 0).all()))
    return {"verified": bool(verified), "state_dim": n, "width": width, "input_dim": q,
            "gamma": float(gamma), "storage_min_eigenvalue": float(storage_eigenvalues.min()),
            "storage_symmetry_error": symmetry_error, "acyclic_upper_triangle_error": acyclic_error,
            "sector_diagonal_min": sector_min, "min_eigenvalue": float(eigenvalues.min()),
            "min_scaled_eigenvalue": float(scaled_eigenvalues.min()),
            "relative_min_eigenvalue": float(eigenvalues.min()/eigenvalues.abs().max()),
            "dynamics": dynamics, "output": output, "supply": supply, "sector": sector,
            "dissipation_matrix": dissipation, "scaled_dissipation_matrix": scaled,
            "congruence": congruence, "storage": P,
            "eigenvalues": eigenvalues, "scaled_eigenvalues": scaled_eigenvalues}

def response_jacobian(ren, realization):
    """True zero-origin state Jacobian, including the acyclic nonlinear feedback."""
    m = realization.matrices
    sector = torch.diag(m["Lambda"])-m["D11"]
    return torch.linalg.solve(m["E"], m["F"]+m["B1"]@torch.linalg.solve(sector, m["C1"]))

def certify(controller):
    maximum, local_min, scaled_min, count, poles = -math.inf, math.inf, math.inf, 0, {}
    with torch.no_grad():
        library = topology_library(controller.registry)
        for topology in library.values():
            real = controller.realize(topology)
            value=float(torch.linalg.eigvalsh(real.certificate).max())
            if not math.isfinite(value):raise RuntimeError("Nonfinite network certificate")
            maximum = max(maximum,value)
            for node, local in real.locals.items():
                check = ren_sector_certificate(local)
                if not check["verified"]:
                    raise RuntimeError(f"Independent local certificate failed:{topology.name}/{node}")
                local_min = min(local_min, check["min_eigenvalue"])
                scaled_min = min(scaled_min, check["min_scaled_eigenvalue"])
                ev = torch.linalg.eigvals(response_jacobian(controller.rens[str(node)], local))
                poles[f"{topology.name}/{node}"] = {"gamma": float(local.gamma),
                    "real": ev.real.tolist(), "imag": ev.imag.tolist(),
                    "spectral_radius": float(ev.abs().max()),
                    "time_constants_ms": (-.05/torch.log(ev.abs())).tolist()}
                count += 1
    if maximum > 1e-12:
        raise RuntimeError("Independent network certificate failed")
    return {"networks": len(library), "locals": count,
        "network_max_eigenvalue": maximum, "local_min_eigenvalue": local_min,
        "local_min_scaled_eigenvalue": scaled_min, "origin_modes": poles,
        "scope": "numerical matrices; no invariance/arbitrary-switching certificate"}

def checked_certificate(model):
    result = certify(model)
    if any(row["spectral_radius"] >= 1 or not math.isfinite(row["spectral_radius"])
           for row in result["origin_modes"].values()):
        raise RuntimeError("Realized REN origin dynamics are not numerically Schur stable")
    return result
