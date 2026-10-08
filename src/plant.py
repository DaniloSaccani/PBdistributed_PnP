"""Physical registry, fixed primary, ZIP plant and topology transitions."""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType


# Registry


@dataclass(frozen=True)
class NodeParameters:
    id: int
    Rt: float
    Lt: float
    Ct: float
    Y: float
    Ibar: float
    P: float
    Vref: float
    Vmax: float = 100.0


@dataclass(frozen=True)
class LineParameters:
    edge: tuple[int, int]
    R: float
    L: float

    def __post_init__(self):
        object.__setattr__(self, "edge", tuple(self.edge))


@dataclass(frozen=True)
class Topology:
    name: str
    active_nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    role: str = "custom"

    def __post_init__(self):
        nodes = tuple(self.active_nodes)
        edges = tuple(tuple(edge) for edge in self.edges)
        if tuple(sorted(set(nodes))) != nodes:
            raise ValueError("Active node IDs must be unique and increasing.")
        if len(set(edges)) != len(edges):
            raise ValueError("A topology cannot duplicate an edge.")
        for edge in edges:
            if len(edge) != 2 or edge[0] >= edge[1]:
                raise ValueError("Edges must retain the smaller-to-larger orientation.")
            if not set(edge).issubset(nodes):
                raise ValueError("An active edge requires both endpoints active.")
        object.__setattr__(self, "active_nodes", nodes)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True)
class Registry:
    nodes: tuple[NodeParameters, ...]
    lines: tuple[LineParameters, ...]
    _node_positions: object = field(init=False, repr=False, compare=False)
    _edge_positions: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "lines", tuple(self.lines))
        ids = tuple(node.id for node in self.nodes)
        if not ids or tuple(sorted(set(ids))) != ids or min(ids) < 1:
            raise ValueError("Permanent node IDs must be unique increasing positive integers.")
        for node in self.nodes:
            values = (node.Rt, node.Lt, node.Ct, node.Y, node.Ibar, node.P, node.Vref, node.Vmax)
            if not all(isfinite(value) for value in values):
                raise ValueError("DGU parameters must be finite.")
            if min(node.Rt, node.Lt, node.Ct, node.Vref) <= 0:
                raise ValueError("DGU resistance, inductance, capacitance, and reference must be positive.")
        edges = tuple(line.edge for line in self.lines)
        if len(set(edges)) != len(edges):
            raise ValueError("Registry line IDs must be unique.")
        for line in self.lines:
            if len(line.edge) != 2 or line.edge[0] >= line.edge[1] or not set(line.edge).issubset(ids):
                raise ValueError("A registered line needs two known, increasing endpoint IDs.")
            if not all(isfinite(value) and value > 0 for value in (line.R, line.L)):
                raise ValueError("Line resistance and inductance must be finite and positive.")
        object.__setattr__(self, "_node_positions", MappingProxyType({node: j for j, node in enumerate(ids)}))
        object.__setattr__(self, "_edge_positions", MappingProxyType({edge: j for j, edge in enumerate(edges)}))

    @property
    def node_ids(self):
        return tuple(node.id for node in self.nodes)

    @property
    def edges(self):
        return tuple(line.edge for line in self.lines)

    @property
    def input_dim(self):
        return len(self.nodes)

    @property
    def state_dim(self):
        return 3 * self.input_dim + len(self.lines)

    @property
    def voltage_indices(self):
        return tuple(3 * j for j in range(self.input_dim))

    def node_indices(self, node):
        if node not in self._node_positions:
            raise ValueError(f"Unknown permanent DGU ID {node!r}.")
        start = 3 * self._node_positions[node]
        return (start, start + 1, start + 2)

    def edge_index(self, edge):
        edge = tuple(edge)
        if edge not in self._edge_positions:
            raise ValueError(f"Unknown or incorrectly oriented line {edge!r}.")
        return 3 * self.input_dim + self._edge_positions[edge]

    def owned_edges(self, node):
        self.node_indices(node)
        return tuple(edge for edge in self.edges if edge[0] == node)

    def incident_edges(self, node):
        self.node_indices(node)
        return tuple(edge for edge in self.edges if node in edge)

    def owned_indices(self, node):
        return self.node_indices(node) + tuple(self.edge_index(edge) for edge in self.owned_edges(node))

    def validate_topology(self, topology):
        if not set(topology.active_nodes).issubset(self.node_ids):
            raise ValueError("Topology contains an unregistered DGU.")
        for edge in topology.edges:
            self.edge_index(edge)
            if not set(edge).issubset(topology.active_nodes):
                raise ValueError("An active line cannot touch an inactive DGU.")

    def physical_indices(self, topology):
        """Canonical ordering of the currently existing physical coordinates."""
        self.validate_topology(topology)
        nodes = tuple(index for node in topology.active_nodes for index in self.node_indices(node))
        lines = tuple(self.edge_index(edge) for edge in self.edges if edge in topology.edges)
        return nodes + lines

    def allocated_indices(self, topology):
        """Active owner blocks, retaining each active owner's unused line slots.

        This order stacks owner blocks; it is not canonical physical ordering.
        The two selectors agree in norm after absent physical slots are zeroed.
        """
        self.validate_topology(topology)
        return tuple(index for node in topology.active_nodes for index in self.owned_indices(node))

    def neighbors(self, topology, node):
        self.validate_topology(topology)
        self.node_indices(node)
        return tuple(sorted(edge[1] if edge[0] == node else edge[0]
                            for edge in topology.edges if node in edge))

    def node_mask(self, topology, device=None):
        self.validate_topology(topology)
        return torch.tensor([node in topology.active_nodes for node in self.node_ids],
                            dtype=torch.float64, device=device)

    def state_mask(self, topology, device=None):
        indices = set(self.physical_indices(topology))
        return torch.tensor([j in indices for j in range(self.state_dim)],
                            dtype=torch.float64, device=device)

    def gather_owned(self, state):
        """Gather one copy of every canonical coordinate; leading batches pass through."""
        if state.shape[-1] != self.state_dim:
            raise ValueError("State does not match the permanent registry.")
        return tuple(state[..., list(self.owned_indices(node))] for node in self.node_ids)

    def local_view(self, state, node, topology):
        """Read own triple and all permanent incident slots, masking absent data.

        Currents use the canonical orientation at BOTH endpoints. These indexed
        measurement views are never separate physical states or residual blocks.
        """
        self.validate_topology(topology)
        if state.shape[-1] != self.state_dim:
            raise ValueError("State does not match the permanent registry.")
        edges = self.incident_edges(node)
        indices = self.node_indices(node) + tuple(self.edge_index(edge) for edge in edges)
        enabled = [node in topology.active_nodes] * 3 + [edge in topology.edges for edge in edges]
        values = state[..., list(indices)]
        mask = torch.tensor(enabled, dtype=torch.bool, device=values.device)
        return torch.where(mask, values, torch.zeros_like(values))


def build_registry():
    """Build the frozen six-DGU/ten-line data registry (REQ-084--085)."""
    nodes = (
        NodeParameters(1, .20, .0018, .0022, .50, 8.0, 6.0, 47.9),
        NodeParameters(2, .30, .0020, .0019, .45, 7.0, 5.0, 48.0),
        NodeParameters(3, .10, .0022, .0017, .55, 7.5, 6.5, 47.7),
        NodeParameters(4, .50, .0030, .0025, .40, 6.0, 4.5, 48.0),
        NodeParameters(5, .40, .0012, .0020, .35, 6.5, 5.0, 47.8),
        NodeParameters(6, .25, .0016, .0021, .42, 7.2, 5.5, 48.1),
    )
    lines = tuple(LineParameters(edge, resistance, .001) for edge, resistance in (
        ((1, 2), .05), ((1, 3), .07), ((2, 4), .04), ((3, 4), .06), ((4, 5), .08),
        ((1, 6), .10), ((2, 6), .09), ((3, 6), .11), ((4, 6), .085), ((5, 6), .08),
    ))
    return Registry(nodes, lines)


def topology_library(registry=None):
    """All nine physical graphs; focused training membership is defined in fixtures.py."""
    registry = build_registry() if registry is None else registry
    base = {(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)}
    descriptions = (
        ("pre_5_dgu", (1, 2, 3, 4, 5), base, "pre"),
        ("plug_05_35", (1, 2, 3, 4, 5, 6), base | {(1, 6), (4, 6)}, "regression"),
        ("plug_15_45", (1, 2, 3, 4, 5, 6), base | {(2, 6), (5, 6)}, "regression"),
        ("plug_05_25", (1, 2, 3, 4, 5, 6), base | {(1, 6), (3, 6)}, "regression"),
        ("plugout_1", (2, 3, 4, 5), base, "regression"),
        ("plugout_5", (1, 2, 3, 4), base, "regression"),
        ("plug_05_45", (1, 2, 3, 4, 5, 6), base | {(1, 6), (5, 6)}, "regression"),
        ("plugout_3", (1, 2, 4, 5), base, "regression"),
        ("plug_05_45_unplug_3", (1, 2, 4, 5, 6), base | {(1, 6), (5, 6)}, "regression"),
    )
    result = {}
    for name, active, candidates, role in descriptions:
        selected = tuple(edge for edge in registry.edges if edge in candidates and set(edge).issubset(active))
        if not candidates.issubset(registry.edges):
            raise ValueError("The frozen topology library requires the complete line registry.")
        topology = Topology(name, active, selected, role)
        registry.validate_topology(topology)
        result[name] = topology
    return MappingProxyType(result)


# Primary


def _immutable(value, name):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return tuple(_immutable(item, name) for item in value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must contain real numbers, not booleans")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PrimaryGains:
    k1: object = .5
    k2_ratio: object = .5
    k3_fraction: object = .5

    def __post_init__(self):
        for name in ("k1", "k2_ratio", "k3_fraction"):
            value = _immutable(getattr(self, name), name)
            values = value if isinstance(value, tuple) else (value,)
            if any(isinstance(item, tuple) for item in values):
                raise ValueError("Primary gains must be scalars or flat vectors")
            if name in ("k1", "k2_ratio") and any(item >= 1 for item in values):
                raise ValueError(f"{name} must be strictly below one")
            if name == "k3_fraction" and any(not 0 < item < 1 for item in values):
                raise ValueError("k3_fraction must be strictly between zero and one")
            object.__setattr__(self, name, value)

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Primary gains require a PrimaryGains object or dictionary")
        return cls(**value)

    def resolve(self, registry):
        count = registry.input_dim
        def vector(value):
            if isinstance(value, tuple):
                if len(value) != count:
                    raise ValueError("Primary vector must match permanent DGU order")
                return torch.tensor(value, dtype=torch.float64)
            return torch.full((count,), value, dtype=torch.float64)
        k1 = vector(self.k1)
        rt = torch.tensor([node.Rt for node in registry.nodes], dtype=torch.float64)
        lt = torch.tensor([node.Lt for node in registry.nodes], dtype=torch.float64)
        k2 = vector(self.k2_ratio)*rt
        upper = (1-k1)*(rt-k2)/lt
        # Preserve the legacy arithmetic order in the default configuration.
        fraction = vector(self.k3_fraction)
        k3 = (1-k1)*(rt-k2)/(2*lt) if bool(torch.all(fraction == .5)) else fraction*upper
        if not bool(torch.isfinite(k3).all() & (k3 > 0).all() & (k3 < upper).all()):
            raise ValueError("Resolved primary gains do not strictly satisfy Nahata inequalities")
        return k1, k2, k3

    def manifest(self, registry):
        k1, k2, k3 = self.resolve(registry)
        return {"node_ids": list(registry.node_ids), "k1": k1.tolist(),
                "k2": k2.tolist(), "k3": k3.tolist(),
                "definition": "k2=ratio*Rt; k3=fraction*(1-k1)*(Rt-k2)/Lt",
                "scope": "offline fixed primary tuning; no online gain changes"}


# Plant


class VoltageDomainError(ValueError):
    """An active physical voltage left the declared positive-voltage domain."""


class Plant:
    def __init__(self, registry, topology, h=5e-5, loads=None, zip_load=True, primary=None):
        registry.validate_topology(topology)
        if not isfinite(h) or h <= 0:
            raise ValueError("The Euler interval must be finite and positive.")
        self.registry, self.topology = registry, topology
        self.h, self.zip_load = float(h), bool(zip_load)
        self.loads = tuple(node.Ibar for node in registry.nodes) if loads is None else tuple(float(x) for x in loads)
        if len(self.loads) != registry.input_dim or not all(isfinite(x) for x in self.loads):
            raise ValueError("Loads must contain one finite current component per permanent DGU.")
        self.params = registry.nodes
        for name in ("Ct", "Lt", "Rt", "Y", "Vref"):
            setattr(self, name, torch.tensor([getattr(node, name) for node in registry.nodes], dtype=torch.float64))
        self.P = torch.tensor([node.P if self.zip_load else 0.0 for node in registry.nodes], dtype=torch.float64)
        self.Ibar = torch.tensor(self.loads, dtype=torch.float64)
        self.Rline = torch.tensor([line.R for line in registry.lines], dtype=torch.float64)
        self.Lline = torch.tensor([line.L for line in registry.lines], dtype=torch.float64)
        self._primary = PrimaryGains.from_value(primary)
        self.k1, self.k2, self.k3 = self.primary.resolve(registry)
        self._frozen_primary = tuple(value.clone() for value in (self.k1, self.k2, self.k3))
        self._node_mask = registry.node_mask(topology).bool()
        self._state_mask = registry.state_mask(topology).bool()
        self._line_mask = torch.tensor([edge in topology.edges for edge in registry.edges], dtype=torch.bool)
        self.incidence = torch.zeros((registry.input_dim, len(registry.lines)), dtype=torch.float64)
        for j, line in enumerate(registry.lines):
            if line.edge in topology.edges:
                tail, head = line.edge
                self.incidence[registry.node_ids.index(tail), j] = 1
                self.incidence[registry.node_ids.index(head), j] = -1
        self.A, self.B = self._build_linear_model()
        self.Bu = self.B
        self._equilibrium = self._build_equilibrium()

    @property
    def primary(self):
        """Read-only experiment definition; use a new Plant for a new tuning."""
        return self._primary

    def check_frozen_primary(self):
        if any(not torch.equal(value, expected) for value, expected in
               zip((self.k1, self.k2, self.k3), self._frozen_primary)):
            raise RuntimeError("Primary gains changed behind cached predictor/equilibrium; construct a new experiment")

    def primary_manifest(self):
        self.check_frozen_primary()
        resolved = self.primary.resolve(self.registry)
        if any(not torch.equal(value, expected) for value, expected in zip(resolved, self._frozen_primary)):
            raise RuntimeError("Primary definition disagrees with the actual cached gain vectors")
        return self.primary.manifest(self.registry)

    def _state(self, value):
        state = torch.as_tensor(value, dtype=torch.float64)
        if state.ndim < 1 or state.shape[-1] != self.registry.state_dim:
            raise ValueError("Expected a canonical state vector, optionally with leading batches.")
        return state

    def _input(self, value, state):
        value = torch.zeros(self.registry.input_dim, dtype=state.dtype, device=state.device) if value is None else torch.as_tensor(value, dtype=state.dtype, device=state.device)
        if value.ndim < 1 or value.shape[-1] != self.registry.input_dim:
            raise ValueError("Expected one boosting input per permanent DGU.")
        return torch.where(self._node_mask.to(state.device), value, torch.zeros_like(value))

    def _physical(self, state):
        return torch.where(self._state_mask.to(state.device), state, torch.zeros_like(state))

    def _check_voltage(self, voltages):
        active = voltages[..., self._node_mask.to(voltages.device)]
        if bool(torch.any((active <= 0) | ~torch.isfinite(active))):
            raise VoltageDomainError("Active voltages must be finite and strictly positive; no denominator clipping is used.")

    def _build_equilibrium(self):
        voltage = self.Vref
        line = (voltage @ self.incidence) / self.Rline
        current = self.Y*voltage + self.Ibar + self.P/voltage + line @ self.incidence.T
        integral = ((1-self.k1)*voltage + (self.Rt-self.k2)*current) / self.k3
        triples = torch.stack((voltage, current, integral), dim=-1).flatten()
        return self._physical(torch.cat((triples, line)))

    def equilibrium(self):
        """Return the analytic physical equilibrium, with absent slots zero."""
        self.check_frozen_primary()
        return self._equilibrium.clone()

    def isolated_equilibrium(self, node):
        """Own triple at Vref with this plant's current load and no line current."""
        self.check_frozen_primary()
        j = self.registry.node_ids.index(node)
        current = self.Y[j]*self.Vref[j] + self.Ibar[j] + self.P[j]/self.Vref[j]
        integral = ((1-self.k1[j])*self.Vref[j] + (self.Rt[j]-self.k2[j])*current)/self.k3[j]
        return torch.stack((self.Vref[j], current, integral))

    def rhs(self, X, u=None):
        """Continuous reference RHS, evaluated once from the common old state."""
        self.check_frozen_primary()
        X = self._physical(self._state(X))
        u = self._input(u, X)
        n = self.registry.input_dim
        triples = X[..., :3*n].reshape(*X.shape[:-1], n, 3)
        V, It, integral = triples.unbind(dim=-1)
        self._check_voltage(V)
        V, It, integral, u = torch.broadcast_tensors(V, It, integral, u)
        node_mask = self._node_mask.to(X.device)
        # Inactive voltages are absent coordinates, not evaluations of P/0.
        denominator = torch.where(node_mask, V, torch.ones_like(V))
        incidence = self.incidence.to(X)
        lines = X[..., 3*n:]
        dV = (It-self.Y.to(X)*V-self.Ibar.to(X)-self.P.to(X)/denominator-lines@incidence.T)/self.Ct.to(X)
        dIt = ((self.k1.to(X)-1)*V+(self.k2.to(X)-self.Rt.to(X))*It+self.k3.to(X)*integral+u)/self.Lt.to(X)
        dIntegral = self.Vref.to(X)-V
        local = torch.stack((dV, dIt, dIntegral), dim=-1).flatten(start_dim=-2)
        dLine = (V@incidence-self.Rline.to(X)*lines)/self.Lline.to(X)
        return self._physical(torch.cat((local, dLine), dim=-1))

    def step(self, X, u=None):
        """One simultaneous explicit Euler step; inactive coordinates are deadbeat."""
        X = self._physical(self._state(X))
        result = X + self.h*self.rhs(X, u)
        self._check_voltage(result[..., list(self.registry.voltage_indices)])
        return result

    def local_step(self, X, u=None):
        """Equivalent endpoint-local evaluation, using only old-state views.

        Each line is updated once by its owner. This explicit loop is separate
        from the incidence-matrix RHS so local/global comparisons are meaningful.
        """
        self.check_frozen_primary()
        X = self._physical(self._state(X))
        u = self._input(u, X)
        self._check_voltage(X[..., list(self.registry.voltage_indices)])
        zero = torch.zeros_like(X[..., 0] + u[..., 0])
        values = [zero for _ in range(self.registry.state_dim)]
        for j, node in enumerate(self.registry.nodes):
            if node.id not in self.topology.active_nodes:
                continue
            iv, ii, iz = self.registry.node_indices(node.id)
            V, It, integral = X[..., iv], X[..., ii], X[..., iz]
            outward = zero
            for edge in self.topology.edges:
                if node.id in edge:
                    sign = 1 if edge[0] == node.id else -1
                    outward = outward + sign*X[..., self.registry.edge_index(edge)]
            power = node.P if self.zip_load else 0.0
            dV = (It-node.Y*V-self.loads[j]-power/V-outward)/node.Ct
            dIt = ((self.k1[j]-1).to(X)*V+(self.k2[j]-node.Rt).to(X)*It+self.k3[j].to(X)*integral+u[..., j])/node.Lt
            values[iv] = V+self.h*dV
            values[ii] = It+self.h*dIt
            values[iz] = integral+self.h*(node.Vref-V)
        for line in self.registry.lines:
            if line.edge in self.topology.edges:
                index = self.registry.edge_index(line.edge)
                left = self.registry.node_indices(line.edge[0])[0]
                right = self.registry.node_indices(line.edge[1])[0]
                values[index] = X[..., index]+self.h*(X[..., left]-X[..., right]-line.R*X[..., index])/line.L
        # Broadcast vector state against a batch of inputs when requested.
        result = torch.stack([value+zero for value in values], dim=-1)
        self._check_voltage(result[..., list(self.registry.voltage_indices)])
        return result

    def _build_linear_model(self):
        n, dim = self.registry.input_dim, self.registry.state_dim
        Ac = torch.zeros((dim, dim), dtype=torch.float64)
        Bu = torch.zeros((dim, n), dtype=torch.float64)
        for j, node in enumerate(self.registry.nodes):
            if node.id not in self.topology.active_nodes:
                continue
            iv, ii, iz = self.registry.node_indices(node.id)
            Ac[iv, iv] = (-node.Y+self.P[j]/node.Vref**2)/node.Ct
            Ac[iv, ii] = 1/node.Ct
            Ac[ii, iv] = (self.k1[j]-1)/node.Lt
            Ac[ii, ii] = (self.k2[j]-node.Rt)/node.Lt
            Ac[ii, iz] = self.k3[j]/node.Lt
            Ac[iz, iv] = -1
            Bu[ii, j] = self.h/node.Lt
        for line in self.registry.lines:
            if line.edge not in self.topology.edges:
                continue
            row = self.registry.edge_index(line.edge)
            Ac[row, row] = -line.R/line.L
            for endpoint, sign in zip(line.edge, (1, -1)):
                j = self.registry.node_ids.index(endpoint)
                voltage = self.registry.node_indices(endpoint)[0]
                Ac[voltage, row] = -sign/self.Ct[j]
                Ac[row, voltage] = sign/line.L
        return torch.diag(self._state_mask.to(torch.float64))+self.h*Ac, Bu

    def linear_model(self):
        """Cached padded A and Bu; inactive auxiliary rows and columns are zero."""
        self.check_frozen_primary()
        return self.A, self.B

    def predict(self, x, u=None):
        """Nominal next deviation for the same applied boosting command."""
        self.check_frozen_primary()
        x = self._state(x)
        u = self._input(u, x)
        return x@self.A.to(x).T + u@self.B.to(x).T

    def remainder(self, x):
        """Exact Euler ZIP remainder from §10.4, in unscaled physical units."""
        x = self._physical(self._state(x))
        v = x[..., list(self.registry.voltage_indices)]
        Vref = self.Vref.to(x)
        self._check_voltage(Vref+v)
        residual = -self.h*self.P.to(x)*v.square()/(self.Ct.to(x)*Vref.square()*(Vref+v))
        residual = torch.where(self._node_mask.to(x.device), residual, torch.zeros_like(residual))
        indices = torch.tensor(self.registry.voltage_indices, dtype=torch.long, device=x.device)
        return torch.zeros_like(x).index_copy(-1, indices, residual)

    def voltage_errors(self, X):
        X = self._state(X)
        errors = X[..., list(self.registry.voltage_indices)]-self.Vref.to(X)
        return torch.where(self._node_mask.to(X.device), errors, torch.zeros_like(errors))


# Events


@dataclass(frozen=True)
class Transition:
    J: torch.Tensor
    s: torch.Tensor

    def _state(self, value):
        value = torch.as_tensor(value, dtype=torch.float64)
        if value.ndim < 1 or value.shape[-1] != self.J.shape[1]:
            raise ValueError("Event state dimension does not match its physical transfer.")
        return value

    def apply(self, X):
        X = self._state(X)
        return X@self.J.to(X).T+self.s.to(X)

    def transport(self, prediction):
        """Carry ONLY J p; adding the equilibrium impulse here would erase it."""
        prediction = self._state(prediction)
        return prediction@self.J.to(prediction).T

    def impulse(self, old_equilibrium, new_equilibrium):
        old_equilibrium = self._state(old_equilibrium)
        new_equilibrium = self._state(new_equilibrium).to(old_equilibrium)
        return self.apply(old_equilibrium)-new_equilibrium


def make_transition(old_plant, new_plant):
    """Preserve survivors; reset new nodes to isolated equilibrium and lines to 0.

    A load-only boundary is identity transport, including padded coordinates.
    At a topology boundary the diagonal selector retains only surviving physical
    coordinates, so reactivation cannot recover stale inactive state values.
    No controller memory is handled here: the PnP controller layer owns it.
    """
    registry = old_plant.registry
    if registry != new_plant.registry:
        raise ValueError("An event cannot silently change the permanent registry.")
    if old_plant.primary_manifest() != new_plant.primary_manifest():
        raise ValueError("Primary gains must remain fixed throughout an event rollout")
    old_top, new_top = old_plant.topology, new_plant.topology
    dim = registry.state_dim
    offset = torch.zeros(dim, dtype=torch.float64)
    if old_top.active_nodes == new_top.active_nodes and set(old_top.edges) == set(new_top.edges):
        return Transition(torch.eye(dim, dtype=torch.float64), offset)
    surviving = set(registry.physical_indices(old_top)) & set(registry.physical_indices(new_top))
    diagonal = torch.tensor([j in surviving for j in range(dim)], dtype=torch.float64)
    for node in set(new_top.active_nodes)-set(old_top.active_nodes):
        indices = torch.tensor(registry.node_indices(node), dtype=torch.long)
        offset = offset.index_copy(0, indices, new_plant.isolated_equilibrium(node))
    return Transition(torch.diag(diagonal), offset)
