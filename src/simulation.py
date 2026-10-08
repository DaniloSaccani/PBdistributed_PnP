"""Physical fixtures, causal IMC rollouts and voltage diagnostics."""
from __future__ import annotations

import torch
from dataclasses import dataclass
from numbers import Integral
from .controller import transition_controller
from .plant import topology_library, Plant, make_transition


# Scenarios


@dataclass
class Scenario:
    name: str
    plant: Plant
    steps: int
    events: dict
    initial_state: torch.Tensor
    event_sample: int
    time_origin_s: float = 0.
    description: str = ""

    def manifest(self):
        return {"name": self.name, "initial_topology": self.plant.topology.name,
                "loads_A": list(self.plant.loads), "h_s": self.plant.h,
                "steps": self.steps, "time_origin_s": self.time_origin_s,
                "duration_s": self.steps*self.plant.h, "event_sample": self.event_sample,
                "event_time_s": self.time_origin_s+self.event_sample*self.plant.h,
                "initial_state": self.initial_state.detach().tolist(),
                "primary_gains": self.plant.primary_manifest(),
                "events": {str(k): {"topology": v.topology.name, "loads_A": list(v.loads),
                                     "primary_gains": v.primary_manifest()}
                           for k, v in self.events.items()}, "description": self.description}


def experiment_loads(registry):
    """New independent scenarios explicitly use4.4A for the sixth current component."""
    return tuple(4.4 if node.id == 6 else node.Ibar for node in registry.nodes)


def fixed_scenario(registry, topology_name, *, h=5e-5, duration=.03, seed=0, zip_load=True,
                   perturbation=.4, primary=None):
    topology = topology_library(registry)[topology_name]
    plant = Plant(registry, topology, h=h, loads=experiment_loads(registry), zip_load=zip_load, primary=primary)
    generator = torch.Generator().manual_seed(seed)
    # New declared distribution in physical units; not residual preprocessing.
    scales = torch.tensor([perturbation, 3*perturbation, .25*perturbation]*6
                          + [1.25*perturbation]*10, dtype=torch.float64)
    dx = torch.randn(28, generator=generator, dtype=torch.float64)*scales*registry.state_mask(topology)
    return Scenario(f"fixed_{topology_name}_seed{seed}", plant, round(duration/h), {},
                    plant.equilibrium()+dx, 0,
                    description="fixed-topology initial physical perturbation; no event labels to controller")


def validation_suite(registry, *, h=5e-5, duration=.15, seed=3100, primary=None, perturbation=.4):
    """Post-selection regression suite; historically inspected during design."""
    result = [fixed_scenario(registry, name, h=h, duration=duration, seed=seed+j, perturbation=perturbation, primary=primary)
              for j, name in enumerate(topology_library(registry))]
    result.extend(representative_scenarios(registry, h=h, before=.005, after=duration, primary=primary))
    return result


def representative_scenarios(registry, *, h=5e-5, before=.025, after=.5, primary=None):
    """Four separately initialized event windows, not a concatenated trajectory.

    Absolute labels4/8/12s locate the event; simulation begins before it at the
    documented equilibrium. We do not claim simulation from time0 or inherited
    loads. Each reference graph/controller initially starts in equilibrium/zero.
    """
    library = topology_library(registry)
    loads = experiment_loads(registry)
    pre = Plant(registry, library["pre_5_dgu"], h=h, loads=loads, primary=primary)
    plug = Plant(registry, library["plug_05_45"], h=h, loads=loads, primary=primary)
    unplug = Plant(registry, library["plug_05_45_unplug_3"], h=h, loads=loads, primary=primary)
    pre_load = list(loads); pre_load[4] += 1.3
    post_load = list(loads); post_load[5] += 1.3
    definitions = (
        ("pre_event_load", pre, Plant(registry, pre.topology, h=h, loads=pre_load, primary=primary), 4., "DGU5 Ibar +1.3A"),
        ("plugin_DGU6", pre, plug, 4., "DGU6 isolated equilibrium4.4A; new line currents0"),
        ("post_event_load", plug, Plant(registry, plug.topology, h=h, loads=post_load, primary=primary), 8., "DGU6 Ibar4.4to5.7A"),
        ("unplug_DGU3", plug, unplug, 12., "remove DGU3 from6DGU representative graph"),
    )
    k = round(before/h)
    return [Scenario(name, p0, round((before+after)/h), {k: p1}, p0.equilibrium(), k,
                     label-k*h, desc+"; independent experiment")
            for name, p0, p1, label, desc in definitions]


# Imc


def reconstruct_residual(X, plant, carried_prediction):
    return X - plant.equilibrium() - carried_prediction


def rollout(initial_plant, controller, steps, events=None, initial_state=None,
            mode="mad", controller_states=None):
    """Full differentiable rollout; no truncation of the temporal graph.

    Event index k is a boundary at k*h: previous propagation, event map, then
    observation/control. X[k] is post-event, u[k] drives interval k to k+1.
    The last observation X[steps] has no applied input entry. ``initial_state``
    means the physical state; dynamic REN state has its own explicit argument.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    events = {} if events is None else dict(events)
    if any(not isinstance(k, Integral) or isinstance(k, bool) for k in events):
        raise ValueError("event indices must be integral sample boundaries")
    if any(k < 0 or k > steps for k in events):
        raise ValueError("event index outside rollout")
    plant = initial_plant
    r = plant.registry
    X = plant.equilibrium() if initial_state is None else initial_state
    if X.shape != (r.state_dim,):
        raise ValueError("rollout expects a single canonical physical-state vector")
    if bool(torch.any(X[~r.state_mask(plant.topology).bool()] != 0)):
        raise ValueError("inactive physical initial coordinates must be zero; no ghost residuals")
    pred = torch.zeros_like(X)
    states = controller.initial_state() if controller_states is None else controller_states
    frozen_weights = controller.effective_edge_weights().detach().clone()
    realization = controller.realize(plant.topology)
    records = {key: [] for key in ("X", "x", "ehat", "prediction", "active", "voltage_error",
                                  "u", "z", "direction", "communication", "features",
                                  "remainder", "event_impulse", "Vc", "residual_identity_error")}
    topologies, reports = [], []
    previous_remainder = torch.zeros_like(X)
    for k in range(steps+1):
        controller.check_frozen_edge_weights(frozen_weights)
        impulse = torch.zeros_like(X)
        diagnostic = previous_remainder if k else X-plant.equilibrium()
        if k in events:
            following = events[k]
            if following.h != initial_plant.h:
                raise ValueError("sample time is fixed during a rollout")
            transfer = make_transition(plant, following)
            impulse = transfer.impulse(plant.equilibrium(), following.equilibrium())
            diagnostic = transfer.transport(diagnostic) + impulse
            X = transfer.apply(X)
            pred = transfer.transport(pred)
            states, realization, info = transition_controller(
                controller, plant, following, states, realization)
            plant = following
            info.update({"sample": k, "time_s": k*plant.h,
                         "J": transfer.J.detach(), "s": transfer.s.detach(),
                         "physical_initial_deviation": (X-plant.equilibrium()).detach(),
                         "prediction_carried": pred.detach(), "event_impulse": impulse.detach(),
                         "model_remainder": previous_remainder.detach()})
            reports.append(info)
        x = X-plant.equilibrium()
        ehat = reconstruct_residual(X, plant, pred)
        active = r.node_mask(plant.topology)
        verr = plant.voltage_errors(X)
        for key, value in (("X", X), ("x", x), ("ehat", ehat), ("prediction", pred),
                           ("active", active), ("voltage_error", verr),
                           ("event_impulse", impulse),
                           ("residual_identity_error", ehat-diagnostic)):
            records[key].append(value)
        topologies.append(plant.topology.name)
        if k == steps:
            break
        if mode == "baseline":
            z = u = direction = communication = torch.zeros(6, dtype=X.dtype)
            features = X.new_zeros((r.input_dim, getattr(controller, "mad_feature_dim", 2)))
        else:
            if hasattr(controller, "step_observed"):
                # Optional external MAD measurements. No change to the residual,
                # prediction, certified interconnection or reference plant step.
                result = controller.step_observed(ehat, verr, plant.topology, states,
                    realization, physical_state=X, mode=mode)
            else:
                result = controller.step(ehat, verr, plant.topology, states, realization, mode=mode)
            u, z, direction, states = result.u, result.z, result.direction, result.next_states
            if isinstance(result.communication, dict):
                communication = torch.stack([result.communication.get(i, X.new_zeros(()))
                                             for i in r.node_ids]).reshape(6)
            else:
                communication = result.communication
            features = result.features
        V, It, integrator = X[:18].reshape(6, 3).unbind(-1)
        Vc = (plant.k1*V+plant.k2*It+plant.k3*integrator+u)*active
        predicted_next = plant.predict(x, u)
        Xnext = plant.step(X, u)
        previous_remainder = Xnext-plant.equilibrium()-predicted_next
        for key, value in (("u", u), ("z", z), ("direction", direction),
                           ("communication", communication), ("features", features),
                           ("remainder", previous_remainder), ("Vc", Vc)):
            records[key].append(value)
        pred, X = predicted_next, Xnext
    result = {key: torch.stack(values) for key, values in records.items()}
    result.update({"topologies": topologies, "event_reports": reports,
                   "final_controller_state": states, "final_realization": realization,
                   "h": initial_plant.h, "mode": mode,
                   "clipping_count": 0, "integrator": "single_simultaneous_Euler"})
    return result


# Simulation

def run_scenario(scenario, controller, mode="mad"):
    trace = rollout(scenario.plant, controller, scenario.steps, scenario.events,
                    initial_state=scenario.initial_state, mode=mode)
    trace["time_origin_s"] = scenario.time_origin_s
    return trace


# Metrics


def trajectory_metrics(trace, start_sample=0):
    error = trace["voltage_error"][start_sample:]
    active = trace["active"][start_sample:]
    absolute = error.abs()*active
    denom = active.sum()
    if float(denom) <= 0:
        raise ValueError("metric window contains no active node")
    candidates = torch.where(active.bool(), absolute, torch.full_like(absolute, -float("inf")))
    flat = int(candidates.argmax())
    t, node = divmod(flat, absolute.shape[-1])
    index = t+start_sample
    u = trace["u"][start_sample:]
    control_mask = trace["active"][start_sample:-1]
    vc = trace["Vc"][start_sample:]
    return {
        "peak_V": float(absolute[t, node]),
        "peak_DGU": node+1, "peak_sample": index,
        "peak_time_s": index*trace["h"], "signed_deviation_V": float(error[t, node]),
        "peak_time_relative_s": index*trace["h"],
        "peak_time_label_s": trace.get("time_origin_s", 0.)+index*trace["h"],
        "rms_V": float(((error.square()*active).sum()/denom).sqrt()),
        "U_pk_V": float((u.abs()*control_mask).max()) if u.numel() else 0.,
        "U_rms_V": float(((u.square()*control_mask).sum()/control_mask.sum()).sqrt())
        if u.numel() and float(control_mask.sum()) > 0 else 0.,
        "sampled_within_10V": bool(torch.all(absolute <= 10)),
        "sampled_within_20V": bool(torch.all(absolute <= 20)),
        "sampled_node_exceedances_10V": int(((absolute > 10)&(active > 0)).sum()),
        "Vc_min_active_V": float(vc[control_mask.bool()].min()) if vc.numel() else None,
        "Vc_max_active_V": float(vc[control_mask.bool()].max()) if vc.numel() else None,
        "would_saturate_sample_nodes": int((((vc < 0)|(vc > 100)) & control_mask.bool()).sum()),
        "saturation_applied": False,
        "max_residual_identity_error": float(trace["residual_identity_error"].abs().max()),
        "max_MAD_amplitude_excess": float((trace["u"].abs()-trace["z"].abs()).max()),
        "forward_invariance": "UNVERIFIED",
    }


def compare_with_baseline(trace, baseline, start_sample=0):
    neural = trajectory_metrics(trace, start_sample)
    base = trajectory_metrics(baseline, start_sample)
    for key in ("peak_V", "rms_V"):
        neural[key.replace("_V", "_reduction_percent")] = (
            100*(1-neural[key]/base[key]) if base[key] > 0 else None)
    shared = trace["active"][start_sample:]*baseline["active"][start_sample:]
    neural["Delta_V_max_V"] = float(((trace["voltage_error"][start_sample:]
                                        -baseline["voltage_error"][start_sample:]).abs()*shared).max())
    neural["baseline"] = base
    return neural


def suite_maximum(metrics):
    if not metrics:
        raise ValueError("empty validation suite")
    name = max(metrics, key=lambda key: metrics[key]["peak_V"])
    return {"scenario": name, **metrics[name]}
