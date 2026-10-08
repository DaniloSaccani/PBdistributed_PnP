"""Physical objective, baseline calibration, balanced fixtures and full-BPTT training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import random
import time
import torch
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from math import isfinite, cos, pi
from numbers import Integral
from pathlib import Path
from .controller import checked_certificate, CurrentAwareController
from .model_io import ROOT, DEFAULT_MODEL, load_model, save_checkpoint
from .plant import Plant, topology_library, VoltageDomainError
from .settings import default_parameters
from .simulation import trajectory_metrics, run_scenario, Scenario, experiment_loads, fixed_scenario


# Losses


BASE_WEIGHTS = {
    "event_excursion": 900., "peak_excursion": 1800., "regulation": 420.,
    "control": 5e-7, "smooth": 5e-5, "terminal": 3800., "settling": 24000.,
    "settling_excursion": 5000., "settling_peak": 2500.,
    "time_weighted_settling": 52000., "time_weighted_excursion": 8000.,
    "voltage_oscillation": 150000., "late_control": .05,
    "tail_bias": 12000., "tail_excursion": 4000., "smooth_tube_count": 180.,
    "tube10_mean": 200., "tube10_peak": 500.,
}


@dataclass(frozen=True)
class LossConfig:
    voltage_tolerance: float = .035
    smooth_tube_tau: float = .005
    tube_radius: float = 10.
    reference_difference_step: float = .0005
    event_window_s: float = .48
    settling_delay_s: float = .021
    tail_window_s: float = .27
    weights: dict = field(default_factory=lambda: dict(BASE_WEIGHTS))


def _mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1.)


def rich_loss(trace, event_samples=(0,), config=None):
    """Return (weighted scalar, raw component dictionary).

    Observation k and the control applied at k are aligned. Terminal observation
    contributes separately; effort never penalizes un-applied pre-MAD z. First
    and second differences are converted to their old .5ms interval equivalents.
    """
    cfg = LossConfig() if config is None else config
    error = trace["voltage_error"][:-1]
    active = trace["active"][:-1]
    u = trace["u"]
    T = len(u)
    if error.shape != u.shape:
        raise ValueError("aligned voltage/control shapes required")
    h = trace["h"]
    sample_indices = torch.arange(T, device=error.device)
    t = sample_indices.to(error.dtype)*h
    event_mask = torch.zeros_like(t)
    settling = torch.zeros_like(t)
    age_mask = torch.zeros_like(t)
    for k in event_samples:
        # Subtract integral indices before multiplication: shifting an event
        # must not change boundary inclusion through cancellation of floats.
        age = (sample_indices-k).to(error.dtype)*h
        window = ((age >= 0) & (age < cfg.event_window_s)).to(error.dtype)
        late = window*(age >= cfg.settling_delay_s).to(error.dtype)
        event_mask = torch.maximum(event_mask, window)
        settling = torch.maximum(settling, late)
        age_weight = (age-cfg.settling_delay_s+h).clamp_min(0)/max(
            min(cfg.event_window_s, T*h)-cfg.settling_delay_s, h)
        age_mask = torch.maximum(age_mask, late*age_weight)
    ev = event_mask[:, None]*active
    st = settling[:, None]*active
    age = age_mask[:, None]*active
    excess = torch.relu(error.abs()-cfg.voltage_tolerance)
    zero = error.sum()*0
    components = {
        "event_excursion": _mean(excess.square(), ev),
        "peak_excursion": (excess*ev).amax(),
        "regulation": _mean(error.square(), active),
        "control": _mean(u.square(), active),
        "terminal": _mean(trace["voltage_error"][-1].square(), trace["active"][-1]),
        "settling": _mean(error.square(), st),
        "settling_excursion": _mean(excess.square(), st),
        "settling_peak": (excess*st).amax(),
        "time_weighted_settling": _mean(error.square(), age),
        "time_weighted_excursion": _mean(excess.square(), age),
        "late_control": _mean(u.square(), st),
    }
    difference_ratio = cfg.reference_difference_step/h
    components["smooth"] = _mean(
        ((u[1:]-u[:-1])*difference_ratio).square(), active[1:]*active[:-1]) if T > 1 else zero
    components["voltage_oscillation"] = _mean(
        ((error[2:]-2*error[1:-1]+error[:-2])*difference_ratio**2).square(),
        age[2:]*active[1:-1]*active[:-2]) if T > 2 else zero
    tail_length = max(1, min(T, round(cfg.tail_window_s/h)))
    components["tail_bias"] = _mean(error[-tail_length:].square(), active[-tail_length:])
    components["tail_excursion"] = _mean(excess[-tail_length:].square(), active[-tail_length:])
    # Historical smooth occupancy penalizes late samples near/outside .035V.
    occupancy = torch.sigmoid((error.abs()-cfg.voltage_tolerance)/cfg.smooth_tube_tau)
    components["smooth_tube_count"] = _mean(occupancy, st)
    tube_excess = torch.relu(trace["voltage_error"].abs()-cfg.tube_radius)*trace["active"]
    components["tube10_mean"] = _mean(tube_excess.square(), trace["active"])
    components["tube10_peak"] = tube_excess.amax().square()
    if set(cfg.weights) != set(components):
        raise ValueError("loss component/weight mismatch")
    total = sum(cfg.weights[name]*value for name, value in components.items())
    return total, components


# Objective


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


_PRIORITIES = {
    "peak_excursion": 4., "event_excursion": 1., "regulation": 1.,
    "terminal": .7, "settling": 1.5, "settling_excursion": 1.,
    "settling_peak": 1.5, "time_weighted_settling": 1.5,
    "time_weighted_excursion": .5, "voltage_oscillation": .1,
    "tail_bias": .5, "tail_excursion": .3, "smooth_tube_count": .2,
}


def calibrated_loss_config(trace, event_samples, base_config, priorities=None,
                           control_scale_V=2., smooth_scale_V=1.):
    """Calibrate LOSS WEIGHTS from a detached, matching physical baseline.

    A voltage component receives priority/max(baseline component, physical
    floor).  V^2 floors use (20 mV)^2, peak V floors 50 mV, the existing
    reference-step second difference uses (1 mV)^2, and occupancy uses .05.
    Effort cannot be divided by baseline effort, since applied baseline delta_u
    is zero: its fixed scales are declared in volts.  The separate 10 V excess
    penalties remain 200/500.  Physical trajectories, residuals, certificate
    norms, objective components and window definitions are never rescaled.
    """
    if not isinstance(base_config, LossConfig):
        raise TypeError("The baseline objective must use LossConfig")
    if not all(isfinite(value) and value > 0 for value in
               (control_scale_V, smooth_scale_V)):
        raise ValueError("Applied-input cost scales must be finite and positive")
    if trace.get("mode", "baseline") != "baseline" or bool((trace["u"] != 0).any()):
        raise ValueError("Loss calibration requires the actual zero-boost baseline trace")
    event_samples = tuple(event_samples)
    if any(not isinstance(k, Integral) or isinstance(k, bool) for k in event_samples):
        raise ValueError("Loss events must be integer sample boundaries")
    selected = dict(_PRIORITIES)
    if priorities is not None:
        if not set(priorities).issubset(selected):
            raise ValueError("Unknown baseline-normalized component priority")
        selected.update(priorities)
    if not all(isfinite(value) and value >= 0 for value in selected.values()):
        raise ValueError("Component priorities must be finite and nonnegative")
    with torch.no_grad():
        _, components = rich_loss(trace, event_samples=event_samples, config=base_config)
        raw = {name: float(value.detach()) for name, value in components.items()}
    if not all(isfinite(value) and value >= 0 for value in raw.values()):
        raise ValueError("Baseline objective components must be finite and nonnegative")
    floors = {name: .02**2 for name in selected}
    floors.update(peak_excursion=.05, settling_peak=.05,
                  voltage_oscillation=.001**2, smooth_tube_count=.05)
    denominators = {name: max(raw[name], floors[name]) for name in selected}
    weights = {name: float(selected[name])/denominators[name] for name in selected}
    weights.update(control=.01/control_scale_V**2,
                   late_control=.01/control_scale_V**2,
                   smooth=.01/smooth_scale_V**2,
                   tube10_mean=200., tube10_peak=500.)
    config = replace(base_config, weights=weights)
    # Audit the actual finite horizon rather than claiming that every declared
    # physical window is fully present in a shorter curriculum trajectory.
    samples = torch.arange(len(trace["u"]), device=trace["u"].device)
    event_window = torch.zeros_like(samples, dtype=torch.bool)
    settling_window = torch.zeros_like(samples, dtype=torch.bool)
    for k in event_samples:
        age = (samples-k).to(trace["u"].dtype)*float(trace["h"])
        window = (age >= 0) & (age < base_config.event_window_s)
        event_window |= window
        settling_window |= window & (age >= base_config.settling_delay_s)
    tail_samples = max(1, min(len(samples), round(base_config.tail_window_s/float(trace["h"]))))
    active = trace["active"][:-1].bool()
    record = {
        "method": "loss_weight=priority/max(detached_baseline_component,physical_floor)",
        "scope": "LOSS WEIGHTS ONLY; no state, residual, feature or certificate normalization",
        "event_samples": [int(k) for k in event_samples],
        "baseline_raw_components": raw, "priorities": dict(selected),
        "physical_floors": floors, "denominators": denominators,
        "weights": dict(weights), "control_scale_V": float(control_scale_V),
        "smooth_scale_V": float(smooth_scale_V), "loss_config": asdict(config),
        "duration_s": len(trace["u"])*float(trace["h"]),
        "h_s": float(trace["h"]), "baseline_steps": len(trace["u"]),
        "effective_window_counts": {
            "event_control_samples": int(event_window.sum()),
            "settling_control_samples": int(settling_window.sum()),
            "tail_control_samples": tail_samples,
            "terminal_observations": 1,
            "event_active_sample_nodes": int((event_window[:, None]*active).sum()),
            "settling_active_sample_nodes": int((settling_window[:, None]*active).sum()),
            "tail_active_sample_nodes": int(active[-tail_samples:].sum()),
        },
        "tail_window_covers_entire_trace": base_config.tail_window_s >= len(trace["u"])*float(trace["h"]),
        "tube_penalty_is_hard_constraint": False,
    }
    return config, record


def _events(scenario):
    return tuple(scenario.events) if scenario.events else (0,)


def _loss_config(cfg):
    return LossConfig(**cfg["loss"])


def _calibration_fixture(scenario):
    """Complete physical fixture; labels alone never identify a calibration."""
    def plant_descriptor(plant):
        return {"zip_load": bool(plant.zip_load), "h_s": float(plant.h),
                "active_nodes": list(plant.topology.active_nodes),
                "edges": [list(edge) for edge in plant.topology.edges]}
    return json_value({
        "scenario": scenario.manifest(),
        "registry": {"nodes": [asdict(node) for node in scenario.plant.registry.nodes],
                     "lines": [asdict(line) for line in scenario.plant.registry.lines]},
        "plants": {"initial": plant_descriptor(scenario.plant),
                   "events": {str(k): plant_descriptor(plant)
                              for k, plant in scenario.events.items()}},
    })


def _fixture_digest(fixture):
    return hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def calibrate_training_scenario(scenario, controller, cfg, *, include_baseline=False):
    """Detached zero-boost calibration on the exact training fixture, no cache.

    The same Scenario object is subsequently scored by the differentiable policy
    rollout. Guard calibrations are separate and remain fixed across epochs.
    No optimizer, controller state, signal scaling or random draw is involved.
    """
    fixture = _calibration_fixture(scenario)
    fingerprint = _fixture_digest(fixture)
    with torch.no_grad():
        baseline = run_scenario(scenario, controller, "baseline")
    if _fixture_digest(_calibration_fixture(scenario)) != fingerprint:
        raise RuntimeError("Loss calibration mutated its physical training fixture")
    registry = scenario.plant.registry
    steps = scenario.steps
    expected = {"u": (steps, registry.input_dim),
                "X": (steps+1, registry.state_dim),
                "voltage_error": (steps+1, registry.input_dim),
                "active": (steps+1, registry.input_dim)}
    if any(tuple(baseline[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("Baseline calibration and training scenario horizons/layouts differ")
    if float(baseline["h"]) != scenario.plant.h or scenario.plant.h != cfg["h_s"]:
        raise ValueError("Baseline calibration and training scenario sample intervals differ")
    initial = scenario.initial_state
    if 0 in scenario.events:
        from .plant import make_transition
        initial = make_transition(scenario.plant, scenario.events[0]).apply(initial)
    if not torch.equal(baseline["X"][0], initial):
        raise ValueError("Baseline calibration and training scenario initial states differ")
    plant = scenario.plant
    expected_topologies, expected_active = [], []
    for k in range(steps+1):
        plant = scenario.events.get(k, plant)
        expected_topologies.append(plant.topology.name)
        expected_active.append(registry.node_mask(plant.topology))
    if (baseline["topologies"] != expected_topologies
            or not torch.equal(baseline["active"], torch.stack(expected_active).to(baseline["active"]))):
        raise ValueError("Baseline calibration and training scenario event layouts differ")
    loss_cfg, calibration = calibrated_loss_config(
        baseline, _events(scenario), _loss_config(cfg), **cfg["loss_calibration"])
    calibration.update(calibration_scope="exact_training_scenario", fixture=fixture,
                       fixture_sha256=fingerprint)
    # New objective extensions may reuse the SAME fully validated baseline.
    # The original two-return-value interface/experiment remains unchanged.
    if include_baseline:
        return loss_cfg, calibration, baseline
    return loss_cfg, calibration


_EARLY = ("early_voltage_energy", "early_voltage_peak_squared", "early_current_energy")


_FLOORS = dict(zip(_EARLY, (.02**2, .05**2, .1**2)))


@dataclass(frozen=True)
class TransientLossConfig:
    base_loss: LossConfig
    early_window_s: float = .05
    weights: dict = field(default_factory=lambda: dict(zip(_EARLY, (4., 4., 1.5))))


def transient_components(trace, event_samples, window_s):
    """All observed states, including the terminal sample if it is in-window.

    Equilibrium-subtracted current belongs ONLY to this model-aware LOSS.
    The runtime MAD current feature is raw measured It, never this deviation.
    """
    if not isfinite(window_s) or window_s <= 0:
        raise ValueError("The early-state window must be finite and positive")
    errors, active, x = trace["voltage_error"], trace["active"], trace["x"]
    if (errors.shape != active.shape or len(errors) != len(trace["u"])+1
            or x.shape != (len(errors), 28) or errors.shape[1] != 6):
        raise ValueError("Complete canonical observed-state layouts required")
    samples = torch.arange(len(errors), device=errors.device)
    selected = torch.zeros(len(errors), dtype=torch.bool, device=errors.device)
    for event in event_samples:
        if not isinstance(event, int) or isinstance(event, bool) or event < 0:
            raise ValueError("Loss events must be integral observation boundaries")
        age = (samples-event).to(errors.dtype)*float(trace["h"])
        selected |= (age >= 0) & (age < window_s)
    mask = selected[:, None]*active
    denominator = mask.sum().clamp_min(1.)
    current_deviation = x[:, :18].reshape(-1, 6, 3)[:, :, 1]
    return {
        "early_voltage_energy": (errors.square()*mask).sum()/denominator,
        "early_voltage_peak_squared": (errors.square()*mask).amax(),
        "early_current_energy": (current_deviation.square()*mask).sum()/denominator,
    }


def transient_loss(trace, event_samples, config):
    base, components = rich_loss(trace, event_samples, config.base_loss)
    early = transient_components(trace, event_samples, config.early_window_s)
    if set(config.weights) != set(early):
        raise ValueError("Transient component/weight mismatch")
    if not all(isfinite(value) and value >= 0 for value in config.weights.values()):
        raise ValueError("Transient weights must be finite and nonnegative")
    return base+sum(config.weights[key]*value for key, value in early.items()), components|early


def calibrate_transient_scenario(scenario, controller, cfg):
    """Exact same-fixture baselines, fixed physical floors; no signal changes."""
    base, rich, baseline = calibrate_training_scenario(scenario, controller, cfg,
                                                       include_baseline=True)
    events = tuple(scenario.events) or (0,)
    settings = cfg["transient_loss"]
    with torch.no_grad():
        raw = {key: float(value) for key, value in transient_components(
            baseline, events, settings["early_window_s"]).items()}
    if not all(isfinite(value) and value >= 0 for value in raw.values()):
        raise FloatingPointError("Nonfinite transient baseline calibration")
    priorities = settings["priorities"]
    if set(priorities) != set(_EARLY) or not all(isfinite(x) and x >= 0 for x in priorities.values()):
        raise ValueError("Three finite nonnegative transient priorities required")
    denominators = {key: max(raw[key], _FLOORS[key]) for key in _EARLY}
    weights = {key: priorities[key]/denominators[key] for key in _EARLY}
    record = {"scope": "LOSS_ONLY", "fixture_sha256": rich["fixture_sha256"],
        "rich": rich, "transient": {"baseline_raw_components": raw,
        "physical_floors": dict(_FLOORS), "denominators": denominators,
        "priorities": priorities, "weights": weights,
        "early_window_s": settings["early_window_s"],
        "terminal_observation_included_if_in_window": True,
        "current_deviation_used_in_policy": False},
        "baseline_rollouts": 1, "same_fixture_and_horizon": True,
        "baseline_metrics": trajectory_metrics(baseline)}
    return TransientLossConfig(base, settings["early_window_s"], weights), record


# Fixtures


_CASES = ("fixed:pre_5_dgu", "fixed:plug_05_45",
          "fixed:plug_05_45_unplug_3", "load:plug_05_45:6",
          "plugin:plug_05_45", "unplug:plug_05_45_unplug_3")


_GROUPS = ("fixed", "load", "plugin", "unplug")


def focused_cases():
    return _CASES


def case_group(case):
    if case not in _CASES:
        raise ValueError("Unsupported focused training case")
    return case.split(":", 1)[0]


def focused_scenario(registry, case, *, h=5e-5, duration=.15, seed=0,
                     primary=None, perturbation=.2, load_step_A=1.3,
                     before_s=.005):
    """Explicit event graphs: removal3 starts from the SIX-DGU graph."""
    kind = case_group(case)
    if (not all(isfinite(x) for x in (h, duration, perturbation, before_s))
            or h <= 0 or duration <= 0 or perturbation < 0
            or not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("Invalid physical scenario interval/perturbation/seed")
    steps = round(duration/h)
    if steps < 1:
        raise ValueError("A focused trajectory needs at least one interval")
    if kind == "fixed":
        return fixed_scenario(registry, case.split(":")[1], h=h,
                              duration=duration, seed=seed, primary=primary,
                              perturbation=perturbation, zip_load=True)
    event = round(before_s/h)
    if not 0 <= before_s < duration or not 0 <= event < steps:
        raise ValueError("An event must leave at least one propagation interval")
    library, loads = topology_library(registry), experiment_loads(registry)
    initial_name = "pre_5_dgu" if kind == "plugin" else "plug_05_45"
    following_name = "plug_05_45_unplug_3" if kind == "unplug" else "plug_05_45"
    next_loads = list(loads)
    if kind == "load":
        if not isfinite(load_step_A) or load_step_A <= 0:
            raise ValueError("Focused load training uses positive current increments")
        next_loads[registry.node_ids.index(6)] += load_step_A
    initial = Plant(registry, library[initial_name], h=h, loads=loads,
                    zip_load=True, primary=primary)
    following = Plant(registry, library[following_name], h=h, loads=next_loads,
                      zip_load=True, primary=primary)
    return Scenario(f"focused_{case.replace(':', '_')}_seed{seed}", initial,
                    steps, {event: following}, initial.equilibrium(), event,
                    description=(f"training target {case}; independent equilibrium start; "
                                 "no metadata reaches policy; plugin/unplug seeds are labels, "
                                 "not independent physical excitations"))


def group_score(case_losses):
    if set(case_losses) != set(_CASES):
        raise ValueError("A guard score needs exactly all six focused cases")
    grouped = defaultdict(list)
    for case, value in case_losses.items():
        if not isfinite(float(value)):
            raise FloatingPointError("Nonfinite guard loss")
        grouped[case_group(case)].append(float(value))
    means = {group: sum(grouped[group])/len(grouped[group]) for group in _GROUPS}
    return {"score": sum(means.values())/4, "group_means": means}


def freeze_edges(controller):
    snapshot = {}
    for name in ("edge_logits", "log_edge_weights"):
        if hasattr(controller, name):
            parameter = getattr(controller, name)
            snapshot[name] = parameter.detach().clone()
            parameter.requires_grad_(False)
            parameter.grad = None
    snapshot["effective_edge_weights"] = controller.effective_edge_weights().detach().clone()
    return snapshot


def assert_edges_frozen(controller, snapshot):
    for name, expected in snapshot.items():
        actual = (controller.effective_edge_weights() if name == "effective_edge_weights"
                  else getattr(controller, name))
        if not torch.equal(actual.detach(), expected):
            raise RuntimeError("Frozen learned edge weights changed")
        if name != "effective_edge_weights" and actual.requires_grad:
            raise RuntimeError("Frozen edge parameter unexpectedly requires gradients")

def epoch_scenarios(registry, cfg, epoch):
    cases = {}
    for j, case in enumerate(focused_cases()):
        generator = random.Random(cfg["seed"]+epoch*1009+j*31)
        limits = cfg["training"]["large_load_step_range_A"] if epoch % cfg["training"]["large_load_every_epochs"] == 0 else cfg["training"]["load_step_range_A"]
        increment = generator.uniform(*limits)
        before = generator.uniform(*cfg["training"]["before_range_s"])
        cases[case] = focused_scenario(registry, case, h=cfg["h_s"],
            duration=cfg["training"]["duration_s"], seed=cfg["seed"]+epoch*100+j,
            primary=cfg["primary"], perturbation=cfg["training"]["perturbation"],
            load_step_A=increment, before_s=before)
    return cases


# Training


def peak_gate(rows, *, radius=10., baseline_rows=None,
              relative_allowance=.05, absolute_allowance=.02):
    """Every trajectory's actual active maximum, including its final sample."""
    failures = []
    for name, row in rows.items():
        peak = row["peak_V"]
        if not np.isfinite(peak) or peak > radius:
            failures.append({"scenario": name, "condition": "sampled_peak", "value": peak, "limit": radius})
        if row.get("would_saturate_sample_nodes", 0):
            failures.append({"scenario": name, "condition": "unsaturated_command_range",
                             "value": row["would_saturate_sample_nodes"]})
        if baseline_rows is not None:
            limit = (1+relative_allowance)*baseline_rows[name]["peak_V"]+absolute_allowance
            if peak > limit:
                failures.append({"scenario": name, "condition": "peak_regression", "value": peak, "limit": limit})
    if not rows:
        raise ValueError("A peak gate needs actual trajectories")
    worst = max(rows, key=lambda name: rows[name]["peak_V"])
    return {"passed": not failures, "failures": failures,
            "maximum": {"scenario": worst, **rows[worst]},
            "scope": "saved active samples; forward invariance UNVERIFIED"}


def learning_rates(optimizer, epoch, settings):
    progress = (epoch-1)/max(settings["epochs"]-1, 1)
    fraction = settings["lr_final_fraction"]
    factor = fraction+(1-fraction)*(1+cos(pi*progress))/2
    rates = {}
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]*factor
        rates[group["kind"]] = group["lr"]
    return rates


def build_guard(registry, controller, cfg):
    fixtures, baselines = {}, {}
    for j, case in enumerate(focused_cases()):
        scenario = focused_scenario(registry, case, h=cfg["h_s"], duration=cfg["guard"]["duration_s"],
            seed=cfg["guard"]["seed"]+j, primary=cfg["primary"], perturbation=cfg["guard"]["perturbation"],
            before_s=cfg["guard"]["before_s"], load_step_A=cfg["guard"]["load_step_A"])
        loss_cfg, calibration = calibrate_transient_scenario(scenario, controller, cfg)
        baselines[case] = calibration["baseline_metrics"]
        fixtures[case] = scenario, loss_cfg
        print(f"Guard baseline {case}: peak={baselines[case]['peak_V']:.6f}V", flush=True)
    gate = peak_gate(baselines, radius=cfg["rho_V"])
    if not gate["passed"]:
        raise RuntimeError("The physical guard baseline failed; no optimization allowed")
    return fixtures, baselines


def evaluate_guard(fixtures, baseline_rows, controller, cfg):
    losses, rows, component_rows = {}, {}, {}
    with torch.no_grad():
        for case, (scenario, loss_cfg) in fixtures.items():
            trace = run_scenario(scenario, controller, "mad")
            loss, components = transient_loss(trace, tuple(scenario.events) or (0,), loss_cfg)
            losses[case] = float(loss)
            rows[case] = trajectory_metrics(trace)
            component_rows[case] = {key: float(value) for key, value in components.items()}
    result = group_score(losses)
    result.update(losses=losses, rows=rows, components=component_rows,
        gate=peak_gate(rows, radius=cfg["rho_V"], baseline_rows=baseline_rows,
            relative_allowance=cfg["guard"]["peak_relative_allowance"],
            absolute_allowance=cfg["guard"]["peak_absolute_allowance_V"]),
        selection_data="fixed training guard; migrated epoch0 included; final suite unused")
    return result


def backward_group(controller, scenarios, cfg):
    cases = [case for case, _ in scenarios]
    if (not cases or len(set(cases)) != len(cases)
            or len({case_group(case) for case in cases}) != 1):
        raise ValueError("One exact mean-gradient group with unique cases required")
    rows = []
    for case, scenario in scenarios:
        start = time.monotonic()
        loss_cfg, calibration = calibrate_transient_scenario(scenario, controller, cfg)
        trace = run_scenario(scenario, controller, "mad")
        loss, components = transient_loss(trace, tuple(scenario.events) or (0,), loss_cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite physical loss; no group update made")
        (loss/len(scenarios)).backward()
        with torch.no_grad():
            metrics = trajectory_metrics(trace)
        rows.append({"case": case, "group": case_group(case), "loss": float(loss.detach()),
            "components": {key: float(value.detach()) for key, value in components.items()},
            "loss_calibration": calibration, "fixture_sha256": calibration["fixture_sha256"],
            "samples": scenario.steps, "duration_s": scenario.steps*cfg["h_s"],
            "scenario": scenario.manifest(), "event_samples": list(scenario.events),
            "sampled_peak_V": metrics["peak_V"], "U_pk_V": metrics["U_pk_V"],
            "elapsed_seconds": time.monotonic()-start})
        del trace, loss, components
    return rows


def parent_guard_gate(checked, parent, relative=.10, absolute=.02):
    result = deepcopy(checked)
    if not checked.get("gate",{}).get("passed",False): return result
    bad = {case:{"candidate":row["peak_V"],"parent":parent["rows"][case]["peak_V"]}
        for case,row in checked["rows"].items()
        if row["peak_V"] > parent["rows"][case]["peak_V"]*(1+relative)+absolute}
    result["gate"]["parent_peak_violations"] = bad
    result["gate"]["passed"] = result["gate"]["passed"] and not bad
    return result


def guard(fixtures, baselines, controller, cfg, parent):
    try:
        checked = evaluate_guard(fixtures, baselines, controller, cfg)
        return parent_guard_gate(checked,parent,cfg["guard"]["parent_relative_allowance"],
                                 cfg["guard"]["parent_absolute_allowance_V"])
    except VoltageDomainError as exc:
        return {"gate":{"passed":False},"physical_domain_error":str(exc),
                "score":None,"rows":{},"selection_data":"infeasible training guard; no fabricated metrics"}


def make_optimizer(model, settings, learn_edges=False):
    rates = {"ren": settings["ren_learning_rate"], "eta": settings["eta_learning_rate"],
        "mad_first": settings["mad_first_learning_rate"], "mad_final": settings["mad_final_learning_rate"],
        "edge": settings.get("edge_learning_rate", .03)}
    groups = []
    for name, parameter in model.named_parameters():
        edge = name in ("edge_logits", "log_edge_weights")
        if edge: parameter.requires_grad_(learn_edges)
        if not parameter.requires_grad: continue
        kind = ("edge" if edge else "ren" if name.startswith("rens.")
            else "eta" if name.startswith("eta_") else "mad_first" if ".mlp.0." in name else "mad_final")
        if not math.isfinite(rates[kind]) or rates[kind] <= 0:
            raise ValueError("Finite positive learning rate required")
        groups.append({"params": [parameter], "lr": rates[kind], "initial_lr": rates[kind],
                       "kind": kind, "name": name})
    return torch.optim.Adam(groups)


# Training entry point


#!/usr/bin/env python3


def validate_config(cfg):
    if cfg["h_s"] != 5e-5 or cfg["controller"]["gamma_R"] != 25:
        raise ValueError("The released physical configuration uses h=50 us and gamma_R=25")
    if cfg["training"]["duration_s"] != .15:
        raise ValueError("Full 150 ms BPTT is required by this training recipe")
    if cfg["training"]["epochs"] < 1:
        raise ValueError("At least one epoch is required")
    if cfg["guard"]["duration_s"] != .5 or cfg["guard"]["load_step_A"] != 1.1:
        raise ValueError("Keep the released fixed guard to compare scores")
    if cfg["runtime"]["wall_time_hours"] <= 0:
        raise ValueError("The wall-time budget must be positive")


def train(args):
    torch.set_num_threads(1)
    model, payload = load_model(args.resume or args.model)
    cfg = deepcopy(payload["config"]) if args.resume else default_parameters()
    if cfg["controller"] != payload["config"]["controller"] or cfg["primary"] != payload["config"]["primary"]:
        raise ValueError("Checkpoint architecture and primary must match the configuration")
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.budget_hours is not None:
        cfg["runtime"]["wall_time_hours"] = args.budget_hours
    cfg["training"]["freeze_edge_weights"] = not args.learn_edges
    validate_config(cfg)
    start_epoch = payload["epoch"] if args.resume else 0
    if cfg["training"]["epochs"] <= start_epoch:
        raise ValueError("For resume, --epochs is the TOTAL target, greater than the saved epoch")
    if args.resume and args.learn_edges:
        raise ValueError("Use a fresh optimizer (--model) when reopening edge optimization")
    optimizer = make_optimizer(model, cfg["training"], args.learn_edges)
    edges = None if args.learn_edges else freeze_edges(model)
    if args.resume:
        if payload["optimizer_state_dict"] is None:
            raise ValueError("Resume needs a training-state checkpoint with Adam state")
        expected = [g["name"] for g in optimizer.param_groups]
        saved = [g["name"] for g in payload["optimizer_state_dict"]["param_groups"]]
        if expected != saved:
            raise ValueError("Optimizer parameter order changed; cannot resume exactly")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["torch_rng_state"])
        random.setstate(payload["python_rng_state"])
    else:
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
    checked_certificate(model)
    output = args.output or ROOT / "models" / ("training_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    if output.exists():
        raise FileExistsError("Training requires a NEW output directory")
    output.mkdir(parents=True)
    deadline = time.monotonic() + cfg["runtime"]["wall_time_hours"]*3600
    reserve = cfg["runtime"]["finalization_reserve_s"]
    fixtures, baselines = build_guard(model.registry, model, cfg)
    initial = guard(fixtures, baselines, model, cfg, payload["guard_parent"])
    if not initial["gate"]["passed"]:
        raise RuntimeError("The initial checkpoint is infeasible on the physical guard")
    parent = payload["guard_parent"] if args.resume else initial
    if args.resume:
        best_guard = deepcopy(payload["best_guard"])
        best_model = deepcopy(payload["best_model_state_dict"])
        best_epoch = payload["best_epoch"]
        best_controller = _selected_model(model, best_model, cfg)
        replay_best = guard(fixtures, baselines, best_controller, cfg, parent)
        if not replay_best["gate"]["passed"] or abs(replay_best["score"]-best_guard["score"]) > 1e-10:
            raise RuntimeError("Saved best guard cannot be reproduced on the resume fixture")
    else:
        best_guard, best_model, best_epoch = initial, deepcopy(model.state_dict()), 0
    payload["guard_parent"] = deepcopy(parent)
    history = deepcopy(payload.get("history", [])) if args.resume else []
    checkpoint_args = dict(config=cfg, history=history, best_guard=best_guard,
                           best_model=best_model, best_epoch=best_epoch)
    start_model = _selected_model(model, best_model, cfg)
    save_checkpoint(output / "selected_controller.pt", start_model, payload, epoch=best_epoch, **checkpoint_args)
    save_checkpoint(output / "last.pt", model, payload, epoch=start_epoch, optimizer=optimizer, **checkpoint_args)
    recent, completed, attempted_updates, rejected = [], 0, 0, 0
    last_checked = initial
    interruption = False
    for epoch in range(start_epoch+1, cfg["training"]["epochs"]+1):
        forecast = max(cfg["runtime"]["minimum_epoch_forecast_s"],
                       max(recent[-4:], default=240)*cfg["runtime"]["epoch_time_safety_factor"])
        if deadline-time.monotonic() < reserve+forecast:
            break
        stamp = time.monotonic()
        rates = learning_rates(optimizer, epoch, cfg["training"])
        scenarios = epoch_scenarios(model.registry, cfg, epoch)
        order = ["fixed", "load", "plugin", "unplug"]
        random.Random(cfg["seed"]+epoch).shuffle(order)
        before_model, before_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
        rows = []
        try:
            for group in order:
                optimizer.zero_grad(set_to_none=True)
                group_rows = backward_group(model, [(case, scenarios[case]) for case in focused_cases()
                                                   if case_group(case) == group], cfg)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError("Missing or nonfinite gradients")
                norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
                optimizer.step()
                attempted_updates += 1
                if edges is not None:
                    assert_edges_frozen(model, edges)
                checked_certificate(model)
                for row in group_rows:
                    row.update(attempted_epoch=epoch, gradient_norm=norm, learning_rates=rates)
                rows.extend(group_rows)
                print(f"epoch {epoch}/{cfg['training']['epochs']} {group}: "
                      f"cost={sum(r['loss'] for r in group_rows)/len(group_rows):.6g}", flush=True)
        except VoltageDomainError as exc:
            model.load_state_dict(before_model)
            optimizer.load_state_dict(before_optimizer)
            for group in optimizer.param_groups:
                group["initial_lr"] *= .5
                group["lr"] *= .5
            rejected += 1
            continue
        except KeyboardInterrupt:
            model.load_state_dict(before_model)
            optimizer.load_state_dict(before_optimizer)
            interruption = True
            break
        if len(rows) != 6:
            raise RuntimeError("A complete epoch must contain six rollouts and four Adam updates")
        completed += 1
        check_now = (not history or (len(history)+1)%cfg["guard"]["every_epochs"] == 0
                     or epoch == cfg["training"]["epochs"])
        last_checked = guard(fixtures, baselines, model, cfg, parent) if check_now else None
        if last_checked is not None:
            if last_checked["gate"]["passed"] and last_checked["score"] < best_guard["score"]:
                best_guard, best_model, best_epoch = last_checked, deepcopy(model.state_dict()), epoch
                save_checkpoint(output / "selected_controller.pt", model, payload, config=cfg,
                    epoch=epoch, best_guard=best_guard, best_model=best_model, best_epoch=best_epoch)
            print(f"guard={last_checked['score']}, best={best_guard['score']:.7g} at {best_epoch}", flush=True)
        seconds = time.monotonic()-stamp
        recent.append(seconds)
        history.append({"attempted_epoch": epoch, "complete_epoch_index": len(history)+1,
            "updates": (len(history)+1)*4, "differentiable_rollouts": (len(history)+1)*6,
            "guard_score": last_checked["score"] if last_checked else None,
            "learning_rates": rates, "elapsed_seconds": seconds})
        save_checkpoint(output / "last.pt", model, payload, config=cfg, epoch=epoch, optimizer=optimizer,
                        history=history, best_guard=best_guard, best_model=best_model, best_epoch=best_epoch)
    # Always check the last COMPLETE epoch before returning; no final-suite selection.
    if completed and last_checked is None:
        last_checked = guard(fixtures, baselines, model, cfg, parent)
        epoch = history[-1]["attempted_epoch"]
        if last_checked["gate"]["passed"] and last_checked["score"] < best_guard["score"]:
            best_guard, best_model, best_epoch = last_checked, deepcopy(model.state_dict()), epoch
            save_checkpoint(output / "selected_controller.pt", model, payload, config=cfg,
                epoch=epoch, best_guard=best_guard, best_model=best_model, best_epoch=best_epoch)
        history[-1]["guard_score"] = last_checked["score"]
        save_checkpoint(output / "last.pt", model, payload, config=cfg, epoch=epoch, optimizer=optimizer,
                        history=history, best_guard=best_guard, best_model=best_model, best_epoch=best_epoch)
    result = {"completed_new_epochs": completed, "accepted_new_updates": completed*4,
        "completed_new_differentiable_rollouts": completed*6, "executed_updates_including_rollbacks": attempted_updates,
        "rejected_epochs": rejected, "interrupted": interruption, "best_epoch": best_epoch,
        "initial_guard": initial, "best_guard": best_guard, "last_complete_guard": last_checked,
        "final_suite_used_for_selection": False, "full_BPTT": True,
        "certificate_selected": checked_certificate(_selected_model(model, best_model, cfg)),
        "output": str(output.resolve())}
    print(f"Completed {completed} new epochs; selected epoch {best_epoch}. Checkpoints: {output.resolve()}", flush=True)
    # Store the final report inside the two checkpoints, with no sidecar files.
    payload["training_summary"] = result
    final_epoch = history[-1]["attempted_epoch"] if completed else start_epoch
    save_checkpoint(output / "last.pt", model, payload, config=cfg, epoch=final_epoch,
                    optimizer=optimizer, history=history, best_guard=best_guard,
                    best_model=best_model, best_epoch=best_epoch)
    save_checkpoint(output / "selected_controller.pt", _selected_model(model, best_model, cfg),
                    payload, config=cfg, epoch=best_epoch, history=history,
                    best_guard=best_guard, best_model=best_model, best_epoch=best_epoch)
    return result


def _selected_model(model, state, cfg):
    # The registry contains immutable mapping proxies, so reconstruct the module.
    # This copy must not consume the RNG state used for exact continuation.
    with torch.random.fork_rng(devices=[]):
        selected = CurrentAwareController(model.registry, **cfg["controller"])
        selected.load_state_dict(state, strict=True)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Start a new optimizer from these weights")
    parser.add_argument("--resume", type=Path, help="Continue an exact complete-epoch Adam/RNG state")
    parser.add_argument("--epochs", type=int, help="Total target epoch count (resume begins after its saved epoch)")
    parser.add_argument("--budget-hours", type=float, help="Cooperative total budget, stopping between complete epochs")
    parser.add_argument("--learn-edges", action="store_true", help="Offline optimization of ten bounded independent weights")
    parser.add_argument("--output", type=Path, help="New output directory")
    return train(parser.parse_args(argv))
