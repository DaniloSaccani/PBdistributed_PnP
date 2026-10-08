#!/usr/bin/env python3
"""Replay baseline/PB and regenerate the three final paper voltage figures."""
import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path
import torch

from src.controller import checked_certificate
from src.model_io import ROOT, DEFAULT_MODEL, load_model
from src.plotting import VIEWS, render_case
from src.simulation import representative_scenarios, run_scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "figure")
    parser.add_argument("--legend", action="store_true", help="Include the top legend")
    parser.add_argument("--controls", action="store_true", help="Also plot the boosting actions")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    model, payload = load_model(args.model)
    checked_certificate(model)
    cfg = payload["config"]
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = representative_scenarios(model.registry, h=cfg["h_s"],
        before=cfg["representative"]["before_s"], after=cfg["representative"]["after_s"],
        primary=cfg["primary"])
    for scenario in scenarios:
        if scenario.name not in VIEWS:
            continue
        print(f"Simulating {scenario.name}...", flush=True)
        with torch.no_grad():
            baseline = run_scenario(scenario, model, "baseline")
            pb = run_scenario(scenario, model, "mad")
        paths = render_case(scenario, baseline, pb, model.registry, args.output,
                            controls=args.controls, legend=args.legend)
        for path in paths:
            print(path.resolve(), flush=True)
    return args.output


if __name__ == "__main__":
    main()
