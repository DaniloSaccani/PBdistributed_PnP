# Distributed performance boosting for PnP DC microgrids

Trained distributed REN controller with an external bounded MAD readout.
This repository contains the selected weights, the three final voltage figures,
and the code to replay the physical experiments or continue training.

## Setup

The reference environment is Python 3.13 on CPU. From this directory:

```
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.4.4 torch==2.11.0 matplotlib==3.11.2
```

All controller and simulation calculations use float64.

## Generate the figures

```
python main_PnP_validation.py
```

The script loads `models/controller.pt`, verifies the realized REN/network
conditions, and simulates matching baseline/PB trajectories. It writes only the
three PDF/PNG voltage figures to `figure/`, replacing those named figures.
Figures have zooms, no title or legend, and use raw samples without smoothing
or decimation.

The case studies are plug-in of DGU 6, a 1.3 A load increase after the plug-in,
and unplugging of DGU 3. They are independent experiments, each starting at
equilibrium 25 ms before the event and ending 500 ms after it. Display times
4/8/12 s are offsets for those separate experiments.

Optional:

```
python main_PnP_validation.py --legend
python main_PnP_validation.py --controls
python main_PnP_validation.py --model models/my_run/selected_controller.pt
```

## Continue training

Start a new Adam optimizer from the selected weights:

```
python main_PnP_training.py --epochs 24 --budget-hours 2.5
```

Resume the bundled last complete epoch, including Adam moments and RNG state:

```
python main_PnP_training.py --resume models/training_state.pt --epochs 152 --budget-hours 2.5
```

For resume, `--epochs` is the total target: the bundled state is epoch 146,
with its best checkpoint at epoch 144. Keeping target 152 preserves its cosine
schedule. This is continuation of an already trained model, not a replay of
the historical adaptive search from random initialization.

Each run writes only `last.pt` and `selected_controller.pt` inside a new
`models/training_<UTC>/` folder. Checkpoints include parameters, training
history, selection scores and, for `last.pt`, optimizer/RNG state.
The time budget is checked between complete epochs.

Parameters for a new optimizer are in `src/settings.py`. Validation and
resume use the parameters stored in their checkpoint. The published recipe
uses sampling 50 us, gamma_R=25, REN6/6, MAD4->8->1 and full 150 ms BPTT.
Each epoch has six differentiable rollouts and four Adam updates, with
baseline calibration on the same fixture. Selection uses the fixed six-case,
500 ms guard with four equally weighted groups. Edge weights stay frozen
unless `--learn-edges` is explicitly requested with a new optimizer.

## Source files

| File | Contents |
| --- | --- |
| `src/plant.py` | DGU/line registry, primary gains, ZIP plant and physical event maps |
| `src/controller.py` | REN realization, interconnection/gain checks, MAD and PnP updates |
| `src/simulation.py` | Scenarios, causal IMC simulation and voltage metrics |
| `src/training.py` | Loss, baseline calibration, fixtures, guard and Adam loop |
| `src/settings.py` | Default numerical parameters for new training |
| `src/model_io.py` | Model loading and checkpoint saving |
| `src/plotting.py` | Raw voltage/control figures and zoom layout |

Physical/controller equations and selected tensor values are unchanged.
The target event families were trained. Numerical matrix checks and sampled
voltage bounds do not establish forward invariance or arbitrary switching.
