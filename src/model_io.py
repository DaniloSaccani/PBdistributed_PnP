"""Load exact model tensors and save complete Adam/RNG checkpoints."""

import random
import torch
from copy import deepcopy
from pathlib import Path
from .controller import CurrentAwareController
from .plant import build_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/controller.pt"


def load_model(path=DEFAULT_MODEL):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "dppb-portable-v1":
        raise ValueError("Expected a portable dppb checkpoint")
    config = deepcopy(payload["config"])
    registry = build_registry()
    descriptor = payload["controller_descriptor"]
    if descriptor["permanent_edge_order"] != [list(e) for e in registry.edges]:
        raise ValueError("Checkpoint line order differs from the physical registry")
    model = CurrentAwareController(registry, **config["controller"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for node, ren in model.rens.items():
        actual = (ren.input_dim, ren.state_dim, ren.width)
        saved = descriptor["rens"][node]
        if actual != (saved["input_dim"], saved["state_dim"], saved["width"]):
            raise ValueError("Controller dimensions and checkpoint descriptor disagree")
    return model, payload


def save_checkpoint(path, model, template, *, config, epoch, optimizer=None,
                    history=(), best_guard=None, best_model=None, best_epoch=0):
    payload = dict(template)
    payload.update(format="dppb-portable-v1", kind="training_state" if optimizer else "selected",
        config=deepcopy(config), epoch=epoch, stage="focused_physical_training",
        selection=("last complete epoch with Adam/RNG" if optimizer else
                   "best feasible fixed training guard; final suite unused"),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer else None,
        torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate(),
        history=list(history), best_guard=best_guard,
        best_model_state_dict=best_model, best_epoch=best_epoch)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
