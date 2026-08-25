"""Safetensors weight loader."""
import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str) -> None:
    """Load all *.safetensors files in path into model parameters."""
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = model.get_parameter(name)
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, f.get_tensor(name))
