from __future__ import annotations

import transformers
from transformers.pytorch_utils import Conv1D 

import torch
import torch.nn.functional as F
from torch import nn


"""Frozen Layers"""


class FrozenConv1D(nn.Module):
    def __init__(self, weight, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.nx, self.nf = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.adapter = None
        self.bias = bias
        
    def forward(self, x):
        x1 = x.clone()
        # with torch.no_grad():
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        output = x.view(*size_out)        
        if self.adapter:
            output += self.adapter(x1)
        return output

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D) -> FrozenConv1D:
        return cls(conv1d.weight, conv1d.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.nf})"


def convert_to_lora(module):
    return FrozenConv1D.from_conv1d(module)


"""Apply Adapters"""


def get_adapters(model) -> dict:
    adapters = dict()
    conv1ds = 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            # print("Conv1D", module.adapter)
            adapters[f"Conv1D{conv1ds}"] = module.adapter
            conv1ds += 1

    return adapters


def set_adapters(model, adapters):
    conv1ds = 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            # print("Conv1D", module.adapter)
            module.adapter = adapters[f"Conv1D{conv1ds}"]
            conv1ds += 1

    return adapters


def add_adapters(model, adapter_dim=4):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            module.adapter = nn.Sequential(
                nn.Linear(
                    module.nx, adapter_dim, bias=False,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.nf, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(module.adapter[1].weight)
