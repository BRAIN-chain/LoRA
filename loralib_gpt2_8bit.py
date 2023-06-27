from __future__ import annotations

import copy

import transformers
from transformers.pytorch_utils import Conv1D 

import torch
import torch.nn.functional as F
from torch import nn
import torch.quantization


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
        # with torch.no_grad():
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        output = x.view(size_out)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D) -> FrozenConv1D:
        return cls(conv1d.weight, conv1d.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.nf})"


def convert_to_lora(module):
    return FrozenConv1D.from_conv1d(module)


"""Quantization"""


class QuantizedModel(nn.Module):
    def __init__(self, model_fp):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp = model_fp
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp(x)
        x = self.dequant(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_fp})"


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


def get_quantized_adapters(model) -> dict:
    quantized_adapters = dict()
    conv1ds = 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            # print("Conv1D", module.adapter)
            adapter = copy.deepcopy(module.adapter)
            _ = adapter.eval()
            adapter.qconfig = torch.quantization.get_default_qat_qconfig("x86")
            model_prepared = torch.quantization.prepare_qat(adapter.train())  # warning?
            _ = model_prepared.eval()
            quantized_model = torch.quantization.convert(model_prepared, inplace=False)

            quantized_adapters[f"Conv1D{conv1ds}"] = quantized_model
            conv1ds += 1
            # print(quantized_model.model_fp[1]._weight_bias())

    return quantized_adapters


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
            float_adapter = nn.Sequential(
                nn.Linear(
                    module.nx, adapter_dim, bias=False,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.nf, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(float_adapter[1].weight)

            qat_model = QuantizedModel(float_adapter)
            module.adapter = copy.deepcopy(qat_model)
