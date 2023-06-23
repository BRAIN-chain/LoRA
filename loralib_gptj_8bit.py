from __future__ import annotations

import copy

import transformers

import torch
import torch.nn.functional as F
from torch import nn
import torch.quantization


"""Frozen Layers"""


class FrozenLinear(nn.Module):
    def __init__(self, weight, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        # with torch.no_grad():
        output = F.linear(input, self.weight, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FrozenLinear:
        return cls(linear.weight, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class FrozenEmbedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            output = F.embedding(input, self.weight, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> FrozenEmbedding:
        return cls(embedding.weight)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"


def convert_to_lora(model):
    # Convert linear and embedding modules with optional adapters

    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    FrozenLinear(
                        weight=torch.zeros(
                            child.out_features, child.in_features,
                            dtype=torch.float32
                        ),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenEmbedding(
                        weight=torch.zeros(
                            child.num_embeddings, child.embedding_dim,
                            dtype=torch.float32
                        ),
                    )
                )


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
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Get", "Linear", module.adapter)
            adapters[f"Linear{linears}"] = module.adapter
            linears += 1
        elif isinstance(module, FrozenEmbedding):
            # print("Get", "Embedding", module.adapter)
            adapters[f"Embedding{embeddings}"] = module.adapter
            embeddings += 1

    return adapters


def get_quantized_adapters(model) -> dict:
    quantized_adapters = dict()
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Get", "Linear", module.adapter)
            adapter = copy.deepcopy(module.adapter)
            _ = adapter.eval()
            adapter.qconfig = torch.quantization.get_default_qat_qconfig("x86")
            model_prepared = torch.quantization.prepare_qat(adapter.train())  # warning?
            _ = model_prepared.eval()
            quantized_model = torch.quantization.convert(model_prepared, inplace=False)

            quantized_adapters[f"Linear{linears}"] = quantized_model
            linears += 1
            # print(quantized_model.model_fp[1]._weight_bias())
        elif isinstance(module, FrozenEmbedding):
            # print("Get", "Embedding", module.adapter)
            adapter = copy.deepcopy(module.adapter)
            _ = adapter.eval()
            adapter.qconfig = torch.quantization.get_default_qat_qconfig("x86")

            qconfig_embedding = torch.quantization.float_qparams_weight_only_qconfig
            # set linear and embedding modules' config
            for adapter_module in list(adapter.modules()):
                for name, child in adapter_module.named_children():
                    if isinstance(child, nn.Embedding):
                        child.qconfig = qconfig_embedding
                        # print("qconfig_embedding")

            model_prepared = torch.quantization.prepare_qat(adapter.train())  # warning?
            _ = model_prepared.eval()
            quantized_model = torch.quantization.convert(model_prepared, inplace=False)

            quantized_adapters[f"Embedding{embeddings}"] = module.adapter
            embeddings += 1
            # print(quantized_model.model_fp[1]._weight_bias())

    return quantized_adapters


def set_adapters(model, adapters):
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Set", "Linear", module.adapter)
            module.adapter = adapters[f"Linear{linears}"]
            linears += 1
        elif isinstance(module, FrozenEmbedding):
            # print("Set", "Embedding", module.adapter)
            module.adapter = adapters[f"Embedding{embeddings}"]
            embeddings += 1

    return adapters


def add_adapters(model, adapter_dim=4):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            float_adapter = nn.Sequential(
                nn.Linear(
                    module.in_features, adapter_dim, bias=False,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.out_features, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(float_adapter[1].weight)
            # module.adapter = copy.deepcopy(float_adapter)

            # 1) Post Training Dynamic Quantization
            # module.adapter = torch.quantization.quantized_dynamic(
            #     float_adapter, {nn.Linear}, dtype=torch.qint8
            # )

            # 2) QAT
            qat_model = QuantizedModel(float_adapter)
            # # qconfig_linear = torch.quantization.get_default_qat_qconfig("fbgemm")
            # qconfig_linear = torch.quantization.get_default_qat_qconfig("x86")
            # # set linear and embedding modules' config
            # for module in list(qat_model.modules()):
            #     for name, child in module.named_children():
            #         if isinstance(child, nn.Linear):
            #             child.qconfig = qconfig_linear
            #             # print("qconfig_linear")
            #         # elif isinstance(child, nn.Embedding):
            #             # child.qconfig = qconfig_embedding
            #             # print("qconfig_embedding")            
            # # model_prepared = torch.quantization.prepare_qat(qat_model.train())  # warning?
            # # _ = model_prepared.eval()
            # # quantized_model = torch.quantization.convert(model_prepared, inplace=False)
            # # _ = quantized_model.eval()
            module.adapter = copy.deepcopy(qat_model)

        elif isinstance(module, FrozenEmbedding):
            float_adapter = nn.Sequential(
                nn.Embedding(
                    module.num_embeddings, adapter_dim,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.embedding_dim, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(float_adapter[1].weight)
            # module.adapter = copy.deepcopy(float_adapter)

            # 1) Post Training Dynamic Quantization
            # module.adapter = torch.quantization.quantized_dynamic(
            #     float_adapter, {nn.Embedding, nn.Linear}, dtype=torch.qint8
            # )

            # 2) QAT
            qat_model = QuantizedModel(float_adapter)
            # # qconfig_linear = torch.quantization.get_default_qat_qconfig("fbgemm")
            # qconfig_linear = torch.quantization.get_default_qat_qconfig("x86")
            # # set linear and embedding modules' config
            # for module in list(qat_model.modules()):
            #     for name, child in module.named_children():
            #         if isinstance(child, nn.Linear):
            #             child.qconfig = qconfig_linear
            #             # print("qconfig_linear")
            #         # elif isinstance(child, nn.Embedding):
            #             # child.qconfig = qconfig_embedding
            #             # print("qconfig_embedding")            
            # # model_prepared = torch.quantization.prepare_qat(qat_model.train())  # warning?
            # # _ = model_prepared.eval()
            # # quantized_model = torch.quantization.convert(model_prepared, inplace=False)
            # # _ = quantized_model.eval()
            module.adapter = copy.deepcopy(qat_model)
