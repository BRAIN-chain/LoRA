from __future__ import annotations

import transformers

import torch
import torch.nn.functional as F
from torch import nn


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


"""Apply Adapters"""


def get_adapters(model) -> dict:
    adapters = dict()
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Linear", module.adapter)
            adapters[f"Linear{linears}"] = module.adapter
            linears += 1
        elif isinstance(module, FrozenEmbedding):
            # print("Embedding", module.adapter)
            adapters[f"Embedding{embeddings}"] = module.adapter
            embeddings += 1

    return adapters


def set_adapters(model, adapters):
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Linear", module.adapter)
            module.adapter = adapters[f"Linear{linears}"]
            linears += 1
        elif isinstance(module, FrozenEmbedding):
            # print("Embedding", module.adapter)
            module.adapter = adapters[f"Embedding{embeddings}"]
            embeddings += 1

    return adapters


def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            module.adapter = nn.Sequential(
                nn.Linear(
                    module.in_features, adapter_dim, bias=False,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.out_features, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(
                    module.num_embeddings, adapter_dim,
                    dtype=torch.float32
                ),
                nn.Linear(
                    adapter_dim, module.embedding_dim, bias=False,
                    dtype=torch.float32
                ),
            )
            nn.init.zeros_(module.adapter[1].weight)


"""Utils"""


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_buffers(model):
    return sum(p.numel() for p in model.buffers())


def summary(model):
    print(f"- # of           params:\t{count_parameters(model)}")
    print(f"- # of trainable params:\t{count_trainable_parameters(model)}")
    print(f"- # of          buffers:\t{count_buffers(model)}")
