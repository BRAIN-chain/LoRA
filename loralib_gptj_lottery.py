from __future__ import annotations

import copy

import transformers

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch.nn.utils.prune as prune

from tqdm.auto import tqdm


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
                            dtype=torch.float16
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
                            dtype=torch.float16
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


def get_sparsity_info(model) -> dict:
    sparsity_info = dict()
    linears, embeddings = 0, 0

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            # print("Linear", module.adapter)
            adapter = copy.deepcopy(module.adapter)
            _ = adapter.eval()

            total_sum_weights = 0
            total_numel = 0
            for module in adapter.children():
                weight = module.weight.detach()
                sparsity = torch.sum(weight == 0) / weight.numel()
                # print(f'Sparsity in Linear{linears}: {sparsity.item():.2%}')
                total_sum_weights += torch.sum(weight == 0)
                total_numel += weight.numel()

            sparsity_info[f"Linear{linears}"] = (total_sum_weights / total_numel).item()
            linears += 1

        elif isinstance(module, FrozenEmbedding):
            # print("Embedding", module.adapter)
            adapter = copy.deepcopy(module.adapter)
            _ = adapter.eval()

            total_sum_weights = 0
            total_numel = 0
            for module in adapter.children():
                weight = module.weight.detach()
                sparsity = torch.sum(weight == 0) / weight.numel()
                # print(f'Sparsity in Embedding{embeddings}: {sparsity.item():.2%}')
                total_sum_weights += torch.sum(weight == 0)
                total_numel += weight.numel()

            sparsity_info[f"Embedding{embeddings}"] = (total_sum_weights / total_numel).item()
            embeddings += 1

    return sparsity_info


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


def add_adapters(
    model, adapter_dim=4,
    train_dataset=None, val_dataset=None,
    tokenizer=None, 
    j=2, p=0.5,
    device='cuda',
    **kwargs
):
    assert adapter_dim > 0
    assert train_dataset != None
    assert val_dataset != None
    assert tokenizer != None

    for module in model.modules():
        if isinstance(module, FrozenLinear):
            module.adapter = nn.Sequential(
                nn.Linear(
                    module.in_features, adapter_dim, bias=False,
                    dtype=torch.float16
                ),
                nn.Linear(
                    adapter_dim, module.out_features, bias=False,
                    dtype=torch.float16
                ),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(
                    module.num_embeddings, adapter_dim,
                    dtype=torch.float16
                ),
                nn.Linear(
                    adapter_dim, module.embedding_dim, bias=False,
                    dtype=torch.float16
                ),
            )
            nn.init.zeros_(module.adapter[1].weight)

    """Lottery Ticket Hypothesis"""

    # [0] Initialize network with theta_0
    theta_0 = copy.deepcopy(get_adapters(model))

    # [1] Train network and get theta_j
    # Iterate j
    model.train()
    model.to(device, non_blocking=True)

    for epoch in range(1, j+1):
        # model.gradient_checkpointing_enable()

        # use the parameters from Appendix D.4, Table 11,12 and 15 at https://arxiv.org/pdf/2106.09685.pdf
        # adjust eps for FP16 (1e-8 => 1e-4)
        optimizer = optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=0.1, eps=1e-4
        )

        with torch.cuda.amp.autocast():
            for row in (pbar := tqdm(train_dataset)):
                if len(row["dialogue"]) <= 1:
                    continue

                batch = tokenizer(
                    row["dialogue"], truncation=True, max_length=2048, return_tensors='pt'
                )
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                out = model.forward(**batch,)
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].flatten(0, -2),
                    batch['input_ids'][:, 1:].flatten(),
                    reduction='mean'
                )
                pbar.set_description(f"loss {loss:.4f}")  # TODO: disable
                loss.backward()
                optimizer.step()

        # Print the statistics of the epoch
        # TODO: train_loss, val_loss, val_accuracy
        print('Completed training batch', epoch)

    # [2] Pruning p% of theta_j
    theta_j = copy.deepcopy(get_adapters(model))

    # prune.random_unstructured(module, name="weight", amount=0.3)
    prune_j = copy.deepcopy(theta_j)
    for name, adapter in prune_j.items():
        for module in adapter.children():
            # print(module)
            
            # Perform the pruning on a GPU, where topk does support torch.float16 tensors.

            # 1) Random
            prune.random_unstructured(module, name='weight', amount=p)

            # # 2) Norm
            # prune.ln_structured(
            #     module, name='weight', amount=p,
            #     n=2,
            #     dim=0  # TODO
            # )

            # 3) Global
            # TODO

            # TODO: permanently remove pruned parameters
            prune.remove(module, 'weight')

    # print(theta_0['Linear168'][1].weight)  # requires_grad  # grad_fn
    # print(theta_j['Linear168'][1].weight)  # requires_grad  # grad_fn
    # print(prune_j['Linear168'][1].weight)  # requires_grad  # grad_fn

    # [3] Initialize network with theta_0
    # set mask into module.adapter
    # print(get_adapters(model)['Linear168'][1].weight)  # requires_grad  # grad_fn
    set_adapters(model, prune_j)
    # print(get_adapters(model)['Linear168'][1].weight)  # requires_grad  # grad_fn
    # print(get_adapters(model)['Linear168'][1].weight.requires_grad)  # requires_grad  # grad_fn
    # print(get_adapters(model)['Linear168'][1].weight.grad_fn)  # requires_grad  # grad_fn

    model.to('cpu')
    model.eval()
