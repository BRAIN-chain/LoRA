from __future__ import annotations

import copy

import transformers
from transformers.pytorch_utils import Conv1D 

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch.nn.utils.prune as prune

from tqdm.auto import tqdm


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


def get_sparsity_info(model) -> dict:
    sparsity_info = dict()
    conv1ds = 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            # print("Conv1D", module.adapter)
            total_sum_weights = 0
            total_numel = 0
            for adapter_module in module.adapter.children():
                weight = adapter_module.weight_mask.detach()
                sparsity = torch.sum(weight == 0) / weight.numel()
                # print(f'Sparsity in Conv1D{conv1ds}: {sparsity.item():.2%}')
                total_sum_weights += torch.sum(weight == 0)
                total_numel += weight.numel()

            sparsity_info[f"Conv1D{conv1ds}"] = (total_sum_weights / total_numel).item()
            conv1ds += 1

    return sparsity_info


def set_adapters(model, adapters):
    conv1ds = 0

    for module in model.modules():
        if isinstance(module, FrozenConv1D):
            # print("Conv1D", module.adapter)
            module.adapter = adapters[f"Conv1D{conv1ds}"]
            conv1ds += 1

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

    """Lottery Ticket Hypothesis"""

    # [0] Initialize network with theta_0
    theta_0 = copy.deepcopy(get_adapters(model))

    # [1] Train network and get theta_j
    model.train()
    model.to(device, non_blocking=True)

    for epoch in range(1, j+1):
        # model.gradient_checkpointing_enable()

        # use the parameters from Appendix D.4, Table 11,12 and 15 at https://arxiv.org/pdf/2106.09685.pdf
        # adjust eps for FP16 (1e-8 => 1e-4)
        optimizer = optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=0.01, eps=1e-4
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

            # # TODO: permanently remove pruned parameters
            # prune.remove(module, 'weight')

    # [3] Initialize network with theta_0
    # set mask into module.adapter
    # print(get_adapters(model)['Linear168'][1].weight)  # requires_grad  # grad_fn
    prune_theta_0 = theta_0
    for t0, pj in zip(prune_theta_0.items(), prune_j.items()):
        (t0_name, t0_adapter) = t0
        (tj_name, tj_adapter) = pj
        for t0_module, pj_module in zip(t0_adapter.children(), tj_adapter.children()):
            # print(next(t0_module.parameters()).device)
            # print(next(pj_module.parameters()).device)
            pj_module.to('cpu')
            mask = pj_module.weight_mask
            prune.custom_from_mask(t0_module, name="weight", mask=mask)

    set_adapters(model, prune_theta_0)

    model.to('cpu')
    model.eval()
