import os  # nopep8
import sys  # nopep8
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8

import loralib_gptj as lora
from utils import *


import argparse

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--r', metavar='R', type=int, required=True, help='low rank r')
args = parser.parse_args()
# print(args)


PATH = "test/adapters"
ADAPTER_PATH = f"{PATH}/gptj.pt"
R = args.r


if __name__ == "__main__":
    """Load Default Model"""

    import torch
    import transformers
    from transformers import GPTJForCausalLM

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float32",
        torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device='cpu', non_blocking=True)
    # _ = model.eval()  # by default
    # print("Model Loaded.")

    # print("\nDefault Model")
    # summary(model)

    """LoRA-applied Model"""

    class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
        def __init__(self, config):
            super().__init__(config)
            lora.convert_to_lora(self.attn)
            lora.convert_to_lora(self.mlp)

    class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
        def __init__(self, config):
            super().__init__(config)
            lora.convert_to_lora(self)

    class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            lora.convert_to_lora(self)

    # monkey-patch GPT-J
    transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float32",
        torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device='cpu', non_blocking=True)
    # _ = model.eval()  # by default
    # print("Model Loaded.")

    # print("\nDefault Model")
    # summary(model)

    # skip gradient calculation of original model
    _ = model.eval()
    for name, param in model.named_parameters():
        # print(f"Setting {name} requires_grad=False")
        param.requires_grad = False

    lora.add_adapters(model, adapter_dim=R)
    model.to(device='cpu', non_blocking=True)

    # print(model)
    # print("\nLoRA-applied Model")
    # summary(model)

    torch.save(lora.get_adapters(model), ADAPTER_PATH)
    # print(f"\nAdapters Saved:\t\t\t{os.path.getsize(ADAPTER_PATH)}")

    # save
    model_ntp = 6050882784
    model_bs = 24207819307
    lora_ntp = count_trainable_parameters(model)
    lora_bs = os.path.getsize(ADAPTER_PATH)
    print(
        f"{model_ntp}, {model_bs}, {lora_ntp}, {lora_bs}, {(1-lora_ntp/model_ntp)*100}, {(1-lora_bs/model_bs)*100}"
    )

"""
Default Model
- # of           params:        6050882784
- # of trainable params:        6050882784
- # of          buffers:        4194305
Model Loaded.

LoRA-applied Model
- # of           params:        9554912
- # of trainable params:        8693504
- # of          buffers:        6054215681

Adapters Saved:                 34954201
"""
