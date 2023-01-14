import os  # nopep8
import sys  # nopep8
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8

import loralib_gpt2 as lora
from utils import *


import argparse

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--r', metavar='R', type=int, required=True, help='low rank r')
parser.add_argument('--size', type=str, required=True, help='gpt2, gpt2-medium, gpt2-large, gpt2-xl')
args = parser.parse_args()
# print(args)


PATH = "test/adapters"
NAME_SIZE = args.size # "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
ADAPTER_PATH = f"{PATH}/{NAME_SIZE}.pt"
R = args.r


if args.size == "gpt2":
    MODEL_NTP = 124439808
    MODEL_BS = 548118077
elif args.size == "gpt2-medium":
    MODEL_NTP = 354823168
    MODEL_BS = 1520013706
elif args.size == "gpt2-large":
    MODEL_NTP = 774030080
    MODEL_BS = 3247202234
elif args.size == "gpt2-xl":
    MODEL_NTP = 1557611200
    MODEL_BS = 6431878936

    
if __name__ == "__main__":
    """Load Default Model"""

    import torch
    import transformers
    from transformers import GPT2Model

    config = transformers.GPT2Config.from_pretrained(NAME_SIZE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(NAME_SIZE)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = GPT2Model.from_pretrained(NAME_SIZE).to(device='cpu', non_blocking=True)
    # _ = model.eval()  # by default
    # print("Model Loaded.")

    # print("\nDefault Model")
    # summary(model)

    """LoRA-applied Model"""
    
    class GPT2Attention(transformers.models.gpt2.modeling_gpt2.GPT2Attention):
        def __init__(self, config, is_cross_attention=False, layer_idx=None):
            super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
            self.c_attn = lora.convert_to_lora(self.c_attn)

    class GPT2Block(transformers.models.gpt2.modeling_gpt2.GPT2Block):
        def __init__(self, config, layer_idx=None):
            super().__init__(config, layer_idx=layer_idx)
            # lora.convert_to_lora(self.attn)
            # lora.convert_to_lora(self.mlp)

    class GPT2Model(transformers.models.gpt2.modeling_gpt2.GPT2Model):
        def __init__(self, config):
            super().__init__(config)
            # lora.convert_to_lora(self)

    # monkey-patch GPT-2
    transformers.models.gpt2.modeling_gpt2.GPT2Attention = GPT2Attention
    transformers.models.gpt2.modeling_gpt2.GPT2Block = GPT2Block

    config = transformers.GPT2Config.from_pretrained(NAME_SIZE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(NAME_SIZE)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = GPT2Model.from_pretrained(NAME_SIZE).to(device='cpu', non_blocking=True)
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
    model_ntp = MODEL_NTP
    model_bs = MODEL_BS
    lora_ntp = count_trainable_parameters(model)
    lora_bs = os.path.getsize(ADAPTER_PATH)
    print(
        f"{model_ntp}, {model_bs}, {lora_ntp}, {lora_bs}, {(1-lora_ntp/model_ntp)*100}, {(1-lora_bs/model_bs)*100}"
    )

"""
Default Model
- # of           params:        1557611200
- # of trainable params:        1557611200
- # of          buffers:        50331696
Model Loaded.

LoRA-applied Model
- # of           params:        1190200000
- # of trainable params:        1228800
- # of          buffers:        418971696

Adapters Saved:                 4965573
"""
