import os  # nopep8
import sys  # nopep8
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8

from lora import *


PATH = "test/adapters"
ADAPTER_PATH = f"{PATH}/gptj.pt"


if __name__ == "__main__":
    """Load Default Model"""

    from transformers import GPTJForCausalLM

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float32",
        torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device='cpu', non_blocking=True)
    # _ = model.eval()  # by default
    print("Model Loaded.")

    print("\nDefault Model")
    summary(model)

    """LoRA-applied Model"""

    class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
        def __init__(self, config):
            super().__init__(config)
            convert_to_lora(self.attn)
            convert_to_lora(self.mlp)


    class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
        def __init__(self, config):
            super().__init__(config)
            convert_to_lora(self)


    class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            convert_to_lora(self)


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
    print("Model Loaded.")

    # print("\nDefault Model")
    # summary(model)

    # skip gradient calculation of original model
    _ = model.eval()
    for name, param in model.named_parameters():
        # print(f"Setting {name} requires_grad=False")
        param.requires_grad = False

    add_adapters(model)
    model.to(device='cpu', non_blocking=True)

    # print(model)
    print("\nLoRA-applied Model")
    summary(model)

    torch.save(get_adapters(model), ADAPTER_PATH)
    print(f"\nAdapters Saved:\t\t\t{os.path.getsize(ADAPTER_PATH)}")
