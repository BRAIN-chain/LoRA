# LoRA

[LoRA](https://github.com/microsoft/LoRA) implementation.

- Change `float32` to `float16` if needed.
- Change `cpu` to `cuda` if available.

# How to Use

Monkey-patch GPT-J for convenience. For example:

```python
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

transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
```

Now you can use LoRA-applying GPT-J just like the original one:

```python
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16",
    torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
```

# Test

For example, get `gpt-j-6B` information through:

```bash
$ python test/gptj_lora.py

Model Loaded.

Default Model
- # of           params:        6050882784
- # of trainable params:        6050882784
- # of          buffers:        117440540
Model Loaded.

LoRA-applied Model
- # of           params:        35635424
- # of trainable params:        34774016
- # of          buffers:        6167461916

Adapters Saved:                 69733529
```

# References

- [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B)
- [LoRA](https://github.com/microsoft/LoRA)
- [Frozen Layers](https://colab.research.google.com/drive/1ft6wQU0BhqG5PRlwgaZJv2VukKKjU4Es?usp=sharing#scrollTo=aIlHG9Wk0WaJ)
